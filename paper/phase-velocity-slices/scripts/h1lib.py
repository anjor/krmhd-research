"""Machinery for the forced-turbulence pilot that tests hypothesis H1 (.loop/specs/rework-H1.md): white-in-time
forcing of the Alfvenic sector and of the density moment g_0 at a fixed injection rate, exact per-step energy
accounting (forcing, hyper-resistive and hypercollisional losses, time-stepping error of the ideal part), the
sampled diagnostics of the H1 test (W(m), W(m, k_perp), field-aligned and k_z-only Hermite fluxes, the sign(k_z)
forward/backward split, D_perp(m), D_nu(m), slice energies and slice correlations, field-aligned wavenumbers, chi,
tilt statistics), the time-step limit of the explicit slice advection, and the gates of CLAUDE.md.

Conventions are those of p1lib (paper section 2): v_A = 1, box side L = 1 (so tau_A = L_z/v_A = 1, the repository's
convention), Phi = (z+ + z-)/2, Psi = (z+ - z-)/2, grad_par = d_z + {Psi, .}, s = sqrt(beta_i), free energy
W = sum_m p_m <g_m^2> with p_0 = 1 - 1/Lambda, p_m = 1 (m >= 1, zero closure), Alfvenic energy
E_A = (1/2)<|grad Phi|^2 + |grad Psi|^2>.  Volume averages of rfft coefficients: <f h> = sum_k w_k Re(f_k^* h_k)/N^6.

GANDALF (the installed krmhd package, v0.7.0) is used read-only: gandalf_step (Lawson-RK4), force_alfven_modes, the
noise generators of krmhd.forcing and poisson_bracket_3d.  Every diagnostic is computed here (GANDALF's
hermite_flux, phase_unmixing diagnostics and CFL routine are not used).

Import p1lib (directly or through this module) before anything else that imports JAX: it switches JAX to float64.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1lib as P  # noqa: E402  (first: float64 JAX on the CPU)
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from krmhd.forcing import (force_alfven_modes, force_hermite_moments, gaussian_white_noise_fourier,  # noqa: E402
                           gaussian_white_noise_fourier_perp_lowkz)
from krmhd.physics import KRMHDState, poisson_bracket_3d  # noqa: E402

SQRT2 = math.sqrt(2.0)

__all__ = ["WhiteForcing", "make_accounting", "make_sampler", "dt_limits", "exact_slice_radius", "state_from",
           "gate_steady_state", "gate_energy_balance", "gate_budget_closure", "gate_spectral_sanity",
           "spectral_slope", "window_slice", "time_average"]


# ------------------------------------------------------------------------------------------------------------------
# GANDALF state helper
# ------------------------------------------------------------------------------------------------------------------
def state_from(template: KRMHDState, zp: Any = None, zm: Any = None, g: Any = None) -> KRMHDState:
    """A KRMHDState with the parameters of `template` and the given fields (others taken from the template)."""
    return KRMHDState(z_plus=template.z_plus if zp is None else zp, z_minus=template.z_minus if zm is None else zm,
                      g=template.g if g is None else g, M=template.M, beta_i=template.beta_i, v_th=template.v_th,
                      nu=template.nu, Lambda=template.Lambda, time=template.time, grid=template.grid)


# ------------------------------------------------------------------------------------------------------------------
# forcing at a fixed injection rate
# ------------------------------------------------------------------------------------------------------------------
class WhiteForcing:
    """White-in-time forcing of Phi (as GANDALF's force_alfven_modes: the same increment on z+ and z-) and of g_0, with the amplitude
    rescaled every step so that the expected injection rates are fixed.

    GANDALF adds the increment (A/sqrt(dt)) xi per step, xi complex Gaussian with unit-variance real and imaginary
    parts on the forced Fourier entries, then projects the k_x = 0 plane onto its Hermitian part (which halves the
    expected power there).  So E|F_k|^2 = v_k A^2/dt with v_k = 2 (k_x > 0) or 1 (k_x = 0 plane), and the expected
    energy added per step is c A^2/dt with
        c_A = sum_forced (1/2) w_k v_k k_perp^2 / N^6      (Alfvenic energy; F is added to z+ and z-, i.e. to Phi),
        c_g = p_0 sum_forced w_k v_k / N^6                   (free energy; F is added to g_0).
    The injection rate is c A^2/dt^2, so A = dt sqrt(eps/c) holds it at eps for any dt (white noise, Ito: the cross
    term with the state averages to zero).  The forced entries are read off one draw of the generator itself.

    Config (mapping `forcing` with sub-mappings `alfven` and `density`):
      alfven:  eps (target rate of E_A), generator "isotropic" (krmhd.forcing.force_alfven_modes: |k| L/2 pi in
               [n_min, n_max], k_z = 0 included; k_perp = 0 entries are forced but carry no energy and do not act) or
               "perp_lowkz" (the same Phi-only forcing, F added to z+ and z- as force_alfven_modes does, with
               krmhd.forcing.gaussian_white_noise_fourier_perp_lowkz as generator: k_perp L/2 pi in [n_min, n_max],
               |n_z| <= max_nz, n_z = 0 only with include_nz0), n_min, n_max, max_nz, include_nz0, seed.
      density: eps (target rate of W), generator "perp_lowkz" (krmhd.forcing.gaussian_white_noise_fourier_perp_lowkz:
               k_perp L/2 pi in [n_min, n_max], |n_z| <= max_nz, n_z = 0 included only with include_nz0) or
               "isotropic" (krmhd.forcing.force_hermite_moments on g_0, the same |k| band as the Alfvenic forcing),
               n_min, n_max, max_nz, include_nz0, seed.
    eps = 0 switches a forcing off."""

    def __init__(self, box: P.Box, cfg: Mapping[str, Any], p0: float) -> None:
        self.box = box
        a, d = cfg["alfven"], cfg["density"]
        self.eps_A = P.parse_number(a["eps"])
        self.gen_A = str(a.get("generator", "isotropic"))
        if self.gen_A not in ("perp_lowkz", "isotropic"):
            raise ValueError(f"unknown Alfvenic forcing generator {self.gen_A!r}")
        self.nA = (int(a["n_min"]), int(a["n_max"]))
        self.max_nz_A = int(a.get("max_nz", 1))
        self.include_nz0_A = bool(a.get("include_nz0", False))
        self.key_A = jax.random.PRNGKey(int(a["seed"]))
        self.eps_g = P.parse_number(d["eps"])
        self.gen_g = str(d.get("generator", "perp_lowkz"))
        if self.gen_g not in ("perp_lowkz", "isotropic"):
            raise ValueError(f"unknown density forcing generator {self.gen_g!r}")
        self.ng = (int(d["n_min"]), int(d["n_max"]))
        self.max_nz = int(d.get("max_nz", 1))
        self.include_nz0 = bool(d.get("include_nz0", False))
        self.key_g = jax.random.PRNGKey(int(d["seed"]))
        self.p0 = float(p0)
        v = np.where(np.arange(box.N // 2 + 1) == 0, 1.0, 2.0)[None, None, :]
        mA = self._mask("alfven")
        mg = self._mask("density")
        self.mask_A, self.mask_g = mA, mg
        self.c_A = float(np.sum(mA * 0.5 * box.w * v * box.KP2)) / box.Ntot ** 2
        self.c_g = self.p0 * float(np.sum(mg * box.w * v)) / box.Ntot ** 2
        self.F_g_last = jnp.zeros((box.N, box.N, box.N // 2 + 1), dtype=jnp.complex128)

    def _mask(self, which: str) -> np.ndarray:
        """Forced entries of a generator (nonzero entries of one unit-amplitude draw)."""
        key = jax.random.PRNGKey(12345)
        g = self.box.grid
        gen = self.gen_A if which == "alfven" else self.gen_g
        n = self.nA if which == "alfven" else self.ng
        if gen == "isotropic":
            F, _ = gaussian_white_noise_fourier(g, 1.0, n[0], n[1], 1.0, key)
        elif which == "alfven":
            F, _ = gaussian_white_noise_fourier_perp_lowkz(g, 1.0, n[0], n[1], self.max_nz_A, self.include_nz0_A, 1.0,
                                                           key)
        else:
            F, _ = gaussian_white_noise_fourier_perp_lowkz(g, 1.0, n[0], n[1], self.max_nz, self.include_nz0, 1.0, key)
        return np.abs(np.asarray(F)) > 0

    def amplitudes(self, dt: float) -> tuple[float, float]:
        """(A_alfven, A_density) for the step dt: A = dt sqrt(eps/c)."""
        aA = dt * math.sqrt(self.eps_A / self.c_A) if self.eps_A > 0 else 0.0
        ag = dt * math.sqrt(self.eps_g / self.c_g) if self.eps_g > 0 else 0.0
        return aA, ag

    def describe(self) -> dict[str, Any]:
        """Forced entries and their wavenumbers (for the run record)."""
        b = self.box
        out: dict[str, Any] = dict(eps_A=self.eps_A, eps_g=self.eps_g, c_A=self.c_A, c_g=self.c_g,
                                   alfven_generator=self.gen_A, alfven_band=list(self.nA),
                                   alfven_max_nz=self.max_nz_A if self.gen_A == "perp_lowkz" else None,
                                   alfven_include_nz0=self.include_nz0_A if self.gen_A == "perp_lowkz" else None,
                                   density_generator=self.gen_g,
                                   density_band=list(self.ng), density_max_nz=self.max_nz,
                                   density_include_nz0=self.include_nz0)
        for name, m in (("alfven", self.mask_A), ("density", self.mask_g)):
            nz = np.broadcast_to(b.nz[:, None, None], m.shape)[m]
            kp = np.broadcast_to(np.sqrt(b.KP2) / b.k1, m.shape)[m]
            out[f"{name}_entries"] = int(m.sum())
            out[f"{name}_nz_values"] = sorted(set(int(x) for x in nz))
            out[f"{name}_nperp_range"] = [float(kp.min()), float(kp.max())]
            out[f"{name}_entries_kperp0"] = int(np.sum(kp < 1e-9))
        return out

    def apply(self, state: KRMHDState, dt: float) -> KRMHDState:
        """Add one step's increments to Phi and g_0 (amplitudes rescaled with dt).  The density increment is kept in
        F_g_last (for the field-aligned wavenumber of the injected pattern)."""
        aA, ag = self.amplitudes(dt)
        if aA > 0:
            if self.gen_A == "isotropic":
                state, self.key_A = force_alfven_modes(state, aA, self.nA[0], self.nA[1], dt, self.key_A)
            else:
                F, self.key_A = gaussian_white_noise_fourier_perp_lowkz(state.grid, aA, self.nA[0], self.nA[1],
                                                                        self.max_nz_A, self.include_nz0_A, dt,
                                                                        self.key_A)
                F = jnp.asarray(F, dtype=jnp.complex128)
                state = state_from(state, zp=state.z_plus + F, zm=state.z_minus + F)
        if ag > 0:
            if self.gen_g == "isotropic":
                g_old = state.g[..., 0]
                state, self.key_g = force_hermite_moments(state, ag, self.ng[0], self.ng[1], dt, self.key_g, (0,))
                self.F_g_last = state.g[..., 0] - g_old
            else:
                F, self.key_g = gaussian_white_noise_fourier_perp_lowkz(state.grid, ag, self.ng[0], self.ng[1],
                                                                        self.max_nz, self.include_nz0, dt,
                                                                        self.key_g)
                F = jnp.asarray(F, dtype=jnp.complex128)
                state = state_from(state, g=state.g.at[..., 0].add(F))
                self.F_g_last = F
        return state

    def keys(self) -> dict[str, np.ndarray]:
        """The PRNG keys (for a restart file)."""
        return dict(key_A=np.asarray(self.key_A), key_g=np.asarray(self.key_g))

    def set_keys(self, key_A: Any, key_g: Any) -> None:
        self.key_A = jnp.asarray(key_A)
        self.key_g = jnp.asarray(key_g)


# ------------------------------------------------------------------------------------------------------------------
# exact per-step accounting
# ------------------------------------------------------------------------------------------------------------------
def make_accounting(box: P.Box, sl: P.Slices, sp: P.StepParams):
    """Jitted per-step energy accounting for GANDALF's Lawson-RK4 step, whose dissipation is applied after the ideal
    update as multiplicative factors: z+-, g -> D_eta(k) z+-, D_eta(k) g with D_eta = exp(-eta (k_perp^2/
    k_perp,max^2)^r dt), then g_m -> D_nu(m) g_m with D_nu = exp(-nu (m/M)^n dt) for m >= 2 (eta_z = 0 here).

    post(zp, zm, g, dt) -> (E_A, pW_m [M+1], loss_A, loss_eta [M+1], loss_nu [M+1]) from the post-step fields:
      E_A, pW_m: Alfvenic energy and p_m <g_m^2> after the step;
      loss_A = sum_k (D_eta^-2 - 1) e_A(k): Alfvenic energy removed by hyper-resistivity in the step;
      loss_eta(m) = p_m sum_k (D_eta^-2 - 1) D_nu^-2 |g_m,k|^2 w/N^6: free energy of g_m removed by hyper-resistivity;
      loss_nu(m) = p_m (D_nu^-2 - 1) <g_m^2>: removed by hypercollisions (GANDALF applies them second).
    The pre-dissipation energies are therefore E_A + loss_A and pW_m + loss_eta + loss_nu exactly, and their change
    from the start of the step is the time-stepping error of the ideal part.
    energies(zp, zm, g) -> (E_A, pW_m)."""
    if sp.eta_z != 0.0:
        raise ValueError("accounting assumes eta_z = 0")
    N6 = float(box.Ntot) ** 2
    w = jnp.asarray(box.w)
    kp2 = jnp.asarray(box.KP2)
    rk = jnp.asarray((box.KP2 / box.kperp_max2_gandalf) ** sp.hyper_r)
    crates = jnp.asarray(P.coll_rates(sl.M, sp.nu, sp.hyper_n))
    p = jnp.asarray(sl.p)
    eta = float(sp.eta)

    @jax.jit
    def energies(zp, zm, g):
        eA = 0.25 * w * kp2 * (jnp.abs(zp) ** 2 + jnp.abs(zm) ** 2) / N6
        g2 = jnp.sum(w[..., None] * jnp.abs(g) ** 2, axis=(0, 1, 2)) / N6
        return jnp.sum(eA), p * g2

    @jax.jit
    def post(zp, zm, g, dt):
        fa = jnp.expm1(2.0 * eta * rk * dt)                     # D_eta^-2 - 1, [1, Ny, Nkx]
        fn = jnp.expm1(2.0 * crates * dt)                       # D_nu^-2 - 1, [M+1]
        eA = 0.25 * w * kp2 * (jnp.abs(zp) ** 2 + jnp.abs(zm) ** 2) / N6
        g2k = w[..., None] * jnp.abs(g) ** 2 / N6
        g2 = jnp.sum(g2k, axis=(0, 1, 2))
        loss_eta = p * jnp.sum(fa[..., None] * g2k, axis=(0, 1, 2)) * (1.0 + fn)
        return jnp.sum(eA), p * g2, jnp.sum(fa * eA), loss_eta, p * fn * g2

    return post, energies


# ------------------------------------------------------------------------------------------------------------------
# sampled diagnostics
# ------------------------------------------------------------------------------------------------------------------
def make_sampler(box: P.Box, sl: P.Slices, s: float, sp: P.StepParams, n_outer: int = 2):
    """Jitted diagnostics at one time, sample(zp, zm, g, F) -> dict of arrays (F: the latest density-forcing
    increment, for the field-aligned wavenumber of the injected pattern; zeros if none).

    Hermite space (W(m) = p_m <g_m^2>, the paper's Hermite spectrum):
      Wm [M+1] (<g_m^2>), pWm [M+1];
      Pi_fa [M]: field-aligned flux Pi_{m+1/2} = 2 s p_m A_{m,m+1} <g_m grad_par g_{m+1}>, grad_par = d_z + {Psi, .}
        (p1lib.hermite_flux_field_aligned);  Pi_kz [M]: the same with d_z only (p1lib.hermite_flux_kz);
      Pi_plus, Pi_minus [M]: sign(k_z) split of Pi_kz (Schekochihin et al. 2016): with g~_m = (i sgn k_z)^m g_m and
        g^+-_m = (g~_m +- g~_{m+1})/sqrt2, Pi_kz(k) = 2 s p_m A_{m,m+1} |k_z| (|g^+_m|^2 - |g^-_m|^2)/2, so
        Pi_plus >= 0 (phase mixing) and Pi_minus <= 0 (unmixing) sum to Pi_kz;  Wplus, Wminus [M] (<|g^+-_m|^2>,
        k_z != 0 only);
      kfa2, kz2 [M+1]: <|grad_par g_m|^2>/<g_m^2> and <|d_z g_m|^2>/<g_m^2> (squared field-aligned and Eulerian
        parallel wavenumbers of each moment);
      Dperp [M+1] = 2 eta p_m sum_k (k_perp^2/k_perp,max^2)^r |g_m|^2 w/N^6, Dnu [M+1] = 2 nu (m/M)^n p_m <g_m^2>
        (instantaneous dissipation per m, p1lib.dissipation_rates per moment);
      Wmk [n_shell, M+1]: p_m <g_m^2> in integer k_perp shells; Wmkz [n_zshell, M+1]: in |k_z| shells;
      E_slice [M+1], C_slice [M+1, M+1]: slice energies <a_j^2> and the slice correlation matrix <a_j a_l>.
    Alfvenic sector: E, E_kin, E_mag, E_plus, E_minus, D_A (2 eta sum r_k e_A), EAk [n_shell] (k_perp spectrum),
      EAkz [n_zshell], kperp_mean, kpar_mean, u_rms, b_rms, z_rms, chi_u, chi_z (p1lib.nonlinearity_parameter);
      outer-scale (k_perp L/2pi <= n_outer + 1/2) quantities: kperp_out, kpar_out (energy-weighted mean |k_z|),
      kpar_out_fa (rms field-aligned wavenumber of Phi_outer, sqrt(<|grad_par Phi_<|^2> / <|grad_perp Phi_<|^2>)
      times kperp_out ... see code), u_out, chi_out = kperp_out u_out/kpar_out, chi_out_fa.
    Injected pattern: kF_fa2, kF_z2 (squared field-aligned and Eulerian wavenumbers of F), tilt statistics
      tiltF2 = <|{Psi, F}|^2>/<F^2>."""
    M = sl.M
    N = box.N
    N6 = float(box.Ntot) ** 2
    kx, ky, kz, mask = box.jk
    w = jnp.asarray(box.w)
    w4 = w[..., None]
    KZ = jnp.asarray(box.KZ)
    KZ4 = KZ[..., None]
    KP2 = jnp.asarray(box.KP2)
    rk = jnp.asarray((box.KP2 / box.kperp_max2_gandalf) ** sp.hyper_r)
    crates = jnp.asarray(P.coll_rates(M, sp.nu, sp.hyper_n))
    p = jnp.asarray(sl.p)
    coef = jnp.asarray(2.0 * s * sl.p[:M] * np.array([sl.A[m, m + 1] for m in range(M)]))
    Wl = jnp.asarray(sl.Wl)
    sig = np.where(box.kz >= 0.0, 1.0, -1.0)
    phase = jnp.asarray((1j * sig[:, None]) ** np.arange(M + 1)[None, :])[:, None, None, :]   # (i sgn k_z)^m
    nzmask = jnp.asarray(np.abs(box.kz) > 0)[:, None, None, None]
    absKZ4 = jnp.abs(KZ4)
    shell = np.broadcast_to(box.nperp, (N, N, N // 2 + 1)).ravel()
    nsh = box.nperp_band + 1
    shell_j = jnp.asarray(np.minimum(shell, nsh - 1))
    zsh = np.broadcast_to(box.nz[:, None, None], (N, N, N // 2 + 1)).ravel()
    nzsh = int(box.nz.max()) + 1
    zsh_j = jnp.asarray(zsh)
    outer = jnp.asarray((np.sqrt(box.KP2) <= box.k1 * (n_outer + 0.5)) & (box.KP2 > 0))      # [1, Ny, Nkx]
    KP = jnp.sqrt(KP2)
    eta = float(sp.eta)

    def br(f, h):
        return poisson_bracket_3d(f, h, kx, ky, N, N, N, mask)

    br_m = jax.vmap(br, in_axes=(None, 3), out_axes=3)

    def vavg(f, h):
        return jnp.sum(w * jnp.real(jnp.conj(f) * h)) / N6

    @jax.jit
    def sample(zp, zm, g, F):
        phi = 0.5 * (zp + zm)
        psi = 0.5 * (zp - zm)
        # ---- Hermite space ----
        dzg = 1j * KZ4 * g
        gpar = dzg + br_m(psi, g)
        g2k = w4 * jnp.abs(g) ** 2 / N6
        Wm = jnp.sum(g2k, axis=(0, 1, 2))
        cfa = jnp.sum(w4 * jnp.real(jnp.conj(g[..., :-1]) * gpar[..., 1:]), axis=(0, 1, 2)) / N6
        cz = jnp.sum(w4 * jnp.real(jnp.conj(g[..., :-1]) * dzg[..., 1:]), axis=(0, 1, 2)) / N6
        gt = phase * g
        gp = (gt[..., :-1] + gt[..., 1:]) / SQRT2
        gm = (gt[..., :-1] - gt[..., 1:]) / SQRT2
        Pp = 0.5 * coef * jnp.sum(w4 * absKZ4 * jnp.abs(gp) ** 2, axis=(0, 1, 2)) / N6
        Pm = -0.5 * coef * jnp.sum(w4 * absKZ4 * jnp.abs(gm) ** 2, axis=(0, 1, 2)) / N6
        Wp = jnp.sum(nzmask * w4 * jnp.abs(gp) ** 2, axis=(0, 1, 2)) / N6
        Wmi = jnp.sum(nzmask * w4 * jnp.abs(gm) ** 2, axis=(0, 1, 2)) / N6
        kfa2 = jnp.sum(w4 * jnp.abs(gpar) ** 2, axis=(0, 1, 2)) / N6 / Wm
        kz2 = jnp.sum(w4 * jnp.abs(dzg) ** 2, axis=(0, 1, 2)) / N6 / Wm
        Dperp = 2.0 * eta * p * jnp.sum(rk[..., None] * g2k, axis=(0, 1, 2))
        Dnu = 2.0 * crates * p * Wm
        flat = (p * g2k).reshape(-1, M + 1)
        Wmk = jax.ops.segment_sum(flat, shell_j, num_segments=nsh)
        Wmkz = jax.ops.segment_sum(flat, zsh_j, num_segments=nzsh)
        a = g @ Wl
        C = jnp.real(jnp.einsum("zyxi,zyxj->ij", w4 * jnp.conj(a), a)) / N6
        # ---- Alfvenic sector ----
        eA = 0.25 * w * KP2 * (jnp.abs(zp) ** 2 + jnp.abs(zm) ** 2) / N6
        E = jnp.sum(eA)
        Ekin = 0.5 * jnp.sum(w * KP2 * jnp.abs(phi) ** 2) / N6
        Emag = 0.5 * jnp.sum(w * KP2 * jnp.abs(psi) ** 2) / N6
        Ep = 0.25 * jnp.sum(w * KP2 * jnp.abs(zp) ** 2) / N6
        Emi = 0.25 * jnp.sum(w * KP2 * jnp.abs(zm) ** 2) / N6
        DA = 2.0 * eta * jnp.sum(rk * eA)
        EAk = jax.ops.segment_sum(eA.ravel(), shell_j, num_segments=nsh)
        EAkz = jax.ops.segment_sum(eA.ravel(), zsh_j, num_segments=nzsh)
        kperp_mean = jnp.sum(KP * eA) / E
        kpar_mean = jnp.sum(jnp.abs(KZ) * eA) / E
        u_rms = jnp.sqrt(2.0 * Ekin)
        b_rms = jnp.sqrt(2.0 * Emag)
        z_rms = jnp.sqrt(2.0 * (Ep + Emi))
        # outer scale
        eo = eA * outer
        Eo = jnp.sum(eo)
        kperp_out = jnp.sum(KP * eo) / Eo
        kpar_out = jnp.sum(jnp.abs(KZ) * eo) / Eo
        phio = phi * outer
        u_out = jnp.sqrt(jnp.sum(w * KP2 * jnp.abs(phio) ** 2) / N6)
        gpar_phio = 1j * KZ * phio + br(psi, phio)
        # rms field-aligned wavenumber of the outer-scale stream function, weighted like its energy
        kpar_out_fa = jnp.sqrt(jnp.sum(w * jnp.abs(gpar_phio) ** 2) / N6 / (jnp.sum(w * jnp.abs(phio) ** 2) / N6))
        kpar_out_z = jnp.sqrt(jnp.sum(w * KZ ** 2 * jnp.abs(phio) ** 2) / jnp.sum(w * jnp.abs(phio) ** 2))
        # injected pattern
        F2 = vavg(F, F)
        tF = br(psi, F)
        gF = 1j * KZ * F + tF
        safe = jnp.where(F2 > 0, F2, 1.0)
        kF_fa2 = jnp.where(F2 > 0, vavg(gF, gF) / safe, 0.0)
        kF_z2 = jnp.where(F2 > 0, vavg(1j * KZ * F, 1j * KZ * F) / safe, 0.0)
        tiltF2 = jnp.where(F2 > 0, vavg(tF, tF) / safe, 0.0)
        return dict(Wm=Wm, pWm=p * Wm, Pi_fa=coef * cfa, Pi_kz=coef * cz, Pi_plus=Pp, Pi_minus=Pm, Wplus=Wp,
                    Wminus=Wmi, kfa2=kfa2, kz2=kz2, Dperp=Dperp, Dnu=Dnu, Wmk=Wmk, Wmkz=Wmkz,
                    E_slice=jnp.diag(C), C_slice=C, E=E, E_kin=Ekin, E_mag=Emag, E_plus=Ep, E_minus=Emi, D_A=DA,
                    EAk=EAk, EAkz=EAkz, kperp_mean=kperp_mean, kpar_mean=kpar_mean, u_rms=u_rms, b_rms=b_rms,
                    z_rms=z_rms, chi_u=kperp_mean * u_rms / kpar_mean, chi_z=kperp_mean * z_rms / kpar_mean,
                    kperp_out=kperp_out, kpar_out=kpar_out, kpar_out_fa=kpar_out_fa, kpar_out_z=kpar_out_z,
                    u_out=u_out, chi_out=kperp_out * u_out / kpar_out, chi_out_fa=kperp_out * u_out / kpar_out_fa,
                    kF_fa2=kF_fa2, kF_z2=kF_z2, tiltF2=tiltF2, F2=F2)

    return sample


# ------------------------------------------------------------------------------------------------------------------
# time step
# ------------------------------------------------------------------------------------------------------------------
def dt_limits(box: P.Box, zp: Any, zm: Any, s: float, umax: float, v_A: float = 1.0, pad: int = 2) -> dict[str, float]:
    """Time-step limits from the current fields (numpy, maxima on a grid refined by `pad`):
      dt_slice = 2 sqrt2 / rho, rho = k_perp,max max_{u = +-umax} max_x |grad_perp(Phi + s u Psi)| (the bound of
        p1lib.slice_advection_rate; |grad(Phi + c Psi)| is convex in c, so the extreme nodes give the maximum over all
        slices): the explicit RK4 stages of Lawson-RK4 are stable for rho dt <= 2 sqrt2;
      dt_alfven = (L/N)/max(v_A, max|grad_perp z+-|) (p1lib.alfvenic_cfl_dt).
    `umax` is the largest |u_j| of the ladder that sets the time step (it may belong to a larger M than the run's,
    e.g. during an Alfvenic spin-up at small M that hands over to the production M)."""
    phi, psi = P.phi_psi(np.asarray(zp), np.asarray(zm))
    fx, fy = box.grad_perp_real(phi, pad)
    sx, sy = box.grad_perp_real(psi, pad)
    V = max(float(np.sqrt((fx + c * sx) ** 2 + (fy + c * sy) ** 2).max()) for c in (s * umax, -s * umax))
    zpx, zpy = fx + sx, fy + sy
    zmx, zmy = fx - sx, fy - sy
    vp = max(float(np.sqrt(zpx ** 2 + zpy ** 2).max()), float(np.sqrt(zmx ** 2 + zmy ** 2).max()))
    rho = box.kperp_max * V
    return dict(V_slice=V, rho_bound=rho, dt_slice=P.LAWSON_RK4_LIMIT / rho if rho > 0 else math.inf,
                v_perp_max=vp, dt_alfven=(box.L / box.N) / max(v_A, vp))


def exact_slice_radius(box: P.Box, zp: Any, zm: Any, s: float, umax: float, tol: float = 1e-4) -> float:
    """Spectral radius of the explicit slice-advection operator P{Phi + s u Psi, .} maximised over u = +-umax
    (p1lib.advection_operator_norm, Lanczos); the Lawson-RK4 limit is 2 sqrt2 / radius."""
    phi, psi = P.phi_psi(np.asarray(zp), np.asarray(zm))
    return max(P.advection_operator_norm(box, phi + c * psi, tol=tol) for c in (s * umax, -s * umax))


# ------------------------------------------------------------------------------------------------------------------
# gates
# ------------------------------------------------------------------------------------------------------------------
def time_average(x: np.ndarray, dts: np.ndarray, first: int) -> np.ndarray:
    """Trapezoidal time average of samples x[first:] (leading axis = samples) with dts[k] the time between samples
    k-1 and k (the sampled runs have a variable dt, so plain sample means are biased in transients)."""
    x = np.asarray(x, dtype=float)
    while first < len(x) - 2 and not np.all(np.isfinite(x[first])):      # e.g. chi at a start from rest (0/0)
        first += 1
    d = np.asarray(dts, dtype=float)[first + 1:]
    mid = 0.5 * (x[first:-1] + x[first + 1:])
    return np.tensordot(d, mid, axes=(0, 0)) / d.sum()


def window_slice(t: np.ndarray, width: float) -> np.ndarray:
    """Boolean mask of the samples in the final window t >= t[-1] - width."""
    t = np.asarray(t, dtype=float)
    return t >= t[-1] - width - 1e-12


def gate_steady_state(t: np.ndarray, series: Mapping[str, np.ndarray], width: float, tol: float = 0.10,
                      t_first: float | None = None) -> dict[str, Any]:
    """CLAUDE.md steady-state gate as in shared/validation.check_steady_state: std/mean of each series over the final
    `width` (in tau_A = L_z/v_A = 1) below tol.  Not applicable if the sampled run is shorter than `width`."""
    t = np.asarray(t, dtype=float)
    t0 = t[0] if t_first is None else t_first
    if t[-1] - t0 < width - 1e-9:
        return dict(applies=False, criterion=f"std/mean < {tol} over the final {width} tau_A", tol=tol,
                    note=f"not applicable: run covers {t[-1] - t0:.2f} tau_A < {width} tau_A")
    m = window_slice(t, width)
    out: dict[str, Any] = dict(applies=True, criterion=f"std/mean < {tol} over the final {width} tau_A "
                               "(tau_A = L_z/v_A = 1), each series", tol=tol, window=[float(t[m][0]), float(t[-1])])
    ok = True
    for k, v in series.items():
        v = np.asarray(v, dtype=float)[m]
        f = float(np.std(v) / np.mean(v))
        out[f"{k}_fluct"] = f
        ok = ok and f < tol
    out["pass"] = bool(ok)
    return out


def gate_energy_balance(name: str, inj: float, diss: float, dE: float, T: float, tol: float = 0.05) -> dict[str, Any]:
    """Energy balance over a window of length T: |<inj> - <diss>|/<inj> < tol, with the time-averaged rates
    <inj> = (injected energy)/T and <diss> = (dissipated energy)/T; also reports the storage term dE/(inj T)."""
    if inj <= 0:
        return dict(applies=False, quantity=name, note="not applicable: no injection (forcing off)",
                    **{"pass": True})
    r = abs(inj - diss) / inj
    return dict(applies=True, quantity=name, criterion=f"|<inj> - <diss>|/<inj> < {tol}", tol=tol,
                inj_rate=inj / T, diss_rate=diss / T, rel_diff=r, storage_over_injection=dE / inj if inj > 0 else 0.0,
                **{"pass": bool(r < tol)})


def gate_budget_closure(name: str, err: float, inj: float, tol: float) -> dict[str, Any]:
    """Energy conservation of the ideal dynamics in a forced-dissipative run: the accumulated change of the energy in
    the ideal part of the steps (time-stepping error; forcing and dissipation are accounted exactly) relative to
    the injected energy over the same window, |sum err|/sum inj < tol."""
    if inj <= 0:
        return dict(applies=False, quantity=name, note="not applicable: no injection (forcing off)", abs_error=err)
    r = abs(err) / inj
    return dict(applies=True, quantity=name, criterion=f"|time-stepping error of the ideal part|/injected < {tol}",
                tol=tol, rel_error=r, **{"pass": bool(r < tol)})


def spectral_slope(k: np.ndarray, E: np.ndarray) -> float:
    """shared/validation.check_spectral_sanity's slope: log-log fit over the middle 60% of the positive bins."""
    k = np.asarray(k, dtype=float)
    E = np.asarray(E, dtype=float)
    v = (k > 0) & (E > 0) & np.isfinite(E)
    lk, lE = np.log10(k[v]), np.log10(E[v])
    n = len(lk)
    if n < 2:
        return float("nan")
    lo, hi = n // 5, 4 * n // 5
    if hi - lo < 3:
        lo, hi = 0, n
    return float(np.polyfit(lk[lo:hi], lE[lo:hi], 1)[0])


def gate_spectral_sanity(nperp: np.ndarray, EAk: np.ndarray, Wk: np.ndarray, nmax: int,
                         outer_frac_A: float, outer_frac_g: float, res_tol: float,
                         pWm: np.ndarray, Dnu: np.ndarray, m_tol: float) -> dict[str, Any]:
    """Spectral sanity: (i) the time-averaged Alfvenic E(k_perp) over the complete shells 1 <= n <= nmax has the
    shared/validation slope in (-5, -1); (ii) resolution in k_perp: fraction of the Alfvenic energy and of the free
    energy in shells n >= nmax (band corners included) below res_tol; (iii) resolution in m: the free energy in
    m > 3M/4 below m_tol of the total and at least half of the hypercollisional dissipation at m >= M/2 (the sink sits
    near m ~ M)."""
    sel = (nperp >= 1) & (nperp <= nmax)
    slope = spectral_slope(nperp[sel], EAk[sel])
    slope_g = spectral_slope(nperp[sel], Wk[sel])
    M = len(pWm) - 1
    frac_hi = float(pWm[3 * M // 4 + 1:].sum() / pWm.sum())
    dnu_hi = float(Dnu[M // 2:].sum() / Dnu.sum()) if Dnu.sum() > 0 else 1.0
    ok = (-5.0 < slope < -1.0) and outer_frac_A < res_tol and outer_frac_g < res_tol and frac_hi < m_tol \
        and dnu_hi > 0.5
    return dict(applies=True, criterion=("Alfvenic E(k_perp) slope (middle 60% of shells 1..nmax) in (-5, -1); "
                                         f"energy fraction in shells n >= nmax < {res_tol} (Alfvenic and free energy); "
                                         f"free energy in m > 3M/4 < {m_tol}; >= half of D_nu at m >= M/2"),
                slope_EA=slope, slope_W=slope_g, outer_frac_A=outer_frac_A, outer_frac_g=outer_frac_g,
                W_frac_m_gt_3M4=frac_hi, Dnu_frac_m_ge_M2=dnu_hi, **{"pass": bool(ok)})
