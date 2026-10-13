"""Shared helpers for the checks in this folder, which verify the dynamical consequences of the phase-velocity
slice structure (Sections 5, 6 and 7 and Appendix E of the paper).  The module wraps GANDALF's grid, Poisson
bracket, Hermite ladder and right-hand side (krmhd 0.6.0, used read-only) and provides volume averages, the slice
decomposition, RK4 steppers built on GANDALF's right-hand side, diagnostics, a pass/fail collector and a run log of
the GANDALF time integrations.  Numerical output goes to paper/phase-velocity-slices/data/derivations/dynamics/,
which is git-ignored.

Conventions (checked in d01 against GANDALF/krmhd 0.6.0):
  v_A = 1, B0 = 1;  Phi = (z+ + z-)/2,  Psi = (z+ - z-)/2;  {f, h} = f_x h_y - f_y h_x;
  u_perp = zhat x grad Phi,  b_perp = zhat x grad Psi;  D_t = d_t + {Phi, .};  grad_par = d_z + {Psi, .};
  one compressive branch in GANDALF's Hermite variables:  d_t g + {Phi, g} + s grad_par (A g) = -C g,
  A = krmhd.hermite.compute_streaming_matrix(M, Lambda) (A[1,0] = (1 - 1/Lambda)/sqrt2), s = sqrt(beta_i),
  C = nu diag((m/M)^n, m >= 2) (GANDALF's normalised hypercollisions).
  Free-energy weights p_0 = 1 - 1/Lambda, p_m = 1 (m >= 1); P0 A is symmetric (zero closure).
Volume averages: <f h> = (1/V) int f h d^3x = sum_k w_k Re(conj(f_k) h_k) / N^6 for GANDALF's unnormalised
rfftn coefficients, w_k = 1 on the kx = 0 and kx = Nyquist planes and 2 elsewhere.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from krmhd.hermite import compute_streaming_matrix  # noqa: E402
from krmhd.physics import KRMHDState, poisson_bracket_3d  # noqa: E402
from krmhd.spectral import SpectralGrid3D  # noqa: E402
from krmhd.timestepping import KRMHDFields, _krmhd_rhs_jit, gandalf_step  # noqa: E402

SQRT5 = float(np.sqrt(5.0))
HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(HERE))                    # paper/phase-velocity-slices
DATA = os.path.join(PAPER, "data", "derivations", "dynamics")    # numerical output (git-ignored)
RUNLOG = os.path.join(DATA, "gandalf_runs.jsonl")


def data_dir() -> str:
    """The folder for numerical output of these checks, created on first use."""
    os.makedirs(DATA, exist_ok=True)
    return DATA


# ----------------------------------------------------------------------------------------------
# grid
# ----------------------------------------------------------------------------------------------
class Box:
    """Triply periodic unit box (L = 1) with GANDALF's SpectralGrid3D, plus numpy helpers."""

    def __init__(self, N: int, L: float = 1.0) -> None:
        self.N = N
        self.grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=L, Ly=L, Lz=L)
        self.kx = np.asarray(self.grid.kx)
        self.ky = np.asarray(self.grid.ky)
        self.kz = np.asarray(self.grid.kz)
        self.mask = np.asarray(self.grid.dealias_mask)
        self.KX = self.kx[None, None, :]
        self.KY = self.ky[None, :, None]
        self.KZ = self.kz[:, None, None]
        self.KP2 = self.KX ** 2 + self.KY ** 2
        w = np.full(N // 2 + 1, 2.0)
        w[0] = 1.0
        w[-1] = 1.0
        self.w = w[None, None, :]
        self.k1 = 2 * np.pi / L
        self.jargs = (jnp.asarray(self.kx), jnp.asarray(self.ky), jnp.asarray(self.kz),
                      jnp.asarray(self.mask))

    # --- inner products (volume averages) ---
    def inner(self, f: np.ndarray, h: np.ndarray) -> float:
        """<f h> for two real fields given by rfftn coefficients."""
        return float(np.sum(self.w * np.real(np.conj(f) * h))) / self.N ** 6

    def norm2(self, f: np.ndarray) -> float:
        return self.inner(f, f)

    def to_real(self, f: np.ndarray, pad: int = 1) -> np.ndarray:
        """Real-space values on a grid refined by `pad` (zero padding; exact for band-limited f)."""
        N = self.N
        if pad == 1:
            return np.fft.irfftn(f, s=(N, N, N), axes=(0, 1, 2))
        Np = pad * N
        F = np.zeros((Np, Np, Np // 2 + 1) + f.shape[3:], dtype=complex)
        h = N // 2
        # copy kz, ky in fftfreq order (drop the Nyquist rows, which the 2/3 mask empties anyway)
        iz = np.r_[0:h, Np - h + 1:Np]
        jz = np.r_[0:h, N - h + 1:N]
        F[np.ix_(iz, iz, np.arange(h))] = f[np.ix_(jz, jz, np.arange(h))]
        return np.fft.irfftn(F, s=(Np, Np, Np), axes=(0, 1, 2)) * pad ** 3

    def rms_grad(self, f: np.ndarray) -> float:
        """sqrt(<|grad_perp f|^2>)."""
        return float(np.sqrt(np.sum(self.w * self.KP2 * np.abs(f) ** 2))) / self.N ** 3

    def random_field(self, rng: np.random.Generator, nperp=(1.0, 2.5), nzmax: int = 2,
                     extra: tuple = ()) -> np.ndarray:
        """Real Gaussian field filtered to n_perp in [a, b], |n_z| <= nzmax, inside the 2/3 mask, k = 0 removed."""
        N = self.N
        x = rng.standard_normal((N, N, N) + tuple(extra))
        f = np.fft.rfftn(x, axes=(0, 1, 2))
        KP = np.sqrt(self.KP2)
        sel = ((KP >= self.k1 * nperp[0] - 1e-9) & (KP <= self.k1 * nperp[1] + 1e-9)
               & (np.abs(self.KZ) <= self.k1 * nzmax + 1e-9) & self.mask)
        if extra:
            sel = sel[..., None]
        f = f * sel
        f[0, 0, 0, ...] = 0.0
        return f

    def embed(self, f0: np.ndarray, N0: int) -> np.ndarray:
        """Embed rfft coefficients from an N0 grid into this grid (same physical field; N0 Nyquist dropped)."""
        N = self.N
        h0 = N0 // 2
        F = np.zeros((N, N, N // 2 + 1) + f0.shape[3:], dtype=complex)
        i_dst = np.r_[0:h0, N - h0 + 1:N]
        i_src = np.r_[0:h0, N0 - h0 + 1:N0]
        F[np.ix_(i_dst, i_dst, np.arange(h0))] = f0[np.ix_(i_src, i_src, np.arange(h0))] * (N / N0) ** 3
        return F * (self.mask[..., None] if f0.ndim == 4 else self.mask)

    def bracket(self, f: np.ndarray, h: np.ndarray) -> np.ndarray:
        """GANDALF's dealiased Poisson bracket {f, h} (rfft coefficients in and out)."""
        N = self.N
        return np.asarray(poisson_bracket_3d(jnp.asarray(f), jnp.asarray(h), self.jargs[0], self.jargs[1],
                                             N, N, N, self.jargs[3]))

    def gradpar(self, psi: np.ndarray, f: np.ndarray) -> np.ndarray:
        """grad_par f = d_z f + {Psi, f}."""
        return 1j * self.KZ * f + self.bracket(psi, f)


# ----------------------------------------------------------------------------------------------
# Hermite ladder and slices
# ----------------------------------------------------------------------------------------------
def ladder(M: int, Lam: float, closure: str = "zero") -> np.ndarray:
    """GANDALF's streaming matrix A (the matrix of u = v_par/v_th on Hermite coefficients)."""
    return np.array(compute_streaming_matrix(M, float(Lam), closure), dtype=float)


def pweights(M: int, Lam: float) -> np.ndarray:
    p = np.ones(M + 1)
    p[0] = 1.0 - 1.0 / Lam
    return p


def slices(M: int, Lam: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Eigen-decomposition of A: u (ascending), right eigenvectors V (columns, V^T P0 V = I), left WL = V^T P0.

    Slice amplitudes a_j = WL[j] . g;  g = sum_j a_j V[:, j];  W = g^T P0 g = sum_j a_j^2.
    """
    A = ladder(M, Lam)
    p = pweights(M, Lam)
    ph = np.sqrt(p)
    B = (ph[:, None] * A) / ph[None, :]
    u, Y = np.linalg.eigh((B + B.T) / 2)
    V = Y / ph[:, None]
    WL = V.T * p[None, :]
    return u, V, WL


def coll_rates(M: int, nu: float, hyper_n: int) -> np.ndarray:
    """GANDALF's normalised hypercollision rates nu (m/M)^n for m >= 2."""
    m = np.arange(M + 1, dtype=float)
    r = nu * (m / M) ** hyper_n if M >= 2 else np.zeros(M + 1)
    r[:2] = 0.0
    return r


# ----------------------------------------------------------------------------------------------
# GANDALF right-hand sides
# ----------------------------------------------------------------------------------------------
def make_rhs(box: Box, M: int, Lam: float, beta: float, closure: str = "zero"):
    """JIT wrappers of GANDALF's _krmhd_rhs_jit (eta = 0, v_A = 1): full (z+, z-, g) and g-only."""
    N = box.N
    kx, ky, kz, mask = box.jargs

    @jax.jit
    def rhs_full(zp, zm, g):
        d = _krmhd_rhs_jit(KRMHDFields(zp, zm, g, 0.0), kx, ky, kz, mask, 0.0, 1.0, beta, 0.0, Lam, M,
                           N, N, N, closure=closure)
        return d.z_plus, d.z_minus, d.g

    @jax.jit
    def rhs_g(zp, zm, g):
        return _krmhd_rhs_jit(KRMHDFields(zp, zm, g, 0.0), kx, ky, kz, mask, 0.0, 1.0, beta, 0.0, Lam, M,
                              N, N, N, closure=closure).g

    return rhs_full, rhs_g


def make_rk4(rhs_full, rhs_g, coll: np.ndarray, dt: float):
    """Classical RK4 steppers built on GANDALF's RHS.  coll: hypercollision rates (m-diagonal), added to g."""
    c = jnp.asarray(coll)

    def fg(zp, zm, g):
        return rhs_g(zp, zm, g) - c * g

    def ff(zp, zm, g):
        a, b, d = rhs_full(zp, zm, g)
        return a, b, d - c * g

    @jax.jit
    def step_full(zp, zm, g):
        k1 = ff(zp, zm, g)
        k2 = ff(zp + 0.5 * dt * k1[0], zm + 0.5 * dt * k1[1], g + 0.5 * dt * k1[2])
        k3 = ff(zp + 0.5 * dt * k2[0], zm + 0.5 * dt * k2[1], g + 0.5 * dt * k2[2])
        k4 = ff(zp + dt * k3[0], zm + dt * k3[1], g + dt * k3[2])
        return (zp + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                zm + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
                g + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]))

    @jax.jit
    def step_frozen(zp, zm, g):
        k1 = fg(zp, zm, g)
        k2 = fg(zp, zm, g + 0.5 * dt * k1)
        k3 = fg(zp, zm, g + 0.5 * dt * k2)
        k4 = fg(zp, zm, g + dt * k3)
        return g + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return step_full, step_frozen


def make_state(box: Box, zp, zm, g, Lam: float, beta: float, nu: float = 0.0, t: float = 0.0) -> KRMHDState:
    M = g.shape[-1] - 1
    return KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(g), M=M, beta_i=beta,
                      v_th=1.0, nu=nu, Lambda=float(Lam), time=t, grid=box.grid)


# ----------------------------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------------------------
def W_m(box: Box, g) -> np.ndarray:
    """<g_m^2> for each m (unweighted per-moment spectrum, volume average)."""
    g = np.asarray(g)
    return np.einsum("zyxm->m", box.w[..., None] * np.abs(g) ** 2) / box.N ** 6


def slice_amps(g, WL: np.ndarray) -> np.ndarray:
    return np.asarray(g) @ WL.T


def corr_matrix(box: Box, a: np.ndarray) -> np.ndarray:
    """C(j, j') = <a_j a_j'> (volume average, k-summed)."""
    return np.real(np.einsum("zyxi,zyxj->ij", box.w[..., None] * np.conj(a), a)) / box.N ** 6


def flux_full(box: Box, g, psi, s: float, A: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Field-aligned Hermite flux Pi_m = 2 s p_m A_{m,m+1} <g_m grad_par g_{m+1}>, m = 0..M-1."""
    g = np.asarray(g)
    psi = np.asarray(psi)
    M = g.shape[-1] - 1
    out = np.zeros(M)
    for m in range(M):
        out[m] = 2 * s * p[m] * A[m, m + 1] * box.inner(g[..., m], box.gradpar(psi, g[..., m + 1]))
    return out


def flux_kz(box: Box, g, s: float, A: np.ndarray, p: np.ndarray, per_mode: bool = False):
    """k_z-only Hermite flux -2 s p_m A_{m,m+1} k_z Im(conj(g_m) g_{m+1}), i.e. s p_m times krmhd's hermite_flux."""
    g = np.asarray(g)
    M = g.shape[-1] - 1
    coef = 2 * s * p[:M] * np.array([A[m, m + 1] for m in range(M)])
    pm = -box.KZ[..., None] * coef * np.imag(np.conj(g[..., :-1]) * g[..., 1:])
    if per_mode:
        return pm
    return np.einsum("zyxm->m", box.w[..., None] * pm) / box.N ** 6


def Wk_perp(box: Box, g, p: np.ndarray) -> np.ndarray:
    """Free energy in integer k_perp shells (not a single-point statistic)."""
    g = np.asarray(g)
    dens = box.w * np.einsum("zyxm,m->zyx", np.abs(g) ** 2, p) / box.N ** 6
    sh = np.rint(np.sqrt(box.KP2) / box.k1).astype(int) * np.ones_like(box.KZ, dtype=int)
    nb = box.N // 3 + 2
    return np.bincount(np.clip(sh, 0, nb - 1).ravel(), weights=dens.ravel(), minlength=nb)[:nb]


def Wk_z(box: Box, g, p: np.ndarray) -> np.ndarray:
    """Free energy in |n_z| shells (Eulerian k_z spectrum; not a single-point statistic)."""
    g = np.asarray(g)
    dens = box.w * np.einsum("zyxm,m->zyx", np.abs(g) ** 2, p) / box.N ** 6
    nz = np.rint(np.abs(box.kz) / box.k1).astype(int)
    nb = box.N // 3 + 2
    out = np.zeros(nb)
    for i, n in enumerate(nz):
        out[min(n, nb - 1)] += dens[i].sum()
    return out


def quartic_and_quantiles(box: Box, g, ms=(0, 1, 2), qs=(0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99), pad: int = 2):
    """Single-point non-quadratic statistics on a padded grid: <g_m^4> and quantiles of g_m(x)."""
    g = np.asarray(g)
    q4 = []
    qq = []
    for m in ms:
        x = box.to_real(g[..., m], pad=pad)
        q4.append(float(np.mean(x ** 4)))
        qq.append(np.quantile(x.ravel(), qs))
    return np.array(q4), np.array(qq)


def relerr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.abs(a - b).sum() / max(np.abs(b).sum(), 1e-300))


def maxrel(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


# ----------------------------------------------------------------------------------------------
# bookkeeping
# ----------------------------------------------------------------------------------------------
def log_run(script: str, purpose: str, N: int, M: int, steps: int, wall: float, extra: dict | None = None) -> None:
    """Append one GANDALF time integration to the run log RUNLOG (one JSON line, with the local end time)."""
    rec = dict(script=script, purpose=purpose, grid=f"{N}^3", M=M, steps=steps, wall_s=round(wall, 1),
               ended_local=time.strftime("%Y%m%d_%H%M%S"))
    if extra:
        rec.update(extra)
    os.makedirs(os.path.dirname(RUNLOG), exist_ok=True)
    with open(RUNLOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


class Checks:
    """Collects named pass/fail checks; exit status 1 if any failed."""

    def __init__(self) -> None:
        self.items: list[tuple[str, bool, str]] = []

    def __call__(self, name: str, ok: bool, info: str = "") -> bool:
        ok = bool(ok)
        self.items.append((name, ok, info))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
        return ok

    def finish(self) -> None:
        nf = sum(1 for _, ok, _ in self.items if not ok)
        print(f"SUMMARY: {len(self.items) - nf}/{len(self.items)} checks passed", flush=True)
        raise SystemExit(1 if nf else 0)


def timer() -> Callable[[], float]:
    t0 = time.time()
    return lambda: time.time() - t0
