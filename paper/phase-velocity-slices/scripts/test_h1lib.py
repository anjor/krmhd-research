#!/usr/bin/env python
"""Self-test of h1lib.py on a 16^3 grid (about a minute on a laptop CPU, mostly JIT compilation).  Exits with status 1
if any check fails.

A  The jitted sampler agrees with p1lib's numpy diagnostics: <g_m^2>, field-aligned and k_z-only Hermite fluxes,
   slice correlation matrix, k_perp spectrum of the free energy, dissipation rates, Alfvenic energies and chi; the
   squared field-aligned wavenumber per moment equals <|grad_par g_m|^2>/<g_m^2> from p1lib's grad_par.
B  The sign(k_z) split: Pi_plus >= 0, Pi_minus <= 0 and Pi_plus + Pi_minus = Pi_kz.
C  Per-m free-energy budget through GANDALF's right-hand side: d/dt p_m <g_m^2> = Pi_{m-1/2} - Pi_{m+1/2} with the
   field-aligned flux (to round-off), and not with the k_z-only flux.
D  Per-step accounting: one Lawson-RK4 step with hyper-resistivity and hypercollisions; the pre-dissipation energies
   reconstructed from the post-step fields (post + losses) equal the energies after the same step with eta = nu = 0,
   for the Alfvenic energy and for every p_m <g_m^2>.
E  Forcing: the expected injection per step c A^2/dt (WhiteForcing constants) agrees with the mean over many draws
   of GANDALF's generators; force_alfven_modes changes Phi only; the density increment goes to g_0 only.
F  Block averages (steady-state block check): exact rates from accumulated increments on a variable time step,
   dt-weighted means of sampled series, and the spread of a series that differs between blocks.
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1lib as H  # noqa: E402  (imports p1lib first)
import p1lib as P  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from krmhd.timestepping import gandalf_step  # noqa: E402

T0 = time.time()
FAILED: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)
    return ok


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


def main() -> int:
    N, M, Lam, beta = 16, 9, P.SQRT5, 1.7
    s = math.sqrt(beta)
    box = P.Box(N)
    sl = P.make_slices(M, Lam, "zero")
    rng = np.random.default_rng(7)
    zp = P.normalise_rms_grad(box, box.random_field(rng, (1, 4), 3), 0.8)
    zm = P.normalise_rms_grad(box, box.random_field(rng, (1, 4), 3), 0.6)
    g = box.random_field(rng, (1, 4), 4, extra=(M + 1,)) * 0.01
    F = box.random_field(rng, (1, 2), 1) * 0.01
    sp = P.StepParams(scheme="lawson_rk4", eta=3.0, hyper_r=2, nu=5.0, hyper_n=6)
    phi, psi = P.phi_psi(zp, zm)

    print("A  sampler vs p1lib")
    sample = H.make_sampler(box, sl, s, sp)
    d = {k: np.asarray(v) for k, v in sample(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g), jnp.asarray(F)).items()}
    check("A1 <g_m^2>", rel(d["Wm"], P.W_m(box, g)) < 1e-12, f"{rel(d['Wm'], P.W_m(box, g)):.1e}")
    pfa = P.hermite_flux_field_aligned(box, g, psi, s, sl)
    pkz = P.hermite_flux_kz(box, g, s, sl)
    check("A2 field-aligned flux", rel(d["Pi_fa"], pfa) < 1e-10, f"{rel(d['Pi_fa'], pfa):.1e}")
    check("A3 k_z-only flux", rel(d["Pi_kz"], pkz) < 1e-10, f"{rel(d['Pi_kz'], pkz):.1e}")
    C = P.slice_correlation(box, sl.amplitudes(g))
    check("A4 slice correlation", rel(d["C_slice"], C) < 1e-10, f"{rel(d['C_slice'], C):.1e}")
    wk = P.W_kperp(box, g, sl.p)
    check("A5 W(k_perp)", rel(d["Wmk"].sum(1), wk) < 1e-10, f"{rel(d['Wmk'].sum(1), wk):.1e}")
    dr = P.dissipation_rates(box, sl, zp, zm, g, sp.eta, sp.hyper_r, sp.nu, sp.hyper_n)
    check("A6 dissipation rates", max(rel(d["Dperp"].sum(), dr["eps_eta_g"]), rel(d["Dnu"].sum(), dr["eps_nu"]),
                                      rel(d["D_A"], dr["eps_eta_A"])) < 1e-10)
    ea = P.alfvenic_energy(box, zp, zm)
    check("A7 Alfvenic energies", max(rel(d[k], ea[k]) for k in ("E", "E_kin", "E_mag", "E_plus", "E_minus")) < 1e-12)
    chi = P.nonlinearity_parameter(box, zp, zm)
    check("A8 chi", max(rel(d["chi_u"], chi["chi_u"]), rel(d["chi_z"], chi["chi_z"])) < 1e-12,
          f"chi_u {float(d['chi_u']):.3f}, chi_out {float(d['chi_out']):.3f}, chi_out_fa {float(d['chi_out_fa']):.3f}")
    kfa = np.array([box.avg2(box.gradpar(psi, g[..., m])) / box.avg2(g[..., m]) for m in range(M + 1)])
    check("A9 field-aligned wavenumber per m", rel(d["kfa2"], kfa) < 1e-10, f"{rel(d['kfa2'], kfa):.1e}")
    kF = box.avg2(box.gradpar(psi, F)) / box.avg2(F)
    check("A10 field-aligned wavenumber of the forcing pattern", rel(d["kF_fa2"], kF) < 1e-10,
          f"kF_fa/k1 = {math.sqrt(float(d['kF_fa2'])) / box.k1:.3f}, kF_z/k1 = {math.sqrt(float(d['kF_z2'])) / box.k1:.3f}")
    check("A11 W(m, k_perp) and W(m, k_z) sum to W(m)", max(rel(d["Wmk"].sum(0), d["pWm"]),
                                                          rel(d["Wmkz"].sum(0), d["pWm"])) < 1e-12)

    print("B  sign(k_z) split")
    check("B1 Pi_plus >= 0, Pi_minus <= 0", bool((d["Pi_plus"] >= 0).all() and (d["Pi_minus"] <= 0).all()))
    check("B2 Pi_plus + Pi_minus = Pi_kz", rel(d["Pi_plus"] + d["Pi_minus"], d["Pi_kz"]) < 1e-10,
          f"{rel(d['Pi_plus'] + d['Pi_minus'], d['Pi_kz']):.1e}")

    print("C  per-m budget through GANDALF's right-hand side")
    _, rhs_g = P.make_rhs(box, M, Lam, beta, "zero")
    dg = np.asarray(rhs_g(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g)))
    dW = np.array([2.0 * sl.p[m] * box.avg(g[..., m], dg[..., m]) for m in range(M + 1)])
    Pi = np.concatenate([[0.0], d["Pi_fa"], [0.0]])
    Pz = np.concatenate([[0.0], d["Pi_kz"], [0.0]])
    div = Pi[:-1] - Pi[1:]
    divz = Pz[:-1] - Pz[1:]
    check("C1 dW_m/dt = Pi_{m-1/2} - Pi_{m+1/2} (field-aligned)", rel(div, dW) < 1e-9, f"{rel(div, dW):.1e}")
    check("C2 ... fails with the k_z-only flux", rel(divz, dW) > 1e-2, f"{rel(divz, dW):.2f}")

    print("D  per-step accounting")
    post, energies = H.make_accounting(box, sl, sp)
    st = P.make_state(box, zp, zm, g, Lam, beta)
    dt = 2e-3
    s1 = P.gandalf_advance(st, dt, 1, sp)
    s0 = P.gandalf_advance(st, dt, 1, P.StepParams(scheme="lawson_rk4", eta=0.0, hyper_r=2, nu=0.0, hyper_n=6))
    EA1, pW1, lA, le, ln = (np.asarray(x) for x in post(s1.z_plus, s1.z_minus, s1.g, dt))
    EA0, pW0 = (np.asarray(x) for x in energies(s0.z_plus, s0.z_minus, s0.g))
    check("D1 Alfvenic: post + loss = undamped step", rel(EA1 + lA, EA0) < 1e-12,
          f"{rel(EA1 + lA, EA0):.1e} (loss fraction {float(lA / EA0):.2e})")
    check("D2 free energy per m: post + losses = undamped step", rel(pW1 + le + ln, pW0) < 1e-11,
          f"{rel(pW1 + le + ln, pW0):.1e} (loss fractions eta {float(le.sum() / pW0.sum()):.2e}, "
          f"nu {float(ln.sum() / pW0.sum()):.2e})")

    print("E  forcing")
    cfg = dict(alfven=dict(eps=1.0, generator="isotropic", n_min=1, n_max=2, seed=1),
               density=dict(eps=1.0, generator="perp_lowkz", n_min=1, n_max=2, max_nz=1, include_nz0=False, seed=2))
    fo = H.WhiteForcing(box, cfg, sl.p[0])
    info = fo.describe()
    check("E1 density forcing excludes k_perp = 0 and n_z = 0", info["density_entries_kperp0"] == 0
          and info["density_nz_values"] == [1], str({k: info[k] for k in info if k.startswith("density")}))
    dt = 1e-3
    zero = P.make_state(box, np.zeros_like(zp), np.zeros_like(zm), np.zeros_like(g), Lam, beta)
    _, energies0 = H.make_accounting(box, sl, P.StepParams(scheme="lawson_rk4"))
    inj_A, inj_g = [], []
    for _ in range(400):
        s2 = fo.apply(zero, dt)
        e, pw = energies0(s2.z_plus, s2.z_minus, s2.g)
        inj_A.append(float(e))
        inj_g.append(float(pw[0]))
    rA = np.mean(inj_A) / dt / 1.0
    rg = np.mean(inj_g) / dt / 1.0
    errA = np.std(inj_A) / np.mean(inj_A) / math.sqrt(400)
    errg = np.std(inj_g) / np.mean(inj_g) / math.sqrt(400)
    check("E2 Alfvenic injection rate = eps (mean of 400 draws)", abs(rA - 1.0) < 4 * errA,
          f"{rA:.4f} +- {errA:.4f}")
    check("E3 density injection rate = eps (mean of 400 draws)", abs(rg - 1.0) < 4 * errg,
          f"{rg:.4f} +- {errg:.4f}")
    s3 = fo.apply(st, dt)
    dphi = P.phi_psi(np.asarray(s3.z_plus), np.asarray(s3.z_minus))
    check("E4 Psi unchanged, Phi changed", np.abs(dphi[1] - psi).max() < 1e-14 * np.abs(psi).max()
          and np.abs(dphi[0] - phi).max() > 0)
    dg3 = np.asarray(s3.g) - g
    check("E5 density increment in g_0 only, equal to F_g_last", np.abs(dg3[..., 1:]).max() == 0.0
          and np.abs(dg3[..., 0] - np.asarray(fo.F_g_last)).max() < 1e-14 * np.abs(dg3[..., 0]).max())

    cfg2 = dict(alfven=dict(eps=1.0, generator="perp_lowkz", n_min=1, n_max=2, max_nz=1, include_nz0=False, seed=3),
                density=dict(eps=0.0, generator="perp_lowkz", n_min=1, n_max=2, seed=4))
    fo2 = H.WhiteForcing(box, cfg2, sl.p[0])
    i2 = fo2.describe()
    inj = []
    for _ in range(400):
        s4 = fo2.apply(zero, dt)
        inj.append(float(energies0(s4.z_plus, s4.z_minus, s4.g)[0]))
    r2 = np.mean(inj) / dt
    e2 = np.std(inj) / np.mean(inj) / math.sqrt(400)
    check("E6 Phi forcing on the perp/low-k_z band: n_z = +-1 only, no k_perp = 0, rate = eps",
          i2["alfven_nz_values"] == [1] and i2["alfven_entries_kperp0"] == 0 and abs(r2 - 1.0) < 4 * e2,
          f"{r2:.4f} +- {e2:.4f}")
    s5 = fo2.apply(st, dt)
    d5 = P.phi_psi(np.asarray(s5.z_plus), np.asarray(s5.z_minus))
    check("E7 ... Psi unchanged", np.abs(d5[1] - psi).max() < 1e-14 * np.abs(psi).max()
          and np.abs(d5[0] - phi).max() > 0)

    # F: block averages on a variable step; rate 3 + t (accumulated) and the sampled series 2 + [t > 5]
    dts_f = np.concatenate([[0.0], np.where(np.arange(1, 41) % 2 == 0, 0.2, 0.3)])
    t_f = np.cumsum(dts_f)
    acc_f = np.concatenate([[0.0], (3 + 0.5 * (t_f[1:] + t_f[:-1])) * dts_f[1:]])
    smp_f = np.stack([2.0 + (t_f > t_f[-1] / 2), np.full_like(t_f, 4.0)], axis=1)
    bf = H.block_averages(t_f, dts_f, 0, 2, dict(x=smp_f), dict(eps=acc_f))
    half = t_f[-1] / 2
    exact = [3 + half / 2, 3 + 1.5 * half]
    check("F1 block rates from accumulated increments are exact", rel(bf["eps"], exact) < 1e-12, f"{bf['eps']} vs {exact}")
    check("F2 dt-weighted block means of sampled series", rel(bf["x"], [[2.0, 4.0], [3.0, 4.0]]) < 1e-12, f"{bf['x']}")
    gf = H.gate_block_convergence(bf, "eps", {"x": ("x", slice(0, 2), False)}, 0.5)
    check("F3 spread (max - min)/mean per column", abs(gf["x"]["max_spread"] - 0.4) < 1e-12 and gf["x"]["at"] == 0
          and gf["pass"], f"{gf['x']}")

    print(f"{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)} ({time.time() - T0:.0f} s)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
