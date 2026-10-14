"""Quick self-test of p1lib.py on 8^3 and 16^3 grids (well under a minute on a laptop CPU, most of it JIT compilation).
Exits with status 1 if any check fails.

A  The slice projection of GANDALF's right-hand side obeys the slice equation to round-off:
   w_j . (dg/dt) = -{Phi + s u_j Psi, a_j} - s u_j d_z a_j for every j (both closures, Lambda = +-sqrt5, beta_i != 1,
   strong fields so that the brackets dominate the streaming); dropping the u_j Psi advection fails.
B  W = sum of slice energies: sum_m p_m <g_m^2> = sum_j <a_j^2>; Q_phi from the slices equals <g^T P phi(A) g>
   for phi = u, u^2, u^3, 1/u (odd M).
C  Normalisations agree with GANDALF's own diagnostics: W(m) and the free energy (hermite_moment_energy,
   hermite_free_energy), the Alfvenic energy (krmhd.physics.energy), the k_z-only flux (hermite_flux).
D  Field-aligned flux budget through GANDALF's right-hand side: d/dt sum_{n<=m} p_n <g_n^2> = -Pi_{m+1/2}, and the
   field-line part {Psi, .} of the flux is not negligible.
E  One gandalf_step (Lawson-RK4 and IMEX-RK222) from a random state conserves Q_1 to time-stepping accuracy: the
   change of Q_1 is orders of magnitude below the change of the individual moment energies and falls at least like
   dt^3 when dt is halved.  One Lawson step with hypercollisions from a single-slice state leaks into slice l at
   the first-order rate dt |K_{l j0}| of slice_coupling_matrix.
F  Frozen-field RK4 conserves every slice energy; with z+- = 0 it reproduces the exact k_z-only solution.
G  Fields: embedding from 8^3 into 16^3 keeps the physical field; the Gaussian blob peaks at 1 at its centre;
   plane_z agrees with the full inverse transform.
H  The slice-advection rate k_perp,max max|grad(Phi + s u Psi)| bounds the spectral radius of GANDALF's dealiased
   advection operator (explicit 512 x 512 matrix on 8^3).
I  Bookkeeping: run IDs, the .npz/.json round trip (in a temporary folder), gate outcome, output-time check,
   parse_number.
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1lib as P  # noqa: E402  (first: switches JAX to float64)
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from krmhd.diagnostics import hermite_flux, hermite_free_energy, hermite_moment_energy  # noqa: E402
from krmhd.physics import energy as krmhd_energy  # noqa: E402
from krmhd.timestepping import gandalf_step  # noqa: E402

T0 = time.time()
FAILED: list[str] = []
NCHK = [0]


def check(label: str, ok: bool, detail: str = "") -> bool:
    """Record and print one PASS/FAIL line."""
    NCHK[0] += 1
    ok = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)
    return ok


def qform(box: P.Box, g: np.ndarray, K: np.ndarray) -> float:
    """<g^T K g> (volume average) for a real symmetric K."""
    dens = np.real(np.einsum("zyxm,mn,zyxn->zyx", np.conj(g), K, g))
    return float(np.sum(box.w * dens)) / box.Ntot ** 2


M = 7                                   # odd, so that Q_{1/u} exists
box = P.Box(16)
rng = np.random.default_rng(20261013)

# ---------------------------------------------------------------------------------------------------------------
print("A/B/D. slice equation, slice energies, Q_phi and the flux budget through GANDALF's right-hand side", flush=True)
for Lam, clo, beta in ((P.SQRT5, "zero", 1.7), (-P.SQRT5, "symmetric", 0.6)):
    s = math.sqrt(beta)
    sl = P.make_slices(M, Lam, clo)
    phi = P.normalise_rms_grad(box, box.random_field(rng, (1, 5), 4), 20.0)
    psi = P.normalise_rms_grad(box, box.random_field(rng, (1, 5), 4), 20.0)
    zp, zm = P.elsasser(phi, psi)
    g = box.random_field(rng, (1, 5), 4, extra=(M + 1,))
    _, rhs_g = P.make_rhs(box, M, Lam, beta, clo)
    dg = np.asarray(rhs_g(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g)))
    a = sl.amplitudes(g)
    da = sl.amplitudes(dg)
    worst, worst_ctrl, ratio = 0.0, np.inf, []
    for j in range(M + 1):
        adv = box.bracket(phi + s * sl.u[j] * psi, a[..., j])
        stream = s * sl.u[j] * box.dz(a[..., j])
        pred = -adv - stream
        pred[0, 0, 0] = 0.0
        sc = np.abs(pred).max()
        worst = max(worst, float(np.abs(da[..., j] - pred).max() / sc))
        ctrl = -box.bracket(phi, a[..., j]) - stream
        ctrl[0, 0, 0] = 0.0
        if abs(sl.u[j]) > 1e-8:
            worst_ctrl = min(worst_ctrl, float(np.abs(da[..., j] - ctrl).max() / sc))
            ratio.append(math.sqrt(box.avg2(adv) / box.avg2(stream)))
    lab = f"Lambda={Lam:+.3f}, {clo} closure, beta_i={beta}"
    check(f"A1 {lab}: w_j.(dg/dt) = -{{Phi + s u_j Psi, a_j}} - s u_j d_z a_j for all j", worst < 1e-12,
          f"max rel. mismatch {worst:.1e}; ||bracket||/||streaming|| = {min(ratio):.1f}..{max(ratio):.1f}")
    check(f"A2 {lab}: control without the u_j Psi advection fails", worst_ctrl > 1e-2, f"{worst_ctrl:.1e}")

    Wfree = P.free_energy(box, g, sl.p)
    E = P.slice_energies(box, a)
    check(f"B1 {lab}: sum_m p_m <g_m^2> = sum_j <a_j^2>", abs(E.sum() / Wfree - 1) < 1e-13,
          f"rel. diff {abs(E.sum() / Wfree - 1):.1e}")
    Q = P.Q_phi(box, a, sl.u)
    Pm = np.diag(sl.p)
    ref = {"Q_1": Pm, "Q_u": Pm @ sl.A, "Q_u2": Pm @ sl.A @ sl.A, "Q_u3": Pm @ sl.A @ sl.A @ sl.A,
           "Q_inv_u": Pm @ np.linalg.inv(sl.A)}
    errs = {k: abs(Q[k] - qform(box, g, K)) / P.Q_phi_scale(box, a, sl.u, n)
            for (n, k), K in zip(P.PHI_NAMES.items(), ref.values())}
    check(f"B2 {lab}: Q_phi (slices) = <g^T P phi(A) g> for phi = 1, u, u^2, u^3, 1/u", max(errs.values()) < 1e-12,
          ", ".join(f"{k} {v:.1e}" for k, v in errs.items()))

    rate_m = np.array([2.0 * sl.p[m] * box.avg(g[..., m], dg[..., m]) for m in range(M + 1)])
    Pi = P.hermite_flux_field_aligned(box, g, psi, s, sl)
    Piz = P.hermite_flux_kz(box, g, s, sl)
    bud = float(np.abs(np.cumsum(rate_m)[:M] + Pi).max() / np.abs(Pi).max())
    fl = float(np.abs(Pi - Piz).max() / np.abs(Pi).max())
    check(f"D1 {lab}: d/dt sum_(n<=m) p_n <g_n^2> = -Pi_(m+1/2) (field-aligned flux)", bud < 1e-11,
          f"rel. residual {bud:.1e}; total dW/dt / scale = {abs(rate_m.sum()) / np.abs(rate_m).max():.1e}")
    check(f"D2 {lab}: the {{Psi, .}} part of the flux is not negligible here", fl > 0.1,
          f"max|Pi - Pi_kz|/max|Pi| = {fl:.2f}")

# ---------------------------------------------------------------------------------------------------------------
print("C. normalisations against GANDALF's diagnostics (Lambda = sqrt5, zero closure)", flush=True)
Lam, beta = P.SQRT5, 1.0
s = 1.0
sl = P.make_slices(M, Lam)
phi = P.normalise_rms_grad(box, box.random_field(rng, (1, 2.5), 2), 1.0)
psi = P.normalise_rms_grad(box, box.random_field(rng, (1, 2.5), 2), 1.0)
zp, zm = P.elsasser(phi, psi)
g = np.zeros((16, 16, 9, M + 1), complex)
g[..., 0] = box.random_field(rng, (1, 2.5), 2)
g[..., 1] = 0.5 * g[..., 0] + 0.5 * box.random_field(rng, (1, 2.5), 2)     # correlated g_0, g_1: Q_u != 0
g[..., 2] = 0.3 * box.random_field(rng, (1, 2.5), 2)
g /= math.sqrt(P.free_energy(box, g, sl.p))
st = P.make_state(box, zp, zm, g, Lam, beta)
Wm = P.W_m(box, g)
e1 = float(np.abs(Wm * box.Ntot ** 2 / np.asarray(hermite_moment_energy(st)) - 1).max())
e2 = abs(P.free_energy(box, g, sl.p) * box.Ntot ** 2 / hermite_free_energy(st) - 1)
EA = P.alfvenic_energy(box, zp, zm)
e3 = abs(EA["E"] * box.N * box.Ntot / krmhd_energy(st)["total"] - 1)
hf = np.asarray(hermite_flux(st))
pz_g = s * sl.p[:M] * np.sum(box.w[..., None] * hf, axis=(0, 1, 2)) / box.Ntot ** 2
pz = P.hermite_flux_kz(box, g, s, sl)
e4 = float(np.abs(pz - pz_g).max() / np.abs(pz).max())
e5 = abs(EA["E_plus"] + EA["E_minus"] - EA["E"]) / EA["E"]
check("C1 W(m) = hermite_moment_energy / N^6; W = hermite_free_energy / N^6", max(e1, e2) < 1e-12, f"{e1:.1e}, {e2:.1e}")
check("C2 Alfvenic energy E = krmhd.physics.energy / (N_z N^3); E = E_plus + E_minus", max(e3, e5) < 1e-12,
      f"{e3:.1e}, {e5:.1e}")
check("C3 k_z-only flux = s p_m x (hermite_flux summed over modes)", e4 < 1e-12, f"{e4:.1e}")
chi = P.nonlinearity_parameter(box, zp, zm)
print(f"     chi_u = {chi['chi_u']:.2f}, chi_z = {chi['chi_z']:.2f}, kbar_perp/k1 = {chi['kperp_mean'] / box.k1:.2f}, "
      f"kbar_par/k1 = {chi['kpar_mean'] / box.k1:.2f}; Q_u/Q_1 = {P.Q_phi(box, sl.amplitudes(g), sl.u)['Q_u']:.3f}")

# ---------------------------------------------------------------------------------------------------------------
print("E. one gandalf_step conserves Q_1 to time-stepping accuracy", flush=True)
lim = P.slice_advection_rate(box, zp, zm, s, sl.u)
dt0 = 0.25 * lim["dt_lawson_max"]
print(f"     slice-advection rate {lim['rate']:.1f} (fastest slice u = {sl.u[lim['j_fastest']]:+.3f}); "
      f"Lawson limit dt = {lim['dt_lawson_max']:.3e}; dt = {dt0:.3e}, {dt0 / 2:.3e}")
a0 = sl.amplitudes(g)
Q0 = P.Q_phi(box, a0, sl.u)
pW0 = sl.p * Wm
for scheme in ("lawson_rk4", "imex_rk222"):
    dQ, dpw, dQu = [], [], []
    for dt in (dt0, dt0 / 2):
        st1 = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, hyper_r=1, hyper_n=6, scheme=scheme)
        _, _, g1 = P.fields_of(st1)
        a1 = sl.amplitudes(g1)
        Q1 = P.Q_phi(box, a1, sl.u)
        dQ.append(abs(Q1["Q_1"] / Q0["Q_1"] - 1))
        dQu.append(abs(Q1["Q_u"] - Q0["Q_u"]) / P.Q_phi_scale(box, a0, sl.u, "u"))
        dpw.append(float(np.abs(sl.p * P.W_m(box, g1) - pW0).max() / Q0["Q_1"]))
    floor = 1e-14
    order_ok = (dQ[0] < floor and dQ[1] < floor) or dQ[0] / max(dQ[1], 1e-300) >= 6.0
    check(f"E1 {scheme}: |dQ_1|/Q_1 << change of the moment energies, falling at least like dt^3",
          dQ[1] < 1e-3 * dpw[1] and order_ok,
          f"|dQ_1|/Q_1 = {dQ[0]:.2e}, {dQ[1]:.2e} (ratio {dQ[0] / max(dQ[1], 1e-300):.1f}); "
          f"max_m |d(p_m W_m)|/Q_1 = {dpw[0]:.2e}, {dpw[1]:.2e}; |dQ_u|/scale = {dQu[0]:.1e}, {dQu[1]:.1e}")

# hypercollisional leakage out of a single slice in one Lawson step, against the first-order coupling
nu, hn = 2.0, 6
j0 = sl.nearest(1.0)
f = box.random_field(rng, (1, 2.5), 2)
gs = sl.load_single(f, j0)
st_s = P.make_state(box, zp, zm, gs, Lam, beta, nu=nu)
dt = dt0 / 2
st1 = gandalf_step(st_s, dt, eta=0.0, v_A=1.0, nu=nu, hyper_r=1, hyper_n=hn, scheme="lawson_rk4")
a1 = sl.amplitudes(np.asarray(st1.g))
K = P.slice_coupling_matrix(sl, P.coll_rates(M, nu, hn))
pred = P.leakage_first_order(K, j0, dt)["amplitude_ratio"]
E1 = P.slice_energies(box, a1)
meas = np.sqrt(np.delete(E1, j0) / E1[j0])
big = pred > 1e-3 * pred.max()
err = float(np.abs(meas[big] / pred[big] - 1).max())
lk = P.slice_leakage(box, a1, j0)
check("E2 one Lawson step with hypercollisions: |a_l|/|a_j0| = dt |K_(l j0)| to first order (slice j0 nearest u = 1)",
      err < 0.02 and np.allclose(K, K.T, atol=1e-14), f"max rel. deviation {err:.1e} (nu dt = {nu * dt:.1e}); "
      f"energy fraction outside j0 = {lk['energy_fraction_outside']:.2e}")

# ---------------------------------------------------------------------------------------------------------------
print("F. frozen-field RK4 and the exact k_z-only solution", flush=True)
dtF, nF = 2e-3, 20
fr = P.FrozenRK4(box, M, Lam, beta, dtF)
gF = np.asarray(fr.advance(psi, -psi, g, nF))
Ej0 = P.slice_energies(box, a0)
EjF = P.slice_energies(box, sl.amplitudes(gF))
e_sl = float(np.abs(EjF / Ej0 - 1).max())
moved = float(np.abs(P.W_m(box, gF) / Wm - 1).max())
check("F1 frozen field (z+, z-) = (Psi, -Psi): every slice energy conserved to RK4 accuracy", e_sl < 1e-9 and moved > 1e-3,
      f"max_j |dE_j/E_j| = {e_sl:.1e} while max_m |dW_m/W_m| = {moved:.2f}")
zero = np.zeros_like(zp)
gZ = np.asarray(fr.advance(zero, zero, g, nF))
gX = P.kz_only_evolve(box, g, sl, s, nF * dtF)
e_z = float(np.abs(gZ - gX).max() / np.abs(gX).max())
check("F2 frozen RK4 with z+- = 0 equals the exact k_z-only solution", e_z < 1e-8, f"max rel. diff {e_z:.1e}")

# ---------------------------------------------------------------------------------------------------------------
print("G. embedding, blob, plane", flush=True)
b8 = P.Box(8)
r8 = np.random.default_rng(1)
f8 = b8.random_field(r8, (1, 2.5), 2, extra=(2,))
f16 = box.embed(f8, 8)
x8 = b8.to_real(f8)
x16 = box.to_real(f16)[::2, ::2, ::2]
e_emb = float(np.abs(x8 - x16).max() / np.abs(x8).max())
e_rg = abs(b8.rms_grad(f8[..., 0]) / box.rms_grad(f16[..., 0]) - 1)
check("G1 embed 8^3 -> 16^3 keeps the physical field (values and rms gradient)", max(e_emb, e_rg) < 1e-13,
      f"{e_emb:.1e}, {e_rg:.1e}")
blob = box.gaussian_blob((box.L / 8, box.L / 8, box.L / 4), remove_mean=False)
xr = box.to_real(blob)
peak = xr[0, 8, 8]                       # centre (L/2, L/2, 0): z index 0, y = x = N/2
pl = box.plane_z(blob, 0.0)
pl2 = box.plane_z(blob, 0.0, pad=2)
e_pl = float(np.abs(pl - xr[0]).max())
e_pl2 = float(np.abs(pl2 - box.to_real(blob, pad=2)[0]).max())
check("G2 Gaussian blob: value 1 at its centre (periodic images < 1e-3); plane_z = full inverse transform",
      abs(peak - 1) < 1e-3 and max(e_pl, e_pl2) < 1e-13 and box.band_leakage(blob) == 0.0,
      f"peak {peak:.6f}; plane errors {e_pl:.1e}, {e_pl2:.1e}")

# ---------------------------------------------------------------------------------------------------------------
print("H. slice-advection rate bounds the spectral radius of the dealiased advection operator", flush=True)
r8b = np.random.default_rng(7)
phi8 = P.normalise_rms_grad(b8, b8.random_field(r8b, (1, 2.5), 2), 1.0)
psi8 = P.normalise_rms_grad(b8, b8.random_field(r8b, (1, 2.5), 2), 1.0)
sl3 = P.make_slices(3, Lam)
zp8, zm8 = P.elsasser(phi8, psi8)
lim8 = P.slice_advection_rate(b8, zp8, zm8, 1.0, sl3.u)
fj = phi8 + sl3.u[lim8["j_fastest"]] * psi8
n8 = 8 ** 3
Bmat = np.zeros((n8, n8))
for i in range(n8):
    e = np.zeros(n8)
    e[i] = 1.0
    ek = np.fft.rfftn(e.reshape(8, 8, 8)) * b8.mask
    Bmat[:, i] = np.fft.irfftn(b8.bracket(fj, ek), s=(8, 8, 8)).ravel()
ev = np.linalg.eigvals(Bmat)
rho = float(np.abs(ev).max())
check("H1 rho(P{Phi + s u_max Psi, .}) <= k_perp,max max|grad(Phi + s u Psi)| (8^3, M = 3)", rho <= lim8["rate"],
      f"rho = {rho:.2f}, estimate = {lim8['rate']:.2f} (ratio {rho / lim8['rate']:.2f}); "
      f"max |Re eig| / rho = {np.abs(ev.real).max() / rho:.1e}")

# ---------------------------------------------------------------------------------------------------------------
print("I. bookkeeping", flush=True)
rid = P.make_run_id("selftest")
okI = rid.startswith(P.RUN_PREFIX + "_selftest_") and len(rid.split("_")[-1]) == 6 and len(rid.split("_")[-2]) == 8
with tempfile.TemporaryDirectory() as tmp:
    cfg = {"N": 16, "M": M, "Lambda": "sqrt(5)", "beta_i": 1.0}
    run = P.Run("selftest", cfg, None, base=tmp)
    gates = P.make_gates(P.gate_energy_conservation([1.0, 1.0 + 1e-9], [2.0, 2.0 - 1e-8]),
                         resolution=P.resolution_check(box, sl, zp, zm, g))
    diag0 = P.diagnostics(box, sl, s, zp, zm, g)
    rec = P.stack_records([diag0, diag0])
    meta = run.save({"t": np.array([0.0, 1.0]), **rec, "nested": {"x": np.arange(3)}}, gates, notes="self-test")
    arrays, meta2 = P.load_run(run.dir)
    okI = okI and meta2["outcome"] == "pass" and meta2["run_id"] == run.run_id and arrays["W_m"].shape == (2, M + 1)
    okI = okI and "nested__x" in arrays and meta2["gates"]["energy_balance"] == P.NOT_APPLICABLE_SHORT
    okI = okI and P.latest_run_dir("selftest", base=tmp) == run.dir and json.loads(json.dumps(meta)) == meta2
    okI = okI and meta2["precision"].startswith("float64") and "krmhd_version" in meta2
    row = P.run_log_row(meta2)
bad = P.make_gates(P.gate_energy_conservation([1.0, 1.02]))
okI = okI and P.gates_outcome(bad) == "fail"
try:
    P.require_gates(bad)
    okI = False
except RuntimeError:
    pass
try:
    P.steps_for_times([0.062, 0.125], 0.002)
    okI = False
except ValueError:
    pass
okI = okI and P.steps_for_times([0.062, 0.124, 0.25], 0.002) == [31, 62, 125]
okI = okI and abs(P.parse_number("sqrt5") - P.SQRT5) < 1e-15 and P.parse_number("-sqrt(5)") == -P.SQRT5
okI = okI and math.isinf(P.parse_number("inf")) and abs(P.parse_number("2*pi") - 2 * math.pi) < 1e-15
check("I1 run ID, .npz/.json round trip, gates, output times, parse_number", okI, row[:90] + " ...")

print(f"== test_p1lib: {NCHK[0] - len(FAILED)}/{NCHK[0]} checks passed ({time.time() - T0:.1f} s)", flush=True)
if FAILED:
    print("FAILED:", *FAILED, sep="\n  ")
    sys.exit(1)
