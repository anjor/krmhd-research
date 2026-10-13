"""c06_cfl: independent check of the time-step limit set by the fastest slices, on a 3-D random state through
gandalf_step (independent of the 2-D shear-flow test in ../../truncated/t7a_cfl_slices.py), and of the IMEX
hyper-resistive stability threshold for r = 1, 2, 4. Supports Appendix D of the paper (summarised in Section 8).

a. A 3-D random (turbulence-like) state, 16^3, M = 64, Lam = sqrt5, beta = 1, max|grad_perp z+-| ~ 5 (so GANDALF's v_max
   is set by |grad z+-|, not v_A): Lawson-RK4 through gandalf_step at GANDALF's dt_cfl. Prediction: the fastest slices
   (|u| ~ u_max = 10.6) blow up within a few tens of steps, the slow ones (|u| < 2) do not; at dt_cfl/8 nothing grows.
   Also IMEX-RK222 at dt_cfl with eta = 0: fastest slices grow.
b. The per-mode amplification of the explicit IMEX stage with a hyper-resistive factor, exact versus the small-y formula
   dt^3 <= 8 eta/(k_max V)^4 (r = 2), and what happens for r = 1 and r = 4 (k-dependence of the threshold).
"""
from __future__ import annotations

import time

import numpy as np

from cc import Checks, hdr, SQRT5, grid3, rand_field, state, gandalf_eig, jnp

C = Checks("c06_cfl")
from krmhd.timestepping import gandalf_step, compute_cfl_timestep
from krmhd.spectral import derivative_x, derivative_y, rfftn_inverse

hdr("a. 3-D random state: slice energies under gandalf_step at GANDALF's dt_cfl")
N = 16
g16 = grid3(N)
M, Lam, beta = 64, SQRT5, 1.0
T, u, P, Pinv = gandalf_eig(M, Lam, "zero")
rng = np.random.default_rng(2024)
zp = rand_field(g16, rng, amp=1.0)
zm = rand_field(g16, rng, amp=1.0)


def maxgrad(f):
    fx = np.asarray(rfftn_inverse(derivative_x(jnp.asarray(f), g16.kx), N, N, N))
    fy = np.asarray(rfftn_inverse(derivative_y(jnp.asarray(f), g16.ky), N, N, N))
    return float(np.sqrt(fx ** 2 + fy ** 2).max())


sc = 5.0 / max(maxgrad(zp), maxgrad(zm))
zp, zm = zp * sc, zm * sc
g0 = rand_field(g16, rng, slope=-1.0, extra=(M + 1,)) * 1e-3
st0 = state(g16, M, Lam, zp, zm, g0, beta=beta)
dtc = compute_cfl_timestep(st0, v_A=1.0)
Vfast = maxgrad((zp + zm) / 2 + u[-1] * (zp - zm) / 2)
kmax = 2 * np.pi * (N // 3) * np.sqrt(2)
print(f"   max|grad z+| = {maxgrad(zp):.2f}, max|grad z-| = {maxgrad(zm):.2f}, s max|grad Psi| = {maxgrad((zp - zm) / 2):.2f}; "
      f"dt_cfl = {dtc:.3e}; fastest slice speed {Vfast:.2f}; y = dt_cfl k_max V = {dtc * kmax * Vfast:.2f} (RK4 limit 2.83)")
fast = np.abs(u) > 8.0
slow = np.abs(u) < 2.0


def slice_E(st):
    a = np.einsum("zyxm,jm->zyxj", np.asarray(st.g), Pinv)
    return (np.abs(a) ** 2).sum(axis=(0, 1, 2))


runs = []
res = {}
for scheme, dt, nst in (("lawson_rk4", dtc, 30), ("lawson_rk4", dtc / 8, 240), ("imex_rk222", dtc, 30)):
    st = st0
    E0 = slice_E(st)
    t0 = time.time()
    for _ in range(nst):
        st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, scheme=scheme)
    wall = time.time() - t0
    E1 = slice_E(st)
    gf = E1[fast].sum() / E0[fast].sum()
    gs = E1[slow].sum() / E0[slow].sum()
    res[(scheme, dt)] = (gf, gs)
    runs.append((scheme, dt, nst, wall))
    print(f"   {scheme}, dt = {dt:.3e} ({dt / dtc:.3f} dt_cfl), {nst} steps to t = {dt * nst:.3f}: energy ratio fast slices "
          f"(|u| > 8) {gf:.3e}, slow slices (|u| < 2) {gs:.3e}  [{wall:.1f} s]")
# effective y: spectral radius of the discrete explicit operator a -> -mask{f_j, a} (anti-self-adjoint on the band),
# by power iteration on -N^2, for the fastest and the u ~ 0 slice
from cc import pb, real_project
mask = np.asarray(g16.dealias_mask)


def Nop(f, a):
    out = -pb(g16, f, a) * mask
    out[0, 0, 0] = 0
    return out


def spec_radius(f, iters=300):
    a = rand_field(g16, np.random.default_rng(1), slope=0.0)
    lam = 0.0
    for _ in range(iters):
        b = Nop(f, Nop(f, a))
        nb = np.sqrt(np.sum(np.abs(b) ** 2))
        lam = np.sqrt(nb / np.sqrt(np.sum(np.abs(a) ** 2)))
        a = -b / nb
    return lam


rho_fast = spec_radius((zp + zm) / 2 + u[-1] * (zp - zm) / 2)
rho_slow = spec_radius((zp + zm) / 2 + u[np.argmin(np.abs(u))] * (zp - zm) / 2)
print(f"   spectral radius of the explicit operator: fastest slice {rho_fast:.1f}, u ~ 0 slice {rho_slow:.1f}; "
      f"y_eff(dt_cfl) = dt_cfl rho = {dtc * rho_fast:.2f} (fast), {dtc * rho_slow:.2f} (slow); nominal k_max V dt_cfl = "
      f"{dtc * kmax * Vfast:.2f}")
# Lawson-RK4 just above and below its own limit y_eff = 2 sqrt2 for the fastest slice
for fac, nst in ((1.25, 40), (0.85, 40)):
    dt = fac * 2 * np.sqrt(2) / rho_fast
    st = st0
    E0 = slice_E(st)
    t0 = time.time()
    for _ in range(nst):
        st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, scheme="lawson_rk4")
    wall = time.time() - t0
    E1 = slice_E(st)
    res[("lawson_y", fac)] = (E1[fast].sum() / E0[fast].sum(), E1[slow].sum() / E0[slow].sum())
    runs.append(("lawson_rk4", dt, nst, wall))
    print(f"   lawson_rk4, dt = {dt:.3e} (y_eff = {dt * rho_fast:.2f}, {dt / dtc:.2f} dt_cfl), {nst} steps: fast-slice energy "
          f"ratio {res[('lawson_y', fac)][0]:.3e}, slow {res[('lawson_y', fac)][1]:.3e}  [{wall:.1f} s]")
okA = (res[("imex_rk222", dtc)][0] > 1e3 and res[("imex_rk222", dtc)][1] < 10 and
       res[("lawson_y", 1.25)][0] > 1e3 and res[("lawson_y", 0.85)][0] < 10 and res[("lawson_y", 1.25)][1] < 10)
C("c06.a IMEX at dt_cfl: fastest slices blow up, slow ones do not. Lawson-RK4 obeys y_eff = dt rho_fast <= 2 sqrt2, with "
  "rho_fast the spectral radius of the discrete advection of the fastest slice (here GANDALF's dt_cfl happens to be "
  "Lawson-stable because y_eff(dt_cfl) < 2 sqrt2 although k_max V dt_cfl > 2 sqrt2)", okA)
print("GANDALF integrations (scheme, dt, steps, wall s), 16^3, M = 64, Lam = sqrt5:")
for r in runs:
    print(f"   {r[0]} dt={r[1]:.3e} steps={r[2]} wall={r[3]:.1f}s")

hdr("b. IMEX explicit stage with hyper-resistivity: exact threshold vs the small-y formula, r = 1, 2, 4")
from scipy.optimize import brentq
Ng = 128
kmax2 = 2 * (2 * np.pi * ((Ng - 1) // 3)) ** 2       # GANDALF's k_perp,max^2 (corner of the 2/3 box)
kmx = np.sqrt(kmax2)
eta, V = 100.0, 30.0
okB = True
for r in (1, 2, 4):
    ks = 2 * np.pi * np.arange(1, (Ng - 1) // 3 + 1) * np.sqrt(2)     # diagonal wavevectors along the flow
    thr = []
    for k in ks:
        G = lambda dt: 0.5 * np.log(1 + (k * V * dt) ** 4 / 4) - eta * (k * k / kmax2) ** r * dt
        # smallest dt > 0 at which the mode becomes unstable
        dts = np.logspace(-9, 0, 2000)
        vals = np.array([G(d) for d in dts])
        i = np.argmax(vals > 0) if np.any(vals > 0) else None
        thr.append(brentq(G, dts[i - 1], dts[i]) if i else np.inf)
    thr = np.array(thr)
    formula = (8 * eta) ** (1 / 3) / (kmx * V) ** (4 / 3)
    imin = int(np.argmin(thr))
    print(f"   r={r}: smallest unstable dt over k = {thr.min():.3e} at |k|/k_max = {ks[imin] / kmx:.3f}; at k_max: "
          f"{thr[-1]:.3e}; formula (8 eta)^(1/3)/(k_max V)^(4/3) = {formula:.3e}")
    if r == 2:
        okB = okB and abs(thr[-1] / formula - 1) < 0.3 and thr.max() / thr.min() < 1.5
    if r == 4:
        okB = okB and thr.min() < 0.05 * formula
C("c06.b r = 2: threshold ~k-independent and close to the formula; r = 4: the lowest k sets a far smaller threshold "
  "(the formula is specific to r <= 2)", okB)
C.done()
