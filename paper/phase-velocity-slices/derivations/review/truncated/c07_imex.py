"""c07_imex: independent checks of the coupling of g to z+- in GANDALF's IMEX-RK222 step, which is second order in dt
in krmhd >= 0.7.0 (gandalf#158; stage 2 samples z+- at t + gamma dt) and was first order in krmhd <= 0.6.0 (z+- at
t + dt/2): a scalar non-autonomous test of the ARS(2,2,2) stage times, and the convergence order of g(T) and of the W
drift through gandalf_step in a configuration different from that of ../../truncated/t7b_imex_order.py. Supports
Appendix D of the paper.

Independent routes:

a. Plain numerics (no SymPy) for the non-autonomous scalar problem y' = lam_I y + i omega(t) y with
   omega(t) = 3 + 2 sin(1.3 t): ARS(2,2,2) with the implicit part lam_I (= 5i or 0) and the explicit stage-2 evaluated at
   time t + c dt, c = gamma (consistent; GANDALF >= 0.7.0) or 1/2 (GANDALF <= 0.6.0). Global order of y(T) and of the
   |y|^2 drift.
b. GANDALF, a configuration different from that of t7b_imex_order.py: 8^3, M = 8, Lam = -sqrt5, beta = 2, a single propagating
   z+ (z- = 0, so the Elsasser update is exact and Lawson-RK4 is a 4th-order reference), dt = T/16 ... T/1024:
   order of g(T) for gandalf_step IMEX (second order since gandalf#158) and of the W drift.
"""
from __future__ import annotations

import time

import numpy as np

from cc import Checks, hdr, SQRT5, grid3, rand_field, state, jnp

C = Checks("c07_imex")
from krmhd.timestepping import gandalf_step, _IMEX_GAMMA as GAM, _IMEX_DELTA as DEL

hdr("a. Scalar non-autonomous test of ARS(2,2,2) stage times")
om = lambda t: 3.0 + 2.0 * np.sin(1.3 * t)
T = 2.0


def exact(lamI):
    # y(T) = exp(lamI T + i int_0^T omega)
    I = 3.0 * T + (2.0 / 1.3) * (1 - np.cos(1.3 * T))
    return np.exp(lamI * T + 1j * I)


def ars(lamI, c, n):
    dt = T / n
    y = 1.0 + 0j
    t = 0.0
    for _ in range(n):
        N1 = 1j * om(t) * y
        y1 = (y + dt * GAM * N1) / (1 - dt * GAM * lamI)
        N2 = 1j * om(t + c * dt) * y1
        y = (y + dt * (1 - GAM) * lamI * y1 + dt * DEL * N1 + dt * (1 - DEL) * N2) / (1 - dt * GAM * lamI)
        t += dt
    return y


okA = True
ns = np.array([64, 128, 256, 512, 1024, 2048, 4096, 8192])
for lamI in (0.0, 5j):
    for c, lab in ((GAM, "gamma"), (0.5, "1/2")):
        errs = np.array([abs(ars(lamI, c, n) - exact(lamI)) for n in ns])
        drift = np.array([abs(abs(ars(lamI, c, n)) ** 2 - abs(exact(lamI)) ** 2) for n in ns])
        o = np.log2(errs[:-1] / errs[1:])
        od = np.log2(drift[:-1] / drift[1:])
        print(f"   lam_I = {lamI}, stage time {lab}: orders of y(T) {np.round(o, 3).tolist()}; orders of the |y|^2 drift "
              f"{np.round(od, 3).tolist()}")
        if lab == "gamma":
            okA = okA and abs(o[-1] - 2.0) < 0.01
        else:
            okA = okA and np.all(np.diff(o) < 0) and o[-1] < 1.06
        okA = okA and np.all(np.diff(od) < 0) and abs(od[-1] - 2.0) < 0.06
C("c07.a stage time 1/2 -> first order, gamma -> second order; |y|^2 drift second order for both", okA)

hdr("b. GANDALF: single z+ wave, 8^3, M = 8, Lam = -sqrt5, beta = 2")
N = 8
g8 = grid3(N)
M, Lam, beta = 8, -SQRT5, 2.0
rng = np.random.default_rng(77)
zp = rand_field(g8, rng, amp=1.0)
from krmhd.spectral import derivative_x, derivative_y, rfftn_inverse
fx = np.asarray(rfftn_inverse(derivative_x(jnp.asarray(zp), g8.kx), N, N, N))
fy = np.asarray(rfftn_inverse(derivative_y(jnp.asarray(zp), g8.ky), N, N, N))
zp = zp * 3.0 / np.sqrt(fx ** 2 + fy ** 2).max()
zm = np.zeros_like(zp)
g = rand_field(g8, rng, slope=-1.0, extra=(M + 1,))
st0 = state(g8, M, Lam, zp, zm, g, beta=beta)
p = np.ones(M + 1); p[0] = 1 - 1 / Lam
w = np.full(N // 2 + 1, 2.0); w[0] = 1.0; w[-1] = 1.0
Wq = lambda gg: float(np.sum(w[None, None, :, None] * p * np.abs(gg) ** 2))
Tend = 0.25


def run(scheme, n):
    st = st0
    for _ in range(n):
        st = gandalf_step(st, Tend / n, eta=0.0, v_A=1.0, nu=0.0, scheme=scheme)
    return np.asarray(st.g)


t0 = time.time()
gref = run("lawson_rk4", 2048)
t_ref = time.time() - t0
t0 = time.time()
lw = [run("lawson_rk4", n) for n in (128, 256)]
t_lw = time.time() - t0
el = [np.linalg.norm(x - gref) / np.linalg.norm(gref) for x in lw]
print(f"   Lawson-RK4 reference (2048 steps, {t_ref:.1f} s); Lawson errors at T/128, T/256: {el[0]:.2e}, {el[1]:.2e} "
      f"(order {np.log2(el[0] / el[1]):.2f})")
t0 = time.time()
ns = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
errs, drift = [], []
W0 = Wq(np.asarray(st0.g))
for n in ns:
    gg = run("imex_rk222", n)
    errs.append(np.linalg.norm(gg - gref) / np.linalg.norm(gref))
    drift.append((Wq(gg) - W0) / W0)
t_imex = time.time() - t0
o = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
od = np.log2(np.abs(np.array(drift[:-1]) / np.array(drift[1:])))
print(f"   IMEX ({sum(ns)} steps, {t_imex:.1f} s): g(T) errors {[f'{e:.2e}' for e in errs]}; orders {np.round(o, 3).tolist()}")
print(f"   IMEX signed W drift (W(T)-W(0))/W(0) {[f'{e:+.2e}' for e in drift]}; local orders {np.round(od, 3).tolist()}")
C("c07.b gandalf_step IMEX: g(T) second order (krmhd >= 0.7.0, gandalf#158; first order in <= 0.6.0) (single wave, "
  "Lam = -sqrt5, beta = 2)", abs(o[-1] - 2) < 0.05, f"finest order {o[-1]:.3f}")
C("c07.c W drift of gandalf_step IMEX tends to second order (local order of the finest pair within 0.15 of 2)",
  abs(od[-1] - 2) < 0.15, f"finest local order {od[-1]:.3f}")
print(f"GANDALF integrations: 8^3, M = 8: Lawson reference 2048 steps + 384 steps, IMEX {sum(ns)} steps; total "
      f"{t_ref + t_imex:.1f} s")
print(f"   lawson_rk4 reference steps=2048 dt={Tend / 2048:.3e} wall={t_ref:.1f}s")
print(f"   lawson_rk4 steps=128+256 wall={t_lw:.1f}s")
print(f"   imex_rk222 steps={'+'.join(str(n) for n in ns)} = {sum(ns)} wall={t_imex:.1f}s")
C.done()
