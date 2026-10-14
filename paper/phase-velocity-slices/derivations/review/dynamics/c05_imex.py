"""c05: stability and order of GANDALF's IMEX-RK222 step for the slice advection, checked independently of d03.  The
script derives the stability function of the coded ARS(2,2,2) stages symbolically, evaluates the explicit-advection
bound with the full advecting field Phi + s u_j Psi, continues the run of d03 with gandalf_step to t = 0.4
(IMEX-RK222 at dt = 0.002 and 0.001, Lawson-RK4 at dt = 0.002), where IMEX-RK222 at dt = 0.002 blows up while the
conserved free energy stays constant in the reference runs, and it measures self-convergence orders in dt.  It
supports Section 6 (the frozen-line test through gandalf_step) and the time-stepping analysis of Appendix D.

Statement under test: the explicit part of ARS(2,2,2) integrates the slice advection s u_j {Psi, a_j} with
R(z) = 1 + z + z^2/2, unstable on the imaginary axis; at N = 32, dt = 0.002, y = s u_max max|grad Psi| k_perp,max dt
= 1.51, and d03 sees a departure from the frozen-line prediction by t = 0.128, where it stops; second order in dt
otherwise (krmhd >= 0.7.0, gandalf#158; krmhd <= 0.6.0 coupled g to z+- at first order).

I1 symbolic: the stage structure of krmhd's _gandalf_step_imex222_jit with GANDALF's own (gamma, delta) gives
   R(zE, zI); R(zE, 0) = 1 + zE + zE^2/2 exactly; |R(iy, 0)|^2 = 1 + y^4/4.
I2 the combined amplification |R(iy, iw)| for the implicit streaming w = s u_j k_z dt of the d03 case.
I3 the explicit advection bound with the full advecting field Phi + s u_j Psi (d03 used Psi only).
I4 numerical: d03's setup (its initial fields, built with derivations/dynamics/dcore.py), gandalf_step IMEX at
   dt = 0.002 continued beyond t = 0.128, with Lawson-RK4 at dt = 0.002 and IMEX at dt = 0.001 as references; the
   conserved free energy W = sum p_m <g_m^2> (eta = nu = 0) and the share of g energy near the cutoff are monitored.
I5 self-convergence orders at t = 0.064 for both schemes (dt = 1e-3, 5e-4, 2.5e-4).
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "dynamics"))   # derivations/dynamics/dcore.py
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from krmhd.timestepping import _IMEX_DELTA, _IMEX_GAMMA, gandalf_step  # noqa: E402
import dcore  # noqa: E402
from dcore import Box, Checks, SQRT5, W_m, ladder, log_run, make_state, pweights, timer  # noqa: E402

# the GANDALF runs of this script are logged with the other checks of this folder (git-ignored data folder)
dcore.RUNLOG = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), "data", "derivations", "review",
                            "dynamics", "gandalf_runs.jsonl")

chk = Checks()

# ------------------------------------------------------------------------------------------------ I1
g_, zE, zI, y, w = sp.symbols("gamma z_E z_I y w")
gam = (2 - sp.sqrt(2)) / 2
dlt = 1 - 1 / (2 * gam)
chk("I1 GANDALF's (gamma, delta) are ARS(2,2,2)'s", abs(float(gam) - _IMEX_GAMMA) < 1e-15 and abs(float(dlt) - _IMEX_DELTA) < 1e-15,
    f"gamma = {_IMEX_GAMMA:.15f}, delta = {_IMEX_DELTA:.15f}")
# stage structure of _gandalf_step_imex222_jit for d_t g = L g + N(g) with L g = (zI/dt) g, N(g) = (zE/dt) g:
#   (1 - gamma zI) g1 = g + gamma zE g
#   (1 - gamma zI) g_new = g + (1 - gamma) zI g1 + delta zE g + (1 - delta) zE g1
g1 = (1 + gam * zE) / (1 - gam * zI)
R = (1 + (1 - gam) * zI * g1 + dlt * zE + (1 - dlt) * zE * g1) / (1 - gam * zI)
RE = sp.simplify(sp.expand(R.subs(zI, 0)))
chk("I1 R(zE, 0) = 1 + zE + zE^2/2 (explicit part of ARS(2,2,2) as coded)", sp.simplify(RE - (1 + zE + zE ** 2 / 2)) == 0, str(RE))
mod2 = sp.simplify(sp.expand((RE.subs(zE, sp.I * y)) * sp.conjugate(RE.subs(zE, sp.I * y))).subs(sp.conjugate(y), y))
chk("I1 |R(iy, 0)|^2 = 1 + y^4/4", sp.simplify(mod2 - (1 + y ** 4 / 4)) == 0, str(mod2))
RI = sp.simplify(R.subs(zE, 0))
Rf = sp.lambdify((zE, zI), R, "numpy")

# ------------------------------------------------------------------------------------------------ I2, I3, I4 setup
N, M, LAM, BETA, AMP, N0 = 32, 8, SQRT5, 1.0, 1.0, 8
S = math.sqrt(BETA)
A = ladder(M, LAM)
P = pweights(M, LAM)
box = Box(N)


def initial_fields(seed: int = 1):           # verbatim recipe of d03 (same seed, same 8^3 embedding)
    b0 = Box(N0)
    rng = np.random.default_rng(seed)
    phi = b0.random_field(rng, (1, 2.5), 2)
    psi = b0.random_field(rng, (1, 2.5), 2)
    g = np.zeros((N0, N0, N0 // 2 + 1, M + 1), complex)
    g[..., 0] = b0.random_field(rng, (1, 2.5), 2)
    g[..., 1] = 0.5 * b0.random_field(rng, (1, 2.5), 2)
    return phi, psi, g


phi8, psi8, g8 = initial_fields()
phi = box.embed(phi8, N0)
psi = box.embed(psi8, N0)
phi *= AMP / box.rms_grad(phi)
psi *= AMP / box.rms_grad(psi)
g0 = box.embed(g8, N0)
g0 /= np.sqrt(np.sum(W_m(box, g0) * P))
u = np.sort(np.real(np.linalg.eigvals(A)))
kperp_max = float(np.sqrt((box.KP2 * box.mask).max()))
kz_max = float(np.abs(box.KZ * box.mask).max())


def grad_max(f):
    gx = np.fft.irfftn(1j * box.KX * f, s=(N, N, N), axes=(0, 1, 2))
    gy = np.fft.irfftn(1j * box.KY * f, s=(N, N, N), axes=(0, 1, 2))
    return float(np.sqrt(gx ** 2 + gy ** 2).max())


for dt in (0.002, 0.001):
    y_psi = S * np.abs(u).max() * grad_max(psi) * kperp_max * dt
    y_full = max(grad_max(phi + S * uj * psi) for uj in u) * kperp_max * dt
    wmax = S * np.abs(u).max() * kz_max * dt
    amp_psi = abs(Rf(1j * y_psi, 0))
    amp_full = abs(Rf(1j * y_full, 0))
    comb = max(abs(Rf(1j * y_full, 1j * ww)) for ww in np.linspace(-wmax, wmax, 401))
    print(f"    dt = {dt}: y (Psi only, as in d03) = {y_psi:.2f}, |R| = {amp_psi:.3f};  y with the full advecting field "
          f"Phi + s u_j Psi = {y_full:.2f}, |R| = {amp_full:.3f};  with implicit streaming |w| <= {wmax:.2f}: max |R| = {comb:.3f}")
    if dt == 0.002:
        chk("I3 the full advecting field gives a larger explicit eigenvalue than the Psi-only bound", y_full > y_psi,
            f"{y_full:.2f} vs {y_psi:.2f}")
        chk("I2 the implicit streaming does not stabilise it", comb > 1.2, f"max |R| = {comb:.3f}")

# ------------------------------------------------------------------------------------------------ I4
T_END = 0.4


def run(scheme: str, dt: float):
    st = make_state(box, phi + psi, phi - psi, g0, LAM, BETA, nu=0.0)
    n_end = int(round(T_END / dt))
    every = int(round(0.016 / dt))
    kk = np.sqrt(box.KP2 + box.KZ ** 2)
    outer = (kk > 0.75 * (2 * np.pi * N / 3)) & box.mask
    hist = []
    W0 = float(np.sum(P * W_m(box, st.g)))
    clock = timer()
    blew = None
    for n in range(1, n_end + 1):
        st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, hyper_r=1, hyper_n=4, scheme=scheme, eta_z=0.0)
        if n % every == 0 or n == n_end:
            g = np.asarray(st.g)
            Wn = float(np.sum(P * W_m(box, g)))
            fo = float(np.sum(box.w[..., None] * outer[..., None] * np.abs(g) ** 2) / np.sum(box.w[..., None] * np.abs(g) ** 2))
            hist.append((n * dt, Wn / W0, fo, float(np.abs(g).max())))
            if not np.isfinite(Wn) or Wn / W0 > 1e6:
                blew = n * dt
                break
    log_run("c05_imex.py", f"d03 setup continued to t = {T_END}: {scheme} dt = {dt}", N, M, n, clock(), dict(dt=dt, T=n * dt))
    return hist, blew


out = {}
for scheme, dt in (("imex_rk222", 0.002), ("lawson_rk4", 0.002), ("imex_rk222", 0.001)):
    hist, blew = run(scheme, dt)
    out[(scheme, dt)] = (hist, blew)
    print(f"  {scheme} dt = {dt}: " + ("BLOW-UP at t = %.3f; " % blew if blew else "")
          + "t, W/W0, cutoff-shell share of g energy: "
          + "; ".join(f"{t:.3f} {r:.3g} {fo:.1e}" for t, r, fo, _ in hist[::2]), flush=True)
hI, bI = out[("imex_rk222", 0.002)]
hL, bL = out[("lawson_rk4", 0.002)]
W_at = lambda h, t: [r for tt, r, _, _ in h if abs(tt - t) < 1e-9]  # noqa: E731
chk("I4 Lawson-RK4 at dt = 0.002 keeps W conserved to the end (reference)", bL is None and abs(hL[-1][1] - 1) < 1e-2,
    f"W/W0 at t = {hL[-1][0]:.3f}: {hL[-1][1]:.4f}")
grow = [r for _, r, _, _ in hI]
print(f"    IMEX dt = 0.002: W/W0 at t = 0.128: {W_at(hI, 0.128)}, final {hI[-1][1]:.3g} at t = {hI[-1][0]:.3f}")
chk("I4 IMEX at dt = 0.002 is unstable: W (conserved by the equations) grows without bound after t = 0.128",
    bI is not None or hI[-1][1] > 10, f"{'blow-up at t = %.3f' % bI if bI else 'W/W0 = %.3g at t = %.3f' % (hI[-1][1], hI[-1][0])}")

# ------------------------------------------------------------------------------------------------ I5
# self-convergence order (no reference solution needed): e(dt) = |X(dt) - X(dt/2)|, order = log2(e(dt)/e(dt/2))
T5 = 0.064


def final_state(scheme, dt):
    st = make_state(box, phi + psi, phi - psi, g0, LAM, BETA, nu=0.0)
    clock = timer()
    n_end = int(round(T5 / dt))
    for _ in range(n_end):
        st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, hyper_r=1, hyper_n=4, scheme=scheme, eta_z=0.0)
    log_run("c05_imex.py", f"self-convergence to t = {T5}: {scheme} dt = {dt}", N, M, n_end, clock(), dict(dt=dt, T=T5))
    return np.asarray(st.g), np.asarray(st.z_plus)


orders = {}
for scheme in ("imex_rk222", "lawson_rk4"):
    sts = {dt: final_state(scheme, dt) for dt in (0.001, 0.0005, 0.00025)}
    dts = sorted(sts, reverse=True)
    e_g = [np.sqrt(np.sum(np.abs(sts[a][0] - sts[b][0]) ** 2) / np.sum(np.abs(sts[b][0]) ** 2)) for a, b in zip(dts[:-1], dts[1:])]
    e_W = [np.abs(W_m(box, sts[a][0]) - W_m(box, sts[b][0])).sum() / W_m(box, sts[b][0]).sum() for a, b in zip(dts[:-1], dts[1:])]
    e_z = [np.sqrt(np.sum(np.abs(sts[a][1] - sts[b][1]) ** 2) / np.sum(np.abs(sts[b][1]) ** 2)) for a, b in zip(dts[:-1], dts[1:])]
    o_g, o_W, o_z = np.log2(e_g[0] / e_g[1]), np.log2(e_W[0] / e_W[1]), np.log2(e_z[0] / e_z[1])
    orders[scheme] = (o_g, o_W, o_z)
    print(f"    {scheme}: self-convergence at t = {T5} (dt = 1e-3, 5e-4, 2.5e-4): g field {e_g[0]:.2e}, {e_g[1]:.2e} -> order "
          f"{o_g:.2f};  W(m) {e_W[0]:.2e}, {e_W[1]:.2e} -> order {o_W:.2f};  z+ {e_z[0]:.2e}, {e_z[1]:.2e} -> order {o_z:.2f}")
chk("I5 IMEX-RK222: the g field converges at second order in dt (krmhd >= 0.7.0, gandalf#158; first order in <= 0.6.0)",
    1.8 < orders["imex_rk222"][0] < 2.3, f"order {orders['imex_rk222'][0]:.2f}")
chk("I5 Lawson-RK4: the g field converges at (at least) second order", orders["lawson_rk4"][0] > 1.8,
    f"order {orders['lawson_rk4'][0]:.2f}")
chk.finish()
