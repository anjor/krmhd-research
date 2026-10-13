"""d03: frozen-line theorem through GANDALF's actual time stepper, gandalf_step (krmhd 0.6.0).  At N = 32 the script
compares gandalf_step (Lawson-RK4 and IMEX-RK222 at dt = 0.002, 0.001, 0.0005, without and with hypercollisions) with
the frozen-line prediction, and checks that the W(m) difference is set by the spatial resolution for Lawson-RK4, is
first order in dt for IMEX-RK222, and that IMEX-RK222 at dt = 0.002 departs from the prediction by t = 0.128.  It
supports Section 6 (frozen-line theorem through GANDALF's stepper); the IMEX results are also used in Appendix D.

NL run: gandalf_step with eta = 0, eta_z = 0 (ideal z+-, no spatial damping of g), scheme lawson_rk4 or imex_rk222,
        hypercollisions nu (m/M)^4 (m >= 2) or none.
FR run: frozen-line prediction d_t h = -s grad_par(0) A h - C h, GANDALF RHS with (z+, z-) = (Psi0, -Psi0),
        classical RK4 with dt_FR = 0.0005 (time error negligible).
Same physical initial fields as d02 (8^3 seed embedded).  Reports the W(m) and Pi_m (full grad_par) differences
at t = 0.0625 and 0.125 for dt and dt/2, so the stepper's time error is visible next to the d02 spatial error.
Expected: Lawson-RK4 error O(dt^2) (midpoint z+- step); IMEX-RK222 error O(dt) (first-order g-z+- coupling,
gandalf#158); both converge towards the d02 RK4 values at the same N.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from krmhd.timestepping import gandalf_step  # noqa: E402
from dcore import (Box, Checks, SQRT5, W_m, coll_rates, flux_full, ladder, log_run, make_rhs, make_rk4,  # noqa: E402
                   make_state, maxrel, pweights, relerr, timer)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
M, LAM, BETA, AMP, N0 = 8, SQRT5, 1.0, 1.0, 8
S = np.sqrt(BETA)
A = ladder(M, LAM)
P = pweights(M, LAM)
TOUT = (0.064, 0.128)          # multiples of every dt used (0.002, 0.001, 0.0005)
box = Box(N)


def initial_fields(seed: int = 1):
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
rhs_full, rhs_g = make_rhs(box, M, LAM, BETA)


def frozen(nu: float) -> dict:
    dtF = 0.0005
    _, stepF = make_rk4(rhs_full, rhs_g, coll_rates(M, nu, 4), dtF)
    g = jnp.asarray(g0)
    out = {}
    n = 0
    zpF, zmF = jnp.asarray(psi), jnp.asarray(-psi)
    clock = timer()
    for t in TOUT:
        while n < int(round(t / dtF)):
            g = stepF(zpF, zmF, g)
            n += 1
        out[t] = (W_m(box, g), flux_full(box, g, psi, S, A, P))
        assert abs(n * dtF - t) < 1e-12
    log_run("d03_frozen_gandalf_step.py", f"frozen-line prediction (RK4 on GANDALF RHS, z+- frozen, nu={nu})", N, M,
            n, clock(), dict(dt=dtF, T=TOUT[-1]))
    return out


def nonlinear(scheme: str, dt: float, nu: float) -> dict:
    st = make_state(box, phi + psi, phi - psi, g0, LAM, BETA, nu=nu)
    out = {}
    n = 0
    clock = timer()
    for t in TOUT:
        while n < int(round(t / dt)):
            st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=nu, hyper_r=1, hyper_n=4, scheme=scheme, eta_z=0.0)
            n += 1
        psiN = np.asarray(st.z_plus - st.z_minus) / 2
        out[t] = (W_m(box, st.g), flux_full(box, st.g, psiN, S, A, P))
        assert abs(n * dt - t) < 1e-12
    print(f"    {scheme} dt={dt}: max|g| at t={TOUT[-1]}: {float(np.abs(np.asarray(st.g)).max()):.3e} "
          f"(initial {float(np.abs(g0).max()):.3e})", flush=True)
    log_run("d03_frozen_gandalf_step.py", f"frozen-line theorem through gandalf_step ({scheme}, nu={nu})", N, M, n,
            clock(), dict(dt=dt, T=TOUT[-1]))
    return out


chk = Checks()
print(f"N = {N}, M = {M}, Lambda = sqrt5, s = 1, amp = {AMP}; NL = gandalf_step (eta = eta_z = 0), "
      f"FR = frozen-line prediction (RK4, dt = 5e-4)", flush=True)
summary = {}
for nu in (0.0, 20.0):
    fr = frozen(nu)
    for scheme in ("lawson_rk4", "imex_rk222"):
        errs = {}
        for dt in (0.002, 0.001, 0.0005):
            nl = nonlinear(scheme, dt, nu)
            errs[dt] = {t: (relerr(nl[t][0], fr[t][0]), maxrel(nl[t][1], fr[t][1])) for t in TOUT}
        for t in TOUT:
            print(f"  nu={nu:4.1f} {scheme:10s} t={t:6.4f}: W(m) rel.diff "
                  + "  ".join(f"dt={dt:.4f}: {errs[dt][t][0]:.2e}" for dt in errs)
                  + "   | Pi_m: " + "  ".join(f"{errs[dt][t][1]:.2e}" for dt in errs), flush=True)
        summary[(nu, scheme)] = errs
        e_last = errs[0.0005][0.128][0]
        e_first = errs[0.002][0.128][0]
        chk(f"nu={nu} {scheme}: W(m) NL - FR at t=0.128 does not grow when dt is reduced",
            e_last <= e_first * 1.05, f"{e_first:.2e} -> {e_last:.2e}")
        chk(f"nu={nu} {scheme}: W(m) NL - FR at t=0.128 below 1e-3 at dt=5e-4 (N={N})", e_last < 1e-3, f"{e_last:.2e}")
# order of the stepper error at t = 0.064 (spatial floor at N = 32 is ~1e-5, d02)
for nu in (0.0, 20.0):
    eL = [summary[(nu, "lawson_rk4")][dt][0.064][0] for dt in (0.002, 0.001, 0.0005)]
    eI = [summary[(nu, "imex_rk222")][dt][0.064][0] for dt in (0.002, 0.001, 0.0005)]
    rI = (eI[0] / eI[1], eI[1] / eI[2])
    print(f"  nu={nu}: t=0.064 W(m) differences  Lawson-RK4 {', '.join(f'{x:.2e}' for x in eL)};  IMEX-RK222 "
          f"{', '.join(f'{x:.2e}' for x in eI)} (ratios {rI[0]:.2f}, {rI[1]:.2f})")
    if nu == 0.0:
        chk("Lawson-RK4 (nu = 0): stepper error below the spatial floor at every dt (difference flat in dt)",
            max(eL) / min(eL) < 1.2, ", ".join(f"{x:.2e}" for x in eL))
    chk(f"IMEX-RK222 (nu = {nu}): first-order convergence in dt (gandalf#158)", all(1.7 < r_ < 2.6 for r_ in rI),
        f"ratios {rI[0]:.2f}, {rI[1]:.2f}")
    eI2 = summary[(nu, "imex_rk222")][0.002][0.128][0]
    chk(f"IMEX-RK222 (nu = {nu}) at dt = 0.002 departs from the frozen-line prediction by t = 0.128 (instability)",
        eI2 > 0.1 and summary[(nu, "imex_rk222")][0.001][0.128][0] < 1e-3, f"{eI2:.2e}")
# explanation: the explicit part of ARS(2,2,2) has stability function R(z) = 1 + z + z^2/2 (|R(iy)|^2 = 1 + y^4/4 > 1)
# and the slice-advection term s u_j {Psi, a_j} has imaginary eigenvalues up to ~ s u_max max|grad Psi| k_perp,max
gx = np.fft.irfftn(1j * box.KX * psi, s=(N, N, N), axes=(0, 1, 2))
gy = np.fft.irfftn(1j * box.KY * psi, s=(N, N, N), axes=(0, 1, 2))
vmax = float(np.sqrt(gx ** 2 + gy ** 2).max())
umax = float(np.abs(np.linalg.eigvals(A)).max())
kmax = float(np.sqrt((box.KP2 * box.mask).max()))          # largest retained k_perp
for dt in (0.002, 0.001, 0.0005):
    y = S * umax * vmax * kmax * dt
    print(f"  slice-advection bound: s u_max max|grad Psi| k_perp,max dt = {S:.0f} x {umax:.2f} x {vmax:.2f} x {kmax:.1f} x {dt}"
          f" = {y:.2f};  explicit RK2 |R(iy)| = {abs(1 + 1j * y - y ** 2 / 2):.3f};  RK4 |R(iy)| = "
          f"{abs(1 + 1j * y - y ** 2 / 2 - 1j * y ** 3 / 6 + y ** 4 / 24):.3f}")
chk.finish()
