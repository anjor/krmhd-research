"""t7a_cfl_slices: checks the explicit perpendicular advection of the slices in GANDALF's time steppers and the
time-step limit it implies: the explicit term advects slice j with Phi + s u_j Psi, krmhd's compute_cfl_timestep
includes the factor u_j (krmhd 0.7.0, gandalf#157, PR #161), the explicit IMEX-RK222 stage has no stable interval on
the imaginary axis while Lawson-RK4 is stable for y <= 2 sqrt2, and in a shear-flow test every slice trajectory
through gandalf_step follows the derived per-slice linear map; with hyper-resistivity GANDALF's dt_cfl keeps every
slice stable, while a limit without the factor u_j leaves the u_max slices unstable. Supports Sections 7 and 8 (the
time-step limit of the fastest slices); the IMEX-RK222 parts concern the time-stepping appendix of an earlier draft.

A  GANDALF's explicit Hermite term (krmhd.timestepping._krmhd_rhs_jit with k_z set to zero, as both steppers call it)
   is N(g) = -{Phi, g} - s {Psi, A g}; projected on slice j it is -{Phi + s u_j Psi, a_j}: slice j is advected by
   z^ x grad(Phi + s u_j Psi) at speed |grad_perp(Phi + s u_j Psi)| <= |grad Phi| + s |u_j| |grad Psi|.
B  krmhd.timestepping.compute_cfl_timestep (krmhd 0.7.0) uses v_max = max(v_A, |grad z+|, |grad z-|, V_g) with the
   fastest slice speed V_g = max_j max_x |grad_perp(Phi + s u_j Psi)|, evaluated at the extreme u_j (|grad(Phi + c Psi)|
   is convex in c), for either closure. Here V_g is taken over every slice j, independently of GANDALF, and the states
   are strong enough that V_g sets v_max, so the factor u_j is tested (u_max = 4.9, 10.6, 15.4 at M = 16, 64, 128 for
   Lam = +-sqrt5). krmhd 0.6.0 used s |grad Psi| in place of V_g, which omits the factor u_j; its v_max is printed
   for comparison.
C  Stage polynomials of the explicit parts: IMEX-RK222 (ARS 2,2,2 with GANDALF's gamma, delta) R2(z) = 1 + z + z^2/2,
   |R2(iy)|^2 = 1 + y^4/4 > 1 for every y != 0 (pure advection is weakly unstable at any dt and needs dissipation);
   Lawson-RK4 at k_z = 0 is classical RK4, |R4(iy)| <= 1 iff |y| <= 2 sqrt2. With GANDALF's post-step factor
   exp(-eta (k_perp^2/k_perp,max^2)^r dt), hyper_r = 2, the IMEX threshold is dt^3 <= 8 eta / (k_perp,max V)^4.
D  GANDALF test: a stationary shear flow (z+- functions of y only, k_z = 0, so the Elsasser step is exact and every
   slice sees a fixed linear operator). The per-step growth of every slice through gandalf_step equals the spectral
   radius of D_eta R(dt N_j) computed from the analytic operator N_j = i k_x f_j'(y) (f_j = Phi + s u_j Psi), for both
   schemes, with and without hyper-resistivity; the fastest slices are |u_j| = u_max. With eta = 2000, GANDALF's
   dt_cfl lies below the IMEX stability threshold of every slice and every slice decays at dt_cfl; the limit without
   the factor u_j (krmhd 0.6.0's v_max, 8.4 times GANDALF's dt_cfl here) exceeds the threshold of the u_max slices but
   not that of the u ~ 0 slices, and at that dt the u_max slices blow up while the u ~ 0 slice decays.
"""
from __future__ import annotations

import time

import numpy as np
import sympy as sp

from tcore import (Checker, section, ladder, weight, slices, make_grid, random_field, make_state, bracket, maxgrad,
                   ksum_weights, SQRT5, jnp)
from krmhd.timestepping import (_krmhd_rhs_jit, _fields_from_state, compute_cfl_timestep, gandalf_step,
                                _IMEX_GAMMA, _IMEX_DELTA)
from krmhd.spectral import rfftn_forward

C = Checker("t7a_cfl_slices")

section("A. The explicit term in slice variables")
N = 16
grid = make_grid(N)
mask = np.asarray(grid.dealias_mask)
okA = True
for M in (10, 16):
    for Lam in (SQRT5, -SQRT5):
        for clo in ("zero", "symmetric"):
            rng = np.random.default_rng(3 + M)
            zp = random_field(grid, rng, amp=50.0)
            zm = random_field(grid, rng, amp=50.0)
            g = random_field(grid, rng, slope=-1.0, extra_shape=(M + 1,))
            beta = 1.3
            s = np.sqrt(beta)
            st = make_state(grid, M, Lam, zp, zm, g, beta_i=beta)
            kz0 = jnp.zeros_like(grid.kz)
            Ng = np.asarray(_krmhd_rhs_jit(_fields_from_state(st), grid.kx, grid.ky, kz0, grid.dealias_mask, 0.0, 1.0,
                                           beta, 0.0, Lam, M, grid.Nz, grid.Ny, grid.Nx, closure=clo).g)
            u, R, W, n = slices(M, Lam, clo)
            a = np.einsum("zyxm,mj->zyxj", g, W)
            aN = np.einsum("zyxm,mj->zyxj", Ng, W)
            Phi = (zp + zm) / 2
            Psi = (zp - zm) / 2
            worst = 0.0
            for j in range(M + 1):
                pred = -bracket(grid, Phi + s * u[j] * Psi, a[..., j]) * mask
                pred[0, 0, 0] = 0
                worst = max(worst, np.abs(pred - aN[..., j]).max() / np.abs(pred).max())
            okA = okA and worst < 1e-12
            print(f"   M={M} Lam={Lam:+.3f} {clo:9s}: max_j |w_j.N(g) + {{Phi + s u_j Psi, a_j}}| / max = {worst:.1e}")
C.check("A1 explicit term (GANDALF's _krmhd_rhs_jit at k_z = 0) = -{Phi + s u_j Psi, a_j} slice by slice", okA)

section("B. GANDALF's CFL speed versus the slice speeds")
okB = True
beta = 1.4                                   # s = sqrt(beta) != 1, so the factor s u_j is tested
s = np.sqrt(beta)
for M in (16, 64, 128):
    for Lam in (SQRT5, -SQRT5):
        for clo in ("zero", "symmetric"):
            rng = np.random.default_rng(100 + M)
            zp = random_field(grid, rng, amp=10.0)
            zm = random_field(grid, rng, amp=10.0)
            g = np.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), complex)
            st = make_state(grid, M, Lam, zp, zm, g, beta_i=beta)
            dt_g = compute_cfl_timestep(st, v_A=1.0, cfl_safety=0.3, closure=clo)
            Phi = (zp + zm) / 2
            Psi = (zp - zm) / 2
            u = np.sort(np.linalg.eigvals(ladder(M, Lam, clo)).real)
            Vj = np.array([maxgrad(grid, Phi + s * uj * Psi) for uj in u])       # every slice, not only the extremes
            Vg = Vj.max()
            vz = max(1.0, maxgrad(grid, zp), maxgrad(grid, zm))
            vG = max(vz, Vg)
            v_old = max(vz, s * maxgrad(grid, Psi))          # krmhd 0.6.0: s |grad Psi|, no factor u_j
            dt_mine = 0.3 * (1.0 / N) / vG
            bound = maxgrad(grid, Phi) + s * np.abs(u).max() * maxgrad(grid, Psi)
            jmax = int(np.argmax(Vj))
            okB = (okB and abs(dt_g / dt_mine - 1) < 1e-10 and dt_g * Vg <= 0.3 / N * (1 + 1e-12) and Vg > vz
                   and Vg <= bound * (1 + 1e-12) and jmax in (0, len(u) - 1))       # an extreme u_j
            print(f"   M={M:3d} Lam={Lam:+.3f} {clo:9s}: u_max = {np.abs(u).max():.4f}; max(v_A, |grad z+-|) = {vz:.3f}; "
                  f"fastest slice speed V_g = {Vg:.3f} at u = {u[jmax]:+.4f} (bound |grad Phi| + s u_max |grad Psi| = "
                  f"{bound:.3f}); GANDALF dt_cfl reproduced with v_max = V_g: {abs(dt_g / dt_mine - 1):.1e}; "
                  f"dt_cfl V_g / dx = {dt_g * Vg * N:.3f}; krmhd 0.6.0 v_max (no u_j) = {v_old:.3f}, V_g / that = "
                  f"{Vg / v_old:.2f}")
C.check("B1 compute_cfl_timestep = 0.3 dx / max(v_A, |grad z+-|, max_j |grad(Phi + s u_j Psi)|) for both closures "
        "(krmhd 0.7.0, gandalf#157, PR #161): the fastest slice, at an extreme u_j, sets dt_cfl and obeys dt_cfl V_g = 0.3 dx",
        okB)

section("C. Stage polynomials")
z, y = sp.symbols("z y")
gam = (2 - sp.sqrt(2)) / 2
dlt = 1 - 1 / (2 * gam)
R2 = sp.expand(1 + z + gam * (1 - dlt) * z ** 2)
okC = sp.simplify(R2 - (1 + z + z ** 2 / 2)) == 0 and abs(_IMEX_GAMMA - float(gam)) < 1e-16 and \
    abs(_IMEX_DELTA - float(dlt)) < 1e-15
mod2 = sp.simplify(sp.expand((R2.subs(z, sp.I * y)) * sp.conjugate(R2.subs(z, sp.I * y)))).subs(sp.conjugate(y), y)
R4 = 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24
mod4 = sp.expand(sp.simplify((R4.subs(z, sp.I * y)) * sp.conjugate(R4.subs(z, sp.I * y))).subs(sp.conjugate(y), y))
print(f"   GANDALF constants gamma = {_IMEX_GAMMA!r}, delta = {_IMEX_DELTA!r}; R2(z) = {sp.nsimplify(R2)}; "
      f"|R2(iy)|^2 = {sp.expand(mod2)}; |R4(iy)|^2 = {mod4}")
okC = okC and sp.simplify(mod2 - (1 + y ** 4 / 4)) == 0 and sp.simplify(mod4 - (1 - y ** 6 / 72 + y ** 8 / 576)) == 0
C.check("C1 explicit IMEX part R2 = 1 + z + z^2/2, |R2(iy)|^2 = 1 + y^4/4; RK4 |R4(iy)|^2 = 1 - y^6/72 + y^8/576", okC)

section("D. GANDALF shear-flow test: per-step growth of every slice")
Ng_ = 32
g2 = make_grid(Ng_, Nz=2)          # Nz = 2: only k_z = 0 survives the 2/3 mask
M = 64
Lam = SQRT5
s = 1.0
u, R, W, n = slices(M, Lam)
aPsi = 1.0 / np.pi                  # max |grad Psi| = 2 pi aPsi = 2
aPhi = 0.3 * aPsi
yv = np.arange(Ng_) / Ng_
Phi_r = aPhi * np.cos(2 * np.pi * yv)[None, :, None] * np.ones((2, Ng_, Ng_))
Psi_r = aPsi * np.cos(2 * np.pi * yv)[None, :, None] * np.ones((2, Ng_, Ng_))
zp = np.asarray(rfftn_forward(jnp.asarray(Phi_r + Psi_r)))
zm = np.asarray(rfftn_forward(jnp.asarray(Phi_r - Psi_r)))
kxs = np.asarray(g2.kx)
kys = np.asarray(g2.ky)
ix = (Ng_ - 1) // 3                 # largest retained k_x index (10)
iy_keep = np.where(np.abs(kys) <= (2.0 / 3.0) * np.abs(kys).max() + 1e-12)[0]
nys = np.round(kys[iy_keep] / (2 * np.pi)).astype(int)
order = np.argsort(nys)
iy_keep = iy_keep[order]
nys = nys[order]
K = len(nys)
# projected multiplication by sin(2 pi y) on the retained k_y set (Hermitian tridiagonal)
Tsin = np.zeros((K, K), complex)
for a_ in range(K):
    for b_ in range(K):
        if nys[a_] == nys[b_] + 1:
            Tsin[a_, b_] = 1 / (2j)
        if nys[a_] == nys[b_] - 1:
            Tsin[a_, b_] = -1 / (2j)
kperp_max2 = kxs[(Ng_ - 1) // 3] ** 2 + kys[(Ng_ - 1) // 3] ** 2


def step_matrix(j: int, dt: float, eta: float, scheme: str, ixx: int = ix, hyper_r: int = 2,
                vel: float = 1.0) -> np.ndarray:
    """One GANDALF step for slice j restricted to the k_x-plane ixx (retained k_y modes, sorted by n_y), as derived:
    D_eta R(dt N_j), N_j = i k_x f_j'(y) projected, f_j = Phi + s u_j Psi, R = 1 + z + gamma(1-delta) z^2 (IMEX) or the
    RK4 polynomial (Lawson at k_z = 0), D_eta = exp(-eta ((k_x^2 + k_y^2)/k_perp,max^2)^r dt)."""
    cj = -2 * np.pi * (aPhi + s * u[j] * aPsi) * vel    # f_j'(y) = cj sin(2 pi y); vel = decay of z+- under eta
    Z = dt * 1j * kxs[ixx] * cj * Tsin
    I = np.eye(K)
    if scheme == "imex_rk222":
        Rm = I + Z + _IMEX_GAMMA * (1 - _IMEX_DELTA) * Z @ Z
    else:
        Z2 = Z @ Z
        Rm = I + Z + Z2 / 2 + Z2 @ Z / 6 + Z2 @ Z2 / 24
    kk2 = kxs[ixx] ** 2 + kys[iy_keep] ** 2
    return np.diag(np.exp(-eta * (kk2 / kperp_max2) ** hyper_r * dt)) @ Rm


def rho(j: int, dt: float, eta: float, scheme: str, all_kx: bool = False) -> float:
    planes = range(1, ix + 1) if all_kx else [ix]
    return max(float(np.abs(np.linalg.eigvals(step_matrix(j, dt, eta, scheme, p))).max()) for p in planes)


Vfast = 2 * np.pi * (aPhi + s * u[-1] * aPsi)
lam_T = np.abs(np.linalg.eigvalsh(Tsin)).max()            # largest |eigenvalue| of the Hermitian projected sin(2 pi y)
rng = np.random.default_rng(5)
g0 = np.zeros((2, Ng_, Ng_ // 2 + 1, M + 1), complex)
g0[0, :, ix, :] = (rng.standard_normal((Ng_, M + 1)) + 1j * rng.standard_normal((Ng_, M + 1)))
g0 *= np.asarray(g2.dealias_mask)[..., None]
st0 = make_state(g2, M, Lam, zp, zm, g0, beta_i=1.0)
dt_cfl = compute_cfl_timestep(st0, v_A=1.0, cfl_safety=0.3)
print(f"   M = {M}, Lam = sqrt5, max|grad Psi| = {maxgrad(g2, (zp - zm) / 2):.3f}, max|grad Phi| = {maxgrad(g2, (zp + zm) / 2):.3f};"
      f" fastest slice speed max|f_j'| = {Vfast:.3f}; GANDALF dt_cfl(0.3) = {dt_cfl:.3e}; y_fast(dt_cfl) = "
      f"{dt_cfl * kxs[ix] * Vfast * lam_T:.2f}")


def run(dt: float, eta: float, scheme: str, nsteps: int):
    """GANDALF trajectory of the slice amplitudes on the k_x = k_x,max plane, and the predicted one."""
    st = st0
    traj = []
    t0 = time.time()
    for k in range(nsteps + 1):
        a = np.einsum("yxm,mj->yxj", np.asarray(st.g)[0], W)
        traj.append(a[iy_keep, ix, :])
        if k < nsteps:
            st = gandalf_step(st, dt, eta=eta, v_A=1.0, nu=0.0, hyper_r=2, hyper_n=6, scheme=scheme)
    wall = time.time() - t0
    traj = np.array(traj)                                     # [step, k_y, j]
    pred = np.zeros_like(traj)
    pred[0] = traj[0]
    # GANDALF damps z+- too: the shear flow (n_y = +-1, n_x = 0) decays by d per step
    d = np.exp(-eta * ((2 * np.pi) ** 2 / kperp_max2) ** 2 * dt)
    for j in range(M + 1):
        for k in range(nsteps):
            pred[k + 1, :, j] = step_matrix(j, dt, eta, scheme, vel=d ** k) @ pred[k, :, j]
    err = max(np.linalg.norm(traj[k] - pred[k], axis=0).max() / np.linalg.norm(pred[k], axis=0).max()
              for k in range(nsteps + 1))
    return traj, pred, err, wall


runs = []
okD = True
okD2 = True
sel_j = [np.argmin(np.abs(u)), np.argmin(np.abs(u - 3.0)), np.argmin(np.abs(u - 6.0)), M - 1, M]
for scheme, ytarget, eta, nsteps in (("imex_rk222", 0.8, 0.0, 60), ("imex_rk222", 0.8, 2000.0, 60),
                                     ("lawson_rk4", 2.6, 0.0, 60), ("lawson_rk4", 3.1, 0.0, 20)):
    dt = ytarget / (kxs[ix] * Vfast * lam_T)
    traj, pred, err, wall = run(dt, eta, scheme, nsteps)
    runs.append((scheme, dt, eta, nsteps, wall))
    amp = np.linalg.norm(traj[-1], axis=0) / np.linalg.norm(traj[0], axis=0)
    rj = np.array([rho(j, dt, eta, scheme) for j in range(M + 1)])
    okD = okD and err < 1e-11
    if rj.max() > 1.0 + 1e-9:
        okD2 = okD2 and abs(u[np.argmax(amp)]) > 0.95 * u[-1] and abs(u[np.argmax(rj)]) > 0.95 * u[-1]
    print(f"   {scheme}, eta={eta:g}, dt={dt:.3e} (y_fast={ytarget}, {dt / dt_cfl:.3f} dt_cfl), {nsteps} steps ({wall:.1f}s): "
          f"max relative deviation of every slice trajectory from the predicted linear map = {err:.1e}")
    print("     u_j                         : " + " ".join(f"{u[j]:9.3f}" for j in sel_j))
    print("     |a_j(n)|/|a_j(0)| (GANDALF) : " + " ".join(f"{amp[j]:9.3e}" for j in sel_j))
    print("     spectral radius per step    : " + " ".join(f"{rj[j]:9.5f}" for j in sel_j))
C.check("D1 every slice trajectory through gandalf_step equals the derived per-slice linear map D_eta R(dt N_j) "
        "(IMEX and Lawson, eta on/off) to round-off", okD)
C.check("D2 when unstable, the fastest-growing slice is at |u_j| ~ u_max", okD2)
# threshold for IMEX with hyper-resistivity (hyper_r = 2): small-y estimate dt^3 <= 8 eta / (k_perp,max V_j)^4, the same
# for every k_x plane; numerical threshold = smallest dt at which some k_x plane is unstable. It is compared with
# GANDALF's dt_cfl and with the limit without the factor u_j, 0.3 dx / max(v_A, |grad z+-|, s |grad Psi|) (the v_max of
# krmhd 0.6.0, whose dt_cfl was this value, 3.606e-3, and whose u_max slices blew up at it).
dt_nou = 0.3 * (1.0 / Ng_) / max(1.0, maxgrad(g2, zp), maxgrad(g2, zm), s * maxgrad(g2, (zp - zm) / 2))
print(f"   limit without the factor u_j: 0.3 dx / max(v_A, |grad z+-|, s |grad Psi|) = {dt_nou:.4e} "
      f"({dt_nou / dt_cfl:.2f} GANDALF dt_cfl)")
eta = 2000.0
okD3 = True
thr = {}
for j in (M, np.argmin(np.abs(u))):
    grid_dt = np.logspace(-5, -0.5, 400)
    unstable = [rho(j, d, eta, "imex_rk222", all_kx=True) > 1.0 + 1e-12 for d in grid_dt]
    if any(unstable):
        i1 = int(np.argmax(unstable))
        lo, hi = grid_dt[i1 - 1], grid_dt[i1]
        for _ in range(50):
            mid = np.sqrt(lo * hi)
            if rho(j, mid, eta, "imex_rk222", all_kx=True) > 1.0 + 1e-12:
                hi = mid
            else:
                lo = mid
        thr[j] = hi
    else:
        thr[j] = np.inf
    Vj = abs(2 * np.pi * (aPhi + s * u[j] * aPsi)) * lam_T
    analytic = (8 * eta / (np.sqrt(kperp_max2) * Vj) ** 4) ** (1 / 3)
    print(f"   IMEX, eta = {eta:g}: slice u = {u[j]:+.3f}: first unstable dt (any k_x plane) = {thr[j]:.4e}; small-y "
          f"formula (8 eta)^(1/3)/(k_perp,max V_j)^(4/3) = {analytic:.4e}; GANDALF dt_cfl = {dt_cfl:.3e}; limit "
          f"without u_j = {dt_nou:.3e}")
    if j == M:
        okD3 = okD3 and dt_cfl < thr[j] < dt_nou and abs(thr[j] / analytic - 1) < 0.25
    else:
        okD3 = okD3 and thr[j] > dt_nou > dt_cfl
C.check("D3 IMEX, eta = 2000: the stability threshold of the fastest slice (formula within 25%) lies above GANDALF's "
        "dt_cfl (krmhd 0.7.0, gandalf#157, PR #161) and below the limit without u_j; that of u ~ 0 lies above both", okD3)
# direct demonstration (IMEX, eta = 2000): 15 steps at GANDALF's dt_cfl and at the limit without u_j
grows = {}
okD4 = True
for lab, dtr in (("GANDALF's dt_cfl", dt_cfl), ("the limit without u_j", dt_nou)):
    traj, pred, err, wall = run(dtr, eta, "imex_rk222", 15)
    runs.append(("imex_rk222", dtr, eta, 15, wall))
    grow = np.linalg.norm(traj[-1], axis=0) / np.linalg.norm(traj[0], axis=0)
    grows[lab] = grow
    okD4 = okD4 and err < 1e-10
    print(f"   gandalf_step at {lab} (dt = {dtr:.3e}), eta = {eta:g}, 15 steps (trajectory deviation from the linear map "
          f"{err:.1e}): amplitude ratio at u = " + ", ".join(f"{u[j]:+.2f}: {grow[j]:.2e}" for j in sel_j)
          + f"; largest over all slices {grow.max():.2e} at u = {u[np.argmax(grow)]:+.2f}")
g_cfl, g_nou = grows["GANDALF's dt_cfl"], grows["the limit without u_j"]
okD4 = okD4 and g_cfl.max() < 1.0 and g_nou[M] > 1e2 and g_nou[np.argmin(np.abs(u))] < 1.0
C.check("D4 eta = 2000: at GANDALF's dt_cfl every slice decays; at the limit without u_j the u_max slices blow up while "
        "the u ~ 0 slice decays", okD4)
print("GANDALF integrations (scheme, dt, eta, steps, wall s), grid 32x32x2, M = 64, Lam = sqrt5, shear flow:")
for r in runs:
    print(f"   {r[0]} dt={r[1]:.3e} eta={r[2]:g} steps={r[3]} wall={r[4]:.1f}s")
C.finish()
