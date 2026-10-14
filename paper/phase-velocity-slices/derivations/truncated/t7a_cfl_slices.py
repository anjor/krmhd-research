"""t7a_cfl_slices: checks the explicit perpendicular advection of the slices in GANDALF's time steppers and the
time-step limit it implies: the explicit term advects slice j with Phi + s u_j Psi, krmhd's compute_cfl_timestep
includes the factor u_j (GANDALF >= 0.7.0, gandalf#157; <= 0.6.0 omitted it), the explicit IMEX-RK222 stage has no
stable interval on the imaginary axis while Lawson-RK4 is stable for y <= 2 sqrt2, and in a shear-flow test every slice
trajectory through gandalf_step follows the derived per-slice linear map, with the u_max slices unstable at the
GANDALF <= 0.6.0 dt_cfl. Supports Appendix D of the paper (summarised in Section 8).

A  GANDALF's explicit Hermite term (krmhd.timestepping._krmhd_rhs_jit with k_z set to zero, as both steppers call it)
   is N(g) = -{Phi, g} - s {Psi, A g}; projected on slice j it is -{Phi + s u_j Psi, a_j}: slice j is advected by
   z^ x grad(Phi + s u_j Psi) at speed |grad_perp(Phi + s u_j Psi)| <= |grad Phi| + s |u_j| |grad Psi|.
B  krmhd.timestepping.compute_cfl_timestep (GANDALF >= 0.7.0, gandalf#157) uses
   v_max = max(v_A, |grad z+|, |grad z-|, max_j |grad(Phi + s u_j Psi)|), so it includes the factor |u_j|
   (u_max = 10.6 at M = 64, 15.4 at M = 128 for Lam = +-sqrt5); GANDALF <= 0.6.0 used s |grad Psi| and omitted it.
C  Stage polynomials of the explicit parts: IMEX-RK222 (ARS 2,2,2 with GANDALF's gamma, delta) R2(z) = 1 + z + z^2/2,
   |R2(iy)|^2 = 1 + y^4/4 > 1 for every y != 0 (pure advection is weakly unstable at any dt and needs dissipation);
   Lawson-RK4 at k_z = 0 is classical RK4, |R4(iy)| <= 1 iff |y| <= 2 sqrt2. With GANDALF's post-step factor
   exp(-eta (k_perp^2/k_perp,max^2)^r dt), hyper_r = 2, the IMEX threshold is dt^3 <= 8 eta / (k_perp,max V)^4.
D  GANDALF test: a stationary shear flow (z+- functions of y only, k_z = 0, so the Elsasser step is exact and every
   slice sees a fixed linear operator). The per-step growth of every slice through gandalf_step equals the spectral
   radius of D_eta R(dt N_j) computed from the analytic operator N_j = i k_x f_j'(y) (f_j = Phi + s u_j Psi), for both
   schemes, with and without hyper-resistivity; the fastest slices are |u_j| = u_max; at the GANDALF <= 0.6.0 dt_cfl
   (no u_j) the fast slices are violently unstable while the u ~ 0 slices are not. At the GANDALF >= 0.7.0 dt_cfl every
   slice decays with eta = 2000, but without dissipation the u_max slices still grow slowly (IMEX-RK222 is weakly
   unstable for pure advection at any dt, gandalf#166).
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
differs = False
for M in (16, 64, 128):
    for Lam in (SQRT5, -SQRT5):
        rng = np.random.default_rng(100 + M)
        zp = random_field(grid, rng, amp=3.0)
        zm = random_field(grid, rng, amp=3.0)
        g = np.zeros((grid.Nz, grid.Ny, grid.Nx // 2 + 1, M + 1), complex)
        st = make_state(grid, M, Lam, zp, zm, g, beta_i=1.0)
        dt_g = compute_cfl_timestep(st, v_A=1.0, cfl_safety=0.3)
        Phi = (zp + zm) / 2
        Psi = (zp - zm) / 2
        u = np.sort(np.linalg.eigvals(ladder(M, Lam)).real)
        Vall = np.array([maxgrad(grid, Phi + uj * Psi) for uj in u])      # every slice speed (s = 1)
        vG = max(1.0, maxgrad(grid, zp), maxgrad(grid, zm), Vall.max())  # GANDALF >= 0.7.0 (gandalf#157)
        vG_old = max(1.0, maxgrad(grid, zp), maxgrad(grid, zm), maxgrad(grid, Psi))   # GANDALF <= 0.6.0 (no u_j)
        dt_mine = 0.3 * (1.0 / N) / vG
        v_gandalf = 0.3 * (1.0 / N) / dt_g
        Vj = np.array([maxgrad(grid, Phi + uj * Psi) for uj in (u[0], u[-1], u[np.argmin(np.abs(u))])])
        bound = maxgrad(grid, Phi) + u[-1] * maxgrad(grid, Psi)
        okB = okB and abs(dt_g / dt_mine - 1) < 1e-10 and Vj[:2].max() <= bound * (1 + 1e-12) and \
            Vall.max() <= v_gandalf * (1 + 1e-10)
        differs = differs or vG / vG_old > 1 + 1e-3
        print(f"   M={M:3d} Lam={Lam:+.3f}: u_max = {u[-1]:.4f}; GANDALF v_max = {v_gandalf:.3f} (dt_cfl reproduced: "
              f"{abs(dt_g / dt_mine - 1):.1e}; <= 0.6.0 formula v_max = {vG_old:.3f}); slice speeds max|grad(Phi + u Psi)| "
              f"at u = -u_max, +u_max, 0: {Vj[0]:.3f}, {Vj[1]:.3f}, {Vj[2]:.3f}; max over all j = {Vall.max():.3f}; ratio "
              f"fastest slice / v_max = {Vall.max() / v_gandalf:.2f}; bound |grad Phi| + u_max |grad Psi| = {bound:.3f}")
C.check("B1 compute_cfl_timestep (GANDALF >= 0.7.0, gandalf#157) = 0.3 dx / max(v_A, |grad z+-|, max_j |grad(Phi + s u_j "
        "Psi)|) (includes u_j; <= 0.6.0 used s|grad Psi| and gives a different dt_cfl for at least one state here); "
        "fastest slice speed obeys the bound and no longer exceeds v_max", okB and differs)

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
dt_cfl = compute_cfl_timestep(st0, v_A=1.0, cfl_safety=0.3)      # GANDALF >= 0.7.0 (includes u_j, gandalf#157)
# GANDALF <= 0.6.0 value: 0.3 min(dx, dy, dz) / max(v_A, |grad z+-|, s |grad Psi|), without the factor u_j
dt_cfl_old = 0.3 * min(g2.Lx / g2.Nx, g2.Ly / g2.Ny, g2.Lz / g2.Nz) / max(
    1.0, maxgrad(g2, zp), maxgrad(g2, zm), s * maxgrad(g2, (zp - zm) / 2))
print(f"   M = {M}, Lam = sqrt5, max|grad Psi| = {maxgrad(g2, (zp - zm) / 2):.3f}, max|grad Phi| = {maxgrad(g2, (zp + zm) / 2):.3f};"
      f" fastest slice speed max|f_j'| = {Vfast:.3f}; GANDALF dt_cfl(0.3) = {dt_cfl:.3e}, y_fast(dt_cfl) = "
      f"{dt_cfl * kxs[ix] * Vfast * lam_T:.2f}; GANDALF <= 0.6.0 dt_cfl(0.3) (no u_j) = {dt_cfl_old:.3e}, y_fast = "
      f"{dt_cfl_old * kxs[ix] * Vfast * lam_T:.2f}")


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
# for every k_x plane; numerical threshold = smallest dt at which some k_x plane is unstable.
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
          f"formula (8 eta)^(1/3)/(k_perp,max V_j)^(4/3) = {analytic:.4e}; GANDALF dt_cfl = {dt_cfl:.3e}, "
          f"<= 0.6.0 dt_cfl = {dt_cfl_old:.3e}")
    if j == M:
        okD3 = okD3 and thr[j] < dt_cfl_old and abs(thr[j] / analytic - 1) < 0.25 and dt_cfl < thr[j]
    else:
        okD3 = okD3 and thr[j] > dt_cfl_old
C.check("D3 IMEX, eta = 2000: GANDALF <= 0.6.0's dt_cfl (no u_j) exceeds the stability threshold of the fastest slice "
        "(formula within 25%) but not that of u ~ 0; GANDALF >= 0.7.0's dt_cfl is below both", okD3)
# direct demonstration at the GANDALF <= 0.6.0 dt_cfl (IMEX, eta = 2000)
traj, pred, err, wall = run(dt_cfl_old, eta, "imex_rk222", 15)
runs.append(("imex_rk222", dt_cfl_old, eta, 15, wall))
grow = np.linalg.norm(traj[-1], axis=0) / np.linalg.norm(traj[0], axis=0)
print(f"   gandalf_step at the <= 0.6.0 dt_cfl, eta = {eta:g}, 15 steps (trajectory deviation from the linear map {err:.1e}): "
      "amplitude ratio at u = " + ", ".join(f"{u[j]:+.2f}: {grow[j]:.2e}" for j in sel_j))
C.check("D4 at GANDALF <= 0.6.0's dt_cfl (no u_j) the u_max slices blow up while the u ~ 0 slice decays",
        grow[M] > 1e2 and grow[np.argmin(np.abs(u))] < 1.0 and err < 1e-10)
# the same 15 steps at GANDALF's (>= 0.7.0) dt_cfl, with eta = 2000 and with eta = 0
okD5 = True
for eta5 in (eta, 0.0):
    traj, pred, err, wall = run(dt_cfl, eta5, "imex_rk222", 15)
    runs.append(("imex_rk222", dt_cfl, eta5, 15, wall))
    grow5 = np.linalg.norm(traj[-1], axis=0) / np.linalg.norm(traj[0], axis=0)
    r5 = np.array([rho(j, dt_cfl, eta5, "imex_rk222") for j in range(M + 1)])
    print(f"   gandalf_step at GANDALF dt_cfl, eta = {eta5:g}, 15 steps (trajectory deviation from the linear map "
          f"{err:.1e}): amplitude ratio at u = " + ", ".join(f"{u[j]:+.2f}: {grow5[j]:.2e}" for j in sel_j) +
          f"; spectral radius per step at u_max {r5[M]:.5f}, u ~ 0 {r5[np.argmin(np.abs(u))]:.5f}, largest at u = "
          f"{u[np.argmax(r5)]:+.2f}; |ratio - 1| at u ~ 0 = {abs(grow5[np.argmin(np.abs(u))] - 1):.1e}")
    okD5 = okD5 and err < 1e-10
    if eta5 > 0:
        okD5 = okD5 and grow5.max() < 1.0 and r5.max() < 1.0
    else:
        okD5 = okD5 and grow5[M] > 1.0 and r5[M] > 1.0 and abs(u[np.argmax(r5)]) > 0.95 * u[-1] and \
            abs(grow5[np.argmin(np.abs(u))] - 1) < 1e-6
C.check("D5 at GANDALF's (>= 0.7.0) dt_cfl, IMEX: with eta = 2000 every slice decays; with eta = 0 the u_max slices "
        "still grow, weakly (largest spectral radius at |u_j| ~ u_max, gandalf#166), and the u ~ 0 slice is neutral", okD5)
print("GANDALF integrations (scheme, dt, eta, steps, wall s), grid 32x32x2, M = 64, Lam = sqrt5, shear flow:")
for r in runs:
    print(f"   {r[0]} dt={r[1]:.3e} eta={r[2]:g} steps={r[3]} wall={r[4]:.1f}s")
C.finish()
