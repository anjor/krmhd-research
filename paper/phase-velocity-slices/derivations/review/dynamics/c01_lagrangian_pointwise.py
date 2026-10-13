"""c01: pointwise Lagrangian test of the frozen-line theorem and of Cauchy's solution, by a route independent of the
scripts in derivations/dynamics.  Lagrangian markers and their deformation gradient are integrated together with
ideal KRMHD (RK4 on GANDALF's right-hand side), and the script checks that g at the markers equals the frozen-line
solution at their labels, that grad_par(t) is carried by the flow, that Cauchy's formula holds and that volume is
preserved, with errors that fall with N far below O(1) controls.  It supports Section 6 (frozen-line theorem).

d02-d05 in derivations/dynamics test the theorem through volume-integrated single-point statistics (W(m), C(j,j'),
moments, quantiles), i.e. through its corollary.  Here the statement itself is tested point by point:

  g(X(xi, t), t) = [exp(-t (s A grad_par(0))) g(0)](xi)                      (frozen-line theorem)
  B(X(xi, t), t) = J(xi, t) B(xi, 0),  J = dX/dxi                           (Cauchy's solution)
  (grad_par(t) g)(X(xi, t), t) = (grad_par(0) h)(xi)                        (grad_par carried by the flow)

X(xi, t) is integrated for 40 Lagrangian markers, dX/dt = u_perp(X, t) = (-Phi_y, Phi_x, 0)(X), together with the
deformation gradient dJ/dt = (grad u)(X) J, in the same RK4 stages as the fields; the fields come from GANDALF's
right-hand side (_krmhd_rhs_jit, eta = 0); u and grad u at the markers are exact Fourier-series sums.
The frozen problem h is GANDALF's right-hand side with (z+, z-) = (Psi0, -Psi0) held fixed.

Box L = 2 pi, M = 8, Lambda = sqrt5, s = 1, rms|grad Phi| = rms|grad Psi| = AMP (argv[2], default 1), fields from an explicit list of
low-order Fourier modes (same physical fields at every N), dt = 0.0025, outputs at TOUT (argv[3], default t = 0.1 and 0.2).
Controls (must be O(1)): Eulerian comparison g(xi, t) vs h(xi, t) (no relabelling); frozen problem along the wrong
field (Psi0 -> -Psi0) and along the unperturbed field (Psi = 0, the k_z-only linear model).
Usage: python c01_lagrangian_pointwise.py [N list, default 16,24,32,48] [AMP] [TOUT] [mode].  With mode = strong the
convergence numbers are printed but not asserted; run_all.sh uses it for a strong-flow run (N = 32, 48, 64,
AMP = 1.5, t = 0.25, 0.5) in which the fields reach the grid scale, which shows the resolution limit of the test.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from c00_core import (Checks, Grid, SQRT5, TWOPI, clock, ladder, log_run, make_rhs, mode_field,  # noqa: E402
                      random_modes)

NS = [int(v) for v in sys.argv[1].split(",")] if len(sys.argv) > 1 else [16, 24, 32, 48]
M, LAM, BETA = 8, SQRT5, 1.0
S = np.sqrt(BETA)
AMP = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
DT = 0.0025
TOUT = tuple(float(v) for v in sys.argv[3].split(",")) if len(sys.argv) > 3 else (0.1, 0.2)
NP = 40
CTRL_N = 32

rng = np.random.default_rng(20261013)
md_phi = random_modes(rng)
md_psi = random_modes(rng)
md_g0 = random_modes(rng)
md_g1 = random_modes(rng)
xi = rng.uniform(0, TWOPI, (NP, 3))          # marker labels (x, y, z)


def derivs_u(grid, E, phi):
    """u = (-Phi_y, Phi_x, 0) and grad u (n, 3, 3) at the points behind E."""
    ev = lambda d: grid.eval(E, phi, d)  # noqa: E731
    px, py = ev((1, 0, 0)), ev((0, 1, 0))
    pxx, pxy, pyy, pxz, pyz = ev((2, 0, 0)), ev((1, 1, 0)), ev((0, 2, 0)), ev((1, 0, 1)), ev((0, 1, 1))
    u = np.stack([-py, px, 0 * px], axis=1)
    G = np.zeros((len(px), 3, 3))
    G[:, 0, 0], G[:, 0, 1], G[:, 0, 2] = -pxy, -pyy, -pyz
    G[:, 1, 0], G[:, 1, 1], G[:, 1, 2] = pxx, pxy, pxz
    return u, G


def gradpar_at(grid, E, psi, f):
    """(d_z f + {Psi, f}) at the points behind E; f may carry a trailing moment axis."""
    fz = grid.eval(E, f, (0, 0, 1))
    fx = grid.eval(E, f, (1, 0, 0))
    fy = grid.eval(E, f, (0, 1, 0))
    psx = grid.eval(E, psi, (1, 0, 0))
    psy = grid.eval(E, psi, (0, 1, 0))
    if fz.ndim == 2:
        psx, psy = psx[:, None], psy[:, None]
    return fz + psx * fy - psy * fx


chk = Checks()
res = {}
for N in NS:
    grid = Grid(N)
    phi0 = mode_field(grid, *md_phi)
    psi0 = mode_field(grid, *md_psi)
    phi0 *= AMP / grid.rms_grad(phi0)
    psi0 *= AMP / grid.rms_grad(psi0)
    g_init = np.zeros((N, N, N // 2 + 1, M + 1), complex)
    g_init[..., 0] = mode_field(grid, *md_g0)
    g_init[..., 1] = 0.5 * mode_field(grid, *md_g1)
    g_init /= np.sqrt(grid.avg(g_init[..., 0], g_init[..., 0]))
    rhs = make_rhs(grid, M, LAM, BETA)

    def full_stage(zp, zm, g, X, J):
        dzp, dzm, dg = rhs(zp, zm, g)
        phi = np.asarray(zp + zm) / 2
        u, G = derivs_u(grid, grid.phase(X), phi)
        return dzp, dzm, dg, u, G @ J

    def rk4_full(zp, zm, g, X, J):
        k1 = full_stage(zp, zm, g, X, J)
        a = [zp + 0.5 * DT * k1[0], zm + 0.5 * DT * k1[1], g + 0.5 * DT * k1[2], X + 0.5 * DT * k1[3], J + 0.5 * DT * k1[4]]
        k2 = full_stage(*a)
        a = [zp + 0.5 * DT * k2[0], zm + 0.5 * DT * k2[1], g + 0.5 * DT * k2[2], X + 0.5 * DT * k2[3], J + 0.5 * DT * k2[4]]
        k3 = full_stage(*a)
        a = [zp + DT * k3[0], zm + DT * k3[1], g + DT * k3[2], X + DT * k3[3], J + DT * k3[4]]
        k4 = full_stage(*a)
        return tuple(y + DT / 6 * (q1 + 2 * q2 + 2 * q3 + q4) for y, q1, q2, q3, q4 in
                     zip((zp, zm, g, X, J), k1, k2, k3, k4))

    @jax.jit
    def rk4_frozen(zpF, zmF, h):
        f = lambda hh: rhs(zpF, zmF, hh)[2]  # noqa: E731
        k1 = f(h)
        k2 = f(h + 0.5 * DT * k1)
        k3 = f(h + 0.5 * DT * k2)
        k4 = f(h + DT * k3)
        return h + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    zp, zm = jnp.asarray(phi0 + psi0), jnp.asarray(phi0 - psi0)
    g = jnp.asarray(g_init)
    X = xi.copy()
    J = np.tile(np.eye(3), (NP, 1, 1))
    frozen = {"psi0": (jnp.asarray(psi0), jnp.asarray(-psi0))}
    if N == CTRL_N:
        frozen["minus_psi0"] = (jnp.asarray(-psi0), jnp.asarray(psi0))
        frozen["kz_only"] = (jnp.zeros_like(zp), jnp.zeros_like(zp))
    hs = {k: jnp.asarray(g_init) for k in frozen}
    nsteps = int(round(TOUT[-1] / DT))
    outs = {int(round(t / DT)): t for t in TOUT}
    tick = clock()
    for n in range(1, nsteps + 1):
        zp, zm, g, X, J = rk4_full(zp, zm, g, X, J)
        for k in hs:
            hs[k] = rk4_frozen(*frozen[k], hs[k])
        if n in outs:
            t = outs[n]
            gn, psit = np.asarray(g), np.asarray(zp - zm) / 2
            EX, Exi = grid.phase(X), grid.phase(xi)
            gX = grid.eval(EX, gn)                                   # (NP, M+1)
            h0 = np.asarray(hs["psi0"])
            hxi = grid.eval(Exi, h0)
            sc = np.abs(hxi).max()
            r = dict(theorem=np.abs(gX - hxi).max() / sc)
            # grad_par(t) U(t) = U(t) grad_par(0): field-aligned derivative carried by the flow
            gpX = gradpar_at(grid, EX, psit, gn)
            gpxi = gradpar_at(grid, Exi, psi0, h0)
            r["gradpar"] = np.abs(gpX - gpxi).max() / np.abs(gpxi).max()
            # Cauchy: B(X, t) = J B(xi, 0)
            BX = np.stack([-grid.eval(EX, psit, (0, 1, 0)), grid.eval(EX, psit, (1, 0, 0)), np.ones(NP)], axis=1)
            B0 = np.stack([-grid.eval(Exi, psi0, (0, 1, 0)), grid.eval(Exi, psi0, (1, 0, 0)), np.ones(NP)], axis=1)
            JB0 = np.einsum("nij,nj->ni", J, B0)
            r["cauchy"] = np.abs(BX - JB0).max() / np.abs(BX[:, :2]).max()
            r["detJ"] = np.abs(np.linalg.det(J) - 1).max()
            r["disp"] = float(np.sqrt(np.mean(np.sum((X - xi) ** 2, axis=1))))
            r["Jdev"] = float(np.abs(J - np.eye(3)).max())
            r["euler"] = np.abs(grid.eval(Exi, gn) - hxi).max() / sc
            kk = np.sqrt(grid.KP2 + grid.KZ ** 2)
            outer = (kk > 0.75 * (N / 3)) & grid.mask
            en = lambda f: np.sum(grid.W[..., None] * np.abs(f) ** 2, axis=(0, 1, 2))  # noqa: E731
            r["outer_NL"] = float(np.sum(grid.W[..., None] * outer[..., None] * np.abs(gn) ** 2) / np.sum(en(gn)))
            r["outer_FR"] = float(np.sum(grid.W[..., None] * outer[..., None] * np.abs(h0) ** 2) / np.sum(en(h0)))
            if N == CTRL_N:
                r["minus_psi0"] = np.abs(gX - grid.eval(Exi, np.asarray(hs["minus_psi0"]))).max() / sc
                r["kz_only"] = np.abs(gX - grid.eval(Exi, np.asarray(hs["kz_only"]))).max() / sc
            res[(N, t)] = r
    wall = tick()
    log_run("c01_lagrangian_pointwise.py",
            "pointwise Lagrangian test of the frozen-line theorem and Cauchy (RK4 on GANDALF RHS + markers)",
            N, M, nsteps, wall, dt=DT, T=TOUT[-1], frozen_runs=len(hs))
    print(f"N = {N}: wall {wall:.0f}s", flush=True)
    for t in TOUT:
        r = res[(N, t)]
        print(f"  t = {t}: max_xi,m |g_m(X(xi,t),t) - h_m(xi,t)| / max|h| = {r['theorem']:.2e};  grad_par: {r['gradpar']:.2e};"
              f"  Cauchy |B(X) - J B0| / max|B_perp| = {r['cauchy']:.2e};  max|det J - 1| = {r['detJ']:.1e};"
              f"  rms displacement {r['disp']:.2f} (wavelength ~ {TWOPI / 1.8:.1f}), max|J - I| = {r['Jdev']:.2f};"
              f"  control (Eulerian, no relabelling) {r['euler']:.2f};  g energy fraction at |k| > 0.75 k_cut: NL {r['outer_NL']:.1e}, FR {r['outer_FR']:.1e}"
              + (f";  control frozen along -Psi0: {r['minus_psi0']:.2f};  control k_z-only: {r['kz_only']:.2f}"
                 if N == CTRL_N else ""), flush=True)

MODE = sys.argv[4] if len(sys.argv) > 4 else "resolved"
for t in TOUT:
    rc = res[(CTRL_N, t)] if (CTRL_N, t) in res else None
    ctrl = min(rc["euler"], rc["minus_psi0"], rc["kz_only"]) if rc else 0.1
    for key, lab in (("theorem", "g(X(xi,t),t) = h(xi,t) pointwise"), ("gradpar", "grad_par(t) U = U grad_par(0) pointwise"),
                     ("cauchy", "Cauchy B(X,t) = J B(xi,0)")):
        e = [res[(N, t)][key] for N in NS]
        if MODE == "resolved":
            # converges: monotone, > 30x over the N range, and the last value far below the O(1) controls
            conv = all(e[i + 1] < e[i] for i in range(len(e) - 1)) and e[-1] < e[0] / 30 and e[-1] < 1e-2 * ctrl
            chk(f"t={t}: {lab}, error falls with N to << the controls", conv, " -> ".join(f"{x:.1e}" for x in e))
        else:
            # strong-flow run: the passive field (and at t = 0.5 the Alfvenic flow itself, see the Cauchy row) reaches
            # the grid scale; nothing is asserted about convergence, the numbers are printed for the record
            print(f"  [INFO] t={t}: {lab} (strong flow, resolution-limited): " + " -> ".join(f"{x:.1e}" for x in e))
    chk(f"t={t}: volume preserved along markers (det J = 1)", max(res[(N, t)]["detJ"] for N in NS) < 1e-8,
        f"{max(res[(N, t)]['detJ'] for N in NS):.1e}")
    if rc:
        chk(f"t={t}: controls are O(1) (test is not vacuous): Eulerian {rc['euler']:.2f}, -Psi0 {rc['minus_psi0']:.2f}, "
            f"k_z-only {rc['kz_only']:.2f}; rms displacement {rc['disp']:.2f}, max|J - I| {rc['Jdev']:.2f}",
            ctrl > 0.05 and rc["Jdev"] > 0.3)
chk.finish()
