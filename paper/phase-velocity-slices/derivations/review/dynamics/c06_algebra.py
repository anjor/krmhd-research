"""c06: algebraic checks of the two-slice law and Hermite flux, the absence of a Fjortoft-type constraint, the
fourth-order statement, the slices with s u = +-1 and the range of Lambda in the frozen-line theorem, by routes
independent of the scripts in derivations/dynamics.  The script checks the Christoffel-Darboux identity symbolically
and with a general eigen-solver, the normalisation of GANDALF's hermite_flux, that correlated and uncorrelated
injections with equal variances differ in slice energies and Gamma, that free phase mixing with Gamma != 0 moves Gamma
to high m while every invariant is conserved, that 0 < Lambda < 1 gives complex ladder eigenvalues, and the s u = -1
slice for M = 9, Lambda = -sqrt5.  It supports Sections 5, 6 and 7 (and Appendix E for A2).

A1 (two-slice law and Hermite flux) Christoffel-Darboux in matrix form, symbolic: for any tridiagonal A with P0 A
   symmetric and P_m = P0 times the projector on n <= m, P_m A - A^T P_m has exactly two non-zero entries,
   +-p_m A_{m,m+1} at (m, m+1), (m+1, m).  Sandwiching with the eigenvectors gives (u_j' - u_j) K_m(j,j') =
   p_m A_{m,m+1}[v_j(m)v_j'(m+1) - v_j(m+1)v_j'(m)], and with a field-aligned derivative it gives the collisionless
   budget d/dt sum_{n<=m} p_n <g_n^2> = -Pi_m.
A2 (flux convention) krmhd.diagnostics.hermite_flux versus the p_m- and s-weighted flux
   2 s p_m A_{m,m+1} Re[g_m^* grad_par g_{m+1}] per mode (with z+- = 0 the k_z flux used in d06 and d07): krmhd 0.7.0
   (gandalf#159, PR #162) includes s, p_m and the field-line term {Psi, .} and is normalised to the budget of
   (1/2) p_m <g_m^2>, so the ratio is 1/2 at every m, m = 0 included, with z+- = 0 and with Psi != 0.  (krmhd 0.6.0
   returned the k_z part without s and p_0: ratio 1/(s p_0) at m = 0 and 1/s at m >= 1.)
A3 (fourth-order statement, Section 7) correlated (f, rho f) and uncorrelated injections with equal variances: W(m)
   equal, but slice energies and Gamma differ at second order (E_j differ by 2 sqrt2 p_0 rho u_j v_j(0)^2;
   Gamma = sqrt2 p_0 rho vs 0).
A4 (no Fjortoft-type constraint) free phase mixing from g(0) = e_0 + r e_1 (Gamma != 0): Gamma is conserved and all
   of it leaves m <= m*, and sum_m m W(m) grows like lambda while every Q_phi is fixed, so no invariant with a
   non-negative m-density B(m) can obey B(m) >= c m W(m) (the Fjortoft-type criterion of Chandran, Mallet & Meyrand
   2026, section 4.1, with W in the role of their invariant A) along this exact solution.
A5 (range of Lambda in the frozen-line theorem) for 0 < Lambda < 1 the ladder has a complex pair, so
   exp(-t s A grad_par(0)) grows like exp(t s |kappa| Im u): the frozen problem is ill-posed in the continuum (growth
   unbounded in kappa); krmhd's Lawson eigensystem refuses such Lambda.
A6 (slices with s u = +-1) the s u = -1 slice through GANDALF's right-hand side for M = 9, Lambda = -sqrt5 (d10 used
   M = 8, +sqrt5): d_t a = -{z-, a} + d_z a, independent of z+.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import sympy as sp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from scipy.linalg import eigh_tridiagonal  # noqa: E402

from krmhd.diagnostics import hermite_flux  # noqa: E402
from krmhd.hermite import compute_streaming_eigensystem  # noqa: E402
from krmhd.physics import KRMHDState, poisson_bracket_3d  # noqa: E402
from c00_core import Checks, Grid, SQRT5, eig_slices, ladder, make_rhs, mode_field, pw, random_modes  # noqa: E402

chk = Checks()

# ------------------------------------------------------------------------------------------------ A1
Ms = 5
a = sp.symbols(f"a0:{Ms}", positive=True)
p = sp.symbols(f"p0:{Ms + 1}", positive=True)
A = sp.zeros(Ms + 1, Ms + 1)
for m in range(Ms):
    A[m, m + 1] = a[m]
    A[m + 1, m] = p[m] * a[m] / p[m + 1]          # makes P0 A symmetric
P0 = sp.diag(*p)
assert sp.simplify(P0 * A - (P0 * A).T) == sp.zeros(Ms + 1, Ms + 1)
ok = True
for m in range(Ms):
    Pm = sp.diag(*[p[n] if n <= m else 0 for n in range(Ms + 1)])
    D = sp.simplify(Pm * A - A.T * Pm)
    target = sp.zeros(Ms + 1, Ms + 1)
    target[m, m + 1] = p[m] * a[m]
    target[m + 1, m] = -p[m] * a[m]
    ok &= sp.simplify(D - target) == sp.zeros(Ms + 1, Ms + 1)
chk("A1 P_m A - A^T P_m = p_m A_{m,m+1} (E_{m,m+1} - E_{m+1,m}) for symbolic P0-symmetric tridiagonal A (M = 5)", ok)
# numerical consequence with GANDALF's ladder and a general eigen-solver
for M, Lam in ((8, SQRT5), (9, -SQRT5), (64, SQRT5)):
    u, V, WL = eig_slices(M, Lam)
    Al = ladder(M, Lam)
    pl = pw(M, Lam)
    worst = 0.0
    for m in range(M):
        K = V[: m + 1].T @ np.diag(pl[: m + 1]) @ V[: m + 1]
        lhs = (u[None, :] - u[:, None]) * K
        rhs = pl[m] * Al[m, m + 1] * (np.outer(V[m], V[m + 1]) - np.outer(V[m + 1], V[m]))
        worst = max(worst, np.abs(lhs - rhs).max() / max(np.abs(rhs).max(), 1e-300))
    chk(f"A1 Christoffel-Darboux with GANDALF's ladder, M = {M}, Lambda = {Lam:+.3f} (general eig solver)", worst < 1e-9,
        f"{worst:.1e}")

# ------------------------------------------------------------------------------------------------ A2
grid = Grid(12)
rng = np.random.default_rng(3)
M, Lam, beta = 6, SQRT5, 2.0
s = np.sqrt(beta)
g = np.zeros((12, 12, 7, M + 1), complex)
for m in range(M + 1):
    g[..., m] = mode_field(grid, *random_modes(rng))
Al = ladder(M, Lam)
pl = pw(M, Lam)
psi_a2 = mode_field(grid, *random_modes(rng))
ok_a2 = True
for lab, psi in (("z+- = 0", np.zeros_like(psi_a2)), ("Psi != 0 (z+ = -z- = Psi)", psi_a2)):
    st = KRMHDState(z_plus=jnp.asarray(psi), z_minus=jnp.asarray(-psi), g=jnp.asarray(g), M=M, beta_i=beta, v_th=1.0,
                    nu=0.0, Lambda=Lam, time=0.0, grid=grid.g)
    hf = np.asarray(hermite_flux(st))
    ratios, dev = [], 0.0
    for m in range(M):
        gp = 1j * grid.KZ * g[..., m + 1] + np.asarray(poisson_bracket_3d(
            jnp.asarray(psi), jnp.asarray(g[..., m + 1]), grid.jk[0], grid.jk[1], 12, 12, 12, grid.jk[3]))
        mine = 2 * s * pl[m] * Al[m, m + 1] * np.real(np.conj(g[..., m]) * gp)
        sel = np.abs(mine) > 1e-8 * np.abs(mine).max()
        ratios.append(np.median(hf[..., m][sel] / mine[sel]))
        dev = max(dev, np.abs(hf[..., m] - mine / 2).max() / np.abs(mine).max())
    ratios = np.array(ratios)
    print(f"    {lab}: hermite_flux / (2 s p_m A_(m,m+1) Re[g_m^* grad_par g_(m+1)]), m = 0..{M - 1}: "
          + ", ".join(f"{r:.4f}" for r in ratios) + f";  max per-mode deviation from 1/2 x {dev:.1e}"
          + f"  (krmhd 0.6.0 with z+- = 0: 1/(s p_0) = {1 / (s * pl[0]):.4f}, 1/s = {1 / s:.4f})")
    ok_a2 = ok_a2 and np.abs(ratios - 0.5).max() < 1e-10 and dev < 1e-12
chk("A2 krmhd 0.7.0's hermite_flux (gandalf#159, PR #162) is 1/2 x the p_m- and s-weighted field-aligned flux at every m, "
    "m = 0 included (z+- = 0 and Psi != 0)", ok_a2)

# ------------------------------------------------------------------------------------------------ A3
for M in (8, 64):
    u, V, WL = eig_slices(M, SQRT5)
    p0 = 1 - 1 / SQRT5
    v0 = V[0]
    for rho in (0.5, 1.0):
        Ec = (v0 * (p0 + np.sqrt(2) * rho * u)) ** 2           # correlated: g = (f, rho f), <f^2> = 1
        Eu = v0 ** 2 * (p0 ** 2 + 2 * rho ** 2 * u ** 2)          # uncorrelated: g = (f, rho f'), independent f'
        Gc, Gu = float(np.sum(u * Ec)), float(np.sum(u * Eu))
        dE = np.abs(Ec - Eu - 2 * np.sqrt(2) * p0 * rho * u * v0 ** 2).max()
        print(f"    M = {M}, rho = {rho}: W(m) at injection equal (both (1, rho^2, 0, ...)); sum_j E_j: {Ec.sum():.6f} vs "
              f"{Eu.sum():.6f};  max_j |E_j(corr) - E_j(uncorr)| = {np.abs(Ec - Eu).max():.4f} (largest E_j "
              f"{max(Ec.max(), Eu.max()):.4f});  Gamma = sum u_j E_j: {Gc:.6f} vs {Gu:.1e} (sqrt2 p0 rho = {np.sqrt(2) * p0 * rho:.6f})")
        chk(f"A3 M={M} rho={rho}: second-order statistics are NOT all identical (slice energies, Gamma differ)",
            abs(Gc - np.sqrt(2) * p0 * rho) < 1e-10 and abs(Gu) < 1e-10 and dE < 1e-12
            and np.abs(Ec - Eu).max() > 1e-2 * max(Ec.max(), Eu.max()) and abs(Ec.sum() - Eu.sum()) < 1e-12)

# ------------------------------------------------------------------------------------------------ A4
for M, Lam, lab in ((3000, 1e300, "free streaming"), (3000, SQRT5, "Landau Lambda = sqrt5")):
    Al = ladder(M, Lam)
    pl = pw(M, Lam)
    ph = np.sqrt(pl)
    e = np.array([ph[m] * Al[m, m + 1] / ph[m + 1] for m in range(M)])
    uu, Q = eigh_tridiagonal(np.zeros(M + 1), e)
    g0 = np.zeros(M + 1)
    g0[0], g0[1] = 1.0, 0.8
    c = Q.T @ (ph * g0)
    Gam_m = lambda gg: 2 * pl[:-1] * Al[np.arange(M), np.arange(1, M + 1)] * np.real(np.conj(gg[:-1]) * gg[1:])  # noqa: E731
    rows = []
    for th in (0.0, 5.0, 10.0, 20.0, 30.0):
        gg = (Q @ (np.exp(-1j * th * uu) * c)) / ph
        Wm = pl * np.abs(gg) ** 2
        Gm = Gam_m(gg)
        Q2 = float(np.real(np.conj(gg) @ ((pl[:, None] * Al) @ (Al @ gg))))
        mW = float(np.sum(np.arange(M + 1) * Wm))
        rows.append((th, Wm.sum(), Gm.sum(), np.abs(Gm[:21]).sum() / abs(Gm.sum()), Q2, mW))
    print(f"    {lab}: (s k t, W, Gamma, sum_(m<=20)|Gamma(m)|/|Gamma|, Q_u^2, sum_m m W(m)): "
          + "; ".join(f"({th:.0f}, {W_:.6f}, {G_:+.6f}, {f_:.1e}, {q_:.6f}, {mw:.1f})" for th, W_, G_, f_, q_, mw in rows))
    cons = max(abs(r[1] - rows[0][1]) for r in rows) + max(abs(r[2] - rows[0][2]) for r in rows) + max(abs(r[4] - rows[0][4]) for r in rows)
    chk(f"A4 {lab}: W, Gamma, Q_u^2 conserved while Gamma(m <= 20) -> 0 and sum m W(m) grows ~ (s k t)^2/2",
        cons < 1e-9 and rows[-1][3] < 1e-6 and rows[-1][5] > 100 * max(rows[0][5], 1), f"conservation residual {cons:.1e}")

# ------------------------------------------------------------------------------------------------ A5
for M in (8, 64):
    ev = np.linalg.eigvals(ladder(M, 0.5))
    im = np.abs(np.imag(ev)).max()
    print(f"    Lambda = 0.5, M = {M}: max |Im u_j| = {im:.4f} -> growth rate s |kappa| {im:.4f}, unbounded in kappa")
    chk(f"A5 Lambda = 0.5, M = {M}: the ladder has complex eigenvalues (ill-posed frozen problem in the continuum)", im > 1e-3)
try:
    compute_streaming_eigensystem(8, 0.5, "zero")
    refused = False
except ValueError:
    refused = True
chk("A5 krmhd.hermite.compute_streaming_eigensystem refuses Lambda = 0.5 (Lawson stepper cannot run it)", refused)

# ------------------------------------------------------------------------------------------------ A6
M, Lam = 9, -SQRT5
u, V, WL = eig_slices(M, Lam)
j = int(np.argmin(np.abs(u + 1.0)))
s_a6 = -1.0 / u[j]                # s = sqrt(beta) that puts slice j at s u_j = -1
grid = Grid(20)
rng = np.random.default_rng(12)
zp = mode_field(grid, *random_modes(rng, (1, 3.5), 3))
zm = mode_field(grid, *random_modes(rng, (1, 3.5), 3))
zp2 = mode_field(grid, *random_modes(rng, (1, 3.5), 3))
zp *= 3 / grid.rms_grad(zp)
zm *= 2 / grid.rms_grad(zm)
zp2 *= 4 / grid.rms_grad(zp2)
g = np.zeros((20, 20, 11, M + 1), complex)
for m in range(M + 1):
    g[..., m] = mode_field(grid, *random_modes(rng, (1, 3.5), 3))
rhs = make_rhs(grid, M, Lam, s_a6 ** 2)
aj = g @ WL[j]
d1 = np.asarray(rhs(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g))[2]) @ WL[j]
d2 = np.asarray(rhs(jnp.asarray(zp2), jnp.asarray(zm), jnp.asarray(g))[2]) @ WL[j]
br = np.asarray(poisson_bracket_3d(jnp.asarray(zm), jnp.asarray(aj), grid.jk[0], grid.jk[1], 20, 20, 20, grid.jk[3]))
pred = -br + 1j * grid.KZ * aj
pred[0, 0, 0] = 0
e1 = np.abs(d1 - pred).max() / np.abs(pred).max()
e2 = np.abs(d2 - d1).max() / np.abs(d1).max()
print(f"    M = 9, Lambda = -sqrt5: u_j = {u[j]:.6f}, beta = {s_a6 ** 2:.6f}; slice equation residual {e1:.1e}; change when z+ "
      f"is replaced: {e2:.1e}")
chk("A6 s u = -1 slice: d_t a = -{z-, a} + d_z a and independent of z+ (M = 9, Lambda = -sqrt5)", e1 < 1e-13 and e2 < 1e-13)
chk.finish()
