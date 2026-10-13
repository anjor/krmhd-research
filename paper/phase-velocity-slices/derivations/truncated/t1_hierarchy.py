"""t1_hierarchy: checks, against the installed krmhd, that GANDALF's Hermite hierarchy for one compressive branch
is exactly a set of M+1 passive slices. It reads the streaming matrix A from krmhd, derives the diagonal weight that
symmetrises it (A = J P_Lam), shows that the spectrum is real and simple for Lam outside [0, 1], builds the slice
basis, verifies the slice equation on krmhd_rhs to round-off, and verifies through gandalf_step that a single-slice
initial condition stays in its slice unless hypercollisions are on. Supports Section 8 (theorem on the truncated
hierarchy) and Appendix C of the paper.

A  krmhd.hermite.compute_streaming_matrix(M, Lam, closure) equals the analytic ladder
     A_{m,m+1} = sqrt((m+1)/2), A_{m+1,m} = sqrt((m+1)/2) (m >= 1), A_{1,0} = c_Lam = (1 - 1/Lam)/sqrt2,
     copy closure ('symmetric', g_{M+1} = g_{M-1}): A_{M,M-1} += sqrt((M+1)/2).
B  (SymPy, symbolic Lam) A = J^(c) P_Lam with P_Lam = I - e0 e0^T/Lam and J^(c) = A(Lam = inf);
     the diagonal weights making Pi A symmetric are unique up to scale: Pi = D_c P_Lam.
C  Pi > 0 iff Lam outside [0, 1]; then A has real, simple eigenvalues. Inside (0, 1) complex pairs appear.
D  krmhd_rhs (float64, brackets comparable to or larger than streaming, beta_i = 1.7) equals
     d_t g = -{Phi, g} - s d_z(A g) - s {Psi, A g},  s = sqrt(beta_i),
   and every slice a_j = w_j . g obeys d_t a_j = -{Phi + s u_j Psi, a_j} - s u_j d_z a_j.
E  Through gandalf_step (IMEX-RK222 and Lawson-RK4; eta, eta_z on; both closures): a single-slice initial
   condition stays in its slice to round-off when nu = 0; with hypercollisions the leakage is proportional to nu.
"""
from __future__ import annotations

import time

import numpy as np
import sympy as sp

from tcore import (Checker, section, ladder, weight, slices, make_grid, random_field, make_state, bracket, dz,
                   SQRT5, jnp)
from krmhd.timestepping import krmhd_rhs, gandalf_step

C = Checker("t1_hierarchy")


def analytic_ladder(M: int, Lam: float, closure: str) -> np.ndarray:
    A = np.zeros((M + 1, M + 1))
    for m in range(M):
        A[m, m + 1] = np.sqrt((m + 1) / 2.0)
        A[m + 1, m] = np.sqrt((m + 1) / 2.0)
    A[1, 0] = (1.0 - (0.0 if np.isinf(Lam) else 1.0 / Lam)) / np.sqrt(2.0)
    if closure == "symmetric":
        A[M, M - 1] += np.sqrt((M + 1) / 2.0)
    return A


section("A. compute_streaming_matrix versus the analytic ladder")
worst = 0.0
for M in (2, 3, 4, 9, 10, 64, 128):
    for Lam in (SQRT5, -SQRT5, 3.0, -0.3, 0.5, np.inf):
        for clo in ("zero", "symmetric"):
            worst = max(worst, np.abs(ladder(M, Lam, clo) - analytic_ladder(M, Lam, clo)).max())
C.check("A1 compute_streaming_matrix == analytic ladder (M=2..128, 6 Lambdas, both closures)", worst < 1e-15,
        f"max |diff| = {worst:.1e}")

section("B. SymPy: factorisation A = J P_Lam and the symmetrising weight")
L = sp.symbols("Lambda", nonzero=True)
okB = True
for M in range(2, 8):
    for clo in ("zero", "symmetric"):
        A = sp.zeros(M + 1, M + 1)
        J = sp.zeros(M + 1, M + 1)
        for m in range(M):
            A[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); A[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
            J[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); J[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
        A[1, 0] = (1 - 1 / L) / sp.sqrt(2)
        if clo == "symmetric":
            A[M, M - 1] += sp.sqrt(sp.Rational(M + 1, 2)); J[M, M - 1] += sp.sqrt(sp.Rational(M + 1, 2))
        P = sp.eye(M + 1); P[0, 0] = 1 - 1 / L
        fact = sp.simplify(A - J * P) == sp.zeros(M + 1, M + 1)
        # unique diagonal weight: unknown diagonal pi_m with pi_0 = 1 - 1/Lam fixed (scale)
        pis = sp.symbols(f"p0:{M + 1}")
        Pi = sp.diag(*pis)
        eqs = [sp.Eq((Pi * A)[i, j], (Pi * A)[j, i]) for i in range(M + 1) for j in range(i + 1, M + 1)]
        sol = sp.solve(eqs + [sp.Eq(pis[0], 1 - 1 / L)], pis, dict=True)
        expect = [1 - 1 / L] + [1] * (M - 1) + [1 if clo == "zero" else sp.sqrt(M) / (sp.sqrt(M) + sp.sqrt(M + 1))]
        uniq = len(sol) == 1 and all(sp.simplify(sol[0][pis[i]] - expect[i]) == 0 for i in range(M + 1))
        okB = okB and fact and uniq
        if not (fact and uniq):
            print("   failure at", M, clo, fact, sol)
C.check("B1 A = J^(c) P_Lam exactly and Pi = D_c P_Lam is the unique diagonal symmetriser (M=2..7, symbolic Lam)", okB)
# numeric confirmation of the weight on GANDALF's own matrix at production M
worst = 0.0
for M in (64, 128):
    for Lam in (SQRT5, -SQRT5, np.inf):
        for clo in ("zero", "symmetric"):
            K = weight(M, Lam, clo)[:, None] * ladder(M, Lam, clo)
            worst = max(worst, np.abs(K - K.T).max())
C.check("B2 Pi A symmetric for GANDALF's matrix at M = 64, 128 (Lam = +-sqrt5, inf; both closures)", worst < 1e-14,
        f"max |Pi A - (Pi A)^T| = {worst:.1e}")

section("C. Positivity of Pi and reality / simplicity of the spectrum")
rows = []
okC = True
for Lam in (SQRT5, -SQRT5, 1.5, 3.0, -0.3, -5.0, np.inf, 0.2, 0.5, 0.9):
    for clo in ("zero", "symmetric"):
        ev = np.linalg.eigvals(ladder(32, Lam, clo))
        imax = np.abs(ev.imag).max()
        er = np.sort(ev.real)
        gap = np.diff(er).min()
        posdef = np.all(weight(32, Lam, clo) > 0)
        outside = not (0.0 <= (0 if np.isinf(Lam) else Lam) <= 1.0) or np.isinf(Lam)
        rows.append((Lam, clo, posdef, imax, gap))
        if outside:
            okC = okC and posdef and imax < 1e-12 and gap > 1e-3
        else:
            okC = okC and (not posdef)
for r in rows:
    print(f"   Lam={r[0]:+8.4f} {r[1]:9s}: Pi>0 {str(r[2]):5s}  max|Im u| = {r[3]:.2e}  min spacing = {r[4]:.3e}")
C.check("C1 Lam outside [0,1]: Pi > 0 and real simple spectrum; Lam in (0,1): Pi indefinite (M = 32)", okC)
# Inside (0,1) the continuum has one purely growing root zeta = i gamma of 1 + zeta Z(zeta) = Lam (Penrose);
# the truncated matrix reproduces it only once its level spacing near u = 0 (~ pi/sqrt(2M)) is small compared
# with gamma, so for Lam near 1 a finite-M GANDALF matrix can have a real spectrum although Pi is indefinite.
from scipy.special import wofz
from scipy.optimize import brentq


def gamma_penrose(Lam: float) -> float:
    f = lambda gm: (1.0 + 1j * gm * (1j * np.sqrt(np.pi) * wofz(1j * gm))).real - Lam
    return brentq(f, 1e-12, 50.0)


okC3 = True
for Lam in (0.2, 0.5, 0.9):
    gp = gamma_penrose(Lam)
    row = f"   Lam={Lam}: continuum growth root gamma = {gp:.6f}; truncated max Im u at M ="
    vals = []
    for M in (16, 32, 64, 128, 256, 512, 1024, 2048):
        vals.append(np.abs(np.linalg.eigvals(ladder(M, Lam)).imag).max())
        row += f" {M}:{vals[-1]:.5f}"
    print(row)
    okC3 = okC3 and abs(vals[-1] - gp) / gp < 0.02
C.check("C3 Lam in (0,1): the truncated spectrum converges to the continuum Penrose root i gamma (M = 2048)", okC3)
okC2 = True
for M in (9, 10, 64, 128):
    for Lam in (SQRT5, -SQRT5):
        for clo in ("zero", "symmetric"):
            A = ladder(M, Lam, clo)
            u, R, W, n = slices(M, Lam, clo)
            e1 = np.abs(W.T @ A - u[:, None] * W.T).max() / np.abs(u).max()
            e2 = np.abs(W.T @ R - np.diag(n)).max()
            e3 = np.abs((R / n[None, :]) @ W.T - np.eye(M + 1)).max()
            okC2 = okC2 and e1 < 1e-12 and e2 < 1e-12 and e3 < 1e-11 and np.all(n > 0)
C.check("C2 left eigenvectors w_j = Pi r_j: W^T A = U W^T, W^T R = N > 0, sum_j r_j w_j^T / n_j = I", okC2)

section("D. krmhd_rhs in matrix form and the exact slice equation")
N = 16
grid = make_grid(N)
mask = np.asarray(grid.dealias_mask)
beta = 1.7
s = np.sqrt(beta)
okD1 = True
okD2 = True
okD3 = True
for M in (2, 9, 10, 16):
    for Lam in (SQRT5, -SQRT5):
        for clo in ("zero", "symmetric"):
            for zamp in (1.0, 300.0):
                rng = np.random.default_rng(1000 * M + int(zamp) + (7 if Lam < 0 else 0))
                zp = random_field(grid, rng, amp=zamp)
                zm = random_field(grid, rng, amp=zamp)
                g = random_field(grid, rng, slope=-1.0, extra_shape=(M + 1,))
                st = make_state(grid, M, Lam, zp, zm, g, beta_i=beta)
                gdot = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure=clo).g)
                A = ladder(M, Lam, clo)
                Phi = (zp + zm) / 2
                Psi = (zp - zm) / 2
                Ag = np.einsum("mn,zyxn->zyxm", A, g)
                pred = np.zeros_like(g)
                sh = np.zeros(3)
                for m in range(M + 1):
                    b1 = bracket(grid, Phi, g[..., m])
                    b2 = s * dz(grid, Ag[..., m])
                    b3 = s * bracket(grid, Psi, Ag[..., m])
                    pred[..., m] = (-b1 - b2 - b3) * mask
                    sh += [np.linalg.norm(b1), np.linalg.norm(b2), np.linalg.norm(b3)]
                pred[0, 0, 0, :] = 0
                err = np.abs(pred - gdot).max() / np.abs(gdot).max()
                okD1 = okD1 and err < 1e-12
                # slice equation
                u, R, W, n = slices(M, Lam, clo)
                a = np.einsum("zyxm,mj->zyxj", g, W)
                adot = np.einsum("zyxm,mj->zyxj", gdot, W)
                worst = 0.0
                ctrl = 0.0
                for j in range(M + 1):
                    bj = bracket(grid, Phi + s * u[j] * Psi, a[..., j])
                    st_j = s * u[j] * dz(grid, a[..., j])
                    pj = (-bj - st_j) * mask
                    pj[0, 0, 0] = 0
                    scale = np.abs(bj).max() + np.abs(st_j).max()
                    worst = max(worst, np.abs(pj - adot[..., j]).max() / scale)
                    if abs(u[j]) > 1e-6:
                        pc = (-bracket(grid, Phi, a[..., j]) - st_j) * mask
                        pc[0, 0, 0] = 0
                        ctrl = max(ctrl, np.abs(pc - adot[..., j]).max() / scale)
                okD2 = okD2 and worst < 1e-12
                okD3 = okD3 and ctrl > 1e-3
                shs = sh / sh.sum()
                print(f"   M={M:2d} Lam={Lam:+.3f} {clo:9s} zamp={zamp:5.0f}: |rhs - matrix form| = {err:.1e}; "
                      f"slice mismatch = {worst:.1e}; control without u_j Psi = {ctrl:.1e}; "
                      f"shares {{Phi,g}}/s dz Ag/s{{Psi,Ag}} = {shs[0]:.2f}/{shs[1]:.2f}/{shs[2]:.2f}")
C.check("D1 krmhd_rhs == -{Phi,g} - s dz(A g) - s{Psi, A g} (beta_i = 1.7)", okD1)
C.check("D2 every slice obeys d_t a_j = -{Phi + s u_j Psi, a_j} - s u_j d_z a_j", okD2)
C.check("D3 control: dropping the u_j Psi advection fails", okD3)

section("E. Slice leakage through gandalf_step (one GANDALF integration per case, recorded below)")
N = 16
grid = make_grid(N)
M = 10
okE1 = True
leak_nu = {}
runs = []
for scheme in ("imex_rk222", "lawson_rk4"):
    for clo in ("zero", "symmetric"):
        for nu in (0.0, 0.5, 1.0):
            Lam = SQRT5
            rng = np.random.default_rng(42)
            zp = random_field(grid, rng, amp=30.0)
            zm = random_field(grid, rng, amp=30.0)
            f = random_field(grid, rng, slope=-1.0)
            u, R, W, n = slices(M, Lam, clo)
            j0 = M - 2          # a fast slice (u ~ +3.7)
            g = f[..., None] * R[:, j0][None, None, None, :]
            st = make_state(grid, M, Lam, zp, zm, g, beta_i=1.0, nu=nu)
            dt, nst = 5e-5, 20
            t0 = time.time()
            for _ in range(nst):
                st = gandalf_step(st, dt, eta=0.5, v_A=1.0, nu=nu, hyper_r=2, hyper_n=2, scheme=scheme,
                                  eta_z=0.3, hyper_rz=2, closure=clo)
            wall = time.time() - t0
            a = np.einsum("zyxm,mj->zyxj", np.asarray(st.g), W)
            norms = np.sqrt((np.abs(a) ** 2).sum(axis=(0, 1, 2)))
            leak = np.delete(norms, j0).max() / norms[j0]
            leak_nu[(scheme, clo, nu)] = leak
            runs.append((scheme, clo, nu, nst, wall))
            print(f"   {scheme:10s} {clo:9s} nu={nu:3.1f}: max_(l != j0) |a_l| / |a_j0| after {nst} steps = {leak:.2e}"
                  f"  (wall {wall:.1f} s incl. compile)")
            if nu == 0.0:
                okE1 = okE1 and leak < 1e-13
C.check("E1 nu = 0: no leakage between slices through gandalf_step (IMEX and Lawson, both closures, eta, eta_z on)",
        okE1)
okE2 = True
for scheme in ("imex_rk222", "lawson_rk4"):
    for clo in ("zero", "symmetric"):
        r = leak_nu[(scheme, clo, 1.0)] / leak_nu[(scheme, clo, 0.5)]
        print(f"   {scheme:10s} {clo:9s}: leakage(nu=1)/leakage(nu=0.5) = {r:.4f} (linear in nu -> 2)")
        okE2 = okE2 and abs(r - 2.0) < 0.01 and leak_nu[(scheme, clo, 1.0)] > 1e-6
C.check("E2 nu > 0: leakage is first order in nu (hypercollisions are the only slice coupling)", okE2)
# first-order prediction: d a_l/dt|_coll = -nu sum_j (W^T C R)_{lj} a_j / n_j, so a slice-j0 start leaks
# |delta a_l| / |a_j0| = nu T |(W^T C R)_{l j0}| / n_j0 over a short time T (here T = 20 * 5e-5)
u, R, W, n = slices(M, SQRT5, "zero")
mm = np.arange(M + 1)
Cd = np.where(mm >= 2, (mm / M) ** 2, 0.0)
coup = W.T @ (Cd[:, None] * R)
j0 = M - 2
pred = 1.0 * (20 * 5e-5) * np.abs(np.delete(coup[:, j0], j0)).max() / n[j0]
meas = leak_nu[("imex_rk222", "zero", 1.0)]
print(f"   first-order estimate of the largest leakage at nu = 1: {pred:.3e}; measured (IMEX, zero closure) {meas:.3e}")
C.check("E3 measured leakage matches the first-order collisional coupling nu T |(W^T C R)_{l j0}| / n_j0",
        abs(meas / pred - 1) < 0.05, f"ratio {meas / pred:.4f}")
print("GANDALF integrations (scheme, closure, nu, steps, wall s):")
for r in runs:
    print(f"   {r[0]} {r[1]} nu={r[2]} steps={r[3]} wall={r[4]:.1f}s grid=16^3 M=10")
C.finish()
