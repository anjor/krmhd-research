"""t2_eigenvalues: checks the phase velocities u_j of the truncated hierarchy, the eigenvalues of GANDALF's ladder
matrix A: Gauss-Hermite nodes and Christoffel numbers at Lam = inf; the characteristic-polynomial identity, the
eigen-condition and the sum rules at finite Lam; the first-order shift and the asymptotic spectral-shift law; the
parity symmetry and the exact zero eigenvalue at even M with its u = 0 slice (Im H); the size and Lam-independence
of u_max and its large-M asymptotics for both closures; and the conventions (Lambda+- of CMM, the Landau root, the
propagation direction of z+). Supports Section 8 and Appendix C of the paper.

A  Lam = inf, zero closure: u_j = zeros x_j of H_{M+1}; right eigenvectors r_j[m] = p_m(x_j) (orthonormal
   Hermite polynomials), n_j = 1/lambda_j (lambda_j = Gauss-Hermite weights normalised to sum 1).
B  Finite Lam (SymPy, symbolic Lam and z, both closures): det(z - A) = det(z - J) + det(z - J^(2)) / (2 Lam);
   eigen-condition rho_{M+1}(u) = 0 (zero closure) or rho_{M+1}(u) = rho_{M-1}(u) (copy closure), where rho_m is
   the GANDALF recurrence started with rho_0 = 1; sum rules sum u_j = 0, sum u_j^2 = M(M+1)/2 - 1/Lam (+ sqrt(M(M+1))
   for the copy closure).
C  First-order shift u_j = x_j - lambda_j x_j / Lam + O(Lam^-2); asymptotic spectral-shift law
   (u_j - x_j)/Delta_j -> arg eps(x_j + i0)/pi with eps(z) = 1 - (1 + z Z(z))/Lam.
D  Parity S A S = -A (S = diag((-1)^m)): spectrum symmetric; even M has an exact zero eigenvalue (any Lam, both
   closures) with left null vector psi_m(0) (zero closure); the u = 0 slice is g(v = 0) and is advected by Phi only
   (checked on krmhd_rhs), its energy (pi/2) sum_k |a_0|^2 is CMM's Im H (conserved, CMM eq. 2.23); odd M: det A != 0.
E  u_max: table for M = 16..256, Lam = +-sqrt5, inf, both closures; Lam-dependence exponentially small
   (mpmath check of the first-order formula for the extreme eigenvalue).
F  Asymptotics of u_max(M):
     zero closure: u_max = nu^(1/2) + 2^(-1/3) a_1 nu^(-1/6) - 2^(-2/3) a_1^2 nu^(-5/6)/10 + O(nu^(-3/2)), nu = 2M+3,
     copy closure: u_max = nu^(1/2) + 2^(-1/3) a'_1 nu^(-1/6) - 2^(-5/3) (4/(5 a'_1) + a'_1^2/5) nu^(-5/6) + O(nu^(-3/2)),
                   nu = 2M+1,
   a_1, a'_1 the first zeros of Ai and Ai'. Checked to M = 16384 and by an unconstrained fit.
G  Conventions: Lam+- = +-sqrt5 from CMM (B7) at beta_i = 1, T_i = T_e; A is the matrix of u = v_par/v_th with
   F_M ~ exp(-u^2) (the least-damped eigenvalue of GANDALF's operator with strong hypercollisions equals -i k s zeta_L,
   zeta_L the Landau root of 1 - (1 + zeta Z(zeta))/Lam = 0); z+ propagates toward -z in krmhd_rhs.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import mpmath as mp
from scipy.special import roots_hermite, wofz, ai_zeros
from scipy.linalg import eigh_tridiagonal

from tcore import (Checker, section, ladder, weight, slices, hermite_functions, orthonormal_polys, make_grid,
                   random_field, make_state, bracket, qform, dqform, SQRT5)
from krmhd.timestepping import krmhd_rhs

C = Checker("t2_eigenvalues")

section("A. Lam = inf: Gauss-Hermite nodes and Christoffel numbers")
okA = True
for M in (8, 9, 64, 128, 255, 256):
    x, w = roots_hermite(M + 1)
    o = np.argsort(x); x = x[o]; lam = (w / np.sqrt(np.pi))[o]
    u, R, W, n = slices(M, np.inf)
    e_nodes = np.abs(x - u).max()
    # eigenvector j is p(x_j)/|p(x_j)| up to sign; compare |r_j . phat_j| with 1 (no division by tiny components)
    P = orthonormal_polys(M, u)
    nrm2 = (P ** 2).sum(axis=0)
    Ph = P / np.sqrt(nrm2)[None, :]
    e_vec = np.abs(1.0 - np.abs(np.einsum("mj,mj->j", R, Ph))).max()
    # Christoffel numbers: |p(x_j)|^2 = 1/lambda_j (checked where lambda_j is representable, > 1e-250)
    ok = lam > 1e-250
    e_chr = np.abs(nrm2[ok] * lam[ok] - 1.0).max()
    okA = okA and e_nodes < 1e-11 and e_vec < 1e-12 and e_chr < 1e-9
    print(f"   M={M:3d}: max|u_j - x_j(GH)| = {e_nodes:.1e}; max(1 - |r_j . p(x_j)/|p(x_j)||) = {e_vec:.1e}; "
          f"max| |p(x_j)|^2 lambda_j - 1 | = {e_chr:.1e}")
C.check("A1 Lam = inf: u_j = zeros of H_{M+1}; r_j proportional to p_m(x_j); |p(x_j)|^2 = 1/lambda_j", okA)

section("B. Finite Lam: characteristic polynomial, eigen-condition, sum rules")
L, z = sp.symbols("Lambda z")
okB = True
for M in range(3, 8):
    for clo in ("zero", "symmetric"):
        def mat(LL):
            A = sp.zeros(M + 1, M + 1)
            for m in range(M):
                A[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); A[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
            if LL is not None:
                A[1, 0] = (1 - 1 / LL) / sp.sqrt(2)
            if clo == "symmetric":
                A[M, M - 1] += sp.sqrt(sp.Rational(M + 1, 2))
            return A
        A = mat(L)
        J = mat(None)
        lhs = (z * sp.eye(M + 1) - A).det()
        rhs = (z * sp.eye(M + 1) - J).det() + (z * sp.eye(M - 1) - J[2:, 2:]).det() / (2 * L)
        okB = okB and sp.simplify(sp.expand(lhs - rhs)) == 0
C.check("B1 det(z - A_Lam) = det(z - J) + det(z - J^(2))/(2 Lam), M = 3..7, both closures (SymPy)", okB)


def rho(M: int, Lam: float, u: np.ndarray) -> np.ndarray:
    """GANDALF recurrence: u rho_0 = rho_1/sqrt2; u rho_1 = c_Lam rho_0 + rho_2; u rho_m = sqrt(m/2) rho_{m-1}
    + sqrt((m+1)/2) rho_{m+1} (m >= 2). Returns rho_0..rho_{M+1}."""
    cL = (1.0 - (0.0 if np.isinf(Lam) else 1.0 / Lam)) / np.sqrt(2.0)
    r = np.zeros((M + 2, u.size))
    r[0] = 1.0
    r[1] = np.sqrt(2.0) * u
    r[2] = u * r[1] - cL * r[0]
    for m in range(2, M + 1):
        r[m + 1] = (u * r[m] - np.sqrt(m / 2.0) * r[m - 1]) / np.sqrt((m + 1) / 2.0)
    return r


okB2 = True
for M in (16, 33, 64):
    for Lam in (SQRT5, -SQRT5, 4.0):
        for clo in ("zero", "symmetric"):
            u, R, W, n = slices(M, Lam, clo)
            r = rho(M, Lam, u)
            cond = r[M + 1] if clo == "zero" else r[M + 1] - r[M - 1]
            rel = np.abs(cond).max() / np.abs(r).max()
            rh = r[:M + 1] / np.linalg.norm(r[:M + 1], axis=0)[None, :]
            Rh = R / np.linalg.norm(R, axis=0)[None, :]
            vec = np.abs(1.0 - np.abs(np.einsum("mj,mj->j", Rh, rh))).max()
            okB2 = okB2 and rel < 1e-10 and vec < 1e-12
            print(f"   M={M} Lam={Lam:+.3f} {clo:9s}: eigen-condition residual {rel:.1e}; 1 - |r_j . rho(u_j)| (unit vectors) = {vec:.1e}")
C.check("B2 eigenvalues are the roots of rho_{M+1} (zero) / rho_{M+1} - rho_{M-1} (copy); r_j[m] = rho_m(u_j)", okB2)
okB3 = True
for M in (10, 64, 128):
    for Lam in (SQRT5, -SQRT5, 1.5):
        for clo in ("zero", "symmetric"):
            u = slices(M, Lam, clo)[0]
            t2 = M * (M + 1) / 2 - 1 / Lam + (np.sqrt(M * (M + 1)) if clo == "symmetric" else 0.0)
            okB3 = okB3 and abs(u.sum()) < 1e-10 and abs((u ** 2).sum() - t2) / t2 < 1e-13
C.check("B3 sum rules: sum u_j = 0, sum u_j^2 = M(M+1)/2 - 1/Lam [+ sqrt(M(M+1)) copy closure]", okB3)

section("C. Effect of finite Lam: first-order shift and the asymptotic spectral-shift law")
M = 128
x, wgh = roots_hermite(M + 1)
o = np.argsort(x); x = x[o]; lam = (wgh / np.sqrt(np.pi))[o]
errs = {}
for Lam in (50.0, 500.0, 5000.0):
    u = slices(M, Lam)[0]
    errs[Lam] = np.abs((u - x) + lam * x / Lam).max()
print("   max_j |u_j - x_j + lambda_j x_j / Lam| at Lam = 50, 500, 5000:",
      ", ".join(f"{errs[k]:.2e}" for k in errs))
C.check("C1 u_j = x_j - lambda_j x_j/Lam + O(Lam^-2) (error falls 100x per 10x in Lam)",
        8e1 < errs[50.0] / errs[500.0] < 1.2e2 and 8e1 < errs[500.0] / errs[5000.0] < 1.2e2)
for Lam in (SQRT5, -SQRT5):
    u = slices(M, Lam)[0]
    D = np.gradient(x)
    first = -lam * x / Lam
    print(f"   Lam={Lam:+.4f}, M=128: max_j |u_j - x_j| = {np.abs(u - x).max():.3e}, max_j |u_j - x_j|/Delta_j = {np.abs((u - x) / D).max():.3e}, "
          f"first-order formula error / Delta_j = {np.abs((u - x - first) / D).max():.3e}")


def eps_real(u: np.ndarray, Lam: float) -> np.ndarray:
    Z = 1j * np.sqrt(np.pi) * wofz(u)          # Z(u + i0) for real u
    return 1.0 - (1.0 + u * Z) / Lam


okC2 = True
for Lam in (SQRT5, -SQRT5):
    row = f"   Lam={Lam:+.4f}: max_(|x_j|<3) |(u_j - x_j)/Delta_j - arg eps(x_j+i0)/pi| at M ="
    vals = []
    for M in (128, 512, 2048):
        x, _ = roots_hermite(M + 1)
        x = np.sort(x)
        u = slices(M, Lam)[0]
        D = np.gradient(x)
        xi = np.angle(eps_real(x, Lam)) / np.pi
        sel = np.abs(x) < 3
        vals.append(np.abs((u - x)[sel] / D[sel] - xi[sel]).max())
        row += f" {M}:{vals[-1]:.2e}"
    print(row + f"  (max |arg eps|/pi on |x|<3 = {np.abs(xi[sel]).max():.3f})")
    okC2 = okC2 and vals[2] < vals[0] and vals[2] < 3e-3
C.check("C2 spectral-shift law (u_j - x_j)/Delta_j -> arg eps(x_j + i0)/pi holds asymptotically in M", okC2)
# the perturbation determinant behind the shift law: det(z - A)/det(z - J) = 1 + e0^T (z - J)^-1 e1 /(Lam sqrt2)
# (matrix determinant lemma, exact) -> eps(z) = 1 - (1 + z Z(z))/Lam for Im z > 0 as M -> infinity
okC3 = True
for Lam in (SQRT5, -SQRT5):
    row = f"   Lam={Lam:+.4f}:"
    for zc in (0.3 + 0.5j, -1.2 + 0.3j, 2.0 + 1.0j):
        errs = []
        for M in (32, 128, 512):
            J = ladder(M, np.inf)
            A = ladder(M, Lam)
            I = np.eye(M + 1)
            s1, l1 = np.linalg.slogdet(zc * I - A)
            s2, l2 = np.linalg.slogdet(zc * I - J)
            ratio = s1 / s2 * np.exp(l1 - l2)
            lemma = 1 + np.linalg.solve(zc * I - J, I[:, 1])[0] / (Lam * np.sqrt(2))
            epsz = 1 - (1 + zc * 1j * np.sqrt(np.pi) * wofz(zc)) / Lam
            okC3 = okC3 and abs(ratio - lemma) < 1e-11
            errs.append(abs(ratio - epsz))
        okC3 = okC3 and errs[-1] < 1e-8 and errs[-1] < errs[0]
        row += f" z={zc}: |det ratio - eps(z)| at M=32,128,512 = " + ", ".join(f"{e:.1e}" for e in errs) + ";"
    print(row)
C.check("C3 det(z - A)/det(z - J) = 1 + e0^T(z-J)^-1 e1/(Lam sqrt2) exactly, -> eps(z) for Im z > 0", okC3)

section("D. Parity, the exact zero eigenvalue at even M, the u = 0 slice and Im H")
okD = True
for M in (10, 11, 64, 65, 128):
    for Lam in (SQRT5, -SQRT5, 0.5, np.inf):
        for clo in ("zero", "symmetric"):
            A = ladder(M, Lam, clo)
            S = np.diag((-1.0) ** np.arange(M + 1))
            par = np.abs(S @ A @ S + A).max()
            ev = np.linalg.eigvals(A)
            mn = np.abs(ev).min()
            okD = okD and par == 0.0 and ((mn < 1e-12) if M % 2 == 0 else (mn > 1e-3))
C.check("D1 S A S = -A; |u|_min = 0 to round-off for even M (any Lam incl. 0.5, both closures), > 0 for odd M", okD)
# symbolic det for odd M
okD2 = True
for M in (3, 5, 7):
    for clo in ("zero", "symmetric"):
        A = sp.zeros(M + 1, M + 1)
        for m in range(M):
            A[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); A[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
        A[1, 0] = (1 - 1 / L) / sp.sqrt(2)
        if clo == "symmetric":
            A[M, M - 1] += sp.sqrt(sp.Rational(M + 1, 2))
        pred = (-1) ** ((M + 1) // 2) * sp.prod([A[2 * i, 2 * i + 1] * A[2 * i + 1, 2 * i] for i in range((M + 1) // 2)])
        okD2 = okD2 and sp.simplify(A.det() - pred) == 0
        print(f"   M={M} {clo:9s}: det A = {sp.simplify(A.det())}")
C.check("D2 odd M: det A = (-1)^((M+1)/2) prod_i A_{2i,2i+1} A_{2i+1,2i} (nonzero for Lam != 1)", okD2)
okD3 = True
for M in (10, 64, 128):
    w0 = hermite_functions(M, [0.0])[:, 0]
    for Lam in (SQRT5, -SQRT5, 0.5, np.inf):
        A = ladder(M, Lam, "zero")
        okD3 = okD3 and np.abs(w0 @ A).max() < 1e-13
        Ac = ladder(M, Lam, "symmetric")
        wc = w0.copy(); wc[M] *= np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
        okD3 = okD3 and np.abs(wc @ Ac).max() < 1e-13
M = 9
w0 = hermite_functions(M, [0.0])[:, 0]
leak = w0 @ ladder(M, SQRT5)
print(f"   M=9: psi(0)^T A = {np.array2string(leak, precision=3)}; predicted last entry psi_8(0) sqrt(9/2) = "
      f"{w0[M - 1] * np.sqrt(M / 2):.6f}")
okD3 = okD3 and np.abs(leak[:M]).max() < 1e-14 and abs(leak[M] - w0[M - 1] * np.sqrt(M / 2)) < 1e-14
C.check("D3 left null vector psi_m(0) (zero closure, any Lam), last entry x sqrt(M)/(sqrt(M)+sqrt(M+1)) (copy); "
        "odd M leaks psi_(M-1)(0) sqrt(M/2) g_M", okD3)
# krmhd_rhs: the u = 0 slice a_0 = sum_m psi_m(0) g_m = g(v=0) is advected by Phi only; Im H conserved
N = 16
grid = make_grid(N)
okD4 = True
for M in (10, 16):
    for Lam in (SQRT5, -SQRT5):
        rng = np.random.default_rng(77 + M)
        zp = random_field(grid, rng, amp=200.0)
        zm = random_field(grid, rng, amp=200.0)
        g = random_field(grid, rng, slope=-1.0, extra_shape=(M + 1,))
        st = make_state(grid, M, Lam, zp, zm, g, beta_i=1.3)
        gdot = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0).g)
        w0 = hermite_functions(M, [0.0])[:, 0]
        a0 = g @ w0
        a0dot = gdot @ w0
        pred = -bracket(grid, (zp + zm) / 2, a0) * np.asarray(grid.dealias_mask)
        pred[0, 0, 0] = 0
        e = np.abs(pred - a0dot).max() / np.abs(pred).max()
        K = (np.pi / 2) * np.outer(w0, w0)
        val, scale = dqform(grid, g, gdot, K)
        okD4 = okD4 and e < 1e-12 and abs(val) / scale < 1e-14
        print(f"   M={M} Lam={Lam:+.3f}: |d_t a0 + {{Phi, a0}}|/|{{Phi, a0}}| = {e:.1e}; Im H = (pi/2) sum|a0|^2 = "
              f"{qform(grid, g, K):.4e}, relative d/dt = {abs(val) / scale:.1e}")
C.check("D4 krmhd_rhs: d_t g(v=0) = -{Phi, g(v=0)} exactly and Im H = (pi/2) sum_k |g(v=0)|^2 is conserved", okD4)

section("E. u_max for GANDALF's matrix")
print("     M |  Lam=+sqrt5 zero | Lam=-sqrt5 zero | Lam=inf zero    | Lam=+sqrt5 copy | Lam=-sqrt5 copy | Lam=inf copy")
table = {}
for M in (16, 32, 64, 65, 128, 129, 256):
    row = f"   {M:3d} |"
    for clo in ("zero", "symmetric"):
        for Lam in (SQRT5, -SQRT5, np.inf):
            um = np.abs(np.linalg.eigvals(ladder(M, Lam, clo)).real).max()
            table[(M, Lam, clo)] = um
            row += f" {um:15.10f} |"
    print(row)
C.check("E1 u_max = 10.6 at M = 64 and 15.4 at M = 128 for Lam = sqrt5 (zero closure)",
        abs(table[(64, SQRT5, 'zero')] - 10.616) < 1e-3 and abs(table[(128, SQRT5, 'zero')] - 15.355) < 1e-3,
        f"{table[(64, SQRT5, 'zero')]:.6f}, {table[(128, SQRT5, 'zero')]:.6f}; Lam=-sqrt5: "
        f"{table[(64, -SQRT5, 'zero')]:.6f}, {table[(128, -SQRT5, 'zero')]:.6f}; copy closure: "
        f"{table[(64, SQRT5, 'symmetric')]:.6f}, {table[(128, SQRT5, 'symmetric')]:.6f}")
dmax = max(abs(table[(M, Lam, c)] - table[(M, np.inf, c)]) for M in (32, 64, 128, 256) for Lam in (SQRT5, -SQRT5)
           for c in ("zero", "symmetric"))
C.check("E2 u_max does not depend on Lam to float64 precision for M >= 32", dmax < 1e-12, f"max diff {dmax:.1e}")
# high precision: first-order shift of the extreme eigenvalue is -lambda_max x_max / Lam (exponentially small)
mp.mp.dps = 80
okE3 = True
for M in (12, 20):
    for Lam in (mp.sqrt(5), -mp.sqrt(5)):
        def mpmat(LL):
            A = mp.zeros(M + 1, M + 1)
            for m in range(M):
                A[m, m + 1] = mp.sqrt(mp.mpf(m + 1) / 2); A[m + 1, m] = mp.sqrt(mp.mpf(m + 1) / 2)
            if LL is not None:
                A[1, 0] = (1 - 1 / LL) / mp.sqrt(2)
            return A
        eJ = sorted([mp.re(e) for e in mp.eig(mpmat(None))[0]])
        eA = sorted([mp.re(e) for e in mp.eig(mpmat(Lam))[0]])
        xm = eJ[-1]
        p = [mp.mpf(1), mp.sqrt(2) * xm]
        for m in range(1, M):
            p.append((xm * p[m] - mp.sqrt(mp.mpf(m) / 2) * p[m - 1]) / mp.sqrt(mp.mpf(m + 1) / 2))
        lam_max = 1 / mp.fsum([pp ** 2 for pp in p])
        # Christoffel numbers of all nodes (for the second-order term)
        lams, xs = [], eJ
        for xk in xs:
            q = [mp.mpf(1), mp.sqrt(2) * xk]
            for m in range(1, M):
                q.append((xk * q[m] - mp.sqrt(mp.mpf(m) / 2) * q[m - 1]) / mp.sqrt(mp.mpf(m + 1) / 2))
            lams.append(1 / mp.fsum([qq ** 2 for qq in q]))
        shift = eA[-1] - xm
        first = -lam_max * xm / Lam
        second = (lam_max * xm / Lam ** 2) * mp.fsum([lams[k] * xs[k] / (xm - xs[k]) for k in range(M)])
        rel1 = abs(shift / first - 1)
        rel2 = abs((shift - first - second) / first)
        okE3 = okE3 and rel2 < 0.1 * rel1 and rel1 < 0.02
        print(f"   M={M} Lam={mp.nstr(Lam, 6)}: u_max - x_max = {mp.nstr(shift, 8)}; first order -lambda_max x_max/Lam = "
              f"{mp.nstr(first, 8)} (rel. diff {mp.nstr(rel1, 3)}); + second order "
              f"(lambda_max x_max/Lam^2) sum_k lambda_k x_k/(x_max - x_k) -> rel. diff {mp.nstr(rel2, 3)}; "
              f"estimate 1/(2 |Lam| x_max^2) = {mp.nstr(1 / (2 * abs(Lam) * xm ** 2), 3)}")
C.check("E3 (mpmath, 80 digits) the Lam-shift of u_max is -lambda_max x_max/Lam to O(1/(Lam x_max^2)) relative; "
        "the exact second-order term removes the residual", okE3)

section("F. Asymptotics of u_max(M)")
a1 = ai_zeros(1)[0][0]
a1p = ai_zeros(1)[1][0]
print(f"   a_1 = {a1:.10f}, a'_1 = {a1p:.10f}; 2^(-1/3)|a_1| = {2 ** (-1 / 3) * abs(a1):.6f}, "
      f"2^(-2/3) a_1^2/10 = {2 ** (-2 / 3) * a1 ** 2 / 10:.6f}, 2^(-1/3)|a'_1| = {2 ** (-1 / 3) * abs(a1p):.6f}, "
      f"-2^(-5/3)(4/(5a'_1) + a'_1^2/5) = {-2 ** (-5 / 3) * (4 / (5 * a1p) + a1p ** 2 / 5):.6f}")


def umax_inf(M: int, clo: str) -> float:
    off = np.sqrt(np.arange(1, M + 1) / 2.0)
    if clo == "symmetric":
        off[-1] = np.sqrt(np.sqrt(M / 2.0) * (np.sqrt(M / 2.0) + np.sqrt((M + 1) / 2.0)))
    return float(eigh_tridiagonal(np.zeros(M + 1), off, eigvals_only=True, select="i", select_range=(M, M))[0])


Ms = np.array([8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
okF = True
for clo in ("zero", "symmetric"):
    nu = 2 * Ms + (3 if clo == "zero" else 1)
    aa = a1 if clo == "zero" else a1p
    c2 = -2 ** (-2 / 3) * a1 ** 2 / 10 if clo == "zero" else -2 ** (-5 / 3) * (4 / (5 * a1p) + a1p ** 2 / 5)
    um = np.array([umax_inf(int(M), clo) for M in Ms])
    two = np.sqrt(nu) + 2 ** (-1 / 3) * aa * nu ** (-1 / 6)
    three = two + c2 * nu ** (-5 / 6)
    r2 = (um - two) * nu ** (5 / 6)
    r3 = (um - three) * nu ** 1.5
    print(f"   {clo} closure (nu = 2M + {3 if clo == 'zero' else 1}):")
    for i, M in enumerate(Ms):
        print(f"     M={M:6d}: u_max = {um[i]:.12f}; (u_max - two-term) nu^(5/6) = {r2[i]:+.6f} (-> {c2:+.6f});"
              f" (u_max - three-term) nu^(3/2) = {r3[i]:+.6f}")
    # unconstrained least-squares fit of the nu^(-1/6) and nu^(-5/6) coefficients (+ a nu^(-3/2) term), M >= 64
    sel = Ms >= 64
    X = np.vstack([nu[sel] ** (-1 / 6), nu[sel] ** (-5 / 6), nu[sel] ** (-1.5)]).T
    coef = np.linalg.lstsq(X, um[sel] - np.sqrt(nu[sel]), rcond=None)[0]
    print(f"     fit (M >= 64): u_max - sqrt(nu) = {coef[0]:+.6f} nu^(-1/6) {coef[1]:+.6f} nu^(-5/6) {coef[2]:+.4f} nu^(-3/2)"
          f"   [predicted {2 ** (-1 / 3) * aa:+.6f}, {c2:+.6f}]")
    okF = okF and abs(r2[-1] - c2) < 2e-3 and abs(r3[-1] - r3[-2]) < 1e-3 and abs(coef[0] - 2 ** (-1 / 3) * aa) < 1e-4 \
        and abs(coef[1] - c2) < 1e-2
    if clo == "zero":
        print(f"     remainder coefficient of nu^(-3/2) (zero closure, from M = 16384): {r3[-1]:+.4f}")
    else:
        print(f"     remainder coefficient of nu^(-3/2) (copy closure, from M = 16384): {r3[-1]:+.4f}")
C.check("F1 u_max asymptotics (both closures): leading Airy term and the derived nu^(-5/6) term confirmed to M = 16384",
        okF)
for M in (64, 128):
    nu = 2 * M + 3
    print(f"   M={M}: three-term zero-closure formula {np.sqrt(nu) + 2 ** (-1 / 3) * a1 * nu ** (-1 / 6) - 2 ** (-2 / 3) * a1 ** 2 / 10 * nu ** (-5 / 6):.6f}"
          f" vs exact {table[(M, SQRT5, 'zero')]:.6f}")

section("G. Conventions")
tau, Zc, bi = sp.symbols("tau Z beta_i", positive=True)
Lpm = [-tau / Zc + 1 / bi + sgn * sp.sqrt((1 + tau / Zc) ** 2 + 1 / bi ** 2) for sgn in (1, -1)]
vals = [sp.simplify(Lx.subs({tau: 1, Zc: 1, bi: 1})) for Lx in Lpm]
C.check("G1 CMM (B7) Lambda+- at beta_i = 1, T_i = T_e (tau = Z = 1) = +-sqrt5", vals == [sp.sqrt(5), -sp.sqrt(5)], f"{vals}")
from krmhd.hermite import build_implicit_operator


def landau_root(Lam, z0):
    f = lambda zz: 1 - (1 + zz * 1j * mp.sqrt(mp.pi) * mp.exp(-zz * zz) * mp.erfc(-1j * zz)) / Lam
    return complex(mp.findroot(f, mp.mpc(z0)))


mp.mp.dps = 30
okG2 = True
kk = 2 * np.pi
for Lam, z0 in ((SQRT5, -0.4j), (-SQRT5, 1.24 - 0.83j)):
    zl = landau_root(Lam, z0)
    for M in (128, 256):
        Lo = np.array(build_implicit_operator(np.array([kk]), 1.0, 300.0, M, Lam, 6, "zero"))[0]
        ev = np.linalg.eigvals(Lo)
        zeta = ev[np.argmax(ev.real)] / (-1j * kk)
        d = min(abs(zeta - zl), abs(zeta + np.conj(zl)))          # mirror pair -conj(zeta)
        d2 = min(abs(zeta - np.sqrt(2) * zl), abs(zeta + np.sqrt(2) * np.conj(zl)))
        d3 = min(abs(zeta - zl / np.sqrt(2)), abs(zeta + np.conj(zl) / np.sqrt(2)))
        okG2 = okG2 and d < 2e-4 and min(d2, d3) > 0.1
        print(f"   Lam={Lam:+.4f}, M={M}, nu=300 (m/M)^6, k s = 2 pi: least-damped eigenvalue / (-i k s) = {zeta:.8f}; "
              f"Landau root zeta_L = {zl:.8f}; |difference| = {d:.1e}; distance to sqrt2 zeta_L, zeta_L/sqrt2: {d2:.3f}, {d3:.3f}")
C.check("G2 GANDALF's operator reproduces the Landau root of 1 - (1 + zeta Z)/Lam (u = v/v_th, F_M ~ e^-u^2), "
        "not a sqrt2-rescaled one", okG2)
Ng = 8
gr = make_grid(Ng)
zpk = np.zeros((Ng, Ng, Ng // 2 + 1), complex)
zpk[1, 1, 1] = 1.0                                   # single mode, k_z = +2 pi
st = make_state(gr, 4, SQRT5, zpk, np.zeros_like(zpk), np.zeros((Ng, Ng, Ng // 2 + 1, 5), complex))
rz = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0).z_plus)
ratio = rz[1, 1, 1] / zpk[1, 1, 1]
kz1 = float(np.asarray(gr.kz)[1])
C.check("G3 krmhd_rhs: d_t z+ = +i k_z z+ for z- = 0, so z+ ~ exp(i k_z (z + t)) propagates toward -z",
        abs(ratio - 1j * kz1) < 1e-12, f"d_t z+/z+ = {ratio:.6f}, i k_z = {1j * kz1:.6f}")
C.finish()
