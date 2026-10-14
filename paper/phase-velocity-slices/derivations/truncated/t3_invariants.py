"""t3_invariants: checks the conserved quadratic forms of the truncated hierarchy and their continuum limits. The
forms with a k-independent matrix K that are conserved for all z+- are exactly Q_phi = g^T Pi phi(A) g, an
(M+1)-dimensional family; krmhd_rhs conserves them, the odd-M helicity Pi A^-1 and the even-M principal-value form,
but not the controls; on the retained Hermite span CMM's W, Gamma and H are Q_1/2, Q_u/2 and Q_{1/u}/2; the Galerkin
matrix of P(1/v) is Pi A^-1 at odd M and differs at even M; and the even-M principal-value helicity converges only
like M^(-1/2). Supports Section 7 (conserved forms, helicity at odd and even M), Section 8 (conservation rates) and
Appendix C of the paper.

A  {K = K^T : K A symmetric} has dimension M+1 and equals span{Pi A^n, n = 0..M} = {W diag(phi_j) W^T}:
   every conserved quadratic form is Q_phi = g^T Pi phi(A) g = sum_j phi(u_j) |a_j|^2 / n_j.
B  krmhd_rhs (float64, strong brackets) conserves every g^T Pi A^n g (n = 0..M), H = g^T Pi A^-1 g (odd M),
   the principal-value form g^T Pi A^D g (even M; A^D the Drazin inverse), and Im H. Controls: the Galerkin
   projection of 1/v at even M and the zero-closure weight under the copy closure (= krmhd.diagnostics.hermite_free_energy)
   are not conserved.
C  (SymPy, symbolic Lam) for G in the span of the first M+1 Hermite functions (zero closure):
   [G,G] = g^T Pi g, [G,LG] = g^T Pi A g, [G,L^2 G] = g^T Pi A^2 g + (M+1) g_M^2 / 2;
   CMM (v_th = 1, n T = 1, no volume average): W^+- (B13, B31) = Q_1/2, Gamma^+- (B25, B32) = Q_u/2,
   H^+- (B18) = Q_{1/u}/2.
D  Galerkin matrix B_M = [P int psi_m psi_n / v dv]: odd M: B_M = J^-1 = Pi A^-1 (any Lam);
   even M: B_M = J^D + lambda_0 (e d^T + d e^T), e_m = p_m(0), d_m = p_m'(0), lambda_0 the Gauss weight of the node 0;
   lambda_0 = [pi/(2M+3)]^(1/2) (1 + O(M^-2)).
E  Convergence for a non-band-limited G = F_M(v - a): W, Gamma (all M) and H (odd M) converge spectrally;
   the even-M principal-value H converges like M^(-1/2), its error being -2 lambda_0 (e.g)(d.g) at Lam = inf, and
   5% to 18% of H at M = 64 and 128.
"""
from __future__ import annotations

from math import lgamma

import numpy as np
import sympy as sp
from scipy.special import roots_hermite, dawsn

from tcore import (Checker, section, ladder, weight, slices, hermite_functions, orthonormal_polys, make_grid,
                   random_field, make_state, qform, dqform, SQRT5, jnp)
from krmhd.timestepping import krmhd_rhs
from krmhd.diagnostics import hermite_free_energy

C = Checker("t3_invariants")


def drazin(A: np.ndarray) -> np.ndarray:
    """Drazin (group) inverse of a diagonalisable A with real spectrum: invert the nonzero eigenvalues only."""
    u, V = np.linalg.eig(A)
    u = u.real; V = V.real
    inv = np.where(np.abs(u) > 1e-9, 1.0 / np.where(np.abs(u) > 1e-9, u, 1.0), 0.0)
    return (V * inv[None, :]) @ np.linalg.inv(V)


section("A. The space of conserved quadratic forms")
okA = True
for M in (5, 6, 9, 10, 13):
    for Lam in (SQRT5, -SQRT5, np.inf):
        for clo in ("zero", "symmetric"):
            A = ladder(M, Lam, clo)
            Pi = np.diag(weight(M, Lam, clo))
            n1 = M + 1
            idx = [(i, j) for i in range(n1) for j in range(i, n1)]
            cols = []
            for (i, j) in idx:
                E = np.zeros((n1, n1)); E[i, j] = 1.0; E[j, i] = 1.0
                Cm = E @ A - (E @ A).T
                cols.append(Cm[np.triu_indices(n1, 1)])
            sv = np.linalg.svd(np.array(cols).T, compute_uv=False)
            nullity = len(idx) - int((sv > 1e-10 * sv.max()).sum())
            rho = np.abs(np.linalg.eigvals(A)).max()
            basis = np.array([(Pi @ np.linalg.matrix_power(A / rho, k))[np.triu_indices(n1)] for k in range(n1)]).T
            rk = np.linalg.matrix_rank(basis, tol=1e-12 * np.linalg.norm(basis, 2))
            u, R, W, nn = slices(M, Lam, clo)
            # each slice projector W e_j e_j^T W^T / n_j lies in span{Pi A^k}
            Qb, _ = np.linalg.qr(basis)
            res = 0.0
            for j in range(n1):
                Kj = np.outer(W[:, j], W[:, j]) / nn[j]
                vj = Kj[np.triu_indices(n1)]
                res = max(res, np.linalg.norm(vj - Qb @ (Qb.T @ vj)) / np.linalg.norm(vj))
            okA = okA and nullity == n1 and rk == n1 and res < 1e-8
    print(f"   M={M:2d}: dim{{K: K A sym}} = M+1, rank span{{Pi A^k}} = M+1, slice projectors in span (worst residual {res:.1e})")
C.check("A1 conserved quadratic forms = span{Pi A^k} = {W diag(phi) W^T}, dimension M+1 (M = 5..13, both closures)", okA)

section("B. Conservation under krmhd_rhs (float64; brackets dominate streaming)")
N = 16
grid = make_grid(N)
okB = True
okB_ctrl = True
for M in (9, 10):
    for Lam in (SQRT5, -SQRT5):
        for clo in ("zero", "symmetric"):
            rng = np.random.default_rng(5 * M + (3 if clo == "zero" else 4) + (1 if Lam > 0 else 2))
            zp = random_field(grid, rng, amp=300.0)
            zm = random_field(grid, rng, amp=300.0)
            g = random_field(grid, rng, slope=-1.0, extra_shape=(M + 1,))
            st = make_state(grid, M, Lam, zp, zm, g, beta_i=1.0)
            gdot = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure=clo).g)
            A = ladder(M, Lam, clo)
            Pi = np.diag(weight(M, Lam, clo))
            worst = 0.0
            for k in range(M + 1):
                v, sc = dqform(grid, g, gdot, Pi @ np.linalg.matrix_power(A, k))
                worst = max(worst, abs(v) / sc)
            if M % 2:
                v, sc = dqform(grid, g, gdot, Pi @ np.linalg.inv(A))
                hname = "Pi A^-1"
            else:
                v, sc = dqform(grid, g, gdot, Pi @ drazin(A))
                hname = "Pi A^D (PV)"
            hres = abs(v) / sc
            ctrl = ""
            ok_ctrl = True
            if M % 2 == 0 and clo == "zero":
                # Galerkin projection of P(1/v) by exact Gauss-Hermite quadrature (even node count, no node at 0)
                xq, wq = roots_hermite(2 * M + 2)
                P = orthonormal_polys(M, xq)
                P0v = orthonormal_polys(M, np.array([0.0]))[:, 0]
                B = ((P[:, None, :] * P[None, :, :] - (P0v[:, None] * P0v[None, :])[..., None]) / xq[None, None, :]
                     ) @ (wq / np.sqrt(np.pi))
                v2, sc2 = dqform(grid, g, gdot, B)
                ctrl += f"; control Galerkin(1/v) rel d/dt = {abs(v2) / sc2:.1e}"
                ok_ctrl = ok_ctrl and abs(v2) / sc2 > 1e-6
            if clo == "symmetric":
                p_zero = np.diag(weight(M, Lam, "zero"))
                v3, sc3 = dqform(grid, g, gdot, p_zero)
                # tie this weight to krmhd.diagnostics.hermite_free_energy
                hf = hermite_free_energy(st)
                q_zero = qform(grid, g, p_zero)
                ctrl += (f"; hermite_free_energy == g^T P_Lam g: {abs(hf - q_zero) / q_zero:.1e}; its rel d/dt under the copy "
                         f"closure = {abs(v3) / sc3:.1e}")
                ok_ctrl = ok_ctrl and abs(v3) / sc3 > 1e-6 and abs(hf - q_zero) / q_zero < 1e-12
            okB = okB and worst < 1e-14 and hres < 1e-14
            okB_ctrl = okB_ctrl and ok_ctrl
            print(f"   M={M:2d} Lam={Lam:+.3f} {clo:9s}: max_k rel d/dt(g^T Pi A^k g) = {worst:.1e}; {hname}: {hres:.1e}{ctrl}")
C.check("B1 all g^T Pi A^k g (k = 0..M) and H (Pi A^-1 odd M, Pi A^D even M) are conserved by krmhd_rhs", okB)
C.check("B2 controls fail: Galerkin 1/v at even M; zero-closure weight (hermite_free_energy) under the copy closure", okB_ctrl)
# Non-quadratic Casimirs of a slice (conserved by the continuum passive-scalar equation) are not conserved by the
# Fourier-Galerkin (dealiased pseudo-spectral) truncation; the quadratic one is.
from krmhd.spectral import rfftn_inverse
from tcore import slices as _slices
okB3 = True
for M, Lam in ((9, SQRT5), (10, -SQRT5)):
    rng = np.random.default_rng(99 + M)
    zp = random_field(grid, rng, amp=300.0)
    zm = random_field(grid, rng, amp=300.0)
    g = random_field(grid, rng, slope=-1.0, extra_shape=(M + 1,))
    st = make_state(grid, M, Lam, zp, zm, g, beta_i=1.0)
    gdot = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0).g)
    u, R, W, nn = _slices(M, Lam)
    worst2, best4 = 0.0, np.inf
    for j in range(M + 1):
        a = np.asarray(rfftn_inverse(jnp.asarray(g @ W[:, j]), N, N, N))
        ad = np.asarray(rfftn_inverse(jnp.asarray(gdot @ W[:, j]), N, N, N))
        q2 = abs(2 * np.sum(a * ad)) / (2 * np.sum(np.abs(a) * np.abs(ad)))
        q4 = abs(4 * np.sum(a ** 3 * ad)) / (4 * np.sum(np.abs(a) ** 3 * np.abs(ad)))
        worst2 = max(worst2, q2)
        best4 = min(best4, q4)
    okB3 = okB3 and worst2 < 1e-13 and best4 > 1e-4
    print(f"   M={M} Lam={Lam:+.3f}: max_j rel d/dt int a_j^2 = {worst2:.1e}; min_j rel d/dt int a_j^4 = {best4:.1e}")
C.check("B3 per slice, int a_j^2 is conserved by krmhd_rhs but int a_j^4 is not (only quadratic Casimirs survive the "
        "Fourier-Galerkin truncation)", okB3)

section("C. SymPy: continuum forms for G in the Hermite span, and CMM's normalisation")
v = sp.symbols("v", real=True)
Ls = sp.symbols("Lambda", nonzero=True)
okC = True
for M in (3, 4):
    gs = sp.symbols(f"g0:{M + 1}", real=True)
    FM = sp.exp(-v ** 2) / sp.sqrt(sp.pi)
    G = sum(gs[m] * sp.hermite(m, v) * FM / sp.sqrt(2 ** m * sp.factorial(m)) for m in range(M + 1))
    h = sp.expand(sp.simplify(G / FM))          # G/F_M, a polynomial

    def vint(poly):
        """int poly(v) F_M dv for a polynomial in v."""
        poly = sp.Poly(sp.expand(poly), v)
        out = 0
        for (k,), c in poly.terms():
            if k % 2 == 0:
                out += c * sp.gamma(sp.Rational(k + 1, 2)) / sp.sqrt(sp.pi)
        return sp.simplify(out)

    M0 = vint(h)                                # int G dv
    M1 = vint(v * h)
    GG = vint(h * h) - M0 ** 2 / Ls             # [G, G]
    # L G = v G - v F_M M0 / Lam  ->  (L G)/F_M = v h - v M0/Lam
    Lh = v * h - v * M0 / Ls
    M0L = vint(Lh)
    GLG = vint(h * Lh) - M0 * M0L / Ls          # [G, L G]
    L2h = v * Lh - v * M0L / Ls
    GL2G = vint(h * L2h) - M0 * vint(L2h) / Ls  # [G, L^2 G]
    A = sp.zeros(M + 1, M + 1)
    for m in range(M):
        A[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); A[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
    A[1, 0] = (1 - 1 / Ls) / sp.sqrt(2)
    Pi = sp.eye(M + 1); Pi[0, 0] = 1 - 1 / Ls
    gv = sp.Matrix(gs)
    d1 = sp.simplify(GG - (gv.T * Pi * gv)[0])
    d2 = sp.simplify(GLG - (gv.T * Pi * A * gv)[0])
    d3 = sp.simplify(GL2G - (gv.T * Pi * A * A * gv)[0] - sp.Rational(M + 1, 2) * gs[M] ** 2)
    # CMM B25: Gamma = int G^2 v/(2 F_M) - M0 M1/Lam; B32: (1/sqrt2)(sum sqrt(m+1) G_m G_{m+1} - G_0 G_1/Lam)
    GamB25 = vint(v * h * h) / 2 - M0 * M1 / Ls
    GamB32 = (sum(sp.sqrt(m + 1) * gs[m] * gs[m + 1] for m in range(M)) - gs[0] * gs[1] / Ls) / sp.sqrt(2)
    d4 = sp.simplify(GamB25 - GLG / 2)
    d5 = sp.simplify(GamB32 - GLG / 2)
    # B31: W = (1/2)[sum G_m^2 - G_0^2/Lam]
    d6 = sp.simplify(sp.Rational(1, 2) * (sum(gs[m] ** 2 for m in range(M + 1)) - gs[0] ** 2 / Ls) - GG / 2)
    print(f"   M={M}: [G,G]-g^T Pi g = {d1}; [G,LG]-g^T Pi A g = {d2}; [G,L^2G]-g^T Pi A^2 g-(M+1)g_M^2/2 = {d3}; "
          f"B25-Q_u/2 = {d4}; B32-Q_u/2 = {d5}; B31-Q_1/2 = {d6}")
    okC = okC and all(d == 0 for d in (d1, d2, d3, d4, d5, d6))
C.check("C1 [G,G], [G,LG] exact on the span; [G,L^2G] misses (M+1)g_M^2/2; CMM W = Q_1/2, Gamma = Q_u/2", okC)

section("D. Phase-space helicity: Galerkin matrix of P(1/v) (SymPy exact)")
okD = True
for M in range(2, 8):
    n1 = M + 1
    p = [sp.hermite(m, v) / sp.sqrt(2 ** m * sp.factorial(m)) for m in range(n1)]
    B = sp.zeros(n1, n1)
    for m in range(n1):
        for nn_ in range(n1):
            if (m + nn_) % 2 == 1:
                q = sp.cancel((p[m] * p[nn_]) / v)          # a polynomial since one factor is odd
                poly = sp.Poly(sp.expand(q), v)
                B[m, nn_] = sum(c * sp.gamma(sp.Rational(k + 1, 2)) / sp.sqrt(sp.pi)
                                for (k,), c in poly.terms() if k % 2 == 0)
    J = sp.zeros(n1, n1)
    for m in range(M):
        J[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); J[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
    if M % 2 == 1:
        A = J.copy(); A[1, 0] = (1 - 1 / Ls) / sp.sqrt(2)
        Pi = sp.eye(n1); Pi[0, 0] = 1 - 1 / Ls
        ok1 = sp.simplify(B - J.inv()) == sp.zeros(n1, n1)
        ok2 = sp.simplify(Pi * A.inv() - J.inv()) == sp.zeros(n1, n1)
        okD = okD and ok1 and ok2
        print(f"   M={M} (odd): B_M == J^-1: {ok1}; Pi A^-1 == J^-1 for symbolic Lam: {ok2}")
    else:
        # Gauss weight of the node x = 0 of H_{M+1}: lambda_0 = 1 / sum_m p_m(0)^2
        e = sp.Matrix([pm.subs(v, 0) for pm in p])
        d = sp.Matrix([sp.diff(pm, v).subs(v, 0) for pm in p])
        lam0 = 1 / sum(ei ** 2 for ei in e)
        JD = J.pinv()       # symmetric J: Moore-Penrose = Drazin inverse
        okk = sp.simplify(B - JD - lam0 * (e * d.T + d * e.T)) == sp.zeros(n1, n1)
        okD = okD and okk
        print(f"   M={M} (even): B_M == J^D + lambda_0 (e d^T + d e^T): {okk}  (lambda_0 = {sp.nsimplify(lam0)})")
C.check("D1 odd M: Galerkin(P 1/v) = J^-1 = Pi A^-1 (any Lam); even M: = J^D + lambda_0 (e d^T + d e^T)", okD)
okD2 = True
for M in (33, 65, 129):
    xq, wq = roots_hermite(2 * M + 2)
    P = orthonormal_polys(M, xq)
    P0v = orthonormal_polys(M, np.array([0.0]))[:, 0]
    B = ((P[:, None, :] * P[None, :, :] - (P0v[:, None] * P0v[None, :])[..., None]) / xq[None, None, :]) @ (wq / np.sqrt(np.pi))
    for Lam in (SQRT5, -SQRT5):
        K = np.diag(weight(M, Lam)) @ np.linalg.inv(ladder(M, Lam))
        err = np.abs(B - K).max() / np.abs(B).max()
        okD2 = okD2 and err < 1e-11
    print(f"   M={M}: max|B_M - Pi A^-1| / max|B_M| = {err:.1e} (Lam = +-sqrt5)")
C.check("D2 odd M = 33, 65, 129: Pi A^-1 equals the Galerkin projection of P(1/v) to round-off", okD2)
# size of the node term at even M: lambda_0 = lambda^inf_{M/2} ~ [pi/(2M+3)]^(1/2) (weights normalised to sum 1)
devs = []
for M in (2, 4, 10, 16, 64, 128, 256, 1024):
    xg, wg = roots_hermite(M + 1)
    j0 = int(np.argmin(np.abs(xg)))
    devs.append((M, wg[j0] / np.sqrt(np.pi) / np.sqrt(np.pi / (2 * M + 3)) - 1))
print("   lambda^inf_(M/2) / [pi/(2M+3)]^(1/2) - 1 at M = " + ", ".join(f"{M}: {d:+.2e}" for M, d in devs))
C.check("D3 even M: the weight of the node u = 0 is lambda^inf_(M/2) = [pi/(2M+3)]^(1/2) (1 + O(M^-2))",
        all(abs(d) * M ** 2 < 0.1 for M, d in devs) and abs(devs[-1][1]) < 1e-7,
        f"max |rel. dev.| M^2 = {max(abs(d) * M ** 2 for M, d in devs):.3f}")

section("E. Convergence to the continuum for G = F_M(v - a), a = 0.5 (all Hermite coefficients nonzero)")
a = 0.5
Wc = {Lam: np.exp(2 * a * a) - (0 if np.isinf(Lam) else 1 / Lam) for Lam in (SQRT5, -SQRT5, np.inf)}
Qc = {Lam: 2 * a * np.exp(2 * a * a) - (0 if np.isinf(Lam) else 2 * a / Lam) for Lam in (SQRT5, -SQRT5, np.inf)}
Hc = 2 * np.exp(2 * a * a) * dawsn(2 * a)
print(f"   continuum: W = e^(2a^2) - 1/Lam, Q_u = 2a(e^(2a^2) - 1/Lam), H = P int G^2/(v F_M) = 2 e^(2a^2) D(2a) = {Hc:.12f}")
okE = True
rowsE = []
for Lam in (SQRT5, -SQRT5, np.inf):
    print(f"   Lam = {Lam:+.4f}")
    for M in (8, 9, 16, 17, 32, 33, 64, 65, 128, 129, 256, 257):
        mm = np.arange(M + 1)
        gm = np.exp(mm * np.log(np.sqrt(2) * a) - 0.5 * np.array([lgamma(k + 1) for k in mm]))
        A = ladder(M, Lam)
        Pi = np.diag(weight(M, Lam))
        eW = gm @ Pi @ gm - Wc[Lam]
        eQ = gm @ Pi @ A @ gm - Qc[Lam]
        if M % 2:
            eH = gm @ Pi @ np.linalg.solve(A, gm) - Hc
            lab = "H (odd, Pi A^-1)"
            extra = ""
        else:
            eH = gm @ Pi @ drazin(A) @ gm - Hc
            lab = "H (even, PV)"
            extra = ""
            if np.isinf(Lam):
                x, w = roots_hermite(M + 1)
                lam0 = (w / np.sqrt(np.pi))[np.argmin(np.abs(x))]
                e = orthonormal_polys(M, np.array([0.0]))[:, 0]
                # p_m'(0) = sqrt(2m) p_{m-1}(0)
                dd = np.array([np.sqrt(2 * m) * e[m - 1] if m >= 1 else 0.0 for m in range(M + 1)])
                pred = -2 * lam0 * (e @ gm) * (dd @ gm)
                extra = f"; predicted -2 lambda_0 (e.g)(d.g) = {pred:+.6e} (ratio {eH / pred:.10f})"
                if M >= 16:          # at M = 8 the Hermite tail beyond M still contributes (ratio 1.0002)
                    okE = okE and abs(eH / pred - 1) < 1e-8
        rowsE.append((Lam, M, eH))
        print(f"     M={M:3d}: W err {eW:+.1e}; Q_u err {eQ:+.1e}; {lab} err {eH:+.3e}{extra}")
        if M >= 32:
            okE = okE and abs(eW) < 1e-13 and abs(eQ) < 1e-13
        if M % 2 and M >= 33:
            okE = okE and abs(eH) < 1e-13
# even-M principal-value error ~ M^(-1/2)
for Lam in (SQRT5, -SQRT5, np.inf):
    ev = np.array([(M, e) for (L_, M, e) in rowsE if L_ == Lam and M % 2 == 0 and M >= 32])
    slope = np.polyfit(np.log(ev[:, 0]), np.log(np.abs(ev[:, 1])), 1)[0]
    print(f"   Lam={Lam:+.4f}: even-M PV error ~ M^({slope:.3f}) for M = 32..256; error x sqrt(M) = "
          + ", ".join(f"{abs(e) * np.sqrt(M):.3f}" for M, e in ev))
    okE = okE and abs(slope + 0.5) < 0.1
C.check("E1 W, Q_u (all M) and H (odd M) converge spectrally; even-M PV H converges ~ M^-1/2 (local slopes within 0.1), "
        "with exactly the predicted node-zero error at Lam = inf (M >= 16)", okE)
# size of the even-M error relative to the helicity at M = 64 and 128 (quoted in the paper as 5% to 18%)
relE = [(Lam, M, abs(e) / Hc) for (Lam, M, e) in rowsE if M in (64, 128)]
print("   |even-M PV error|/H at M = 64, 128: " + "; ".join(f"Lam={L_:+.4f} M={M}: {100 * r:.1f}%" for L_, M, r in relE))
C.check("E2 at M = 64 and 128 the even-M PV helicity error is between 5% and 18% of H (Lam = sqrt5, -sqrt5, inf)",
        all(0.05 <= r <= 0.18 for _, _, r in relE),
        f"min {100 * min(r for *_, r in relE):.1f}%, max {100 * max(r for *_, r in relE):.1f}%")
C.finish()
