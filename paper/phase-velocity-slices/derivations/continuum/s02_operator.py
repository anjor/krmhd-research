"""s02: the branch operator L G = u G - (u F_M/Lambda) M0[G] and the free-energy product [f, g], checked in exact
SymPy arithmetic: self-adjointness of L^n in [.,.], the forms of P L and P L^{-1} = 1/u, the orthogonal split of
[G, G], the cases Lambda = 1 and 0 < Lambda < 1, the Hermite (ladder) matrix A of L and the weight matrix P0 with
P0 A symmetric, agreement of A with GANDALF's streaming matrix, and the equivalence of the norms [G, G] and <G, G>
for Lambda outside [0, 1]. It supports Section 3 and Appendix B of the paper.

u = v_par/v_th, F_M = exp(-u^2)/sqrt(pi).  Test functions G = q(u) F_M with q a polynomial with rational
coefficients; then every integral is a Gaussian moment, int u^k F_M du = (k-1)!!/2^(k/2) (k even), 0 (k odd).
  L G = u G - (u F_M / Lambda) M0[G]
  <f, g> = int f g / F_M du (weighted L^2),  [f, g] = <f, g> - M0[f] M0[g] / Lambda = <f, P g>,
  P g = g - (F_M / Lambda) M0[g]   (note M0[g] = <F_M, g>).
Checks:
 O1  [f, L^n g] = [L^n f, g], n = 0..6 (symbolic Lambda, random degree-6 polynomials).
 O2  [f, L g] = int u f g/F_M - (M0f M1g + M1f M0g)/Lambda.
 O3  P L G = u G - (u F_M/Lambda) M0 - (F_M/Lambda) M1;  P L^{-1} = 1/u on G with G(0) = 0 (no Lambda term).
 O4  [G, G] = int (G - M0 F_M)^2/F_M du + (1 - 1/Lambda) M0^2   (orthogonal split along F_M).
 O5  <F_M, F_M> = 1, <u F_M, u F_M> = 1/2 (the rank-one part of L is bounded).
 O6  Lambda = 1: L F_M = 0 and [F_M, g] = 0 for all g;  0 < Lambda < 1: [F_M, F_M] = 1 - 1/Lambda < 0.
 O7  Hermite matrix of L (zero closure is the truncation of the exact infinite matrix): A as stated; [.,.] = diag P0.
 O8  P0 A is symmetric.
 O9  A equals GANDALF's krmhd.hermite.compute_streaming_matrix (both closures).
 O10 norm equivalence min(1, 1-1/Lambda) <= [G,G]/<G,G> <= max(1, 1-1/Lambda) on random G.
Exit 1 on failure.
"""
import sys
import random
import numpy as np
import sympy as sp

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


u = sp.Symbol('u', real=True)
Lam = sp.Symbol('Lambda', nonzero=True)


def gmom(k):
    if k % 2:
        return sp.Integer(0)
    return sp.factorial2(k - 1) / 2**sp.Rational(k, 2) if k > 0 else sp.Integer(1)


def integ(p):
    """int p(u) F_M du for a polynomial p."""
    P = sp.Poly(sp.expand(p), u)
    return sp.expand(sum(c * gmom(m[0]) for m, c in zip(P.monoms(), P.coeffs())))


# functions are represented by their polynomial factor q (G = q F_M)
M = lambda k, q: integ(u**k * q)
Lop = lambda q: sp.expand(u * q - u * M(0, q) / Lam)
Pop = lambda q: sp.expand(q - M(0, q) / Lam)
ipw = lambda p, q: integ(p * q)                         # <pF, qF>
br = lambda p, q: sp.expand(ipw(p, q) - M(0, p) * M(0, q) / Lam)


def Ln(q, n):
    for _ in range(n):
        q = Lop(q)
    return q


rng = random.Random(20261013)
rp = lambda deg: sum(sp.Rational(rng.randint(-9, 9), rng.randint(1, 6)) * u**k for k in range(deg + 1))
F, G = rp(6), rp(6)

print("O1  self-adjointness of L^n in [.,.]")
res = [sp.simplify(br(F, Ln(G, n)) - br(Ln(F, n), G)) for n in range(7)]
check("[F, L^n G] - [L^n F, G] = 0, n = 0..6", all(x == 0 for x in res), str(res))
print("O2  bilinear form of L")
check("[F, L G] closed form", sp.simplify(br(F, Lop(G)) - (integ(u * F * G) - (M(0, F) * M(1, G) + M(1, F) * M(0, G)) / Lam)) == 0)
check("M0[L G] = M1[G] (since M1[F_M] = 0)", sp.simplify(M(0, Lop(G)) - M(1, G)) == 0)
print("O3  P L and P L^{-1}")
check("P L G = u G - (u F_M/Lambda) M0 - (F_M/Lambda) M1", sp.simplify(Pop(Lop(G)) - (u * G - u * M(0, G) / Lam - M(1, G) / Lam)) == 0)
q0 = sp.expand(G - G.subs(u, 0))           # G(0) = 0
qv = sp.expand(sp.cancel(q0 / u))          # G/u (polynomial)
Nn = integ(qv)                             # N = int G/u du
Linv = sp.expand(qv + Nn / (Lam - 1))      # L^{-1} G = G/u + F_M N/(Lambda - 1)
check("L (G/u + F_M N/(Lambda-1)) = G for G(0) = 0", sp.simplify(Lop(Linv) - q0) == 0)
check("P L^{-1} G = G/u exactly (no Lambda)", sp.simplify(Pop(Linv) - qv) == 0)
print("O4  orthogonal split")
check("[G,G] = int (G - M0 F_M)^2/F_M + (1 - 1/Lambda) M0^2",
      sp.simplify(br(G, G) - (ipw(G - M(0, G), G - M(0, G)) + (1 - 1 / Lam) * M(0, G)**2)) == 0)
check("<G - M0 F_M, F_M> = 0", sp.simplify(ipw(G - M(0, G), sp.Integer(1))) == 0)
print("O5  norms")
check("<F_M, F_M> = 1", ipw(sp.Integer(1), sp.Integer(1)) == 1)
check("<u F_M, u F_M> = 1/2", ipw(u, u) == sp.Rational(1, 2))
print("O6  Lambda = 1 and 0 < Lambda < 1")
check("Lambda = 1: L F_M = 0", sp.simplify(Lop(sp.Integer(1)).subs(Lam, 1)) == 0)
check("Lambda = 1: [F_M, G] = 0 for the random G", sp.simplify(br(sp.Integer(1), G).subs(Lam, 1)) == 0)
check("[F_M, F_M] = 1 - 1/Lambda", sp.simplify(br(sp.Integer(1), sp.Integer(1)) - (1 - 1 / Lam)) == 0)

print("O7  Hermite matrix of L and of [.,.]")
Mx = 10
H = [sp.hermite(m, u) for m in range(Mx + 3)]
nrm = [sp.sqrt(2**m * sp.factorial(m)) for m in range(Mx + 3)]
basis = [H[m] / nrm[m] for m in range(Mx + 2)]          # e_m = H_m F_M / sqrt(2^m m!)  (polynomial factor)
coef = lambda q, m: sp.simplify(integ(H[m] * q) / nrm[m])  # g_m = int H_m G du / sqrt(2^m m!)
Aex = sp.zeros(Mx + 1, Mx + 1)
for n in range(Mx + 1):
    Le = Lop(basis[n])
    for m in range(Mx + 1):
        Aex[m, n] = coef(Le, m)
Afor = sp.zeros(Mx + 1, Mx + 1)
for m in range(Mx):
    Afor[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2))
    Afor[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
Afor[1, 0] = (1 - 1 / Lam) / sp.sqrt(2)
check("exact A_mn = ladder formula (m, n <= 10)", (Aex - Afor).applyfunc(sp.simplify) == sp.zeros(Mx + 1, Mx + 1))
# L e_M has a component on e_{M+1} only through A[M+1, M]: zero closure = drop it
check("L e_10 has e_11 coefficient sqrt(11/2) (dropped by the zero closure)", sp.simplify(coef(Lop(basis[Mx]), Mx + 1) - sp.sqrt(sp.Rational(11, 2))) == 0)
P0ex = sp.Matrix(Mx + 1, Mx + 1, lambda m, n: sp.simplify(br(basis[m], basis[n])))
P0for = sp.diag(*([1 - 1 / Lam] + [1] * Mx))
check("[e_m, e_n] = diag(1 - 1/Lambda, 1, 1, ...)", (P0ex - P0for).applyfunc(sp.simplify) == sp.zeros(Mx + 1, Mx + 1))
print("O8  P0 A symmetric")
check("P0 A - (P0 A)^T = 0", (P0for * Afor - (P0for * Afor).T).applyfunc(sp.simplify) == sp.zeros(Mx + 1, Mx + 1))

print("O9  comparison with GANDALF (krmhd.hermite.compute_streaming_matrix)")
try:
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cc_common import ladder
    from krmhd.hermite import compute_streaming_matrix
    import krmhd
    worst = 0.0
    for Mv in (2, 8, 64, 128):
        for Lv in (np.sqrt(5.0), -np.sqrt(5.0), 3.7, 1e6):
            for clo in ("zero", "symmetric"):
                Tg = np.asarray(compute_streaming_matrix(Mv, float(Lv), clo))
                worst = max(worst, np.abs(Tg - ladder(Mv, Lv, clo)).max())
    check(f"krmhd {krmhd.__version__} streaming matrix == A (max abs diff {worst:.1e})", worst < 1e-15)
except Exception as e:  # pragma: no cover
    check(f"krmhd import ({e})", False)

print("O10 norm equivalence")
okn = True
for Lv in (np.sqrt(5.0), -np.sqrt(5.0), 1.0001, -1e-3, 40.0):
    lo, hi = min(1, 1 - 1 / Lv), max(1, 1 - 1 / Lv)
    for _ in range(200):
        Gr = rp(5)
        b = float(br(Gr, Gr).subs(Lam, Lv)); w = float(ipw(Gr, Gr))
        okn &= (lo * w - 1e-12 <= b <= hi * w + 1e-12)
check("min(1,1-1/Lambda) <= [G,G]/<G,G> <= max(1,1-1/Lambda) (1000 random G, 5 Lambda)", okn)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s02 CHECKS PASSED")
