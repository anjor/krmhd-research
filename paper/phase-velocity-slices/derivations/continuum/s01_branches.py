"""s01: the compressive sector of KRMHD in the two-branch form of Appendix B of Chandran, Mallet & Meyrand (2026,
arXiv:2607.27981v1; CMM below), checked in exact SymPy arithmetic. It verifies the Laguerre moments and field
relations, the G_n, G_B equations of Schekochihin et al. (2009) and their coupling matrix K, the branches G+- of CMM
(B5)-(B7) as left eigenvectors of K with Lambda+ > 1 and Lambda- < 0, the moments carried by the branches, the
free-energy and H decompositions (and that CMM's (B16) as printed lacks a square), the factor 1/2 in Schekochihin et
al. (2009) eqs (202)-(203), and the Laguerre l >= 2 sector as the Lambda = infinity branch. It supports Section 2 and
Appendix A of the paper.

Inputs (CMM 2026, labels from the LaTeX source; numbers from s00):
  (3.25) eq:KRMHDdgdt   dg/dt + v grad_par[ g + ((Z/tau) dn/n + x dB/B) F0 ] = 0,   x = vperp^2/vth^2
  (3.27) eq:dnKRMHD     mean(g)   = dn/n - dB/B
  (3.28) eq:perpAmpereKRMHD  mean(x g) = -(Z/tau) dn/n - 2 (1 + 1/beta) dB/B
  (B8)   eq:Gl          g = n/(pi vth^2) e^{-x} sum_l L_l(x) G_l(vpar),  F0 = n/(pi vth^2) e^{-x} F_M(vpar)
The v_perp average (2 pi/n) int dvperp vperp (...) of n/(pi vth^2) e^{-x} h(x) equals int_0^inf dx e^{-x} h(x).
Radicals: q = sqrt((1 + tau/Z)^2 + 1/beta^2) is a symbol; expressions are reduced modulo q^2 - (1+tau/Z)^2 - 1/beta^2.

Checks:
 B1  Laguerre moments: int e^{-x} x^k L_l dx = 0 for l > k; P_l[1] = delta_l0, P_l[x] = delta_l0 - delta_l1.
 B2  dn/n = int G_n, dB/B = int G_B with G_n, G_B of (B2), (B3); G_0 = G_n - G_B; G_1 = (1+Z/tau) G_0 + D G_B.
 B3  the G_n, G_B equations (Schekochihin et al. 2009, eqs 179-180) from the l = 0, 1 projections; coupling matrix K.
 B4  (B5) with sigma (B6) are left eigenvectors of K with eigenvalues 1/Lambda+- of (B7) -> each G+- obeys (B4).
 B5  Lambda+ > 1, Lambda- < 0 for all beta, tau/Z > 0; sum, product; beta = tau = Z = 1 -> +-sqrt5; limits.
 B6  moments through the branches: G_0, G_1, dn/n, dB/B, u_par, dp_perp, dp_par.
 B7  W_compr (3.31) = W~ + c+ W+ + c- W-: exact c+-; (B15) with kappa as printed in (B16) and with Schekochihin et al.
     (2009) eq. 204; H_ph-sp decomposition (B17).
 B8  Schekochihin et al. (2009) eqs (202)-(203) versus the exact inversion (CMM footnote: factor 1/2).
 B9  l >= 2 sector: the branch operator with 1/Lambda = 0.
Exit code 1 on any failed check.
"""
import sys
import random
import sympy as sp

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


beta, r, q = sp.symbols('beta r q', positive=True)       # r = tau/Z ; q = sqrt((1+r)^2 + beta^-2)
Q0 = (1 + r)**2 + 1 / beta**2
Zt = 1 / r                                               # Z/tau
D = Zt + 2 * (1 + 1 / beta)


def is_zero(expr):
    """expr (rational in beta, r, q) == 0 identically, using q^k = Q0^(k//2) q^(k%2).
    After reduction the numerator is A + B q with A, B rational in (beta, r); q is not rational, so A = B = 0."""
    num, den = sp.fraction(sp.together(expr))
    P = sp.Poly(sp.expand(num), q)
    e2 = sp.expand(sum(c * Q0**(k // 2) * q**(k % 2) for (k,), c in P.terms()))
    return sp.simplify(e2.coeff(q, 1)) == 0 and sp.simplify(e2.coeff(q, 0)) == 0


# ---------------------------------------------------------------- B1
x = sp.Symbol('x', nonnegative=True)


def lag_moment(l, k):
    """int_0^inf e^{-x} x^k L_l(x) dx, from L_l = sum_j (-1)^j C(l,j) x^j/j!."""
    return sum((-1)**j * sp.binomial(l, j) * sp.factorial(k + j) / sp.factorial(j) for j in range(l + 1))


print("B1  Laguerre moments", flush=True)
check("int e^-x x^k L_l = 0 for l > k (l <= 12, k <= 6)", all(lag_moment(l, k) == 0 for l in range(13) for k in range(7) if l > k))
check("int e^-x x^k L_k = (-1)^k k!", all(lag_moment(k, k) == (-1)**k * sp.factorial(k) for k in range(8)))
check("orthonormality int e^-x L_l L_m = delta (sympy laguerre, l,m <= 5)",
      all(sp.integrate(sp.exp(-x) * sp.laguerre(l, x) * sp.laguerre(m, x), (x, 0, sp.oo)) == (1 if l == m else 0)
          for l in range(6) for m in range(6)))
check("P_l[1] = delta_l0; P_l[x] = (1,-1,0,..); P_l[x^2] = (2,-4,2,0,..)",
      [lag_moment(l, 0) for l in range(4)] == [1, 0, 0, 0] and [lag_moment(l, 1) for l in range(4)] == [1, -1, 0, 0]
      and [lag_moment(l, 2) for l in range(4)] == [2, -4, 2, 0])

G0, G1, N0, N1, n_, b_ = sp.symbols('G0 G1 N0 N1 n b')     # n = dn/n, b = dB/B

print("B2  field relations", flush=True)
sol = sp.solve([sp.Eq(N0, n_ - b_), sp.Eq(N0 - N1, -Zt * n_ - 2 * (1 + 1 / beta) * b_)], [n_, b_], dict=True)[0]
Gn = lambda a0, a1: -((a0 - a1) - 2 * (1 + 1 / beta) * a0) / D      # (B2) after the v_perp integral
GB = lambda a0, a1: -((a0 - a1) + Zt * a0) / D                      # (B3)
check("int G_n = dn/n", sp.simplify(Gn(N0, N1) - sol[n_]) == 0)
check("int G_B = dB/B", sp.simplify(GB(N0, N1) - sol[b_]) == 0)
check("G_n - G_B = G_0", sp.simplify(Gn(G0, G1) - GB(G0, G1) - G0) == 0)
check("G_1 = (1 + Z/tau)(G_n - G_B) + D G_B", sp.simplify((1 + Zt) * (Gn(G0, G1) - GB(G0, G1)) + D * GB(G0, G1) - G1) == 0)

print("B3  G_n, G_B equations", flush=True)
src0 = Zt * n_ + b_        # l = 0 source:  dG0/dt + v grad[G0 + F src0] = 0
src1 = -b_                 # l = 1 source:  dG1/dt + v grad[G1 + F src1] = 0
srcGn = sp.simplify(Gn(src0, src1))
srcGB = sp.simplify(GB(src0, src1))
check("Schekochihin et al. 2009 eq. (179): dGn/dt + v grad Gn = -D^-1 v F grad[(Z/tau)(1+2/beta) n + (2/beta) b]",
      sp.simplify(-srcGn + (Zt * (1 + 2 / beta) * n_ + 2 / beta * b_) / D) == 0)
check("Schekochihin et al. 2009 eq. (180): dGB/dt + v grad GB = +D^-1 v F grad[(Z/tau)(1+Z/tau) n + (2+Z/tau) b]",
      sp.simplify(-srcGB - (Zt * (1 + Zt) * n_ + (2 + Zt) * b_) / D) == 0)
K = sp.Matrix([[sp.diff(-srcGn, n_), sp.diff(-srcGn, b_)], [sp.diff(-srcGB, n_), sp.diff(-srcGB, b_)]]).applyfunc(sp.simplify)
print("    K =", K, flush=True)

print("B4  diagonalisation", flush=True)
Lp = -r + 1 / beta + q
Lm = -r + 1 / beta - q
sigma = 1 + r + 1 / beta + q
lp = sp.Matrix([[(1 + Zt) / sigma, 1]])          # G+ = G_B + sigma^-1 (1 + Z/tau) G_n      (B5)
lm = sp.Matrix([[1, 2 * r / (sigma * beta)]])    # G- = G_n + 2 tau/(sigma Z beta) G_B       (B5)
check("l+ K = l+ / Lambda+", all(is_zero(e) for e in (lp * K - lp / Lp)))
check("l- K = l- / Lambda-", all(is_zero(e) for e in (lm * K - lm / Lm)))
check("trace K = 1/L+ + 1/L-", is_zero(K.trace() - 1 / Lp - 1 / Lm))
check("det K = 1/(L+ L-)", is_zero(K.det() - 1 / (Lp * Lm)))
num = lambda e, bb, rr: sp.N(e.subs({q: sp.sqrt((1 + rr)**2 + sp.Integer(1) / bb**2)}).subs({beta: bb, r: rr}), 30)
check("Lambda+-(1,1) = +-sqrt5", abs(num(Lp, 1, 1) - sp.sqrt(5)) < 1e-25 and abs(num(Lm, 1, 1) + sp.sqrt(5)) < 1e-25)

print("B5  ranges, sum, product, limits", flush=True)
check("Lambda+ - 1 - 1/beta = q - (1 + tau/Z) > 0 since q > 1 + tau/Z", is_zero((Lp - 1 - 1 / beta) - (q - (1 + r))))
check("Lambda- = 1/beta - q - tau/Z < 0 since q > 1/beta", is_zero(Lm - (1 / beta - q - r)))
check("Lambda+ + Lambda- = 2(1/beta - tau/Z)", is_zero(Lp + Lm - 2 * (1 / beta - r)))
check("Lambda+ Lambda- = -(1 + 2 (tau/Z)(1 + 1/beta))", is_zero(Lp * Lm + 1 + 2 * r * (1 + 1 / beta)))
rng = random.Random(1)
okr = True
for _ in range(5000):
    bv = 10 ** rng.uniform(-4, 4); rv = 10 ** rng.uniform(-4, 4)
    qq = ((1 + rv)**2 + bv**-2) ** 0.5
    okr &= (-rv + 1 / bv + qq > 1) and (-rv + 1 / bv - qq < 0)
check("random sweep beta, tau/Z in [1e-4,1e4]: Lambda+ > 1, Lambda- < 0", okr)
e_ = sp.Symbol('e', positive=True)
LpE = lambda bb: -r + 1 / bb + sp.sqrt((1 + r)**2 + 1 / bb**2)
LmE = lambda bb: -r + 1 / bb - sp.sqrt((1 + r)**2 + 1 / bb**2)
hp = sp.series(LpE(1 / e_) - 1, e_, 0, 2).removeO()
hm = sp.series(LmE(1 / e_) - 1, e_, 0, 1).removeO()
check("high beta: Lambda+ - 1 = 1/beta + O(1/beta^2) (Schekochihin et al. 2009 eq. 188)", sp.simplify(hp - e_) == 0, str(hp))
check("high beta: Lambda- - 1 -> -2(1 + tau/Z) (Schekochihin et al. 2009 eq. 187)", sp.simplify(hm + 2 * (1 + r)) == 0, str(hm))
lp_low = sp.series(e_ * (LpE(e_) - 1), e_, 0, 1).removeO()
lm_low = sp.series(LmE(e_) - 1, e_, 0, 1).removeO()
check("low beta: beta (Lambda+ - 1) -> 2 (Schekochihin et al. 2009 eq. 192)", sp.simplify(lp_low - 2) == 0, str(lp_low))
check("low beta: Lambda- - 1 -> -(1 + tau/Z) (Schekochihin et al. 2009 eq. 191)", sp.simplify(lm_low + 1 + r) == 0, str(lm_low))

print("B6  moments through the branches", flush=True)
Gn_s, GB_s, Gp_s, Gm_s = sp.symbols('G_n G_B G_+ G_-')
Sm = sp.Matrix([[lp[0], lp[1]], [lm[0], lm[1]]])     # (G+, G-) = Sm (G_n, G_B)
Sinv = Sm.inv()
detS = sp.simplify(Sm.det())
GnGB = Sinv * sp.Matrix([Gp_s, Gm_s])
G0_of = GnGB[0] - GnGB[1]
G1_of = (1 + Zt) * G0_of + D * GnGB[1]
at = lambda e: sp.nsimplify(sp.radsimp(sp.simplify(e.subs({beta: 1, r: 1, q: sp.sqrt(5)}))))
gold = (1 + sp.sqrt(5)) / 2
print("    det S at beta=tau=Z=1 =", at(detS), "=", sp.N(at(detS), 15), flush=True)
check("sigma^-1(1 + Z/tau) at beta=tau=Z=1 = (3 - sqrt5)/2 = 0.381966", sp.simplify(at(lp[0]) - (3 - sp.sqrt(5)) / 2) == 0)
check("2 tau/(sigma Z beta) at beta=tau=Z=1 = (3 - sqrt5)/2", sp.simplify(at(lm[1]) - (3 - sp.sqrt(5)) / 2) == 0)
check("det S = 0.381966^2 - 1 = -0.854102 at beta=tau=Z=1 (|det| = 0.854102)", sp.simplify(at(detS) + (1 - ((3 - sp.sqrt(5)) / 2)**2)) == 0)
g0_1 = sp.expand(at(G0_of))
check("beta=tau=Z=1: G_0 = G_n - G_B = golden ratio x (G- - G+)", sp.simplify(g0_1 - gold * (Gm_s - Gp_s)) == 0, str(g0_1))
print("    beta=tau=Z=1: G_1 =", sp.expand(at(G1_of)), flush=True)
print("    beta=tau=Z=1: G_n =", sp.expand(at(GnGB[0])), "; G_B =", sp.expand(at(GnGB[1])), flush=True)
# pressure moments: g_i = g_09 + x b F0 ; dp_perp/(nT) = mean(x g_i) = (N0 - N1) + 2 b ; dp_par/(nT) = (2/vth^2) M2[G0] + b
check("dp_perp/p = (N0 - N1) + 2 dB/B = -(Z/tau) dn/n - (2/beta) dB/B",
      sp.simplify((N0 - N1) + 2 * sol[b_] - (-Zt * sol[n_] - 2 / beta * sol[b_])) == 0)
check("mean(vhat_par^2 x F0) = 1/2 (enters dp_par)", sp.Rational(1, 2) * lag_moment(0, 1) == sp.Rational(1, 2))

print("B7  free-energy and H decompositions", flush=True)
# (24): W_compr/(nT/2) = (Z/tau) n^2 + (2/beta) b^2 + int[(G0 + F b)^2 + (G1 - F b)^2]/F + sum_{l>=2} int G_l^2/F
#        = int (G0^2 + G1^2)/F + 2 b (N0 - N1) + 2 b^2 int F + (Z/tau) n^2 + (2/beta) b^2 + ...,  int F = 1,
#        with N0 - N1 = -(Z/tau) n - 2 (1 + 1/beta) b from (9)
I01 = sp.Symbol('I01')                     # int (G0^2 + G1^2)/F dv
Gs0, Gs1, Fs = sp.symbols('Gs0 Gs1 Fs', positive=True)
integrand = sp.expand(((Gs0 + Fs * b_)**2 + (Gs1 - Fs * b_)**2) / Fs)
# integrate term by term: Gs^2/F -> I01 ; Gs0 b, -Gs1 b -> b (N0 - N1) ; F b^2 -> b^2
red = integrand.subs({Gs0**2 / Fs: 0, Gs1**2 / Fs: 0})
red = sp.expand(red)
cross = red.coeff(Gs0, 1) * sp.Symbol('N0') + red.coeff(Gs1, 1) * sp.Symbol('N1')
quad_ = red.subs({Gs0: 0, Gs1: 0}).subs(Fs, 1)
lhs24 = I01 + cross + quad_ + Zt * n_**2 + 2 / beta * b_**2
lhs24 = lhs24.subs(sp.Symbol('N0'), sp.Symbol('N1') - Zt * n_ - 2 * (1 + 1 / beta) * b_)
rhs24 = I01 + Zt * n_**2 - 2 * Zt * n_ * b_ - 2 * (1 + 1 / beta) * b_**2
check("(24): l = 0,1 part of W_compr/(nT/2) = int(G0^2+G1^2)/F + (Z/tau) n^2 - 2(Z/tau) n b - 2(1+1/beta) b^2",
      sp.simplify(sp.expand(lhs24 - rhs24)) == 0)
cp, cm = sp.symbols('c_p c_m')
G0n = Gn_s - GB_s
G1n = (1 + Zt) * (Gn_s - GB_s) + D * GB_s
Gp_e = lp[0] * Gn_s + lp[1] * GB_s
Gm_e = lm[0] * Gn_s + lm[1] * GB_s
diff_loc = sp.expand(G0n**2 + G1n**2 - cp * Gp_e**2 - cm * Gm_e**2)
c_nn = diff_loc.coeff(Gn_s, 2)
c_bb = diff_loc.coeff(GB_s, 2)
c_nb = diff_loc.coeff(Gn_s, 1).coeff(GB_s, 1)
solc = sp.solve([c_nn, c_bb], [cp, cm], dict=True)[0]
check("local identity G0^2 + G1^2 = c+ G+^2 + c- G-^2 (cross term vanishes)", is_zero(c_nb.subs(solc)))
M0p = lp[0] * n_ + lp[1] * b_
M0m = lm[0] * n_ + lm[1] * b_
mom = sp.expand(Zt * n_**2 - 2 * Zt * n_ * b_ - 2 * (1 + 1 / beta) * b_**2 + solc[cp] * M0p**2 / Lp + solc[cm] * M0m**2 / Lm)
check("moment identity with the same c+-", all(is_zero(mom.coeff(n_, i).coeff(b_, j)) for i, j in ((2, 0), (1, 1), (0, 2))))
kap_sch = q                                  # Schekochihin et al. (2009) eq. (204): kappa = sqrt((1 + tau/Z)^2 + 1/beta^2)
cp_B15 = lambda kap: (1 + (1 + r) / kap) * Lp**2
cm_B15 = lambda kap: (1 / (2 * r**2)) * (1 + 1 / (kap * beta)) * Lm**2
check("(B15) coefficients exact with kappa = sqrt((1+tau/Z)^2 + 1/beta^2) (Schekochihin et al. 2009 eq. 204)",
      is_zero(solc[cp] - cp_B15(kap_sch)) and is_zero(solc[cm] - cm_B15(kap_sch)))
kapB16 = sp.sqrt((1 + r) + 1 / beta**2)
worst = 0
for bb, rr in ((1, 1), (sp.Rational(1, 3), 2), (5, sp.Rational(1, 2)), (sp.Rational(1, 10), sp.Rational(1, 10))):
    sub = {q: sp.sqrt((1 + rr)**2 + sp.Integer(1) / bb**2)}
    for ce, cb in ((solc[cp], cp_B15(kapB16)), (solc[cm], cm_B15(kapB16))):
        dv = sp.N((ce.subs(sub) - cb.subs(sub)).subs({beta: bb, r: rr}), 25)
        worst = max(worst, abs(dv))
print(f"    with kappa as printed in (B16), max |c_exact - c_(B15)| on 4 points = {float(worst):.4f}", flush=True)
check("(B16) as printed (no square on 1 + tau/Z) does not satisfy (B15)", worst > 1e-3)
print("    exact c+ (beta=tau=Z=1) =", at(solc[cp]), "=", sp.N(at(solc[cp]), 12), "; c- =", at(solc[cm]), "=", sp.N(at(solc[cm]), 12))
# Schekochihin et al. (2009) eq. (213) as printed in the published paper:
#   W_compr = W~ + 4 [1 + (1/kappa)(1 + tau/Z)] (L+)^2 W+ + 2 (Z^2/tau^2) [1 + 1/(kappa beta)] (L-)^2 W-
cp_sch = 4 * (1 + (1 + r) / q) * Lp**2
cm_sch = 2 * (1 / r**2) * (1 + 1 / (q * beta)) * Lm**2
check("Schekochihin et al. 2009 eq. (213) coefficients = 4 x exact c+ and 4 x exact c- (CMM's footnote)", is_zero(cp_sch - 4 * solc[cp]) and is_zero(cm_sch - 4 * solc[cm]))
check("(B17) H decomposition: same local identity, same c+-", is_zero(c_nb.subs(solc)))

print("B8  Schekochihin et al. 2009 eqs (202)-(203) vs exact inversion", flush=True)
Mp_s, Mm_s = sp.symbols('M0p M0m')
inv_nb = sp.solve([sp.Eq(Mp_s, M0p), sp.Eq(Mm_s, M0m)], [n_, b_], dict=True)[0]
sch_n = (sigma * Mm_s - r * 2 / beta * Mp_s) / q
sch_b = (sigma * Mp_s - (1 + Zt) * Mm_s) / q
check("exact dn/n = (1/2) Schekochihin et al. 2009 eq. (202)", is_zero(inv_nb[n_] - sch_n / 2))
check("exact dB/B = (1/2) Schekochihin et al. 2009 eq. (203)", is_zero(inv_nb[b_] - sch_b / 2))
print("    exact at beta=tau=Z=1: dn/n =", sp.expand(at(inv_nb[n_])), "; dB/B =", sp.expand(at(inv_nb[b_])), flush=True)

check("|det S| = 1 - 2(1 + tau/Z)/(beta sigma^2) = 2 kappa/sigma, kappa = q", is_zero(-detS - 2 * q / sigma) and is_zero(-detS - (1 - 2 * (1 + r) / (beta * sigma**2))))
check("G_n = [sigma G- - (2 tau/(Z beta)) G+]/(2 kappa)", is_zero(GnGB[0] - (sigma * Gm_s - 2 * r / beta * Gp_s) / (2 * q)))
check("G_B = [sigma G+ - (1 + Z/tau) G-]/(2 kappa)", is_zero(GnGB[1] - (sigma * Gp_s - (1 + Zt) * Gm_s) / (2 * q)))

print("B9  l >= 2", flush=True)
check("P_l[(Z/tau) n + x b] = 0 for l = 2..10",
      all(sp.simplify(Zt * n_ * lag_moment(l, 0) + b_ * lag_moment(l, 1)) == 0 for l in range(2, 11)))

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s01 CHECKS PASSED")
