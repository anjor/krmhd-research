"""s07: the invariant family Q_phi = [G, phi(L) G] in exact SymPy arithmetic: the integrands of W (B13), Gamma
(B25) and H (B18) of Chandran, Mallet & Meyrand (2026; CMM below) in dimensional v_par and their Hermite forms
(B31)-(B32), closed forms of [G, L^n G] from a generating function, conservation of every Q_{u^n} (n <= 8) from the
moment hierarchy alone, the local field-aligned flux law d_t w + div(w u_E + gamma bhat) = 0, the u = 0 slice, and
the advection of products of slices at equal u. It supports Section 4 of the paper.

u = v_par/v_th, F_M(u) = e^{-u^2}/sqrt(pi); c_k = int u^k F_M du; s = v_th/v_A, v_A = 1; d_t = d_t + {Phi, .};
grad_par = d_z + {Psi, .}.  Branch: d_t G = -s grad_par(L G).
Checks:
 I1  change of variable: [G,G] and the CMM integrands in dimensional v_par (F_M of CMM below B4) versus u:
     CMM (B13) integrand = [G,G];  CMM (B25) integrand gamma = (v_th/2)[G, L G];  CMM (B18) integrand = (1/(2 v_th)) P int G^2/(u F_M).
 I2  Hermite forms: [G,G] = sum g_m^2 - g_0^2/Lambda (CMM B31); (1/2)[G,LG] = (1/sqrt2)(sum sqrt(m+1) g_m g_{m+1} - g_0 g_1/Lambda)
     (CMM B32 / v_th).
 I3  generating function: Q_{u^n} = int u^n G^2/F_M - (1/Lambda) sum_{a+b+j=n} e_j M_a M_b, 1/eps(z) = sum_j e_j z^-j; n <= 8;
     explicit Q_{u^2}, Q_{u^3}.
 I4  moment hierarchy d_t M_k = -s grad_par M_{k+1} + s (c_{k+1}/Lambda) grad_par M_0 (CMM B23, B24 for k = 0, 1) and
     d/dt int d^3r Q_{u^n} = 0 for n <= 8 using only the antisymmetry of [a,b] = int d^3r M_a grad_par M_b.
 I5  local law: [G, grad_par L G] = grad_par( (1/2)[G, L G] ) pointwise (generic coefficient fields), hence
     d_t w + grad_par gamma = 0 with w = (1/2)[G,G], gamma = (s/2)[G, L G]; and d_t + {Phi,.} + grad_par = div of (w u_E + gamma bhat).
 I6  u = 0: (L G)(0) = 0, so G(r, 0, t) obeys d_t G(0) + {Phi, G(0)} = 0 (CMM GKvpar0 in KRMHD); a_0 = (1-1/Lambda) G(0)/F_M(0).
 I7  products of slices at equal u (same flow) are advected: cross-branch quadratic invariants at fixed u.
Exit 1 on failure.
"""
import sys
import random
import sympy as sp

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


u, v, vt = sp.symbols('u v v_t', positive=True)
u = sp.Symbol('u', real=True)
vv = sp.Symbol('v', real=True)
Lam = sp.Symbol('Lambda', nonzero=True)


def gmom(k):
    if k % 2:
        return sp.Integer(0)
    return sp.factorial2(k - 1) / 2**sp.Rational(k, 2) if k > 0 else sp.Integer(1)


def integ(p, var=None):
    var = var or u
    P = sp.Poly(sp.expand(p), var)
    return sp.expand(sum(c * gmom(m[0]) for m, c in zip(P.monoms(), P.coeffs())))


Mk = lambda k, q: integ(u**k * q)
Lop = lambda q: sp.expand(u * q - u * Mk(0, q) / Lam)
br = lambda p, q: sp.expand(integ(p * q) - Mk(0, p) * Mk(0, q) / Lam)


def Ln(q, n):
    for _ in range(n):
        q = Lop(q)
    return q


rng = random.Random(7)
rp = lambda deg: sum(sp.Rational(rng.randint(-9, 9), rng.randint(1, 6)) * u**k for k in range(deg + 1))
G = rp(6)

print("I1  CMM integrands in dimensional v_par (direct integrals, symbolic v_th)")
vt = sp.Symbol('v_t', positive=True)
Fd = sp.exp(-vv**2 / vt**2) / (sp.sqrt(sp.pi) * vt)         # CMM's F_M(v_par), text below (B4)
qd = sp.Rational(1, 2) - sp.Rational(3, 4) * u + u**2 - sp.Rational(1, 3) * u**3   # test polynomial, q(0) != 0
q0 = sp.expand(qd - qd.subs(u, 0))                              # q(0) = 0 (for the PV-free H check)
Gd = lambda q: q.subs(u, vv / vt) * Fd                          # G(v) = q(v/v_t) F(v): G dv = q(u) F_M(u) du
I = lambda f: sp.simplify(sp.integrate(sp.expand(f), (vv, -sp.oo, sp.oo)))
M0d = I(Gd(qd)); M1d = I(vv * Gd(qd))
W_dim = I(Gd(qd)**2 / Fd) - M0d**2 / Lam                        # (B13) bracket
gam_dim = I(vv * Gd(qd)**2 / (2 * Fd)) - M0d * M1d / Lam          # (B25) integrand
H_dim = I(Gd(q0)**2 / (2 * vv * Fd))                             # (B18) integrand (no PV needed since q0(0) = 0)
check("(B13) bracket in v_par = [G,G] in u", sp.simplify(W_dim - br(qd, qd)) == 0)
check("(B25) integrand gamma = (v_th/2)[G, L G]", sp.simplify(gam_dim - vt / 2 * br(qd, Lop(qd))) == 0)
check("(B18) integrand = (1/(2 v_th)) P int G^2/(u F_M) du", sp.simplify(H_dim - integ(sp.cancel(q0 * q0 / u)) / (2 * vt)) == 0)
Nn = integ(sp.cancel(q0 / u))
Linv = sp.expand(sp.cancel(q0 / u) + Nn / (Lam - 1))
check("Q_{1/u} = [G, L^{-1} G] = P int G^2/(u F_M) (no Lambda term)", sp.simplify(br(q0, Linv) - integ(sp.cancel(q0 * q0 / u))) == 0)

print("I2  Hermite forms (CMM B31, B32)")
Mx = 8
gs = sp.symbols(f'g0:{Mx + 1}')
Hq = sum(gs[m] * sp.hermite(m, u) / sp.sqrt(2**m * sp.factorial(m)) for m in range(Mx + 1))   # G = Hq F_M
check("[G,G] = sum g_m^2 - g_0^2/Lambda", sp.simplify(br(Hq, Hq) - (sum(g**2 for g in gs) - gs[0]**2 / Lam)) == 0)
B32 = (sum(sp.sqrt(m + 1) * gs[m] * gs[m + 1] for m in range(Mx)) - gs[0] * gs[1] / Lam) / sp.sqrt(2)
check("(1/2)[G, L G] = (1/sqrt2)(sum sqrt(m+1) g_m g_{m+1} - g_0 g_1/Lambda)", sp.simplify(br(Hq, Lop(Hq)) / 2 - B32) == 0)

print("I3  generating function and Q_{u^n}")
w = sp.Symbol('w')       # w = 1/z
NMAX = 8
eps_w = 1 + sum(gmom(k + 1) * w**(k + 1) for k in range(NMAX + 2)) / Lam     # eps = 1 - (1/Lam) int uF/(u - z) = 1 + (1/Lam) sum c_{k+1} z^{-k-1}
inv_eps = sp.series(1 / eps_w, w, 0, NMAX + 2).removeO()
e = [sp.expand(inv_eps.coeff(w, j)) for j in range(NMAX + 1)]
print("    1/eps = sum e_j z^-j:", e[:5])
Mm = [Mk(k, G) for k in range(NMAX + 2)]
res = []
for n in range(NMAX + 1):
    closed = integ(u**n * G * G) - sum(e[j] * sum(Mm[a] * Mm[n - j - a] for a in range(n - j + 1)) for j in range(n + 1)) / Lam
    res.append(sp.simplify(br(G, Ln(G, n)) - closed))
check("[G, L^n G] = closed form, n <= 8", all(r == 0 for r in res), str(res))
c2 = gmom(2)
Qu2 = integ(u**2 * G * G) - (2 * Mm[0] * Mm[2] + Mm[1]**2) / Lam + c2 * Mm[0]**2 / Lam**2
Qu3 = integ(u**3 * G * G) - 2 * (Mm[0] * Mm[3] + Mm[1] * Mm[2]) / Lam + 2 * c2 * Mm[0] * Mm[1] / Lam**2
check("Q_{u^2} = int u^2 G^2/F_M - (2 M0 M2 + M1^2)/Lambda + M0^2/(2 Lambda^2)", sp.simplify(br(G, Ln(G, 2)) - Qu2) == 0)
check("Q_{u^3} = int u^3 G^2/F_M - 2(M0 M3 + M1 M2)/Lambda + M0 M1/Lambda^2", sp.simplify(br(G, Ln(G, 3)) - Qu3) == 0)

print("I4  conservation from the moment hierarchy")
sym = {}


def pair(a, b):
    """[a,b] = int d^3r M_a grad_par M_b, antisymmetric."""
    if a == b:
        return sp.Integer(0)
    if a > b:
        return -pair(b, a)
    if (a, b) not in sym:
        sym[(a, b)] = sp.Symbol(f'P_{a}_{b}')
    return sym[(a, b)]


s_ = sp.Symbol('s', positive=True)
# d/dt int d^3r int u^n G^2/F_M = (2 s/Lambda) [n+1, 0]
ddt_vmom = lambda n: 2 * s_ * pair(n + 1, 0) / Lam
# d/dt int d^3r M_a M_b with d_t M_k = -s grad_par M_{k+1} + s (c_{k+1}/Lambda) grad_par M_0
ddt_MM = lambda a, b: s_ * (-pair(a, b + 1) + gmom(b + 1) / Lam * pair(a, 0) - pair(b, a + 1) + gmom(a + 1) / Lam * pair(b, 0))
res = []
for n in range(NMAX + 1):
    dq = ddt_vmom(n) - sum(e[j] * sum(ddt_MM(a, n - j - a) for a in range(n - j + 1)) for j in range(n + 1)) / Lam
    res.append(sp.simplify(sp.expand(dq)))
check("d/dt int d^3r Q_{u^n} = 0, n <= 8 (moment hierarchy only)", all(r == 0 for r in res), str(res))
check("control: d/dt int d^3r int u^2 G^2/F_M alone != 0", sp.simplify(ddt_vmom(2)) != 0)
# CMM B23, B24 in dimensional form: d_t M0 = -grad M1 ; d_t M1 = -grad M2 + (v_th^2/(2 Lambda)) grad M0
check("c_1 = 0 (B23) and v_th^2 c_2 = v_th^2/2 (B24)", gmom(1) == 0 and gmom(2) == sp.Rational(1, 2))
# derive the hierarchy from the kinetic equation: int u^k (L G) du = M_{k+1} - c_{k+1} M0/Lambda
check("int u^k L G du = M_{k+1} - c_{k+1} M_0/Lambda (k <= 6)", all(sp.simplify(Mk(k, Lop(G)) - (Mk(k + 1, G) - gmom(k + 1) * Mk(0, G) / Lam)) == 0 for k in range(7)))

print("I5  local field-aligned flux law")
x, y, z = sp.symbols('x y z', real=True)
Psi = sp.Function('Psi')(x, y, z)
K = 3
cf = [sp.Function(f'c{k}')(x, y, z) for k in range(K + 1)]
Gq = sum(cf[k] * u**k for k in range(K + 1))          # G = Gq F_M with space-dependent coefficients
pb = lambda f, g: sp.diff(f, x) * sp.diff(g, y) - sp.diff(f, y) * sp.diff(g, x)
gpar = lambda f: sp.diff(f, z) + pb(Psi, f)


def gpar_poly(q):
    """grad_par acting on the coefficient fields of a polynomial in u."""
    P = sp.Poly(sp.expand(q), u)
    return sp.expand(sum(gpar(c) * u**m[0] for m, c in zip(P.monoms(), P.coeffs())))


lhs = br(Gq, gpar_poly(Lop(Gq)))
rhs = gpar(br(Gq, Lop(Gq)) / 2)
check("[G, grad_par L G] = grad_par((1/2)[G, L G]) (generic fields, cubic G)", sp.simplify(sp.expand(lhs - rhs)) == 0)
Phi = sp.Function('Phi')(x, y, z)
wfun = sp.Function('w')(x, y, z); gfun = sp.Function('gamma')(x, y, z)
# {Phi, w} + grad_par gamma = div( w zhat x grad Phi + gamma (zhat + zhat x grad Psi) )
zx = lambda f: (-sp.diff(f, y), sp.diff(f, x))
div = (sp.diff(wfun * zx(Phi)[0] + gfun * zx(Psi)[0], x) + sp.diff(wfun * zx(Phi)[1] + gfun * zx(Psi)[1], y) + sp.diff(gfun, z))
check("{Phi, w} + grad_par gamma = div(w u_E + gamma bhat)", sp.simplify(pb(Phi, wfun) + gpar(gfun) - div) == 0)

print("I6  the u = 0 slice")
check("(L G)(u = 0) = 0", sp.simplify(Lop(G).subs(u, 0)) == 0)

print("I7  products at equal u")
a, b = sp.Function('a')(x, y, z), sp.Function('b')(x, y, z)
uu, tt = sp.symbols('u_0 t')
A = Phi + s_ * uu * Psi
opr = lambda f: pb(A, f) + s_ * uu * sp.diff(f, z)
check("(d_t + U.grad)(a b) = a (d_t + U.grad) b + b (d_t + U.grad) a", sp.simplify(opr(a * b) - a * opr(b) - b * opr(a)) == 0)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s07 CHECKS PASSED")
