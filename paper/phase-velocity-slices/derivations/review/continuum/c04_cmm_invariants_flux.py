"""c04 (independent check): the invariants W+-, Gamma+- and H+- of Chandran, Mallet & Meyrand (2026; CMM below) as
members of the family Q_phi (dimensional quadrature with CMM's Maxwellian), their Hermite forms (B31)-(B32), the
coefficient in (B24), the local field-aligned flux law in Hermite form (with the symmetric closure as a control),
and Gamma/W as the energy-weighted mean slice velocity. It supports Section 4 of the paper.

 N1  dimensional quadrature with v_th = 1.7 and CMM's F_M(v) = pi^{-1/2} v_th^{-1} e^{-v^2/v_th^2} (text below B4):
       (B13) integrand  = (n T/2) [G,G]_u,   (B25) integrand = (v_th/2) [G, L G]_u,   (B18) integrand = (n T/(2 v_th)) P int G^2/(u F_M) du,
     where [.,.]_u and L G = u G - u F_M M0/Lambda are evaluated pointwise from their definitions (not from closed forms).
 N2  (B31), (B32): Hermite coefficients by quadrature (CMM B30) reproduce W and Gamma.
 N3  (B23), (B24): int v^2 F_M dv = v_th^2/2, so d M1/dt = -grad M2 + v_th^2/(2 Lambda) grad M0.
 N4  local law in Hermite form (SymPy, generic fields g_m(x,y,z), M = 5, generic Phi, Psi, s, Lambda):
       sum_m P0_mm g_m d_t g_m + {Phi, w} + grad_par gamma = 0,  w = g^T P0 g/2,  gamma = s g^T P0 A g/2,
     with d_t g = -{Phi, g} - s grad_par(A g), and {Phi,w} + grad_par gamma = div(w u_E + gamma bhat), bhat = zhat + zhat x grad Psi.
     Control: the same identity with the non-symmetric (symmetric-closure) matrix fails.
 N5  Gamma/W as the energy-weighted mean phase velocity: for G = Gaussian mixtures the ratio int u rho T^2 / int rho T^2 equals
     [G,LG]/[G,G] (spectral form, from the c03 machinery re-implemented here by quadrature).
Exit 1 on any failure.
"""
import sys
from math import factorial
import numpy as np
import sympy as sp
from scipy.integrate import quad
from numpy.polynomial.hermite import hermval

fails = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


def Q(f, a=-20.0, b=20.0):
    val = quad(f, a, b, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
    if not np.isfinite(val):
        raise FloatingPointError('non-finite integral')
    return val
PV0 = lambda f, L: quad(f, -L, L, weight='cauchy', wvar=0.0, limit=400, epsabs=1e-14, epsrel=1e-13)[0]

print("N1  CMM integrands (dimensional) versus Q_phi members")
vth, nT = 1.7, 2.3
FMd = lambda v: np.exp(-(v / vth)**2) / (np.sqrt(np.pi) * vth)          # CMM F_M(v_par)
FMu = lambda u: np.exp(-u * u) / np.sqrt(np.pi)
worst = 0
for Lam in (np.sqrt(5.0), -np.sqrt(5.0), 3.3):
    for coeffs in ((0.5, -0.75, 1.0, -1 / 3), (1.2, 0.4, -0.3, 0.2)):
        q = lambda u: coeffs[0] + coeffs[1] * u + coeffs[2] * u**2 + coeffs[3] * u**3
        Gd = lambda v: q(v / vth) * FMd(v) * 1.3 * np.exp(-0.2 * (v / vth)**2)   # G(v_par)
        Gu = lambda u: vth * Gd(vth * u)                                      # G(u) du = G(v) dv
        # CMM integrands
        M0 = Q(Gd); M1 = Q(lambda v: v * Gd(v))
        W13 = nT / 2 * (Q(lambda v: Gd(v)**2 / FMd(v)) - M0**2 / Lam)
        G25 = Q(lambda v: Gd(v)**2 * v / (2 * FMd(v))) - M0 * M1 / Lam
        H18 = nT * PV0(lambda v: Gd(v)**2 / (2 * FMd(v)), 12 * vth)
        # u-variables from definitions
        m0 = Q(Gu)
        LG = lambda u: u * Gu(u) - u * FMu(u) * m0 / Lam
        br = lambda f, g: Q(lambda u: f(u) * g(u) / FMu(u)) - Q(f) * Q(g) / Lam
        W_u = br(Gu, Gu); Gam_u = br(Gu, LG); H_u = PV0(lambda u: Gu(u)**2 / FMu(u), 12)
        e = (abs(W13 - nT / 2 * W_u) / abs(W13), abs(G25 - vth / 2 * Gam_u) / abs(G25), abs(H18 - nT / (2 * vth) * H_u) / abs(H18))
        assert all(np.isfinite(e)), e
        worst = max(worst, *e)
check("(B13) = (nT/2)[G,G], (B25) = (v_th/2)[G,LG], (B18) = (nT/(2 v_th)) P int G^2/(u F_M)", worst < 1e-9, f"max rel {worst:.1e}")

print("N2  Hermite forms (B31), (B32)")
worst = 0
for Lam in (np.sqrt(5.0), -np.sqrt(5.0)):
    Gd = lambda v: (1 + 0.3 * v - 0.2 * v**2) * np.exp(-(v - 0.4)**2 / (1.1 * vth**2)) / vth
    Mh = 60
    Gt = np.array([Q(lambda v, m=m: hermval(v / vth, np.eye(Mh + 2)[m]) * Gd(v)) / np.sqrt(2.0**m * factorial(m)) for m in range(Mh + 2)])
    M0 = Q(Gd); M1 = Q(lambda v: v * Gd(v))
    W13 = nT / 2 * (Q(lambda v: Gd(v)**2 / FMd(v)) - M0**2 / Lam)
    G25 = Q(lambda v: Gd(v)**2 * v / (2 * FMd(v))) - M0 * M1 / Lam
    W31 = nT / 2 * (np.sum(Gt**2) - Gt[0]**2 / Lam)
    G32 = vth / np.sqrt(2) * (np.sum(Gt[:-1] * Gt[1:] * np.sqrt(np.arange(1, Mh + 2))) - Gt[0] * Gt[1] / Lam)
    assert np.isfinite([W31, W13, G32, G25]).all()
    worst = max(worst, abs(W31 - W13) / abs(W13), abs(G32 - G25) / abs(G25))
    print(f"    Lambda={Lam:+.3f}: |G~_{Mh + 1}| = {abs(Gt[-1]):.1e}")
check("(B31) and (B32) reproduce (B13) and (B25)", worst < 1e-9, f"max rel {worst:.1e}")

print("N3  (B24) coefficient")
c2 = Q(lambda v: v * v * FMd(v))
check("int v^2 F_M dv = v_th^2/2", abs(c2 - vth**2 / 2) < 1e-12, f"{c2:.15f} vs {vth**2 / 2:.15f}")

print("N4  local law in Hermite form (SymPy)")
x, y, z, s, Lm = sp.symbols('x y z s Lambda', real=True)
M = 5
g = sp.Matrix([sp.Function(f'g{m}')(x, y, z) for m in range(M + 1)])
Phi = sp.Function('Phi')(x, y, z); Psi = sp.Function('Psi')(x, y, z)
pb = lambda f, h: sp.diff(f, x) * sp.diff(h, y) - sp.diff(f, y) * sp.diff(h, x)
gpar = lambda f: sp.diff(f, z) + pb(Psi, f)
A = sp.zeros(M + 1, M + 1)
for m in range(M):
    A[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); A[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
A[1, 0] = (1 - 1 / Lm) / sp.sqrt(2)
P0 = sp.diag(*([1 - 1 / Lm] + [1] * M))
check("P0 A symmetric (zero closure)", sp.simplify(P0 * A - (P0 * A).T) == sp.zeros(M + 1, M + 1))
Ag = A * g
dtg = sp.Matrix([-pb(Phi, g[m]) - s * gpar(Ag[m]) for m in range(M + 1)])
w = (g.T * P0 * g)[0] / 2
gam = s * (g.T * P0 * A * g)[0] / 2
lhs = sp.expand((g.T * P0 * dtg)[0] + pb(Phi, w) + gpar(gam))
check("d_t w + {Phi, w} + grad_par gamma = 0 pointwise (generic fields)", sp.simplify(lhs) == 0)
zx = lambda f: (-sp.diff(f, y), sp.diff(f, x))
div = sp.diff(w * zx(Phi)[0] + gam * zx(Psi)[0], x) + sp.diff(w * zx(Phi)[1] + gam * zx(Psi)[1], y) + sp.diff(gam, z)
check("{Phi, w} + grad_par gamma = div(w u_E + gamma bhat)", sp.simplify(sp.expand(pb(Phi, w) + gpar(gam) - div)) == 0)
# control: symmetric closure (g_{M+1} = g_{M-1}) breaks P0-symmetry, and the local law with the same gamma fails
Asym = A.copy(); Asym[M, M - 1] += sp.sqrt(sp.Rational(M + 1, 2))
dtg2 = sp.Matrix([-pb(Phi, g[m]) - s * gpar((Asym * g)[m]) for m in range(M + 1)])
gam2 = s * (g.T * P0 * Asym * g)[0] / 2
lhs2 = sp.expand((g.T * P0 * dtg2)[0] + pb(Phi, w) + gpar(gam2))
check("control: with the symmetric closure the identity fails (non-vacuous)", sp.simplify(lhs2) != 0)

print("N5  Gamma/W as mean phase velocity")
PVq = lambda f, u0: quad(f, -14, 14, weight='cauchy', wvar=u0, limit=400, epsabs=1e-13, epsrel=1e-12)[0]
ug = np.linspace(-9, 9, 721); hu = ug[1] - ug[0]
PVvFM = np.array([PVq(lambda v: v * FMu(v), u0) for u0 in ug])
worst = 0
for Lam in (np.sqrt(5.0), -np.sqrt(5.0)):
    Gu = lambda u: (1 + 0.6 * u) * np.exp(-(u - 0.3)**2 / 1.2)
    m0 = Q(Gu)
    epsR = 1 - PVvFM / Lam; epsI = -np.pi * ug * FMu(ug) / Lam
    PVvG = np.array([PVq(lambda v: v * Gu(v), u0) for u0 in ug])
    FMT = PVvG * FMu(ug) / Lam + epsR * Gu(ug) - m0 * FMu(ug) / Lam
    rT2 = FMT**2 / (FMu(ug) * (epsR**2 + epsI**2))
    tr = lambda f: hu * (f.sum() - 0.5 * (f[0] + f[-1]))
    LG = lambda u: u * Gu(u) - u * FMu(u) * m0 / Lam
    br = lambda f, h: Q(lambda u: f(u) * h(u) / FMu(u)) - Q(f) * Q(h) / Lam
    ratio_spec = tr(ug * rT2) / tr(rT2); ratio_def = br(Gu, LG) / br(Gu, Gu)
    assert np.isfinite([ratio_spec, ratio_def]).all()
    worst = max(worst, abs(ratio_spec - ratio_def) / abs(ratio_def))
    print(f"    Lambda={Lam:+.3f}: [G,LG]/[G,G] = {ratio_def:.12f}, int u rho T^2/int rho T^2 = {ratio_spec:.12f}")
check("Gamma/W (per v_th/(nT)) = energy-weighted mean slice velocity", worst < 1e-9, f"rel {worst:.1e}")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL c04 CHECKS PASSED")
