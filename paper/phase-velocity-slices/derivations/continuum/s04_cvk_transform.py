"""s04: the Case-Van Kampen modes F_u and the slice transform T[G](u) = [F_u, G] (SymPy, SciPy and mpmath): the
resolvent of L and Stone's formula for the spectral density, the eigen-relation T[L G] = u T[G], Parseval's
identity and the spectral forms of [G, phi(L) G], completeness and orthogonality (the inverse transform), the total
mass of the spectral density, and the unstable pair that completes Parseval's identity for 0 < Lambda < 1. It
supports Section 3 and Appendix B of the paper.

Notation: Hc[f](u) = (1/pi) P int f(v)/(v-u) dv; eps(u+i0) = eR + i eI, eR = 1 + Hc[eI], eI = -pi u F_M/Lambda;
rho = F_M/(eR^2 + eI^2).
  CVK mode   F_u(v) = (1/Lambda) P[v F_M(v)/(v - u)] + eR(u) delta(v - u)      (M0[F_u] = 1)
  transform  T[G](u) = [F_u, G] = (eR G - eI Hc[G])/F_M = eR G/F_M + (u/Lambda) P int G/(v-u) dv
  inverse    G = int rho T F_u du = eR c + eI Hc[c],   c = rho T          (Morrison's G-transform of c)
Checks:
 C1  resolvent: (z - L)^{-1} G has M0 = C(z)/eps(z), C(z) = int G/(z-v) dv, and
     Q(z) = [G,(z-L)^{-1}G] = int G^2/((z-v)F_M) dv - (z/Lambda) C(z)^2/eps(z)       (SymPy algebra + Hermite solve)
 C2  Stone: -(1/pi) Im Q(u+i0) = (eR G - eI Hc G)^2/(F_M |eps|^2) = rho T^2                (SymPy algebra)
 C3  F_u: M0[F_u] = 1, M1[F_u] = u (PV quadrature); T[F_M] = 1 - 1/Lambda; T[L G] = u T[G]
 C4  Parseval [G1,G2] = int rho T[G1] T[G2] du; Q_u, Q_{u^2}, Q_{1/u} spectral forms; Q(z) = int rho T^2/(z-u) du
 C5  inverse o forward: eR c + eI Hc[c] = G with c = rho T[G]  (completeness)
 C6  forward o inverse: rho T[eR c + eI Hc c] = c                (orthogonality [F_u, F_u'] = delta(u-u')/rho(u))
 C7  int rho du = Lambda/(Lambda - 1); rho(0) = F_M(0)/(1 - 1/Lambda)^2; rho -> F_M at large |u|
 C8  0 < Lambda < 1: Parseval fails by exactly the unstable pair: [G,G] = int rho T^2 + 2 Re([F+,G]^2/[F+,F+]),
     F+ = v F_M/(Lambda (v - i gamma)), [F+,G] = (i gamma/Lambda) int G/(v - i gamma), [F+,F+] = (Lambda - 1 + 2 gamma^2 Lambda)/Lambda^2
Exit 1 on failure.
"""
import os
import sys
import numpy as np
import sympy as sp
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_common import (FM, ReZ, Zc, eps_c, eR, eI, rho, Gauss, T_of, FMT_of, rhoT2, ladder, P0diag, SQRT5, SQPI)

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


Q = lambda f, a=-14, b=14, **kw: quad(f, a, b, limit=1000, epsabs=1e-14, epsrel=1e-13, **kw)[0]
PVq = lambda f, u0, a=-14, b=14: quad(f, a, b, weight='cauchy', wvar=u0, limit=1000, epsabs=1e-14, epsrel=1e-13)[0]

# ---------------------------------------------------------------------------------------------------------- C1, C2
print("C1  resolvent and generating function (SymPy algebra)")
z, Lam, m0, CG, IGG, RZ = sp.symbols('z Lambda m0 C_G I_GG R')   # R = 1 + z Z(z) = int v F_M/(v - z)
# X = (G - (v F_M/Lambda) m)/(z - v), m = M0[X]:  M0[X] = C_G - (m/Lambda) int v F_M/(z - v) = C_G + (m/Lambda) R
msol = sp.solve(sp.Eq(m0, CG + m0 * RZ / Lam), m0)[0]
epsz = 1 - RZ / Lam
check("M0[(z-L)^{-1}G] = C_G/eps(z)", sp.simplify(msol - CG / epsz) == 0)
# [G, X] = int G X/F_M - M0[G] M0[X]/Lambda;  int G X/F_M = I_GG - (m/Lambda) int v G/(z-v) = I_GG - (m/Lambda)(z C_G - M0G)
M0G = sp.Symbol('M0G')
QG = IGG - (msol / Lam) * (z * CG - M0G) - M0G * msol / Lam
check("Q(z) = I_GG - (z/Lambda) C_G^2/eps(z)", sp.simplify(QG - (IGG - z * CG**2 / (Lam * epsz))) == 0)

print("C2  Stone formula (SymPy algebra)")
Gs, hs, eRs, Fs, us = sp.symbols('G h e_R F_M u', real=True)
Ls = sp.Symbol('Lambda', real=True, nonzero=True)
eIs = -sp.pi * us * Fs / Ls
CGb = -sp.pi * (hs + sp.I * Gs)                 # C_G(u + i0) = P int G/(u-v) - i pi G = -pi (Hc G + i G)
ImQ = -sp.pi * Gs**2 / Fs + sp.im(sp.expand(-(us / Ls) * CGb**2 * (eRs - sp.I * eIs) / (eRs**2 + eIs**2)))
target = (eRs * Gs - eIs * hs)**2 / (Fs * (eRs**2 + eIs**2))
check("-(1/pi) Im Q(u+i0) = (eR G - eI Hc G)^2/(F_M |eps|^2)", sp.simplify(-ImQ / sp.pi - target) == 0)

# numerical check of the closed-form Q(z) against a large Hermite solve (z - A) x = g, Q = g^T P0 x
G1 = Gauss([(1.0, 0.5, 1.0), (-0.7, -0.3, 0.8)])
G2 = Gauss([(0.4, -0.9, 1.0), (1.0, 0.6, 1.2)])
G3 = Gauss([(1.0, 0.0, 0.6), (0.5, 1.4, 0.9)])
worst = 0
for Lv in (SQRT5, -SQRT5, 5.0):
    for zc in (0.7 + 0.8j, -1.3 + 0.6j, 0.2 - 0.9j):
        Mh = 800
        g = G1.hermite(Mh)
        x = np.linalg.solve(zc * np.eye(Mh + 1) - ladder(Mh, Lv), g)
        Qh = g @ (P0diag(Mh, Lv) * x)
        # closed form; for Im z < 0 use C(z) = -int G/(v - z) with the integral evaluated directly (no continuation)
        if zc.imag > 0:
            Cz = -G1.Cz(zc)
            epz = eps_c(zc, Lv)
        else:
            Cz = np.conj(-G1.Cz(np.conj(zc)))
            epz = np.conj(eps_c(np.conj(zc), Lv))
        IGG = Q(lambda v: (G1(v)**2 / FM(v) / (zc - v)).real) + 1j * Q(lambda v: (G1(v)**2 / FM(v) / (zc - v)).imag)
        Qc = IGG - zc * Cz**2 / (Lv * epz)
        worst = max(worst, abs(Qh - Qc) / abs(Qc))
check("closed-form Q(z) = Hermite resolvent g^T P0 (z-A)^{-1} g (M=800, Im z = +-0.6..0.9)", worst < 1e-10, f"max rel {worst:.1e}")

# ---------------------------------------------------------------------------------------------------------- C3
print("C3  CVK modes and eigen-relation")
import mpmath as mp
mp.mp.dps = 25
FMm = lambda x: mp.e**(-x**2) / mp.sqrt(mp.pi)
dawm = lambda x: mp.sqrt(mp.pi) / 2 * mp.e**(-x**2) * mp.erfi(x)
w = 0
for Lv in (mp.sqrt(5), -mp.sqrt(5), mp.mpf(7)):
    for u0 in (mp.mpf('-1.1'), mp.mpf('0.35'), mp.mpf('2.2')):
        eRu = 1 - (1 + u0 * (-2 * dawm(u0))) / Lv
        for k in (0, 1):
            # M_k[F_u] = (1/Lambda) P int v^{k+1} F_M/(v-u) dv + eR(u) u^k
            pv = mp.quad(lambda t: ((u0 + t)**(k + 1) * FMm(u0 + t) - (u0 - t)**(k + 1) * FMm(u0 - t)) / t, [0, 1, 4, mp.inf])
            val = pv / Lv + eRu * u0**k
            w = max(w, abs(val - (1 if k == 0 else u0)))
check("M0[F_u] = 1 and M1[F_u] = u (PV quadrature, 3 Lambda x 3 u)", w < 1e-20, mp.nstr(w, 3))
uu = np.linspace(-4, 4, 41)
w1 = w2 = 0
for Lv in (SQRT5, -SQRT5, 5.0, -0.4):
    TFM = eR(uu, Lv) + (uu / Lv) * ReZ(uu)          # T[F_M] with P int F_M/(v-u) = ReZ(u)
    w1 = max(w1, np.abs(TFM - (1 - 1 / Lv)).max())
    for Gt in (G1, G2):
        # L G = u G - u F_M M0/Lambda ;  P int LG/(v-u) dv = M0 + u PV[G] - (M0/Lambda)(1 + u ReZ)
        LG = uu * Gt(uu) - uu * FM(uu) * Gt.M0() / Lv
        PV_LG = Gt.M0() + uu * Gt.PV(uu) - (Gt.M0() / Lv) * (1 + uu * ReZ(uu))
        TLG = eR(uu, Lv) * LG / FM(uu) + (uu / Lv) * PV_LG
        w2 = max(w2, np.abs(TLG - uu * T_of(Gt, uu, Lv)).max() / np.abs(uu * T_of(Gt, uu, Lv)).max())
check("T[F_M](u) = 1 - 1/Lambda for all u", w1 < 1e-14, f"{w1:.1e}")
check("T[L G] = u T[G]", w2 < 1e-14, f"{w2:.1e}")

# ---------------------------------------------------------------------------------------------------------- C4
print("C4  Parseval and spectral forms")


def br(Ga, Gb, Lv, wfun=lambda v: 1.0):
    I = Q(lambda v: wfun(v) * Ga(v) * Gb(v) / FM(v), -12, 12)
    return I


rows = []
worst = 0
for Lv in (SQRT5, -SQRT5, 5.0, -0.4, 50.0):
    for Ga, Gb in ((G1, G1), (G1, G2), (G2, G3), (G3, G3)):
        lhs = br(Ga, Gb, Lv) - Ga.M0() * Gb.M0() / Lv
        # rho T1 T2 evaluated stably as (F_M T1)(F_M T2)/(F_M |eps|^2)
        rhs = Q(lambda x: FMT_of(Ga, x, Lv) * FMT_of(Gb, x, Lv) / (FM(x) * (eR(x, Lv)**2 + eI(x, Lv)**2)), -14, 14)
        worst = max(worst, abs(lhs - rhs) / abs(lhs))
check("[G1,G2] = int rho T[G1] T[G2] du (5 Lambda x 4 pairs)", worst < 1e-11, f"max rel {worst:.1e}")
worst = 0
for Lv in (SQRT5, -SQRT5, 5.0):
    for Gt in (G1, G2, G3):
        Qu = br(Gt, Gt, Lv, lambda v: v) - 2 * Gt.M0() * Gt.M1() / Lv
        Qu_s = Q(lambda x: x * rhoT2(Gt, x, Lv), -14, 14)
        # [G, L^2 G] = [LG, LG];  LG = u G - u F_M M0/Lambda
        LG = lambda v: v * Gt(v) - v * FM(v) * Gt.M0() / Lv
        M0LG = Gt.M1()
        Qu2 = Q(lambda v: LG(v)**2 / FM(v), -12, 12) - M0LG**2 / Lv
        Qu2_s = Q(lambda x: x * x * rhoT2(Gt, x, Lv), -14, 14)
        f = lambda v: Gt(v)**2 / FM(v)
        Qinv = quad(lambda v: (f(v) - f(-v)) / v if v > 0 else 0.0, 0, 12, limit=1000, epsabs=1e-14, epsrel=1e-13)[0]
        Qinv_s = PVq(lambda x: rhoT2(Gt, x, Lv), 0.0)
        worst = max(worst, abs(Qu - Qu_s) / abs(Qu), abs(Qu2 - Qu2_s) / abs(Qu2), abs(Qinv - Qinv_s) / abs(Qinv))
check("Q_u = [G,LG], Q_{u^2} = [LG,LG], Q_{1/u} = P int G^2/(uF_M) equal int phi rho T^2", worst < 1e-10, f"max rel {worst:.1e}")
worst = 0
for Lv in (SQRT5, -SQRT5):
    for zc in (0.7 + 0.5j, -1.3 + 0.2j, 2.0 + 1.0j):
        lhs = Q(lambda x: (rhoT2(G1, x, Lv) / (zc - x)).real) + 1j * Q(lambda x: (rhoT2(G1, x, Lv) / (zc - x)).imag)
        IGG = Q(lambda v: (G1(v)**2 / FM(v) / (zc - v)).real, -12, 12) + 1j * Q(lambda v: (G1(v)**2 / FM(v) / (zc - v)).imag, -12, 12)
        rhs = IGG - zc * (-G1.Cz(zc))**2 / (Lv * eps_c(zc, Lv))
        worst = max(worst, abs(lhs - rhs) / abs(rhs))
check("Q(z) = int rho T^2/(z - u) du (Stieltjes transform of the spectral density)", worst < 1e-11, f"max rel {worst:.1e}")
# u = 0 slice: rho(0) T(0)^2 = G(0)^2/F_M(0)
w = 0
for Lv in (SQRT5, -SQRT5, 5.0):
    for Gt in (G1, G2, G3):
        w = max(w, abs(rhoT2(Gt, 0.0, Lv) - Gt(0.0)**2 / FM(0.0)))
check("rho(0) T[G](0)^2 = G(0)^2/F_M(0) (u = 0 slice density, Lambda-independent)", w < 1e-14, f"{w:.1e}")

# ---------------------------------------------------------------------------------------------------------- C5
print("C5  inverse o forward (completeness)")
worst = 0
for Lv in (SQRT5, -SQRT5, 5.0):
    for Gt in (G1, G2):
        c = lambda x: FMT_of(Gt, x, Lv) / (eR(x, Lv)**2 + eI(x, Lv)**2)       # c = rho T
        for v0 in (-2.0, -0.7, 0.0, 0.45, 1.3, 2.6):
            Hc_c = PVq(c, v0) / np.pi
            rec = eR(v0, Lv) * c(v0) + eI(v0, Lv) * Hc_c
            worst = max(worst, abs(rec - Gt(v0)) / max(abs(Gt(np.linspace(-3, 3, 61))).max(), 1e-300))
check("eR c + eI Hc[c] = G, c = rho T[G]", worst < 1e-11, f"max rel {worst:.1e}")

# ---------------------------------------------------------------------------------------------------------- C6
print("C6  forward o inverse (orthogonality of CVK modes)")


class Cfun:
    """c(u) = sum (al + be u) F_M((u-a)/s)/s with exact Hc."""

    def __init__(self, terms):
        self.t = terms

    def __call__(self, u):
        return sum((al + be * u) * FM((u - a) / s) / s for al, be, a, s in self.t)

    def H(self, u):
        out = 0.0
        for al, be, a, s in self.t:
            h0 = ReZ((u - a) / s) / s / np.pi                 # Hc[F_M((.-a)/s)/s]
            out = out + al * h0 + be * (u * h0 + 1.0 / np.pi)  # Hc[v g] = u Hc[g] + (1/pi) int g
        return out


worst = 0
for Lv in (SQRT5, -SQRT5, 5.0):
    for cf in (Cfun([(1.0, 0.5, 0.3, 1.0)]), Cfun([(0.3, -1.0, -0.6, 0.8), (1.0, 0.0, 1.1, 1.1)])):
        Gc = lambda v: eR(v, Lv) * cf(v) + eI(v, Lv) * cf.H(v)
        for u0 in (-1.8, -0.5, 0.0, 0.7, 1.9):
            HG = PVq(Gc, u0) / np.pi
            back = (eR(u0, Lv) * Gc(u0) - eI(u0, Lv) * HG) / (eR(u0, Lv)**2 + eI(u0, Lv)**2)
            worst = max(worst, abs(back - cf(u0)) / abs(cf(np.linspace(-2, 2, 41))).max())
check("rho T[eR c + eI Hc c] = c (two c, three Lambda, five u)", worst < 1e-11, f"max rel {worst:.1e}")

# ---------------------------------------------------------------------------------------------------------- C7
print("C7  spectral density")
w = 0
for Lv in (SQRT5, -SQRT5, 5.0, -0.4, 50.0, 1.05):
    mass = Q(lambda x: rho(x, Lv), -14, 14)
    w = max(w, abs(mass - Lv / (Lv - 1)))
check("int rho du = Lambda/(Lambda-1)", w < 1e-10, f"{w:.1e}")
check("rho(0) = F_M(0)/(1 - 1/Lambda)^2", abs(rho(0.0, SQRT5) - FM(0.0) / (1 - 1 / SQRT5)**2) < 1e-15)
check("rho/F_M -> 1 at |u| = 8", abs(rho(8.0, SQRT5) / FM(8.0) - 1) < 2e-2 and abs(rho(8.0, -SQRT5) / FM(8.0) - 1) < 2e-2,
      f"{rho(8.0, SQRT5) / FM(8.0):.4f}, {rho(8.0, -SQRT5) / FM(8.0):.4f}")

# ---------------------------------------------------------------------------------------------------------- C8
print("C8  0 < Lambda < 1: the unstable pair completes Parseval")
from scipy.special import erfcx
from scipy.optimize import brentq
worst = 0
for Lv in (0.5, 0.2):
    gam = brentq(lambda g: 1 - g * SQPI * erfcx(g) - Lv, 1e-9, 50)
    zeta0 = 1j * gam
    for Gt in (G1, G2, G3):
        lhs = br(Gt, Gt, Lv) - Gt.M0()**2 / Lv
        cont = Q(lambda x: rhoT2(Gt, x, Lv), -14, 14)
        FpG = (zeta0 / Lv) * Gt.Cz(zeta0)
        FpFp = (Lv - 1 + 2 * gam**2 * Lv) / Lv**2
        disc = 2 * (FpG**2 / FpFp).real
        worst = max(worst, abs(lhs - cont - disc) / abs(lhs))
        print(f"    Lambda={Lv}: [G,G]={lhs:+.10f}, continuum={cont:+.10f}, unstable pair={disc:+.10f}")
check("[G,G] = int rho T^2 + 2 Re([F+,G]^2/[F+,F+])", worst < 1e-10, f"max rel {worst:.1e}")
# check of [F+,F+] formula by quadrature at Lambda = 0.5
Lv = 0.5
gam = brentq(lambda g: 1 - g * SQPI * erfcx(g) - Lv, 1e-9, 50)
Fp = lambda v: v * FM(v) / (Lv * (v - 1j * gam))
FpFp_q = Q(lambda v: (Fp(v)**2 / FM(v)).real) + 1j * Q(lambda v: (Fp(v)**2 / FM(v)).imag) - 1 / Lv
check("[F+,F+] = (Lambda - 1 + 2 gamma^2 Lambda)/Lambda^2 (quadrature)", abs(FpFp_q - (Lv - 1 + 2 * gam**2 * Lv) / Lv**2) < 1e-12,
      f"{FpFp_q:.12f}")
M0Fp = Q(lambda v: Fp(v).real) + 1j * Q(lambda v: Fp(v).imag)
check("M0[F+] = 1", abs(M0Fp - 1) < 1e-12)

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s04 CHECKS PASSED")
