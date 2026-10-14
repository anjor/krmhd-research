"""s10: the case 0 < Lambda < 1 of the branch operator L, where the free-energy product is indefinite and the
dielectric function has one zero i gamma in the upper half plane (SciPy quadrature). The script checks the expansion
of a distribution function over the Case-Van Kampen modes and the two discrete modes F_+- that appendix B gives for
this case: the reconstruction G = int rho T[G] F_u du + alpha_+ F_+ + alpha_- F_-, the coefficient alpha_+ = -pi res_G
with res_G the residue of chi_G/eps at i gamma, its form [F_+,G]/[F_+,F_+] in the bilinear product, the orthogonality
of F_+- to each other and to every F_u, and the norm [F_+,F_+]. It supports Appendix B (the case 0 < Lambda < 1) of
the paper; s04 C8 checks the corresponding Parseval identity by another route.

Notation as in cc_common.py: u = v_par/v_th, F_M = exp(-u^2)/sqrt(pi), eps(u+i0) = eR + i eI, eI = -pi u F_M/Lambda,
rho = F_M/|eps|^2, T[G] = (eR G - eI Hc[G])/F_M with Hc[f](u) = (1/pi) P int f(v)/(v-u) dv, chi_f(zeta) =
(1/pi) int f(v)/(v - zeta) dv (Im zeta > 0). The bilinear product is [f,g] = int f g/F_M dv - M0[f] M0[g]/Lambda,
without complex conjugation.
Checks (Lambda = 0.5 and 0.2; test functions G are sums of Gaussians, whose Hilbert transforms are exact):
 U1  the zero: 1 - sqrt(pi) gamma e^{gamma^2} erfc(gamma) = Lambda gives eps(i gamma) = 0, and eps(-conj z) = conj eps(z)
 U2  F_+- = v F_M/(Lambda (v -+ i gamma)): M0[F_+-] = 1 and L F_+- = +-i gamma F_+- (pointwise)
 U3  bilinear orthogonality: [F_+, F_-] = 0 (quadrature) and T[F_+](u) = [F_u, F_+] = 0 for real u (PV quadrature)
 U4  reconstruction at 6 points: G = eR c + eI Hc[c] + 2 Re(alpha_+ F_+), c = rho T[G],
     alpha_+ = -pi chi_G(i gamma)/eps'(i gamma) = -pi res_G, eps' = -(Z + zeta Z')/Lambda, Z' = -2(1 + zeta Z)
 U5  alpha_+ = [F_+,G]/[F_+,F_+], with [F_+,G] by direct quadrature and [F_+,F_+] = (Lambda - 1 + 2 gamma^2 Lambda)/Lambda^2
     (closed form against quadrature)
 U6  [G,G] = int rho T[G]^2 du + 2 Re([F_+,G]^2/[F_+,F_+])
Run from the repository root:  uv run python paper/phase-velocity-slices/derivations/continuum/s10_unstable_pair.py
Exit 1 on failure.
"""
import os
import sys
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import erfcx

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_common import FM, Zc, eps_c, eR, eI, rho, Gauss, FMT_of, rhoT2, SQPI  # noqa: E402

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


Q = lambda f, a=-14, b=14: quad(f, a, b, limit=1000, epsabs=1e-14, epsrel=1e-13)[0]
PVq = lambda f, u0, a=-14.3, b=13.9: quad(f, a, b, weight='cauchy', wvar=u0, limit=1000, epsabs=1e-14, epsrel=1e-13)[0]


def cquad(f, a=-14, b=14):
    """int f dv for complex f on [a, b]."""
    return Q(lambda v: f(v).real, a, b) + 1j * Q(lambda v: f(v).imag, a, b)


def gamma_of(Lv):
    """gamma > 0 with 1 - sqrt(pi) gamma e^{gamma^2} erfc(gamma) = Lambda (erfcx(g) = e^{g^2} erfc(g))."""
    return brentq(lambda g: 1 - SQPI * g * erfcx(g) - Lv, 1e-12, 60, xtol=1e-15)


def deps(z, Lv):
    """eps'(zeta) = -(Z + zeta Z')/Lambda with Z' = -2(1 + zeta Z)."""
    Z = Zc(z)
    return -(Z - 2 * z * (1 + z * Z)) / Lv


TESTS = {"G1": Gauss([(1.0, 0.3, 1.0)]), "G2": Gauss([(1.0, -0.5, 0.7), (-0.4, 1.0, 0.9)])}
VS = np.array([-1.7, -0.6, 0.05, 0.45, 1.1, 2.3])

for Lv in (0.5, 0.2):
    gam = gamma_of(Lv)
    z0 = 1j * gam
    print(f"Lambda = {Lv}: gamma = {gam:.10f}")

    # U1 ------------------------------------------------------------------------------------------------------
    e0 = abs(eps_c(z0, Lv))
    zt = 0.37 + 0.81j
    sym = abs(eps_c(-np.conj(zt), Lv) - np.conj(eps_c(zt, Lv)))
    check(f"U1 Lambda={Lv}: eps(i gamma) = 0 and eps(-conj z) = conj eps(z)", e0 < 1e-13 and sym < 1e-14,
          f"|eps(i gamma)| = {e0:.1e}, symmetry {sym:.1e}")

    # U2 ------------------------------------------------------------------------------------------------------
    Fp = lambda v: v * FM(v) / (Lv * (v - z0))
    Fm = lambda v: v * FM(v) / (Lv * (v + z0))
    M0p, M0m = cquad(Fp), cquad(Fm)
    vv = np.linspace(-4, 4, 41)
    LFp = vv * Fp(vv) - vv * FM(vv) / Lv * M0p
    LFm = vv * Fm(vv) - vv * FM(vv) / Lv * M0m
    eL = max(np.abs(LFp - z0 * Fp(vv)).max(), np.abs(LFm + z0 * Fm(vv)).max()) / np.abs(Fp(vv)).max()
    check(f"U2 Lambda={Lv}: M0[F_+-] = 1 and L F_+- = +-i gamma F_+-", abs(M0p - 1) < 1e-12 and abs(M0m - 1) < 1e-12
          and eL < 1e-12, f"|M0 - 1| = {max(abs(M0p - 1), abs(M0m - 1)):.1e}, eigen-relation {eL:.1e}")

    # U3 ------------------------------------------------------------------------------------------------------
    FpFm = cquad(lambda v: Fp(v) * Fm(v) / FM(v)) - M0p * M0m / Lv
    tmax = 0.0
    for u0 in (-2.2, -0.9, 0.3, 1.3, 2.7):
        HFp = (PVq(lambda v: Fp(v).real, u0) + 1j * PVq(lambda v: Fp(v).imag, u0)) / np.pi
        TFp = (eR(u0, Lv) * Fp(u0) - eI(u0, Lv) * HFp) / FM(u0)
        tmax = max(tmax, abs(TFp) / abs(Fp(u0) / FM(u0)))
    check(f"U3 Lambda={Lv}: [F_+, F_-] = 0 and T[F_+](u) = [F_u, F_+] = 0 at real u", abs(FpFm) < 1e-12 and tmax < 1e-11,
          f"|[F+,F-]| = {abs(FpFm):.1e}, max |T[F+]|/|F+/F_M| = {tmax:.1e}")

    FpFp_closed = (Lv - 1 + 2 * gam ** 2 * Lv) / Lv ** 2
    FpFp_q = cquad(lambda v: Fp(v) ** 2 / FM(v)) - M0p ** 2 / Lv
    for name, G in TESTS.items():
        # U4 --------------------------------------------------------------------------------------------------
        c = lambda u: FMT_of(G, u, Lv) / (eR(u, Lv) ** 2 + eI(u, Lv) ** 2)          # c = rho T[G]
        Hc = np.array([PVq(c, v) / np.pi for v in VS])
        Gcont = eR(VS, Lv) * c(VS) + eI(VS, Lv) * Hc
        chiG = G.Cz(z0) / np.pi                                                       # chi_G(i gamma)
        res = chiG / deps(z0, Lv)                                                     # residue of chi_G/eps at i gamma
        alpha = -np.pi * res
        pole = 2 * (alpha * Fp(VS)).real
        err = np.abs(G(VS) - Gcont - pole).max() / np.abs(G(VS)).max()
        frac = np.abs(pole).max() / np.abs(G(VS)).max()
        check(f"U4 Lambda={Lv} {name}: G = eR c + eI Hc[c] + 2 Re(alpha_+ F_+), alpha_+ = -pi res_G", err < 1e-9,
              f"max rel {err:.1e} (discrete part {frac:.2f} of max|G|)")

        # U5 --------------------------------------------------------------------------------------------------
        FpG_q = cquad(lambda v: Fp(v) * G(v) / FM(v)) - M0p * G.M0() / Lv
        FpG_closed = (z0 / Lv) * G.Cz(z0)
        e2 = abs(FpG_q / FpFp_q - alpha) / abs(alpha)
        e3 = abs(FpFp_q - FpFp_closed) / abs(FpFp_closed)
        e4 = abs(FpG_q - FpG_closed) / abs(FpG_closed)
        check(f"U5 Lambda={Lv} {name}: alpha_+ = [F_+,G]/[F_+,F_+]; [F_+,F_+] = (Lambda - 1 + 2 gamma^2 Lambda)/Lambda^2",
              e2 < 1e-9 and e3 < 1e-10 and e4 < 1e-10,
              f"alpha {e2:.1e}, [F+,F+] {e3:.1e}, [F+,G] = (i gamma/Lambda) int G/(v - i gamma) {e4:.1e}")

        # U6 --------------------------------------------------------------------------------------------------
        GG = Q(lambda v: G(v) ** 2 / FM(v)) - G.M0() ** 2 / Lv
        cont = Q(lambda u: rhoT2(G, u, Lv), -12, 12)
        disc = 2 * (FpG_q ** 2 / FpFp_q).real
        e5 = abs(GG - cont - disc) / max(abs(GG), abs(cont))
        check(f"U6 Lambda={Lv} {name}: [G,G] = int rho T^2 + 2 Re([F_+,G]^2/[F_+,F_+])", e5 < 1e-9,
              f"[G,G] = {GG:+.6f}, continuum {cont:+.6f}, pair {disc:+.6f}, rel {e5:.1e}")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s10 CHECKS PASSED")
