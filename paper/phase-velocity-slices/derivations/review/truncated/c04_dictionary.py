"""c04_dictionary: independent checks of the dictionary between slice correlations and the Hermite spectrum: the
exact spectrum of delta-correlated slices by Gauss-Hermite quadrature, Kanekar et al. (2015) eq. (4.10) evaluated
directly at finite Lam with its Lam-independent asymptote, their normalisation integral (4.15) in GANDALF variables,
and the large-m dictionary against a direct 2-D quadrature. Supports Section 5 (dictionary and prefactor) of the
paper.

Independent routes:

a. int F_M psi_m^2 du = Gamma(m+1/2)/(sqrt2 pi m!) by Gauss-Hermite quadrature with weight e^{-2u^2} (exact for
   polynomials), m = 0..150, in mpmath; and the 1/(pi sqrt(2m)) (1 - 1/(8m)) expansion.
b. Kanekar et al. (2015) eq. (4.10) evaluated directly (mpmath, Z^(m) by its recurrence at 80 digits, quadrature in zeta)
   at finite Lam: with alpha = -1/Lam, C_m = (eps/(2 pi |k|)) (1 - 1/Lam)^2 (2^m m!)^-1 int |Z^(m)|^2/|eps(zeta)|^2.
   Checked: at Lam = inf it equals (pi eps/|k|) Gamma(m+1/2)/(sqrt2 pi m!); at Lam = +-sqrt5 the product
   C_m |k| sqrt(2m)/eps_W -> 1 with eps_W = (1 - 1/Lam) eps (Lam-independent asymptote).
c. Kanekar (4.15) in GANDALF variables, int F_M/|eps|^2 = 1/(1 - 1/Lam), with mpmath (Lam = +-sqrt5, 1.2, -0.05).
d. The large-m dictionary against a direct 2-D quadrature of W(m) = int int sqrt(F(u)F(u')) c(u-u') psi_m(u) psi_m(u') for an
   exponential c (p = 2) and a Gaussian c (smooth: cutoff), not through the Fourier route used in t4_dictionary.py.
"""
from __future__ import annotations

import numpy as np
import mpmath as mp
from numpy.polynomial.hermite import hermgauss

from cc import Checks, hdr, SQRT5

C = Checks("c04_dictionary")

hdr("a. Gauss-Hermite (weight e^{-2u^2}) evaluation of int F_M psi_m^2")
mp.mp.dps = 50
okA = True
worst = mp.mpf(0)
x, w = hermgauss(200)                       # nodes/weights for e^{-t^2}; u = t/sqrt2
for m in list(range(0, 13)) + [25, 50, 100, 150]:
    # int F_M psi_m^2 du = (1/pi) int e^{-2u^2} H_m(u)^2/(2^m m!) du = (1/(pi sqrt2)) int e^{-t^2} H_m(t/sqrt2)^2/(2^m m!) dt
    s = mp.fsum(mp.mpf(wi) * mp.hermite(m, mp.mpf(xi) / mp.sqrt(2)) ** 2 for xi, wi in zip(x, w))
    val = s / (mp.pi * mp.sqrt(2) * mp.mpf(2) ** m * mp.factorial(m))
    tgt = mp.gamma(m + mp.mpf(1) / 2) / (mp.sqrt(2) * mp.pi * mp.factorial(m))
    worst = max(worst, abs(val / tgt - 1))
    if m in (50, 150):
        print(f"   m={m}: value x pi sqrt(2m) = {mp.nstr(val * mp.pi * mp.sqrt(2 * m), 12)}; 1 - 1/(8m) + 1/(128 m^2) = "
              f"{mp.nstr(1 - mp.mpf(1) / (8 * m) + mp.mpf(1) / (128 * m * m), 12)}")
print(f"   max relative difference from Gamma(m+1/2)/(sqrt2 pi m!) over m <= 150: {mp.nstr(worst, 3)} "
      f"(float64 nodes limit this to ~1e-14)")
C("c04.a int F_M psi_m^2 = Gamma(m+1/2)/(sqrt2 pi m!) (Gauss-Hermite, weight e^{-2u^2})", worst < 1e-12)

hdr("b. Kanekar et al. (4.10) at finite Lam, evaluated directly")
mp.mp.dps = 80


def Zm_all(z, mmax):
    """Z^(m)(z) for real z on the Landau contour, m = 0..mmax (Z' = -2(1 + zZ), Z^(m+1) = -2(z Z^(m) + m Z^(m-1)))."""
    Z0 = 1j * mp.sqrt(mp.pi) * mp.exp(-z * z) * mp.erfc(-1j * z)
    out = [Z0, -2 * (1 + z * Z0)]
    for m in range(1, mmax):
        out.append(-2 * (z * out[m] + m * out[m - 1]))
    return out


def eps_of(z, Lam):
    Z0 = 1j * mp.sqrt(mp.pi) * mp.exp(-z * z) * mp.erfc(-1j * z)
    return 1 - (1 + z * Z0) / Lam


ms = [10, 40, 100]
mmax = max(ms)
# integrate in zeta: the integrand ~ e^{-zeta^2} sqrt(2 pi/m) inside |zeta| < sqrt(2m), and decays as zeta^{-2m-2} outside
Zs = {}
zgrid = {}
for m in ms:
    Lz = float(np.sqrt(2 * m)) + 12.0
    zgrid[m] = Lz
nodes = {}


GLx, GLw = np.polynomial.legendre.leggauss(64)


def integral(m, Lam):
    """(2^m m!)^-1 int |Z^(m)(zeta)|^2 / |eps(zeta)|^2 dzeta (eps = 1 for Lam = None): 64-point Gauss-Legendre on unit-ish
    panels over |zeta| < sqrt(2m) + 12, plus the analytic tail |Z^(m)| ~ m!/|zeta|^(m+1)."""
    Lz = zgrid[m]
    edges = np.linspace(-Lz, Lz, int(2 * Lz) + 1)
    tot = mp.mpf(0)
    norm = mp.mpf(2) ** m * mp.factorial(m)
    for a_, b_ in zip(edges[:-1], edges[1:]):
        for xi, wi in zip(GLx, GLw):
            z = mp.mpf((b_ - a_) / 2 * xi + (a_ + b_) / 2)
            val = abs(Zm_all(z, m)[m]) ** 2 / norm
            if Lam is not None:
                val = val / abs(eps_of(z, Lam)) ** 2
            tot += mp.mpf(wi * (b_ - a_) / 2) * val
    tail = 2 * mp.factorial(m) ** 2 / ((2 * m + 1) * mp.mpf(Lz) ** (2 * m + 1)) / norm
    return tot + tail


okB = True
for m in ms:
    I_inf = integral(m, None)
    exact = mp.sqrt(2) * mp.pi * mp.gamma(m + mp.mpf(1) / 2) / mp.factorial(m)
    rel = abs(I_inf / exact - 1)
    line = f"   m={m}: Lam=inf: (2^m m!)^-1 int|Z^(m)|^2 vs sqrt2 pi Gamma(m+1/2)/m!: rel. diff {mp.nstr(rel, 3)}"
    okB = okB and rel < 1e-10
    for Lam in (mp.sqrt(5), -mp.sqrt(5)):
        I = integral(m, Lam)
        # C_m |k|/eps = (1/(2 pi)) (1 - 1/Lam)^2 I ;  eps_W = (1 - 1/Lam) eps  ->  C_m |k| sqrt(2m)/eps_W:
        ratio = (1 - 1 / Lam) * I * mp.sqrt(2 * m) / (2 * mp.pi)
        line += f"; Lam={mp.nstr(Lam, 4)}: C_m |k| sqrt(2m)/eps_W = {mp.nstr(ratio, 6)}"
        if m == ms[-1]:
            okB = okB and abs(ratio - 1) < 0.01
    print(line)
C("c04.b Kanekar (4.10) evaluated directly: closed form at Lam = inf; Lam-independent asymptote eps_W/(|k| sqrt(2m)) at "
  "Lam = +-sqrt5 (within 1% at m = 100)", okB)

hdr("c. Kanekar (4.15): int F_M/|eps|^2 = 1/(1 - 1/Lam)  (break points at the real parts of weakly damped roots)")
mp.mp.dps = 30
okC = True
for Lam in (mp.sqrt(5), -mp.sqrt(5), mp.mpf("1.2"), mp.mpf("-0.05"), mp.mpf("-0.2")):
    brk = [-mp.inf, -3, 0, 3, mp.inf]
    roots = []
    if Lam < 0 and Lam > -1:
        # weakly damped (Langmuir-like) roots of eps: 1 + zeta Z = Lam, near zeta^2 ~ 1/(2|Lam|)
        z0 = mp.sqrt(1 / (2 * abs(Lam)))
        zr = mp.findroot(lambda zz: 1 + zz * 1j * mp.sqrt(mp.pi) * mp.exp(-zz * zz) * mp.erfc(-1j * zz) - Lam,
                         mp.mpc(z0, -1e-4))
        roots.append(zr)
        g_ = abs(mp.im(zr))
        x0 = abs(mp.re(zr))
        extra = []
        for sgn in (-1, 1):
            for k in (1e4, 1e3, 1e2, 30, 10, 3, 1, 0.3, 0):
                extra += [sgn * x0 - k * g_, sgn * x0 + k * g_]
        brk = sorted(set([float(b) for b in brk[1:-1]] + [float(e) for e in extra]))
        brk = [-mp.inf] + [mp.mpf(b) for b in brk] + [mp.inf]
    val = mp.quad(lambda z: mp.exp(-z * z) / mp.sqrt(mp.pi) / abs(eps_of(z, Lam)) ** 2, brk, maxdegree=10)
    tgt = 1 / (1 - 1 / Lam)
    okC = okC and abs(val / tgt - 1) < 1e-10
    extra_txt = f" (root {mp.nstr(roots[0], 8)})" if roots else ""
    print(f"   Lam={mp.nstr(Lam, 5)}: integral {mp.nstr(val, 15)}; 1/(1 - 1/Lam) = {mp.nstr(tgt, 15)}{extra_txt}")
C("c04.c (4.15) in GANDALF variables, including Lam in (-1, 0) where eps has weakly damped roots", okC)

hdr("d. Large-m dictionary against direct 2-D quadrature")
u = np.linspace(-9, 9, 6001)
du = u[1] - u[0]
FMu = np.exp(-u ** 2) / np.sqrt(np.pi)


def psi_all(mmax, x):
    P = np.zeros((mmax + 1, x.size))
    P[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    P[1] = np.sqrt(2.0) * x * P[0]
    for m in range(1, mmax):
        P[m + 1] = np.sqrt(2.0 / (m + 1)) * x * P[m] - np.sqrt(m / (m + 1.0)) * P[m - 1]
    return P


Psi = psi_all(200, u)
D = np.abs(u[:, None] - u[None, :])
okD = True
for name, ell, cfun, chat in (("exponential (p = 2)", 0.3, lambda d, l: np.exp(-d / l), lambda k, l: 2 * l / (1 + (k * l) ** 2)),
                              ("Gaussian (smooth)", 0.3, lambda d, l: np.exp(-d ** 2 / (2 * l * l)),
                               lambda k, l: np.sqrt(2 * np.pi) * l * np.exp(-(k * l) ** 2 / 2))):
    Ck = np.sqrt(FMu)[:, None] * cfun(D, ell) * np.sqrt(FMu)[None, :]
    line = f"   {name}, l = {ell}: W(m) / [int e chat(sqrt2m)/(pi sqrt2m)] at m ="
    vals = []
    for m in (8, 20, 50, 100, 200):
        f = Psi[m] * du
        Wm = f @ Ck @ f
        pred = 1.0 * chat(np.sqrt(2 * m), ell) / (np.pi * np.sqrt(2 * m))
        vals.append(Wm / pred)
        line += f" {m}: {Wm / pred:.4f} (W = {Wm:.3e})"
    print(line)
    if name.startswith("exp"):
        okD = okD and abs(vals[-1] - 1) < 0.02 and abs(vals[-2] - 1) < 0.03
        Wa = [(Psi[m] * du) @ Ck @ (Psi[m] * du) for m in (100, 200)]
        slope = np.log(Wa[1] / Wa[0]) / np.log(2)
        sp_ = np.log2((1 + 200 * ell ** 2) / (1 + 400 * ell ** 2) / np.sqrt(2))
        print(f"     local slope of W(m) between m = 100 and 200: {slope:.3f} (prediction with chat = 2l/(1 + 2m l^2): "
              f"{sp_:.3f}; pure p = 2 law: -1.5)")
C("c04.d W(m) ~ [int e] chat(sqrt(2m))/(pi sqrt(2m)) confirmed by 2-D quadrature for exponential slice correlation", okD)
C.done()
