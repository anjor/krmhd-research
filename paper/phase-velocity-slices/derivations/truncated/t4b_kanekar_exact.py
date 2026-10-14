"""t4b_kanekar_exact: checks that the exact steady-state Hermite spectrum of Kanekar et al. (2015) eq. (4.10), for a
density moment driven by white noise, is the spectrum of delta-correlated slices at every finite Lambda, not only at
Lambda = inf; that (1 - 1/Lambda) times the Lambda = inf closed form misses it at small m (by 4.7% at m = 1 for
Lambda = sqrt5 and 5.2% for Lambda = -sqrt5); and that the antisymmetric part of the forced slice correlation carries
the injection rate eps_W = (1 - 1/Lambda) eps_k through every m. All integrals are mpmath quadratures over the whole
real line (30 digits). Supports Section 5 (the forced spectrum and the dephasing argument) and the m = 1 numbers of
Section 8 of the paper; t4_dictionary.py checks the Lambda = inf closed form and the large-m asymptote by other routes.

Variables: u = v_par/v_th, F_M = exp(-u^2)/sqrt(pi), eps(u+i0) = 1 - [1 + u Z(u+i0)]/Lambda, rho = F_M/|eps|^2,
p_0 = 1 - 1/Lambda, p_m = 1 (m >= 1), r_m(u) the ladder polynomials (r_0 = 1, r_1 = sqrt2 u,
sqrt((m+1)/2) r_{m+1} = u r_m - A_{m,m-1} r_{m-1}, A_{10} = (1 - 1/Lambda)/sqrt2), phi_m = sqrt(p_m rho) r_m.
Kanekar's alpha is -1/Lambda and their |k| is our s|k_z|; spectra are in units of eps_k/(s|k_z|).
A  int rho du = 1/p_0 = Lambda/(Lambda - 1)  (Kanekar's eq. 4.15 in these variables)
B  Lambda = inf: (4.10) = (1/(2 pi)) (2^m m!)^-1 int |Z^(m)(x)|^2 dx = Gamma(m+1/2)/(sqrt2 m!)  (m = 1, 2, 3, 6)
C  finite Lambda: (4.10) = (1 - 1/Lambda)^2/(2 pi) (2^m m!)^-1 int |Z^(m)(x+i0)|^2/|eps(x+i0)|^2 dx equals the spectrum
   of delta-correlated slices with e = pi (1 - 1/Lambda)^2 rho, W(m) = int e phi_m^2 du  (Lambda = +-sqrt5, m = 1..8)
D  m = 1: W(1)/[(1 - 1/Lambda) Gamma(3/2)/sqrt2] = 0.9527 (Lambda = sqrt5), 1.0523 (Lambda = -sqrt5)
E  flux through m from the antisymmetric part S(u,u') = (1 - 1/Lambda)^2 sqrt(rho(u) rho(u')) of the forced
   correlation, which dephases at the rate s k_z (u - u'): Pi_{m+1/2} = int int K_m S du du'
   = (1 - 1/Lambda)^2 sum_{n<=m} (int phi_n sqrt(rho) du)^2 equals eps_W = (1 - 1/Lambda) for m = 0, 1, 3, 6
Run from the repository root:
    uv run python paper/phase-velocity-slices/derivations/truncated/t4b_kanekar_exact.py
Exit status 1 if any check fails.
"""
from __future__ import annotations

import mpmath as mp

from tcore import Checker, section

C = Checker("t4b_kanekar_exact")
mp.mp.dps = 30
SQ5 = mp.sqrt(5)
XBIG = 12
BREAKS = [-mp.inf, -24, -12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12, 24, mp.inf]


def Zders(x, mmax: int) -> list:
    """Z^(m)(x + i0), m = 0..mmax, for real x: closed form with 40 guard digits for |x| < XBIG, and the asymptotic
    series (term-by-term derivatives; the imaginary part ~ exp(-x^2) is negligible there) for |x| >= XBIG."""
    x = mp.mpf(x)
    if abs(x) < XBIG:
        with mp.workdps(mp.mp.dps + 40):
            z = [1j * mp.sqrt(mp.pi) * mp.exp(-x * x) * mp.erfc(-1j * x)]
            z.append(-2 * (1 + x * z[0]))
            for m in range(1, mmax):
                z.append(-2 * (x * z[m] + m * z[m - 1]))
        return [+zz for zz in z]
    out = []
    for m in range(mmax + 1):
        s = mp.mpf(0)
        for k in range(60):
            c = mp.fac2(2 * k - 1) / mp.mpf(2) ** k
            p = 2 * k + 1
            s += -c * (-1) ** m * mp.rf(p, m) * x ** (-(p + m))
        out.append(mp.mpc(s, 0))
    return out


def eps(x, Lam):
    return mp.mpf(1) if Lam is None else 1 - (1 + x * Zders(x, 0)[0]) / Lam


def rho(x, Lam):
    return mp.exp(-x * x) / mp.sqrt(mp.pi) / abs(eps(x, Lam)) ** 2


def r_poly(u, mmax: int, Lam) -> list:
    A10 = (1 - 1 / Lam) / mp.sqrt(2) if Lam is not None else 1 / mp.sqrt(2)
    r = [mp.mpf(1), u / mp.sqrt(mp.mpf(1) / 2)]
    for n in range(1, mmax):
        down = A10 if n == 1 else mp.sqrt(mp.mpf(n) / 2)
        r.append((u * r[n] - down * r[n - 1]) / mp.sqrt(mp.mpf(n + 1) / 2))
    return r


def p0(Lam):
    return mp.mpf(1) if Lam is None else 1 - 1 / Lam


def W410(m: int, Lam) -> mp.mpf:
    """Kanekar et al. (2015) eq. (4.10) in our variables, units eps_k/(s|k_z|)."""
    f = lambda x: abs(Zders(x, m)[m]) ** 2 / abs(eps(x, Lam)) ** 2
    return p0(Lam) ** 2 / (2 * mp.pi) * mp.quad(f, BREAKS) / (2 ** m * mp.factorial(m))


def Wslice(m: int, Lam) -> mp.mpf:
    """Delta-correlated slices with e = pi (1 - 1/Lambda)^2 rho: W(m) = int e p_m rho r_m^2 du."""
    pm = p0(Lam) if m == 0 else 1
    f = lambda u: rho(u, Lam) ** 2 * pm * r_poly(u, m, Lam)[m] ** 2
    return mp.pi * p0(Lam) ** 2 * mp.quad(f, BREAKS)


def closed(m: int) -> mp.mpf:
    """Gamma(m+1/2)/(sqrt2 m!) = pi int F_M psi_m^2 du."""
    return mp.gamma(m + mp.mpf(1) / 2) / (mp.sqrt(2) * mp.factorial(m))


section("A. Normalisation of the spectral density (Kanekar et al. 2015 eq. 4.15)")
worst = mp.mpf(0)
for Lam in (SQ5, -SQ5):
    I = mp.quad(lambda u: rho(u, Lam), [-mp.inf, -4, -2, 0, 2, 4, mp.inf])
    worst = max(worst, abs(I - Lam / (Lam - 1)) / abs(Lam / (Lam - 1)))
    print(f"   Lam={mp.nstr(Lam, 6)}: int rho du = {mp.nstr(I, 20)}; Lam/(Lam - 1) = {mp.nstr(Lam / (Lam - 1), 20)}")
C.check("A1 int rho du = 1/p_0 at Lam = +-sqrt5", worst < 1e-25, f"max rel diff {mp.nstr(worst, 3)}")

section("B. Lambda = inf: eq. (4.10) is the closed form Gamma(m+1/2)/(sqrt2 m!)")
worst = mp.mpf(0)
for m in (1, 2, 3, 6):
    a, b = W410(m, None), closed(m)
    worst = max(worst, abs(a - b) / b)
    print(f"   m={m}: (4.10) = {mp.nstr(a, 15)}; closed form = {mp.nstr(b, 15)}")
C.check("B1 Lam = inf: (4.10) = Gamma(m+1/2)/(sqrt2 m!) (m = 1, 2, 3, 6)", worst < 1e-25, f"max rel diff {mp.nstr(worst, 3)}")

section("C. Finite Lambda: eq. (4.10) is the spectrum of delta-correlated slices with e = pi (1 - 1/Lam)^2 rho")
worst = mp.mpf(0)
for Lam in (SQ5, -SQ5):
    for m in (1, 2, 3, 5, 8):
        a, b = W410(m, Lam), Wslice(m, Lam)
        worst = max(worst, abs(a - b) / a)
        print(f"   Lam={mp.nstr(Lam, 6)} m={m}: (4.10) = {mp.nstr(a, 15)}; slices = {mp.nstr(b, 15)}")
C.check("C1 Lam = +-sqrt5: (4.10) = int pi (1 - 1/Lam)^2 rho phi_m^2 du exactly (m = 1, 2, 3, 5, 8)", worst < 1e-25,
        f"max rel diff {mp.nstr(worst, 3)}")

section("D. (1 - 1/Lambda) times the Lambda = inf closed form holds only asymptotically")
ratios = {}
for Lam in (SQ5, -SQ5):
    ratios[Lam] = W410(1, Lam) / (p0(Lam) * closed(1))
    print(f"   Lam={mp.nstr(Lam, 6)}: W(1)/[(1 - 1/Lam) Gamma(3/2)/sqrt2] = {mp.nstr(ratios[Lam], 8)} "
          f"({mp.nstr(100 * (ratios[Lam] - 1), 3)}%)")
C.check("D1 m = 1: 4.7% below (1 - 1/Lam) x closed form at Lam = sqrt5, 5.2% above at Lam = -sqrt5",
        abs(ratios[SQ5] - mp.mpf('0.9527')) < 5e-5 and abs(ratios[-SQ5] - mp.mpf('1.0523')) < 5e-5,
        f"{mp.nstr(ratios[SQ5], 6)}, {mp.nstr(ratios[-SQ5], 6)}")

section("E. The antisymmetric part of the forced correlation carries eps_W through every m")
ok = True
for Lam in (SQ5, -SQ5):
    vals = []
    for m in (0, 1, 3, 6):
        tot = mp.mpf(0)
        for n in range(m + 1):
            pn = p0(Lam) if n == 0 else 1
            I = mp.quad(lambda u: rho(u, Lam) * mp.sqrt(pn) * r_poly(u, n, Lam)[n], [-mp.inf, -4, -2, 0, 2, 4, mp.inf])
            tot += I ** 2
        flux = p0(Lam) ** 2 * tot
        vals.append(flux / p0(Lam))
        ok = ok and abs(flux / p0(Lam) - 1) < 1e-20
    print(f"   Lam={mp.nstr(Lam, 6)}: Pi_(m+1/2)/eps_W at m = 0, 1, 3, 6: {[mp.nstr(v, 15) for v in vals]}")
C.check("E1 Pi_(m+1/2) = eps_W = (1 - 1/Lam) eps_k for every m (Lam = +-sqrt5)", ok)

C.finish()
