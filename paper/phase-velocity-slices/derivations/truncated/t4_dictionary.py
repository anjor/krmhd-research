"""t4_dictionary: checks the dictionary between slice correlations and the Hermite spectrum W(m), in the continuum
and for the truncated hierarchy: the exact spectrum of delta-correlated slices, its identity with Kanekar et al.
(2015) eq. (4.10) at alpha = 0, the Lam-independent asymptote eps_W/(|k| s sqrt(2m)) on GANDALF's operator, the
large-m dictionary W(m) ~ [int e] chat(sqrt(2m))/(pi sqrt(2m)) and its power-law correspondence, and the truncated
(slice-diagonal) form with its recurrence-equilibrium level. Supports Section 5 (dictionary and prefactor) and
Section 8 and Appendix C (truncated form) of the paper.

A  int F_M(u) psi_m(u)^2 du = Gamma(m+1/2)/(sqrt2 pi m!)  (SymPy exact m <= 12; mpmath m = 50, 200);
   large m: = (1/(pi sqrt(2m))) (1 - 1/(8m) + O(m^-2)).
B  Kanekar et al. (2015) eq. (4.10) at alpha = 0 (Lam = inf): (1/(2^m m!)) int |Z^(m)(zeta)|^2 dzeta =
   sqrt2 pi Gamma(m+1/2)/m!  (Hilbert-transform Parseval), so their exact spectrum is
   C_m = eps Gamma(m+1/2)/(sqrt2 |k| m!) = (pi eps/|k|) int F_M psi_m^2 du : delta-correlated slices with density
   e(u) = (pi eps/|k|) F_M(u).
C  Finite Lam: Kanekar's Kramers-Kronig integral (4.15) is int F_M(z)/|eps(z)|^2 dz = 1/(1 - 1/Lam); the asymptote
   W(m) -> eps_W/(|k| s sqrt(2m)) (eps_W the injection of Q_1 = sum_m p_m |g_m|^2) does not depend on Lam.
   Checked on GANDALF's truncated operator (krmhd.hermite.build_implicit_operator) with an outgoing closure at m = M,
   forced steady state by a Lyapunov solve.
D  Dictionary at large m: C(u,u') = sqrt(e(u)e(u')) c(u-u')  ->  W(m) ~ [int e du] chat(sqrt(2m)) / (pi sqrt(2m));
   chat(kappa) ~ kappa^-p  ->  W ~ m^(-(p+1)/2); second-order structure function S_2(dv) ~ dv^(p-1) for 1 < p < 3.
E  Truncated form: W(m) = p_m sum_jl R_mj R_ml C_jl/(n_j n_l) exactly; uncorrelated slices give
   W(m) = p_m sum_j E_j R_mj^2 / n_j; the closed collisionless system time-averages to this slice-diagonal form,
   whose large-m level is [sum_j E_j Delta_j]/(pi sqrt(2m)) (recurrence equilibrium ~ W_tot / (2 sqrt(M m)) for
   energy injected in g_0), against eps_W/(|k| s sqrt(2m)) for the open (flux) state.
"""
from __future__ import annotations

from math import lgamma

import numpy as np
import sympy as sp
import mpmath as mp
from scipy.special import wofz, roots_hermite
from scipy.integrate import quad
from scipy.linalg import solve_continuous_lyapunov

from tcore import Checker, section, ladder, weight, slices, hermite_functions, SQRT5
from krmhd.hermite import build_implicit_operator

C = Checker("t4_dictionary")


def gh_ratio(m: int) -> float:
    """Gamma(m+1/2)/(sqrt2 pi m!)."""
    return float(np.exp(lgamma(m + 0.5) - lgamma(m + 1.0)) / (np.sqrt(2.0) * np.pi))


section("A. delta-correlated slices with density F_M: exact Hermite spectrum")
u = sp.symbols("u", real=True)
okA = True
for m in range(0, 13):
    psi2 = (sp.hermite(m, u) ** 2) * sp.exp(-u ** 2) / (2 ** m * sp.factorial(m) * sp.sqrt(sp.pi))
    val = sp.integrate(psi2 * sp.exp(-u ** 2) / sp.sqrt(sp.pi), (u, -sp.oo, sp.oo))
    target = sp.gamma(m + sp.Rational(1, 2)) / (sp.sqrt(2) * sp.pi * sp.factorial(m))
    okA = okA and sp.simplify(val - target) == 0
C.check("A1 int F_M psi_m^2 du = Gamma(m+1/2)/(sqrt2 pi m!), m = 0..12 (SymPy exact)", okA)
mp.mp.dps = 30
okA2 = True
for m in (50, 200):
    f = lambda x: (mp.exp(-x * x) / mp.sqrt(mp.pi)) * (mp.hermite(m, x) ** 2 * mp.exp(-x * x)
                                                         / (mp.mpf(2) ** m * mp.factorial(m) * mp.sqrt(mp.pi)))
    val = mp.quad(f, mp.linspace(-12, 12, 49))
    tgt = mp.gamma(m + mp.mpf(1) / 2) / (mp.sqrt(2) * mp.pi * mp.factorial(m))
    okA2 = okA2 and abs(val / tgt - 1) < 1e-20
    print(f"   m={m}: quadrature/exact - 1 = {mp.nstr(val / tgt - 1, 3)}; x pi sqrt(2m) = {mp.nstr(val * mp.pi * mp.sqrt(2 * m), 10)}"
          f" (1 - 1/(8m) = {1 - 1 / (8 * m):.10f})")
C.check("A2 the same at m = 50, 200 (mpmath); W(m) pi sqrt(2m) = 1 - 1/(8m) + O(m^-2)", okA2)

section("B. Kanekar et al. 2015 eq. (4.10) at alpha = 0 is the delta-correlated slice spectrum")


def Zderivs(z: np.ndarray, mmax: int) -> np.ndarray:
    """Z^(m)(z) for real z (Landau contour), m = 0..mmax, from Z' = -2(1 + z Z), Z^(m+1) = -2(z Z^(m) + m Z^(m-1))."""
    out = np.zeros((mmax + 1, z.size), complex)
    out[0] = 1j * np.sqrt(np.pi) * wofz(z)
    if mmax >= 1:
        out[1] = -2.0 * (1.0 + z * out[0])
    for m in range(1, mmax):
        out[m + 1] = -2.0 * (z * out[m] + m * out[m - 1])
    return out


okB = True
for m in (1, 2, 3, 5, 8):
    # integrate |Z^(m)|^2 over the real line; |Z^(m)| ~ m!/z^(m+1) at large z, so a finite range suffices
    f = lambda zz: float(np.abs(Zderivs(np.array([zz]), m)[m, 0]) ** 2)
    val = sum(quad(f, a_, b_, limit=400)[0] for a_, b_ in [(-60, -6), (-6, 0), (0, 6), (6, 60)])
    tail = 2 * (np.exp(lgamma(m + 1)) ** 2) / ((2 * m + 1) * 60.0 ** (2 * m + 1))
    lhs = (val + tail) / (2 ** m * np.exp(lgamma(m + 1)))
    rhs = np.sqrt(2) * np.pi * np.exp(lgamma(m + 0.5) - lgamma(m + 1))
    okB = okB and abs(lhs / rhs - 1) < 1e-7
    print(f"   m={m}: (1/(2^m m!)) int |Z^(m)|^2 = {lhs:.10f}; sqrt2 pi Gamma(m+1/2)/m! = {rhs:.10f}")
C.check("B1 (1/(2^m m!)) int |Z^(m)(zeta)|^2 dzeta = sqrt2 pi Gamma(m+1/2)/m! (Kanekar 4.10 at alpha = 0)", okB)

section("C. Finite Lam: Kramers-Kronig normalisation and the Lam-independent asymptote")
okC = True
for Lam in (SQRT5, -SQRT5, 4.0, -0.3):
    f = lambda zz: float(np.exp(-zz * zz) / np.sqrt(np.pi) /
                         np.abs(1 - (1 + zz * 1j * np.sqrt(np.pi) * wofz(zz)) / Lam) ** 2)
    val = quad(f, -12, 12, limit=400)[0]
    okC = okC and abs(val * (1 - 1 / Lam) - 1) < 1e-9
    print(f"   Lam={Lam:+.4f}: int F_M/|eps|^2 dz = {val:.12f}; 1/(1 - 1/Lam) = {1 / (1 - 1 / Lam):.12f}")
C.check("C1 int F_M(z)/|eps(z + i0)|^2 dz = 1/(1 - 1/Lam) (Kanekar 4.15 in GANDALF variables)", okC)


def forced_spectrum(M: int, Lam: float, kz: float, s: float = 1.0, outgoing: bool = True, nu: float = 0.0,
                    hyper_n: int = 6) -> np.ndarray:
    """Steady state of d_t g = L g + white noise on g_0 (unit intensity) with GANDALF's implicit operator
    L = -i s kz A + nu D (krmhd.hermite.build_implicit_operator), optionally with the outgoing (radiation) closure
    g_{M+1} = -i sgn(kz) g_M added in the last row. Returns W(m) = p_m <|g_m|^2>."""
    Lop = np.array(build_implicit_operator(np.array([kz]), s * s, nu, M, Lam, hyper_n, "zero"))[0]
    if outgoing:
        Lop[M, M] += -abs(kz) * s * np.sqrt((M + 1) / 2.0)
    Q = np.zeros((M + 1, M + 1), complex)
    Q[0, 0] = 1.0
    S = solve_continuous_lyapunov(Lop, -Q)
    return weight(M, Lam) * np.real(np.diag(S))


okC2 = True
kz = 2 * np.pi
Mbig = 600
for Lam in (np.inf, SQRT5, -SQRT5):
    W = forced_spectrum(Mbig, Lam, kz)
    epsW = weight(Mbig, Lam)[0] * 1.0          # injection of Q_1 by unit white noise on g_0
    ms = np.array([1, 2, 4, 8, 16, 32, 64, 128, 200])
    ratio_asym = W[ms] * abs(kz) * np.sqrt(2 * ms) / epsW
    line = "   Lam={:+.4f}: W(m)|k| s sqrt(2m)/eps_W at m = {}: {}".format(Lam, ms.tolist(), np.round(ratio_asym, 4).tolist())
    if np.isinf(Lam):
        exact = np.array([np.pi * gh_ratio(m) / abs(kz) for m in ms])      # (pi eps/|k|) Gamma(m+1/2)/(sqrt2 pi m!)
        dev = np.abs(W[ms] / exact - 1).max()
        line += f"; vs exact Kanekar(4.10) at alpha=0: max rel dev = {dev:.1e}"
        okC2 = okC2 and dev < 2e-3
    else:
        okC2 = okC2 and abs(ratio_asym[-1] - 1) < 0.01 and abs(ratio_asym[-2] - 1) < 0.01
    print(line)
C.check("C2 GANDALF operator, outgoing closure, M = 600: W(m) -> eps_W/(|k| s sqrt(2m)) for Lam = inf, +-sqrt5; "
        "Lam = inf matches Kanekar (4.10) exactly", okC2)

section("D. Large-m dictionary and the power-law correspondence")
x = np.linspace(-14, 14, 2 ** 13 + 1)
dx = x[1] - x[0]
e = np.exp(-x ** 2) / np.sqrt(np.pi)
mmax = 400
Psi = hermite_functions(mmax, x)
# h_m(kappa) = int sqrt(e) psi_m e^{i kappa u} du; W(m) = (1/2pi) int chat(kappa) |h_m(kappa)|^2 dkappa
kap = np.linspace(0, 80, 8001)
dk = kap[1] - kap[0]
ms = np.array([4, 8, 16, 32, 64, 128, 200, 300, 400])
H2 = np.zeros((len(ms), kap.size))
cosM = np.cos(np.outer(kap, x))
sinM = np.sin(np.outer(kap, x))
for i, m in enumerate(ms):
    fm = np.sqrt(e) * Psi[m] * dx
    H2[i] = (cosM @ fm) ** 2 + (sinM @ fm) ** 2


def W_of(chat: np.ndarray) -> np.ndarray:
    # chat even in kappa: (1/2pi) int_{-inf}^{inf} = (1/pi) int_0^inf (trapezoid)
    integ = H2 * chat[None, :]
    return (integ.sum(axis=1) - 0.5 * (integ[:, 0] + integ[:, -1])) * dk / np.pi


okD = True
Wd = W_of(np.ones_like(kap))
dev = np.abs(Wd / np.array([gh_ratio(m) for m in ms]) - 1).max()
print(f"   chat = 1 (delta): Fourier-route W(m) vs Gamma(m+1/2)/(sqrt2 pi m!): max rel dev {dev:.1e}")
okD = okD and dev < 1e-6
for ell in (0.15, 1.0):
    for p in (0.5, 1.0, 1.5, 2.0, 2.5):
        chat_f = lambda kk_: (1 + (kk_ * ell) ** 2) ** (-p / 2)
        Wm = W_of(chat_f(kap))
        pred = chat_f(np.sqrt(2 * ms)) / (np.pi * np.sqrt(2 * ms))
        slope = np.polyfit(np.log(ms[-4:]), np.log(Wm[-4:]), 1)[0]
        slope_pred = np.polyfit(np.log(ms[-4:]), np.log(pred[-4:]), 1)[0]
        print(f"   chat = (1+kappa^2 l^2)^(-p/2), l={ell}, p={p}: W(m)/[chat(sqrt2m)/(pi sqrt2m)] at m={ms.tolist()} = "
              f"{np.round(Wm / pred, 3).tolist()}; slope m=128..400: {slope:.3f} (of the prediction {slope_pred:.3f}; "
              f"pure power law {-(p + 1) / 2:.3f})")
        okD = okD and abs(Wm[-1] / pred[-1] - 1) < 0.03 and abs(slope - slope_pred) < 0.02
        if ell == 1.0:
            okD = okD and abs(slope + (p + 1) / 2) < 0.03
C.check("D1 W(m) ~ [int e] chat(sqrt(2m))/(pi sqrt(2m)); chat ~ kappa^-p gives W ~ m^-(p+1)/2 (p = 0.5..2.5)", okD)
# structure function S_2(dv) = (1/pi) int_0^inf 2 (1 - cos(kappa dv)) chat(kappa) dkappa ~ dv^(p-1), 1 < p < 3
okD2 = True
kk = np.logspace(-3, 7, 400001)
for p in (1.5, 2.0, 2.5):
    chat = (1 + (kk * ell) ** 2) ** (-p / 2)
    dvs = np.array([1e-4, 3e-4, 1e-3, 3e-3])
    S2 = np.array([np.trapezoid(2 * (1 - np.cos(kk * d)) * chat, kk) / np.pi for d in dvs])
    sl = np.polyfit(np.log(dvs), np.log(S2), 1)[0]
    print(f"   p={p}: S_2(dv) slope for dv << l = {sl:.3f} (predicted p - 1 = {p - 1:.3f})")
    okD2 = okD2 and abs(sl - (p - 1)) < 0.03
C.check("D2 S_2(dv) ~ dv^(p-1) for 1 < p < 3", okD2)

section("E. Truncated form of the dictionary")
okE = True
rng = np.random.default_rng(3)
for M in (16, 33):
    for Lam in (SQRT5, -SQRT5):
        for clo in ("zero", "symmetric"):
            uj, R, W, n = slices(M, Lam, clo)
            p = weight(M, Lam, clo)
            g = rng.standard_normal((500, M + 1)) + 1j * rng.standard_normal((500, M + 1))
            a = g @ W
            Cjl = (a.T @ np.conj(a))                      # sum over samples of a_j a_l^*
            Wm_direct = p * (np.abs(g) ** 2).sum(axis=0)
            Wm_slices = p * np.real(np.einsum("mj,ml,jl->m", R / n[None, :], R / n[None, :], Cjl))
            okE = okE and np.abs(Wm_direct - Wm_slices).max() / Wm_direct.max() < 1e-12
C.check("E1 W(m) = p_m sum_jl R_mj R_ml C_jl/(n_j n_l) exactly (both closures)", okE)
# closed collisionless single-k system: time average is slice-diagonal; recurrence-equilibrium level
okE2 = True
for Lam in (np.inf, SQRT5, -SQRT5):
    M = 128
    uj, R, W, n = slices(M, Lam)
    p = weight(M, Lam)
    g0 = np.zeros(M + 1); g0[0] = 1.0
    a0 = W.T @ g0
    E = a0 ** 2 / n
    Wbar = p * (((R / n[None, :]) ** 2 * n[None, :]) @ E)        # p_m sum_j E_j R_mj^2 / n_j
    # explicit long-time average for comparison
    ts = np.linspace(0, 3000, 30001)
    G = (R / n[None, :]) @ (a0[:, None] * np.exp(-1j * 2 * np.pi * np.outer(uj, ts)))
    Wavg = p * (np.abs(G) ** 2).mean(axis=1)
    err = np.abs(Wavg - Wbar).max() / Wbar.max()
    Dj = np.gradient(uj)
    level = (E * Dj).sum()
    mm = np.array([4, 8, 16, 32, 48, 64])
    r1 = Wbar[mm] * np.pi * np.sqrt(2 * mm) / level
    r2 = Wbar[mm] * 2 * np.sqrt(M * mm) / p[0]
    print(f"   Lam={Lam:+.4f}: |<W(m)>_t - p_m sum_j E_j R_mj^2/n_j| / max = {err:.1e}; W(m) pi sqrt(2m)/sum_j E_j Delta_j at "
          f"m={mm.tolist()}: {np.round(r1, 3).tolist()}; W(m) 2 sqrt(M m)/W_tot: {np.round(r2, 3).tolist()}")
    okE2 = okE2 and err < 2e-2 and np.all(np.abs(r1[1:5] - 1) < 0.08)
C.check("E2 closed system: time average = slice-diagonal spectrum, level [sum_j E_j Delta_j]/(pi sqrt(2m)) "
        "(~ W_tot/(2 sqrt(M m)))", okE2)
C.finish()
