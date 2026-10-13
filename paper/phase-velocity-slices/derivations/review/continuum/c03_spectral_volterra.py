"""c03 (independent check): the spectral theorem for L checked through time evolution. The autocorrelation
[G0, exp(-itL) G0], obtained from a Volterra equation for the density, equals the Fourier transform of rho T[G0]^2
for several Lambda; it decays at the Landau rate (no point spectrum); its time integral gives the H kernel and the
u = 0 slice; for 0 < Lambda < 1 the density grows at the unstable rate; and the inverse transform reproduces a
non-Gaussian slice profile. It supports Section 3 and Appendix B of the paper.

Single-k linear problem (time in units 1/(k_par s)):  d_t G = -i L G,  L G = u G - u F_M M0[G]/Lambda.
Exact solution:  G(u,t) = e^{-iut} G0 + (i/Lambda) u F_M int_0^t e^{-iu(t-t')} m(t') dt',  m = M0[G], which gives the
Volterra equation  m(t) = mfree(t) + int_0^t k(t-t') m(t') dt',  k(tau) = (i/Lambda) int u F_M e^{-iu tau} du.
The autocorrelation C(t) = [G0, G(t)] = int e^{-iut} dmu_G0(u) is the Fourier transform of the spectral measure of L in G0.
Claim tested:  dmu_G0 = rho T[G0]^2 du with rho = F_M/|eps(u+i0)|^2 and T[G](u) = [F_u, G], F_u the Case-Van Kampen mode
normalised by M0[F_u] = 1. Every ingredient is computed from its definition by quadrature (no Dawson / Faddeeva closed form,
no code from the derivation scripts in ../../continuum). Checks:
 V1  kernel k(tau) by quadrature equals tau e^{-tau^2/4}/(2 Lambda) (used in the solver).
 V2  C(t) (Volterra, Richardson-extrapolated) = int rho T^2 e^{-iut} du for t in [0, 12], Lambda = sqrt5, -sqrt5, 5, -0.4, 50;
     t = 0 is Parseval [G0,G0] = int rho T^2.
 V3  |C(t)| -> 0 (no point spectrum in the cyclic subspace), at the Landau rate for Lambda = sqrt5 (-0.41 i root).
 V4  H kernel: -Im int_0^inf C dt = P int G0^2/(u F_M) du and Re int_0^inf C dt = pi G0(0)^2/F_M(0) (u = 0 slice).
 V5  0 < Lambda < 1: m(t) grows at the rate gamma solving 1 - sqrt(pi) gamma e^{gamma^2} erfc(gamma) = Lambda (Lambda = 0.5).
 V6  surjectivity (inverse G-transform): for c(u) = (1+u) e^{-0.8u^2}/(1+0.3u^2) (not Gaussian), G = eps_R c + eps_I H[c] has
     rho T[G] = c, with H by the Kress-Martensen odd-even rule (validated on a Gaussian).
Exit 1 on any failure.
"""
import sys
import time
import numpy as np
from scipy.integrate import quad
from scipy.special import erfcx, dawsn
from scipy.optimize import brentq

fails = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


t0 = time.time()
SQPI = np.sqrt(np.pi)
FM = lambda u: np.exp(-u * u) / SQPI

# velocity grid for transforms in u (trapezoid on [-11, 11]; integrands are entire and decay like Gaussians)
hu = 0.02
ug = np.arange(-11.0, 11.0 + hu / 2, hu)


def trap(f):
    return hu * (f.sum(axis=-1) - 0.5 * (f[..., 0] + f[..., -1]))


def G_a(u):
    return (1 + 0.8 * u - 0.5 * u**2 + 0.3 * u**3) * np.exp(-u**2 / 1.4) / SQPI


def G_b(u):
    return np.exp(-(u - 0.7)**2 / 0.9) - 0.6 * u * np.exp(-(u + 0.4)**2 / 0.5)


# ---------------------------------------------------------------------------------- V1 kernel
print("V1  Volterra kernel")
worst = 0
for Lam in (np.sqrt(5.0), -0.4):
    for tau in (0.3, 1.7, 4.0):
        kq = 1j / Lam * trap(ug * FM(ug) * np.exp(-1j * ug * tau))
        worst = max(worst, abs(kq - tau * np.exp(-tau**2 / 4) / (2 * Lam)))
check("k(tau) = (i/Lambda) int u F_M e^{-iu tau} du = tau e^{-tau^2/4}/(2 Lambda)", worst < 1e-13, f"max abs {worst:.1e}")


def volterra(G0, Lam, T, h):
    """Return t, m(t), C(t) by trapezoid product integration (O(h^2))."""
    n = int(round(T / h)) + 1
    t = np.arange(n) * h
    E = np.exp(-1j * np.outer(t, ug))                     # [t, u]
    mfree = trap(E * G0(ug)[None, :])
    Pt = trap(E * (G0(ug)**2 / FM(ug))[None, :])
    Rt = trap(E * (ug * G0(ug))[None, :])
    k = t * np.exp(-t**2 / 4) / (2 * Lam)                  # k(0) = 0: explicit scheme
    m = np.zeros(n, complex)
    m[0] = mfree[0]
    for i in range(1, n):
        s = 0.5 * k[i] * m[0] + np.dot(k[i - 1:0:-1], m[1:i])
        m[i] = mfree[i] + h * s
    # C(t) = P(t) + (i/Lambda) int_0^t R(t-t') m(t') dt' - M0[G0] m(t)/Lambda
    conv = np.zeros(n, complex)
    for i in range(1, n):
        conv[i] = h * (0.5 * Rt[i] * m[0] + np.dot(Rt[i - 1:0:-1], m[1:i]) + 0.5 * Rt[0] * m[i])
    C = Pt + 1j / Lam * conv - mfree[0] * m / Lam
    return t, m, C


def volterra_rich(G0, Lam, T, h):
    t1, m1, C1 = volterra(G0, Lam, T, h)
    t2, m2, C2 = volterra(G0, Lam, T, h / 2)
    return t1, (4 * m2[::2] - m1) / 3, (4 * C2[::2] - C1) / 3, np.abs(C2[::2] - C1).max()


# ---------------------------------------------------------------------------------- spectral side from definitions
def pv(f, u0, a=-14, b=14):
    """P int_a^b f(v)/(v - u0) dv (QUADPACK Cauchy weight)."""
    return quad(f, a, b, weight='cauchy', wvar=u0, limit=400, epsabs=1e-13, epsrel=1e-12)[0]


PVvFM = np.array([pv(lambda v: v * FM(v), u0) for u0 in ug])      # P int v F_M/(v-u) dv


def spectral_density(G0, Lam):
    epsR = 1 - PVvFM / Lam
    epsI = -np.pi * ug * FM(ug) / Lam
    PVvG = np.array([pv(lambda v: v * G0(v), u0) for u0 in ug])     # P int v G/(v-u)
    M0 = trap(G0(ug))
    # T[G](u) = [F_u, G] = (1/Lambda) P int v G/(v-u) + lambda(u) G(u)/F_M(u) - M0[G]/Lambda,  lambda = epsR
    FMT = PVvG * FM(ug) / Lam + epsR * G0(ug) - M0 * FM(ug) / Lam      # F_M T (stable at large |u|)
    rhoT2 = FMT**2 / (FM(ug) * (epsR**2 + epsI**2))
    return rhoT2, epsR, epsI


print("V2/V3  autocorrelation C(t) = Fourier transform of rho T^2")
T_end, h = 12.0, 0.01
for Lam in (np.sqrt(5.0), -np.sqrt(5.0), 5.0, -0.4, 50.0):
    for name, G0 in (("G_a", G_a), ("G_b", G_b)):
        t, m, C, rich = volterra_rich(G0, Lam, T_end, h)
        rhoT2, _, _ = spectral_density(G0, Lam)
        S = trap(np.exp(-1j * np.outer(t, ug)) * rhoT2[None, :])
        W = trap(G0(ug)**2 / FM(ug)) - trap(G0(ug))**2 / Lam          # [G0, G0]
        err = np.abs(C - S).max() / abs(W)
        check(f"Lambda={Lam:+.3f} {name}: [G0, e^(-itL) G0] = int rho T^2 e^(-iut) du, t in [0,12]", err < 1e-6,
              f"max |C-S|/[G0,G0] = {err:.1e} (Richardson correction {rich / abs(W):.1e}); Parseval {abs(C[0] - W) / abs(W):.1e},"
              f" {abs(S[0] - W) / abs(W):.1e}")
# decay at the Landau rate (Lambda = sqrt5): |C| ~ e^{-0.41 t} at late times
t, m, C, _ = volterra_rich(G_a, np.sqrt(5.0), 40.0, 0.01)
sel = (t > 20) & (t < 40)
rate = np.polyfit(t[sel], np.log(np.abs(m[sel])), 1)[0]
print(f"    Lambda=sqrt5: |m(t)| late-time decay rate {rate:.4f} (Landau root -0.409958 i); |C(40)|/|C(0)| = {abs(C[-1]) / abs(C[0]):.1e}")
check("V3 no point spectrum: C(t) -> 0, density decays at the Landau rate", abs(rate + 0.409958) < 2e-3 and abs(C[-1]) / abs(C[0]) < 1e-6)

print("V4  H kernel and the u = 0 slice from int_0^inf C(t) dt")
for Lam in (np.sqrt(5.0), -np.sqrt(5.0), 5.0, -0.4):
    for name, G0 in (("G_a", G_a), ("G_b", G_b)):
        t, m, C, _ = volterra_rich(G0, Lam, 60.0, 0.01)
        # Simpson's rule (the trapezoid rule has an O(h^2) endpoint error (h^2/12) C'(0), and C'(0) = -i[G0, L G0]
        # is purely imaginary, which would bias only the imaginary part)
        wS = np.ones(len(C)); wS[1:-1:2] = 4; wS[2:-1:2] = 2
        Iint = 0.01 / 3 * np.sum(wS * C)
        Itrap = 0.01 * (C.sum() - 0.5 * (C[0] + C[-1]))
        f = lambda v: G0(v)**2 / FM(v)
        Hpv = pv(f, 0.0)                                   # P int G0^2/(u F_M) du
        e1 = abs(-Iint.imag - Hpv) / abs(Hpv); e2 = abs(Iint.real - np.pi * G0(0.0)**2 / FM(0.0)) / (np.pi * G0(0.0)**2 / FM(0.0))
        check(f"Lambda={Lam:+.3f} {name}: -Im int C = P int G^2/(uF_M) (no Lambda term), Re int C = pi G(0)^2/F_M(0)",
              e1 < 1e-6 and e2 < 1e-6, f"rel {e1:.1e}, {e2:.1e}; |C(60)| = {abs(C[-1]):.1e}; trapezoid would give {abs(-Itrap.imag - Hpv) / abs(Hpv):.1e}")

print("V5  0 < Lambda < 1: unstable growth")
Lam = 0.5
gam = brentq(lambda g: 1 - SQPI * g * erfcx(g) - Lam, 1e-9, 50)
t, m, C, _ = volterra_rich(G_b, Lam, 30.0, 0.01)
sel = t > 15
rate = np.polyfit(t[sel], np.log(np.abs(m[sel])), 1)[0]
check("Lambda = 0.5: M0 grows at gamma = root of 1 - sqrt(pi) g e^{g^2} erfc(g) = Lambda", abs(rate - gam) < 1e-3,
      f"measured {rate:.5f}, gamma {gam:.5f}")

print("V6  surjectivity of the slice map (inverse G-transform)")
hk = 0.01
vg = np.arange(-30.0, 30.0 + hk / 2, hk)


def hilbert_km(f):
    """Morrison convention Hc[f](u_k) = (1/pi) P int f(v)/(v - u_k) dv on a uniform grid: odd-even rule
    P int f/(v-u_k) ~ 2 h sum_{j-k odd} f_j/(v_j - u_k)."""
    n = len(f)
    idx = np.arange(n)
    out = np.zeros(n)
    for k in range(n):
        j = idx[(idx - k) % 2 == 1]
        out[k] = 2 * hk * np.sum(f[j] / (vg[j] - vg[k])) / np.pi
    return out


gauss = np.exp(-vg**2)
Hg = hilbert_km(gauss)
ref = -2 * dawsn(vg) / np.sqrt(np.pi)                      # (1/pi) P int e^{-v^2}/(v-u) dv = -(2/sqrt(pi)) D(u)
sel = np.abs(vg) < 8
check("odd-even Hilbert rule reproduces the Gaussian's Hilbert transform", np.abs(Hg - ref)[sel].max() < 1e-10,
      f"max abs {np.abs(Hg - ref)[sel].max():.1e}")
c = (1 + vg) * np.exp(-0.8 * vg**2) / (1 + 0.3 * vg**2)
Hc = hilbert_km(c)
worst = 0
for Lam in (np.sqrt(5.0), -np.sqrt(5.0), 5.0):
    FMv = FM(vg)
    PVv = 2 * np.sqrt(np.pi) * 0 + np.pi * hilbert_km(vg * FMv)       # P int v F_M/(v-u) dv on the grid
    epsR = 1 - PVv / Lam; epsI = -np.pi * vg * FMv / Lam
    G = epsR * c + epsI * Hc
    HG = hilbert_km(G)
    M0 = hk * G.sum()
    PVvG = np.pi * hilbert_km(vg * G)
    FMT = PVvG * FMv / Lam + epsR * G - M0 * FMv / Lam
    rhoT = FMT / (epsR**2 + epsI**2)                       # rho T = F_M T/|eps|^2
    sel = np.abs(vg) < 4
    worst = max(worst, np.abs(rhoT - c)[sel].max() / np.abs(c).max())
check("rho T[eps_R c + eps_I H c] = c for a non-Gaussian c (|u| < 4, three Lambda)", worst < 1e-8, f"max rel {worst:.1e}")

print(f"wall time {time.time() - t0:.1f} s")
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL c03 CHECKS PASSED")
