"""s03: the plasma dispersion function Z in the normalisation used here and the dielectric function
eps(zeta) = 1 - (1 + zeta Z)/Lambda of one branch (mpmath): the Kramers-Kronig relation, Nyquist winding numbers
and an argument-principle count showing no zero in the upper half plane unless 0 < Lambda < 1, the unstable root
for 0 < Lambda < 1, the degenerate case Lambda = 1, the least-damped Landau roots at Lambda = +-sqrt5, and the
correspondence with the Vlasov-Poisson and Jeans dielectric functions. It supports Section 3 and Appendix B of the
paper (the correspondence is the remark in Section 2).

Z(zeta) = int F_M(x)/(x - zeta) dx with F_M = e^{-x^2}/sqrt(pi) (Im zeta > 0) = i sqrt(pi) e^{-zeta^2} erfc(-i zeta).
eps(zeta) = 1 - (1/Lambda) int u F_M/(u - zeta) du = 1 - (1 + zeta Z(zeta))/Lambda.
Checks:
 D1  Z normalisation: closed form vs direct quadrature at points of the upper half plane.
 D2  real axis: P int F_M/(x-u) dx = -2 Dawson(u) = Re Z(u+i0); Im Z(u+i0) = sqrt(pi) e^{-u^2} = pi F_M(u).
 D3  Z' = -2(1 + zeta Z); 1 + zeta Z = -1/(2 zeta^2) - 3/(4 zeta^4) + O(zeta^-6) for |zeta| -> inf in Im zeta >= 0.
 D4  eps by quadrature = 1 - (1 + zeta Z)/Lambda; eps -> 1 at infinity.
 D5  eps_R - 1 = Hc[eps_I] (Kramers-Kronig / Morrison's identity), eps_I = -pi u F_M/Lambda.
 D6  Nyquist curve R(u) = 1 + u Z(u+i0): Im R = pi u F_M(u); real-axis crossings only at u = 0 (R = 1) and u -> +-inf (R -> 0);
     winding number of eps(u+i0) about 0 on a Lambda grid: 1 for 0 < Lambda < 1, else 0.
 D7  argument principle: number of zeros of eps in a large box of the upper half plane (contour integral of eps'/eps).
 D8  0 < Lambda < 1: the zero is zeta = i gamma, 1 - sqrt(pi) gamma e^{gamma^2} erfc(gamma) = Lambda; the left side decreases
     monotonically from 1 to 0, so the root is unique; truncated ladder matrices converge to +-i gamma.
 D9  Lambda = 1: eps(0 + i0) = 0, eps(u+i0) = -i sqrt(pi) u + 2 u^2 + O(u^3), rho ~ 1/(pi^{3/2} u^2) (not integrable).
 D10 Landau roots (zeros of the continuation of eps into Im zeta < 0) at Lambda = +-sqrt5 and Lambda -> 1+, where
     zeta_L = -i (delta/sqrt(pi)) [1 - 2 delta/pi + O(delta^2)] for Lambda = 1 + delta.
 D11 Vlasov-Poisson / Jeans correspondence: with F0 Maxwellian, eps_VP = 1 + (1 + zeta Z)/(k lambda_D)^2, i.e. Lambda = -(k lambda_D)^2;
     self-gravitating: Lambda = (k/k_J)^2.
Exit 1 on failure.
"""
import os
import sys
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_common import ladder, eR as eR_np, eI as eI_np, SQRT5

mp.mp.dps = 30
fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


FMm = lambda x: mp.e**(-x**2) / mp.sqrt(mp.pi)
Zm = lambda z: 1j * mp.sqrt(mp.pi) * mp.e**(-z**2) * mp.erfc(-1j * z)
dawson = lambda x: mp.sqrt(mp.pi) / 2 * mp.e**(-x**2) * mp.erfi(x)
epsm = lambda z, L: 1 - (1 + z * Zm(z)) / L


def pv_int(f, u0):
    """P int f(v)/(v - u0) dv = int_0^inf [f(u0+t) - f(u0-t)]/t dt."""
    return mp.quad(lambda t: (f(u0 + t) - f(u0 - t)) / t, [0, 1, 4, mp.inf])


print("D1  Z normalisation")
worst = 0
for z in (mp.mpc(0.3, 0.7), mp.mpc(-1.2, 0.4), mp.mpc(2.0, 0.15), mp.mpc(0, 2.5), mp.mpc(-0.05, 0.05)):
    direct = mp.quad(lambda x: FMm(x) / (x - z), [-mp.inf, mp.re(z) - 1, mp.re(z), mp.re(z) + 1, mp.inf])
    worst = max(worst, abs(direct - Zm(z)) / abs(Zm(z)))
check("int F_M/(x - zeta) = i sqrt(pi) e^{-zeta^2} erfc(-i zeta)", worst < 1e-20, f"max rel err {mp.nstr(worst, 3)}")
check("Z(0) = i sqrt(pi)", abs(Zm(mp.mpf(0)) - 1j * mp.sqrt(mp.pi)) < 1e-28)

print("D2  real axis")
w1 = w2 = 0
for u0 in (mp.mpf('-2.3'), mp.mpf('-0.4'), mp.mpf('0.1'), mp.mpf('1.7'), mp.mpf('3.2')):
    pv = pv_int(FMm, u0)
    w1 = max(w1, abs(pv + 2 * dawson(u0)))
    zb = Zm(u0 + 1e-25j)
    w2 = max(w2, abs(mp.re(zb) + 2 * dawson(u0)) + abs(mp.im(zb) - mp.pi * FMm(u0)))
check("P int F_M/(x-u) dx = -2 Dawson(u)", w1 < 1e-20, mp.nstr(w1, 3))
check("Z(u + i0) = -2 Dawson(u) + i pi F_M(u)", w2 < 1e-20, mp.nstr(w2, 3))

print("D3  derivative and asymptotics")
w = 0
for z in (mp.mpc(0.3, 0.7), mp.mpc(-1.2, -0.4), mp.mpc(2.0, 0.15)):
    w = max(w, abs(mp.diff(Zm, z) + 2 * (1 + z * Zm(z))))
check("Z' = -2(1 + zeta Z)", w < 1e-20, mp.nstr(w, 3))
w = 0
for z in (mp.mpc(30, 0.0), mp.mpc(0, 30), mp.mpc(-21, 21), mp.mpc(40, 1e-20)):
    zz = z if mp.im(z) > 0 else z + 1e-30j
    approx = -1 / (2 * zz**2) - 3 / (4 * zz**4)
    w = max(w, abs((1 + zz * Zm(zz)) - approx) * abs(zz)**6)
check("|1 + zeta Z + 1/(2 zeta^2) + 3/(4 zeta^4)| |zeta|^6 bounded (<= 15/8 + small) on Im zeta >= 0", w < 2.0, mp.nstr(w, 4))

print("D4  dielectric function")
w = 0
for L in (mp.sqrt(5), -mp.sqrt(5), mp.mpf('0.5')):
    for z in (mp.mpc(0.3, 0.7), mp.mpc(-1.2, 0.4)):
        direct = 1 - mp.quad(lambda x: x * FMm(x) / (x - z), [-mp.inf, mp.re(z), mp.inf]) / L
        w = max(w, abs(direct - epsm(z, L)))
check("eps(zeta) = 1 - (1/Lambda) int u F_M/(u-zeta) du = 1 - (1 + zeta Z)/Lambda", w < 1e-20, mp.nstr(w, 3))
check("eps -> 1 at |zeta| -> inf (Im zeta > 0)", abs(epsm(mp.mpc(0, 200), mp.sqrt(5)) - 1) < 1e-4)

print("D5  eps_R - 1 = Hc[eps_I]")
w = 0
for L in (mp.sqrt(5), -mp.sqrt(5)):
    eI = lambda x: -mp.pi * x * FMm(x) / L
    for u0 in (mp.mpf('-1.3'), mp.mpf('0.2'), mp.mpf('2.4')):
        H = pv_int(eI, u0) / mp.pi
        eRv = 1 - (1 + u0 * (-2 * dawson(u0))) / L
        w = max(w, abs(eRv - 1 - H))
check("eps_R(u) - 1 = (1/pi) P int eps_I(v)/(v-u) dv", w < 1e-20, mp.nstr(w, 3))

print("D6  Nyquist curve and winding numbers")
uu = np.concatenate([-np.logspace(2, -6, 200001), [0.0], np.logspace(-6, 2, 200001)])
Rre = 1 + uu * (-2 * __import__('scipy').special.dawsn(uu))
Rim = np.pi * uu * np.exp(-uu**2) / np.sqrt(np.pi)
inner = np.abs(uu) < 25   # beyond |u| ~ 27 F_M underflows to 0 in float64; the sign statement is analytic there
sgn_ok = np.all(Rim[inner & (uu > 0)] > 0) and np.all(Rim[inner & (uu < 0)] < 0)
check("Im R(u) = pi u F_M(u) has the sign of u (crossings only at u = 0 and +-inf)", bool(sgn_ok))
check("R(0) = 1, R(+-100) ~ -1/(2u^2) -> 0", abs(Rre[200001] - 1) < 1e-15 and abs(Rre[0] + 1 / (2 * 100.0**2)) < 1e-6)
okw = True
rows = []
for L in (-5.0, -SQRT5, -0.5, -0.01, 0.01, 0.2, 0.5, 0.9, 0.999, 1.001, 1.5, SQRT5, 10.0):
    er = eR_np(uu, L); ei = eI_np(uu, L)
    ph = np.unwrap(np.arctan2(ei, er))
    wind = (ph[-1] - ph[0]) / (2 * np.pi)
    expect = 1 if 0 < L < 1 else 0
    rows.append(f"{L:+.4f}:{wind:+.6f}")
    okw &= abs(wind - expect) < 1e-6
print("    winding of eps(u+i0) on [-100,100]: " + ", ".join(rows))
check("winding = 1 for 0 < Lambda < 1, 0 otherwise", okw)

print("D7  argument principle in the box [-6,6] x [1e-3, 6]")


def count_zeros(L, X=6, y0=mp.mpf('1e-3'), Y=6):
    f = lambda z: epsm(z, L)
    fp = lambda z: -(Zm(z) - 2 * z * (1 + z * Zm(z))) / L      # eps' = -(Z + zeta Z')/Lambda
    g = lambda z: fp(z) / f(z)
    corners = [mp.mpc(-X, y0), mp.mpc(X, y0), mp.mpc(X, Y), mp.mpc(-X, Y), mp.mpc(-X, y0)]
    tot = 0
    for a, b in zip(corners[:-1], corners[1:]):
        tot += mp.quad(lambda t: g(a + (b - a) * t) * (b - a), [0, 0.25, 0.5, 0.75, 1])
    return tot / (2j * mp.pi)


okc = True
rows = []
for L in (mp.sqrt(5), -mp.sqrt(5), mp.mpf(3), mp.mpf('-0.3'), mp.mpf('1.5'), mp.mpf('0.2'), mp.mpf('0.5'), mp.mpf('0.9')):
    n = count_zeros(L)
    expect = 1 if 0 < L < 1 else 0
    rows.append(f"{float(L):+.3f}: {float(mp.re(n)):+.10f} {float(mp.im(n)):+.1e}i")
    okc &= abs(n - expect) < 1e-8
print("    zeros counted: " + ", ".join(rows))
check("one zero in the upper half plane for 0 < Lambda < 1, none otherwise", okc)

print("D8  unstable root for 0 < Lambda < 1")
f_g = lambda g: 1 - mp.sqrt(mp.pi) * g * mp.e**(g**2) * mp.erfc(g)
check("1 + i gamma Z(i gamma) = 1 - sqrt(pi) gamma e^{gamma^2} erfc(gamma) (real)",
      abs((1 + 1j * mp.mpf('0.7') * Zm(1j * mp.mpf('0.7'))) - f_g(mp.mpf('0.7'))) < 1e-25)
grid = [mp.mpf(x) for x in np.linspace(1e-3, 30, 3001)]
vals = [f_g(g) for g in grid]
check("f(gamma) strictly decreasing on (0, 30], f(0+) = 1, f -> 0",
      all(vals[i + 1] < vals[i] for i in range(len(vals) - 1)) and abs(f_g(mp.mpf('1e-12')) - 1) < 1e-11 and vals[-1] < 1e-3)
check("f(gamma) ~ 1/(2 gamma^2) at large gamma", abs(f_g(mp.mpf(200)) * 2 * 200**2 - 1) < 1e-4)
okm = True
for Lv, ref in ((0.01, 6.966267), (0.2, 1.175216), (0.5, 0.432752), (0.9, 0.060314), (0.999, 0.000565)):
    g = mp.findroot(lambda g: f_g(g) - Lv, ref)
    e0 = abs(epsm(1j * g, mp.mpf(Lv)))
    msg = f"Lambda={Lv}: gamma={mp.nstr(g, 10)}, |eps(i gamma)|={mp.nstr(e0, 2)}"
    okm &= e0 < 1e-20 and abs(g - ref) < 2e-6
    if Lv in (0.2, 0.5, 0.9):
        ims = []
        for Mv in (64, 128, 256, 512):
            ims.append(np.abs(np.linalg.eigvals(ladder(Mv, Lv)).imag).max())
        msg += "; max Im eig A (M=64,128,256,512): " + ", ".join(f"{x:.6f}" for x in ims)
        if Lv in (0.2, 0.5):
            okm &= abs(ims[-1] - float(g)) < 1e-5
    print("    " + msg)
check("unstable root zeta = i gamma (reference values) and truncated-matrix convergence (Lambda = 0.2, 0.5)", okm)

print("D9  Lambda = 1")
check("eps(0 + i0) = 0 at Lambda = 1", abs(epsm(mp.mpc(0, 1e-25), 1)) < 1e-20)
w = 0
for u0 in (mp.mpf('1e-3'), mp.mpf('-2e-3'), mp.mpf('5e-4')):
    e = epsm(u0 + 1e-28j, 1)
    w = max(w, abs(e - (-1j * mp.sqrt(mp.pi) * u0 + 2 * u0**2)) / abs(u0)**3)
check("eps(u+i0) = -i sqrt(pi) u + 2 u^2 + O(u^3) at Lambda = 1", w < 10, f"|remainder|/|u|^3 <= {mp.nstr(w, 3)}")
u0 = mp.mpf('1e-4')
e = epsm(u0 + 1e-28j, 1)
rho0 = FMm(u0) / abs(e)**2
check("rho(u) u^2 -> 1/pi^{3/2} at Lambda = 1 (non-integrable)", abs(rho0 * u0**2 * mp.pi**1.5 - 1) < 1e-3, mp.nstr(rho0 * u0**2 * mp.pi**1.5, 8))

print("D10 Landau roots (zeros of the continuation into Im zeta < 0)")
r1 = mp.findroot(lambda z: epsm(z, mp.sqrt(5)), mp.mpc(0, -0.4))
r2 = mp.findroot(lambda z: epsm(z, -mp.sqrt(5)), mp.mpc(1.24, -0.84))
print(f"    Lambda=+sqrt5: {mp.nstr(r1, 10)};  Lambda=-sqrt5: {mp.nstr(r2, 10)}")
check("Lambda=+sqrt5 least-damped root -0.409958 i (reference value)", abs(r1 - mp.mpc(0, -0.409958)) < 2e-6)
check("Lambda=-sqrt5 least-damped roots +-1.242247 - 0.835300 i (reference values)", abs(r2 - mp.mpc(1.242247, -0.835300)) < 2e-6)
okl = True
ok2 = True
for d in (mp.mpf('1e-2'), mp.mpf('1e-3'), mp.mpf('1e-4')):
    rz = mp.findroot(lambda z: epsm(z, 1 + d), mp.mpc(0, -d / mp.sqrt(mp.pi)))
    rel = abs(rz - mp.mpc(0, -d / mp.sqrt(mp.pi))) / (d / mp.sqrt(mp.pi))
    okl &= rel < 2 * d
    # second order: zeta_L = -i (delta/sqrt(pi)) [1 - 2 delta/pi + O(delta^2)]  (Appendix B)
    c2 = (rz / mp.mpc(0, -d / mp.sqrt(mp.pi)) - 1) / d
    ok2 &= abs(c2 + 2 / mp.pi) < d
    print(f"    Lambda = 1 + {mp.nstr(d, 2)}: root {mp.nstr(rz, 8)}, rel. dev. from -i(Lambda-1)/sqrt(pi) = {mp.nstr(rel, 3)}; "
          f"(root/lead - 1)/delta = {mp.nstr(mp.re(c2), 6)} (-2/pi = {mp.nstr(-2 / mp.pi, 6)})")
check("Lambda -> 1+: zeta_L = -i (Lambda - 1)/sqrt(pi) (1 + O(Lambda - 1))  [Schekochihin et al. 2009 eq. 190 with Lambda+ - 1 = 1/beta]", okl)
check("Lambda = 1 + delta: zeta_L = -i (delta/sqrt(pi)) [1 - 2 delta/pi + O(delta^2)]  (|(root/lead - 1)/delta + 2/pi| < delta)", ok2)

print("D11 Vlasov-Poisson / Jeans correspondence")
w = 0
for z in (mp.mpc(0.4, 0.3), mp.mpc(-1.1, 0.8)):
    dFM = lambda x: -2 * x * FMm(x)
    I = mp.quad(lambda x: dFM(x) / (x - z), [-mp.inf, mp.re(z), mp.inf])   # int F_M'/(x - zeta)
    for kl2 in (mp.mpf('0.3'), mp.sqrt(5)):
        # VP (repulsive): eps = 1 - (1/(2 k^2 lambda_D^2)) int F_M'/(x - zeta) dx   (u = v/v_t, lambda_D^2 = v_t^2/(2 w_p^2))
        epsVP = 1 - I / (2 * kl2)
        w = max(w, abs(epsVP - epsm(z, -kl2)))
        # gravitational: sign of the coupling reversed, (k lambda_D)^2 -> -(k/k_J)^2
        epsJ = 1 + I / (2 * kl2)
        w = max(w, abs(epsJ - epsm(z, kl2)))
check("eps_VP(k lambda_D) = eps(Lambda = -(k lambda_D)^2); eps_Jeans(k/k_J) = eps(Lambda = (k/k_J)^2)", w < 1e-20, mp.nstr(w, 3))

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s03 CHECKS PASSED")
