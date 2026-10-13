"""s09: the advection lemmas used at finite k_perp rho_i, illustrated numerically: the triad formula for dQ/dt of a
quadratic form under v-independent and gyroaveraged advection, four cases that separate conserved from
non-conserved kernels, a toy model with a k-dependent Lambda_k in which the W- and Gamma-type forms change while the
H-type form (kernel 1/(u F_M)) does not, and the orders of the finite-k_perp rho_i corrections in the equations of
Chandran, Mallet & Meyrand (2026; CMM below). It supports Section 4 (finite k_perp rho_i) of the paper.

Lemma A (v-independent advection). theta_k(v) obeys d_t theta = -{chi, theta} with chi(r) arbitrary. For
Q = sum_k theta_k^dag K_k theta_k (K_k Hermitian on velocity space, K_{-k} = conj K_k):
   for theta supported on {+-q, +-(q+p)} and chi on {+-p}:   dQ/dt = 4 (p x q)_z Re[ chi_p theta_{q+p}^dag (K_{q+p} - K_q) theta_q ],
so dQ/dt = 0 for all states and flows iff K_{q+p} = K_q whenever p x q != 0, i.e. iff K_k does not depend on k.
Lemma A' (gyroaveraged advection). If velocity v is advected by J0(k_perp v_perp) chi_k (multiplication M_p in v for each
Fourier mode p of chi), then dQ/dt = 4 (p x q)_z Re[ chi_p theta_{q+p}^dag (K_{q+p} M_p - M_p K_q) theta_q ]; for a
k-independent K the condition is [K, M_p] = 0 for all p: K must act locally in v_perp; a multiplication operator in v always works.
Lemma B. P0 L^{-1} = 1/u for every Lambda (s02 O3), so the H kernel does not change if Lambda becomes k-dependent, while the
kernels of W (P0) and Gamma (P0 L) contain 1/Lambda.
Checks:
 K1  triad formula for dQ/dt (Lemma A and A') against a direct pseudo-spectral evaluation.
 K2  four cases on a 2D grid: (a) K const, flat advection: conserved; (b) K_k = K0 + |k|^2 K1: not conserved;
     (c) K = diag(w(v)), gyroaveraged advection: conserved; (d) K const but nonlocal in v_perp, gyroaveraged: not conserved.
 K3  toy compressive model with a k-dependent Lambda_k (illustration only, not gyrokinetics): with v-independent advection
     the W-form and Gamma-form change, the H-form (kernel 1/(u F_M)) does not.
 K4  orders of the finite-k_perp rho_i corrections in CMM's ITEF equations (series of J0, 2J1/a, Gamma0 = I0 e^{-alpha}).
Exit 1 on failure.
"""
import sys
import numpy as np
import sympy as sp
from scipy.special import j0

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


N = 32
k1 = np.fft.fftfreq(N, d=1.0 / N)
KX, KY = np.meshgrid(k1, k1, indexing='ij')
KK = np.sqrt(KX**2 + KY**2)
keep = (np.abs(KX) < N / 3) & (np.abs(KY) < N / 3)            # 2/3-rule Galerkin set


def bracket_hat(chi_hat, th_hat):
    """Fourier coefficients (normalised: f(r) = sum f_k e^{ikr}) of {chi, theta}, projected on the Galerkin set."""
    dx = lambda f: np.fft.ifft2(1j * KX * f * N * N).real
    dy = lambda f: np.fft.ifft2(1j * KY * f * N * N).real
    b = dx(chi_hat) * dy(th_hat) - dy(chi_hat) * dx(th_hat)
    return np.fft.fft2(b) / (N * N) * keep


def dQdt(chi_hat, th_hat, Kfun, gyro_vperp=None, with_scale=False):
    """dQ/dt = 2 Re sum_k theta_k^dag K_k (d_t theta)_k, d_t theta_v = -{chi_v, theta_v};
    th_hat [nv, N, N]; Kfun(k) -> nv x nv Hermitian; gyro_vperp: array of v_perp (then chi_v = J0(k v_perp) chi)."""
    nv = th_hat.shape[0]
    dth = np.zeros_like(th_hat)
    for i in range(nv):
        ch = chi_hat if gyro_vperp is None else chi_hat * j0(KK * gyro_vperp[i])
        dth[i] = -bracket_hat(ch, th_hat[i])
    tot = 0.0
    sc = 0.0
    for ix in range(N):
        for iy in range(N):
            if not keep[ix, iy]:
                continue
            Kk = Kfun(KX[ix, iy], KY[ix, iy])
            term = 2 * np.real(np.conj(th_hat[:, ix, iy]) @ Kk @ dth[:, ix, iy])
            tot += term
            sc += 2 * np.abs(np.conj(th_hat[:, ix, iy])) @ np.abs(Kk) @ np.abs(dth[:, ix, iy])
    return (tot, sc) if with_scale else tot


def real_field(nv, rng, kmax=8):
    out = np.zeros((nv, N, N), complex)
    for i in range(nv):
        f = rng.standard_normal((N, N))
        fh = np.fft.fft2(f) / (N * N)
        fh *= (KK <= kmax) * keep
        out[i] = fh
    return out


rng = np.random.default_rng(2)
nv = 4
vperp = np.array([0.3, 0.8, 1.4, 2.1])

print("K1  triad formula")
q = (2, 1); p = (1, 3)
idx = lambda k: (k[0] % N, k[1] % N)
th = np.zeros((nv, N, N), complex)
a1 = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
a2 = rng.standard_normal(nv) + 1j * rng.standard_normal(nv)
qp = (q[0] + p[0], q[1] + p[1])
th[:, idx(q)[0], idx(q)[1]] = a1; th[:, idx((-q[0], -q[1]))[0], idx((-q[0], -q[1]))[1]] = np.conj(a1)
th[:, idx(qp)[0], idx(qp)[1]] = a2; th[:, idx((-qp[0], -qp[1]))[0], idx((-qp[0], -qp[1]))[1]] = np.conj(a2)
chi = np.zeros((N, N), complex)
cp_ = 0.7 - 0.4j
chi[idx(p)] = cp_; chi[idx((-p[0], -p[1]))] = np.conj(cp_)
B0 = rng.standard_normal((nv, nv)); K0 = B0 + B0.T
B1 = rng.standard_normal((nv, nv)); K1 = B1 + B1.T
Kfun_b = lambda kx, ky: K0 + (kx**2 + ky**2) / 10 * K1
pxq = p[0] * q[1] - p[1] * q[0]
Kq = Kfun_b(*q); Kqp = Kfun_b(*qp)
formula = 4 * pxq * np.real(cp_ * np.conj(a2) @ (Kqp - Kq) @ a1)
direct = dQdt(chi, th, Kfun_b)
check("dQ/dt = 4 (p x q) Re[chi_p theta_{q+p}^dag (K_{q+p} - K_q) theta_q]", abs(direct - formula) < 1e-12 * max(1, abs(formula)),
      f"direct {direct:.12f}, formula {formula:.12f}")
Mp = np.diag(j0(np.hypot(*p) * vperp))
formula_g = 4 * pxq * np.real(cp_ * np.conj(a2) @ (Kqp @ Mp - Mp @ Kq) @ a1)
direct_g = dQdt(chi, th, Kfun_b, gyro_vperp=vperp)
check("gyroaveraged: dQ/dt = 4 (p x q) Re[chi_p theta_{q+p}^dag (K_{q+p} M_p - M_p K_q) theta_q]",
      abs(direct_g - formula_g) < 1e-12 * max(1, abs(formula_g)), f"direct {direct_g:.12f}, formula {formula_g:.12f}")

print("K2  four cases on a random state")
th = real_field(nv, rng)
chi = real_field(1, rng)[0] * 5
norm = sum(np.real(np.conj(th[:, ix, iy]) @ th[:, ix, iy]) for ix in range(N) for iy in range(N))
Kconst = lambda kx, ky: K0
Kdiag = lambda kx, ky: np.diag([1.0, 2.0, 0.5, 3.0])
ca = dQdt(chi, th, Kconst); cb = dQdt(chi, th, Kfun_b)
cc = dQdt(chi, th, Kdiag, gyro_vperp=vperp); cd = dQdt(chi, th, Kconst, gyro_vperp=vperp)
scale = abs(cb)
print(f"    (a) {ca:+.3e}  (b) {cb:+.3e}  (c) {cc:+.3e}  (d) {cd:+.3e}")
check("(a) K const, flat advection: dQ/dt = 0", abs(ca) < 1e-12 * scale)
check("(b) K_k = K0 + |k|^2 K1/10: dQ/dt != 0", abs(cb) > 1e-3 * norm)
check("(c) K = diag(w(v)), gyroaveraged advection: dQ/dt = 0", abs(cc) < 1e-12 * scale)
check("(d) K const, nonlocal in v_perp, gyroaveraged advection: dQ/dt != 0", abs(cd) > 1e-3 * abs(cb))

print("K3  toy model with k-dependent Lambda_k (illustration)")
# velocity grid in u (parallel) only; advection v-independent; forms built with Lambda_k
nu_ = 12
ug = np.linspace(-2.75, 2.75, nu_)                     # uniform grid, even count: u = 0 excluded
FMg = np.exp(-ug**2) / np.sqrt(np.pi)
wq = np.full(nu_, ug[1] - ug[0])                        # int f du ~ sum wq_i f_i (quadrature details irrelevant here)
Lam0 = np.sqrt(5.0)
LamK = lambda kx, ky: Lam0 * (1 + 0.3 * (kx**2 + ky**2) / 10)
one = np.ones(nu_)


def KW(kx, ky):
    Lk = LamK(kx, ky)
    return np.diag(wq / FMg) - np.outer(wq, wq) / Lk          # [f,g] = sum wq f g/F_M - (sum wq f)(sum wq g)/Lambda_k


def KGam(kx, ky):
    Lk = LamK(kx, ky)
    M0 = np.outer(wq, one); U = np.diag(ug)
    # [G, L G] = int u G^2/F_M - 2 M0 M1/Lambda
    return np.diag(wq * ug / FMg) - (np.outer(wq, wq * ug) + np.outer(wq * ug, wq)) / Lk


KH = lambda kx, ky: np.diag(wq / (ug * FMg))
thu = real_field(nu_, rng)
chi = real_field(1, rng)[0] * 5
dW, sW = dQdt(chi, thu, KW, with_scale=True); dG, sG = dQdt(chi, thu, KGam, with_scale=True)
dH, sH = dQdt(chi, thu, KH, with_scale=True)
dW0, sW0 = dQdt(chi, thu, lambda kx, ky: np.diag(wq / FMg) - np.outer(wq, wq) / Lam0, with_scale=True)
print(f"    dW/dt = {dW:+.3e}, dGamma/dt = {dG:+.3e}, dH/dt = {dH:+.3e}; control with constant Lambda: dW/dt = {dW0:+.3e}")
check("k-dependent Lambda_k: W-form and Gamma-form not conserved by advection", abs(dW) > 1e-10 * sW and abs(dG) > 1e-10 * sG,
      f"|dW|/scale = {abs(dW) / sW:.1e}, |dGamma|/scale = {abs(dG) / sG:.1e}")
check("H-form (kernel 1/(u F_M), Lambda-free) conserved; W-form conserved when Lambda is constant",
      abs(dH) < 1e-13 * sH and abs(dW0) < 1e-13 * sW0, f"|dH|/scale = {abs(dH) / sH:.1e}, |dW0|/scale = {abs(dW0) / sW0:.1e}")

print("K4  orders of the finite-k_perp rho_i corrections (CMM ITEF, eq:J0J1, eq:Gamma0)")
a, al = sp.symbols('a alpha', positive=True)
J0s = sp.series(sp.besselj(0, a), a, 0, 4).removeO()
J1s = sp.series(2 * sp.besselj(1, a) / a, a, 0, 4).removeO()
G0s = sp.series(sp.besseli(0, al) * sp.exp(-al), al, 0, 2).removeO()
G1s = sp.series(-sp.diff(sp.besseli(0, al) * sp.exp(-al), al), al, 0, 2).removeO()
check("J0(a) = 1 - a^2/4 + ...", sp.simplify(J0s - (1 - a**2 / 4)) == 0, str(J0s))
check("2 J1(a)/a = 1 - a^2/8 + ...", sp.simplify(J1s - (1 - a**2 / 8)) == 0, str(J1s))
check("Gamma0 = I0 e^-alpha = 1 - alpha + ..., Gamma1 = 1 - 3 alpha/2 + ... (alpha = k^2 rho^2/2)",
      sp.simplify(G0s - (1 - al)) == 0 and sp.simplify(G1s - (1 - 3 * al / 2)) == 0, f"{G0s}; {G1s}")
print("    KRMHD ordering phi ~ (dB/B)/(k_perp rho_i) (CMM sec. 3.2): the advection of g by vperp^2 J1 dB/B in CMM eq:dgdt")
print("    is O(k_perp rho_i) relative to E x B advection; gyroaveraging corrections are O(k_perp^2 rho_i^2).")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s09 CHECKS PASSED")
