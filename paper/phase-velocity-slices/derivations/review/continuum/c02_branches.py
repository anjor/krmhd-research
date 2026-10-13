"""c02 (independent check): the KRMHD compressive sector in the two-branch form of Appendix B of Chandran, Mallet &
Meyrand (2026; CMM below), re-derived by quadrature in v_perp and v_par applied to CMM's equations as printed: the
coupling matrix K, the branches and Lambda+- on random (beta_i, tau/Z), the Laguerre l >= 2 sector, the free-energy
and H decompositions with the corrected kappa (and their failure with (B16) as printed), the factor 1/2 in
Schekochihin et al. (2009) eqs (202)-(203), the moments u_par, dp_perp, dp_par, and the golden-ratio forms at
beta_i = tau = Z = 1. It supports Section 2 and Appendix A of the paper.

Route: explicit quadrature in x = vperp^2/vth^2 (Gauss-Laguerre nodes, weight e^{-x}) and in v_par (Gauss-Hermite or
fine trapezoid), applied to CMM's equations as printed in the LaTeX source:
  (3.25) dg/dt + v grad_par[g + ((Z/tau) n + x b) F0] = 0      (g = g_09, n = dn/n, b = dB/B)
  (3.27) mean(g) = n - b,   (3.28) mean(x g) = -(Z/tau) n - 2(1 + 1/beta) b
  (B2) G_n = -D^-1 (2pi/n) int dvperp vperp [x - 2(1+1/beta)] g,  (B3) G_B = -D^-1 (2pi/n) int dvperp vperp (x + Z/tau) g,
  D = Z/tau + 2(1 + 1/beta);  (B5) G+, G-;  (B6) sigma;  (B7) Lambda+-;  (B13)-(B19);  (3.31) W_compr with g_i = g + x b F0;
  (3.33) H_ph-sp with g_09.
Nothing is imported from the derivation scripts in ../../continuum. Checks:
 BR1 the v_perp moments of the source define K numerically; 1/eig(K) = Lambda+- of (B7) and the left eigenvectors are (B5)
     with sigma of (B6), on 400 random (beta, tau/Z) in [1e-3, 1e3]^2; (B1) int G_n = n, int G_B = b from (3.27)-(3.28).
 BR2 Lambda+ > 1 and Lambda- < 0 on the same sample; beta = tau = Z = 1 gives +-sqrt5 exactly (SymPy).
 BR3 Laguerre l >= 2 projection of the source vanishes (l = 2..12): the l >= 2 sector is the Lambda = infinity branch.
 BR4 W_compr (3.31) by direct 2D velocity quadrature of a random g equals W~ + c+ W+ + c- W- with (B15) coefficients
     only if kappa = sqrt((1+tau/Z)^2 + beta^-2); with (B16) as printed it fails.  Same for H_ph-sp (3.33) and (B17).
 BR5 dn/n and dB/B from M0+- (exact inversion) vs Schekochihin et al. (2009) eqs (202)-(203) (factor 1/2); u_par,
     dp_perp, dp_par by direct velocity integrals of g_i vs the moment formulas of Appendix A; beta = tau = Z = 1
     golden-ratio forms.
Exit 1 on any failure.
"""
import sys
import numpy as np
import sympy as sp
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.hermite import hermgauss
from scipy.special import eval_laguerre

fails = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


xq, wq = laggauss(40)            # int_0^inf e^{-x} f(x) dx = sum wq f(xq)
rng = np.random.default_rng(7)


def Kmatrix(beta, r):
    """r = tau/Z.  Source S(x) = (Z/tau) n + x b enters as v grad[ ... + S F_M].  For weight w(x) the equation of
    G_w = -D^-1 sum wq w(x) ghat is dG_w/dt + v grad G_w = v F_M grad [D^-1 sum wq w(x) S(x)]."""
    Zt = 1.0 / r
    D = Zt + 2 * (1 + 1 / beta)
    wn = xq - 2 * (1 + 1 / beta)          # (B2) weight
    wB = xq + Zt                          # (B3) weight
    # columns: coefficient of n and of b in D^-1 sum wq w S
    K = np.array([[(wq * wn * Zt).sum(), (wq * wn * xq).sum()],
                  [(wq * wB * Zt).sum(), (wq * wB * xq).sum()]]) / D
    # sign: G_w = -D^-1 sum w ghat, so its source is -D^-1 sum w S F_M inside grad_par with a + sign on the left:
    # dG_w/dt + v grad G_w + v grad(-D^-1 sum w S F_M) = 0  =>  RHS = + v F_M grad(D^-1 sum w S)
    return K


def cmm(beta, r):
    q = np.sqrt((1 + r)**2 + beta**-2)
    Lp = -r + 1 / beta + q; Lm = -r + 1 / beta - q
    sigma = 1 + r + 1 / beta + q
    lp = np.array([(1 + 1 / r) / sigma, 1.0])            # G+ = G_B + sigma^-1 (1 + Z/tau) G_n  -> (coef G_n, coef G_B)
    lm = np.array([1.0, 2 * r / (sigma * beta)])         # G- = G_n + 2 tau/(sigma Z beta) G_B
    return Lp, Lm, lp, lm, q, sigma


print("BR1/BR2  K from velocity quadrature; (B5)-(B7)")
worst_L = worst_v = 0.0
ok_range = True
for _ in range(400):
    beta = 10 ** rng.uniform(-3, 3); r = 10 ** rng.uniform(-3, 3)
    K = Kmatrix(beta, r)
    ev, VL = np.linalg.eig(K.T)                           # left eigenvectors of K
    Lp, Lm, lp, lm, q, sigma = cmm(beta, r)
    Ls = np.sort(1 / ev.real)
    worst_L = max(worst_L, abs(Ls[1] - Lp) / abs(Lp), abs(Ls[0] - Lm) / abs(Lm))
    for target, L in ((lp, Lp), (lm, Lm)):
        res = target @ K - target / L                      # left-eigenvector residual
        worst_v = max(worst_v, np.abs(res).max() / np.abs(target @ K).max())
    ok_range &= (Lp > 1) and (Lm < 0)
check("1/eig(K) = Lambda+- of (B7) (400 random beta, tau/Z)", worst_L < 1e-10, f"max rel {worst_L:.1e}")
check("(B5) with sigma (B6) are left eigenvectors: l K = l/Lambda", worst_v < 1e-10, f"max rel {worst_v:.1e}")
check("Lambda+ > 1 and Lambda- < 0 on the sample", ok_range)
b_, r_ = sp.Integer(1), sp.Integer(1)
q_ = sp.sqrt((1 + r_)**2 + 1 / b_**2)
check("beta = tau = Z = 1: Lambda+- = +-sqrt5 (exact)", sp.simplify(-r_ + 1 / b_ + q_ - sp.sqrt(5)) == 0 and sp.simplify(-r_ + 1 / b_ - q_ + sp.sqrt(5)) == 0)
# (B1): solve (3.27)-(3.28) for n, b given N0 = int mean(g) dv, Nx = int mean(x g) dv
worst = 0
for _ in range(50):
    beta = 10 ** rng.uniform(-2, 2); r = 10 ** rng.uniform(-2, 2); Zt = 1 / r
    ghat = rng.standard_normal(len(xq)) * np.exp(-0.05 * xq)    # an arbitrary v_par-integrated ghat(x)
    N0 = (wq * ghat).sum(); Nx = (wq * xq * ghat).sum()
    n, b = np.linalg.solve([[1, -1], [-Zt, -2 * (1 + 1 / beta)]], [N0, Nx])
    D = Zt + 2 * (1 + 1 / beta)
    Gn = -(wq * (xq - 2 * (1 + 1 / beta)) * ghat).sum() / D
    GB = -(wq * (xq + Zt) * ghat).sum() / D
    worst = max(worst, abs(Gn - n) / (abs(n) + abs(b)), abs(GB - b) / (abs(n) + abs(b)))
check("(B1): int G_n = dn/n, int G_B = dB/B from (3.27)-(3.28)", worst < 1e-12, f"max rel {worst:.1e}")

print("BR3  Laguerre l >= 2 sector")
proj = max(abs((wq * eval_laguerre(l, xq) * (0.7 + 1.3 * xq)).sum()) for l in range(2, 13))
check("projection of the source (Z/tau) n + x b on L_l vanishes for l = 2..12", proj < 1e-12, f"max {proj:.1e}")

print("BR4  W_compr and H_ph-sp decompositions by direct 2D velocity quadrature")
# v_par grid (units v_th = 1): trapezoid on [-9, 9] (spectrally accurate for these integrands)
v = np.linspace(-9, 9, 3601); dv = v[1] - v[0]
FM = np.exp(-v**2) / np.sqrt(np.pi)
I = lambda f: f.sum(axis=-1) * dv                        # int dv_par
# a random g_09 with Laguerre content l = 0..4 and smooth v_par dependence; G_l(v) vanish at v = 0 (H needs no PV)
Lmax = 4


def rand_Gl():
    c = rng.standard_normal((Lmax + 1, 4))
    return np.array([v * (c[l, 0] + c[l, 1] * v + c[l, 2] * v**2 + c[l, 3] * v**3) * FM for l in range(Lmax + 1)])


xg, wg = laggauss(30)
Lmat = np.array([eval_laguerre(l, xg) for l in range(Lmax + 1)])     # [l, x]
for beta, r in ((1.0, 1.0), (0.3, 2.0), (5.0, 0.5), (0.1, 0.1), (20.0, 3.0)):
    Zt = 1 / r
    Gl = rand_Gl()
    ghat = np.einsum('lx,lv->xv', Lmat, Gl)             # ghat(x, v);  g = (1/pi) e^{-x} ghat  (n = v_th = 1)
    # field relations (3.27)-(3.28) with mean(...) = int dx e^{-x} int dv (...)
    N0 = (wg[:, None] * ghat).sum(0); Nx = (wg[:, None] * xg[:, None] * ghat).sum(0)
    n, b = np.linalg.solve([[1, -1], [-Zt, -2 * (1 + 1 / beta)]], [I(N0), I(Nx)])
    # (3.31): W/(n T/2) = (Z/tau) n^2 + (2/beta) b^2 + mean(g_i^2/F0),  g_i/F0 = ghat/F_M + x b
    gi_over_F0 = ghat / FM[None, :] + xg[:, None] * b
    W31 = Zt * n**2 + 2 / beta * b**2 + I((wg[:, None] * gi_over_F0**2 * FM[None, :]).sum(0))
    # branches: G_n, G_B by (B2)-(B3) as v_perp quadratures
    D = Zt + 2 * (1 + 1 / beta)
    Gn = -(wg[:, None] * (xg[:, None] - 2 * (1 + 1 / beta)) * ghat).sum(0) / D
    GB = -(wg[:, None] * (xg[:, None] + Zt) * ghat).sum(0) / D
    Lp, Lm, lp, lm, q, sigma = cmm(beta, r)
    Gp = lp[0] * Gn + lp[1] * GB; Gm = lm[0] * Gn + lm[1] * GB
    Wp = I(Gp**2 / FM) - I(Gp)**2 / Lp                   # (B13) / (n T/2)
    Wm = I(Gm**2 / FM) - I(Gm)**2 / Lm
    Wt = sum(I(Gl[l]**2 / FM) for l in range(2, Lmax + 1))   # (B14) / (n T/2)
    coef = lambda kap: ((1 + (1 + r) / kap) * Lp**2, (1 / (2 * r**2)) * (1 + 1 / (kap * beta)) * Lm**2)
    kap_true = np.sqrt((1 + r)**2 + beta**-2); kap_B16 = np.sqrt((1 + r) + beta**-2)
    cp, cm = coef(kap_true); cpB, cmB = coef(kap_B16)
    e_true = abs(W31 - (Wt + cp * Wp + cm * Wm)) / abs(W31)
    e_B16 = abs(W31 - (Wt + cpB * Wp + cmB * Wm)) / abs(W31)
    # (3.33): H/(T) = int dvperp pi vperp P int g^2/(v F0) = (1/2) int dx e^{-x} int ghat^2/(v F_M) dv  (n = v_th = 1)
    H33 = 0.5 * I((wg[:, None] * ghat**2).sum(0) / (v * FM + (v == 0)) * (v != 0))
    Hp = I(Gp**2 / (2 * v * FM + (v == 0)) * (v != 0)); Hm = I(Gm**2 / (2 * v * FM + (v == 0)) * (v != 0))
    Ht = sum(I(Gl[l]**2 / (2 * v * FM + (v == 0)) * (v != 0)) for l in range(2, Lmax + 1))
    h_true = abs(H33 - (Ht + cp * Hp + cm * Hm)) / max(abs(H33), 1e-300)
    h_B16 = abs(H33 - (Ht + cpB * Hp + cmB * Hm)) / max(abs(H33), 1e-300)
    print(f"    beta={beta}, tau/Z={r}: W rel err with kappa(Schekochihin et al. 2009 eq. 204) {e_true:.1e}, with kappa(B16 printed) {e_B16:.2e};"
          f" H: {h_true:.1e} vs {h_B16:.2e}")
    check(f"beta={beta}, tau/Z={r}: (B15) and (B17) hold with kappa^2 = (1+tau/Z)^2 + beta^-2", e_true < 1e-10 and h_true < 1e-10)
    if abs(1 + r - (1 + r)**2) > 1e-12:
        check(f"beta={beta}, tau/Z={r}: (B15) fails with (B16) as printed", e_B16 > 1e-4 and h_B16 > 1e-4)

print("BR5  moments")
worst202 = worst_mom = 0.0
for _ in range(100):
    beta = 10 ** rng.uniform(-2, 2); r = 10 ** rng.uniform(-2, 2); Zt = 1 / r
    Lp, Lm, lp, lm, q, sigma = cmm(beta, r)
    n, b = rng.standard_normal(2)
    M0p = lp @ [n, b]; M0m = lm @ [n, b]
    n09 = (sigma * M0m - r * 2 / beta * M0p) / q          # Schekochihin et al. (2009) eq. (202) as printed
    b09 = (sigma * M0p - (1 + Zt) * M0m) / q              # Schekochihin et al. (2009) eq. (203) as printed
    worst202 = max(worst202, abs(n09 / 2 - n) / (abs(n) + abs(b)), abs(b09 / 2 - b) / (abs(n) + abs(b)))
check("exact (dn/n, dB/B) = (1/2) x Schekochihin et al. 2009 eqs (202)-(203) (CMM footnote)", worst202 < 1e-12, f"max rel {worst202:.1e}")
# u_par, dp_perp, dp_par by direct velocity integrals of g_i = g_09 + x b F0 (k_perp rho_i -> 0)
for beta, r in ((1.0, 1.0), (0.4, 3.0)):
    Zt = 1 / r
    Gl = rand_Gl() + np.array([(0.3 + 0.2 * v**2) * FM] + [0 * v] * Lmax)   # give G_0 a nonzero M0
    ghat = np.einsum('lx,lv->xv', Lmat, Gl)
    N0 = (wg[:, None] * ghat).sum(0); Nx = (wg[:, None] * xg[:, None] * ghat).sum(0)
    n, b = np.linalg.solve([[1, -1], [-Zt, -2 * (1 + 1 / beta)]], [I(N0), I(Nx)])
    gi = ghat + xg[:, None] * b * FM[None, :]             # g_i e^{x} pi  (per unit n, v_th = 1)
    upar = I((wg[:, None] * gi).sum(0) * v)                # (1/n) int d^3v v_par g_i
    dpperp = I((wg[:, None] * xg[:, None] * gi).sum(0))    # (1/n) int d^3v (m vperp^2 / 2T) g_i
    dppar = I((wg[:, None] * gi).sum(0) * 2 * v**2)        # (1/n) int d^3v (m vpar^2 / T) g_i, m vpar^2/T = 2 v^2/v_th^2
    G0 = Gl[0]
    f_upar = I(v * G0); f_perp = -Zt * n - 2 / beta * b; f_par = 2 * I(v**2 * G0) + b
    worst_mom = max(worst_mom, abs(upar - f_upar), abs(dpperp - f_perp), abs(dppar - f_par))
check("u_par = int v G_0, dp_perp/p = -(Z/tau) n - (2/beta) b, dp_par/p = 2 int v^2 G_0 + b (direct integrals)", worst_mom < 1e-10,
      f"max abs {worst_mom:.1e}")
# beta = tau = Z = 1: G_0 = phi (G- - G+), G_1 = phi^2 G+ + G-, c+ = 5 + 2 sqrt5, c- = (5 + sqrt5)/2  (exact)
Gn_, GB_ = sp.symbols('G_n G_B')
s5 = sp.sqrt(5); sig = 3 + s5; rr = 2 / sig
Gp_ = GB_ + rr * Gn_; Gm_ = Gn_ + rr * GB_
G0_ = Gn_ - GB_; G1_ = 2 * G0_ + 5 * GB_                  # G_1 = (1 + Z/tau) G_0 + D G_B with D = 1 + 4 = 5
phi = (1 + s5) / 2
check("beta=tau=Z=1: G_0 = phi (G- - G+), G_1 = phi^2 G+ + G-",
      sp.simplify(sp.expand(G0_ - phi * (Gm_ - Gp_))) == 0 and sp.simplify(sp.expand(G1_ - (phi**2 * Gp_ + Gm_))) == 0)
cp_, cm_ = sp.symbols('c_p c_m')
eqs = sp.Poly(sp.expand(G0_**2 + G1_**2 - cp_ * Gp_**2 - cm_ * Gm_**2), Gn_, GB_).coeffs()
solc = sp.solve(eqs[:2], [cp_, cm_], dict=True)[0]
check("beta=tau=Z=1: c+ = 5 + 2 sqrt5, c- = (5 + sqrt5)/2, cross term zero",
      sp.simplify(solc[cp_] - (5 + 2 * s5)) == 0 and sp.simplify(solc[cm_] - (5 + s5) / 2) == 0
      and all(sp.simplify(e.subs(solc)) == 0 for e in eqs))

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL c02 CHECKS PASSED")
