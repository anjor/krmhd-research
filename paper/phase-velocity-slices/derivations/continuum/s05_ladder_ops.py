"""s05: Case-Van Kampen theory compared with the truncated Hermite ladder matrix (GANDALF's streaming matrix) as M
grows. The Hermite coefficients of the CVK modes are the ladder polynomials r_m(u), orthogonal for the spectral
measure rho du; the zero-closure truncation is the Gauss quadrature of that measure (nodes, eigenvectors,
Christoffel weights, exactness); discrete slice amplitudes equal the continuum transform at the nodes; the
slice-energy distribution converges as M grows; the late-time density response is a sum over Landau roots; and the
node u = 0 is present exactly for even M. It supports Section 3 and Appendix B of the paper (the Gauss-quadrature
statement is also used in Section 8).

Claim: r_m(u) := g_m(F_u) (Hermite coefficients of the CVK mode, eq. (B30) of Chandran, Mallet & Meyrand 2026) is a
polynomial of degree m with
   r_m(u) = [(1 - 1/Lambda) H_m(u) + q_m(u)/Lambda] / sqrt(2^m m!),   q_m(u) = int [v H_m(v) - u H_m(u)]/(v - u) F_M(v) dv,
satisfying the three-term recurrence of the ladder A; the r_m are orthogonal for the spectral measure rho(u) du:
   int rho r_m r_n du = delta_mn / P0_nn ;
hence the zero-closure truncation at order M is the (M+1)-point Gauss quadrature of rho:
   nodes u_j = eig(A) = zeros of r_{M+1}, right eigenvectors (r_0(u_j), ..., r_M(u_j)), left eigenvectors w_j = P0 r_j,
   weights lam_j = 1/sum_m P0_mm r_m(u_j)^2 = mu0 V_0j^2 (Golub-Welsch), mu0 = Lambda/(Lambda-1);
   for G in the span of the first M+1 Hermite functions the discrete slice amplitude w_j . g equals T[G](u_j) exactly.
Checks:
 L1  r_m from the CVK definition (PV quadrature, mpmath) = recurrence polynomial;  explicit form via q_m (SymPy).
 L2  orthogonality int rho r_m r_n = delta/P0 (m, n <= 14) and moments int rho u^k = (P0 A^k)_00/(1-1/Lambda)^2.
 L3  eig(A) = zeros of r_{M+1}; eigenvectors; Golub-Welsch weights = 1/n_j; Gauss exactness for degree <= 2M+1.
 L4  exactness w_j . g = T[G](u_j) for G in the truncated space (M = 12).
 L5  convergence as M grows (M = 16 ... 1024): slice-energy distribution vs int rho T^2; lam_j vs rho(u_j) du_j.
 L6  density (Landau) response: discrete Gauss sum vs continuum (1 - 1/Lambda) int rho e^{-i tau u} du before recurrence;
     late-time continuum = -(Lambda-1) sum_n e^{-i tau zeta_n}/(zeta_n eps'(zeta_n)) over Landau roots.
 L7  parity r_m(-u) = (-1)^m r_m(u): A has the eigenvalue 0 exactly when M is even.
Exit 1 on failure.
"""
import os
import sys
import math
import numpy as np
import sympy as sp
import mpmath as mp
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_common import (FM, ReZ, Zc, eps_c, eR, eI, rho, Gauss, FMT_of, rhoT2, ladder, P0diag, jacobi, gauss_rule, SQRT5)

fails = []


def check(name, cond, info=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {info}", flush=True)
    if not cond:
        fails.append(name)


Q = lambda f, a=-14, b=14: quad(f, a, b, limit=2000, epsabs=1e-14, epsrel=1e-13)[0]


def r_rec(M, Lam, u):
    """r_0..r_M at u (array) by the ladder recurrence: u r_m = A[m,m+1] r_{m+1} + A[m,m-1] r_{m-1}."""
    A = ladder(M + 1, Lam)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    r = np.zeros((M + 1, u.size))
    r[0] = 1.0
    if M >= 1:
        r[1] = u * r[0] / A[0, 1]
    for m in range(1, M):
        r[m + 1] = (u * r[m] - A[m, m - 1] * r[m - 1]) / A[m, m + 1]
    return r


def p_orthonormal(M, Lam, u):
    """sqrt(rho(u)) p_m(u), p_m = sqrt(P0_mm) r_m, by the scaled symmetric recurrence (stable for large M)."""
    b = jacobi(M + 1, Lam)
    u = np.atleast_1d(np.asarray(u, dtype=float))
    out = np.zeros((M + 1, u.size))
    out[0] = np.sqrt(rho(u, Lam)) * np.sqrt(1 - (0 if np.isinf(Lam) else 1 / Lam))
    if M >= 1:
        out[1] = u * out[0] / b[0]
    for m in range(1, M):
        out[m + 1] = (u * out[m] - b[m - 1] * out[m - 1]) / b[m]
    return out


# ---------------------------------------------------------------------------------------------------------- L1
print("L1  CVK Hermite coefficients are the ladder polynomials")
mp.mp.dps = 25
FMm = lambda x: mp.e**(-x**2) / mp.sqrt(mp.pi)
dawm = lambda x: mp.sqrt(mp.pi) / 2 * mp.e**(-x**2) * mp.erfi(x)
worst = 0
for Lv in (mp.sqrt(5), -mp.sqrt(5), mp.mpf('3.3')):
    for u0 in (mp.mpf('-1.7'), mp.mpf('0.25'), mp.mpf('1.1')):
        eRu = 1 - (1 + u0 * (-2 * dawm(u0))) / Lv
        rr = r_rec(8, float(Lv), float(u0))[:, 0]
        for m in range(9):
            Hm = lambda v: mp.hermite(m, v)
            pv = mp.quad(lambda t: (Hm(u0 + t) * (u0 + t) * FMm(u0 + t) - Hm(u0 - t) * (u0 - t) * FMm(u0 - t)) / t, [0, 1, 4, mp.inf])
            gm = (pv / Lv + eRu * Hm(u0)) / mp.sqrt(2**m * mp.factorial(m))
            worst = max(worst, abs(float(gm) - rr[m]) / max(1.0, abs(rr[m])))
check("g_m(F_u) by PV quadrature = r_m(u) by recurrence (m <= 8)", worst < 1e-12, f"max rel {worst:.1e}")
u, v, L = sp.symbols('u v Lambda')
gm_ = lambda k: (sp.factorial2(k - 1) / 2**sp.Rational(k, 2) if k > 0 else 1) if k % 2 == 0 else 0
okq = True
rs = [sp.Integer(1), sp.sqrt(2) * u]
Asym = lambda m, n: (1 - 1 / L) / sp.sqrt(2) if (m, n) == (1, 0) else sp.sqrt(sp.Rational(max(m, n), 2))
for m in range(1, 11):
    rs.append(sp.expand((u * rs[m] - Asym(m, m - 1) * rs[m - 1]) / Asym(m, m + 1)))
for m in range(0, 11):
    Hv = sp.hermite(m, v); Hu = sp.hermite(m, u)
    quo = sp.cancel((v * Hv - u * Hu) / (v - u))                   # polynomial in u, v
    P = sp.Poly(sp.expand(quo), v)
    qm = sp.expand(sum(c * gm_(mon[0]) for mon, c in zip(P.monoms(), P.coeffs())))
    explicit = ((1 - 1 / L) * Hu + qm / L) / sp.sqrt(2**m * sp.factorial(m))
    okq &= sp.simplify(sp.expand(explicit - rs[m])) == 0
check("r_m = [(1-1/Lambda) H_m + q_m/Lambda]/sqrt(2^m m!) (exact, m <= 10)", okq)
print("    r_2 =", sp.factor(rs[2]), ";  r_3 =", sp.factor(rs[3]))

# ---------------------------------------------------------------------------------------------------------- L2
print("L2  orthogonality for rho du")
worst = 0
for Lv in (SQRT5, -SQRT5, 5.0, -0.4):
    Mx = 14
    G = np.zeros((Mx + 1, Mx + 1))
    for mm in range(Mx + 1):
        for nn in range(mm, Mx + 1):
            f = lambda x: (p_orthonormal(Mx, Lv, x)[mm] * p_orthonormal(Mx, Lv, x)[nn])[0]
            G[mm, nn] = G[nn, mm] = Q(f, -16, 16)
    worst = max(worst, np.abs(G - np.eye(Mx + 1)).max())
check("int rho p_m p_n du = delta_mn, p_m = sqrt(P0_mm) r_m (m,n <= 14; 4 Lambda)", worst < 1e-10, f"max {worst:.1e}")
worst = 0
for Lv in (SQRT5, -SQRT5, 5.0):
    A = ladder(20, Lv); P0 = P0diag(20, Lv)
    Ak = np.eye(21)
    for k in range(13):
        mat = (P0[:, None] * Ak)[0, 0] / (1 - 1 / Lv)**2
        cont = Q(lambda x: x**k * rho(x, Lv), -16, 16)
        worst = max(worst, abs(mat - cont) / max(1.0, abs(cont)))
        Ak = A @ Ak
check("int rho u^k du = (P0 A^k)_00/(1 - 1/Lambda)^2, k <= 12", worst < 1e-11, f"max rel {worst:.1e}")

# ---------------------------------------------------------------------------------------------------------- L3
print("L3  truncation = Gauss quadrature")
worst_z = worst_v = worst_w = 0
for Lv in (SQRT5, -SQRT5, 5.0):
    for Mv in (10, 20, 40):
        A = ladder(Mv, Lv); P0 = P0diag(Mv, Lv)
        uj, lam, V = gauss_rule(Mv, Lv)
        ev = np.sort(np.linalg.eigvals(A).real)
        worst_z = max(worst_z, np.abs(ev - uj).max())
        pM1 = p_orthonormal(Mv + 1, Lv, uj)                  # sqrt(rho) p_m at the nodes
        scale = np.sqrt((pM1[:Mv + 1]**2).sum(axis=0))
        worst_z = max(worst_z, np.abs(pM1[Mv + 1] / scale).max())
        R = r_rec(Mv, Lv, uj)                                # columns: r(u_j)
        worst_v = max(worst_v, np.abs(A @ R - R * uj[None, :]).max() / np.abs(R).max())
        n_j = (P0[:, None] * R**2).sum(axis=0)
        worst_w = max(worst_w, np.abs(1 / n_j - lam).max() / lam.max())
check("eig(A) = Golub-Welsch nodes = zeros of r_{M+1} (M = 10,20,40)", worst_z < 1e-10, f"{worst_z:.1e}")
check("A r(u_j) = u_j r(u_j): eigenvectors are the CVK Hermite coefficients", worst_v < 1e-10, f"{worst_v:.1e}")
check("Christoffel weights 1/sum P0 r_m(u_j)^2 = mu0 V_0j^2", worst_w < 1e-10, f"{worst_w:.1e}")
# Gauss exactness in 50-digit arithmetic (float64 loses ~1e-9 on degree-21 moments)
mp.mp.dps = 50
worst = mp.mpf(0)
for Lv in (mp.sqrt(5), -mp.sqrt(5)):
    for Mv in (6, 10):
        def jac(n):
            Jm = mp.zeros(n + 1, n + 1)
            for m in range(n):
                bm = mp.sqrt((1 - 1 / Lv) / 2) if m == 0 else mp.sqrt(mp.mpf(m + 1) / 2)
                Jm[m, m + 1] = Jm[m + 1, m] = bm
            return Jm
        E, Vm = mp.eigsy(jac(Mv))
        mu0 = 1 / (1 - 1 / Lv)
        lam = [mu0 * Vm[0, j]**2 for j in range(Mv + 1)]
        Jbig = jac(3 * Mv)
        Pk = mp.eye(3 * Mv + 1)
        for k in range(2 * Mv + 2):
            mom = mu0 * Pk[0, 0]                      # int rho u^k = mu0 (J^k)_00 = (A^k)_00/(1 - 1/Lambda)
            gs = mp.fsum(lam[j] * E[j]**k for j in range(Mv + 1))
            worst = max(worst, abs(gs - mom) / max(1, abs(mom)))
            Pk = Jbig * Pk
check("sum_j lam_j u_j^k = int rho u^k for k <= 2M+1 (M = 6, 10; 50 digits)", worst < mp.mpf("1e-40"), f"max rel {mp.nstr(worst, 3)}")

# ---------------------------------------------------------------------------------------------------------- L4
print("L4  discrete slice amplitude = continuum transform at the nodes (truncated G)")
rng = np.random.default_rng(3)
worst = 0
for Lv in (SQRT5, -SQRT5):
    Mv = 12
    g = rng.standard_normal(Mv + 1)
    uj, lam, V = gauss_rule(Mv, Lv)
    R = r_rec(Mv, Lv, uj); P0 = P0diag(Mv, Lv)
    a_disc = (P0[:, None] * R * g[:, None]).sum(axis=0)           # w_j . g
    # continuum T[G](u) for G = sum g_m H_m F_M/sqrt(2^m m!):  T = eR G/F_M + (u/Lambda) P int G/(v-u)
    # P int H_m F_M/(v-u) dv = int (H_m(v)-H_m(u))/(v-u) F_M dv + H_m(u) ReZ(u)
    for j, u0 in enumerate(uj):
        Gu = 0.0; PV = 0.0
        for m in range(Mv + 1):
            Hm = np.polynomial.hermite.Hermite.basis(m)
            nm = np.sqrt(2.0**m * math.factorial(m))
            Gu += g[m] * Hm(u0) / nm                                   # G/F_M at u0
            # (H_m(v) - H_m(u0))/(v - u0) as a polynomial in v, integrated against F_M exactly by Gauss-Hermite
            xg, wg = np.polynomial.hermite.hermgauss(40)
            with np.errstate(divide='ignore', invalid='ignore'):
                quo = np.where(np.abs(xg - u0) > 1e-12, (Hm(xg) - Hm(u0)) / (xg - u0), Hm.deriv()(u0))
            PV += g[m] / nm * ((wg * quo).sum() / np.sqrt(np.pi) + Hm(u0) * ReZ(u0))
        T = eR(u0, Lv) * Gu + (u0 / Lv) * PV
        worst = max(worst, abs(T - a_disc[j]) / np.abs(a_disc).max())
check("w_j . g = T[G](u_j) exactly for G in span(e_0..e_12)", worst < 1e-11, f"max rel {worst:.1e}")

# ---------------------------------------------------------------------------------------------------------- L5
print("L5  convergence as M grows")
Gt = Gauss([(1.0, 0.5, 1.0), (-0.7, -0.3, 0.8)])
for Lv in (SQRT5, -SQRT5):
    Wc = Q(lambda x: rhoT2(Gt, x, Lv), -14, 14)
    rows = []
    errs = []
    for Mv in (16, 32, 64, 128, 256, 512, 1024):
        uj, lam, V = gauss_rule(Mv, Lv)
        g = Gt.hermite(Mv)
        P0 = P0diag(Mv, Lv)
        E = (V.T @ (np.sqrt(P0) * g))**2                     # discrete slice energies, sum = g^T P0 g
        cum = np.cumsum(E)
        mids = 0.5 * (uj[1:] + uj[:-1])
        sel = np.abs(mids) < 4
        contcum = np.array([Q(lambda x: rhoT2(Gt, x, Lv), -14, m_) for m_ in mids[sel]])
        dcdf = np.abs(cum[:-1][sel] - contcum).max() / Wc
        du = np.gradient(uj)
        inner = np.abs(uj) < 3
        dw = np.abs(lam[inner] / (rho(uj[inner], Lv) * du[inner]) - 1).max()
        errs.append(dcdf)
        rows.append(f"M={Mv}: sumE-W={E.sum() - Wc:+.1e}, sup|CDF_M - CDF| /W={dcdf:.2e}, max|lam/(rho du)-1|(|u|<3)={dw:.2e}")
    print(f"    Lambda={Lv:+.4f}:\n      " + "\n      ".join(rows))
    rate = np.polyfit(np.log([16, 32, 64, 128, 256, 512, 1024]), np.log(errs), 1)[0]
    print(f"      CDF error ~ M^{rate:.2f}")
    check(f"Lambda={Lv:+.3f}: slice-energy distribution converges (sup CDF error < 0.02 W at M=1024, decreasing)",
          errs[-1] < 2e-2 and errs[-1] < errs[0] / 5)

# ---------------------------------------------------------------------------------------------------------- L6
print("L6  density response: Gauss sum, continuum, Landau roots")
for Lv in (SQRT5, -SQRT5):
    taus = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 12.0])

    def M0cont(t):
        re = quad(lambda x: rho(x, Lv) * np.cos(t * x), -14, 14, limit=4000, epsabs=1e-15)[0]
        im = quad(lambda x: -rho(x, Lv) * np.sin(t * x), -14, 14, limit=4000, epsabs=1e-15)[0]
        return (1 - 1 / Lv) * (re + 1j * im)

    cont = np.array([M0cont(t) for t in taus])
    for Mv in (128, 512):
        uj, lam, V = gauss_rule(Mv, Lv)
        disc = np.array([(1 - 1 / Lv) * (lam * np.exp(-1j * t * uj)).sum() for t in taus])
        d = np.abs(disc - cont).max()
        print(f"    Lambda={Lv:+.4f} M={Mv}: max |g0_disc - M0_cont| over tau <= 12: {d:.1e}")
        check(f"Lambda={Lv:+.3f}, M={Mv}: Gauss sum = continuum before recurrence (tau <= 12)", d < 1e-10)
    # Landau-root asymptotics
    mp.mp.dps = 30
    Zm = lambda z: 1j * mp.sqrt(mp.pi) * mp.e**(-z**2) * mp.erfc(-1j * z)
    epsm = lambda z: 1 - (1 + z * Zm(z)) / Lv
    depsm = lambda z: -(Zm(z) - 2 * z * (1 + z * Zm(z))) / Lv
    starts = [mp.mpc(0, -0.41), mp.mpc(2.6, -2.25), mp.mpc(-2.6, -2.25)] if Lv > 0 else \
        [mp.mpc(1.24, -0.835), mp.mpc(-1.24, -0.835), mp.mpc(2.26, -1.89), mp.mpc(-2.26, -1.89)]
    roots = [mp.findroot(epsm, s0) for s0 in starts]
    for t in (4.0, 8.0, 12.0):
        asym = sum(-(Lv - 1) * mp.e**(-1j * t * zn) / (zn * depsm(zn)) for zn in roots)
        c = M0cont(t)
        rel = abs(complex(asym) - c) / abs(c)
        print(f"    Lambda={Lv:+.4f} tau={t}: |M0| = {abs(c):.3e}, Landau-root sum rel. diff = {rel:.1e}")
    check(f"Lambda={Lv:+.3f}: late-time density response = Landau-root residue sum (tau = 12, rel < 1e-6)", rel < 1e-6)

# ---------------------------------------------------------------------------------------------------------- L7
print("L7  parity and the u = 0 node")
xs = np.linspace(-3, 3, 13)
par = max(np.abs(r_rec(15, Lv, -xs)[m] - (-1)**m * r_rec(15, Lv, xs)[m]).max() for Lv in (SQRT5, -SQRT5) for m in range(16))
check("r_m(-u) = (-1)^m r_m(u)", par < 1e-9, f"{par:.1e}")
z_even = max(np.abs(gauss_rule(Mv, SQRT5)[0]).min() for Mv in (64, 128))
z_odd = min(np.abs(gauss_rule(Mv, SQRT5)[0]).min() for Mv in (63, 127))
check("even M (64, 128): eigenvalue 0 present; odd M (63, 127): absent", z_even < 1e-12 and z_odd > 1e-3, f"{z_even:.1e}, {z_odd:.3f}")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL s05 CHECKS PASSED")
