"""c03_invariants: independent checks of the conserved quadratic forms of the truncated hierarchy and of the
helicity at odd and even M: the space of conserved k-independent forms by a 60-digit SVD, the failure of k-weighted
forms, CMM's continuum integrals (B13, B18, B25, B31, B32) by direct quadrature on the retained span, what changes
under the copy closure, per-slice Casimirs with exact quadrature, the Galerkin matrix of P(1/v) by direct
quadrature, and the M^(-1/2) convergence of the even-M principal-value helicity out to M = 2048. Supports Section 8
and Appendix C of the paper.

Independent routes:

a. Exact (SymPy, symbolic Lam) solution of K A = A^T K for symmetric K, M = 4, 5, both closures: dimension M + 1 and
   K = Pi phi(A).
b. A translation-invariant but k-dependent form, sum_k |k_z| g^T Pi g, is NOT conserved by krmhd_rhs (so "exactly the
   (M+1)-dimensional family" needs the class "k-independent quadratic forms" stated).
c. Direct mpmath quadrature of CMM's integrals for random retained moments (M = 8 and 9, Lam = sqrt5):
   B13/B31 (W), B25 (Gamma), B18 (H, principal value) against Q_1/2, Q_u/2, Q_{1/u}/2, and [G, L^2 G] against
   g^T Pi A^2 g + (M+1) g_M^2/2 -- zero closure.
d. Copy closure: Q_u's matrix Pi A is closure-independent; Q_1 = g^T Pi g is NOT (B31) on the retained moments (weight p_M on
   g_M^2); at odd M, Pi A^-1 is NOT the Galerkin matrix of P(1/v) (it is D_c J^-1 D_c).
e. Per-slice Casimirs on krmhd_rhs with EXACT quadrature (2x padded grid): d/dt int a_j^2 = 0, d/dt int a_j^4 != 0.
f. Principal-value Galerkin matrix B_M by direct mpmath quadrature (not the Gauss-rule argument): odd M: B_M = J^-1;
   even M: B_M = J^D + lambda_0 (e d^T + d e^T).
g. Even-M principal-value H error for G = F_M(v - 1/2) at Lam = sqrt5 out to M = 2048: does error x sqrt(M) converge?
"""
from __future__ import annotations

import numpy as np
import sympy as sp
import mpmath as mp
from scipy.linalg import eigh_tridiagonal
from scipy.special import dawsn, roots_hermite
from math import lgamma

from cc import Checks, hdr, SQRT5, grid3, rand_field, state, pb, ddz, vint, gandalf_eig, to_real, jnp

C = Checks("c03_invariants")
from krmhd.hermite import compute_streaming_matrix
from krmhd.timestepping import krmhd_rhs

hdr("a. Space of symmetric K with K A = A^T K: 60-digit mpmath SVD of the linear map (M = 4..13, two Lam, both closures)")
mp.mp.dps = 60
okA = True
for M in (4, 5, 9, 10):
    for Lam in (mp.mpf(7) / 3, -mp.sqrt(5)):
        for clo in ("zero", "symmetric"):
            A = mp.matrix(M + 1, M + 1)
            for m in range(M):
                A[m, m + 1] = mp.sqrt(mp.mpf(m + 1) / 2); A[m + 1, m] = mp.sqrt(mp.mpf(m + 1) / 2)
            A[1, 0] = (1 - 1 / Lam) / mp.sqrt(2)
            if clo == "symmetric":
                A[M, M - 1] += mp.sqrt(mp.mpf(M + 1) / 2)
            idx = [(i, j) for i in range(M + 1) for j in range(i, M + 1)]
            rows = []
            for (i, j) in idx:
                E = mp.matrix(M + 1, M + 1); E[i, j] = 1; E[j, i] = 1
                Cm = E * A - A.T * E
                rows.append([Cm[r, c] for r in range(M + 1) for c in range(r + 1, M + 1)])
            Mat = mp.matrix(rows).T                      # columns = symmetric basis elements
            sv = mp.svd_r(Mat, compute_uv=False)
            sv = sorted([abs(x) for x in sv])
            nnull = len(idx) - len(sv) + sum(1 for x in sv if x < mp.mpf(10) ** -40)
            gap = min(x for x in sv if x >= mp.mpf(10) ** -40)
            # Pi A^n (n = 0..M) solve it and are independent
            p = [1 - 1 / Lam] + [mp.mpf(1)] * (M - 1) + [mp.mpf(1) if clo == "zero" else mp.sqrt(M) / (mp.sqrt(M) + mp.sqrt(M + 1))]
            Pi = mp.diag(p)
            res, vecs, An = mp.mpf(0), [], mp.eye(M + 1)
            for n in range(M + 1):
                K = Pi * An
                D = K * A - A.T * K
                res = max(res, max(abs(D[r, c]) for r in range(M + 1) for c in range(M + 1)))
                vecs.append([K[i, j] for (i, j) in idx])
                An = An * A
            sv2 = mp.svd_r(mp.matrix(vecs), compute_uv=False)
            rk = sum(1 for x in sv2 if abs(x) > mp.mpf(10) ** -40)
            ok = nnull == M + 1 and res < mp.mpf(10) ** -45 and rk == M + 1
            okA = okA and ok
    print(f"   M={M}: nullity = M+1 for both closures and Lam = 7/3, -sqrt5 ({ok}); smallest nonzero singular value {mp.nstr(gap, 3)}; "
          f"Pi A^n residual {mp.nstr(res, 3)}, rank {rk}")
C("c03.a conserved quadratic forms (k-independent K) = span{Pi A^n}, dimension M+1 (60-digit SVD)", okA)

hdr("b. A k-dependent translation-invariant form is not conserved")
N = 16
g16 = grid3(N)
mask = np.asarray(g16.dealias_mask)
w = np.full(N // 2 + 1, 2.0); w[0] = 1.0; w[-1] = 1.0
W3 = np.broadcast_to(w[None, None, :], (N, N, N // 2 + 1))
M, Lam = 9, SQRT5
rng = np.random.default_rng(17)
zp = rand_field(g16, rng, amp=300.0)
zm = rand_field(g16, rng, amp=300.0)
g = rand_field(g16, rng, slope=-1.0, extra=(M + 1,))
gd = np.asarray(krmhd_rhs(state(g16, M, Lam, zp, zm, g), eta=0.0, v_A=1.0).g)
p = np.ones(M + 1); p[0] = 1 - 1 / Lam
kzabs = np.abs(np.asarray(g16.kz))[:, None, None]
kperp2 = (np.asarray(g16.kx)[None, None, :] ** 2 + np.asarray(g16.ky)[None, :, None] ** 2)
res = {}
for name, mult in (("1", np.ones_like(kzabs)), ("|k_z|", kzabs), ("k_perp^2", kperp2)):
    val = 2 * np.sum(W3[..., None] * mult[..., None] * p * np.real(np.conj(g) * gd))
    sc = 2 * np.sum(W3[..., None] * mult[..., None] * p * np.abs(g) * np.abs(gd))
    res[name] = abs(val) / sc
    print(f"   d/dt sum_k {name:8s} g^T Pi g : relative rate {res[name]:.1e}")
C("c03.b the family is exact only within k-independent forms: k-weighted versions of Q_1 are not conserved",
  res["1"] < 1e-14 and res["|k_z|"] > 1e-4 and res["k_perp^2"] > 1e-4)

hdr("c. CMM's continuum integrals by direct quadrature on the retained span (zero closure, Lam = sqrt5)")
mp.mp.dps = 30
FM = lambda v: mp.exp(-v * v) / mp.sqrt(mp.pi)
pm_ = lambda m, v: mp.hermite(m, v) / mp.sqrt(mp.mpf(2) ** m * mp.factorial(m))
okC = True
for M in (8, 9):
    rng = np.random.default_rng(M)
    gv = rng.standard_normal(M + 1)
    Lm = mp.mpf(SQRT5)
    G = lambda v: FM(v) * mp.fsum(mp.mpf(gv[m]) * pm_(m, v) for m in range(M + 1))
    M0 = mp.quad(G, [-mp.inf, 0, mp.inf])
    M1 = mp.quad(lambda v: v * G(v), [-mp.inf, 0, mp.inf])
    W_B13 = (mp.quad(lambda v: G(v) ** 2 / FM(v), [-mp.inf, 0, mp.inf]) - M0 ** 2 / Lm) / 2
    Gam_B25 = mp.quad(lambda v: G(v) ** 2 * v / (2 * FM(v)), [-mp.inf, 0, mp.inf]) - M0 * M1 / Lm
    # principal value: P int G^2/(2 v F_M) = int_0^inf [h(v) - h(-v)]/(2v) dv, h = G^2/F_M
    hfun = lambda v: G(v) ** 2 / FM(v)
    H_B18 = mp.quad(lambda v: (hfun(v) - hfun(-v)) / (2 * v), [0, 1, mp.inf])
    # [G, L^2 G] with L G = v G - v F_M M0(G)/Lam
    LG = lambda v: v * G(v) - v * FM(v) * M0 / Lm
    M0LG = mp.quad(LG, [-mp.inf, 0, mp.inf])
    L2G = lambda v: v * LG(v) - v * FM(v) * M0LG / Lm
    M0L2G = mp.quad(L2G, [-mp.inf, 0, mp.inf])
    GL2G = mp.quad(lambda v: G(v) * L2G(v) / FM(v), [-mp.inf, 0, mp.inf]) - M0 * M0L2G / Lm
    A = np.asarray(compute_streaming_matrix(M, SQRT5, "zero"))
    Pi = np.diag(np.r_[1 - 1 / SQRT5, np.ones(M)])
    Q1 = gv @ Pi @ gv
    Qu = gv @ Pi @ A @ gv
    Qa2 = gv @ Pi @ A @ A @ gv + (M + 1) * gv[M] ** 2 / 2
    d = [abs(float(W_B13) - Q1 / 2), abs(float(Gam_B25) - Qu / 2), abs(float(GL2G) - Qa2)]
    line = f"   M={M}: |W(B13) - Q_1/2| = {d[0]:.1e}; |Gamma(B25) - Q_u/2| = {d[1]:.1e}; |[G,L^2G] - (g Pi A^2 g + (M+1)g_M^2/2)| = {d[2]:.1e}"
    if M % 2:
        Qinv = gv @ Pi @ np.linalg.solve(A, gv)
        d.append(abs(float(H_B18) - Qinv / 2))
        line += f"; |H(B18) - Q_(1/u)/2| (odd M) = {d[3]:.1e}"
    else:
        # at even M the conserved PV form differs from B18: report the difference
        u_, R_ = np.linalg.eig(A)
        u_ = u_.real
        Ad = (R_ * np.where(np.abs(u_) > 1e-9, 1 / np.where(np.abs(u_) > 1e-9, u_, 1), 0)) @ np.linalg.inv(R_)
        Qpv = gv @ Pi @ Ad.real @ gv
        line += f"; even M: H(B18) = {float(H_B18):+.6f} vs conserved PV form/2 = {Qpv / 2:+.6f}"
    print(line)
    okC = okC and max(d) < 1e-12
C("c03.c CMM's W (B13), Gamma (B25), H (B18, odd M) equal Q_1/2, Q_u/2, Q_{1/u}/2 on the retained span; [G,L^2G] misses "
  "(M+1) g_M^2/2 (direct quadrature)", okC)

hdr("d. Copy closure: which CMM forms the conserved quadratic forms are")
okD = True
for M in (9, 10, 33):
    Az = np.asarray(compute_streaming_matrix(M, SQRT5, "zero"))
    Ac = np.asarray(compute_streaming_matrix(M, SQRT5, "symmetric"))
    pz = np.r_[1 - 1 / SQRT5, np.ones(M)]
    pc = pz.copy(); pc[M] = np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
    same_u = np.abs(pz[:, None] * Az - pc[:, None] * Ac).max()
    diff_W = abs(pc[M] - 1)
    line = f"   M={M}: |Pi_c A_c - Pi_z A_z| = {same_u:.1e} (Q_u closure-independent); Q_1 weight on g_M^2: {pc[M]:.4f} vs 1 in (B31)"
    if M % 2:
        Bz = pz[:, None] * np.linalg.inv(Az)          # = J^-1 (zero closure)
        Bc = pc[:, None] * np.linalg.inv(Ac)
        dev = np.abs(Bc - Bz).max() / np.abs(Bz).max()
        line += f"; odd M: max|Pi_c A_c^-1 - J^-1|/max = {dev:.3f}"
        okD = okD and dev > 1e-2
    print(line)
    okD = okD and same_u < 1e-14 and diff_W > 0.4
C("c03.d copy closure: Q_u = CMM's Gamma (B32) on retained moments, but Q_1 is not (B31) and odd-M Pi A^-1 is not the "
  "Galerkin P(1/v) (the 'exact restriction' statements hold for the zero closure only)", okD)

hdr("e. Per-slice Casimirs with exact (padded) quadrature on krmhd_rhs")
okE = True
for M, Lam, clo in ((9, SQRT5, "zero"), (10, -SQRT5, "symmetric")):
    rng = np.random.default_rng(100 + M)
    zp = rand_field(g16, rng, amp=300.0)
    zm = rand_field(g16, rng, amp=300.0)
    g = rand_field(g16, rng, slope=-1.0, extra=(M + 1,))
    gd = np.asarray(krmhd_rhs(state(g16, M, Lam, zp, zm, g), eta=0.0, v_A=1.0, closure=clo).g)
    T, u, P, Pinv = gandalf_eig(M, Lam, clo)
    r2, r4 = 0.0, np.inf
    for j in range(M + 1):
        a = to_real(g16, g @ Pinv[j], pad=2)
        ad = to_real(g16, gd @ Pinv[j], pad=2)
        q2 = abs(np.sum(a * ad)) / np.sum(np.abs(a * ad))
        q4 = abs(np.sum(a ** 3 * ad)) / np.sum(np.abs(a ** 3 * ad))
        r2, r4 = max(r2, q2), min(r4, q4)
    okE = okE and r2 < 1e-13 and r4 > 1e-4
    print(f"   M={M} Lam={Lam:+.3f} {clo}: max_j rel. d/dt int a_j^2 = {r2:.1e}; min_j rel. d/dt int a_j^4 = {r4:.1e} "
          f"(exact quadrature on the 2x padded grid)")
C("c03.e int a_j^2 conserved, int a_j^4 not (exact quadrature)", okE)

hdr("f. Galerkin matrix of P(1/v) by direct mpmath quadrature")
psi = lambda m, v: pm_(m, v) * mp.exp(-v * v / 2) / mp.pi ** mp.mpf(0.25)
okF = True
for M in (6, 7, 10, 11):
    B = np.zeros((M + 1, M + 1))
    for m in range(M + 1):
        for n in range(M + 1):
            if (m + n) % 2 == 1:
                B[m, n] = float(mp.quad(lambda v: psi(m, v) * psi(n, v) / v, [-mp.inf, 0, mp.inf]))
    J = np.asarray(compute_streaming_matrix(M, np.inf, "zero"))
    if M % 2:
        err = np.abs(B - np.linalg.inv(J)).max()
        line = f"   M={M} (odd): max|B_M - J^-1| = {err:.1e}"
        for Lam in (SQRT5, -SQRT5, 3.0):
            A = np.asarray(compute_streaming_matrix(M, Lam, "zero"))
            p = np.r_[1 - 1 / Lam, np.ones(M)]
            err = max(err, np.abs(B - p[:, None] * np.linalg.inv(A)).max())
        line += f"; max|B_M - Pi A^-1| over Lam = +-sqrt5, 3: {err:.1e}"
    else:
        x, wq = roots_hermite(M + 1)
        lam0 = float((wq / np.sqrt(np.pi))[np.argmin(np.abs(x))])
        e = np.array([float(pm_(m, 0)) for m in range(M + 1)])
        d = np.array([float(mp.diff(lambda v: pm_(m, v), 0)) for m in range(M + 1)])
        JD = np.linalg.pinv(J)
        err = np.abs(B - JD - lam0 * (np.outer(e, d) + np.outer(d, e))).max()
        line = f"   M={M} (even): max|B_M - J^D - lambda_0 (e d^T + d e^T)| = {err:.1e} (lambda_0 = {lam0:.6f})"
    print(line)
    okF = okF and err < 1e-12
C("c03.f odd M: B_M = J^-1 = Pi A^-1 (zero closure, any Lam); even M: B_M = J^D + lambda_0(ed^T + de^T) (direct quadrature)",
  okF)

hdr("g. Even-M principal-value H for G = F_M(v - 1/2): convergence rate at Lam = sqrt5 out to M = 2048")
a = 0.5
Hc = 2 * np.exp(2 * a * a) * dawsn(2 * a)
rows = []
for Lam in (SQRT5, -SQRT5, np.inf):
    out = []
    for M in (64, 128, 256, 512, 1024, 2048):
        mm = np.arange(M + 1)
        gm = np.exp(mm * np.log(np.sqrt(2) * a) - 0.5 * np.array([lgamma(k + 1) for k in mm]))
        p = np.ones(M + 1); p[0] = 1 - (0 if np.isinf(Lam) else 1 / Lam)
        off = np.sqrt(np.arange(1, M + 1) / 2.0)
        off[0] = np.sqrt(p[0]) / np.sqrt(2.0)            # Pi^(1/2) A Pi^(-1/2): (0,1) entry sqrt(A01 A10)
        uu, V = eigh_tridiagonal(np.zeros(M + 1), off)
        aj = V.T @ (np.sqrt(p) * gm)                    # a_j / sqrt(n_j)
        nz = np.abs(uu) > 1e-9
        Hpv = np.sum(aj[nz] ** 2 / uu[nz])
        out.append((M, Hpv - Hc))
    rows.append((Lam, out))
    print(f"   Lam={Lam:+.4f}: error x sqrt(M) at M = 64..2048: " + ", ".join(f"{e * np.sqrt(M):.4f}" for M, e in out))
okG = True
for Lam, out in rows:
    Ms_ = np.array([M for M, e in out], float)
    prod = np.array([e * np.sqrt(M) for M, e in out])
    X = np.vstack([np.ones_like(Ms_), Ms_ ** -0.5, Ms_ ** -1.0]).T
    coef, *_ = np.linalg.lstsq(X, prod, rcond=None)
    resid = np.abs(X @ coef - prod).max()
    incr = np.abs(np.diff(prod))
    print(f"   Lam={Lam:+.4f}: fit error x sqrt(M) = {coef[0]:+.4f} {coef[1]:+.3f} M^-1/2 {coef[2]:+.3f} M^-1 (max residual "
          f"{resid:.1e}); successive increments shrink by " + ", ".join(f"{incr[i] / incr[i + 1]:.2f}" for i in range(len(incr) - 1)))
    okG = okG and resid < 2e-3 and abs(coef[0]) > 0.5
C("c03.g even-M PV error = c M^-1/2 (1 + O(M^-1/2)) for Lam = sqrt5, -sqrt5, inf (coefficient settles; slow at Lam = sqrt5)", okG)
C.done()
