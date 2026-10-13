"""d08: no Fjortoft-type constraint from Gamma, H or any Q_phi acts on the Hermite transfer.  With GANDALF's Hermite
ladder the script checks that W is the only member of the invariant family Q_phi that is diagonal in m, that a
positive invariant can still carry an m-weight growing like m + 1/2, that free phase mixing (and Landau damping)
conserves every Q_phi while moving all of W to high m, and that Gamma and H vanish on mirror-symmetric states with
arbitrary W(m).  It supports Section 5 (no Fjortoft-type constraint).

F1 Only W is m-diagonal.  For GANDALF's ladder A (irreducible tridiagonal), the quadratic invariants
   Q_phi = g^T P0 phi(A) g = g^T WL^T diag(phi(u)) WL g are of the form sum_m w_m g_m^2 only for phi = const:
   (a) the diagonal matrices commuting with A form a 1-dimensional space;
   (b) the linear map phi -> offdiag(WL^T diag(phi) WL) has a 1-dimensional kernel (phi = const).
F2 A positive invariant can still have an m-weight growing with m: for a pure Hermite state e_m,
   Q_{u^2}/W = (P0 A^2)_{mm}/p_m = A_{m,m+1}A_{m+1,m} + A_{m,m-1}A_{m-1,m} ~ m + 1/2, which naively would satisfy
   the Fjortoft-type criterion B(m) >= c m^n W(m) of Chandran, Mallet & Meyrand (2026, section 4.1) with n = 1 ...
F3 ... yet free phase mixing (an exact solution of KRMHD with Phi = Psi = 0) conserves every Q_phi and moves all of
   W to high m: from g_0 (free streaming) W(m <= m*, t) = P[Poisson(lam) <= m*] -> 0, lam = (s k t)^2/2, while the
   m-diagonal part of Q_{u^2} is lam + 1/2 and the off-diagonal part is exactly -lam.  Landau case (Lambda = sqrt5)
   also checked: all Q_phi constant, W(m <= m*) -> 0 before recurrence.
F4 Gamma (and H) are mirror-odd: every mirror-symmetric state has Gamma(m) = 0 for every m with arbitrary W(m), so no
   inequality |Gamma(m)| >= c m^n W(m) holds (checked on a symmetrised random GANDALF state).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import mpmath as mp  # noqa: E402

from dcore import Box, Checks, SQRT5, W_m, ladder, pweights, slices  # noqa: E402

chk = Checks()
BIG = 1e300

# F1
for M in (4, 8, 9, 16, 64, 128):
    for Lam in (SQRT5, -SQRT5, BIG):
        A = ladder(M, Lam)
        n = M + 1
        # (a) D diagonal with D A - A D = 0: linear map d -> (D A - A D) entries
        rows = []
        for i in range(n):
            for j in range(n):
                if A[i, j] != 0 and i != j:
                    r = np.zeros(n)
                    r[i] += A[i, j]
                    r[j] -= A[i, j]
                    rows.append(r)
        sv = np.linalg.svd(np.array(rows), compute_uv=False)
        dim_a = n - int(np.sum(sv > 1e-10 * sv.max()))
        # (b) phi -> off-diagonal part of WL^T diag(phi) WL
        u, V, WL = slices(M, Lam)
        iu = np.triu_indices(n, 1)
        Mat = np.array([np.outer(WL[j], WL[j])[iu] for j in range(n)]).T     # (pairs, n)
        sv2 = np.linalg.svd(Mat, compute_uv=False)
        dim_b = n - int(np.sum(sv2 > 1e-9 * sv2.max()))
        lab = "inf" if Lam == BIG else f"{Lam:+.3f}"
        ok = dim_a == 1 and dim_b == 1
        if not ok or M in (8, 128):
            chk(f"F1 M={M}, Lambda={lab}: commutant of A in diagonals has dim 1; m-diagonal Q_phi only for phi = const",
                ok, f"dims {dim_a}, {dim_b}")
        elif not ok:
            chk(f"F1 M={M} Lambda={lab}", False)
print("    F1 also holds for M = 4, 9, 16, 64 and all three Lambda (silent passes)")

# F2
M = 64
for Lam in (SQRT5, BIG):
    A = ladder(M, Lam)
    p = pweights(M, Lam)
    q = np.diag((p[:, None] * A) @ A) / p
    lab = "inf" if Lam == BIG else "sqrt5"
    print(f"    F2 Lambda={lab}: Q_(u^2)/W for pure Hermite states e_m, m = 0, 1, 2, 4, 16, 32: "
          + ", ".join(f"{q[m]:.3f}" for m in (0, 1, 2, 4, 16, 32)))
    ok = np.allclose(q[2:M - 1], np.arange(2, M - 1) + 0.5)
    chk(f"F2 Lambda={lab}: Q_(u^2)/W = m + 1/2 for 2 <= m < M (a positive weight growing with m)", ok)

# F3 free streaming and Landau damping
M = 512
mp.mp.dps = 30
for Lam, lab in ((BIG, "free streaming"), (SQRT5, "Landau, Lambda = sqrt5")):
    A = ladder(M, Lam)
    p = pweights(M, Lam)
    u, V, WL = slices(M, Lam)
    g0 = np.zeros(M + 1)
    g0[0] = 1.0
    a0 = WL @ g0
    phis = {"1": np.ones_like(u), "u": u, "u^2": u ** 2, "u^4": u ** 4, "|u|": np.abs(u), "sgn u": np.sign(u)}
    Q0 = {k: float(np.sum(f * a0 ** 2)) for k, f in phis.items()}
    worst_Q, worst_pois, worst_diag = 0.0, 0.0, 0.0
    lows = []
    for kt in (0.5, 2.0, 5.0, 10.0, 14.0):
        lam = 0.5 * kt ** 2
        a = a0 * np.exp(-1j * kt * u)                  # slices just rotate: a_j(t) = a_j(0) e^{-i s k u_j t}
        g = V @ a
        for k, f in phis.items():
            Qt = float(np.real(np.sum(f * np.abs(a) ** 2)))
            Qg = float(np.real(np.conj(g) @ ((p[:, None] * V) @ np.diag(f) @ WL) @ g))   # g^H P0 phi(A) g
            sc = max(abs(Q0[k]), Q0["1"])            # Q_u and Q_sgn(u) vanish for a pure g_0 state: normalise by W
            worst_Q = max(worst_Q, abs(Qt - Q0[k]) / sc, abs(Qg - Q0[k]) / sc)
        Wm = p * np.abs(g) ** 2
        lows.append((kt, Wm[:11].sum() / Wm.sum()))
        if Lam == BIG:
            pois = np.array([float(mp.e ** (-lam) * mp.mpf(lam) ** m / mp.factorial(m)) for m in range(M + 1)])
            worst_pois = max(worst_pois, np.abs(Wm - pois)[: M // 2].max())
            A2 = (p[:, None] * A) @ A
            diag_part = float(np.sum(np.diag(A2) * np.abs(g) ** 2))
            off_part = float(np.real(np.conj(g) @ A2 @ g)) - diag_part
            worst_diag = max(worst_diag, abs(diag_part - (lam + 0.5)), abs(off_part + lam))
    chk(f"F3 {lab}: every Q_phi (phi = 1, u, u^2, u^4, |u|, sgn u) constant along free phase mixing", worst_Q < 1e-10,
        f"max change / max(|Q_phi|, W) = {worst_Q:.1e}; Q_phi(0) = " + ", ".join(f"{k}: {v:.3g}" for k, v in Q0.items()))
    print(f"    F3 {lab}: fraction of W in m <= 10 at s k t = " + ", ".join(f"{kt}: {f:.2e}" for kt, f in lows))
    chk(f"F3 {lab}: W(m <= 10) -> 0 while all invariants are conserved", lows[-1][1] < 1e-3 * lows[0][1] or lows[-1][1] < 1e-6,
        f"{lows[0][1]:.2e} -> {lows[-1][1]:.2e}")
    if Lam == BIG:
        chk("F3 free streaming: W(m) is Poisson(lam) (GANDALF ladder, M = 512, before recurrence)", worst_pois < 1e-10,
            f"{worst_pois:.1e}")
        chk("F3 free streaming: m-diagonal part of Q_(u^2) = lam + 1/2, off-diagonal part = -lam", worst_diag < 1e-8,
            f"{worst_diag:.1e}")

# F4 mirror-symmetric states have Gamma(m) = 0 with arbitrary W(m)
N, M = 16, 9
box = Box(N)
rng = np.random.default_rng(2)
for Lam in (SQRT5, -SQRT5):
    A = ladder(M, Lam)
    p = pweights(M, Lam)
    g = box.random_field(rng, (1, 4), 3, extra=(M + 1,))
    Sg = (-1.0) ** np.arange(M + 1)
    gm = np.roll(np.flip(g, 0), 1, 0) * Sg                      # M g: k_z -> -k_z, g_m -> (-1)^m g_m
    gs = 0.5 * (g + gm)
    Gm = np.array([2 * p[m] * A[m, m + 1] * box.inner(gs[..., m], gs[..., m + 1]) for m in range(M)])
    Wm = W_m(box, gs)
    u, V, WL = slices(M, Lam)
    a = gs @ WL.T
    E = np.array([box.norm2(a[..., j]) for j in range(M + 1)])
    H = float(np.sum(E / u))                                    # odd M: no zero eigenvalue
    chk(f"F4 Lambda={Lam:+.3f}: mirror-symmetric state has Gamma(m) = 0 for all m, H = 0, W(m) > 0",
        np.abs(Gm).max() < 1e-14 * Wm.max() and abs(H) < 1e-12 * E.sum() and Wm.min() > 0,
        f"max|Gamma(m)|/max W = {np.abs(Gm).max() / Wm.max():.1e}, |H|/W = {abs(H) / E.sum():.1e}, "
        f"W(m) range {Wm.min():.2e}..{Wm.max():.2e}")
chk.finish()
