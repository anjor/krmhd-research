"""t6_reflection: checks the hypercollisional reflection law of a finite Hermite hierarchy, R = exp(-X) with
X = 2 nu t_w/(2n+1) and t_w = sqrt(2M)/(|k| s), by a ray (packet) picture, a mode picture and exact revivals of
GANDALF's linear operator over M, n, nu, k_z and Lam, and its consequence for the forced steady state,
W_zero/W_open -> coth X at m << M, against Lyapunov solves. Supports Section 8 (reflection law) and Appendix C of
the paper.

Linear problem at one k_z (k_perp plays no role), GANDALF's operator L = -i s k_z A + nu D,
D = -diag((m/M)^n) for m >= 2 (krmhd.hermite.build_implicit_operator), zero closure unless stated.
A  Ray (packet) picture: for G(t=0) = F_M and Lam = inf the free-streaming solution has |g_m|^2 Poisson with mean
   (k s t)^2/2, so the packet reaches m = M at t_w; integrating nu (m(t)/M)^n along the ray to M and back gives
   2 nu t_w/(2n+1) (SymPy).
B  Mode picture: first-order slice damping gamma_j = nu w_j^T D r_j / n_j -> nu/(2n+1) for |u_j| << sqrt(2M)
   (eigenvector energy spread as m^(-1/2) over 0..M), and the revival time T_rec = 2 pi/(k s Delta_0) -> 2 t_w;
   R = exp(-gamma_0 T_rec) is the same law.
C  The law against the exact revival of g_0(t) (matrix exponential of GANDALF's L) over M, n, nu, n_z and Lam, and
   against the first-order mode sum; domain of validity.
D  Forced steady state at M = 128, n = 6, nu = 3, Lam = sqrt5, k_z = 2 pi, s = 1: reflection with
   repeated round trips predicts W_zero(m)/W_open(m) = [1 + exp(-2X(1 - mu))]/(1 - exp(-2X)), X = 2 nu t_w/(2n+1),
   mu = (m/M)^(n+1/2), against Lyapunov solves with the zero closure and an outgoing (radiation) closure.
"""
from __future__ import annotations

import numpy as np
import sympy as sp
from scipy.linalg import expm, solve_continuous_lyapunov

from tcore import Checker, section, ladder, weight, slices, SQRT5
from krmhd.hermite import build_implicit_operator

C = Checker("t6_reflection")


def Lop(M: int, Lam: float, kz: float, nu: float, n: int, s: float = 1.0, closure: str = "zero") -> np.ndarray:
    return np.array(build_implicit_operator(np.array([kz]), s * s, nu, M, Lam, n, closure))[0]


section("A. Ray picture")
t, tw, nu_, n_ = sp.symbols("t t_w nu n", positive=True)
ray = sp.integrate(nu_ * (t / tw) ** (2 * n_), (t, 0, tw))
C.check("A1 int_0^{t_w} nu (m(t)/M)^n dt = nu t_w/(2n+1) with m(t) = M (t/t_w)^2", sp.simplify(ray - nu_ * tw / (2 * n_ + 1)) == 0,
        f"{sp.simplify(ray)}")
okA2 = True
M = 256
kz = 2 * np.pi
L0 = Lop(M, np.inf, kz, 0.0, 6)
for tt in (0.2, 0.5, 1.0, 1.5):
    g = expm(L0 * tt)[:, 0]
    kap = kz * tt
    mm = np.arange(M + 1)
    from math import lgamma
    exact = np.exp(-kap ** 2 / 4 + mm * np.log(kap + 1e-300) - 0.5 * np.array([lgamma(k + 1) for k in mm]) - mm * 0.5 * np.log(2.0))
    err = np.abs(np.abs(g) - exact).max()
    mbar = (mm * np.abs(g) ** 2).sum() / (np.abs(g) ** 2).sum()
    okA2 = okA2 and err < 1e-10 and abs(mbar - kap ** 2 / 2) < 1e-8 * max(1, kap ** 2)
    print(f"   t={tt}: max | |g_m| - kappa^m e^(-kappa^2/4)/sqrt(2^m m!) | = {err:.1e}; mean m = {mbar:.6f} vs (k s t)^2/2 = "
          f"{kap ** 2 / 2:.6f}")
C.check("A2 free streaming from g_0 (Lam = inf, GANDALF operator, before the packet reaches M): |g_m|^2 Poisson, "
        "mean m = (k s t)^2/2", okA2)

section("B. Mode picture: first-order slice damping and revival time")
okB = True
for Lam in (np.inf, SQRT5):
    for n in (2, 4, 6):
        row = f"   Lam={Lam:+.4f} n={n}: gamma_0 (2n+1)/nu at M ="
        vals = []
        for M in (64, 256, 1024, 2048):
            u, R, W, nn = slices(M, Lam)
            mm = np.arange(M + 1)
            Dd = np.where(mm >= 2, (mm / M) ** n, 0.0)
            gam = np.einsum("mj,m,mj->j", W, Dd, R) / nn
            i0 = np.argsort(np.abs(u))[:6]                 # slices nearest u = 0
            vals.append(gam[i0].mean() * (2 * n + 1))
            row += f" {M}:{vals[-1]:.4f}"
        print(row)
        if np.isinf(Lam):
            okB = okB and abs(vals[-1] - 1) < 0.005 and abs(vals[-1] - 1) < abs(vals[0] - 1)
        else:
            okB = okB and abs(vals[-1] - 1) < 0.03
C.check("B1 first-order damping of the slices near u = 0 -> nu/(2n+1): Lam = inf to 0.3% at M = 2048 (O(1/M)); "
        "Lam = sqrt5 within 3% (slower approach from below)", okB)
okB2 = True
for M in (64, 128, 256, 1024):
    for Lam in (np.inf, SQRT5, -SQRT5):
        u = slices(M, Lam)[0]
        i0 = np.argmin(np.abs(u))
        d0 = (u[i0 + 1] - u[i0 - 1]) / 2
        ratio = (2 * np.pi / d0) / (2 * np.sqrt(2 * M))     # T_rec k s / (2 t_w k s)
        print(f"   M={M} Lam={Lam:+.4f}: spacing at u=0 = {d0:.6f}; T_rec/(2 t_w) = {ratio:.4f}")
        if np.isinf(Lam):
            okB2 = okB2 and abs(ratio - np.sqrt((2 * M + 3) / (2 * M))) < 2e-3
C.check("B2 Lam = inf: T_rec = 2 pi/(k s Delta_0) = 2 sqrt(2M+3)/(k s) ~ 2 t_w (finite Lam shifts Delta_0 by O(1/Lam))", okB2)

section("C. Exact revival versus the law and versus the first-order mode sum")


def revival(M, n, nu, nz, Lam, s=1.0):
    """Revival amplitude of |g_0(t)| near T_rec, collisional / collisionless (maxima over [0.8, 1.2] T_rec), the
    first-order mode-sum prediction, the law, X, the collisionless peak, and the collisional |g_0| at 0.75 T_rec
    (an upper bound on the non-recurrent tail, e.g. the Landau-damped direct response at finite Lam)."""
    kz = 2 * np.pi * nz
    Lc = Lop(M, Lam, kz, nu, n, s)
    Lf = Lop(M, Lam, kz, 0.0, n, s)
    Trec = 2 * np.sqrt(2 * M + 3) / (kz * s)
    ts = np.linspace(0.8 * Trec, 1.2 * Trec, 1601)
    out = []
    tail = None
    for Lx in (Lc, Lf):
        lam, V = np.linalg.eig(Lx)
        c = np.linalg.solve(V, np.eye(M + 1)[:, 0])
        g0 = (V[0] * c) @ np.exp(np.outer(lam, ts))
        out.append(np.abs(g0))
        if tail is None:
            tail = abs((V[0] * c) @ np.exp(lam * 0.75 * Trec))
    R_exact = out[0].max() / out[1].max()
    # first-order mode sum with exact u_j and gamma_j
    u, R, W, nn = slices(M, Lam)
    mm = np.arange(M + 1)
    Dd = np.where(mm >= 2, (mm / M) ** n, 0.0)
    gam = nu * np.einsum("mj,m,mj->j", W, Dd, R) / nn
    a0 = W[0]
    pf = np.abs((R[0] * a0 / nn) @ np.exp(np.outer(-1j * kz * s * u, ts)))
    pc = np.abs((R[0] * a0 / nn) @ np.exp(np.outer(-1j * kz * s * u, ts) - np.outer(gam, ts)))
    R_mode = pc.max() / pf.max()
    tw = np.sqrt(2 * M) / (kz * s)
    X = 2 * nu * tw / (2 * n + 1)
    return R_exact, R_mode, np.exp(-X), X, out[1].max(), tail


rows = []
for Lam in (np.inf, SQRT5):
    for M in (32, 64, 128, 256):
        for n in (2, 4, 6):
            for nu in (0.3, 1.0, 3.0, 10.0, 30.0):
                for nz in (1, 2, 4):
                    Re, Rm, Rw, X, free, tail = revival(M, n, nu, nz, Lam)
                    rows.append((Lam, M, n, nu, nz, X, Re, Rm, Rw, free, tail))
rows = np.array(rows, dtype=float)
resolved = rows[:, 6] * rows[:, 9] > 20 * rows[:, 10]
un = rows[~resolved]
print(f"   {(~resolved).sum()} of {len(rows)} cases unresolved (revival peak below 20x the non-recurrent |g_0| at 0.75 T_rec): "
      f"{int((un[:, 0] == SQRT5).sum())} at Lam = sqrt5 (Landau-damped direct response; M <= 64 or X >= "
      f"{un[un[:, 0] == SQRT5, 5].min():.1f}), {int(np.isinf(un[:, 0]).sum())} at Lam = inf (X >= "
      f"{un[np.isinf(un[:, 0]), 5].min():.1f}); they are excluded below")
rows_all = rows
rows = rows[resolved]
print("   selected cases (Lam, M, n, nu, n_z): X = 2 nu t_w/(2n+1) | R_exact | R_mode (1st order) | R_law = e^-X")
for r in rows:
    if (r[1] in (32, 128) and r[3] in (1.0, 3.0, 30.0) and r[4] in (1, 4) and r[2] == 6) or \
            (r[1] == 128 and r[2] == 6 and r[3] == 3.0 and r[4] == 1):
        print(f"     Lam={r[0]:+.3f} M={int(r[1]):3d} n={int(r[2])} nu={r[3]:5.1f} nz={int(r[4])}: X={r[5]:6.3f} | "
              f"{r[6]:.4f} | {r[7]:.4f} | {r[8]:.4f}   (collisionless revival peak {r[9]:.3f})")
# domain of validity: relative error of the law in log space, |ln R_law - ln R_exact| / X, binned
print("   relative error of the exponent, |ln R_law/R_exact| / X, by M and by X (all n, nu, n_z; Lam = inf and sqrt5):")
for M in (32, 64, 128, 256):
    sel = rows[:, 1] == M
    rel = np.abs(np.log(rows[sel, 8] / rows[sel, 6])) / rows[sel, 5]
    small = rows[sel, 5] < 3
    print(f"     M={M:3d}: median {np.median(rel):.3f}, max {rel.max():.3f}; for X < 3: median {np.median(rel[small]):.3f}, "
          f"max {rel[small].max():.3f}")
for lo, hi in ((0, 0.5), (0.5, 1.5), (1.5, 3), (3, 6), (6, 100)):
    sel = (rows[:, 5] >= lo) & (rows[:, 5] < hi) & (rows[:, 1] >= 128)
    if sel.sum():
        rel = np.abs(np.log(rows[sel, 8] / rows[sel, 6])) / rows[sel, 5]
        relm = np.abs(np.log(rows[sel, 7] / rows[sel, 6])) / rows[sel, 5]
        print(f"     M>=128, {lo} <= X < {hi}: law exponent error median {np.median(rel):.3f} max {rel.max():.3f}; "
              f"first-order mode sum exponent error median {np.median(relm):.3f} max {relm.max():.3f}  ({sel.sum()} cases)")
sel = (rows[:, 1] >= 128) & (rows[:, 5] < 3)
rel = np.abs(np.log(rows[sel, 8] / rows[sel, 6])) / rows[sel, 5]
C.check("C1 the law fixes the exponent to within 5% for M >= 128 and X < 3 (error ~ 1/M: 14%, 7%, 3.4%, 1.7% median "
        "at M = 32..256)", rel.max() < 0.051, f"max exponent error {rel.max():.3f}")
sel2 = (rows[:, 1] >= 128) & (rows[:, 5] < 1.5)
relm = np.abs(np.log(rows[sel2, 7] / rows[sel2, 6])) / rows[sel2, 5]
C.check("C2 the first-order mode sum (exact u_j, gamma_j) reproduces the exact exponent to within 2% for M >= 128, X < 1.5",
        relm.max() < 0.02, f"max {relm.max():.3f}")
# breakdown parameter: damping per Hermite step at m = M, delta = nu/(|k| s sqrt(2M)) = nu t_w/(2M)
delta = rows[:, 3] / (2 * np.pi * rows[:, 4] * np.sqrt(2 * rows[:, 1]))
relall = np.abs(np.log(rows[:, 8] / rows[:, 6])) / rows[:, 5]
print("   exponent error of the law binned by delta = nu/(|k| s sqrt(2M)) (all M, n, nu, n_z, Lam):")
for lo, hi in ((0, 0.01), (0.01, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 3)):
    sb = (delta >= lo) & (delta < hi)
    if sb.sum():
        print(f"     {lo} <= delta < {hi}: median {np.median(relall[sb]):.3f}, max {relall[sb].max():.3f} ({sb.sum()} cases, "
              f"X up to {rows[sb, 5].max():.1f})")
for Mmin in (128,):
    print(f"   the same for M >= {Mmin}:")
    for lo, hi in ((0, 0.03), (0.03, 0.1), (0.1, 0.3), (0.3, 3)):
        sb = (delta >= lo) & (delta < hi) & (rows[:, 1] >= Mmin)
        if sb.sum():
            print(f"     {lo} <= delta < {hi}: median {np.median(relall[sb]):.3f}, max {relall[sb].max():.3f} "
                  f"({sb.sum()} cases, X up to {rows[sb, 5].max():.1f})")
sb = (delta < 0.1) & (rows[:, 1] >= 128)
C.check("C3 M >= 128 and delta = nu/(|k| s sqrt(2M)) < 0.1: exponent error <= 6% at any X (here up to 14)",
        relall[sb].max() < 0.06, f"max {relall[sb].max():.3f}, X up to {rows[sb, 5].max():.1f}")
meds = []
for M in (32, 64, 128, 256):
    sel = rows[:, 1] == M
    meds.append(np.median(np.abs(np.log(rows[sel, 8] / rows[sel, 6])) / rows[sel, 5]))
meds = np.array(meds)
C.check("C4 the residual error is a finite-M effect: the median exponent error halves per doubling of M, at any X and "
        "delta = nu/(|k| s sqrt(2M)) tested (X up to 30, delta up to 0.6)",
        np.all(np.abs(meds[:-1] / meds[1:] - 2) < 0.15), f"medians {np.round(meds, 4).tolist()}")
print("   exponent bias: R_exact < R_law in", int((rows[:, 6] < rows[:, 8]).sum()), "of", len(rows), "cases")

section("D. Forced steady state at M = 128, n = 6, nu = 3, Lam = sqrt5, k_z = 2 pi, s = 1: what the reflection "
        "does to W(m)")
M, n, nu, Lam, kz, s = 128, 6, 3.0, SQRT5, 2 * np.pi, 1.0


def lyap(outgoing: bool, closure: str = "zero") -> np.ndarray:
    Lx = Lop(M, Lam, kz, nu, n, s, closure)
    if outgoing:
        Lx[M, M] += -abs(kz) * s * np.sqrt((M + 1) / 2.0)       # radiation closure g_{M+1} = -i sgn(k) g_M
    Q = np.zeros((M + 1, M + 1), complex); Q[0, 0] = 1.0
    S = solve_continuous_lyapunov(Lx, -Q)
    return weight(M, Lam, closure) * np.real(np.diag(S))


Wz = lyap(False)
Wo = lyap(True)
Wc = lyap(False, "symmetric")
epsW = weight(M, Lam)[0]
tw_ = np.sqrt(2 * M) / (kz * s)
X = 2 * nu * tw_ / (2 * n + 1)
Re_, Rm_, Rw_, X_, _, _t = revival(M, n, nu, 1, Lam)
Xm = -np.log(Re_)
print(f"   X = 2 nu t_w/(2n+1) = {X:.4f} (R = {np.exp(-X):.4f}); measured revival ratio {Re_:.4f} -> X_meas = {Xm:.4f}")
print("     m  | W_zero |k|s sqrt(2m)/eps_W | W_open ... | W_copy ... | W_zero/W_open | law, X | law, X_meas")
okD = True
for m in (2, 4, 8, 16, 32, 48, 64, 80, 96, 112, 120, 128):
    mu = (m / M) ** (n + 0.5)
    law = (1 + np.exp(-2 * X * (1 - mu))) / (1 - np.exp(-2 * X))
    lawm = (1 + np.exp(-2 * Xm * (1 - mu))) / (1 - np.exp(-2 * Xm))
    asy = epsW / (abs(kz) * s * np.sqrt(2 * m))
    print(f"   {m:4d} | {Wz[m] / asy:.4f} | {Wo[m] / asy:.4f} | {Wc[m] / asy:.4f} | {Wz[m] / Wo[m]:.4f} | {law:.4f} | {lawm:.4f}")
    if 2 <= m <= 48:
        okD = okD and abs(Wz[m] / Wo[m] / lawm - 1) < 0.01
C.check("D1 W_zero/W_open follows [1 + e^(-2X(1-mu))]/(1 - e^(-2X)) within 1% for 2 <= m <= 48 (X from the exact revival)",
        okD)
print("   (the law with the WKB X gives coth(X) = %.4f at low m; with X_meas coth = %.4f)" % (1 / np.tanh(X), 1 / np.tanh(Xm)))
C.finish()
