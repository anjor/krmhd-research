"""c09_reflection: independent checks of the hypercollisional reflection law: the revival ratio through GANDALF's
own stepper (gandalf_step, IMEX-RK222) rather than a matrix exponential, the revival time at finite Lam, and the
forced steady-state ratio W_zero/W_open against two different open references. Supports Section 8 (reflection law)
and Appendix C of the paper.

Independent routes:

a. The revival through GANDALF's actual stepper (gandalf_step, IMEX-RK222) instead of the matrix exponential of
   build_implicit_operator: one Fourier mode k = (0, 0, 2 pi), z+- = 0, M = 128, n = 6, nu = 3, Lam = sqrt5, s = 1.
   R = max|g_0| (nu = 3) / max|g_0| (nu = 0) near T_rec, against e^{-X}, X = 2 nu t_w/(2n+1), t_w = sqrt(2M)/(|k| s).
b. Revival TIME (matrix exponential of the ladder operator built here): is the revival at ~2 t_w, or at
   2 pi/(k s Delta_0) with Delta_0 the (Lam-dependent) spacing of the slices at u = 0, as the mode picture suggests?
   Lam = inf, sqrt5, -sqrt5; M = 128, 512.
c. Forced steady state: W_zero/W_open with two different "open" references -- the radiation closure used in
   t6_reflection.py and an extended hierarchy (M' = 3M) whose hypercollision profile nu (m/M)^n simply continues beyond M,
   absorbing the packet without a wall -- against [1 + e^{-2X(1-mu)}]/(1 - e^{-2X}) with X from the exact revival.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.linalg import expm, solve_continuous_lyapunov, eig

from cc import Checks, hdr, SQRT5, grid3, state

C = Checks("c09_reflection")
from krmhd.timestepping import gandalf_step


def ladder(M, Lam):
    A = np.zeros((M + 1, M + 1))
    for m in range(M):
        A[m, m + 1] = A[m + 1, m] = np.sqrt((m + 1) / 2.0)
    A[1, 0] = (1 - (0 if np.isinf(Lam) else 1 / Lam)) / np.sqrt(2)
    return A


def damp(M, n, Mref=None):
    Mref = M if Mref is None else Mref
    m = np.arange(M + 1)
    return np.where(m >= 2, (m / Mref) ** n, 0.0)


def g0_series(L, ts):
    lam, V = eig(L)
    c = np.linalg.solve(V, np.eye(L.shape[0])[:, 0])
    return np.abs((V[0] * c) @ np.exp(np.outer(lam, ts)))


hdr("a. Revival through gandalf_step (IMEX), M = 128, n = 6, nu = 3, Lam = sqrt5, k_z = 2 pi")
M, n, nu, Lam, s = 128, 6, 3.0, SQRT5, 1.0
kz = 2 * np.pi
tw = np.sqrt(2 * M) / (kz * s)
X = 2 * nu * tw / (2 * n + 1)
Trec = 2 * np.sqrt(2 * M + 3) / (kz * s)
g4 = grid3(4)
gk = np.zeros((4, 4, 3, M + 1), complex)
gk[1, 0, 0, 0] = 1.0
gk[3, 0, 0, 0] = 1.0                       # Hermitian partner of k_z = -2 pi on the k_x = 0 plane
z0 = np.zeros((4, 4, 3), complex)
nsteps = 1500
dt = 1.25 * Trec / nsteps
peaks = {}
t_all = {}
runs = []
for nuv in (0.0, nu):
    st = state(g4, M, Lam, z0, z0, gk, beta=s * s, nu=nuv)
    amp = []
    t0 = time.time()
    for k in range(nsteps):
        st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=nuv, hyper_n=n, scheme="imex_rk222")
        amp.append(abs(complex(np.asarray(st.g)[1, 0, 0, 0])))
    wall = time.time() - t0
    runs.append((nuv, nsteps, dt, wall))
    amp = np.array(amp)
    ts = dt * np.arange(1, nsteps + 1)
    win = (ts > 0.8 * Trec) & (ts < 1.2 * Trec)
    peaks[nuv] = amp[win].max()
    t_all[nuv] = ts[win][np.argmax(amp[win])]
R_g = peaks[nu] / peaks[0.0]
# the same with the matrix exponential of the operator built here (no GANDALF), for comparison
Lc = -1j * s * kz * ladder(M, Lam) - nu * np.diag(damp(M, n))
Lf = -1j * s * kz * ladder(M, Lam)
tt = np.linspace(0.8 * Trec, 1.2 * Trec, 2001)
R_x = g0_series(Lc, tt).max() / g0_series(Lf, tt).max()
print(f"   X = {X:.4f}, e^-X = {np.exp(-X):.4f}; gandalf_step revival ratio = {R_g:.4f} (peak at t = {t_all[0.0]:.3f}, "
      f"T_rec = {Trec:.3f}); matrix exponential of the operator built here: {R_x:.4f}; exact value from "
      f"t6_reflection.py 0.2913")
C("c09.a GANDALF's stepper reproduces the revival ratio of the linear operator (0.291) to 1%, i.e. e^-X to 6%",
  abs(R_g / R_x - 1) < 0.01 and abs(np.log(R_g) / -X - 1) < 0.06, f"R_gandalf {R_g:.4f}, R_expm {R_x:.4f}")
print("GANDALF integrations: grid 4^3, single mode k = (0,0,2pi), M = 128, IMEX:")
for r in runs:
    print(f"   nu={r[0]} steps={r[1]} dt={r[2]:.3e} wall={r[3]:.1f}s")

hdr("b. Revival time vs 2 t_w and vs 2 pi/(k s Delta_0)")
okB = True
for M in (128, 512):
    tw = np.sqrt(2 * M) / (kz * s)
    for Lam in (np.inf, SQRT5, -SQRT5):
        A = ladder(M, Lam)
        p = np.ones(M + 1); p[0] = 1 - (0 if np.isinf(Lam) else 1 / Lam)
        Ssym = (np.sqrt(p)[:, None] * A) / np.sqrt(p)[None, :]
        uu = np.linalg.eigvalsh(0.5 * (Ssym + Ssym.T))
        i0 = np.argmin(np.abs(uu))
        D0 = 0.5 * (uu[i0 + 1] - uu[i0 - 1])
        Tmode = 2 * np.pi / (kz * s * D0)
        tt = np.linspace(1.6 * tw, 2.6 * tw, 4001)
        gg = g0_series(-1j * s * kz * A, tt)
        tpk = tt[np.argmax(gg)]
        print(f"   M={M} Lam={Lam:+.4f}: revival peak at t/(2 t_w) = {tpk / (2 * tw):.4f}; 2 pi/(k s Delta_0)/(2 t_w) = "
              f"{Tmode / (2 * tw):.4f}; sqrt((2M+3)/2M) = {np.sqrt((2 * M + 3) / (2 * M)):.4f}; peak |g_0| = {gg.max():.3f}")
        if not np.isinf(Lam):
            okB = okB and abs(tpk / (2 * tw) - np.sqrt((2 * M + 3) / (2 * M))) < abs(Tmode / (2 * tw) - tpk / (2 * tw))
C("c09.b at finite Lam the revival occurs at ~2 t_w sqrt((2M+3)/2M), not at 2 pi/(k s Delta_0): the mode picture's "
  "T_rec = 2 pi/(k s Delta_0) is the Lam = inf statement only", okB)

hdr("c. Forced steady state: two different open references")
M, n, nu, Lam = 128, 6, 3.0, SQRT5
tw = np.sqrt(2 * M) / (kz * s)
Xw = 2 * nu * tw / (2 * n + 1)


def lyap(L, M_out):
    Q = np.zeros(L.shape, complex); Q[0, 0] = 1.0
    Sx = solve_continuous_lyapunov(L, -Q)
    p = np.ones(L.shape[0]); p[0] = 1 - 1 / Lam
    return (p * np.real(np.diag(Sx)))[:M_out + 1]


Lz = -1j * s * kz * ladder(M, Lam) - nu * np.diag(damp(M, n))
Wz = lyap(Lz, M)
Lr = Lz.copy()
Lr[M, M] += -abs(kz) * s * np.sqrt((M + 1) / 2.0)
Wr = lyap(Lr, M)
Me = 3 * M
Le = -1j * s * kz * ladder(Me, Lam) - nu * np.diag(damp(Me, n, Mref=M))
We = lyap(Le, M)
tt = np.linspace(0.8 * Trec, 1.2 * Trec, 2001)
Rex = g0_series(Lz, tt).max() / g0_series(-1j * s * kz * ladder(M, Lam), tt).max()
Xm = -np.log(Rex)
print(f"   X_WKB = {Xw:.4f}; X from the exact revival = {Xm:.4f}")
print("     m | W_zero/W_open(radiation) | W_zero/W_open(extended) | law with X_meas | law with X_WKB")
okC = True
for m in (2, 4, 8, 16, 32, 48, 64, 96):
    mu = (m / M) ** (n + 0.5)
    lawm = (1 + np.exp(-2 * Xm * (1 - mu))) / (1 - np.exp(-2 * Xm))
    laww = (1 + np.exp(-2 * Xw * (1 - mu))) / (1 - np.exp(-2 * Xw))
    print(f"   {m:4d} | {Wz[m] / Wr[m]:.4f} | {Wz[m] / We[m]:.4f} | {lawm:.4f} | {laww:.4f}")
    if 2 <= m <= 48:
        okC = okC and abs(Wz[m] / We[m] / lawm - 1) < 0.02 and abs(We[m] / Wr[m] - 1) < 0.02
C("c09.c the coth-type law with the measured X holds against both open references for 2 <= m <= 48 (2%)", okC)
C.done()
