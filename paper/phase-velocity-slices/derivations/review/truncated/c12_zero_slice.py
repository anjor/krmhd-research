"""c12_zero_slice: independent check of the u = 0 slice at even M with GANDALF's own streaming matrix at M = 64 and
128:
the left eigenvector of the zero eigenvalue is psi_m(0) (zero closure; last entry x p_M for the copy closure), for
Lam = sqrt5, -sqrt5, 0.5 (the zero eigenvalue exists for every Lam, even inside the Penrose range), and on krmhd_rhs
the u = 0 slice obeys d_t a_0 = -{Phi, a_0} and conserves (pi/2) sum_k |a_0|^2 (CMM's Im H for one branch).
Supports Section 8 (helicity at odd and even M) of the paper.
"""
from __future__ import annotations

import numpy as np

from cc import Checks, hdr, SQRT5, grid3, rand_field, state, pb, vint

C = Checks("c12_zero_slice")
from krmhd.hermite import compute_streaming_matrix
from krmhd.timestepping import krmhd_rhs


def psi0(M):
    P = np.zeros(M + 1)
    P[0] = np.pi ** -0.25
    for m in range(1, M + 1):
        P[m] = -np.sqrt((m - 1) / m) * P[m - 2] if m >= 2 else 0.0      # psi_m(0) recurrence at x = 0
    return P


ok = True
for M in (64, 128):
    w0 = psi0(M)
    for Lam in (SQRT5, -SQRT5, 0.5):
        for clo in ("zero", "symmetric"):
            A = np.asarray(compute_streaming_matrix(M, Lam, clo))
            w = w0.copy()
            if clo == "symmetric":
                w[M] *= np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
            res = np.abs(w @ A).max() / np.abs(w).max()
            ok = ok and res < 1e-13
    print(f"   M={M}: max |w0^T A|/max|w0| over Lam = +-sqrt5, 0.5 and both closures < 1e-13: {ok}")
C("c12.a left null vector psi_m(0) (x p_M at m = M for the copy closure), any Lam, M = 64, 128", ok)
N = 16
g16 = grid3(N)
mask = np.asarray(g16.dealias_mask)
okD = True
for M, Lam, clo in ((64, SQRT5, "zero"), (64, -SQRT5, "symmetric")):
    rng = np.random.default_rng(M + 3)
    zp = rand_field(g16, rng, amp=300.0)
    zm = rand_field(g16, rng, amp=300.0)
    g = rand_field(g16, rng, slope=-1.0, extra=(M + 1,))
    gd = np.asarray(krmhd_rhs(state(g16, M, Lam, zp, zm, g, beta=1.9), eta=0.0, v_A=1.0, closure=clo).g)
    w = psi0(M)
    if clo == "symmetric":
        w[M] *= np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
    a0, a0d = g @ w, gd @ w
    pred = -pb(g16, (zp + zm) / 2, a0) * mask
    pred[0, 0, 0] = 0
    e = np.abs(pred - a0d).max() / np.abs(pred).max()
    rate = abs(2 * vint(g16, a0, a0d)) / (2 * np.sum(np.abs(a0) * np.abs(a0d)))
    okD = okD and e < 1e-12 and rate < 1e-14
    print(f"   M={M} Lam={Lam:+.3f} {clo}: |d_t a0 + {{Phi, a0}}|/max = {e:.1e}; relative d/dt of sum|a0|^2 = {rate:.1e}")
C("c12.b u = 0 slice is advected by Phi only and (pi/2) sum|a_0|^2 is conserved on krmhd_rhs (beta = 1.9)", okD)
C.done()
