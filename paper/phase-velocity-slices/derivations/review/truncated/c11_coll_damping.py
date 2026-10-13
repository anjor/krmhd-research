"""c11_coll_damping: checks the first-order hypercollisional damping of each slice, gamma_j = nu (P_inv D P)_jj with
D = diag((m/M)^n), m >= 2, from GANDALF's own eigensystem. The fastest slices, which bind the explicit-stage
stability (../../truncated/t7a_cfl_slices.py, c06_cfl.py), are also the most strongly damped by hypercollisions, so
the IMEX threshold dt^3 <= 8 eta/(k_perp,max V_j)^4, which includes only hyper-resistivity, omits a stabilising term
of order nu for exactly those slices. Supports Section 8 (hypercollisions and slice damping) and Appendix D of the
paper.
"""
from __future__ import annotations

import numpy as np

from cc import Checks, hdr, SQRT5, gandalf_eig

C = Checks("c11_coll_damping")
ok = True
for M in (64, 128):
    for n in (2, 6):
        T, u, P, Pinv = gandalf_eig(M, SQRT5, "zero")
        m = np.arange(M + 1)
        D = np.where(m >= 2, (m / M) ** n, 0.0)
        gam = np.einsum("jm,m,mj->j", Pinv, D, P)
        picks = [np.argmin(np.abs(u - x)) for x in (0.0, 1.0, 3.0, 6.0)] + [M]
        print(f"   M={M} n={n}: gamma_j/nu at u = " + ", ".join(f"{u[j]:+.2f}: {gam[j]:.4f}" for j in picks)
              + f"; nu/(2n+1) = {1 / (2 * n + 1):.4f}")
        ok = ok and gam[M] > 4 * gam[picks[0]] and gam[M] > 0.6
C("hypercollisional damping of the u_max slice exceeds that of the u ~ 0 slice by > 4x and is > 0.6 nu (n = 2, 6; M = 64, 128)", ok)
C.done()
