"""c05_flux: independent check of the Christoffel-Darboux form of the Hermite flux at production size M = 64/65 with
GANDALF's own right eigenvectors P (arbitrary normalisation), beta = 0.37, copy and zero closures, Lam = -sqrt5 and
3: the collisionless budget, the Christoffel-Darboux identity, the slice form of the flux, the sine-kernel limit at
M = 1024, and the pair weight of the flux as a sine transform. Supports Section 5 (Christoffel-Darboux and
sine-transform forms of the flux) and Section 8 of the paper.

a. Collisionless budget dE_{<=m}/dt = -Pi_{m+1/2}, Pi = s b_m int (g_m grad_par g_{m+1} - g_{m+1} grad_par g_m), on krmhd_rhs.
b. Generalised Christoffel-Darboux identity b_m (P_mj P_{m+1,l} - P_{m+1,j} P_ml) = (u_l - u_j) K_m(j,l),
   K_m(j,l) = sum_{n<=m} p_n P_nj P_nl (normalisation-covariant).
c. Slice form Pi_{m+1/2} = -sum_jl K_m(j,l) Cdot_jl/(n_j n_l), n_j = P_j^T Pi P_j, a_j = (Pi P_j).g.
d. Diagonal pairs carry no flux (Cdot_jj = 0) and the share of |K Cdot| in pairs with |u_j - u_l| < 2 pi/sqrt(2m+1).
e. Sine kernel at Lam = inf, M = 1024: K_m(j,l)/sqrt(n_j n_l) vs sqrt(D_j D_l) sin(sqrt(2m+1)(u_j-u_l))/(pi (u_j-u_l)).
"""
from __future__ import annotations

import numpy as np

from cc import Checks, hdr, SQRT5, grid3, rand_field, state, pb, ddz, vint, gandalf_eig

C = Checks("c05_flux")
from krmhd.timestepping import krmhd_rhs

N = 16
g16 = grid3(N)
mask = np.asarray(g16.dealias_mask)
okA = okB = okC = okD = True
for M, Lam, clo in ((64, -SQRT5, "symmetric"), (65, 3.0, "zero")):
    beta = 0.37
    s = np.sqrt(beta)
    rng = np.random.default_rng(M + 1)
    zp = rand_field(g16, rng, amp=300.0)
    zm = rand_field(g16, rng, amp=200.0)
    g = rand_field(g16, rng, slope=-1.0, extra=(M + 1,))
    gd = np.asarray(krmhd_rhs(state(g16, M, Lam, zp, zm, g, beta=beta), eta=0.0, v_A=1.0, closure=clo).g)
    T, u, P, Pinv = gandalf_eig(M, Lam, clo)
    p = np.ones(M + 1); p[0] = 1 - 1 / Lam
    if clo == "symmetric":
        p[M] = np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
    b = np.array([p[m] * T[m, m + 1] for m in range(M)])
    Psi = (zp - zm) / 2
    gpar = lambda f: (ddz(g16, f) + pb(g16, Psi, f)) * mask
    gp = np.stack([gpar(g[..., m]) for m in range(M + 1)], -1)
    Pi_dir = np.array([s * b[m] * (vint(g16, g[..., m], gp[..., m + 1]) - vint(g16, g[..., m + 1], gp[..., m])) for m in range(M)])
    dE = np.cumsum([2 * p[n] * vint(g16, g[..., n], gd[..., n]) for n in range(M)])
    eA = np.abs(dE + Pi_dir).max() / np.abs(Pi_dir).max()
    okA = okA and eA < 1e-12
    # CD identity with GANDALF's P
    eB = 0.0
    for m in (0, 1, M // 2, M - 1):
        K = (P[:m + 1].T * p[:m + 1]) @ P[:m + 1]
        lhs = b[m] * (np.outer(P[m], P[m + 1]) - np.outer(P[m + 1], P[m]))
        rhs = (u[None, :] - u[:, None]) * K
        eB = max(eB, np.abs(lhs - rhs).max() / np.abs(lhs).max())
    okB = okB and eB < 1e-10
    # slice form
    n = np.einsum("mj,m,mj->j", P, p, P)
    a = np.einsum("zyxm,mj->zyxj", g, p[:, None] * P)
    ap = np.stack([gpar(a[..., j]) for j in range(M + 1)], -1)
    I = np.array([[vint(g16, a[..., l], ap[..., j]) for j in range(M + 1)] for l in range(M + 1)])   # I[l, j]
    Cdot = -s * (u[:, None] - u[None, :]) * I.T
    eC = 0.0
    shares = []
    for m in range(M):
        K = (P[:m + 1].T * p[:m + 1]) @ P[:m + 1]
        contrib = -K * Cdot / np.outer(n, n)
        Pi_sl = contrib.sum()
        eC = max(eC, abs(Pi_sl - Pi_dir[m]) / np.abs(Pi_dir).max())
        if m in (8, 16, 32):
            near = np.abs(u[:, None] - u[None, :]) < 2 * np.pi / np.sqrt(2 * m + 1)
            shares.append((m, np.abs(contrib[near]).sum() / np.abs(contrib).sum()))
    okC = okC and eC < 1e-11
    diag = np.abs(np.diag(Cdot)).max() / np.abs(Cdot).max()
    okD = okD and diag == 0.0
    print(f"   M={M} Lam={Lam:+.3f} {clo:9s}: budget {eA:.1e}; CD identity {eB:.1e}; slice form {eC:.1e}; "
          f"max|Cdot_jj|/max|Cdot| = {diag:.1e}; share of sum|K Cdot/nn| from pairs with |u_j - u_l| < 2pi/sqrt(2m+1): "
          + ", ".join(f"m={m}: {sh:.2f}" for m, sh in shares))
C("c05.a collisionless budget with the full grad_par (M = 64, 65)", okA)
C("c05.b generalised Christoffel-Darboux identity with GANDALF's eigenvectors", okB)
C("c05.c flux = CD-filtered slice dephasing; d. diagonal pairs carry none", okC and okD)

hdr("e. Sine kernel at Lam = inf, M = 1024 (independent of the M = 512 check in t5_flux_cd.py)")
from scipy.linalg import eigh_tridiagonal
M = 1024
uu, V = eigh_tridiagonal(np.zeros(M + 1), np.sqrt(np.arange(1, M + 1) / 2.0))
V = V * np.sign(V[0])[None, :]
D = np.gradient(uu)
sel = np.where(np.abs(uu) < 1.5)[0]
okE = True
for m in (128, 256, 512):
    K = V[:m + 1].T @ V[:m + 1]          # n_j = 1 in this normalisation
    du = uu[sel][:, None] - uu[sel][None, :]
    km = np.sqrt(2 * m + 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sk = np.where(np.abs(du) > 0, np.sin(km * du) / (np.pi * du), km / np.pi)
    pred = np.sqrt(np.outer(D[sel], D[sel])) * sk
    err = np.abs(K[np.ix_(sel, sel)] - pred).max() / np.abs(pred).max()
    okE = okE and err < 0.01
    print(f"   m={m}: max|K_m - sine kernel|/max = {err:.4f}")
C("c05.e sine-kernel form of the CD kernel for |u| < 1.5 (Lam = inf)", okE)
# the pair weight that multiplies the dephasing rate int a_l grad_par a_j is K_m(j,l)(u_j - u_l)/sqrt(n_j n_l):
# it does not decay with |u_j - u_l| (|sin| has the same envelope at every separation), so it is not localised at
# |u_j - u_l| ~ 1/sqrt(2m)
m = 128
K = V[:m + 1].T @ V[:m + 1]
sel2 = np.where(np.abs(uu) < 3.0)[0]
du = uu[sel2][:, None] - uu[sel2][None, :]
wgt = np.abs(K[np.ix_(sel2, sel2)] * du) / np.sqrt(np.outer(D[sel2], D[sel2]))
env = []
for lo, hi in ((0.0, 0.25), (0.5, 1.0), (1.0, 2.0), (2.0, 4.0), (4.0, 6.0)):
    bsel = (np.abs(du) >= lo) & (np.abs(du) < hi)
    env.append(wgt[bsel].max())
    print(f"   m={m}: max pair weight |K(u_j-u_l)|/sqrt(D_j D_l n_j n_l) for {lo} <= |u_j-u_l| < {hi}: {wgt[bsel].max():.4f} (1/pi = {1/np.pi:.4f})")
C("c05.f the pair weight in the flux does not decay with |u_j - u_l| (envelope ~1/pi out to |u_j - u_l| = 6): it is a sine "
  "transform at kappa = sqrt(2m+1), not a filter localised at |u_j - u_l| ~ 1/sqrt(2m)", min(env[2:]) >= env[0] and
  all(abs(e * np.pi - 1) < 0.2 for e in env))
C.done()
