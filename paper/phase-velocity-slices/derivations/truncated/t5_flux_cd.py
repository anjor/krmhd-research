"""t5_flux_cd: checks the Hermite flux Pi_{m+1/2} of the truncated hierarchy as Christoffel-Darboux-filtered
dephasing of slice pairs: the collisionless budget on krmhd_rhs with the field-line part of grad_par (and what
krmhd.diagnostics.hermite_flux omits), the generalised Christoffel-Darboux identity, the slice form of the flux, and
the sine-kernel limit of the kernel. Supports Section 5 (Christoffel-Darboux and sine-transform forms of the flux)
and Section 8 and Appendix C of the paper.

Definitions (real-space integrals over the periodic box; Pi the weight, b_m = (Pi A)_{m,m+1}):
  E_{<=m} = sum_{n<=m} p_n int g_n^2,   Pi_{m+1/2} = s b_m int (g_m grad_par g_{m+1} - g_{m+1} grad_par g_m),
  grad_par = d_z + {Psi, .},  C_jl = int a_j a_l,  Cdot_jl = -s (u_j - u_l) int a_l grad_par a_j,
  K_m(j,l) = sum_{n<=m} p_n R_nj R_nl  (generalised Christoffel-Darboux kernel).
A  Collisionless budget on krmhd_rhs: dE_{<=m}/dt = -Pi_{m+1/2} exactly (E x B advection drops out), with the
   field-line part {Psi,.} included; s * krmhd.diagnostics.hermite_flux reproduces only the d_z part (m >= 1).
B  Christoffel-Darboux identity b_m (R_mj R_{m+1,l} - R_{m+1,j} R_ml) = (u_l - u_j) K_m(j,l) (both closures), hence
   Pi_{m+1/2} = -sum_{jl} K_m(j,l) Cdot_jl / (n_j n_l), and Cdot_jl = dC_jl/dt on krmhd_rhs.
C  The kernel is a low-pass filter: for |u| << sqrt(2m), K_m(j,l)/sqrt(n_j n_l) ~ sqrt(D_j D_l) sin(k_m (u_j-u_l))/(pi (u_j-u_l)),
   k_m = sqrt(2m+1), D_j the local slice spacing (sine kernel), so Pi_{m+1/2} is carried by slice pairs with
   |u_j - u_l| ~ pi/sqrt(2m).
"""
from __future__ import annotations

import numpy as np

from tcore import (Checker, section, ladder, weight, slices, make_grid, random_field, make_state, bracket, dz,
                   ksum_weights, SQRT5, jnp)
from krmhd.timestepping import krmhd_rhs
from krmhd.diagnostics import hermite_flux

C = Checker("t5_flux_cd")
N = 16
grid = make_grid(N)
mask = np.asarray(grid.dealias_mask)
wk = ksum_weights(grid)


def integ(f: np.ndarray, h: np.ndarray) -> float:
    return float(np.sum(wk * np.real(np.conj(f) * h)))


section("A/B. Budget, field-line part, and the slice (Christoffel-Darboux) form on krmhd_rhs")
okA = True
okA2 = True
okB1 = True
okB2 = True
okB3 = True
for M in (9, 12):
    for Lam in (SQRT5, -SQRT5):
        for clo in ("zero", "symmetric"):
            rng = np.random.default_rng(11 * M + (1 if clo == "zero" else 2))
            zp = random_field(grid, rng, amp=150.0)
            zm = random_field(grid, rng, amp=150.0)
            g = random_field(grid, rng, slope=-1.0, extra_shape=(M + 1,))
            beta = 1.4
            s = np.sqrt(beta)
            st = make_state(grid, M, Lam, zp, zm, g, beta_i=beta)
            gdot = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure=clo).g)
            A = ladder(M, Lam, clo)
            p = weight(M, Lam, clo)
            b = np.array([p[m] * A[m, m + 1] for m in range(M)])
            Psi = (zp - zm) / 2

            def gpar(f):
                return (dz(grid, f) + bracket(grid, Psi, f)) * mask

            gp = np.stack([gpar(g[..., m]) for m in range(M + 1)], axis=-1)
            gz = np.stack([dz(grid, g[..., m]) * mask for m in range(M + 1)], axis=-1)
            Pi_full = np.array([s * b[m] * (integ(g[..., m], gp[..., m + 1]) - integ(g[..., m + 1], gp[..., m]))
                                for m in range(M)])
            Pi_z = np.array([s * b[m] * (integ(g[..., m], gz[..., m + 1]) - integ(g[..., m + 1], gz[..., m]))
                             for m in range(M)])
            dE = np.array([2 * sum(p[n] * integ(g[..., n], gdot[..., n]) for n in range(m + 1)) for m in range(M)])
            scale = np.abs(Pi_full).max()
            e1 = np.abs(dE + Pi_full).max() / scale
            e1z = np.abs(dE + Pi_z).max() / scale
            # GANDALF diagnostic: per-k flux -k_z sqrt(2(m+1)) Im[g_{m+1} g_m^*]; summed with rfft weights, times s
            hf = np.asarray(hermite_flux(st))
            hf_sum = s * np.array([np.sum(wk * hf[..., m]) for m in range(M)])
            e_hf = np.abs(hf_sum[1:] - Pi_z[1:]).max() / np.abs(Pi_z).max()
            okA = okA and e1 < 1e-12
            okA2 = okA2 and e1z > 1e-3 and e_hf < 1e-12
            # slice form
            uj, R, W, n = slices(M, Lam, clo)
            a = np.einsum("zyxm,mj->zyxj", g, W)
            adot = np.einsum("zyxm,mj->zyxj", gdot, W)
            ap = np.stack([gpar(a[..., j]) for j in range(M + 1)], axis=-1)
            # Cdot_jl = -s (u_j - u_l) int a_l gpar a_j  and its check against d/dt int a_j a_l
            Ilj = np.array([[integ(a[..., l], ap[..., j]) for j in range(M + 1)] for l in range(M + 1)])  # [l, j]
            Cdot = -s * (uj[:, None] - uj[None, :]) * Ilj.T
            Cdot_dyn = np.array([[integ(adot[..., j], a[..., l]) + integ(a[..., j], adot[..., l])
                                  for l in range(M + 1)] for j in range(M + 1)])
            e2 = np.abs(Cdot - Cdot_dyn).max() / np.abs(Cdot_dyn).max()
            okB2 = okB2 and e2 < 1e-12
            worst_cd = 0.0
            Pi_sl = np.zeros(M)
            for m in range(M):
                Km = (R[:m + 1].T * p[:m + 1]) @ R[:m + 1]
                lhs = b[m] * (np.outer(R[m], R[m + 1]) - np.outer(R[m + 1], R[m]))
                rhs = (uj[None, :] - uj[:, None]) * Km
                worst_cd = max(worst_cd, np.abs(lhs - rhs).max() / np.abs(lhs).max())
                Pi_sl[m] = -np.sum(Km * Cdot / np.outer(n, n))
            e3 = np.abs(Pi_sl - Pi_full).max() / scale
            okB1 = okB1 and worst_cd < 1e-11
            okB3 = okB3 and e3 < 1e-11
            print(f"   M={M} Lam={Lam:+.3f} {clo:9s}: |dE/dt + Pi| = {e1:.1e}; with d_z part only {e1z:.1e} "
                  f"(field-line share max|Pi_full - Pi_z|/max|Pi| = {np.abs(Pi_full - Pi_z).max() / scale:.2f}); "
                  f"s*hermite_flux vs d_z part (m>=1) {e_hf:.1e}; m=0: hermite_flux/(Pi_z) = {hf_sum[0] / Pi_z[0]:.4f} "
                  f"(1/(1-1/Lam) = {1 / (1 - 1 / Lam):.4f}); CD identity {worst_cd:.1e}; Cdot vs dynamics {e2:.1e}; "
                  f"slice form vs direct {e3:.1e}")
C.check("A1 dE_{<=m}/dt = -s b_m int(g_m gpar g_{m+1} - g_{m+1} gpar g_m) on krmhd_rhs (brackets on, both closures)",
        okA)
C.check("A2 the d_z part alone does not close the budget; s * hermite_flux equals the d_z part for m >= 1", okA2)
C.check("B1 generalised Christoffel-Darboux identity b_m(R_mj R_{m+1,l} - R_{m+1,j} R_ml) = (u_l - u_j) K_m(j,l)", okB1)
C.check("B2 Cdot_jl = -s (u_j - u_l) int a_l gpar a_j equals dC_jl/dt on krmhd_rhs (E x B advection drops out)", okB2)
C.check("B3 Pi_{m+1/2} = -sum_jl K_m(j,l) Cdot_jl / (n_j n_l) (CD-filtered slice dephasing)", okB3)

section("C. The Christoffel-Darboux kernel is a sine-kernel low-pass filter in u")
okC = True
for Lam in (np.inf, SQRT5, -SQRT5):
    M = 512
    uj, R, W, n = slices(M, Lam)
    R = R * np.sign(R[0])[None, :]      # sign convention r_j[0] > 0, i.e. r_j[m] = rho_m(u_j) up to a positive factor
    p = weight(M, Lam)
    D = np.gradient(uj)
    sel = np.where(np.abs(uj) < 1.5)[0]
    row = f"   Lam={Lam:+.4f}, M={M}:"
    for m in (64, 128, 256):
        Km = (R[:m + 1].T * p[:m + 1]) @ R[:m + 1]
        Kn = Km / np.sqrt(np.outer(n, n))
        km = np.sqrt(2 * m + 1)
        du = uj[sel][:, None] - uj[sel][None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            sk = np.where(np.abs(du) > 0, np.sin(km * du) / (np.pi * du), km / np.pi)
        pred = np.sqrt(np.outer(D[sel], D[sel])) * sk
        err = np.abs(Kn[np.ix_(sel, sel)] - pred).max() / np.abs(pred).max()
        row += f" m={m}: max|K_hat - sine kernel|/max = {err:.3f};"
        okC = okC and err < (0.05 if np.isinf(Lam) else 0.15)
    print(row)
C.check("C1 K_m(j,l)/sqrt(n_j n_l) ~ sqrt(D_j D_l) sin(sqrt(2m+1)(u_j - u_l))/(pi (u_j - u_l)) for |u| < 1.5", okC)
C.finish()
