"""c02_slices: independent check of the exact slice decomposition of GANDALF's hierarchy and of the two-slice law,
using GANDALF's own eigensystem (krmhd.hermite.compute_streaming_eigensystem: P_inv rows are the left eigenvectors)
instead of the symmetrised construction in ../../truncated/tcore.py, at production-size M (64 even, with a u = 0
slice, and 65 odd), beta_i = 0.37 and 2.3, Lam = -sqrt5 and 3, both closures. Supports Section 8 (theorem on the
truncated hierarchy; hypercollisions as the only slice coupling) and Section 5 (two-slice law) of the paper.

a. krmhd_rhs projected on GANDALF's left eigenvectors obeys d_t a_j = -{Phi + s u_j Psi, a_j} - s u_j d_z a_j
   (strong brackets; also with z+- that are NOT 2/3-dealiased, since only bilinearity is used).
b. GANDALF's left eigenvectors are proportional to Pi r_j with Pi = diag(1 - 1/Lam, 1, ..., 1, p_M) (p_M = 1 or
   sqrt(M)/(sqrt(M)+sqrt(M+1))): an independent confirmation of the symmetriser.
c. Two-slice law: d/dt int a_j a_l = -s (u_j - u_l) int a_l grad_par a_j on krmhd_rhs (control: d_z only fails).
d. gandalf_step, M = 64, Lam = -sqrt5, copy closure, beta = 2.3, eta (r=2) and eta_z on, 40 steps: a single-slice start
   (u = 0 slice and fastest slice) stays in its slice at nu = 0; with nu > 0 (n = 6) the leakage matches the first-order
   coupling nu T (P_inv D P)_{l j0}.
"""
from __future__ import annotations

import time

import numpy as np

from cc import Checks, hdr, SQRT5, grid3, rand_field, state, pb, ddz, vint, gandalf_eig, jnp

C = Checks("c02_slices")
from krmhd.timestepping import krmhd_rhs, gandalf_step

N = 16
g16 = grid3(N)
mask = np.asarray(g16.dealias_mask)

hdr("a/b/c. Slice equation, symmetriser and two-slice law on krmhd_rhs with GANDALF's eigenvectors")
okA = okA2 = okB = okC = okCc = True
for M in (64, 65):
    for Lam in (-SQRT5, 3.0):
        for clo in ("zero", "symmetric"):
            beta = 0.37
            s = np.sqrt(beta)
            rng = np.random.default_rng(M * 7 + (1 if clo == "zero" else 2) + (5 if Lam > 0 else 0))
            zp = rand_field(g16, rng, amp=400.0)
            zm = rand_field(g16, rng, amp=250.0)
            g = rand_field(g16, rng, slope=-1.0, extra=(M + 1,))
            gd = np.asarray(krmhd_rhs(state(g16, M, Lam, zp, zm, g, beta=beta), eta=0.0, v_A=1.0, closure=clo).g)
            T, u, P, Pinv = gandalf_eig(M, Lam, clo)
            a = np.einsum("zyxm,jm->zyxj", g, Pinv)
            ad = np.einsum("zyxm,jm->zyxj", gd, Pinv)
            Phi, Psi = (zp + zm) / 2, (zp - zm) / 2
            worst, brshare = 0.0, 0.0
            for j in range(M + 1):
                b = pb(g16, Phi + s * u[j] * Psi, a[..., j])
                st_ = s * u[j] * ddz(g16, a[..., j])
                pred = (-b - st_) * mask
                pred[0, 0, 0] = 0
                sc = np.abs(b).max() + np.abs(st_).max()
                worst = max(worst, np.abs(pred - ad[..., j]).max() / sc)
                brshare = max(brshare, np.abs(b).max() / sc)
            okA = okA and worst < 1e-11
            # b: left eigenvectors vs Pi r_j
            p = np.ones(M + 1); p[0] = 1 - 1 / Lam
            if clo == "symmetric":
                p[M] = np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1))
            Wp = p[:, None] * P
            cos = np.abs(np.einsum("mj,jm->j", Wp, Pinv)) / (np.linalg.norm(Wp, axis=0) * np.linalg.norm(Pinv, axis=1))
            eB = np.abs(1 - cos).max()
            okB = okB and eB < 1e-12
            # c: two-slice law on a subset of pairs (all pairs among 12 slices incl. extremes and u ~ 0)
            js = np.unique(np.r_[0, 1, M // 4, M // 2 - 1, M // 2, M // 2 + 1, 3 * M // 4, M - 1, M,
                                 np.argmin(np.abs(u)), np.argmin(np.abs(u - 1 / s)), np.argmin(np.abs(u + 1 / s))])
            gpar = lambda f: (ddz(g16, f) + pb(g16, Psi, f)) * mask
            err, errc, scale = 0.0, 0.0, 0.0
            for j in js:
                gpj = gpar(a[..., j])
                gzj = ddz(g16, a[..., j]) * mask
                for l in js:
                    lhs = vint(g16, ad[..., j], a[..., l]) + vint(g16, a[..., j], ad[..., l])
                    rhs = -s * (u[j] - u[l]) * vint(g16, a[..., l], gpj)
                    rhs_c = -s * (u[j] - u[l]) * vint(g16, a[..., l], gzj)
                    scale = max(scale, abs(lhs))
                    err = max(err, abs(lhs - rhs))
                    errc = max(errc, abs(lhs - rhs_c))
            okC = okC and err / scale < 1e-12
            okCc = okCc and errc / scale > 1e-2
            print(f"   M={M} Lam={Lam:+.3f} {clo:9s} beta={beta}: slice eq. {worst:.1e} (max bracket share {brshare:.2f}); "
                  f"1-|cos(w_j, Pi r_j)| {eB:.1e}; two-slice law {err / scale:.1e} (d_z-only control {errc / scale:.2f})")
C("c02.a slice equation on krmhd_rhs with GANDALF's own left eigenvectors (M = 64, 65; Lam = -sqrt5, 3; both closures)", okA)
C("c02.b GANDALF's left eigenvectors are Pi r_j (Pi = D_c P_Lam)", okB)
C("c02.c two-slice law dC_jl/dt = -s (u_j - u_l) int a_l grad_par a_j (E x B absent); control without {Psi,.} fails",
  okC and okCc)
# non-dealiased z+-: the slice equation needs only bilinearity of GANDALF's bracket
M, Lam, clo, beta = 64, SQRT5, "zero", 1.0
rng = np.random.default_rng(3)
zp = rand_field(g16, rng, amp=300.0, dealias=False)
zm = rand_field(g16, rng, amp=300.0, dealias=False)
g = rand_field(g16, rng, slope=-1.0, extra=(M + 1,))
gd = np.asarray(krmhd_rhs(state(g16, M, Lam, zp, zm, g, beta=beta), eta=0.0, v_A=1.0, closure=clo).g)
T, u, P, Pinv = gandalf_eig(M, Lam, clo)
a = np.einsum("zyxm,jm->zyxj", g, Pinv)
ad = np.einsum("zyxm,jm->zyxj", gd, Pinv)
Phi, Psi = (zp + zm) / 2, (zp - zm) / 2
worst = 0.0
for j in range(M + 1):
    b = pb(g16, Phi + u[j] * Psi, a[..., j])
    st_ = u[j] * ddz(g16, a[..., j])
    pred = (-b - st_) * mask
    pred[0, 0, 0] = 0
    worst = max(worst, np.abs(pred - ad[..., j]).max() / (np.abs(b).max() + np.abs(st_).max()))
C("c02.a' the slice equation also holds for z+- with power outside the 2/3 band (bilinearity only)", worst < 1e-11,
  f"{worst:.1e}")

hdr("d. gandalf_step: slice leakage at M = 64 (copy closure, Lam = -sqrt5, beta = 2.3), 40 steps")
M, Lam, clo, beta = 64, -SQRT5, "symmetric", 2.3
s = np.sqrt(beta)
T, u, P, Pinv = gandalf_eig(M, Lam, clo)
rng = np.random.default_rng(99)
zp = rand_field(g16, rng, amp=30.0)
zm = rand_field(g16, rng, amp=20.0)
f0 = rand_field(g16, rng, slope=-1.0)
from krmhd.timestepping import compute_cfl_timestep
from krmhd.spectral import derivative_x, derivative_y, rfftn_inverse


def maxgrad(f):
    fx = np.asarray(rfftn_inverse(derivative_x(jnp.asarray(f), g16.kx), N, N, N))
    fy = np.asarray(rfftn_inverse(derivative_y(jnp.asarray(f), g16.ky), N, N, N))
    return float(np.sqrt(fx ** 2 + fy ** 2).max())


Vfast = maxgrad((zp + zm) / 2) + s * u[-1] * maxgrad((zp - zm) / 2)
kmax = np.sqrt(2) * 2 * np.pi * (N // 3)
dt = 0.3 / (kmax * Vfast)               # explicit stage safely stable for every slice
print(f"   u_max = {u[-1]:.4f}; fastest-slice speed bound {Vfast:.2f}; dt = {dt:.3e} "
      f"(GANDALF dt_cfl = {compute_cfl_timestep(state(g16, M, Lam, zp, zm, np.zeros((N, N, N // 2 + 1, M + 1), complex), beta=beta), 1.0):.3e})")
okD = True
lk = {}
runs = []
for scheme in ("imex_rk222", "lawson_rk4"):
    for j0 in (int(np.argmin(np.abs(u))), M):
        for nu in (0.0, 2.0):
            g = f0[..., None] * P[:, j0][None, None, None, :]
            st = state(g16, M, Lam, zp, zm, g, beta=beta, nu=nu)
            t0 = time.time()
            nst = 40
            for _ in range(nst):
                st = gandalf_step(st, dt, eta=1.0, v_A=1.0, nu=nu, hyper_r=2, hyper_n=6, scheme=scheme, eta_z=0.5,
                                  hyper_rz=2, closure=clo)
            wall = time.time() - t0
            runs.append((scheme, j0, nu, nst, wall))
            a = np.einsum("zyxm,jm->zyxj", np.asarray(st.g), Pinv)
            nrm = np.sqrt((np.abs(a) ** 2).sum(axis=(0, 1, 2)))
            leak = np.delete(nrm, j0).max() / nrm[j0]
            lk[(scheme, j0, nu)] = (leak, nrm, nst * dt)
            if nu == 0.0:
                okD = okD and leak < 1e-12
            print(f"   {scheme} start in slice u = {u[j0]:+.3f}, nu = {nu}: max_(l != j0) |a_l|/|a_j0| = {leak:.2e} "
                  f"({nst} steps, {wall:.1f} s)")
C("c02.d nu = 0: no leakage between slices through gandalf_step (IMEX, Lawson; eta, eta_z, copy closure, M = 64)", okD)
# first-order hypercollisional coupling with GANDALF's own eigenvectors: d a/dt = -nu (P_inv D P) a
mm = np.arange(M + 1)
Dd = np.where(mm >= 2, (mm / M) ** 6, 0.0)
Kc = Pinv @ (Dd[:, None] * P)
okE = True
for scheme in ("imex_rk222", "lawson_rk4"):
    for j0 in (int(np.argmin(np.abs(u))), M):
        leak, nrm, Ttot = lk[(scheme, j0, 2.0)]
        # predicted |a_l(T)| / |a_j0(T)| at first order: nu T |K_{l j0}| (ignoring the j0 decay, a second-order effect)
        pred = 2.0 * Ttot * np.abs(np.delete(Kc[:, j0], j0)).max()
        okE = okE and abs(leak / pred - 1) < 0.05
        print(f"   {scheme} u = {u[j0]:+.3f}: measured leakage {leak:.4e}, first-order nu T max|(P_inv D P)_(l j0)| = "
              f"{pred:.4e} (ratio {leak / pred:.4f}); diagonal damping nu (P_inv D P)_jj = {2.0 * Kc[j0, j0]:.3e}")
C("c02.e with hypercollisions the leakage is the first-order coupling nu T (P_inv D P)_{l j0}", okE)
print("GANDALF integrations: grid 16^3, M = 64, Lam = -sqrt5, copy closure, beta = 2.3, eta = 1 (r = 2), eta_z = 0.5:")
for r in runs:
    print(f"   {r[0]} start slice {r[1]} nu={r[2]} steps={r[3]} dt={dt:.2e} wall={r[4]:.1f}s")
C.done()
