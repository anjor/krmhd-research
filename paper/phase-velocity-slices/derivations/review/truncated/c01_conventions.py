"""c01_conventions: independent checks of the conventions that the paper and the truncated-hierarchy scripts share
with CMM and GANDALF: GANDALF's streaming matrix is the matrix of L G = u G - (u F_M/Lam) M0[G] in CMM's Hermite
basis (by mpmath quadrature), GANDALF's Hermite basis functions, the sign of its Poisson bracket, the streaming
factor s = sqrt(beta_i), the propagation direction of z+ through gandalf_step, and the signs of the Alfvenic
equations. Supports Section 2 (conventions) and Section 8 of the paper.

Independent routes:

1. The matrix of L G = u G - (u F_M/Lam) int G du' in CMM's basis phi_m = H_m(u) F_M/sqrt(2^m m!) (B27, B30), computed by
   mpmath quadrature with mpmath's Hermite polynomials (no hand-written three-term recurrence), equals GANDALF's
   compute_streaming_matrix (zero closure). The copy closure differs by sqrt((M+1)/2) e_M e_{M-1}^T only.
2. GANDALF's own Hermite basis (krmhd.hermite.hermite_basis_function) is H_m(x) exp(-x^2/2)/sqrt(2^m m! sqrt(pi)),
   x = v/v_th, i.e. G/sqrt(F_M) with F_M ~ exp(-v^2/v_th^2); the Rayleigh quotient of x in that basis is A at Lam = inf.
3. Bracket sign: poisson_bracket_3d(sin 2pi x, cos 2pi y) = -4 pi^2 cos 2pi x sin 2pi y, i.e. {f,h} = f_x h_y - f_y h_x.
4. s = sqrt(beta_i): the streaming part of krmhd_rhs scales as sqrt(beta) (beta = 0.37 and 4.0, z+- = 0).
5. z+ direction through gandalf_step (both schemes), not only krmhd_rhs: a single z+ mode acquires phase +k_z t.
6. KRMHD field equations: krmhd_rhs gives d_t Psi = grad_par Phi and d_t lap Phi = -{Phi, lap Phi} + grad_par lap Psi
   (CMM (gkeKRMHD) and (vorticity) with A = -Psi/(rho v_A), varphi = 2 Phi/(rho v_th)), so Phi, Psi carry CMM's signs.
"""
from __future__ import annotations

import time

import numpy as np
import mpmath as mp

from cc import Checks, hdr, SQRT5, grid3, rand_field, state, pb, ddz, jnp, vint

C = Checks("c01_conventions")
from krmhd.hermite import compute_streaming_matrix, hermite_basis_function

hdr("1. Matrix of L in CMM's basis by quadrature (mpmath, 30 digits)")
mp.mp.dps = 30
M = 12
FM = lambda u: mp.exp(-u * u) / mp.sqrt(mp.pi)
pm = lambda m, u: mp.hermite(m, u) / mp.sqrt(mp.mpf(2) ** m * mp.factorial(m))
Jq = mp.matrix(M + 1, M + 1)
for n in range(M + 1):
    for m in range(M + 1):
        Jq[n, m] = mp.quad(lambda u: pm(n, u) * u * pm(m, u) * FM(u), [-mp.inf, 0, mp.inf])
# int phi_m du = delta_{m0};  int p_n u F_M du = delta_{n1}/sqrt2  (also by quadrature)
I0 = [mp.quad(lambda u: pm(m, u) * FM(u), [-mp.inf, mp.inf]) for m in range(M + 1)]
I1 = [mp.quad(lambda u: pm(n, u) * u * FM(u), [-mp.inf, mp.inf]) for n in range(M + 1)]
worst = 0.0
for Lam in (SQRT5, -SQRT5, 3.0, np.inf):
    Lq = np.array([[float(Jq[n, m] - (0 if np.isinf(Lam) else (I1[n] * I0[m]) / mp.mpf(Lam))) for m in range(M + 1)]
                   for n in range(M + 1)])
    Ag = np.asarray(compute_streaming_matrix(M, Lam, "zero"))
    worst = max(worst, np.abs(Lq - Ag).max())
    Ac = np.asarray(compute_streaming_matrix(M, Lam, "symmetric"))
    dc = Ac - Ag
    dexp = np.zeros_like(dc); dexp[M, M - 1] = np.sqrt((M + 1) / 2)
    worst = max(worst, np.abs(dc - dexp).max())
print(f"   M = {M}: max |quadrature matrix of L - compute_streaming_matrix| over Lam = +-sqrt5, 3, inf "
      f"(and copy-closure difference) = {worst:.1e}")
C("c01.1 GANDALF's A is the matrix of L G = uG - u F_M M0/Lam in CMM's basis H_m F_M/sqrt(2^m m!) (u = v/v_th, "
  "F_M ~ e^-u^2)", worst < 1e-14, f"{worst:.1e}")

hdr("2. GANDALF's Hermite basis functions")
x = np.linspace(-9, 9, 4001)
dx = x[1] - x[0]
B = np.array([np.asarray(hermite_basis_function(m, jnp.asarray(x), 1.0)) for m in range(M + 1)])
gram = (B * dx) @ B.T
X = (B * x * dx) @ B.T
e_gram = np.abs(gram - np.eye(M + 1)).max()
e_X = np.abs(X - np.asarray(compute_streaming_matrix(M, np.inf, "zero"))).max()
ref = np.array([[float(mp.hermite(m, xx) * mp.exp(-xx * xx / 2) / mp.sqrt(2 ** m * mp.factorial(m) * mp.sqrt(mp.pi)))
                 for xx in x[::400]] for m in range(M + 1)])
e_fun = np.abs(B[:, ::400] - ref).max()
print(f"   orthonormality {e_gram:.1e}; <psi_m|x|psi_n> - A(Lam=inf) {e_X:.1e}; basis vs H_m e^(-x^2/2)/norm {e_fun:.1e}")
C("c01.2 hermite_basis_function = H_m(x) e^{-x^2/2}/sqrt(2^m m! sqrt pi) (= G/sqrt(F_M) basis, F_M ~ e^{-x^2}) and "
  "its x-matrix is A(Lam = inf)", e_gram < 1e-10 and e_X < 1e-10 and e_fun < 1e-12)

hdr("3. Sign of GANDALF's Poisson bracket")
N = 16
g16 = grid3(N)
xs = np.arange(N) / N
Xr = xs[None, None, :] * np.ones((N, N, N))
Yr = xs[None, :, None] * np.ones((N, N, N))
f = np.fft.rfftn(np.sin(2 * np.pi * Xr), axes=(0, 1, 2))
h = np.fft.rfftn(np.cos(2 * np.pi * Yr), axes=(0, 1, 2))
br = np.fft.irfftn(pb(g16, f, h), s=(N, N, N), axes=(0, 1, 2))
exp_ = -4 * np.pi ** 2 * np.cos(2 * np.pi * Xr) * np.sin(2 * np.pi * Yr)
e_br = np.abs(br - exp_).max() / np.abs(exp_).max()
C("c01.3 {f,h} = d_x f d_y h - d_y f d_x h", e_br < 1e-12, f"{e_br:.1e}")

hdr("4. Streaming factor s = sqrt(beta_i)")
from krmhd.timestepping import krmhd_rhs, gandalf_step
rng = np.random.default_rng(7)
M4 = 7
g = rand_field(g16, rng, slope=-1.0, extra=(M4 + 1,))
z0 = np.zeros((N, N, N // 2 + 1), complex)
A4 = np.asarray(compute_streaming_matrix(M4, SQRT5, "zero"))
okS = True
for beta in (0.37, 4.0):
    gd = np.asarray(krmhd_rhs(state(g16, M4, SQRT5, z0, z0, g, beta=beta), eta=0.0, v_A=1.0).g)
    pred = -np.sqrt(beta) * ddz(g16, np.einsum("mn,zyxn->zyxm", A4, g)[..., :]) * np.asarray(g16.dealias_mask)[..., None]
    pred[0, 0, 0] = 0
    e = np.abs(gd - pred).max() / np.abs(pred).max()
    okS = okS and e < 1e-13
    print(f"   beta = {beta}: |krmhd_rhs.g - (-sqrt(beta) d_z A g)| / max = {e:.1e}")
C("c01.4 streaming term = -sqrt(beta_i) d_z (A g)", okS)

hdr("5. z+ propagation direction through gandalf_step")
g8 = grid3(8)
zp = np.zeros((8, 8, 5), complex)
zp[1, 1, 1] = 0.3                      # k = 2pi (1, 1, 1): z+ ~ exp(i(kx x + ky y + kz z)) + c.c.
zm = np.zeros_like(zp)
okZ = True
kz1 = float(np.asarray(g8.kz)[1])
runs = []
for scheme in ("imex_rk222", "lawson_rk4"):
    st = state(g8, 4, SQRT5, zp, zm, np.zeros((8, 8, 5, 5), complex))
    dt, nst = 0.01, 7
    t0 = time.time()
    for _ in range(nst):
        st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, scheme=scheme)
    ph = np.angle(np.asarray(st.z_plus)[1, 1, 1] / zp[1, 1, 1])
    runs.append((scheme, nst, dt, time.time() - t0))
    okZ = okZ and abs(ph - kz1 * dt * nst) < 1e-12
    print(f"   {scheme}: phase of z+(k) after t = {dt * nst}: {ph:+.12f}; +k_z t = {kz1 * dt * nst:+.12f}")
C("c01.5 z+ ~ exp(i k_z (z + t)): propagates toward -z under gandalf_step (both schemes)", okZ)

hdr("6. Alfvenic equations: d_t Psi = grad_par Phi, d_t lap Phi = -{Phi, lap Phi} + grad_par lap Psi")
rng = np.random.default_rng(11)
zp = rand_field(g16, rng, amp=300.0)
zm = rand_field(g16, rng, amp=300.0)
st = state(g16, 4, SQRT5, zp, zm, np.zeros((N, N, N // 2 + 1, 5), complex))
r = krmhd_rhs(st, eta=0.0, v_A=1.0)
dzp, dzm = np.asarray(r.z_plus), np.asarray(r.z_minus)
Phi, Psi = (zp + zm) / 2, (zp - zm) / 2
mask = np.asarray(g16.dealias_mask)
kx = np.asarray(g16.kx)[None, None, :]
ky = np.asarray(g16.ky)[None, :, None]
lap = lambda q: -(kx ** 2 + ky ** 2) * q
psi_dot = (dzp - dzm) / 2
phi_dot = (dzp + dzm) / 2
pred_psi = ddz(g16, Phi) + pb(g16, Psi, Phi)
pred_vort = -pb(g16, Phi, lap(Phi)) + ddz(g16, lap(Psi)) + pb(g16, Psi, lap(Psi))
sel = mask.astype(bool)
sel[0, 0, 0] = False
e1 = np.abs((psi_dot - pred_psi)[sel]).max() / np.abs(pred_psi[sel]).max()
e2 = np.abs((lap(phi_dot) - pred_vort)[sel]).max() / np.abs(pred_vort[sel]).max()
nl_share = np.abs(pb(g16, Psi, Phi)[sel]).max() / np.abs(pred_psi[sel]).max()
print(f"   |Psi_dot - (d_z Phi + {{Psi,Phi}})| = {e1:.1e}; |lap Phi_dot - (...)| = {e2:.1e} (bracket share {nl_share:.2f})")
C("c01.6 krmhd_rhs: d_t Psi = grad_par Phi and the vorticity equation with CMM's signs (B0 along +z)",
  e1 < 1e-12 and e2 < 1e-12 and nl_share > 0.1)
print("GANDALF integrations (scheme, steps, dt, wall s), grid 8^3, M = 4, Lam = sqrt5, single z+ mode:")
for r in runs:
    print(f"   {r[0]} steps={r[1]} dt={r[2]:.2e} wall={r[3]:.1f}s")
C.done()
