"""d07: the two-slice law with and without the k-sum, and its relation to the Hermite flux.  Using GANDALF's
right-hand side on random, strongly nonlinear states, the script checks the k-summed two-slice law (the E x B part
drops out), the per-wavenumber law with its E x B transfer between wavenumbers, the Christoffel-Darboux identity and
the Hermite flux as a Christoffel-Darboux-filtered sum of two-slice correlation rates, and the two-slice correlation
along frozen field lines.  It supports Section 5 (two-slice law and Hermite flux); L4 also supports the two-slice
form of the single-point corollary in Section 6.

Slices a_j = w_j . g (left eigenvectors of A), C(j,j') = <a_j a_j'> (volume average = k-sum).  From GANDALF's RHS:
 L1 k-summed law:  dC(j,j')/dt = -s (u_j - u_j') <a_j' grad_par a_j>,  grad_par = d_z + {Psi, .};
    the E x B part  -<{Phi, a_j} a_j'> - <a_j {Phi, a_j'}>  vanishes.
 L2 per-k law:  dC_k/dt = -i s k_z (u_j - u_j') C_k + N^Phi_k + N^Psi_k,  C_k = a_j(k) conj(a_j'(k)),
    N^Phi_k = -[{Phi,a_j}]_k conj(a_j'(k)) - a_j(k) conj([{Phi,a_j'}]_k)  (E x B transfer between wavenumbers),
    N^Psi_k = -s u_j [{Psi,a_j}]_k conj(a_j'(k)) - s u_j' a_j(k) conj([{Psi,a_j'}]_k);
    N^Phi_k is O(1) mode by mode but sums to zero; sum_k N^Psi_k = -s (u_j - u_j') <a_j' {Psi, a_j}>.
 L3 Hermite flux:  Pi_m = 2 s p_m A_{m,m+1} <g_m grad_par g_{m+1}> = -sum_{j,j'} K_m(j,j') dC(j,j')/dt,
    K_m(j,j') = sum_{n<=m} p_n v_j(n) v_j'(n)  (Christoffel-Darboux kernel), with the CD identity
    (u_j' - u_j) K_m(j,j') = p_m A_{m,m+1} [v_j(m) v_j'(m+1) - v_j(m+1) v_j'(m)];
    per mode, the k_z-only flux (dcore.flux_kz; with Psi = 0 it is twice krmhd 0.7.0's hermite_flux) is the
    CD-filtered linear dephasing:
    Pi^z_m(k) = -sum K_m(j,j') Re[-i s k_z (u_j - u_j') C_k(j,j')].
 L4 frozen field lines: in the frozen-line problem C(j,j'; t) = <a_j(0), T_{s(u_j' - u_j) t} a_j'(0)>, with
    T_l = exp(-l grad_par(0)) the translation along the frozen field lines (checked for the pair u_i = -u_j, with
    classical RK4 and GANDALF's bracket).  The discrete grad_par(0) is anti-Hermitian (2/3-dealiased bracket) and the
    RK4 polynomial R4 has real coefficients, so R4(-c dt L)^H = R4(c dt L) for L = -grad_par(0); for u_i = -u_j the
    two transported slices then give exactly the relative transport computed with dt/2:
    <a_i(t) a_j(t)>_dt = <a_i(0), T a_j(0)>_{dt/2} to round-off.  The mismatch at equal dt is therefore the RK4 error
    of the transport T alone, and falls at fourth order in dt.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dcore import (Box, Checks, SQRT5, flux_full, flux_kz, ladder, log_run, make_rhs, pweights, slices,  # noqa: E402
                   timer)

chk = Checks()
N = 24
box = Box(N)
rng = np.random.default_rng(5)
for M, Lam, beta in ((8, SQRT5, 1.0), (9, -SQRT5, 2.0)):
    s = np.sqrt(beta)
    A = ladder(M, Lam)
    p = pweights(M, Lam)
    u, V, WL = slices(M, Lam)
    rhs_full, rhs_g = make_rhs(box, M, Lam, beta)
    phi = box.random_field(rng, (1, 4), 3)
    psi = box.random_field(rng, (1, 4), 3)
    phi *= 5.0 / box.rms_grad(phi)
    psi *= 5.0 / box.rms_grad(psi)
    g = box.random_field(rng, (1, 5), 4, extra=(M + 1,))
    dg = np.asarray(rhs_g(jnp.asarray(phi + psi), jnp.asarray(phi - psi), jnp.asarray(g)))
    a = g @ WL.T
    ad = dg @ WL.T
    J = M + 1
    lab = f"M={M}, Lambda={Lam:+.3f}, s={s:.3f}"
    # L1
    Cdot = np.array([[box.inner(ad[..., i], a[..., j]) + box.inner(a[..., i], ad[..., j]) for j in range(J)]
                     for i in range(J)])
    gp = [box.gradpar(psi, a[..., i]) for i in range(J)]
    law = np.array([[-s * (u[i] - u[j]) * box.inner(a[..., j], gp[i]) for j in range(J)] for i in range(J)])
    e1 = np.abs(Cdot - law).max() / np.abs(Cdot).max()
    chk(f"L1 {lab}: dC/dt = -s (u - u') <a_u' grad_par a_u> (k-summed)", e1 < 1e-12, f"{e1:.1e}")
    brphi = [box.bracket(phi, a[..., i]) for i in range(J)]
    exb = np.array([[-box.inner(brphi[i], a[..., j]) - box.inner(a[..., i], brphi[j]) for j in range(J)]
                    for i in range(J)])
    e_exb = np.abs(exb).max() / np.abs(Cdot).max()
    chk(f"L1 {lab}: E x B contribution to dC/dt vanishes", e_exb < 1e-12, f"{e_exb:.1e}")
    # L2 per-k law (complex, rfft half space); per-mode sums use the rfft weights w and the Re part
    brpsi = [box.bracket(psi, a[..., i]) for i in range(J)]
    worst, nphi_rel, sum_nphi, sum_npsi = 0.0, 0.0, 0.0, 0.0
    for i in range(J):
        for j in range(J):
            Ck = a[..., i] * np.conj(a[..., j])
            Cdk = ad[..., i] * np.conj(a[..., j]) + a[..., i] * np.conj(ad[..., j])
            lin = -1j * s * box.KZ * (u[i] - u[j]) * Ck
            nphi = -brphi[i] * np.conj(a[..., j]) - a[..., i] * np.conj(brphi[j])
            npsi = -s * u[i] * brpsi[i] * np.conj(a[..., j]) - s * u[j] * a[..., i] * np.conj(brpsi[j])
            sc = np.abs(Cdk).max()
            worst = max(worst, np.abs(Cdk - (lin + nphi + npsi)).max() / sc)
            nphi_rel = max(nphi_rel, np.abs(nphi).max() / sc)
            sum_nphi = max(sum_nphi, abs(np.sum(box.w * np.real(nphi))) / box.N ** 6 / np.abs(Cdot).max())
            tgt = -s * (u[i] - u[j]) * box.inner(a[..., j], brpsi[i])
            sum_npsi = max(sum_npsi, abs(np.sum(box.w * np.real(npsi)) / box.N ** 6 - tgt) / np.abs(Cdot).max())
    chk(f"L2 {lab}: per-k law dC_k/dt = linear dephasing + N^Phi_k + N^Psi_k (every mode, every pair)",
        worst < 1e-12, f"{worst:.1e}")
    chk(f"L2 {lab}: E x B transfer N^Phi_k is O(1) mode by mode", nphi_rel > 0.1, f"max |N^Phi_k| / max |dC_k/dt| = {nphi_rel:.2f}")
    chk(f"L2 {lab}: sum_k N^Phi_k = 0", sum_nphi < 1e-12, f"{sum_nphi:.1e}")
    chk(f"L2 {lab}: sum_k N^Psi_k = -s (u - u') <a_u' {{Psi, a_u}}>", sum_npsi < 1e-12, f"{sum_npsi:.1e}")
    # L3 flux relation and CD identity
    Pi = flux_full(box, g, psi, s, A, p)
    worst_cd, worst_flux = 0.0, 0.0
    for m in range(M):
        K = (V[: m + 1].T * p[: m + 1]) @ V[: m + 1]
        cd = (u[None, :] - u[:, None]) * K
        rhs_cd = p[m] * A[m, m + 1] * (np.outer(V[m], V[m + 1]) - np.outer(V[m + 1], V[m]))
        worst_cd = max(worst_cd, np.abs(cd - rhs_cd).max())
        worst_flux = max(worst_flux, abs(Pi[m] + np.sum(K * Cdot)) / np.abs(Pi).max())
    chk(f"L3 {lab}: Christoffel-Darboux identity for K_m", worst_cd < 1e-12, f"{worst_cd:.1e}")
    chk(f"L3 {lab}: Pi_m (full grad_par) = -sum K_m(j,j') dC(j,j')/dt", worst_flux < 1e-11, f"{worst_flux:.1e}")
    Kall = [(V[: m + 1].T * p[: m + 1]) @ V[: m + 1] for m in range(M)]
    Pz = flux_kz(box, g, s, A, p, per_mode=True) * box.w[..., None] / box.N ** 6
    worst_kz = 0.0
    for m in (0, 1, M // 2, M - 1):
        acc = np.zeros(box.KZ.shape[:1] + box.KY.shape[1:2] + box.KX.shape[2:3])
        for i in range(J):
            for j in range(J):
                Ck = a[..., i] * np.conj(a[..., j])
                acc = acc - Kall[m][i, j] * np.real(-1j * s * box.KZ * (u[i] - u[j]) * Ck)
        acc = acc * box.w / box.N ** 6
        worst_kz = max(worst_kz, np.abs(acc - Pz[..., m]).max() / np.abs(Pz[..., m]).max())
    chk(f"L3 {lab}: per mode, k_z-only flux = CD-filtered linear dephasing", worst_kz < 1e-11, f"{worst_kz:.1e}")
    Pzs = flux_kz(box, g, s, A, p)
    print(f"    {lab}: field-line share of the flux |Pi - Pi^z| / |Pi| per m: "
          + " ".join(f"{abs(Pi[m] - Pzs[m]) / abs(Pi[m]):.2f}" for m in range(M)))

# L4 frozen field lines: relative displacement along frozen lines
M, Lam, beta = 8, SQRT5, 1.0
s = 1.0
u, V, WL = slices(M, Lam)
psi = box.random_field(rng, (1, 3), 2)
psi *= 1.0 / box.rms_grad(psi)
a0 = box.random_field(rng, (1, 3), 2, extra=(2,))
i, j = 2, 6
t_end = 0.1
kz = jnp.asarray(box.KZ)
mask = jnp.asarray(box.mask)
psij = jnp.asarray(psi)
from krmhd.physics import poisson_bracket_3d  # noqa: E402


def make_transport(dt):
    @jax.jit
    def transport(f, speed):
        """RK4 step of d_t f = -speed grad_par(0) f with GANDALF's bracket."""
        def r(h):
            return -speed * (1j * kz * h + poisson_bracket_3d(psij, h, box.jargs[0], box.jargs[1], N, N, N, mask))
        k1 = r(f)
        k2 = r(f + 0.5 * dt * k1)
        k3 = r(f + 0.5 * dt * k2)
        k4 = r(f + dt * k3)
        return f + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return transport


errs, lhs_dt, rhs_dt = [], [], []
for dt in (0.001, 0.0005, 0.00025):
    transport = make_transport(dt)
    fi, fj, fr = jnp.asarray(a0[..., 0]), jnp.asarray(a0[..., 1]), jnp.asarray(a0[..., 1])
    clock = timer()
    for _ in range(int(round(t_end / dt))):
        fi = transport(fi, s * u[i])
        fj = transport(fj, s * u[j])
        fr = transport(fr, s * (u[j] - u[i]))          # relative displacement only
    log_run("d07_two_slice_law.py", "L4 transport along frozen lines, 3 fields (RK4 with GANDALF's Poisson bracket, "
            "not the full right-hand side)", N, M, int(round(t_end / dt)), clock(), dict(dt=dt, T=t_end))
    lhs = box.inner(np.asarray(fi), np.asarray(fj))
    rhs = box.inner(a0[..., 0], np.asarray(fr))
    errs.append(abs(lhs - rhs) / abs(rhs))
    lhs_dt.append(lhs)
    rhs_dt.append(rhs)
    print(f"    L4 dt = {dt}: <a_i(t) a_j(t)> = {lhs:.10e}, <a_i(0), T a_j(0)> = {rhs:.10e}, rel diff {errs[-1]:.2e}")
rate = np.log2(errs[0] / errs[1]), np.log2(errs[1] / errs[2])
# exact discrete identity for the pair u_i = -u_j: <a_i a_j> with step dt equals <a_i(0), T a_j(0)> with step dt/2
half = max(abs(lhs_dt[k] - rhs_dt[k + 1]) / abs(rhs_dt[k + 1]) for k in range(2))
print(f"    L4 <a_i(t) a_j(t)> at dt vs <a_i(0), T a_j(0)> at dt/2: max rel diff {half:.1e}")
# krmhd 0.6.0 (2/3 mask with 3|n| <= N, one more shell at N = 24): rel diff 2.1e-7 -> 1.4e-8 -> 8.6e-10 (orders 3.94,
# 3.98). The earlier absolute bound 1e-9 at dt = 0.00025 was not part of the statement; it failed at 1.9e-9 with the
# corrected mask of krmhd 0.7.0 (gandalf#172), which changes the transported fields near the cutoff (<a_i(t) a_j(t)>
# moves from 3.912e-4 to 4.059e-4), and with them the size of the RK4 error
chk(f"L4 frozen lines: <a_i(t) a_j(t)> = <a_i(0), T_(s (u_j - u_i) t) a_j(0)> (u_i={u[i]:.3f}, u_j={u[j]:.3f}); the "
    "mismatch is the RK4 error of the transport (equal to round-off with dt/2 for the transport, 4th order in dt)",
    abs(u[i] + u[j]) < 1e-12 and half < 1e-12 and min(rate) > 3.5,
    f"rel diff {errs[0]:.1e} -> {errs[1]:.1e} -> {errs[2]:.1e}, observed orders {rate[0]:.2f}, {rate[1]:.2f}; "
    f"dt vs dt/2 for the transport {half:.1e}")
chk.finish()
