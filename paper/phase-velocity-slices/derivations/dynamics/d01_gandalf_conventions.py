"""d01: the conventions used in these checks and in the paper, checked against GANDALF's own right-hand side (krmhd
0.6.0).  The script verifies that GANDALF's induction equation is the ideal one (no other sign choice fits), that
each eigenvector of the Hermite ladder gives a slice advected by Phi + s u_j Psi and streaming at s u_j, that the z+
equation has the Elsasser vorticity form, that z+ propagates toward -z, and that every slice energy is conserved by
the g right-hand side.  It supports Section 6 (GANDALF's induction equation is ideal, as the frozen-line theorem
requires); C3 is also used in Section 7.

C1  Flux freezing: GANDALF's z+- RHS (eta = 0) satisfies  R = d_t Psi + {Phi, Psi} - d_z Phi = 0  at round-off;
    the three other sign variants do not.  (So the induction equation in GANDALF is the ideal one of d00/E1.)
C2  Slice equation through GANDALF's g RHS: for every eigenpair (u_j, w_j) of A,
    w_j . (d_t g) = -{Phi + s u_j Psi, a_j} - s u_j d_z a_j,  a_j = w_j . g, with brackets 10-100x the streaming.
C3  Elsasser vorticity form of GANDALF's z+ equation (for the slices with s u = +-1, Section 7):
    lap_perp(d_t z+) - d_z lap_perp z+ = P[-{z-, lap z+} + {d_x z+, d_x z-} + {d_y z+, d_y z-}]  (k_perp != 0),
    i.e. (d_t - d_z + {z-, .}) lap z+ = (terms that vanish if z+ or z- = 0).
C4  Direction: with z- = 0 the z+ RHS is +i k_z z+, so z+(z + t): z+ propagates toward -z.
C5  Energy: the slice energies <a_j^2> are conserved by the g RHS (d/dt <a_j^2> = 0) for every j.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from dcore import Box, Checks, SQRT5, make_rhs, slices  # noqa: E402

chk = Checks()
N, M = 24, 8
box = Box(N)
rng = np.random.default_rng(11)
for Lam in (SQRT5, -SQRT5):
    beta = 1.0
    s = np.sqrt(beta)
    rhs_full, rhs_g = make_rhs(box, M, Lam, beta)
    u, V, WL = slices(M, Lam)
    for amp in (1.0, 40.0):
        phi = box.random_field(rng, (1, 4), 3)
        psi = box.random_field(rng, (1, 4), 3)
        phi *= amp / box.rms_grad(phi)
        psi *= amp / box.rms_grad(psi)
        zp, zm = phi + psi, phi - psi
        g = box.random_field(rng, (1, 5), 4, extra=(M + 1,))
        dzp, dzm, dg = (np.asarray(a) for a in rhs_full(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g)))
        dpsi = (dzp - dzm) / 2
        br = box.bracket(phi, psi)
        dzphi = 1j * box.KZ * phi
        scale = max(np.abs(dpsi).max(), np.abs(br).max(), np.abs(dzphi).max())
        res = {}
        for sb in (+1, -1):
            for sz in (+1, -1):
                Rr = dpsi + sb * br - sz * dzphi
                Rr[0, 0, 0] = 0
                res[(sb, sz)] = np.abs(Rr).max() / scale
        lab = f"Lambda={Lam:+.3f}, amp={amp}"
        if Lam > 0:
            chk(f"C1 {lab}: d_t Psi + {{Phi,Psi}} - d_z Phi = 0", res[(1, 1)] < 1e-13,
                f"{res[(1, 1)]:.1e}; other signs: {res[(1, -1)]:.1e}, {res[(-1, 1)]:.1e}, {res[(-1, -1)]:.1e}; "
                f"|{{Phi,Psi}}|/scale = {np.abs(br).max() / scale:.2f}")
            chk(f"C1 {lab}: wrong-sign variants fail", min(res[(1, -1)], res[(-1, 1)], res[(-1, -1)]) > 1e-3)
        # C2 slice equation
        a = g @ WL.T
        lhs = dg @ WL.T
        worst = 0.0
        ratios = []
        for j in range(M + 1):
            brj = box.bracket(phi + s * u[j] * psi, a[..., j])
            strj = s * u[j] * 1j * box.KZ * a[..., j]
            pred = -brj - strj
            pred[0, 0, 0] = 0
            sc = np.abs(pred).max()
            worst = max(worst, np.abs(lhs[..., j] - pred).max() / sc)
            if abs(u[j]) > 1e-8:
                ratios.append(np.sqrt(box.norm2(brj) / box.norm2(strj)))
        chk(f"C2 {lab}: slice equation through GANDALF RHS (M={M})", worst < 1e-12,
            f"max rel. mismatch {worst:.1e}; ||bracket||/||streaming|| over slices with u != 0: "
            f"{min(ratios):.1f} to {max(ratios):.1f}")
        # C5 slice energy conservation: d/dt <a_j^2> = 2 <a_j, d_t a_j> = 0
        rates = np.array([2 * box.inner(a[..., j], lhs[..., j]) for j in range(M + 1)])
        en = np.array([box.norm2(a[..., j]) for j in range(M + 1)])
        rel = np.abs(rates).max() / (np.abs(lhs).max() * np.sqrt(en.max()) * 1.0)
        chk(f"C5 {lab}: d/dt <a_j^2> = 0 for every slice", np.abs(rates / en).max() < 1e-11 * (1 + amp),
            f"max |d ln E_j/dt| = {np.abs(rates / en).max():.1e}")

# C3 Elsasser vorticity form, C4 direction
Lam = SQRT5
rhs_full, rhs_g = make_rhs(box, M, Lam, 1.0)
zp = box.random_field(rng, (1, 4), 3)
zm = box.random_field(rng, (1, 4), 3)
zp *= 3.0 / box.rms_grad(zp)
zm *= 2.0 / box.rms_grad(zm)
g0 = np.zeros((N, N, N // 2 + 1, M + 1), complex)
dzp, dzm, _ = (np.asarray(x) for x in rhs_full(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g0)))
lapz = lambda f: -box.KP2 * f  # noqa: E731
lhs = lapz(dzp) - 1j * box.KZ * lapz(zp)
rhs = (-box.bracket(zm, lapz(zp)) + box.bracket(1j * box.KX * zp, 1j * box.KX * zm)
       + box.bracket(1j * box.KY * zp, 1j * box.KY * zm))
rhs = rhs * box.mask
sel = np.broadcast_to(box.KP2 > 0, lhs.shape)
err = np.abs((lhs - rhs)[sel]).max() / np.abs(lhs[sel]).max()
chk("C3 (d_t - d_z + {z-,.}) lap z+ = {z+_x, z-_x} + {z+_y, z-_y} in GANDALF", err < 1e-12, f"{err:.1e}")
dzp1, dzm1, _ = (np.asarray(x) for x in rhs_full(jnp.asarray(zp), jnp.asarray(0 * zm), jnp.asarray(g0)))
e4 = np.abs(dzp1 - 1j * box.KZ * zp).max() / np.abs(dzp1).max()
chk("C4 z- = 0 => d_t z+ = +i k_z z+ (z+ moves toward -z) and d_t z- = 0", e4 < 1e-13 and np.abs(dzm1).max() < 1e-12,
    f"{e4:.1e}, max|d_t z-| = {np.abs(dzm1).max():.1e}")
chk.finish()
