"""d10: the slices with s u = -1 and s u = +1 are each advected by one Elsasser field only.  With beta chosen so that
s u_j = -1 for an eigenvalue of GANDALF's ladder, the script checks through GANDALF's right-hand side that the
s u = -1 slice is advected by z- alone and the s u = +1 slice by z+ alone while every other slice sees both, and, with
gandalf_step in an exact pure-z+ RMHD solution, that the s u = -1 slice translates rigidly with z+ while every other
slice is distorted.  It supports Section 7 (slices advected by a single Elsasser field).

Slice equation: (d_t + s u d_z) a_u + {Phi + s u Psi, a_u} = 0 with Phi + s u Psi = [(1 + su) z+ + (1 - su) z-]/2, so
   s u = -1:  (d_t - d_z) a + {z-, a} = 0      (moves toward -z with z+, advected by z- only),
   s u = +1:  (d_t + d_z) a + {z+, a} = 0      (moves toward +z with z-, advected by z+ only).
GANDALF's Elsasser vorticity (d01/C3): (d_t - d_z + {z-, .}) lap z+ = {d_x z+, d_x z-} + {d_y z+, d_y z-}, so the
s u = -1 slice is transported exactly like the Elsasser field Z+ (vector form d_t Z+ - d_z Z+ + Z-.grad Z+ = -grad p),
without the pressure-type term.

E1 GANDALF RHS with beta chosen so that s u_j = -1 for an eigenvalue u_j of A: d_t a_j = -{z-, a_j} + d_z a_j, and it
   does not change when z+ is replaced by an unrelated field; the mirror slice (s u = +1) does not see z-.
E2 Generic slices (s u != +-1) see both fields.
E3 gandalf_step (Lawson-RK4, eta = nu = 0) from z- = 0 (an exact nonlinear RMHD solution: z+ translates rigidly):
   the s u = -1 slice translates rigidly with z+ (a_j(t) = a_j(0) e^{i k_z t}) while every other slice is distorted.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from krmhd.timestepping import gandalf_step  # noqa: E402
from dcore import Box, Checks, SQRT5, log_run, make_rhs, make_state, slices, timer  # noqa: E402

chk = Checks()
N, M, Lam = 24, 8, SQRT5
box = Box(N)
rng = np.random.default_rng(8)
u, V, WL = slices(M, Lam)
jn = int(np.argmin(np.abs(u + 1.0)))                 # eigenvalue closest to -1
jp = int(np.argmin(np.abs(u - u[jn] * -1)))          # its mirror partner (+|u_jn|)
s = -1.0 / u[jn]
beta = s ** 2
print(f"    M = {M}, Lambda = sqrt5: eigenvalues u = {np.round(u, 4).tolist()}")
print(f"    choose u_j = {u[jn]:.6f} and beta = {beta:.6f} so that s u_j = -1 (mirror slice u = {u[jp]:.6f}, s u = {s * u[jp]:+.6f})")
rhs_full, rhs_g = make_rhs(box, M, Lam, beta)
zp = box.random_field(rng, (1, 4), 3)
zm = box.random_field(rng, (1, 4), 3)
zp *= 3 / box.rms_grad(zp)
zm *= 2 / box.rms_grad(zm)
zp2 = box.random_field(rng, (1, 4), 3)
zp2 *= 5 / box.rms_grad(zp2)
zm2 = box.random_field(rng, (1, 4), 3)
zm2 *= 4 / box.rms_grad(zm2)
g = box.random_field(rng, (1, 5), 4, extra=(M + 1,))
a = g @ WL.T


def adot(zp_, zm_):
    return np.asarray(rhs_g(jnp.asarray(zp_), jnp.asarray(zm_), jnp.asarray(g))) @ WL.T


d0 = adot(zp, zm)
d_newzp = adot(zp2, zm)
d_newzm = adot(zp, zm2)
pred_n = -box.bracket(zm, a[..., jn]) + 1j * box.KZ * a[..., jn]
pred_n[0, 0, 0] = 0
pred_p = -box.bracket(zp, a[..., jp]) - 1j * box.KZ * a[..., jp]
pred_p[0, 0, 0] = 0
rel = lambda x, y: float(np.abs(x - y).max() / np.abs(y).max())  # noqa: E731
chk("E1 s u = -1 slice: d_t a = -{z-, a} + d_z a (GANDALF RHS)", rel(d0[..., jn], pred_n) < 1e-13, f"{rel(d0[..., jn], pred_n):.1e}")
chk("E1 s u = -1 slice: unchanged when z+ is replaced", rel(d_newzp[..., jn], d0[..., jn]) < 1e-13,
    f"{rel(d_newzp[..., jn], d0[..., jn]):.1e}")
chk("E1 s u = +1 slice: d_t a = -{z+, a} - d_z a (GANDALF RHS)", rel(d0[..., jp], pred_p) < 1e-13, f"{rel(d0[..., jp], pred_p):.1e}")
chk("E1 s u = +1 slice: unchanged when z- is replaced", rel(d_newzm[..., jp], d0[..., jp]) < 1e-13,
    f"{rel(d_newzm[..., jp], d0[..., jp]):.1e}")
others = [j for j in range(M + 1) if j not in (jn, jp)]
sens = min(min(rel(d_newzp[..., j], d0[..., j]), rel(d_newzm[..., j], d0[..., j])) for j in others)
chk("E2 every other slice sees both z+ and z-", sens > 1e-2, f"min sensitivity {sens:.2f}")

# E3: pure z+ wave (z- = 0), Lawson-RK4
N3 = 16
b3 = Box(N3)
rng3 = np.random.default_rng(9)
zp3 = b3.random_field(rng3, (1, 3), 2)
zp3 *= 2 / b3.rms_grad(zp3)
g3 = b3.random_field(rng3, (1, 3), 2, extra=(M + 1,))
st = make_state(b3, zp3, 0 * zp3, g3, Lam, beta)
dt, nst = 0.005, 100
clock = timer()
for _ in range(nst):
    st = gandalf_step(st, dt, eta=0.0, v_A=1.0, nu=0.0, scheme="lawson_rk4")
log_run("d10_elsasser_slices.py", "pure z+ wave: s u = -1 slice co-moves rigidly (gandalf_step lawson_rk4)", N3, M, nst,
        clock(), dict(dt=dt, T=dt * nst))
T = dt * nst
ph = np.exp(1j * b3.KZ * T)
e_zp = rel(np.asarray(st.z_plus), zp3 * ph)
e_zm = float(np.abs(np.asarray(st.z_minus)).max())
a0 = g3 @ WL.T
aT = np.asarray(st.g) @ WL.T
dev = []
for j in range(M + 1):
    shift = np.exp(-1j * s * u[j] * b3.KZ * T)     # pure streaming along z at speed s u_j
    dev.append(np.sqrt(b3.norm2(aT[..., j] - a0[..., j] * shift) / b3.norm2(a0[..., j])))
dev = np.array(dev)
print(f"    E3 t = {T}: |z+(t) - z+(0) e^(i k_z t)| = {e_zp:.1e}, max|z-(t)| = {e_zm:.1e}")
print("    E3 deviation of each slice from rigid streaming along z, ||a_j(t) - a_j(0) e^(-i s u_j k_z t)|| / ||a_j(0)||:")
print("       " + "  ".join(f"su={s * u[j]:+.3f}: {dev[j]:.1e}" for j in range(M + 1)))
chk("E3 z- stays zero and z+ translates rigidly toward -z", e_zp < 1e-12 and e_zm < 1e-12, f"{e_zp:.1e}, {e_zm:.1e}")
chk("E3 the s u = -1 slice co-moves rigidly with z+", dev[jn] < 1e-12, f"{dev[jn]:.1e}")
chk("E3 every other slice is distorted by z+", min(dev[j] for j in range(M + 1) if j != jn) > 1e-2,
    f"min {min(dev[j] for j in range(M + 1) if j != jn):.2f}")

chk.finish()
