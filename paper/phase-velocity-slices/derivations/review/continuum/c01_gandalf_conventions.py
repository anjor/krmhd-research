"""c01 (independent check): GANDALF's right-hand side krmhd_rhs against an independent NumPy evaluation of the
KRMHD equations of Chandran, Mallet & Meyrand (2026; CMM below): the Alfvenic right-hand sides with the bracket and
Elsasser conventions of the paper, the kinetic branch equation with a Hermite matrix built by quadrature (both
closures, Lambda = +-sqrt5, two values of beta_i), the slice equation for the left eigenvectors of that matrix, and
the direction of propagation of z+ under time stepping with gandalf_step. It supports Section 2 (conventions) and
Section 3 (slice theorem) of the paper.

Independent route (nothing imported from the derivation scripts in ../../continuum):
  * wavenumbers rebuilt with numpy from the box lengths; products evaluated on a 2x zero-padded grid (no reliance on
    GANDALF's 2/3-rule aliasing argument), then projected on GANDALF's retained modes;
  * Hermite matrix of L built by Gauss-Hermite quadrature of u * e_n(u) - u F_M M0[e_n]/Lambda against H_m/sqrt(2^m m!)
    (CMM B27, B30), not from the ladder formula;
  * Alfvenic equations written from CMM (3.23)-(3.24) in Phi, Psi with v_A = 1:
        d_t Psi = d_z Phi + {Psi, Phi},   d_t lap Phi = -{Phi, lap Phi} + d_z lap Psi + {Psi, lap Psi},
    with Phi = (z+ + z-)/2, Psi = (z+ - z-)/2, {f,g} = f_x g_y - f_y g_x (CMM 2.9);
  * branch equation (B4) in u:  d_t g = -{Phi, g} - s d_z(A g) - s {Psi, A g},  s = sqrt(beta_i);
  * slices from numpy.linalg.eig of A^T (left eigenvectors), no symmetrisation.
Checks:
  G1  GANDALF dz+/dt, dz-/dt equal the CMM (3.23)-(3.24) right-hand sides (nonlinear, random fields, Lz != Lx);
  G2  GANDALF dg/dt equals the branch equation with the quadrature matrix A (zero and symmetric closures; Lambda = +-sqrt5,
      beta_i = 1, 0.3);
  G3  slices w_j . g of GANDALF's dg/dt equal -{Phi + s u_j Psi, a_j} - s u_j d_z a_j (own brackets);
  G4  direction: gandalf_step on a pure z+ wave for t = 0.5 moves the pattern toward -z (time integration, 16 steps).
Exit 1 on any failure.
"""
import sys
import time
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import krmhd_rhs, gandalf_step
import krmhd

fails = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


t0 = time.time()
print(f"krmhd {krmhd.__version__}, jax {jax.__version__}")

# ------------------------------------------------------------------ own spectral machinery (rfft layout [z, y, x])
Nx = Ny = Nz = 16
Lx, Ly, Lz = 2 * np.pi, 2 * np.pi, 3.0 * np.pi      # Lz != Lx to catch any kz/kx mix-up
grid = SpectralGrid3D.create(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz)
kx1 = 2 * np.pi / Lx * np.arange(Nx // 2 + 1)
ky1 = 2 * np.pi / Ly * np.fft.fftfreq(Ny, 1.0 / Ny)
kz1 = 2 * np.pi / Lz * np.fft.fftfreq(Nz, 1.0 / Nz)
check("own wavenumbers equal GANDALF's grid.kx/ky/kz",
      np.allclose(kx1, np.asarray(grid.kx)) and np.allclose(ky1, np.asarray(grid.ky)) and np.allclose(kz1, np.asarray(grid.kz)))
KZ, KY, KX = np.meshgrid(kz1, ky1, kx1, indexing='ij')
mask = np.asarray(grid.dealias_mask).astype(float)
print(f"  retained modes: {int(mask.sum())} of {mask.size}")


def to_pad(F, p=2):
    """rfft coefficients [Nz,Ny,Nx/2+1] (GANDALF: unnormalised forward) -> real field on a p-times finer grid."""
    Pz, Py, Px = p * Nz, p * Ny, p * Nx
    G = np.zeros((Pz, Py, Px // 2 + 1), complex)
    iz = np.where(np.arange(Nz) < Nz // 2, np.arange(Nz), np.arange(Nz) + (Pz - Nz))
    iy = np.where(np.arange(Ny) < Ny // 2, np.arange(Ny), np.arange(Ny) + (Py - Ny))
    Fm = F * mask                                    # Nyquist and |k| > 2/3 k_max are zero anyway
    G[np.ix_(iz, iy, np.arange(Nx // 2 + 1))] = Fm
    return np.fft.irfftn(G, s=(Pz, Py, Px), axes=(0, 1, 2)) * p**3


def from_pad(f, p=2):
    Pz, Py, Px = p * Nz, p * Ny, p * Nx
    G = np.fft.rfftn(f, axes=(0, 1, 2)) / p**3
    iz = np.where(np.arange(Nz) < Nz // 2, np.arange(Nz), np.arange(Nz) + (Pz - Nz))
    iy = np.where(np.arange(Ny) < Ny // 2, np.arange(Ny), np.arange(Ny) + (Py - Ny))
    return G[np.ix_(iz, iy, np.arange(Nx // 2 + 1))] * mask


dx = lambda F: 1j * KX * F
dy = lambda F: 1j * KY * F
dz = lambda F: 1j * KZ * F
lap = lambda F: -(KX**2 + KY**2) * F
K2 = KX**2 + KY**2


def ilap(F):
    out = np.where(K2 > 0, -F / np.where(K2 > 0, K2, 1.0), 0.0)
    return out


def pb(F, G):
    """{f, g} = f_x g_y - f_y g_x (CMM 2.9), exact product on the padded grid, projected on the retained modes."""
    return from_pad(to_pad(dx(F)) * to_pad(dy(G)) - to_pad(dy(F)) * to_pad(dx(G)))


rng = np.random.default_rng(20261013)


def rand_field(amp, slope=-1.5, no_kperp0=True):
    f = rng.standard_normal((Nz, Ny, Nx))
    F = np.fft.rfftn(f, axes=(0, 1, 2))
    kk = np.sqrt(KX**2 + KY**2 + KZ**2)
    F = F * mask * np.where(kk > 0, kk, 1.0)**slope
    if no_kperp0:
        F = F * (K2 > 0)
    F[0, 0, 0] = 0
    F = F / np.abs(to_pad(F)).max() * amp
    return F


# ------------------------------------------------------------------ Hermite matrix of L by quadrature (independent)
def hermite_matrix(M, Lam, closure):
    """A[m, n] = int H_m(u)/sqrt(2^m m!) (L e_n)(u) du,  e_n = H_n F_M/sqrt(2^n m!),  L G = u G - u F_M M0[G]/Lambda.
    Gauss-Hermite with weight e^{-u^2}: int q(u) F_M du = sum w_i q(x_i)/sqrt(pi) (exact for deg < 2*nq)."""
    from numpy.polynomial.hermite import hermgauss, hermval
    from math import factorial
    nq = M + 12
    x, w = hermgauss(nq)
    norm = lambda m: np.sqrt(2.0**m * factorial(m))
    H = np.array([hermval(x, np.eye(M + 4)[m]) for m in range(M + 4)])     # physicists' H_m at nodes
    A = np.zeros((M + 3, M + 2))                          # rows 0..M+2, columns 0..M+1 of the exact infinite matrix
    for n in range(M + 2):
        en_over_FM = H[n] / norm(n)                       # e_n / F_M
        M0n = (w * en_over_FM).sum() / np.sqrt(np.pi)     # = delta_n0
        LeFM = x * en_over_FM - x * M0n / Lam            # (L e_n)/F_M
        for m in range(M + 3):
            A[m, n] = (w * H[m] * LeFM).sum() / np.sqrt(np.pi) / norm(m)
    At = A[:M + 1, :M + 1].copy()                         # zero closure: g_{M+1} = 0
    if closure == "symmetric":                            # g_{M+1} := g_{M-1}: row M gains (L e_{M+1})_M g_{M-1}
        At[M, M - 1] += A[M, M + 1]
    return At, A


# sanity of the quadrature matrix (not a check of the claim): structure
A0, Afull = hermite_matrix(10, np.sqrt(5.0), "zero")
print("  quadrature A (M=10, Lambda=sqrt5): A[0,1]=%.15f A[1,0]=%.15f A[1,2]=%.15f A[2,1]=%.15f, max|diag|=%.1e" %
      (A0[0, 1], A0[1, 0], A0[1, 2], A0[2, 1], np.abs(np.diag(A0)).max()))

# ------------------------------------------------------------------ G1: Alfvenic sector
print("G1  Alfvenic right-hand sides (CMM 3.23-3.24, Phi = (z+ + z-)/2, Psi = (z+ - z-)/2, v_A = 1)")
zp = rand_field(12.0); zm = rand_field(8.0)
M = 12
gg = np.stack([rand_field(1.0, no_kperp0=False) for _ in range(M + 1)], axis=-1)
st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(gg), M=M, beta_i=1.0, v_th=1.0,
                nu=0.0, Lambda=float(np.sqrt(5.0)), time=0.0, grid=grid)
d = krmhd_rhs(st, eta=0.0, v_A=1.0, closure="zero")
Phi = (zp + zm) / 2; Psi = (zp - zm) / 2
dPsi = dz(Phi) + pb(Psi, Phi)
dlapPhi = -pb(Phi, lap(Phi)) + dz(lap(Psi)) + pb(Psi, lap(Psi))
dPhi = ilap(dlapPhi)
dzp_own = dPhi + dPsi; dzm_own = dPhi - dPsi
dzp_own[0, 0, 0] = 0; dzm_own[0, 0, 0] = 0
e1 = np.abs(np.asarray(d.z_plus) - dzp_own).max() / np.abs(dzp_own).max()
e2 = np.abs(np.asarray(d.z_minus) - dzm_own).max() / np.abs(dzm_own).max()
nl = np.abs(pb(Psi, Phi)).max(); lin = np.abs(dz(Phi)).max()
check("GANDALF dz+/dt = CMM RHS", e1 < 1e-12, f"rel {e1:.1e}")
check("GANDALF dz-/dt = CMM RHS", e2 < 1e-12, f"rel {e2:.1e}; max|{{Psi,Phi}}|/max|d_z Phi| = {nl / lin:.1f}")
# the opposite field-line sign (grad_par = d_z - {Psi,.}) must NOT match: makes the check non-vacuous
dPsi_bad = dz(Phi) - pb(Psi, Phi)
e_bad = np.abs(np.asarray(d.z_plus) - (dPhi + dPsi_bad)).max() / np.abs(dzp_own).max()
check("control: grad_par = d_z - {Psi,.} does not match GANDALF", e_bad > 1e-2, f"rel {e_bad:.2f}")

# ------------------------------------------------------------------ G2, G3: kinetic sector and slices
print("G2/G3  kinetic sector and slices")
for Lam in (np.sqrt(5.0), -np.sqrt(5.0)):
    for beta in (1.0, 0.3):
        for clo in ("zero", "symmetric"):
            M = 12
            zp = rand_field(40.0); zm = rand_field(30.0)
            gg = np.stack([rand_field(1.0, no_kperp0=False) for _ in range(M + 1)], axis=-1)
            st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(gg), M=M, beta_i=beta,
                            v_th=1.0, nu=0.0, Lambda=float(Lam), time=0.0, grid=grid)
            dg_G = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure=clo).g)
            A, _ = hermite_matrix(M, Lam, clo)
            s = np.sqrt(beta)
            Phi = (zp + zm) / 2; Psi = (zp - zm) / 2
            Ag = gg @ A.T
            own = np.stack([-pb(Phi, gg[..., m]) - s * dz(Ag[..., m]) - s * pb(Psi, Ag[..., m]) for m in range(M + 1)], -1)
            own[0, 0, 0, :] = 0
            e = np.abs(dg_G - own).max() / np.abs(own).max()
            br = np.abs(np.stack([pb(Phi, gg[..., m]) for m in range(M + 1)], -1)).max()
            sm = np.abs(s * 1j * KZ[..., None] * Ag).max()
            check(f"G2 Lambda={Lam:+.3f} beta={beta} {clo}: GANDALF dg/dt = branch eq. with quadrature A", e < 1e-12,
                  f"rel {e:.1e}; max|{{Phi,g}}|/max|s d_z Ag| = {br / sm:.1f}")
            # controls: flipping the field-line sign or the Lambda term must break the agreement
            bad1 = np.stack([-pb(Phi, gg[..., m]) - s * dz(Ag[..., m]) + s * pb(Psi, Ag[..., m]) for m in range(M + 1)], -1)
            Abad, _ = hermite_matrix(M, -Lam, clo)
            Agb = gg @ Abad.T
            bad2 = np.stack([-pb(Phi, gg[..., m]) - s * dz(Agb[..., m]) - s * pb(Psi, Agb[..., m]) for m in range(M + 1)], -1)
            eb1 = np.abs(dg_G - bad1).max() / np.abs(own).max(); eb2 = np.abs(dg_G - bad2).max() / np.abs(own).max()
            check(f"   controls: -{{Psi,.}} gives rel {eb1:.2f}, Lambda -> -Lambda gives rel {eb2:.2f} (both must be > 1e-3)",
                  eb1 > 1e-3 and eb2 > 1e-3)
            # slices from left eigenvectors of A (no symmetrisation)
            lam_, W = np.linalg.eig(A.T)                  # columns: left eigenvectors w_j (A^T w = u w)
            if np.abs(lam_.imag).max() > 1e-10 or np.abs(W.imag).max() > 1e-10:
                check("real spectrum of A", False, f"max imag {np.abs(lam_.imag).max():.1e}")
                continue
            uj = lam_.real; W = W.real
            a = gg @ W                                    # a[..., j] = w_j . g
            lhs = dg_G @ W
            rhs = np.stack([-pb(Phi + s * uj[j] * Psi, a[..., j]) - s * uj[j] * dz(a[..., j]) for j in range(M + 1)], -1)
            rhs[0, 0, 0, :] = 0
            e3 = np.abs(lhs - rhs).max() / np.abs(rhs).max()
            check(f"G3 Lambda={Lam:+.3f} beta={beta} {clo}: slices of GANDALF's dg/dt obey the slice equation", e3 < 1e-11,
                  f"rel {e3:.1e}; u_j in [{uj.min():.2f}, {uj.max():.2f}]")

# ------------------------------------------------------------------ G4: direction of propagation by time stepping
print("G4  direction of z+ propagation (gandalf_step, pure z+ wave, no g)")
grid8 = SpectralGrid3D.create(Nx=8, Ny=8, Nz=8)
zp = np.zeros((8, 8, 5), complex); zm = np.zeros_like(zp)
zp[1, 0, 1] = 64.0                                       # k_z = 1, k_x = 1  (unnormalised rfft coefficient)
g0 = np.zeros((8, 8, 5, 3), complex)
st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(g0), M=2, beta_i=1.0, v_th=1.0,
                nu=0.0, Lambda=float(np.sqrt(5.0)), time=0.0, grid=grid8)
T = 0.5; nst = 16
for _ in range(nst):
    st = gandalf_step(st, dt=T / nst, eta=0.0, v_A=1.0)
ratio = complex(np.asarray(st.z_plus)[1, 0, 1] / zp[1, 0, 1])
# a pattern f(x, z + c t) has Fourier phase e^{+i k_z c t}: c = +1 means motion toward -z
print(f"    z+ coefficient ratio after t = {T}: {ratio:.6f}; exp(+i k_z t) = {np.exp(1j * T):.6f}")
check("z+ = f(z + t): propagates toward -z (antiparallel to B0)", abs(ratio - np.exp(1j * T)) < 1e-10)

print(f"wall time {time.time() - t0:.1f} s")
if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL c01 CHECKS PASSED")
