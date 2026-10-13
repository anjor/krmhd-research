"""c06 (independent check): the exact zero eigenvalue of the Hermite ladder matrix at even M, by exact arithmetic
(determinant and parity of the characteristic polynomial) and on GANDALF's streaming matrix (M = 63, 64, 127, 128,
exact null vector), and the u = 0 slice it defines, which is a pure E x B passive scalar in GANDALF's right-hand
side. It supports Section 4 (the u = 0 slice) and Section 8 (even and odd M) of the paper.

 Z1  SymPy, symbolic Lambda: det(A_M) = 0 for even M and != 0 for odd M (M = 1..16), both closures, with A built from
     krmhd.hermite.compute_streaming_matrix's structure rebuilt exactly (A[m,m+1] = A[m+1,m] = sqrt((m+1)/2), A[1,0] =
     (1 - 1/Lambda)/sqrt2; symmetric closure adds sqrt((M+1)/2) to A[M,M-1]); the characteristic polynomial has parity (-1)^(M+1).
 Z2  GANDALF's own matrix (compute_streaming_matrix) at M = 64, 128 (even) and 63, 127 (odd), Lambda = +-sqrt5, both closures:
     smallest |eigenvalue|; exact rational null vector of A^T at M = 64 evaluated with GANDALF's matrix.
 Z3  the u = 0 slice is a pure E x B passive scalar in GANDALF: w0 . (dg/dt) = -{Phi, w0 . g} with krmhd_rhs (M = 16),
     brackets computed with numpy (padded grid); the s-dependent streaming and the {Psi,.} term vanish identically for it.
Exit 1 on any failure.
"""
import sys
import numpy as np
import sympy as sp

fails = []


def check(name, ok, info=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)
    if not ok:
        fails.append(name)


print("Z1  exact determinant parity")
Lam = sp.Symbol('Lambda', nonzero=True)
lam = sp.Symbol('lam')


def A_exact(M, closure):
    A = sp.zeros(M + 1, M + 1)
    for m in range(M):
        A[m, m + 1] = sp.sqrt(sp.Rational(m + 1, 2)); A[m + 1, m] = sp.sqrt(sp.Rational(m + 1, 2))
    A[1, 0] = (1 - 1 / Lam) / sp.sqrt(2)
    if closure == "symmetric":
        A[M, M - 1] += sp.sqrt(sp.Rational(M + 1, 2))
    return A


ok_even = ok_odd = ok_par = True
for clo in ("zero", "symmetric"):
    for M in range(2, 17):
        A = A_exact(M, clo)
        d = sp.simplify(A.det())
        if M % 2 == 0:
            ok_even &= (d == 0)
        else:
            ok_odd &= (d != 0) and sp.simplify(d.subs(Lam, sp.sqrt(5))) != 0 and sp.simplify(d.subs(Lam, -sp.sqrt(5))) != 0
        if M <= 10:
            p = sp.expand((lam * sp.eye(M + 1) - A).det())
            ok_par &= sp.simplify(p.subs(lam, -lam) - (-1)**(M + 1) * p) == 0
check("det A = 0 for even M = 2..16 (both closures, symbolic Lambda)", ok_even)
check("det A != 0 for odd M = 3..15 at Lambda = +-sqrt5 and symbolically", ok_odd)
check("characteristic polynomial has parity (-1)^(M+1) (M <= 10)", ok_par)
d1 = sp.factor(A_exact(3, "zero").det())
print(f"    e.g. M = 3: det A = {d1}  (vanishes only at Lambda = 1)")

print("Z2  GANDALF's matrix")
from krmhd.hermite import compute_streaming_matrix
import krmhd
for clo in ("zero", "symmetric"):
    for Lv in (np.sqrt(5.0), -np.sqrt(5.0)):
        row = []
        for M in (63, 64, 127, 128):
            T = np.asarray(compute_streaming_matrix(M, float(Lv), clo), dtype=float)
            ev = np.linalg.eigvals(T)
            row.append((M, np.abs(ev).min(), np.abs(ev.imag).max()))
        txt = ", ".join(f"M={M}: min|u|={a:.1e}" for M, a, _ in row)
        ok = row[1][1] < 1e-10 and row[3][1] < 1e-10 and row[0][1] > 1e-2 and row[2][1] > 1e-2 and max(r[2] for r in row) < 1e-8
        check(f"krmhd {krmhd.__version__} {clo}, Lambda={Lv:+.3f}: zero eigenvalue at even M only; spectrum real", ok, txt)
# exact null vector of A^T at M = 64 (left eigenvector for u = 0) from the recurrence, checked with GANDALF's matrix
M = 64
Tg = np.asarray(compute_streaming_matrix(M, float(np.sqrt(5.0)), "zero"), dtype=float)
A64 = A_exact(M, "zero").subs(Lam, sp.sqrt(5))
ns = (A64.T).nullspace()
w0 = np.array([float(x) for x in ns[0]])
res = np.abs(Tg.T @ w0).max() / np.abs(w0).max()
check("exact rational-radical null vector of A^T (M = 64) annihilated by GANDALF's matrix", len(ns) == 1 and res < 1e-13, f"residual {res:.1e}")
print(f"    w0 has nonzero entries only at even m: {np.all(np.abs(w0[1::2]) < 1e-14)}")

print("Z3  the u = 0 slice in GANDALF's right-hand side")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D
from krmhd.timestepping import krmhd_rhs
N = 16; Mg = 16
grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N)
mask = np.asarray(grid.dealias_mask).astype(float)
kx = np.asarray(grid.kx)[None, None, :]; ky = np.asarray(grid.ky)[None, :, None]
rng = np.random.default_rng(3)


def rf(amp):
    F = np.fft.rfftn(rng.standard_normal((N, N, N)), axes=(0, 1, 2)) * mask
    F[0, 0, 0] = 0
    return F / np.abs(np.fft.irfftn(F, s=(N, N, N), axes=(0, 1, 2))).max() * amp


def pad(F):
    P = 2 * N
    G = np.zeros((P, P, N + 1), complex)
    iz = np.where(np.arange(N) < N // 2, np.arange(N), np.arange(N) + N)
    G[np.ix_(iz, iz, np.arange(N // 2 + 1))] = F * mask
    return np.fft.irfftn(G, s=(P, P, P), axes=(0, 1, 2)) * 8


def unpad(f):
    G = np.fft.rfftn(f, axes=(0, 1, 2)) / 8
    iz = np.where(np.arange(N) < N // 2, np.arange(N), np.arange(N) + N)
    return G[np.ix_(iz, iz, np.arange(N // 2 + 1))] * mask


pb = lambda F, G: unpad(pad(1j * kx * F) * pad(1j * ky * G) - pad(1j * ky * F) * pad(1j * kx * G))
for Lv in (np.sqrt(5.0), -np.sqrt(5.0)):
    for beta in (1.0, 2.5):
        zp, zm = rf(10.0), rf(7.0)
        gg = np.stack([rf(1.0) for _ in range(Mg + 1)], -1)
        st = KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(gg), M=Mg, beta_i=beta, v_th=1.0,
                        nu=0.0, Lambda=float(Lv), time=0.0, grid=grid)
        dg = np.asarray(krmhd_rhs(st, eta=0.0, v_A=1.0, closure="zero").g)
        Tg = np.asarray(compute_streaming_matrix(Mg, float(Lv), "zero"), dtype=float)
        w0 = np.array([float(x) for x in (A_exact(Mg, "zero").subs(Lam, sp.nsimplify(Lv) if abs(Lv) != np.sqrt(5) else sp.sign(Lv) * sp.sqrt(5)).T).nullspace()[0]])
        a0 = gg @ w0; da0 = dg @ w0
        Phi = (zp + zm) / 2
        rhs = -pb(Phi, a0); rhs[0, 0, 0] = 0
        e = np.abs(da0 - rhs).max() / np.abs(rhs).max()
        full = np.abs(dg).max()
        check(f"Lambda={Lv:+.3f} beta={beta}: d_t a_0 = -{{Phi, a_0}} exactly (no streaming, no {{Psi,.}})", e < 1e-12,
              f"rel {e:.1e}; |A^T w0| = {np.abs(Tg.T @ w0).max():.1e}")

if fails:
    print("FAILED:", fails)
    sys.exit(1)
print("ALL c06 CHECKS PASSED")
