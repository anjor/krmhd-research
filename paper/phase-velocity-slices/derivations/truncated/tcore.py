"""Shared helpers for the truncated-hierarchy check scripts in this folder (t1_*.py ... t7b_*.py), which support
Sections 5 and 8 and Appendices C and D of the paper.

Every GANDALF-specific object is taken from the installed krmhd package (krmhd 0.6.0):
the streaming ("ladder") matrix A comes from krmhd.hermite.compute_streaming_matrix, the right-hand
side from krmhd.timestepping.krmhd_rhs, brackets and derivatives from krmhd.physics / krmhd.spectral.
What is re-derived here (and checked in t1) is the weight that symmetrises A and the slice basis.

Conventions: u = v_par / v_th, F_M = exp(-u^2)/sqrt(pi), s = sqrt(beta_i) = v_th / v_A, v_A = 1.
GANDALF's g_m are CMM's tilde G_m: G = sum_m g_m H_m(u) F_M / sqrt(2^m m!), so G / sqrt(F_M) = sum_m g_m psi_m(u)
with psi_m the orthonormal Hermite functions.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from krmhd.hermite import compute_streaming_matrix  # noqa: E402

SQRT5 = float(np.sqrt(5.0))
HERE = os.path.dirname(os.path.abspath(__file__))


class Checker:
    """Collects PASS/FAIL checks; finish() exits with status 1 if any check failed."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.failed: list[str] = []
        self.n = 0
        self.t0 = time.time()

    def check(self, label: str, ok: bool, detail: str = "") -> bool:
        self.n += 1
        ok = bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""), flush=True)
        if not ok:
            self.failed.append(label)
        return ok

    def finish(self) -> None:
        print(f"== {self.name}: {self.n - len(self.failed)}/{self.n} checks passed "
              f"({time.time() - self.t0:.1f} s)", flush=True)
        if self.failed:
            print("FAILED CHECKS:", *self.failed, sep="\n  ")
            sys.exit(1)


def section(title: str) -> None:
    print("=" * 100 + "\n" + title, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# Velocity-space linear algebra
# ----------------------------------------------------------------------------------------------------------------
def ladder(M: int, Lam: float, closure: str = "zero") -> np.ndarray:
    """GANDALF's streaming matrix A (krmhd.hermite.compute_streaming_matrix), as a float64 numpy array."""
    return np.array(compute_streaming_matrix(int(M), float(Lam), closure), dtype=float)


def weight(M: int, Lam: float, closure: str = "zero") -> np.ndarray:
    """Diagonal of the weight Pi that makes Pi A symmetric (derived in t1):
    Pi = D_c P_Lam,  P_Lam = diag(1 - 1/Lam, 1, ..., 1),  D_c = I (zero closure) or
    diag(1, ..., 1, sqrt(M)/(sqrt(M)+sqrt(M+1))) (copy closure g_{M+1} = g_{M-1})."""
    p = np.ones(M + 1)
    p[0] = 1.0 - (0.0 if np.isinf(Lam) else 1.0 / Lam)
    if closure == "symmetric":
        p[M] = np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1.0))
    return p


def slices(M: int, Lam: float, closure: str = "zero"):
    """Slice basis of A: eigenvalues u (ascending), right eigenvectors R[:, j], left eigenvectors
    W[:, j] = Pi R[:, j] (so W^T A = diag(u) W^T), and norms n_j = W_j . R_j > 0.
    Uses the symmetric similarity transform Pi^{1/2} A Pi^{-1/2} (valid for Lam outside [0, 1])."""
    A = ladder(M, Lam, closure)
    p = weight(M, Lam, closure)
    if np.any(p <= 0):
        raise ValueError("weight not positive definite (Lambda in [0,1])")
    sp = np.sqrt(p)
    S = (sp[:, None] * A) / sp[None, :]
    S = 0.5 * (S + S.T)
    u, Q = np.linalg.eigh(S)
    R = Q / sp[:, None]          # right eigenvectors, normalised so that R^T Pi R = I
    W = p[:, None] * R           # left eigenvectors
    n = np.einsum("mj,mj->j", W, R)
    return u, R, W, n


def hermite_functions(M: int, x) -> np.ndarray:
    """Orthonormal Hermite functions psi_m(x), m = 0..M, by the stable three-term recurrence. Shape (M+1, len(x))."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    P = np.zeros((M + 1, x.size))
    P[0] = np.pi ** -0.25 * np.exp(-x ** 2 / 2)
    if M >= 1:
        P[1] = np.sqrt(2.0) * x * P[0]
    for m in range(1, M):
        P[m + 1] = np.sqrt(2.0 / (m + 1)) * x * P[m] - np.sqrt(m / (m + 1.0)) * P[m - 1]
    return P


def orthonormal_polys(M: int, x) -> np.ndarray:
    """p_m(x) = H_m(x)/sqrt(2^m m!), orthonormal with weight F_M = exp(-x^2)/sqrt(pi). Shape (M+1, len(x))."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    P = np.zeros((M + 1, x.size))
    P[0] = 1.0
    if M >= 1:
        P[1] = np.sqrt(2.0) * x
    for m in range(1, M):
        P[m + 1] = (x * P[m] - np.sqrt(m / 2.0) * P[m - 1]) / np.sqrt((m + 1) / 2.0)
    return P


# ----------------------------------------------------------------------------------------------------------------
# Spatial helpers (GANDALF grid conventions: arrays [Nz, Ny, Nx//2+1(, M+1)], L = 1 box)
# ----------------------------------------------------------------------------------------------------------------
def make_grid(N: int, Nz: int | None = None):
    from krmhd.spectral import SpectralGrid3D
    return SpectralGrid3D.create(Nx=N, Ny=N, Nz=N if Nz is None else Nz, Lx=1.0, Ly=1.0, Lz=1.0)


def ksum_weights(grid) -> np.ndarray:
    """rfft weights so that sum_k w |f_k|^2 is the full-k sum: 1 on kx = 0 and Nyquist planes, 2 elsewhere."""
    Nx = grid.Nx
    w = np.full(Nx // 2 + 1, 2.0)
    w[0] = 1.0
    w[-1] = 1.0
    return np.broadcast_to(w[None, None, :], (grid.Nz, grid.Ny, Nx // 2 + 1)).copy()


def hermitian_project(f: np.ndarray) -> np.ndarray:
    """Project an rfft array [Nz, Ny, Nkx, ...] onto real fields: f(-k) = conj f(k) on the kx = 0 and kx = Nyquist planes."""
    f = f.copy()
    for plane in (0, f.shape[2] - 1):
        sub = f[:, :, plane, ...]
        flipped = np.conj(np.roll(np.roll(sub[::-1, ::-1, ...], 1, axis=0), 1, axis=1))
        f[:, :, plane, ...] = 0.5 * (sub + flipped)
    return f


def random_field(grid, rng, slope: float = -2.0, extra_shape=(), amp: float = 1.0) -> np.ndarray:
    """Random real (Hermitian) dealiased field with |f_k| ~ (|k|/2pi)^slope, zero mean."""
    shp = (grid.Nz, grid.Ny, grid.Nx // 2 + 1) + tuple(extra_shape)
    kx = np.asarray(grid.kx)[None, None, :]
    ky = np.asarray(grid.ky)[None, :, None]
    kz = np.asarray(grid.kz)[:, None, None]
    kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    kk[0, 0, 0] = 1.0
    env = amp * (kk / (2 * np.pi)) ** slope * np.asarray(grid.dealias_mask)
    env = env.reshape(env.shape + (1,) * len(extra_shape))
    f = (rng.standard_normal(shp) + 1j * rng.standard_normal(shp)) * env
    f[0, 0, 0, ...] = 0.0
    return hermitian_project(f)


def make_state(grid, M: int, Lam: float, zp: np.ndarray, zm: np.ndarray, g: np.ndarray, beta_i: float = 1.0,
               nu: float = 0.0):
    from krmhd.physics import KRMHDState
    return KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(g), M=M, beta_i=float(beta_i),
                      v_th=1.0, nu=float(nu), Lambda=float(Lam), time=0.0, grid=grid)


def bracket(grid, f: np.ndarray, h: np.ndarray) -> np.ndarray:
    """GANDALF's dealiased Poisson bracket {f, h} = d_x f d_y h - d_y f d_x h (krmhd.physics.poisson_bracket_3d)."""
    from krmhd.physics import poisson_bracket_3d
    return np.asarray(poisson_bracket_3d(jnp.asarray(f), jnp.asarray(h), grid.kx, grid.ky, grid.Nz, grid.Ny,
                                         grid.Nx, grid.dealias_mask))


def dz(grid, f: np.ndarray) -> np.ndarray:
    """GANDALF's parallel derivative (krmhd.spectral.derivative_z): i k_z f."""
    from krmhd.spectral import derivative_z
    return np.asarray(derivative_z(jnp.asarray(f), grid.kz))


def inner(grid, f: np.ndarray, h: np.ndarray) -> float:
    """Volume integral int f h d^3r of two real fields given as rfft arrays (up to a fixed normalisation)."""
    w = ksum_weights(grid)
    return float(np.sum(w * np.real(np.conj(f) * h)))


def qform(grid, g: np.ndarray, K: np.ndarray) -> float:
    """sum_k w_k g(k)^H K g(k) for g of shape [Nz, Ny, Nkx, M+1] and a real symmetric K."""
    w = ksum_weights(grid)
    return float(np.real(np.einsum("zyx,zyxm,mn,zyxn->", w, np.conj(g), K, g)))


def dqform(grid, g: np.ndarray, gdot: np.ndarray, K: np.ndarray):
    """d/dt of qform with the symmetric part of K, and a cancellation scale (same sum with absolute values)."""
    w = ksum_weights(grid)
    Ks = 0.5 * (K + K.T)
    val = 2.0 * float(np.real(np.einsum("zyx,zyxm,mn,zyxn->", w, np.conj(g), Ks, gdot)))
    scale = 2.0 * float(np.einsum("zyx,zyxm,mn,zyxn->", w, np.abs(g), np.abs(Ks), np.abs(gdot)))
    return val, scale


def maxgrad(grid, f: np.ndarray) -> float:
    """max over real space of |grad_perp f| (same construction as krmhd.timestepping.compute_cfl_timestep)."""
    from krmhd.spectral import derivative_x, derivative_y, rfftn_inverse
    fx = rfftn_inverse(derivative_x(jnp.asarray(f), grid.kx), grid.Nz, grid.Ny, grid.Nx)
    fy = rfftn_inverse(derivative_y(jnp.asarray(f), grid.ky), grid.Nz, grid.Ny, grid.Nx)
    return float(jnp.sqrt(fx ** 2 + fy ** 2).max())
