"""Shared helpers for the independent review checks of the truncated-hierarchy results in this folder (c01_*.py ...
c12_*.py), which support Sections 2, 5 and 8 and Appendices C and D of the paper. Written independently of
../../truncated/tcore.py: the slice basis here is taken from GANDALF's own
krmhd.hermite.compute_streaming_eigensystem (P, P_inv) wherever a check is about GANDALF, and from mpmath/SymPy where
it is about the mathematics.

Conventions: u = v_par/v_th, F_M = exp(-u^2)/sqrt(pi), s = sqrt(beta_i), v_A = 1, Phi = (z+ + z-)/2, Psi = (z+ - z-)/2.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

SQRT5 = float(np.sqrt(5.0))


class Checks:
    def __init__(self, name: str) -> None:
        self.name, self.fail, self.n, self.t0 = name, [], 0, time.time()

    def __call__(self, label: str, ok: bool, detail: str = "") -> bool:
        self.n += 1
        ok = bool(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""), flush=True)
        if not ok:
            self.fail.append(label)
        return ok

    def done(self) -> None:
        print(f"== {self.name}: {self.n - len(self.fail)}/{self.n} passed ({time.time() - self.t0:.1f} s)", flush=True)
        if self.fail:
            print("FAILED:", *self.fail, sep="\n  ")
            sys.exit(1)


def hdr(t: str) -> None:
    print("-" * 100 + "\n" + t, flush=True)


def gandalf_eig(M: int, Lam: float, closure: str = "zero"):
    """GANDALF's own eigensystem of the streaming matrix: T, eigenvalues (sorted), P (right), P_inv (rows = left)."""
    from krmhd.hermite import compute_streaming_eigensystem
    T, ev, P, Pinv = compute_streaming_eigensystem(int(M), float(Lam), closure)
    T, ev, P, Pinv = (np.asarray(x) for x in (T, ev, P, Pinv))
    ev = np.real(ev)
    o = np.argsort(ev)
    return np.real(T), ev[o], np.real(P[:, o]), np.real(Pinv[o, :])


def grid3(N: int, Nz: int | None = None):
    from krmhd.spectral import SpectralGrid3D
    return SpectralGrid3D.create(Nx=N, Ny=N, Nz=N if Nz is None else Nz, Lx=1.0, Ly=1.0, Lz=1.0)


def real_project(f: np.ndarray) -> np.ndarray:
    """Make an rfft array Hermitian on the kx = 0 and Nyquist planes (so it represents a real field)."""
    f = f.copy()
    for pl in (0, f.shape[2] - 1):
        sub = f[:, :, pl, ...]
        flip = np.conj(sub[(-np.arange(sub.shape[0])) % sub.shape[0]][:, (-np.arange(sub.shape[1])) % sub.shape[1]])
        f[:, :, pl, ...] = 0.5 * (sub + flip)
    return f


def rand_field(grid, rng, amp: float = 1.0, slope: float = -2.0, extra=(), dealias: bool = True) -> np.ndarray:
    shp = (grid.Nz, grid.Ny, grid.Nx // 2 + 1) + tuple(extra)
    kx = np.asarray(grid.kx)[None, None, :]
    ky = np.asarray(grid.ky)[None, :, None]
    kz = np.asarray(grid.kz)[:, None, None]
    kk = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    kk[0, 0, 0] = 1.0
    env = amp * (kk / (2 * np.pi)) ** slope
    if dealias:
        env = env * np.asarray(grid.dealias_mask)
    env = env.reshape(env.shape + (1,) * len(extra))
    f = (rng.standard_normal(shp) + 1j * rng.standard_normal(shp)) * env
    f[0, 0, 0, ...] = 0.0
    return real_project(f)


def state(grid, M, Lam, zp, zm, g, beta=1.0, nu=0.0):
    from krmhd.physics import KRMHDState
    return KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(g), M=int(M), beta_i=float(beta),
                      v_th=1.0, nu=float(nu), Lambda=float(Lam), time=0.0, grid=grid)


def pb(grid, f, h):
    from krmhd.physics import poisson_bracket_3d
    return np.asarray(poisson_bracket_3d(jnp.asarray(f), jnp.asarray(h), grid.kx, grid.ky, grid.Nz, grid.Ny, grid.Nx,
                                         grid.dealias_mask))


def ddz(grid, f):
    kz = np.asarray(grid.kz).reshape((-1,) + (1,) * (np.ndim(f) - 1))
    return 1j * kz * f


def to_real(grid, f, pad: int = 1):
    """Inverse rfft (GANDALF convention: forward unnormalised, inverse = irfftn) on a grid padded by the integer
    factor `pad`, so that grid sums of products of up to 2*pad band-limited (2/3-dealiased) fields are exact."""
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    if pad == 1:
        return np.fft.irfftn(f, s=(Nz, Ny, Nx), axes=(0, 1, 2))
    Pz, Py, Px = pad * Nz, pad * Ny, pad * Nx
    F = np.zeros((Pz, Py, Px // 2 + 1), complex)
    iz = np.r_[0:Nz // 2, Pz - Nz // 2:Pz]
    iy = np.r_[0:Ny // 2, Py - Ny // 2:Py]
    F[np.ix_(iz, iy, np.arange(Nx // 2 + 1))] = f
    return np.fft.irfftn(F, s=(Pz, Py, Px), axes=(0, 1, 2)) * pad ** 3


def kweights(grid) -> np.ndarray:
    w = np.full(grid.Nx // 2 + 1, 2.0)
    w[0] = 1.0
    if grid.Nx % 2 == 0:
        w[-1] = 1.0
    return np.broadcast_to(w[None, None, :], (grid.Nz, grid.Ny, grid.Nx // 2 + 1))


def vint(grid, f, h) -> float:
    """sum over all k of conj(f_k) h_k (Parseval form of int f h for real fields), from rfft arrays."""
    return float(np.sum(kweights(grid) * np.real(np.conj(f) * h)))
