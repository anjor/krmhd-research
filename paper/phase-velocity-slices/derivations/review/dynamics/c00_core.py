"""Shared helpers for the independent checks in this folder, written separately from derivations/dynamics/dcore.py
so that results of Sections 5, 6 and 7 are re-derived by different routes.  Numerical output goes to
paper/phase-velocity-slices/data/derivations/review/dynamics/, which is git-ignored.

Differences from dcore on purpose: box of side L = 2 pi (integer wavenumbers), fields built from an explicit list of
Fourier modes with random amplitudes (the same physical field at every N, no embedding), point evaluation of the
Fourier series at arbitrary positions (for Lagrangian tracking), and separate RK4 wrappers.  The physics (right-hand
side, stepper, ladder matrix) is GANDALF's (installed krmhd 0.7.0, read-only).

Conventions checked in the krmhd source: z+ = Phi + Psi, z- = Phi - Psi; {f,h} = f_x h_y - f_y h_x;
u_perp = zhat x grad Phi = (-Phi_y, Phi_x, 0); b_perp = zhat x grad Psi; grad_par = d_z + {Psi, .};
d_t g + {Phi, g} + sqrt(beta) grad_par (A g) = 0 with A = hermite.compute_streaming_matrix(M, Lambda).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from krmhd.hermite import compute_streaming_matrix  # noqa: E402
from krmhd.spectral import SpectralGrid3D  # noqa: E402
from krmhd.timestepping import KRMHDFields, _krmhd_rhs_jit  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))          # paper/phase-velocity-slices
DATA = os.path.join(PAPER, "data", "derivations", "review", "dynamics")  # numerical output (git-ignored)
RUNLOG = os.path.join(DATA, "gandalf_runs.jsonl")
SQRT5 = float(np.sqrt(5.0))
TWOPI = 2.0 * np.pi


class Grid:
    """GANDALF grid with L = 2 pi in every direction (integer wavenumbers)."""

    def __init__(self, N: int):
        self.N = N
        self.g = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=TWOPI, Ly=TWOPI, Lz=TWOPI)
        self.kx = np.asarray(self.g.kx, dtype=float)
        self.ky = np.asarray(self.g.ky, dtype=float)
        self.kz = np.asarray(self.g.kz, dtype=float)
        self.mask = np.asarray(self.g.dealias_mask)
        self.KX = self.kx[None, None, :] + 0 * self.ky[None, :, None] + 0 * self.kz[:, None, None]
        self.KY = self.ky[None, :, None] + 0 * self.KX
        self.KZ = self.kz[:, None, None] + 0 * self.KX
        self.KP2 = self.KX ** 2 + self.KY ** 2
        w = np.full(N // 2 + 1, 2.0)
        w[0] = 1.0
        if N % 2 == 0:
            w[-1] = 1.0
        self.W = (w[None, None, :] + 0 * self.KX)
        self.jk = (jnp.asarray(self.kx), jnp.asarray(self.ky), jnp.asarray(self.kz), jnp.asarray(self.mask))
        x = np.arange(N) * TWOPI / N
        # GANDALF arrays are [z, y, x]
        self.Z, self.Y, self.X = np.meshgrid(x, x, x, indexing="ij")
        # flattened list of stored modes (for point evaluation)
        sel = self.mask & (self.W > 0)
        self.sel = sel
        self.kvec = np.stack([self.KX[sel], self.KY[sel], self.KZ[sel]], axis=1)   # (nm, 3) as (kx, ky, kz)
        self.wsel = self.W[sel]

    # volume average <f h> of two real fields from rfftn coefficients
    def avg(self, f, h) -> float:
        return float(np.sum(self.W * np.real(np.conj(f) * h))) / self.N ** 6

    def from_real(self, f):
        return np.fft.rfftn(f, axes=(0, 1, 2)) * self.mask

    def to_real(self, F):
        return np.fft.irfftn(F, s=(self.N,) * 3, axes=(0, 1, 2))

    def rms_grad(self, F) -> float:
        return float(np.sqrt(np.sum(self.W * self.KP2 * np.abs(F) ** 2))) / self.N ** 3

    # Fourier series evaluated at arbitrary points P (n, 3) given as (x, y, z)
    def phase(self, P):
        return np.exp(1j * (P @ self.kvec.T))                               # (n, nm)

    def eval(self, E, F, d=(0, 0, 0)):
        """F: rfftn coefficients (Nz, Ny, Nx//2+1[, extra]); d: derivative orders in (x, y, z)."""
        fac = (1j * self.kvec[:, 0]) ** d[0] * (1j * self.kvec[:, 1]) ** d[1] * (1j * self.kvec[:, 2]) ** d[2]
        c = F[self.sel]
        if c.ndim == 1:
            return np.real(E @ (self.wsel * fac * c)) / self.N ** 3
        return np.real(E @ ((self.wsel * fac)[:, None] * c)) / self.N ** 3


def mode_field(grid: Grid, modes: np.ndarray, amps: np.ndarray, phases: np.ndarray):
    """Real field sum_n a_n cos(n . x + theta_n) (n integer vectors (nx, ny, nz)); returned as rfftn coefficients."""
    f = np.zeros_like(grid.X)
    for n, a, th in zip(modes, amps, phases):
        f += a * np.cos(n[0] * grid.X + n[1] * grid.Y + n[2] * grid.Z + th)
    return grid.from_real(f)


def random_modes(rng, nperp=(1.0, 2.5), nzmax=2, n=None):
    """All integer (nx, ny, nz) with |n_perp| in [a, b], |nz| <= nzmax, one of each +-pair; random amps and phases."""
    out = []
    for nx in range(-3, 4):
        for ny in range(-3, 4):
            for nz in range(-nzmax, nzmax + 1):
                kp = np.hypot(nx, ny)
                if nperp[0] - 1e-9 <= kp <= nperp[1] + 1e-9:
                    v = (nx, ny, nz)
                    if (-nx, -ny, -nz) not in out:
                        out.append(v)
    modes = np.array(out, dtype=float)
    amps = rng.standard_normal(len(modes))
    phases = rng.uniform(0, TWOPI, len(modes))
    return modes, amps, phases


def ladder(M: int, Lam: float, closure: str = "zero") -> np.ndarray:
    return np.array(compute_streaming_matrix(M, float(Lam), closure), dtype=float)


def pw(M: int, Lam: float) -> np.ndarray:
    p = np.ones(M + 1)
    p[0] = 1.0 - 1.0 / Lam
    return p


def eig_slices(M: int, Lam: float):
    """Right eigenvectors V (columns) of A with V^T P0 V = I, eigenvalues u, left W = V^{-1} (general solver)."""
    A = ladder(M, Lam)
    u, V = np.linalg.eig(A)
    assert np.abs(np.imag(u)).max() < 1e-12, "complex spectrum"
    u, V = np.real(u), np.real(V)
    o = np.argsort(u)
    u, V = u[o], V[:, o]
    p = pw(M, Lam)
    nrm = np.sqrt(np.einsum("mj,m,mj->j", V, p, V))
    V = V / nrm
    WL = np.linalg.inv(V)
    return u, V, WL


def make_rhs(grid: Grid, M: int, Lam: float, beta: float, closure: str = "zero"):
    N = grid.N
    kx, ky, kz, mask = grid.jk

    @jax.jit
    def rhs(zp, zm, g):
        d = _krmhd_rhs_jit(KRMHDFields(zp, zm, g, 0.0), kx, ky, kz, mask, 0.0, 1.0, beta, 0.0, Lam, M, N, N, N,
                           closure=closure)
        return d.z_plus, d.z_minus, d.g

    return rhs


def log_run(script: str, purpose: str, N: int, M: int, steps: int, wall: float, **extra):
    """Append one GANDALF time integration to the run log RUNLOG (one JSON line, with the local end time)."""
    rec = dict(script=script, purpose=purpose, grid=f"{N}^3", M=M, steps=steps, wall_s=round(wall, 1),
               ended_local=time.strftime("%Y%m%d_%H%M%S"))
    rec.update(extra)
    os.makedirs(os.path.dirname(RUNLOG), exist_ok=True)
    with open(RUNLOG, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


class Checks:
    def __init__(self):
        self.items = []

    def __call__(self, name, ok, info=""):
        ok = bool(ok)
        self.items.append((name, ok))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({info})" if info else ""), flush=True)
        return ok

    def finish(self):
        nf = sum(1 for _, ok in self.items if not ok)
        print(f"SUMMARY: {len(self.items) - nf}/{len(self.items)} checks passed", flush=True)
        raise SystemExit(1 if nf else 0)


def clock():
    t0 = time.time()
    return lambda: time.time() - t0
