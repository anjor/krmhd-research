#!/usr/bin/env python
"""SPEC.md §1, §3 and §6 against the installed GANDALF.

Study 04, Phase 0, derivation D03.  D01 checks the invariants of the Hermite
ladder; this script checks the statements of SPEC.md about how the pinned
GANDALF applies that ladder, its dissipation and its diagnostics, so that Gate 1
accepts those statements on a test rather than on a reading of the source.
Everything runs in float64 (`jax_enable_x64`).

Part A -- `gandalf_step` keyword defaults (SPEC.md §3)
    The §3 bullet that starts "- `gandalf_step` keyword defaults" lists every
    keyword default as `name=value`.  The set of names must equal the set of
    parameters of `krmhd.timestepping.gandalf_step` that have a default, and
    each value must equal the signature's default, with the same type.

Part B -- the IMEX implicit operator (SPEC.md §1 and §3)
    `krmhd.hermite.build_implicit_operator(kz, beta_i, nu, M, Lambda, hyper_n,
    closure)` must equal, for every k_z,

        L(k_z) = -i s k_z A  -  diag(nu (m/M)^n for m >= 2, 0 for m = 0, 1),

    s = sqrt(beta_i), with A the matrix of SPEC.md §1:
        A_01 = 1/sqrt2, A_10 = c_Lambda = (1 - 1/Lambda)/sqrt2,
        A_{m,m+1} = sqrt((m+1)/2) (m >= 1), A_{m,m-1} = sqrt(m/2) (m >= 2),
    and, for closure="symmetric" (g_{M+1} = g_{M-1}), A_{M,M-1} increased by
    sqrt((M+1)/2).  Checked for M in {6, 13}, Lambda = +-sqrt5, both closures,
    beta_i in {1, 2}, n = 6, nu = 2.5, to relative tolerance TOL.

Part C -- the resistive factor (SPEC.md §3)
    One `gandalf_step(scheme="imex_rk222")` on a state whose only modes have
    k_z = 0 isolates the post-step factor: streaming vanishes at k_z = 0, nu = 0
    makes the implicit operator zero, and the nonlinear terms vanish.
    (C1) z+ = z- = 0 and random g: Phi = Psi = 0, so g only meets the factor.
    (C2) g = 0, z- = 0 and z+ a single Fourier mode: every Poisson bracket of a
    single mode with itself or with z- = 0 vanishes, and the k_z = 0 integrating
    factor is 1, so z+ only meets the factor.
    In both, every mode must be multiplied by exp(-eta (k_perp^2/k_perp,max^2)^r dt),
    k_perp,max^2 = kx[(Nx-1)//3]^2 + ky[(Ny-1)//3]^2, to relative tolerance TOL.

Each part has a negative control that must fail by a clear margin (B2, C3, D3),
so that a check cannot pass because it compares something with itself.

Part D -- the diagnostics of SPEC.md §6
    `hermite_flux(state)` must equal -k_z sqrt(2(m+1)) Im[g_{m+1} g_m^*] (so its
    m = 0 prefactor is sqrt2, not c_0 = sqrt2 (1 - 1/Lambda), SPEC.md §2), and
    `hermite_moment_energy(state)` must equal sum_k w |g_m|^2 with the rfft
    weights w = 1 on the kx = 0 and Nyquist planes and 2 elsewhere, both to
    relative tolerance TOL on a random state with k = 0 zeroed.

Exit code 0 iff every check passes; the last line is then ALL CHECKS PASSED.
Usage: uv run python studies/04-phase-space-helicity/derivations/03_spec_solver.py
"""

from __future__ import annotations

import ast
import inspect
import re
import sys
from pathlib import Path

import numpy as np

SPEC_REL = "studies/04-phase-space-helicity/SPEC.md"


def find_spec(start: Path) -> Path:
    """The study's tracked SPEC.md, at a fixed path below the root of the git repository
    that holds this file. The Gate 1 check runs a copy of this script under the ignored
    data/scratch/, so a search upwards could find a stray untracked copy instead."""
    import subprocess

    top = subprocess.run(["git", "-C", str(start.resolve().parent), "rev-parse", "--show-toplevel"],
                         check=True, capture_output=True, text=True).stdout.strip()
    return Path(top) / SPEC_REL


SPEC = find_spec(Path(__file__))
TOL = 1e-12  # relative; float64 throughout, so only round-off may differ
N_GRID = 16
SEED = 3
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record a named check."""
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILURES.append(name)


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    """max |a - b| / max |b| (b nonzero by construction in every use)."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b))))


def spec_section(text: str, number: int) -> str:
    """Body of the SPEC.md section `## <number>.` up to the next `## ` heading."""
    m = re.search(rf"(?ms)^## {number}\. .*?(?=^## )", text)
    if not m:
        raise ValueError(f"SPEC.md has no section {number}")
    return m.group(0)


def spec_defaults(text: str) -> dict[str, object]:
    """The `name=value` pairs of the SPEC.md §3 bullet on `gandalf_step` keyword defaults."""
    lines = [ln for ln in spec_section(text, 3).splitlines()
             if ln.startswith("- `gandalf_step` keyword defaults")]
    if len(lines) != 1:
        raise ValueError(f"SPEC.md §3 has {len(lines)} gandalf_step defaults bullets, not one")
    pairs = re.findall(r"`(\w+)=([^`]+)`", lines[0])
    out: dict[str, object] = {}
    for name, value in pairs:
        if name in out:
            raise ValueError(f"SPEC.md §3 lists {name} twice")
        out[name] = ast.literal_eval(value)
    return out


def ladder_A(M: int, Lambda: float, closure: str) -> np.ndarray:
    """The matrix A of SPEC.md §1 (with the closure applied at m = M)."""
    A = np.zeros((M + 1, M + 1))
    A[0, 1] = 1.0 / np.sqrt(2.0)
    A[1, 0] = (1.0 - 1.0 / Lambda) / np.sqrt(2.0)
    for m in range(1, M):
        A[m, m + 1] = np.sqrt((m + 1) / 2.0)
    for m in range(2, M + 1):
        A[m, m - 1] = np.sqrt(m / 2.0)
    if closure == "symmetric":
        A[M, M - 1] += np.sqrt((M + 1) / 2.0)
    return A


def part_a() -> None:
    print("Part A: gandalf_step keyword defaults (SPEC.md §3)")
    import hashlib

    from krmhd.timestepping import gandalf_step

    print(f"  SPEC.md read from {SPEC}, sha256 {hashlib.sha256(SPEC.read_bytes()).hexdigest()}")
    sig = {name: p.default for name, p in inspect.signature(gandalf_step).parameters.items()
           if p.default is not inspect.Parameter.empty}
    spec = spec_defaults(SPEC.read_text(encoding="utf-8"))
    check("A1 SPEC lists exactly the defaulted parameters", set(spec) == set(sig),
          f"SPEC {sorted(spec)}; signature {sorted(sig)}")
    for name in sorted(sig):
        if name in spec:
            same = spec[name] == sig[name] and type(spec[name]) is type(sig[name])
            check(f"A2 default {name}", same, f"SPEC {spec[name]!r}; signature {sig[name]!r}")


def part_b() -> None:
    print("Part B: IMEX implicit operator (SPEC.md §1 and §3)")
    from krmhd.hermite import build_implicit_operator

    kz = 2.0 * np.pi * np.fft.fftfreq(N_GRID, d=1.0 / N_GRID)
    nu, n = 2.5, 6
    for M in (6, 13):
        m = np.arange(M + 1)
        damp = np.where(m >= 2, nu * (m / M) ** n, 0.0)
        for Lambda in (np.sqrt(5.0), -np.sqrt(5.0)):
            for closure in ("zero", "symmetric"):
                for beta_i in (1.0, 2.0):
                    L = np.asarray(build_implicit_operator(kz, beta_i, nu, M, float(Lambda), n, closure))
                    A = ladder_A(M, float(Lambda), closure)
                    ref = (-1j * np.sqrt(beta_i) * kz[:, None, None] * A[None, :, :]
                           - np.diag(damp)[None, :, :])
                    err = rel_err(L, ref)
                    tag = f"M={M}, Lambda={Lambda:+.4f}, {closure}, beta_i={beta_i:g}"
                    check(f"B1 L = -i s kz A - diag(nu (m/M)^n, m>=2) ({tag})", err < TOL,
                          f"rel err {err:.1e}")
        # negative control: the other closure's A does not match
        L = np.asarray(build_implicit_operator(kz, 1.0, nu, M, float(np.sqrt(5.0)), n, "zero"))
        wrong = (-1j * kz[:, None, None] * ladder_A(M, float(np.sqrt(5.0)), "symmetric")[None]
                 - np.diag(damp)[None])
        err = rel_err(L, wrong)
        check(f"B2 control: symmetric-closure A does not match the zero-closure L (M={M})",
              err > 1e-3, f"rel err {err:.1e}")


def _state(grid, M: int, z_plus: np.ndarray, z_minus: np.ndarray, g: np.ndarray, nu: float,
           Lambda: float):
    import jax.numpy as jnp
    from krmhd.physics import KRMHDState

    return KRMHDState(z_plus=jnp.asarray(z_plus), z_minus=jnp.asarray(z_minus), g=jnp.asarray(g),
                      M=M, beta_i=1.0, v_th=1.0, nu=nu, Lambda=Lambda, time=0.0, grid=grid)


def _random_g(rng: np.random.Generator, shape: tuple[int, ...], mask: np.ndarray) -> np.ndarray:
    """Random complex g, dealiased, k = 0 zeroed, reality imposed on the kx = 0 and Nyquist planes."""
    g = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    g *= mask[..., None]
    g[0, 0, 0, :] = 0.0
    for plane in (0, g.shape[2] - 1):
        sub = g[:, :, plane, :]
        flipped = np.conj(np.roll(np.roll(sub[::-1, ::-1, :], 1, axis=0), 1, axis=1))
        g[:, :, plane, :] = 0.5 * (sub + flipped)
    return g


def part_c() -> None:
    print("Part C: resistive factor after one gandalf_step (SPEC.md §3)")
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import gandalf_step

    grid = SpectralGrid3D.create(Nx=N_GRID, Ny=N_GRID, Nz=N_GRID, Lx=1.0, Ly=1.0, Lz=1.0)
    mask = np.asarray(grid.dealias_mask)
    kx, ky = np.asarray(grid.kx), np.asarray(grid.ky)
    kperp2 = kx[None, None, :] ** 2 + ky[None, :, None] ** 2
    kperp2_max = kx[(N_GRID - 1) // 3] ** 2 + ky[(N_GRID - 1) // 3] ** 2
    M, Lambda, eta, dt = 6, float(np.sqrt(5.0)), 1.5, 0.01
    rng = np.random.default_rng(SEED)
    zero_z = np.zeros((N_GRID, N_GRID, N_GRID // 2 + 1), complex)
    for r in (1, 2):
        factor = np.exp(-eta * (kperp2 / kperp2_max) ** r * dt)  # [Nz, Ny, Nx//2+1]
        g = _random_g(rng, (N_GRID, N_GRID, N_GRID // 2 + 1, M + 1), mask)
        g[1:, :, :, :] = 0.0  # keep the k_z = 0 plane only
        new = gandalf_step(_state(grid, M, zero_z, zero_z, g, 0.0, Lambda), dt, eta, 1.0,
                           nu=0.0, hyper_r=r, hyper_n=6, scheme="imex_rk222", eta_z=0.0)
        err = rel_err(np.asarray(new.g), g * factor[..., None])
        check(f"C1 g damped by exp(-eta (kperp^2/kperp_max^2)^r dt), r={r}", err < TOL,
              f"rel err {err:.1e}")
        zp = zero_z.copy()
        zp[0, 2, 1] = 0.7 - 0.4j  # k_z = 0, ky index 2, kx index 1: a single mode
        new = gandalf_step(_state(grid, M, zp, zero_z, np.zeros_like(g), 0.0, Lambda), dt, eta, 1.0,
                           nu=0.0, hyper_r=r, hyper_n=6, scheme="imex_rk222", eta_z=0.0)
        err = rel_err(np.asarray(new.z_plus), zp * factor)
        check(f"C2 z+ damped by the same factor, r={r}", err < TOL, f"rel err {err:.1e}")
        # negative control: k_perp,max^2 = kx_max^2 alone (the edge, not the corner)
        edge = np.exp(-eta * (kperp2 / kx[(N_GRID - 1) // 3] ** 2) ** r * dt)
        err = rel_err(np.asarray(new.z_plus), zp * edge)
        check(f"C3 control: the edge normalisation kx_max^2 does not match, r={r}", err > 1e-6,
              f"rel err {err:.1e}")


def part_d() -> None:
    print("Part D: hermite_flux and hermite_moment_energy (SPEC.md §6)")
    from krmhd.diagnostics import hermite_flux, hermite_moment_energy
    from krmhd.spectral import SpectralGrid3D

    grid = SpectralGrid3D.create(Nx=N_GRID, Ny=N_GRID, Nz=N_GRID, Lx=1.0, Ly=1.0, Lz=1.0)
    mask = np.asarray(grid.dealias_mask)
    M, Lambda = 9, float(np.sqrt(5.0))
    rng = np.random.default_rng(SEED + 1)
    g = _random_g(rng, (N_GRID, N_GRID, N_GRID // 2 + 1, M + 1), mask)
    zero_z = np.zeros(g.shape[:3], complex)
    state = _state(grid, M, zero_z, zero_z, g, 0.0, Lambda)
    kz = np.asarray(grid.kz)[:, None, None, None]
    ref_flux = -kz * np.sqrt(2.0 * (np.arange(M) + 1.0)) * np.imag(g[..., 1:] * np.conj(g[..., :-1]))
    err = rel_err(np.asarray(hermite_flux(state)), ref_flux)
    check("D1 hermite_flux = -kz sqrt(2(m+1)) Im[g_{m+1} g_m*], prefactor sqrt2 at m = 0",
          err < TOL, f"rel err {err:.1e}")
    wrong = ref_flux.copy()
    wrong[..., 0] *= 1.0 - 1.0 / Lambda  # the Gamma weight c_0 / sqrt2 at m = 0
    err = rel_err(np.asarray(hermite_flux(state)), wrong)
    check("D3 control: the m = 0 prefactor sqrt2 (1 - 1/Lambda) does not match", err > 1e-3,
          f"rel err {err:.1e}")
    w = np.full(N_GRID // 2 + 1, 2.0)
    w[0] = 1.0
    w[-1] = 1.0  # N_GRID is even: Nyquist plane once
    ref_W = np.sum(np.abs(g) ** 2 * w[None, None, :, None], axis=(0, 1, 2))
    err = rel_err(np.asarray(hermite_moment_energy(state)), ref_W)
    check("D2 hermite_moment_energy = sum_k w |g_m|^2 (rfft weights 1, 2, ..., 2, 1)",
          err < TOL, f"rel err {err:.1e}")


def main() -> int:
    import jax
    jax.config.update("jax_enable_x64", True)
    try:
        part_a()
        part_b()
        part_c()
        part_d()
    except Exception as exc:  # a crash is a failed check, not a pass
        check(f"script raised {type(exc).__name__}", False, str(exc))
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): " + "; ".join(FAILURES))
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
