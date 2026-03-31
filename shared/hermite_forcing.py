"""Study-local Hermite forcing helpers.

The upstream Hermite forcing uses a full-|k| shell. For Study 02 we sometimes
want the Hermite drive to respect the same low-|n_z| mask as the Alfvén drive
so the two sectors are forced on comparable large-scale supports.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from krmhd.forcing import force_hermite_moments
from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D


@dataclass(frozen=True)
class HermiteForcingOptions:
    """Study-specific Hermite forcing settings."""

    mode: str = "shell"
    max_nz: int = 1
    include_nz0: bool = False


def pop_hermite_forcing_options(cfg_dict: dict[str, Any]) -> HermiteForcingOptions:
    """Extract study-local Hermite forcing keys before use."""
    mode = str(cfg_dict.pop("mode", "shell"))
    max_nz = int(cfg_dict.pop("max_nz", 1))
    include_nz0 = bool(cfg_dict.pop("include_nz0", False))
    return HermiteForcingOptions(
        mode=mode,
        max_nz=max_nz,
        include_nz0=include_nz0,
    )


def apply_hermite_forcing(
    state: KRMHDState,
    amplitude: float,
    n_min: int,
    n_max: int,
    dt: float,
    key: Array,
    forced_moments: tuple[int, ...],
    options: HermiteForcingOptions,
) -> tuple[KRMHDState, Array]:
    """Apply the configured Hermite forcing and return the updated state/key."""
    if amplitude <= 0.0:
        return state, key

    if options.mode == "shell":
        return force_hermite_moments(
            state,
            amplitude=amplitude,
            n_min=n_min,
            n_max=n_max,
            dt=dt,
            key=key,
            forced_moments=forced_moments,
        )

    if options.mode == "perp_lowkz":
        forcing, key = gaussian_white_noise_fourier_perp_lowkz_local(
            state.grid,
            amplitude=amplitude,
            n_min=n_min,
            n_max=n_max,
            max_nz=options.max_nz,
            include_nz0=options.include_nz0,
            dt=dt,
            key=key,
        )
        g_new = jnp.array(state.g)
        for m in forced_moments:
            g_new = g_new.at[:, :, :, m].add(forcing)
        new_state = KRMHDState(
            z_plus=state.z_plus,
            z_minus=state.z_minus,
            B_parallel=state.B_parallel,
            g=g_new,
            M=state.M,
            beta_i=state.beta_i,
            v_th=state.v_th,
            nu=state.nu,
            Lambda=state.Lambda,
            time=state.time,
            grid=state.grid,
        )
        return new_state, key

    raise ValueError(
        f"Unknown Hermite forcing mode {options.mode!r}. "
        "Expected 'shell' or 'perp_lowkz'."
    )


def gaussian_white_noise_fourier_perp_lowkz_local(
    grid: SpectralGrid3D,
    amplitude: float,
    n_min: int,
    n_max: int,
    max_nz: int,
    include_nz0: bool,
    dt: float,
    key: Array,
) -> tuple[Array, Array]:
    """Local low-|n_z| white-noise forcing with the correct JIT signature."""
    if amplitude <= 0:
        raise ValueError(f"amplitude must be positive, got {amplitude}")
    if n_min <= 0 or n_max < n_min:
        raise ValueError(f"Invalid band: n_min={n_min}, n_max={n_max}")
    if max_nz < 0:
        raise ValueError(f"max_nz must be >= 0, got {max_nz}")

    kperp_min = 2.0 * jnp.pi * n_min / grid.Lx
    kperp_max = 2.0 * jnp.pi * n_max / grid.Lx

    k1 = 2.0 * jnp.pi / grid.Lz
    nz_float = jnp.round(jnp.abs(grid.kz) / k1)
    nz_int = nz_float.astype(jnp.int32)
    kz_allowed = nz_int <= max_nz
    if not include_nz0:
        kz_allowed = kz_allowed & (nz_int != 0)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
    real_part = jax.random.normal(subkey1, shape=shape, dtype=jnp.float32)
    imag_part = jax.random.normal(subkey2, shape=shape, dtype=jnp.float32)

    forcing = _gaussian_white_noise_fourier_perp_lowkz_jit_local(
        grid.kx,
        grid.ky,
        grid.kz,
        amplitude,
        float(kperp_min),
        float(kperp_max),
        kz_allowed,
        float(dt),
        real_part,
        imag_part,
        grid.Nx,
    )
    return forcing, key


@partial(jax.jit, static_argnums=(10,))
def _gaussian_white_noise_fourier_perp_lowkz_jit_local(
    kx: Array,
    ky: Array,
    kz: Array,
    amplitude: float,
    kperp_min: float,
    kperp_max: float,
    kz_allowed: Array,
    dt: float,
    real_part: Array,
    imag_part: Array,
    nx_full: int,
) -> Array:
    """JIT core for local low-|n_z| Gaussian forcing."""
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
    k_perp = jnp.sqrt(kx_3d**2 + ky_3d**2)

    perp_mask = (k_perp >= kperp_min) & (k_perp <= kperp_max)
    kz_mask = kz_allowed[:, jnp.newaxis, jnp.newaxis]
    mask = perp_mask & kz_mask

    scale = amplitude / jnp.sqrt(dt)
    noise = (real_part + 1j * imag_part) * scale
    forced_field = noise * mask.astype(noise.dtype)

    forced_field = forced_field.at[:, :, 0].set(
        forced_field[:, :, 0].real.astype(forced_field.dtype)
    )
    if nx_full % 2 == 0:
        nyquist_idx = forced_field.shape[2] - 1
        forced_field = forced_field.at[:, :, nyquist_idx].set(
            forced_field[:, :, nyquist_idx].real.astype(forced_field.dtype)
        )

    forced_field = forced_field.at[0, 0, 0].set(0.0 + 0.0j)
    return forced_field
