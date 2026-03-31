"""Study-local Alfvén forcing helpers.

This module provides a config-driven wrapper around the upstream KRMHD forcing
functions so the study can switch between:

- Gaussian white-noise forcing in a total-|k| shell
- the original GANDALF shell forcing in total |k|
- balanced Elsasser forcing in a perpendicular band and low-|nz|
- a GANDALF-amplitude variant restricted to a perpendicular band and low-|nz|

The low-|nz| path preserves the current "force phi only" behavior by applying
the same forcing field to z+ and z-.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array

from krmhd.forcing import (
    force_alfven_modes,
    force_alfven_modes_balanced,
    force_alfven_modes_gandalf,
)
from krmhd.physics import KRMHDState
from krmhd.spectral import SpectralGrid3D


@dataclass(frozen=True)
class AlfvenForcingOptions:
    """Study-specific Alfvén forcing settings."""

    mode: str = "gandalf_shell"
    max_nz: int = 1
    include_nz0: bool = False
    correlation: float = 0.0


def pop_alfven_forcing_options(cfg_dict: dict[str, Any]) -> AlfvenForcingOptions:
    """Extract study-local forcing keys before SimulationConfig validation."""
    forcing_dict = cfg_dict.get("forcing") or {}
    mode = str(forcing_dict.pop("mode", "gandalf_shell"))
    max_nz = int(forcing_dict.pop("max_nz", 1))
    include_nz0 = bool(forcing_dict.pop("include_nz0", False))
    correlation = float(forcing_dict.pop("correlation", 0.0))
    return AlfvenForcingOptions(
        mode=mode,
        max_nz=max_nz,
        include_nz0=include_nz0,
        correlation=correlation,
    )


def apply_alfven_forcing(
    state: KRMHDState,
    forcing_cfg: Any,
    dt: float,
    key: Array,
    options: AlfvenForcingOptions,
) -> tuple[KRMHDState, Array]:
    """Apply the configured Alfvén forcing and return the updated state/key."""
    if not forcing_cfg.enabled:
        return state, key

    n_min = int(forcing_cfg.k_min)
    n_max = int(forcing_cfg.k_max)

    if options.mode == "gaussian_shell":
        return force_alfven_modes(
            state,
            amplitude=forcing_cfg.amplitude,
            n_min=n_min,
            n_max=n_max,
            dt=dt,
            key=key,
        )

    if options.mode == "gandalf_shell":
        return force_alfven_modes_gandalf(
            state,
            fampl=forcing_cfg.amplitude,
            n_min=n_min,
            n_max=n_max,
            dt=dt,
            key=key,
        )

    if options.mode == "balanced_elsasser_lowkz":
        return force_alfven_modes_balanced(
            state,
            amplitude=forcing_cfg.amplitude,
            n_min=n_min,
            n_max=n_max,
            dt=dt,
            key=key,
            max_nz=options.max_nz,
            include_nz0=options.include_nz0,
            correlation=options.correlation,
        )

    if options.mode == "gandalf_perp_lowkz":
        return force_alfven_modes_gandalf_perp_lowkz(
            state,
            fampl=forcing_cfg.amplitude,
            n_min=n_min,
            n_max=n_max,
            max_nz=options.max_nz,
            include_nz0=options.include_nz0,
            dt=dt,
            key=key,
        )

    raise ValueError(
        f"Unknown Alfvén forcing mode {options.mode!r}. "
        "Expected 'gaussian_shell', 'gandalf_shell', "
        "'balanced_elsasser_lowkz', or 'gandalf_perp_lowkz'."
    )


def force_alfven_modes_gandalf_perp_lowkz(
    state: KRMHDState,
    fampl: float,
    n_min: int,
    n_max: int,
    max_nz: int,
    include_nz0: bool,
    dt: float,
    key: Array,
) -> tuple[KRMHDState, Array]:
    """Apply GANDALF-amplitude forcing on a perp band and low-|nz| planes."""
    forcing, key = gandalf_forcing_fourier_perp_lowkz(
        state.grid,
        fampl=fampl,
        n_min=n_min,
        n_max=n_max,
        max_nz=max_nz,
        include_nz0=include_nz0,
        dt=dt,
        key=key,
    )

    z_plus_new = state.z_plus + forcing
    z_minus_new = state.z_minus + forcing

    new_state = KRMHDState(
        z_plus=z_plus_new,
        z_minus=z_minus_new,
        B_parallel=state.B_parallel,
        g=state.g,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        Lambda=state.Lambda,
        time=state.time,
        grid=state.grid,
    )
    return new_state, key


def gandalf_forcing_fourier_perp_lowkz(
    grid: SpectralGrid3D,
    fampl: float,
    n_min: int,
    n_max: int,
    max_nz: int,
    include_nz0: bool,
    dt: float,
    key: Array,
) -> tuple[Array, Array]:
    """Generate GANDALF-style forcing restricted to low-|nz|."""
    if fampl <= 0:
        raise ValueError(f"fampl must be positive, got {fampl}")
    if n_min <= 0 or n_max < n_min:
        raise ValueError(f"Invalid perpendicular band: n_min={n_min}, n_max={n_max}")
    if max_nz < 0:
        raise ValueError(f"max_nz must be >= 0, got {max_nz}")

    kperp_min = 2.0 * jnp.pi * n_min / grid.Lx
    kperp_max = 2.0 * jnp.pi * n_max / grid.Lx

    k1z = 2.0 * jnp.pi / grid.Lz
    nz_float = jnp.round(jnp.abs(grid.kz) / k1z)
    nz_int = nz_float.astype(jnp.int32)
    kz_allowed = nz_int <= max_nz
    if not include_nz0:
        kz_allowed = kz_allowed & (nz_int != 0)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    random_amplitude = jax.random.uniform(
        subkey1,
        shape=(grid.Nz, grid.Ny, grid.Nx // 2 + 1),
        minval=1e-10,
        maxval=1.0,
    )
    random_phase = jax.random.uniform(
        subkey2,
        shape=(grid.Nz, grid.Ny, grid.Nx // 2 + 1),
        minval=0.0,
        maxval=2.0 * jnp.pi,
    )

    forced_field = _gandalf_forcing_fourier_perp_lowkz_jit(
        grid.kx,
        grid.ky,
        grid.kz,
        fampl,
        float(kperp_min),
        float(kperp_max),
        kz_allowed,
        float(dt),
        random_amplitude,
        random_phase,
        grid.Nx,
    )
    return forced_field, key


@partial(jax.jit, static_argnums=(10,))
def _gandalf_forcing_fourier_perp_lowkz_jit(
    kx: Array,
    ky: Array,
    kz: Array,
    fampl: float,
    kperp_min: float,
    kperp_max: float,
    kz_allowed: Array,
    dt: float,
    random_amplitude: Array,
    random_phase: Array,
    nx_full: int,
) -> Array:
    """JIT core for GANDALF-style forcing on a perp band and low-|nz| planes."""
    kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
    ky_3d = ky[jnp.newaxis, :, jnp.newaxis]

    k_perp = jnp.sqrt(kx_3d**2 + ky_3d**2)
    perp_mask = (k_perp >= kperp_min) & (k_perp <= kperp_max)
    kz_mask = kz_allowed[:, jnp.newaxis, jnp.newaxis]
    forcing_mask = perp_mask & kz_mask

    k_perp_safe = jnp.maximum(k_perp, 1e-10)
    log_factor = jnp.sqrt(jnp.abs((fampl / dt) * jnp.log(random_amplitude)))
    forced_field = (1.0 / k_perp_safe) * log_factor * jnp.exp(1j * random_phase)
    forced_field = forced_field * forcing_mask.astype(forced_field.dtype)

    forced_field = forced_field.at[0, 0, 0].set(0.0 + 0.0j)
    forced_field = forced_field.at[:, :, 0].set(
        forced_field[:, :, 0].real.astype(forced_field.dtype)
    )

    if nx_full % 2 == 0:
        nyquist_idx = forced_field.shape[2] - 1
        forced_field = forced_field.at[:, :, nyquist_idx].set(
            forced_field[:, :, nyquist_idx].real.astype(forced_field.dtype)
        )

    return forced_field
