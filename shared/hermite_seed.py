from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HermiteSeedOptions:
    enabled: bool = True
    amplitude: float = 1e-3
    seed: int = 137


def pop_hermite_seed_options(cfg_dict: dict[str, Any]) -> HermiteSeedOptions:
    """Extract study-local Hermite seed options from a YAML config dict."""
    seed_cfg = cfg_dict.pop("hermite_seed", {})
    return HermiteSeedOptions(
        enabled=bool(seed_cfg.get("enabled", True)),
        amplitude=float(seed_cfg.get("amplitude", 1e-3)),
        seed=int(seed_cfg.get("seed", 137)),
    )


def apply_hermite_seed(state, *, options: HermiteSeedOptions):
    """Seed g with a configurable perturbation to avoid the g=0 fixed point."""
    if not options.enabled or options.amplitude <= 0.0:
        return state

    from krmhd.physics import KRMHDState, initialize_hermite_moments

    g_seed = initialize_hermite_moments(
        state.grid,
        state.M,
        perturbation_amplitude=options.amplitude,
        seed=options.seed,
    )
    return KRMHDState(
        z_plus=state.z_plus,
        z_minus=state.z_minus,
        B_parallel=state.B_parallel,
        g=g_seed,
        M=state.M,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=state.nu,
        Lambda=state.Lambda,
        time=state.time,
        grid=state.grid,
    )
