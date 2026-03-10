"""Run a single KRMHD simulation locally and save diagnostics.

Usage:
    uv run python studies/01-hermite-convergence/scripts/run_local.py configs/M004_dev.yaml

Relative config paths are resolved from studies/01-hermite-convergence/.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Add project root to path for shared imports
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from krmhd.config import SimulationConfig
from krmhd.diagnostics import (
    EnergyHistory,
    compute_energy,
    energy_spectrum_perpendicular,
    hermite_moment_energy,
)
from krmhd.forcing import force_alfven_modes_gandalf
from krmhd.io import save_checkpoint, save_timeseries
from krmhd.timestepping import compute_cfl_timestep, gandalf_step

from shared.run_utils import detect_hardware, generate_run_id, log_run
from shared.validation import print_gate_results, run_all_gates


STUDY_DIR = Path(__file__).resolve().parents[1]


def resolve_config_path(config_arg: str) -> Path:
    """Resolve config path relative to study dir or as absolute."""
    p = Path(config_arg)
    if p.is_absolute():
        return p
    # Try relative to study dir
    candidate = STUDY_DIR / p
    if candidate.exists():
        return candidate
    # Try relative to cwd
    candidate = Path.cwd() / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def run_simulation(config: SimulationConfig) -> tuple:
    """Run the KRMHD simulation loop.

    Returns
    -------
    state : KRMHDState
        Final simulation state.
    history : EnergyHistory
        Time series of energy components.
    total_injection : float
        Cumulative energy injected by forcing.
    """
    grid = config.create_grid()
    state = config.create_initial_state(grid)

    physics = config.physics
    ti = config.time_integration
    forcing_cfg = config.forcing

    history = EnergyHistory()
    history.append(state)

    rng_key = jax.random.PRNGKey(forcing_cfg.seed if forcing_cfg.seed is not None else 42)
    total_injection = 0.0

    print(f"Running {ti.n_steps} steps, M={config.initial_condition.M}, "
          f"grid={config.grid.Nx}x{config.grid.Ny}x{config.grid.Nz}")

    for step in range(1, ti.n_steps + 1):
        dt = compute_cfl_timestep(state, physics.v_A, ti.cfl_safety)

        # Time integration step
        state = gandalf_step(
            state, dt, physics.eta, physics.v_A,
            nu=physics.nu, hyper_r=physics.hyper_r, hyper_n=physics.hyper_n,
        )

        # Apply forcing if enabled
        if forcing_cfg.enabled:
            rng_key, subkey = jax.random.split(rng_key)
            state, inj_energy = force_alfven_modes_gandalf(
                state,
                fampl=forcing_cfg.amplitude,
                n_min=int(forcing_cfg.k_min),
                n_max=int(forcing_cfg.k_max),
                dt=dt,
                key=subkey,
            )
            total_injection += float(jnp.sum(inj_energy))

        # Record diagnostics at save intervals
        if step % ti.save_interval == 0:
            history.append(state)
            energy = compute_energy(state)
            print(f"  step {step:5d}/{ti.n_steps}  t={state.time:.3f}  "
                  f"E_total={energy['total']:.6f}  dt={dt:.6f}")

    return state, history, total_injection


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config_path = resolve_config_path(sys.argv[1])
    print(f"Loading config: {config_path}")
    config = SimulationConfig.from_yaml(str(config_path))

    # Extract param label from config name (e.g. M004_dev -> M004)
    param_label = config_path.stem.split("_")[0]
    run_id = generate_run_id("01", param_label)
    print(f"Run ID: {run_id}")

    # Run simulation
    wall_start = time.time()
    state, history, total_injection = run_simulation(config)
    wall_time = time.time() - wall_start
    print(f"\nSimulation complete in {wall_time:.1f}s")

    # Save diagnostics
    data_dir = Path(config.io.output_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Save energy time series
    ts_path = str(data_dir / f"{run_id}_timeseries.h5")
    save_timeseries(history, ts_path, metadata={"run_id": run_id}, overwrite=True)
    print(f"Saved timeseries: {ts_path}")

    # Save final state checkpoint
    ckpt_path = str(data_dir / f"{run_id}_checkpoint.h5")
    save_checkpoint(state, ckpt_path, metadata={"run_id": run_id}, overwrite=True)
    print(f"Saved checkpoint: {ckpt_path}")

    # Save diagnostic summary as npz
    k_perp, E_kperp = energy_spectrum_perpendicular(state)
    W_m = hermite_moment_energy(state)
    npz_path = str(data_dir / f"{run_id}_diagnostics.npz")
    np.savez(
        npz_path,
        times=np.array(history.times),
        E_total=np.array(history.E_total),
        E_magnetic=np.array(history.E_magnetic),
        E_kinetic=np.array(history.E_kinetic),
        E_compressive=np.array(history.E_compressive),
        k_perp=np.array(k_perp),
        E_kperp=np.array(E_kperp),
        W_m=np.array(W_m),
        M=config.initial_condition.M,
    )
    print(f"Saved diagnostics: {npz_path}")

    # Compute dissipation rate estimate from energy history
    diss_rates = np.array(history.dissipation_rate())
    mean_diss = float(np.mean(np.abs(diss_rates[len(diss_rates) // 2:]))) if len(diss_rates) > 2 else 0.0
    n_steps = config.time_integration.n_steps
    sim_time = float(state.time)
    mean_injection = total_injection / sim_time if sim_time > 0 else 0.0

    # Run validation gates
    results = run_all_gates(
        history, state,
        injection_rate=mean_injection,
        dissipation_rate=mean_diss,
        tau_A=1.0,
    )
    print_gate_results(results)

    # Determine outcome
    outcome = "pass" if all(r.passed for r in results) else "fail"
    gate_summary = ", ".join(
        f"{r.name}:{'ok' if r.passed else 'FAIL'}" for r in results
    )

    # Log the run
    hardware = detect_hardware()
    log_run(
        run_id=run_id,
        config_path=str(config_path.relative_to(PROJECT_ROOT)),
        hardware=hardware,
        wall_time=wall_time,
        outcome=outcome,
        notes=gate_summary,
    )
    print(f"Logged to docs/run_log.md")


if __name__ == "__main__":
    main()
