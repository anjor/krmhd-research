"""Run a single KRMHD simulation locally for Study 02 (collisionality scan).

Usage:
    uv run python studies/02-collisionality-scan/scripts/run_local.py configs/nu1e-3.yaml

Relative config paths are resolved from studies/02-collisionality-scan/.

Key differences from Study 01 run_local.py:
    - Saves Hermite moment spectrum W(m) at every save_interval
    - Computes collisional dissipation rate time series
    - Saves nu, eta, hyper_n, hyper_r in diagnostics npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import jax

# Enable persistent JIT compilation cache to avoid recompiling for each nu value
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

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
from krmhd.forcing import force_alfven_modes_gandalf, force_hermite_moments
from krmhd.io import save_checkpoint, save_timeseries
from krmhd.timestepping import compute_cfl_timestep, gandalf_step

from shared.dissipation import compute_collisional_dissipation
from shared.run_utils import detect_hardware, generate_run_id, log_run
from shared.validation import print_gate_results, run_all_gates


STUDY_DIR = Path(__file__).resolve().parents[1]


def resolve_config_path(config_arg: str) -> Path:
    """Resolve config path relative to study dir or as absolute."""
    p = Path(config_arg)
    if p.is_absolute():
        return p
    candidate = STUDY_DIR / p
    if candidate.exists():
        return candidate
    candidate = Path.cwd() / p
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {config_arg}")


def run_simulation(config: SimulationConfig) -> tuple:
    """Run the KRMHD simulation loop with extended diagnostics.

    Returns
    -------
    state : KRMHDState
        Final simulation state.
    history : EnergyHistory
        Time series of energy components.
    total_injection : float
        Cumulative energy injected by forcing.
    W_m_history : list[np.ndarray]
        Hermite moment spectrum at each save interval.
    epsilon_nu_history : list[float]
        Collisional dissipation rate at each save interval.
    save_times : list[float]
        Simulation times at each save interval.
    """
    grid = config.create_grid()
    state = config.create_initial_state(grid)

    # Seed g with tiny perturbation to break the g=0 fixed point.
    # With Lambda != 1, the natural coupling from z+/z- will amplify this
    # into a physical Hermite cascade.
    from krmhd.physics import initialize_hermite_moments, KRMHDState
    g_seed = initialize_hermite_moments(
        grid, config.initial_condition.M,
        perturbation_amplitude=1e-3,
        seed=137,
    )
    state = KRMHDState(
        z_plus=state.z_plus, z_minus=state.z_minus,
        B_parallel=state.B_parallel, g=g_seed,
        M=state.M, beta_i=state.beta_i, v_th=state.v_th,
        nu=state.nu, Lambda=state.Lambda, time=state.time, grid=state.grid,
    )

    physics = config.physics
    ti = config.time_integration
    forcing_cfg = config.forcing
    M = config.initial_condition.M

    history = EnergyHistory()
    history.append(state)

    rng_key = jax.random.PRNGKey(forcing_cfg.seed if forcing_cfg.seed is not None else 42)
    total_injection = 0.0

    # Extended diagnostics for Study 02
    W_m_history: list[np.ndarray] = []
    epsilon_nu_history: list[float] = []
    save_times: list[float] = []

    # Record initial state diagnostics
    E_m_init = np.array(hermite_moment_energy(state))
    W_m_history.append(E_m_init)
    epsilon_nu_history.append(
        compute_collisional_dissipation(E_m_init, physics.nu, M, physics.hyper_n)
    )
    save_times.append(float(state.time))

    print(f"Running {ti.n_steps} steps, M={M}, nu={physics.nu}, "
          f"grid={config.grid.Nx}x{config.grid.Ny}x{config.grid.Nz}")

    for step in range(1, ti.n_steps + 1):
        dt = compute_cfl_timestep(state, physics.v_A, ti.cfl_safety)

        state = gandalf_step(
            state, dt, physics.eta, physics.v_A,
            nu=physics.nu, hyper_r=physics.hyper_r, hyper_n=physics.hyper_n,
        )

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

        # Continuously force g_0 (density) and g_1 (momentum) to provide
        # a steady energy source for the Hermite cascade. Matches the
        # GANDALF hermite_cascade_benchmark.py default.
        rng_key, subkey = jax.random.split(rng_key)
        state, _ = force_hermite_moments(
            state,
            amplitude=0.15,
            n_min=int(forcing_cfg.k_min),
            n_max=int(forcing_cfg.k_max),
            dt=dt,
            key=subkey,
            forced_moments=(0, 1),
        )

        if step % ti.save_interval == 0:
            history.append(state)
            energy = compute_energy(state)

            # Extended diagnostics: Hermite spectrum and dissipation rate
            E_m = np.array(hermite_moment_energy(state))
            W_m_history.append(E_m)
            save_times.append(float(state.time))
            eps_nu = compute_collisional_dissipation(
                E_m, physics.nu, M, physics.hyper_n
            )
            epsilon_nu_history.append(eps_nu)

            print(f"  step {step:5d}/{ti.n_steps}  t={state.time:.3f}  "
                  f"E_total={energy['total']:.6f}  eps_nu={eps_nu:.6e}  dt={dt:.6f}")

    return state, history, total_injection, W_m_history, epsilon_nu_history, save_times


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <config.yaml>")
        sys.exit(1)

    config_path = resolve_config_path(sys.argv[1])
    print(f"Loading config: {config_path}")
    config = SimulationConfig.from_yaml(str(config_path))

    # Extract param label from config filename (e.g. nu1e-3_dev -> nu1e-3)
    param_label = config_path.stem.split("_")[0]
    run_id = generate_run_id("02", param_label)
    print(f"Run ID: {run_id}")

    # Run simulation
    wall_start = time.time()
    state, history, total_injection, W_m_history, epsilon_nu_history, save_times = (
        run_simulation(config)
    )
    wall_time = time.time() - wall_start
    print(f"\nSimulation complete in {wall_time:.1f}s")

    # Save diagnostics — resolve output_dir relative to study dir
    data_dir = Path(config.io.output_dir)
    if not data_dir.is_absolute():
        data_dir = STUDY_DIR / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    ts_path = str(data_dir / f"{run_id}_timeseries.h5")
    save_timeseries(history, ts_path, metadata={"run_id": run_id}, overwrite=True)
    print(f"Saved timeseries: {ts_path}")

    ckpt_path = str(data_dir / f"{run_id}_checkpoint.h5")
    save_checkpoint(state, ckpt_path, metadata={"run_id": run_id}, overwrite=True)
    print(f"Saved checkpoint: {ckpt_path}")

    # Save diagnostic summary with extended Study 02 data
    k_perp, E_kperp = energy_spectrum_perpendicular(state)
    W_m = hermite_moment_energy(state)
    npz_path = str(data_dir / f"{run_id}_diagnostics.npz")
    np.savez(
        npz_path,
        # Energy time series
        times=np.array(history.times),
        E_total=np.array(history.E_total),
        E_magnetic=np.array(history.E_magnetic),
        E_kinetic=np.array(history.E_kinetic),
        E_compressive=np.array(history.E_compressive),
        # Final-state spectra
        k_perp=np.array(k_perp),
        E_kperp=np.array(E_kperp),
        W_m=np.array(W_m),
        M=config.initial_condition.M,
        # Study 02 extended diagnostics
        W_m_history=np.array(W_m_history),
        epsilon_nu_history=np.array(epsilon_nu_history),
        save_times=np.array(save_times),
        nu=config.physics.nu,
        eta=config.physics.eta,
        hyper_n=config.physics.hyper_n,
        hyper_r=config.physics.hyper_r,
        total_injection=total_injection,
    )
    print(f"Saved diagnostics: {npz_path}")

    # Compute dissipation rate estimate from energy history
    diss_rates = np.array(history.dissipation_rate())
    mean_diss = float(np.mean(np.abs(diss_rates[len(diss_rates) // 2:]))) if len(diss_rates) > 2 else 0.0
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

    outcome = "pass" if all(r.passed for r in results) else "fail"
    gate_summary = ", ".join(
        f"{r.name}:{'ok' if r.passed else 'FAIL'}" for r in results
    )

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
