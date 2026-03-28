"""Modal app for running KRMHD simulations on cloud GPUs.

Usage:
    # Run a single config remotely:
    modal run infrastructure/modal_app.py --config-yaml studies/02-collisionality-scan/configs/nu1e-2.yaml

    # The results (diagnostics.npz, timeseries.h5) are saved to the study's data/ dir.
"""

from __future__ import annotations

import modal

app = modal.App("krmhd-research")

# Image: install GANDALF + dependencies into a GPU-capable container
krmhd_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]",
        "gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@v0.4.2",
        "numpy",
        "h5py",
        "pyyaml",
    )
)


@app.function(
    image=krmhd_image,
    gpu="T4",
    timeout=7200,  # 2 hours max
)
def run_simulation_remote(config_yaml: str) -> dict:
    """Run a KRMHD simulation on a cloud GPU.

    Parameters
    ----------
    config_yaml : str
        Contents of the YAML config file (not a path — the actual YAML string).

    Returns
    -------
    dict with keys:
        'diagnostics_npz': bytes — the npz file contents
        'config_yaml': str — echo back the config
        'wall_time': float — wall clock seconds
        'outcome': str — 'pass' or 'fail'
        'gate_summary': str — validation gate results
    """
    import io
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np
    import yaml

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

    from krmhd.config import SimulationConfig
    from krmhd.diagnostics import (
        EnergyHistory,
        compute_energy,
        energy_spectrum_perpendicular,
        hermite_moment_energy,
    )
    from krmhd.forcing import force_alfven_modes_gandalf, force_hermite_moments
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step
    from krmhd.physics import initialize_hermite_moments, KRMHDState

    # Parse config from YAML string
    config = SimulationConfig(**yaml.safe_load(config_yaml))

    # Set up state
    grid = config.create_grid()
    state = config.create_initial_state(grid)

    # Seed g
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

    rng_key = jax.random.PRNGKey(
        forcing_cfg.seed if forcing_cfg.seed is not None else 42
    )
    total_injection = 0.0

    W_m_history: list = []
    epsilon_nu_history: list = []
    save_times: list = []

    # Initial diagnostics
    E_m_init = np.array(hermite_moment_energy(state))
    W_m_history.append(E_m_init)
    save_times.append(float(state.time))

    # Compute collisional dissipation inline (avoid importing shared/)
    def _eps_nu(E_m: np.ndarray) -> float:
        m_idx = np.arange(len(E_m))
        rates = physics.nu * (m_idx / M) ** physics.hyper_n
        rates[:2] = 0.0
        return float(2.0 * np.sum(rates * E_m))

    epsilon_nu_history.append(_eps_nu(E_m_init))

    print(f"Running {ti.n_steps} steps, M={M}, nu={physics.nu}, "
          f"grid={config.grid.Nx}x{config.grid.Ny}x{config.grid.Nz}")
    print(f"Device: {jax.devices()}")

    wall_start = time.time()

    for step in range(1, ti.n_steps + 1):
        dt = compute_cfl_timestep(state, physics.v_A, ti.cfl_safety)

        state = gandalf_step(
            state, dt, physics.eta, physics.v_A,
            nu=physics.nu, hyper_r=physics.hyper_r, hyper_n=physics.hyper_n,
        )
        # Workaround: gandalf_step v0.4.2 RK4 returns time as JAX array,
        # which fails Pydantic validation. Convert back to float.
        if not isinstance(state.time, float):
            state = KRMHDState(
                z_plus=state.z_plus, z_minus=state.z_minus,
                B_parallel=state.B_parallel, g=state.g,
                M=state.M, beta_i=state.beta_i, v_th=state.v_th,
                nu=state.nu, Lambda=state.Lambda,
                time=float(state.time), grid=state.grid,
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

        # Force g_0 (density) and g_1 (momentum) to provide continuous
        # energy source for the Hermite cascade (benchmark default).
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
            E_m = np.array(hermite_moment_energy(state))
            W_m_history.append(E_m)
            save_times.append(float(state.time))
            eps_nu = _eps_nu(E_m)
            epsilon_nu_history.append(eps_nu)

            print(f"  step {step:5d}/{ti.n_steps}  t={state.time:.3f}  "
                  f"E_total={energy['total']:.6f}  eps_nu={eps_nu:.6e}  dt={dt:.6f}")

    wall_time = time.time() - wall_start

    # Final spectra
    k_perp, E_kperp = energy_spectrum_perpendicular(state)
    W_m = hermite_moment_energy(state)

    # Pack diagnostics into npz bytes
    buf = io.BytesIO()
    np.savez(
        buf,
        times=np.array(history.times),
        E_total=np.array(history.E_total),
        E_magnetic=np.array(history.E_magnetic),
        E_kinetic=np.array(history.E_kinetic),
        E_compressive=np.array(history.E_compressive),
        k_perp=np.array(k_perp),
        E_kperp=np.array(E_kperp),
        W_m=np.array(W_m),
        M=M,
        W_m_history=np.array(W_m_history),
        epsilon_nu_history=np.array(epsilon_nu_history),
        save_times=np.array(save_times),
        nu=physics.nu,
        eta=physics.eta,
        hyper_n=physics.hyper_n,
        hyper_r=physics.hyper_r,
        total_injection=total_injection,
    )
    npz_bytes = buf.getvalue()

    # Simple validation
    diss_rates = np.array(history.dissipation_rate())
    mean_diss = float(np.mean(np.abs(diss_rates[len(diss_rates) // 2:]))) if len(diss_rates) > 2 else 0.0
    sim_time = float(state.time)
    mean_injection = total_injection / sim_time if sim_time > 0 else 0.0

    # Energy conservation check
    E = np.array(history.E_total)
    E_ss = E[len(E) // 2:]
    E_mean = np.mean(E_ss)
    rel_error = float(np.max(np.abs(E_ss - E_mean)) / E_mean) if E_mean > 0 else 1.0

    # Steady state check
    fluct = float(np.std(E_ss) / E_mean) if E_mean > 0 else 1.0

    gates = {
        "energy_conservation": rel_error < 0.01,
        "steady_state": fluct < 0.10,
    }
    outcome = "pass" if all(gates.values()) else "fail"
    gate_summary = ", ".join(f"{k}:{'ok' if v else 'FAIL'}" for k, v in gates.items())

    print(f"\nSimulation complete in {wall_time:.1f}s")
    print(f"Gates: {gate_summary}")

    return {
        "diagnostics_npz": npz_bytes,
        "config_yaml": config_yaml,
        "wall_time": wall_time,
        "outcome": outcome,
        "gate_summary": gate_summary,
    }


@app.local_entrypoint()
def main(config_yaml: str):
    """CLI entrypoint: run a config on Modal and save results locally."""
    from pathlib import Path

    config_path = Path(config_yaml)
    if not config_path.exists():
        print(f"Config not found: {config_yaml}")
        raise SystemExit(1)

    yaml_contents = config_path.read_text()
    print(f"Submitting {config_path.name} to Modal...")

    result = run_simulation_remote.remote(yaml_contents)

    print(f"Completed in {result['wall_time']:.1f}s — {result['outcome']}")
    print(f"Gates: {result['gate_summary']}")

    # Save diagnostics locally
    # Determine output dir from config
    import yaml
    cfg = yaml.safe_load(yaml_contents)
    output_dir = Path(cfg.get("io", {}).get("output_dir", "data"))
    if not output_dir.is_absolute():
        # Resolve relative to the config file's study dir
        study_dir = config_path.resolve().parents[1]
        output_dir = study_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate run ID
    from datetime import datetime
    param_label = config_path.stem.split("_")[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"02_{param_label}_{timestamp}"

    npz_path = output_dir / f"{run_id}_diagnostics.npz"
    npz_path.write_bytes(result["diagnostics_npz"])
    print(f"Saved: {npz_path}")
