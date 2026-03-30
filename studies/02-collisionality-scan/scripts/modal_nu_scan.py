"""Submit multiple ν test runs to Modal in parallel.

Creates temporary YAML configs and submits each to Modal.
Shorter runs (20k steps) to quickly find the stability threshold.

Usage:
    modal run studies/02-collisionality-scan/scripts/modal_nu_scan.py
"""
from __future__ import annotations

import io
import time

import modal
import yaml

app = modal.App("krmhd-nu-scan")

krmhd_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]",
        "gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@v0.4.3",
        "numpy",
        "h5py",
        "pyyaml",
    )
)

# Base config — same as production but shorter runs
BASE_CONFIG = {
    "name": "nu_stability_test",
    "description": "Quick stability test",
    "grid": {"Nx": 64, "Ny": 64, "Nz": 32, "Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
    "physics": {
        "v_A": 1.0, "eta": 2.0, "beta_i": 1.0, "Lambda": 2.236,
        "hyper_r": 2, "hyper_n": 6,
    },
    "initial_condition": {
        "type": "random_spectrum", "amplitude": 0.05, "alpha": 1.667,
        "k_min": 1.0, "k_max": 10.0, "k_wave": [0.0, 0.0, 1.0], "M": 32,  # M=32 stabilizes
    },
    "hermite_forcing": {"amplitude": 0.0035, "forced_moments": [0]},
    "forcing": {"enabled": True, "amplitude": 0.005, "k_min": 1.0, "k_max": 2.0},
    "time_integration": {"n_steps": 40000, "cfl_safety": 0.3, "save_interval": 2000},
    "io": {
        "output_dir": "data", "save_spectra": True,
        "save_energy_history": True, "save_fields": False,
        "save_final_state": True, "overwrite": False,
    },
}

# Test these ν values — wider range, extending to low ν
NU_VALUES = [100.0, 10.0, 1.0, 0.25, 0.1, 0.01, 0.001, 0.0001]


@app.function(image=krmhd_image, gpu="A100", timeout=7200)
def run_nu_test(nu: float) -> dict:
    """Run a short stability test at a given ν."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

    from krmhd.config import SimulationConfig
    from krmhd.diagnostics import compute_energy, hermite_moment_energy
    from krmhd.forcing import force_alfven_modes_gandalf, force_hermite_moments
    from krmhd.physics import initialize_hermite_moments, KRMHDState
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step

    cfg = dict(BASE_CONFIG)
    cfg["physics"] = dict(cfg["physics"])
    cfg["physics"]["nu"] = nu
    cfg["name"] = f"nu_test_{nu:.1e}"

    hermite_cfg = cfg.pop("hermite_forcing", {})
    config = SimulationConfig(**cfg)

    hermite_amplitude = hermite_cfg.get("amplitude", 0.0035)
    hermite_moments = tuple(hermite_cfg.get("forced_moments", [0]))

    grid = config.create_grid()
    state = config.create_initial_state(grid)
    M = config.initial_condition.M

    g_seed = initialize_hermite_moments(grid, M, perturbation_amplitude=1e-3, seed=137)
    state = KRMHDState(
        z_plus=state.z_plus, z_minus=state.z_minus,
        B_parallel=state.B_parallel, g=g_seed,
        M=state.M, beta_i=state.beta_i, v_th=state.v_th,
        nu=state.nu, Lambda=state.Lambda, time=state.time, grid=state.grid,
    )

    physics = config.physics
    ti = config.time_integration
    forcing_cfg = config.forcing
    rng_key = jax.random.PRNGKey(42)

    def _eps_nu(E_m):
        m_idx = np.arange(len(E_m))
        rates = nu * (m_idx / M) ** physics.hyper_n
        rates[:2] = 0.0
        return float(2.0 * np.sum(rates * E_m))

    eps_history = []
    time_history = []
    etotal_history = []

    print(f"\n=== nu={nu:.1e}, M={M}, hyper_n={physics.hyper_n}, {ti.n_steps} steps ===")
    print(f"Device: {jax.devices()}")

    wall_start = time.time()

    for step in range(1, ti.n_steps + 1):
        dt = compute_cfl_timestep(state, physics.v_A, ti.cfl_safety)
        state = gandalf_step(
            state, dt, physics.eta, physics.v_A,
            nu=nu, hyper_r=physics.hyper_r, hyper_n=physics.hyper_n,
        )
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
            state, _ = force_alfven_modes_gandalf(
                state, fampl=forcing_cfg.amplitude,
                n_min=int(forcing_cfg.k_min), n_max=int(forcing_cfg.k_max),
                dt=dt, key=subkey,
            )
        rng_key, subkey = jax.random.split(rng_key)
        state, _ = force_hermite_moments(
            state, amplitude=hermite_amplitude,
            n_min=int(forcing_cfg.k_min), n_max=int(forcing_cfg.k_max),
            dt=dt, key=subkey, forced_moments=hermite_moments,
        )

        if step % ti.save_interval == 0:
            energy = compute_energy(state)
            E_m = np.array(hermite_moment_energy(state))
            eps = _eps_nu(E_m)
            eps_history.append(eps)
            time_history.append(float(state.time))
            etotal_history.append(float(energy["total"]))
            print(f"  step {step:5d}  t={state.time:.2f}  E_total={energy['total']:.4f}  eps_nu={eps:.3e}")

    wall_time = time.time() - wall_start

    # Check stability: is eps_nu growing or stable?
    if len(eps_history) >= 4:
        last_quarter = eps_history[len(eps_history)*3//4:]
        first_quarter = eps_history[:len(eps_history)//4+1]
        mean_last = np.mean([abs(e) for e in last_quarter if np.isfinite(e)])
        mean_first = np.mean([abs(e) for e in first_quarter if np.isfinite(e)])
        growth = mean_last / mean_first if mean_first > 0 else float('inf')
        stable = growth < 10  # Less than 10× growth
    else:
        growth = float('inf')
        stable = False

    verdict = "STABLE" if stable else "BLOWUP"
    print(f"\n  nu={nu:.1e}: {verdict} (growth={growth:.1f}x, wall={wall_time:.0f}s)")

    # Return only plain Python types to avoid numpy deserialization issues
    return {
        "nu": float(nu),
        "eps_history": [float(e) for e in eps_history],
        "time_history": [float(t) for t in time_history],
        "etotal_history": [float(e) for e in etotal_history],
        "growth_factor": float(growth),
        "stable": bool(stable),
        "wall_time": float(wall_time),
    }


@app.local_entrypoint()
def main():
    """Submit all ν tests in parallel."""
    print(f"Submitting {len(NU_VALUES)} stability tests to Modal...")
    print(f"ν values: {NU_VALUES}")

    # Submit all in parallel
    results = []
    for result in run_nu_test.map(NU_VALUES):
        results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print("STABILITY SCAN SUMMARY")
    print("=" * 70)
    print(f"{'nu':>10s}  {'verdict':>10s}  {'growth':>10s}  {'final_eps':>12s}  {'wall(s)':>8s}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["nu"], reverse=True):
        eps_final = r["eps_history"][-1] if r["eps_history"] else float('nan')
        print(f"{r['nu']:10.1e}  {'STABLE' if r['stable'] else 'BLOWUP':>10s}  "
              f"{r['growth_factor']:10.1f}x  {eps_final:12.3e}  {r['wall_time']:8.0f}")
