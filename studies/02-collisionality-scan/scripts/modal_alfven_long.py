"""Extended Alfvénic steady state test — 200k steps on best candidates.

From the scan, the best candidates are:
  eta=2, fampl=0.001 (E_final=26, var=35%)
  eta=10, fampl=0.002 (E_final=46, var=34%)
  eta=20, fampl=0.005 (E_final=99, var=33%)

Run these for 200k steps to confirm true steady state.
Also add Hermite forcing to the best one to test combined stability.
"""
from __future__ import annotations
import time
import modal

app = modal.App("krmhd-alfven-long")

krmhd_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]",
        "gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@v0.4.3",
        "numpy", "h5py", "pyyaml",
    )
)

# (eta, fampl, include_hermite, label)
RUNS = [
    # Alfvénic-only long runs
    (2.0, 0.001, False, "alfven_eta2_f0.001"),
    (10.0, 0.002, False, "alfven_eta10_f0.002"),
    (20.0, 0.005, False, "alfven_eta20_f0.005"),
    # Combined: Alfvénic + Hermite forcing
    (2.0, 0.001, True, "combined_eta2_f0.001"),
    (10.0, 0.002, True, "combined_eta10_f0.002"),
    (20.0, 0.005, True, "combined_eta20_f0.005"),
]


@app.function(image=krmhd_image, gpu="A100", timeout=14400)
def run_test(eta: float, fampl: float, include_hermite: bool, label: str) -> dict:
    """Run extended Alfvénic cascade test."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

    from krmhd.config import SimulationConfig
    from krmhd.diagnostics import compute_energy, hermite_moment_energy
    from krmhd.forcing import force_alfven_modes_gandalf, force_hermite_moments
    from krmhd.physics import initialize_hermite_moments, KRMHDState
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step

    M = 32
    n_steps = 200000
    save_interval = 5000
    nu = 1.0
    hyper_n = 6

    config = SimulationConfig(
        name=label,
        description=f"Extended test: eta={eta}, fampl={fampl}, hermite={include_hermite}",
        grid={"Nx": 64, "Ny": 64, "Nz": 32, "Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
        physics={"v_A": 1.0, "eta": eta, "nu": nu, "beta_i": 1.0,
                 "Lambda": 2.236, "hyper_r": 2, "hyper_n": hyper_n},
        initial_condition={"type": "random_spectrum", "amplitude": 0.05,
                          "alpha": 1.667, "k_min": 1.0, "k_max": 10.0,
                          "k_wave": [0.0, 0.0, 1.0], "M": M},
        forcing={"enabled": True, "amplitude": fampl, "k_min": 1.0, "k_max": 2.0},
        time_integration={"n_steps": n_steps, "cfl_safety": 0.3,
                         "save_interval": save_interval},
        io={"output_dir": "data", "save_spectra": True,
            "save_energy_history": True, "save_fields": False,
            "save_final_state": False, "overwrite": True},
    )

    grid = config.create_grid()
    state = config.create_initial_state(grid)
    physics = config.physics
    ti = config.time_integration
    forcing_cfg = config.forcing
    rng_key = jax.random.PRNGKey(42)

    if include_hermite:
        g_seed = initialize_hermite_moments(grid, M, perturbation_amplitude=1e-3, seed=137)
        state = KRMHDState(
            z_plus=state.z_plus, z_minus=state.z_minus,
            B_parallel=state.B_parallel, g=g_seed,
            M=state.M, beta_i=state.beta_i, v_th=state.v_th,
            nu=state.nu, Lambda=state.Lambda, time=state.time, grid=state.grid,
        )

    etotal_history = []
    eps_nu_history = []
    time_history = []

    def _eps_nu(E_m):
        m_idx = np.arange(len(E_m))
        rates = nu * (m_idx / M) ** hyper_n
        rates[:2] = 0.0
        return float(2.0 * np.sum(rates * E_m))

    hermite_amplitude = 0.0035
    hermite_moments_forced = (0,)

    print(f"\n=== {label}: eta={eta}, fampl={fampl}, hermite={include_hermite} ===")
    print(f"  M={M}, nu={nu}, hyper_n={hyper_n}, {n_steps} steps")
    print(f"  Device: {jax.devices()}")

    wall_start = time.time()

    for step in range(1, n_steps + 1):
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

        if include_hermite:
            rng_key, subkey = jax.random.split(rng_key)
            state, _ = force_hermite_moments(
                state, amplitude=hermite_amplitude,
                n_min=int(forcing_cfg.k_min), n_max=int(forcing_cfg.k_max),
                dt=dt, key=subkey, forced_moments=hermite_moments_forced,
            )

        if step % save_interval == 0:
            energy = compute_energy(state)
            etotal = float(energy["total"])
            etotal_history.append(etotal)
            time_history.append(float(state.time))
            E_m = np.array(hermite_moment_energy(state))
            eps = _eps_nu(E_m)
            eps_nu_history.append(eps)
            print(f"  step {step:6d}  t={state.time:.1f}  E_total={etotal:.4f}  eps_nu={eps:.3e}")

    wall_time = time.time() - wall_start

    # Steady state check on second half
    half = len(etotal_history) // 2
    second_half_e = etotal_history[half:]
    mean_e = sum(second_half_e) / len(second_half_e)
    max_dev = max(abs(e - mean_e) for e in second_half_e) if second_half_e else float('inf')
    rel_var_e = max_dev / mean_e if mean_e > 0 else float('inf')

    second_half_eps = eps_nu_history[half:]
    mean_eps = sum(abs(e) for e in second_half_eps) / len(second_half_eps) if second_half_eps else 0
    max_dev_eps = max(abs(e) - mean_eps for e in second_half_eps) if second_half_eps else float('inf')
    rel_var_eps = abs(max_dev_eps / mean_eps) if mean_eps > 0 else float('inf')

    alfven_steady = rel_var_e < 0.3
    hermite_steady = rel_var_eps < 0.5 if include_hermite else True
    both_steady = alfven_steady and hermite_steady

    print(f"\n  {label}: E_var={rel_var_e:.1%}, eps_var={rel_var_eps:.1%}")
    print(f"  Alfvén={'STEADY' if alfven_steady else 'NOT'}, "
          f"Hermite={'STEADY' if hermite_steady else 'NOT'} → "
          f"{'BOTH STEADY' if both_steady else 'NEEDS WORK'}")

    return {
        "label": label,
        "eta": float(eta),
        "fampl": float(fampl),
        "include_hermite": include_hermite,
        "etotal_history": [float(e) for e in etotal_history],
        "eps_nu_history": [float(e) for e in eps_nu_history],
        "time_history": [float(t) for t in time_history],
        "alfven_steady": bool(alfven_steady),
        "hermite_steady": bool(hermite_steady),
        "both_steady": bool(both_steady),
        "e_variation": float(rel_var_e),
        "eps_variation": float(rel_var_eps),
        "wall_time": float(wall_time),
    }


@app.local_entrypoint()
def main():
    """Submit all tests in parallel."""
    print(f"Submitting {len(RUNS)} extended tests...")

    inputs = [(eta, fampl, hermite, label) for eta, fampl, hermite, label in RUNS]
    results = list(run_test.starmap(inputs))

    print("\n" + "=" * 80)
    print("EXTENDED STABILITY RESULTS (200k steps)")
    print("=" * 80)
    print(f"{'Label':<28s}  {'E_var':>8s}  {'eps_var':>8s}  {'E_final':>12s}  {'Verdict'}")
    print("-" * 80)
    for r in results:
        verdict = "BOTH STEADY" if r["both_steady"] else (
            "Alfvén OK" if r["alfven_steady"] else "NOT STEADY"
        )
        print(f"{r['label']:<28s}  {r['e_variation']:>7.1%}  {r['eps_variation']:>7.1%}  "
              f"{r['etotal_history'][-1]:>12.4f}  {verdict}")
