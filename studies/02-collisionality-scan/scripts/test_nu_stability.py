"""Quick ν stability scan — find the threshold where Hermite cascade stabilizes.

Runs 5000 steps at each ν value with M=128, hyper_n=6, and prints eps_nu trend.
"""
import sys
import yaml
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
DEV_CONFIG = STUDY_DIR / "configs" / "nu1e-3_dev.yaml"

NU_VALUES = [100.0, 10.0, 1.0, 0.5, 0.25, 0.1]
N_STEPS = 3000
SAVE_INTERVAL = 500
M_OVERRIDE = 32  # Use small M for fast CPU iteration; stability threshold is similar


def run_one(nu: float) -> None:
    import jax
    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")
    import jax.numpy as jnp
    import numpy as np

    from krmhd.config import SimulationConfig
    from krmhd.diagnostics import compute_energy, hermite_moment_energy
    from krmhd.forcing import force_alfven_modes_gandalf, force_hermite_moments
    from krmhd.physics import initialize_hermite_moments, KRMHDState
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step

    cfg_dict = yaml.safe_load(DEV_CONFIG.read_text())
    hermite_cfg = cfg_dict.pop("hermite_forcing", {})
    cfg_dict["physics"]["nu"] = nu
    cfg_dict["initial_condition"]["M"] = M_OVERRIDE
    cfg_dict["time_integration"]["n_steps"] = N_STEPS
    cfg_dict["time_integration"]["save_interval"] = SAVE_INTERVAL
    config = SimulationConfig(**cfg_dict)

    grid = config.create_grid()
    state = config.create_initial_state(grid)

    g_seed = initialize_hermite_moments(
        grid, config.initial_condition.M, perturbation_amplitude=1e-3, seed=137
    )
    state = KRMHDState(
        z_plus=state.z_plus, z_minus=state.z_minus,
        B_parallel=state.B_parallel, g=g_seed,
        M=state.M, beta_i=state.beta_i, v_th=state.v_th,
        nu=state.nu, Lambda=state.Lambda, time=state.time, grid=state.grid,
    )

    physics = config.physics
    forcing_cfg = config.forcing
    M = config.initial_condition.M
    hermite_amplitude = hermite_cfg.get("amplitude", 0.0035)
    hermite_moments = tuple(hermite_cfg.get("forced_moments", [0]))

    rng_key = jax.random.PRNGKey(42)

    def _eps_nu(E_m):
        m_idx = np.arange(len(E_m))
        rates = nu * (m_idx / M) ** physics.hyper_n
        rates[:2] = 0.0
        return float(2.0 * np.sum(rates * E_m))

    print(f"\n  nu={nu:.1e}  (M={M}, hyper_n={physics.hyper_n}, {N_STEPS} steps)")

    for step in range(1, N_STEPS + 1):
        dt = compute_cfl_timestep(state, physics.v_A, config.time_integration.cfl_safety)
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

        if step % SAVE_INTERVAL == 0:
            E_m = np.array(hermite_moment_energy(state))
            eps = _eps_nu(E_m)
            energy = compute_energy(state)
            print(f"    step {step:5d}  t={state.time:.2f}  E_total={energy['total']:.4f}  eps_nu={eps:.3e}")

    # Final verdict
    E_m = np.array(hermite_moment_energy(state))
    eps_final = _eps_nu(E_m)
    print(f"  → FINAL eps_nu={eps_final:.3e}  {'STABLE' if eps_final < 100 else 'BLOWUP'}")


if __name__ == "__main__":
    for nu in NU_VALUES:
        run_one(nu)
