"""Quick ν stability scan — find the threshold where Hermite cascade stabilizes.

By default this runs the historical CPU scan. It also supports focused A/B tests
of the Alfvén forcing path so we can compare the old full-|k| shell drive
against the new low-|n_z| variant without editing the study configs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

STUDY_DIR = Path(__file__).resolve().parents[1]
DEV_CONFIG = STUDY_DIR / "configs" / "nu1e-3_dev.yaml"

NU_VALUES = [100.0, 10.0, 1.0, 0.5, 0.25, 0.1]
N_STEPS = 3000
SAVE_INTERVAL = 500
M_OVERRIDE = 32  # Use small M for fast CPU iteration; stability threshold is similar


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEV_CONFIG,
        help="Config file to use (default: Study 02 dev config).",
    )
    parser.add_argument(
        "--nu",
        type=float,
        action="append",
        dest="nu_values",
        help="Single ν value to test. Repeat to test multiple values.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=N_STEPS,
        help=f"Number of timesteps to run (default: {N_STEPS}).",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=SAVE_INTERVAL,
        help=f"Print interval in timesteps (default: {SAVE_INTERVAL}).",
    )
    parser.add_argument(
        "--m-override",
        type=int,
        default=M_OVERRIDE,
        help=f"Hermite truncation to use for the test (default: {M_OVERRIDE}).",
    )
    parser.add_argument(
        "--forcing-mode",
        choices=(
            "gaussian_shell",
            "gandalf_shell",
            "balanced_elsasser_lowkz",
            "gandalf_perp_lowkz",
        ),
        help="Override the Alfvén forcing mode from the YAML config.",
    )
    parser.add_argument(
        "--alfven-amplitude",
        type=float,
        help="Override the Alfvén forcing amplitude from the YAML config.",
    )
    parser.add_argument(
        "--max-nz",
        type=int,
        help="Maximum |n_z| to force when using gandalf_perp_lowkz.",
    )
    parser.add_argument(
        "--include-nz0",
        action="store_true",
        help="Include n_z=0 in the low-|n_z| forcing mask.",
    )
    parser.add_argument(
        "--hermite-amplitude",
        type=float,
        help="Override Hermite forcing amplitude from the YAML config.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        help="Override resistivity eta from the YAML config.",
    )
    parser.add_argument(
        "--k-min",
        type=int,
        help="Override the minimum forced shell from the YAML config.",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        help="Override the maximum forced shell from the YAML config.",
    )
    parser.add_argument(
        "--skip-hermite-seed",
        action="store_true",
        help="Do not seed g with the tiny initial Hermite perturbation.",
    )
    parser.add_argument(
        "--hermite-seed-amplitude",
        type=float,
        help="Override Hermite seed amplitude from the YAML config.",
    )
    return parser.parse_args()


def run_one(
    nu: float,
    *,
    config_path: Path,
    n_steps: int,
    save_interval: int,
    m_override: int,
    forcing_mode: str | None,
    alfven_amplitude_override: float | None,
    max_nz: int | None,
    include_nz0: bool,
    hermite_amplitude_override: float | None,
    eta_override: float | None,
    k_min_override: int | None,
    k_max_override: int | None,
    skip_hermite_seed: bool,
    hermite_seed_amplitude_override: float | None,
) -> None:
    import jax

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")
    import jax.numpy as jnp
    import numpy as np

    from krmhd.config import SimulationConfig
    from krmhd.diagnostics import compute_energy, hermite_moment_energy
    from krmhd.physics import KRMHDState
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step
    from shared.alfven_forcing import apply_alfven_forcing, pop_alfven_forcing_options
    from shared.hermite_forcing import apply_hermite_forcing, pop_hermite_forcing_options
    from shared.hermite_seed import apply_hermite_seed, pop_hermite_seed_options

    cfg_dict = yaml.safe_load(config_path.read_text())
    alfven_forcing_options = pop_alfven_forcing_options(cfg_dict)
    hermite_cfg = cfg_dict.pop("hermite_forcing", {})
    hermite_forcing_options = pop_hermite_forcing_options(hermite_cfg)
    hermite_seed_options = pop_hermite_seed_options(cfg_dict)
    cfg_dict["physics"]["nu"] = nu
    if eta_override is not None:
        cfg_dict["physics"]["eta"] = eta_override
    if k_min_override is not None:
        cfg_dict["forcing"]["k_min"] = k_min_override
    if k_max_override is not None:
        cfg_dict["forcing"]["k_max"] = k_max_override
    cfg_dict["initial_condition"]["M"] = m_override
    cfg_dict["time_integration"]["n_steps"] = n_steps
    cfg_dict["time_integration"]["save_interval"] = save_interval
    if alfven_amplitude_override is not None:
        cfg_dict["forcing"]["amplitude"] = alfven_amplitude_override
    if forcing_mode is not None:
        alfven_forcing_options = alfven_forcing_options.__class__(
            mode=forcing_mode,
            max_nz=max_nz if max_nz is not None else alfven_forcing_options.max_nz,
            include_nz0=include_nz0,
        )
    elif max_nz is not None:
        alfven_forcing_options = alfven_forcing_options.__class__(
            mode=alfven_forcing_options.mode,
            max_nz=max_nz,
            include_nz0=include_nz0,
        )
    config = SimulationConfig(**cfg_dict)

    grid = config.create_grid()
    state = config.create_initial_state(grid)
    seed_amplitude = (
        hermite_seed_options.amplitude
        if hermite_seed_amplitude_override is None
        else hermite_seed_amplitude_override
    )
    seed_enabled = hermite_seed_options.enabled and not skip_hermite_seed
    hermite_seed_options = hermite_seed_options.__class__(
        enabled=seed_enabled,
        amplitude=seed_amplitude,
        seed=hermite_seed_options.seed,
    )
    state = apply_hermite_seed(state, options=hermite_seed_options)

    physics = config.physics
    forcing_cfg = config.forcing
    M = config.initial_condition.M
    hermite_amplitude = (
        hermite_cfg.get("amplitude", 0.0035)
        if hermite_amplitude_override is None
        else hermite_amplitude_override
    )
    hermite_moments = tuple(hermite_cfg.get("forced_moments", [0]))

    rng_key = jax.random.PRNGKey(42)

    def _eps_nu(E_m):
        if M <= 1 or nu == 0.0 or len(E_m) < 3:
            return 0.0
        m_idx = np.arange(len(E_m))
        rates = nu * (m_idx / M) ** physics.hyper_n
        rates[:2] = 0.0
        return float(2.0 * np.sum(rates * E_m))

    print(
        f"\n  nu={nu:.1e}  (M={M}, hyper_n={physics.hyper_n}, {n_steps} steps, "
        f"forcing={alfven_forcing_options.mode}, max_nz={alfven_forcing_options.max_nz}, "
        f"include_nz0={alfven_forcing_options.include_nz0}, "
        f"alfven_amp={forcing_cfg.amplitude}, k=[{forcing_cfg.k_min}, {forcing_cfg.k_max}], "
        f"eta={physics.eta}, "
        f"hermite_amp={hermite_amplitude}, seed_g={hermite_seed_options.enabled}, "
        f"seed_amp={hermite_seed_options.amplitude})"
    )

    for step in range(1, n_steps + 1):
        dt = compute_cfl_timestep(state, physics.v_A, config.time_integration.cfl_safety)
        if forcing_cfg.enabled:
            rng_key, subkey = jax.random.split(rng_key)
            state, _ = apply_alfven_forcing(
                state,
                forcing_cfg=forcing_cfg,
                dt=dt,
                key=subkey,
                options=alfven_forcing_options,
            )
        if hermite_amplitude > 0.0:
            rng_key, subkey = jax.random.split(rng_key)
            state, _ = apply_hermite_forcing(
                state, amplitude=hermite_amplitude,
                n_min=int(forcing_cfg.k_min), n_max=int(forcing_cfg.k_max),
                dt=dt, key=subkey, forced_moments=hermite_moments,
                options=hermite_forcing_options,
            )
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

        if step % save_interval == 0:
            E_m = np.array(hermite_moment_energy(state))
            eps = _eps_nu(E_m)
            energy = compute_energy(state)
            total_w = float(np.sum(E_m))
            tail_w = float(np.sum(E_m[-5:])) if len(E_m) >= 5 else total_w
            tail_frac = tail_w / total_w if total_w > 0 else float("nan")
            print(
                f"    step {step:5d}  t={state.time:.2f}  E_total={energy['total']:.4f}  "
                f"eps_nu={eps:.3e}  sumW={total_w:.3e}  tail5/sumW={tail_frac:.3e}"
            )

    # Final verdict
    E_m = np.array(hermite_moment_energy(state))
    eps_final = _eps_nu(E_m)
    total_w = float(np.sum(E_m))
    tail_w = float(np.sum(E_m[-5:])) if len(E_m) >= 5 else total_w
    tail_frac = tail_w / total_w if total_w > 0 else float("nan")
    print(
        f"  → FINAL eps_nu={eps_final:.3e}  sumW={total_w:.3e}  "
        f"tail5/sumW={tail_frac:.3e}  {'STABLE' if eps_final < 100 else 'BLOWUP'}"
    )


if __name__ == "__main__":
    args = parse_args()
    nu_values = args.nu_values if args.nu_values is not None else NU_VALUES
    for nu in nu_values:
        run_one(
            nu,
            config_path=args.config,
            n_steps=args.steps,
            save_interval=args.save_interval,
            m_override=args.m_override,
            forcing_mode=args.forcing_mode,
            alfven_amplitude_override=args.alfven_amplitude,
            max_nz=args.max_nz,
            include_nz0=args.include_nz0,
            hermite_amplitude_override=args.hermite_amplitude,
            eta_override=args.eta,
            k_min_override=args.k_min,
            k_max_override=args.k_max,
            skip_hermite_seed=args.skip_hermite_seed,
            hermite_seed_amplitude_override=args.hermite_seed_amplitude,
        )
