#!/usr/bin/env python3
"""Task-1 forcing gate for Study 3 (fluid-only, config-driven).

The run is deliberately M=0: the gate calibrates the RMHD advector that will
later carry the passive Hermite hierarchy.  It writes the prediction before
evolving and never changes it after inspecting the result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import jax
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from krmhd.config import SimulationConfig
from krmhd.timestepping import compute_cfl_timestep, gandalf_step

from shared.alfven_diagnostics import (
    elsasser_energies,
    elsasser_perpendicular_spectra,
    high_k_fraction,
    sigma_c,
)
from shared.alfven_forcing import apply_alfven_forcing, pop_alfven_forcing_options
from shared.run_utils import detect_hardware, generate_run_id, log_run


def _config_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Config not found: {value}")


def _write_prediction(output_dir: Path, gate: dict, target_sigma_c: float, run_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction.md").write_text(
        "# Pre-run prediction\n\n"
        f"Run ID: `{run_id}`\n\n"
        f"Target injection cross-helicity: `{target_sigma_c:.3f}`. "
        "Because z+ and z- are independently forced with squared amplitudes "
        "proportional to 1+sigma_c and 1-sigma_c, respectively, the steady "
        "volume-integrated RMHD sigma_c is predicted to agree with the target "
        f"within `{gate['sigma_tolerance']:.1%}` over the final "
        f"`{gate['averaging_outer_times']}` outer times. Both E+ and E- are "
        "predicted to fall away from the dealiasing cutoff, with the combined "
        f"top-20%-of-resolved-shell energy fraction below `{gate['max_high_k_fraction']:.1%}`.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="YAML config path relative to repository root")
    args = parser.parse_args()
    config_path = _config_path(args.config)
    raw_cfg = yaml.safe_load(config_path.read_text())
    gate = raw_cfg.pop("study3_forcing_gate")
    forcing_options = pop_alfven_forcing_options(raw_cfg)
    config = SimulationConfig(**raw_cfg)
    if config.initial_condition.M != 0:
        raise ValueError("Task 1 forcing gate requires initial_condition.M: 0")
    if gate["target_outer_times"] < 10.0:
        raise ValueError("Task 1 gate requires at least 10 outer times")

    run_id = generate_run_id("03", f"forcing_sig{forcing_options.target_sigma_c:g}")
    output_dir = PROJECT_ROOT / config.io.output_dir / run_id
    _write_prediction(output_dir, gate, forcing_options.target_sigma_c, run_id)
    shutil.copy2(config_path, output_dir / "config.yaml")

    grid = config.create_grid()
    state = config.create_initial_state(grid)
    physics = config.physics
    forcing_cfg = config.forcing
    outer_time = grid.Lz / physics.v_A
    target_time = gate["target_outer_times"] * outer_time
    sample_interval = gate["sample_interval_outer_times"] * outer_time
    next_sample = 0.0
    rng_key = jax.random.PRNGKey(forcing_cfg.seed)
    records: list[dict[str, float]] = []
    wall_start = time.time()
    step = 0

    print(f"Run ID: {run_id}")
    print(f"Task 1 fluid-only forcing gate; target t={target_time:.3f} ({gate['target_outer_times']} tau_outer)")
    print(f"Pre-run prediction: {output_dir / 'prediction.md'}")
    while float(state.time) < target_time:
        dt = compute_cfl_timestep(state, physics.v_A, config.time_integration.cfl_safety)
        state = gandalf_step(
            state, dt, physics.eta, physics.v_A, nu=0.0,
            hyper_r=physics.hyper_r, hyper_n=physics.hyper_n,
        )
        rng_key, force_key = jax.random.split(rng_key)
        state, _ = apply_alfven_forcing(state, forcing_cfg, dt, force_key, forcing_options)
        step += 1
        if float(state.time) >= next_sample:
            eplus, eminus = elsasser_energies(state)
            nperp, spec_plus, spec_minus = elsasser_perpendicular_spectra(state)
            cutoff = min(grid.Nx, grid.Ny) // 3
            record = {
                "time": float(state.time), "outer_times": float(state.time) / outer_time,
                "E_plus": eplus, "E_minus": eminus, "sigma_c": sigma_c(state),
                "high_k_fraction": high_k_fraction(nperp, spec_plus, spec_minus, cutoff),
            }
            records.append(record)
            print("step=%d t=%.3f sigma_c=%+.4f E+=%.4e E-=%.4e tail=%.3e" % (
                step, record["time"], record["sigma_c"], eplus, eminus, record["high_k_fraction"]
            ))
            next_sample += sample_interval

    nperp, spec_plus, spec_minus = elsasser_perpendicular_spectra(state)
    times = np.array([r["outer_times"] for r in records])
    steady = times >= gate["target_outer_times"] - gate["averaging_outer_times"]
    sigma_values = np.array([r["sigma_c"] for r in records])[steady]
    tail_values = np.array([r["high_k_fraction"] for r in records])[steady]
    sigma_mean = float(np.mean(sigma_values))
    sigma_std = float(np.std(sigma_values))
    sigma_error = abs(sigma_mean - forcing_options.target_sigma_c)
    tail_max = float(np.max(tail_values))
    passed = bool(
        sigma_error <= gate["sigma_tolerance"]
        and tail_max <= gate["max_high_k_fraction"]
        and len(sigma_values) >= 2
    )
    wall_time = time.time() - wall_start
    summary = {
        "run_id": run_id, "config": str(config_path.relative_to(PROJECT_ROOT)),
        "hardware": detect_hardware(), "wall_time_s": wall_time, "steps": step,
        "target_sigma_c": forcing_options.target_sigma_c, "mean_sigma_c": sigma_mean,
        "std_sigma_c": sigma_std, "absolute_sigma_error": sigma_error,
        "steady_window_outer_times": gate["averaging_outer_times"],
        "max_high_k_fraction": tail_max, "pass": passed, "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    np.savez(output_dir / "elsasser_spectra.npz", nperp=nperp, E_plus=spec_plus, E_minus=spec_minus)
    with (output_dir / "result.md").open("w") as result:
        result.write("# Task 1 forcing-gate result\n\n")
        result.write(f"Outcome: **{'PASS' if passed else 'FAIL'}**\n\n")
        result.write(f"Mean sigma_c over final {gate['averaging_outer_times']} outer times: {sigma_mean:+.4f} ± {sigma_std:.4f}; target {forcing_options.target_sigma_c:+.4f}.\n\n")
        result.write(f"Maximum high-k energy fraction: {tail_max:.3e}.\n")
    log_run(run_id, str(config_path.relative_to(PROJECT_ROOT)), detect_hardware(), wall_time,
            "pass" if passed else "fail",
            f"Task 3 forcing gate: target sigma_c={forcing_options.target_sigma_c:+.3f}, mean={sigma_mean:+.3f}, tail={tail_max:.2e}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
