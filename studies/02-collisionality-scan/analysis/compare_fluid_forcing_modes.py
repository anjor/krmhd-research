"""Run and compare fluid-only forcing modes on the production grid.

Generates a compact figure comparing:
- fluid energy history
- final perpendicular spectrum

for several Alfvén forcing families using true RMHD-only M=0 runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import numpy as np
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = STUDY_DIR / "figures"
BASE_CONFIG = STUDY_DIR / "configs" / "alfven_fluid_benchmark_smoke.yaml"
sys.path.insert(0, str(PROJECT_ROOT))

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

from krmhd.config import SimulationConfig
from krmhd.diagnostics import compute_energy, energy_spectrum_perpendicular
from krmhd.physics import KRMHDState
from krmhd.timestepping import compute_cfl_timestep, gandalf_step
from shared.alfven_forcing import apply_alfven_forcing, pop_alfven_forcing_options


RUNS = [
    {
        "label": "Gaussian shell",
        "mode": "gaussian_shell",
        "amplitude": 0.005,
        "eta": 2.0,
        "color": "#4c78a8",
    },
    {
        "label": "Balanced low-kz",
        "mode": "balanced_elsasser_lowkz",
        "amplitude": 0.005,
        "eta": 2.0,
        "color": "#0b6e4f",
    },
    {
        "label": "Phi-only low-kz",
        "mode": "gandalf_perp_lowkz",
        "amplitude": 0.005,
        "eta": 2.0,
        "color": "#c84c09",
    },
]

N_STEPS = 5000
SAVE_INTERVAL = 500


def run_case(case: dict) -> dict:
    cfg_dict = yaml.safe_load(BASE_CONFIG.read_text())
    alfven_forcing_options = pop_alfven_forcing_options(cfg_dict)
    cfg_dict.pop("hermite_forcing", None)
    cfg_dict.pop("hermite_seed", None)
    cfg_dict["physics"]["nu"] = 0.0
    cfg_dict["physics"]["eta"] = case["eta"]
    cfg_dict["initial_condition"]["M"] = 0
    cfg_dict["time_integration"]["n_steps"] = N_STEPS
    cfg_dict["time_integration"]["save_interval"] = SAVE_INTERVAL
    cfg_dict["forcing"]["amplitude"] = case["amplitude"]
    alfven_forcing_options = alfven_forcing_options.__class__(
        mode=case["mode"],
        max_nz=alfven_forcing_options.max_nz,
        include_nz0=alfven_forcing_options.include_nz0,
        correlation=alfven_forcing_options.correlation,
    )
    config = SimulationConfig(**cfg_dict)

    grid = config.create_grid()
    state = config.create_initial_state(grid)
    physics = config.physics
    forcing_cfg = config.forcing
    ti = config.time_integration
    rng_key = jax.random.PRNGKey(42)

    times = [float(state.time)]
    energies = [float(compute_energy(state)["total"])]

    for step in range(1, ti.n_steps + 1):
        dt = compute_cfl_timestep(state, physics.v_A, ti.cfl_safety)
        if forcing_cfg.enabled:
            rng_key, subkey = jax.random.split(rng_key)
            state, _ = apply_alfven_forcing(
                state,
                forcing_cfg=forcing_cfg,
                dt=dt,
                key=subkey,
                options=alfven_forcing_options,
            )
        state = gandalf_step(
            state,
            dt,
            physics.eta,
            physics.v_A,
            nu=physics.nu,
            hyper_r=physics.hyper_r,
            hyper_n=physics.hyper_n,
        )
        if not isinstance(state.time, float):
            state = KRMHDState(
                z_plus=state.z_plus,
                z_minus=state.z_minus,
                B_parallel=state.B_parallel,
                g=state.g,
                M=state.M,
                beta_i=state.beta_i,
                v_th=state.v_th,
                nu=state.nu,
                Lambda=state.Lambda,
                time=float(state.time),
                grid=state.grid,
            )

        if step % ti.save_interval == 0:
            times.append(float(state.time))
            energies.append(float(compute_energy(state)["total"]))

    k_perp, e_k = energy_spectrum_perpendicular(state)
    return {
        "label": case["label"],
        "color": case["color"],
        "times": np.array(times),
        "E_total": np.array(energies),
        "k_perp": np.array(k_perp),
        "E_kperp": np.array(e_k),
    }


def add_reference_slope(ax: plt.Axes, k: np.ndarray, e: np.ndarray) -> None:
    mask = (k > 0) & (e > 0)
    if np.sum(mask) < 2:
        return
    k_ref = k[mask][1]
    e_ref = e[mask][1]
    k_line = np.array([k_ref, k[mask][-1]])
    e_line = e_ref * (k_line / k_ref) ** (-5.0 / 3.0)
    ax.loglog(k_line, e_line, "--", color="0.4", lw=1.2, label=r"$k_\perp^{-5/3}$ ref")


def plot() -> Path:
    results = [run_case(case) for case in RUNS]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    ax = axes[0]
    for result in results:
        ax.plot(
            result["times"],
            result["E_total"],
            color=result["color"],
            lw=2,
            label=result["label"],
        )
    ax.set_title("Fluid Energy History")
    ax.set_xlabel("t")
    ax.set_ylabel("E_total")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    for result in results:
        k = result["k_perp"]
        e = result["E_kperp"]
        mask = (k > 0) & (e > 0)
        ax.loglog(k[mask], e[mask], "o-", color=result["color"], lw=1.6, ms=4, label=result["label"])
    add_reference_slope(ax, results[0]["k_perp"], results[0]["E_kperp"])
    ax.set_title("Final Perpendicular Spectrum")
    ax.set_xlabel(r"$k_\perp$")
    ax.set_ylabel(r"$E(k_\perp)$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("Fluid-only forcing comparison at 64^2 x 32, M=0 (benchmark forcing order)", fontsize=13)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / "fluid_forcing_mode_comparison.png"
    pdf_path = FIGURES_DIR / "fluid_forcing_mode_comparison.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    output = plot()
    print(output)
