#!/usr/bin/env python3
"""Plot benchmark spectra directly from saved checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from krmhd.diagnostics import (
    energy_spectrum_perpendicular_kinetic,
    energy_spectrum_perpendicular_magnetic,
)
from krmhd.io import load_checkpoint


def plot_single_checkpoint(checkpoint_path: Path, output_dir: Path) -> tuple[float, Path]:
    state, _, _ = load_checkpoint(str(checkpoint_path))
    k_perp, e_kin = energy_spectrum_perpendicular_kinetic(state)
    _, e_mag = energy_spectrum_perpendicular_magnetic(state)

    n_perp = np.asarray(k_perp) * float(state.grid.Lx) / (2 * np.pi)
    time_tau_a = float(state.time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.loglog(n_perp, np.asarray(e_kin), color="tab:blue", linewidth=2, label=f"t={time_tau_a:.1f}")
    if len(n_perp) > 1 and e_kin[1] > 0:
        k_ref = np.array([2.0, 10.0])
        e_ref = k_ref ** (-5 / 3) * float(e_kin[1]) / (float(n_perp[1]) ** (-5 / 3))
        ax1.loglog(k_ref, e_ref, "k--", linewidth=1.5, label="n^(-5/3)")
    ax1.axvspan(1, 2, alpha=0.2, color="green", label="Forcing modes 1-2")
    ax1.set_xlabel("Mode number n")
    ax1.set_ylabel("E_kin(n)")
    ax1.set_title("Kinetic Spectrum")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.loglog(n_perp, np.asarray(e_mag), color="tab:red", linewidth=2, label=f"t={time_tau_a:.1f}")
    if len(n_perp) > 1 and e_mag[1] > 0:
        k_ref = np.array([2.0, 10.0])
        e_ref = k_ref ** (-5 / 3) * float(e_mag[1]) / (float(n_perp[1]) ** (-5 / 3))
        ax2.loglog(k_ref, e_ref, "k--", linewidth=1.5, label="n^(-5/3)")
    ax2.axvspan(1, 2, alpha=0.2, color="green", label="Forcing modes 1-2")
    ax2.set_xlabel("Mode number n")
    ax2.set_ylabel("E_mag(n)")
    ax2.set_title("Magnetic Spectrum")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle(f"64^3 Benchmark Checkpoint Spectrum at t={time_tau_a:.1f} tau_A")
    fig.tight_layout()

    output_path = output_dir / f"checkpoint_spectrum_t{time_tau_a:06.1f}.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return time_tau_a, output_path


def plot_comparison(checkpoint_paths: list[Path], output_dir: Path) -> Path:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for checkpoint_path in checkpoint_paths:
        state, _, _ = load_checkpoint(str(checkpoint_path))
        k_perp, e_kin = energy_spectrum_perpendicular_kinetic(state)
        _, e_mag = energy_spectrum_perpendicular_magnetic(state)
        n_perp = np.asarray(k_perp) * float(state.grid.Lx) / (2 * np.pi)
        time_tau_a = float(state.time)

        ax1.loglog(n_perp, np.asarray(e_kin), linewidth=2, label=f"t={time_tau_a:.0f}")
        ax2.loglog(n_perp, np.asarray(e_mag), linewidth=2, label=f"t={time_tau_a:.0f}")

    for ax, ylabel, ref_values in (
        (ax1, "E_kin(n)", e_kin),
        (ax2, "E_mag(n)", e_mag),
    ):
        if len(n_perp) > 1 and ref_values[1] > 0:
            k_ref = np.array([2.0, 10.0])
            e_ref = k_ref ** (-5 / 3) * float(ref_values[1]) / (float(n_perp[1]) ** (-5 / 3))
            ax.loglog(k_ref, e_ref, "k--", linewidth=1.5, label="n^(-5/3)")
        ax.axvspan(1, 2, alpha=0.2, color="green")
        ax.set_xlabel("Mode number n")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    ax1.set_title("Kinetic Spectra by Checkpoint")
    ax2.set_title("Magnetic Spectra by Checkpoint")
    fig.suptitle("64^3 Benchmark Resume Checkpoint Spectra")
    fig.tight_layout()

    output_path = output_dir / "checkpoint_spectra_comparison.png"
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("studies/02-collisionality-scan/benchmark_output/alfven64_resume_t150/checkpoints"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("studies/02-collisionality-scan/figures/alfven64_resume_checkpoint_spectra"),
    )
    args = parser.parse_args()

    checkpoint_paths = sorted(args.input_dir.glob("checkpoint_t*.h5"))
    if not checkpoint_paths:
        raise SystemExit(f"No checkpoints found in {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_path in checkpoint_paths:
        plot_single_checkpoint(checkpoint_path, args.output_dir)

    plot_comparison(checkpoint_paths, args.output_dir)


if __name__ == "__main__":
    main()
