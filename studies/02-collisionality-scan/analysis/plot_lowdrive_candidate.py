"""Compare the low-drive candidate branch across dev and production grids.

Produces a compact figure with:
- fluid energy history
- Hermite dissipation history
- truncation-tail fraction history
- final perpendicular energy spectra
- final Hermite spectra
- final Hermite dissipation spectra

Usage:
    uv run python studies/02-collisionality-scan/analysis/plot_lowdrive_candidate.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

STUDY_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = STUDY_DIR / "figures"

DEV_RUN = STUDY_DIR / "data_lowdrive" / "02_nu1e-3_20260330_161023_diagnostics.npz"
PROD_RUN = STUDY_DIR / "data_lowdrive_prod" / "02_nu1e-3_20260330_163006_diagnostics.npz"


def load_run(path: Path, label: str) -> dict[str, np.ndarray | str]:
    data = np.load(path)
    run = {key: data[key] for key in data.files}
    run["label"] = label
    return run


def tail_fraction_history(run: dict[str, np.ndarray | str]) -> np.ndarray:
    w_hist = np.asarray(run["W_m_history"])
    total = np.sum(w_hist, axis=1)
    tail = np.sum(w_hist[:, -5:], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, tail / total, np.nan)
    return frac


def dissipation_spectrum(run: dict[str, np.ndarray | str]) -> tuple[np.ndarray, np.ndarray]:
    w_m = np.asarray(run["W_m"])
    nu = float(run["nu"])
    m_max = int(run["M"])
    hyper_n = int(run["hyper_n"])
    m = np.arange(len(w_m))
    d_m = 2.0 * nu * (m / m_max) ** hyper_n * w_m
    d_m[:2] = 0.0
    return m, d_m


def plot() -> Path:
    dev = load_run(DEV_RUN, "Dev grid 32^3")
    prod = load_run(PROD_RUN, "Prod grid 64^2 x 32")
    runs = [dev, prod]
    colors = ["#0b6e4f", "#c84c09"]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    ax = axes[0, 0]
    for run, color in zip(runs, colors, strict=True):
        ax.plot(run["times"], run["E_total"], color=color, lw=2, label=run["label"])
    ax.set_title("Fluid Energy History")
    ax.set_xlabel("t")
    ax.set_ylabel("E_total")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    for run, color in zip(runs, colors, strict=True):
        ax.plot(run["save_times"], run["epsilon_nu_history"], color=color, lw=2, label=run["label"])
    ax.set_title("Hermite Dissipation History")
    ax.set_xlabel("t")
    ax.set_ylabel("eps_nu")
    ax.set_yscale("log")

    ax = axes[0, 2]
    for run, color in zip(runs, colors, strict=True):
        ax.plot(run["save_times"], tail_fraction_history(run), color=color, lw=2, label=run["label"])
    ax.set_title("High-m Tail Fraction")
    ax.set_xlabel("t")
    ax.set_ylabel("sum(W[M-4:M])/sum(W)")
    ax.set_yscale("log")

    ax = axes[1, 0]
    for run, color in zip(runs, colors, strict=True):
        k = np.asarray(run["k_perp"])
        e = np.asarray(run["E_kperp"])
        mask = (k > 0) & (e > 0)
        ax.loglog(k[mask], e[mask], "o-", color=color, lw=1.5, ms=4, label=run["label"])
    ax.set_title("Final Perpendicular Spectrum")
    ax.set_xlabel("k_perp")
    ax.set_ylabel("E(k_perp)")

    ax = axes[1, 1]
    for run, color in zip(runs, colors, strict=True):
        w_m = np.asarray(run["W_m"])
        m = np.arange(len(w_m))
        mask = (m >= 2) & (w_m > 0)
        ax.loglog(m[mask], w_m[mask], "o-", color=color, lw=1.5, ms=3, label=run["label"])
    ax.set_title("Final Hermite Spectrum")
    ax.set_xlabel("m")
    ax.set_ylabel("W(m)")

    ax = axes[1, 2]
    for run, color in zip(runs, colors, strict=True):
        m, d_m = dissipation_spectrum(run)
        mask = (m >= 2) & (d_m > 0)
        ax.semilogy(m[mask], d_m[mask], "o-", color=color, lw=1.5, ms=3, label=run["label"])
    ax.set_title("Final Hermite Dissipation Spectrum")
    ax.set_xlabel("m")
    ax.set_ylabel("D(m)")

    for ax in axes.flat:
        ax.grid(alpha=0.2)

    fig.suptitle("Low-drive candidate branch: dev vs production grid", fontsize=14)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / "lowdrive_candidate_comparison.png"
    pdf_path = FIGURES_DIR / "lowdrive_candidate_comparison.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    output = plot()
    print(output)
