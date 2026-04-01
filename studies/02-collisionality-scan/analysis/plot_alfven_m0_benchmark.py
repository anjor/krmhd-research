"""Compare the pre-fix M=2 workaround and true M=0 RMHD benchmark runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = STUDY_DIR / "figures"

WORKAROUND_RUN = (
    STUDY_DIR / "data" / "alfven_benchmark" / "02_alfven_20260331_094512_diagnostics.npz"
)
M0_RUN = (
    STUDY_DIR / "data" / "alfven_benchmark" / "02_alfven_20260331_114340_diagnostics.npz"
)


def load_run(path: Path, label: str) -> dict[str, np.ndarray | str]:
    data = np.load(path)
    run = {key: data[key] for key in data.files}
    run["label"] = label
    return run


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
    runs = [
        load_run(WORKAROUND_RUN, "M=2 workaround"),
        load_run(M0_RUN, "True M=0"),
    ]
    colors = ["#c84c09", "#0b6e4f"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    ax = axes[0]
    for run, color in zip(runs, colors, strict=True):
        ax.plot(run["times"], run["E_total"], color=color, lw=2, label=run["label"])
    ax.set_title("Fluid Energy History")
    ax.set_xlabel("t")
    ax.set_ylabel("E_total")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    ax = axes[1]
    for run, color in zip(runs, colors, strict=True):
        k = np.asarray(run["k_perp"])
        e = np.asarray(run["E_kperp"])
        mask = (k > 0) & (e > 0)
        ax.loglog(k[mask], e[mask], "o-", color=color, lw=1.6, ms=4, label=run["label"])
    add_reference_slope(ax, np.asarray(runs[-1]["k_perp"]), np.asarray(runs[-1]["E_kperp"]))
    ax.set_title("Final Perpendicular Spectrum")
    ax.set_xlabel(r"$k_\perp$")
    ax.set_ylabel(r"$E(k_\perp)$")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)

    fig.suptitle("Fluid-only benchmark: M=2 workaround vs true M=0", fontsize=13)
    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES_DIR / "alfven_m0_benchmark.png"
    pdf_path = FIGURES_DIR / "alfven_m0_benchmark.pdf"
    fig.savefig(png_path, dpi=180)
    fig.savefig(pdf_path, dpi=300)
    plt.close(fig)
    return png_path


if __name__ == "__main__":
    output = plot()
    print(output)
