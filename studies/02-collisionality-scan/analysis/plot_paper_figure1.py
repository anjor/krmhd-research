#!/usr/bin/env python3
"""Analysis: headline Figure 1 for the dissipative-anomaly paper.

Single panel: time-averaged collisional dissipation \\bar\\epsilon_\\nu as a
function of \\nu across the six IMEX runs (the plateau), plus legacy
Lawson-RK4 short-probe points for reference.

The time-averaged dissipation values are taken from Table 1 of the paper
(computed from diagnostics_timeseries.npz on the Modal volume; pulled in by
hand here to avoid an additional 6-file download just for one summary number
per run).

Usage:
    uv run python studies/02-collisionality-scan/analysis/plot_paper_figure1_with_pi.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = PROJECT_ROOT / "studies/02-collisionality-scan"
FIGURES_DIR = STUDY_DIR / "figures"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper/dissipative-anomaly/figures"

# JPP figure standards (matches analysis/dissipation_plateau.py).
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    }
)

# Run summary, ordered by nu. Numbers are taken from Table 1 of main.tex
# (and match docs/hermite_handoff.md). The time-series mean is the averaging
# window t - t_0 > 30 tau_A; the spec-window mean is t in [2100, 2200] tau_A.
RUNS = [
    {"nu": 1.0,  "eps_ts": 48.42, "eps_ts_std": 11.41, "eps_spec": 48.56},
    {"nu": 3.0,  "eps_ts": 49.21, "eps_ts_std": 10.65, "eps_spec": 48.15},
    {"nu": 5.0,  "eps_ts": 49.26, "eps_ts_std": 10.61, "eps_spec": 47.99},
    {"nu": 10.0, "eps_ts": 49.23, "eps_ts_std": 10.81, "eps_spec": 47.99},
    {"nu": 20.0, "eps_ts": 49.21, "eps_ts_std": 10.99, "eps_spec": 48.26},
    {"nu": 50.0, "eps_ts": 49.21, "eps_ts_std": 11.21, "eps_spec": 48.79},
]
PLATEAU_MEAN = 49.09

# Legacy Lawson-RK4 short-probe points (50 tau_A before numerical blowup;
# see Appendix A). Shown for reference only on the left panel.
LEGACY_LAWSON = [
    {"nu": 20.0, "eps": 60.0},
    {"nu": 50.0, "eps": 53.0},
    {"nu": 100.0, "eps": 46.0},
]


def main() -> None:
    # Single-panel headline figure: eps_nu vs nu plateau.
    fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.4))

    nus = np.array([r["nu"] for r in RUNS])
    eps_ts = np.array([r["eps_ts"] for r in RUNS])
    eps_ts_std = np.array([r["eps_ts_std"] for r in RUNS])
    eps_spec = np.array([r["eps_spec"] for r in RUNS])
    ax.errorbar(nus, eps_ts, yerr=eps_ts_std, fmt="o", color="#1f77b4",
                ms=5, capsize=3, lw=1.0,
                label=r"IMEX $200\,\tau_A$ (mean $\pm$ std)")
    ax.plot(nus, eps_spec, "D", color="#d62728", ms=4, mfc="none",
            label=r"IMEX spectrum window ($t \in [2100, 2200]$)")
    ax.axhline(PLATEAU_MEAN, ls="--", color="0.3", lw=1.0,
               label=rf"plateau $\bar\varepsilon_\nu = {PLATEAU_MEAN:.2f}$")
    legacy_nus = np.array([p["nu"] for p in LEGACY_LAWSON])
    legacy_eps = np.array([p["eps"] for p in LEGACY_LAWSON])
    ax.plot(legacy_nus, legacy_eps, "s", color="0.5", ms=5, mfc="0.85",
            label=r"Lawson-RK4 short probes ($50\,\tau_A$, pre-blowup)")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\bar\varepsilon_\nu$")
    ax.set_title(r"Dissipative anomaly at $M = 128$", fontsize=10)
    ax.set_ylim(20, 80)
    ax.legend(loc="lower left", frameon=False, fontsize=7)

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "hermite128_imex_eps_nu_plateau"
    fig.savefig(FIGURES_DIR / f"{stem}.pdf")
    fig.savefig(FIGURES_DIR / f"{stem}.png", dpi=200)
    fig.savefig(PAPER_FIGURES_DIR / f"{stem}.pdf")
    plt.close(fig)

    print(f"Wrote {FIGURES_DIR / (stem + '.pdf')}")
    print(f"Wrote {FIGURES_DIR / (stem + '.png')}")
    print(f"Wrote {PAPER_FIGURES_DIR / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
