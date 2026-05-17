#!/usr/bin/env python3
"""Analysis: Figure 3 for the dissipative-anomaly paper.

Documents the base state and the absence of a Hermite pileup in the nu=3 IMEX
run:

    Panel (a): time-averaged perpendicular energy spectrum E(k_perp) of the
               driven KRMHD state, showing a legitimate inertial-range cascade.
    Panel (b): Hermite moment spectrum W(m, t) as a heatmap over the
               statistically stationary window, showing that energy stays
               confined to low/intermediate m -- no pileup at the m = M
               truncation over the run.

Data source: the nu=3 IMEX run on the Modal volume `krmhd-benchmark-vol`,
branch `hermite128_nu3_imex`. Each `spectra/spectrum_t*.npz` snapshot carries
`k_perp`, `E_total` (= E(k_perp)), `E_m` (= W(m)), `eps_nu`, `time`, `step`.

Usage:
    # Download data from the Modal volume if absent, then plot:
    uv run python studies/02-collisionality-scan/analysis/plot_paper_figure3.py

    # Skip the Modal download and use already-fetched local data:
    uv run python studies/02-collisionality-scan/analysis/plot_paper_figure3.py --local

Idempotent: reads saved npz snapshots and regenerates the figure without
re-running any simulation (CLAUDE.md rule 5).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = PROJECT_ROOT / "studies/02-collisionality-scan"
FIGURES_DIR = STUDY_DIR / "figures"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper/dissipative-anomaly/figures"

VOLUME_NAME = "krmhd-benchmark-vol"
BRANCH = "hermite128_nu3_imex"
SPECTRA_DIR = STUDY_DIR / "data" / BRANCH / "spectra"

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

# Inertial-range fit window for E(k_perp), in k_perp units. The forcing band
# sits at k_perp < 12; the resistive cutoff sets in beyond k_perp ~ 42.
KPERP_FIT_MIN = 12.0
KPERP_FIT_MAX = 42.0


def download_spectra() -> None:
    """Download the nu=3 IMEX spectra from the Modal volume into the data dir.

    Mirrors the download pattern in plot_benchmark_spectra_from_volume.py:
    `modal volume get` drops files into a nested `spectra/` subdirectory.
    """
    SPECTRA_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", "--force", VOLUME_NAME,
         f"{BRANCH}/spectra", str(SPECTRA_DIR.parent)],
        check=True,
        capture_output=True,
    )


def load_snapshots() -> list[dict]:
    """Load all spectrum snapshots, sorted by simulation time."""
    paths = sorted(
        SPECTRA_DIR.glob("spectrum_t*.npz"),
        key=lambda p: float(p.stem.split("_t")[1].split("_step")[0]),
    )
    snaps = []
    for p in paths:
        d = np.load(p)
        snaps.append({k: d[k] for k in d.files})
    return snaps


def fit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope of log(y) vs log(x)."""
    coeffs = np.polyfit(np.log(x), np.log(y), 1)
    return float(coeffs[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use already-downloaded local data; skip the Modal download.",
    )
    args = parser.parse_args()

    if not args.local and not SPECTRA_DIR.exists():
        print(f"Downloading {BRANCH}/spectra from Modal volume {VOLUME_NAME} ...")
        download_spectra()

    if not SPECTRA_DIR.exists():
        sys.exit(
            f"No spectra found at {SPECTRA_DIR}. "
            "Run without --local to fetch from the Modal volume."
        )

    snaps = load_snapshots()
    if not snaps:
        sys.exit(f"No spectrum_t*.npz files in {SPECTRA_DIR}.")

    times = np.array([float(s["time"]) for s in snaps])
    print(f"Loaded {len(snaps)} snapshots, t = {times[0]:.1f} to {times[-1]:.1f} tau_A")

    # --- Panel (a): time-averaged E(k_perp) ---
    k_perp = snaps[0]["k_perp"]
    E_kperp = np.mean([s["E_total"] for s in snaps], axis=0)
    valid = (k_perp > 0) & (E_kperp > 0)
    kp, Ekp = k_perp[valid], E_kperp[valid]

    fit_mask = (kp >= KPERP_FIT_MIN) & (kp <= KPERP_FIT_MAX)
    slope = fit_slope(kp[fit_mask], Ekp[fit_mask])

    # --- Panel (b): W(m, t) heatmap ---
    W_mt = np.array([s["E_m"] for s in snaps]).T  # shape (M+1, n_times)
    M = W_mt.shape[0] - 1
    m_edges = np.arange(-0.5, M + 1.5)
    dt = np.median(np.diff(times))
    t_edges = np.concatenate([times - dt / 2, [times[-1] + dt / 2]])

    # --- Figure ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.0))

    # Panel (a) -- focus on the cascade; the spectrum drops to the dealiasing
    # noise floor (~1e-16) well beyond the resistive cutoff, which we crop out.
    ax_a.loglog(kp, Ekp, "o-", color="#1b9e77", ms=3, lw=1.0)
    ref_k = kp[fit_mask]
    ref_E = 1.6 * Ekp[fit_mask][0] * (ref_k / ref_k[0]) ** slope
    ax_a.loglog(ref_k, ref_E, "k--", lw=1.2,
                label=rf"$k_\perp^{{{slope:.2f}}}$")
    ax_a.axvspan(kp.min() / 1.3, KPERP_FIT_MIN, color="0.85", zorder=0)
    ax_a.text(0.05, 0.08, "forcing", transform=ax_a.transAxes,
              fontsize=7, color="0.4")
    cascade = Ekp[kp <= 90.0]
    ax_a.set_xlim(kp.min() / 1.3, 110.0)
    ax_a.set_ylim(cascade.min() / 3, cascade.max() * 3)
    ax_a.set_xlabel(r"$k_\perp$")
    ax_a.set_ylabel(r"$E(k_\perp)$")
    ax_a.set_title(r"(a) base-state perpendicular spectrum", fontsize=9)
    ax_a.legend(loc="upper right", frameon=False)

    # Panel (b)
    pcm = ax_b.pcolormesh(
        t_edges, m_edges, W_mt,
        cmap="viridis",
        norm=LogNorm(vmin=max(W_mt[W_mt > 0].min(), 1e-10), vmax=W_mt.max()),
        shading="flat",
    )
    ax_b.set_xlabel(r"$t\ [\tau_A]$")
    ax_b.set_ylabel(r"$m$")
    ax_b.set_ylim(0, M)
    ax_b.set_title(r"(b) Hermite spectrum $W(m,t)$", fontsize=9)
    cbar = fig.colorbar(pcm, ax=ax_b, pad=0.02)
    cbar.set_label(r"$W(m)$")

    fig.tight_layout()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stem = "hermite128_nu3_imex_basestate_wmt"
    pdf_path = FIGURES_DIR / f"{stem}.pdf"
    png_path = FIGURES_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=200)
    fig.savefig(PAPER_FIGURES_DIR / f"{stem}.pdf")
    plt.close(fig)

    print(f"E(k_perp) inertial-range slope (k_perp in "
          f"[{KPERP_FIT_MIN}, {KPERP_FIT_MAX}]): {slope:.3f}")
    print(f"W(m,t): peak m = {int(np.argmax(W_mt.mean(axis=1)))}, "
          f"max m carrying >1% of peak = "
          f"{int(np.max(np.where(W_mt.mean(axis=1) > 0.01 * W_mt.mean(axis=1).max())))}")
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    print(f"Wrote {PAPER_FIGURES_DIR / (stem + '.pdf')}")


if __name__ == "__main__":
    main()
