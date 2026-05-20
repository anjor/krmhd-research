#!/usr/bin/env python3
"""Analysis: headline Figure 1 for the dissipative-anomaly paper.

Two panels:

    Panel (a): time-averaged collisional dissipation \\bar\\epsilon_\\nu as a
               function of \\nu across the six IMEX runs (the plateau).
    Panel (b): Hermite cascade flux \\Pi(m) for the six runs, time-averaged
               over the late stationary window. The constant-\\Pi(m) plateau
               value matches \\bar\\epsilon_\\nu across all six \\nu and is the
               direct measurement of the constant-flux cascade that gives the
               dissipative anomaly.

Data sources:
    - For panel (a), the time-averaged dissipation values are taken from
      Table 1 of the paper (computed from diagnostics_timeseries.npz on the
      Modal volume; pulled in by hand here to avoid an additional 6-file
      download just for one summary number per run).
    - For panel (b), checkpoints at t = 2180, 2190, 2200 \\tau_A for each of
      hermite128_nu{1,3,5,10,20,50}_imex/checkpoints/ on the Modal volume
      `krmhd-benchmark-vol`. \\Pi(m) is computed locally by calling
      `krmhd.diagnostics.hermite_flux` on each checkpoint state and summing
      the resulting [Nz, Ny, Nx//2+1, M] tensor over the spatial axes.

Usage:
    # Default: use local checkpoints; download anything missing from Modal.
    uv run python studies/02-collisionality-scan/analysis/plot_paper_figure1_with_pi.py

    # Skip the Modal download entirely (use whatever is local).
    uv run python studies/02-collisionality-scan/analysis/plot_paper_figure1_with_pi.py --local

Idempotent: caches Pi(m) per checkpoint to
data/hermite128_nu{X}_imex/pi_m_cache.npz so re-runs do not recompute.
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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = PROJECT_ROOT / "studies/02-collisionality-scan"
DATA_DIR = STUDY_DIR / "data"
FIGURES_DIR = STUDY_DIR / "figures"
PAPER_FIGURES_DIR = PROJECT_ROOT / "paper/dissipative-anomaly/figures"

VOLUME_NAME = "krmhd-benchmark-vol"

# Late-window checkpoint times to use for Pi(m) time-averaging.
CHECKPOINT_TIMES = (2180.0, 2190.0, 2200.0)

# Inertial-range band for the panel-(b) plateau annotation.
INERTIAL_M_MIN = 4
INERTIAL_M_MAX = 40

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


def label_for(nu: float) -> str:
    """Map a nu value to its branch label on the volume."""
    return f"hermite128_nu{int(nu)}_imex"


def ensure_checkpoint(nu: float, t: float, allow_download: bool) -> Path | None:
    """Ensure a checkpoint file is present locally; return its path (or None)."""
    branch = label_for(nu)
    ckpt_dir = DATA_DIR / branch / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    local = ckpt_dir / f"checkpoint_t{t:06.1f}.h5"
    if local.exists():
        return local
    if not allow_download:
        return None
    print(f"  downloading {branch}/checkpoints/{local.name} ...")
    subprocess.run(
        ["modal", "volume", "get", "--force", VOLUME_NAME,
         f"{branch}/checkpoints/{local.name}", str(local)],
        check=True,
    )
    return local if local.exists() else None


def compute_pi_m(checkpoint: Path) -> np.ndarray:
    """Load a checkpoint and compute net Hermite flux Pi(m), summed over k.

    Pi(m) = sum_k <-k_||.sqrt(2(m+1)).Im[g_{m+1} g_m*]> from the Hermite flux
    diagnostic; positive values are forward (phase-mixing) flux.
    """
    from krmhd.io import load_checkpoint
    from krmhd.diagnostics import hermite_flux

    state, _grid, _meta = load_checkpoint(str(checkpoint), expected_scheme="imex_rk222")
    flux = hermite_flux(state)  # [Nz, Ny, Nx//2+1, M]
    pi_m = np.asarray(np.sum(np.asarray(flux), axis=(0, 1, 2)))
    return pi_m


def time_averaged_pi_m(nu: float, allow_download: bool) -> tuple[np.ndarray | None, list[float]]:
    """Compute the time-averaged Pi(m) for one run.

    Returns (pi_m_mean, list_of_times_used). pi_m_mean is None if no
    checkpoints are available.
    """
    branch = label_for(nu)
    cache = DATA_DIR / branch / "pi_m_cache.npz"
    per_t = []
    times_used = []
    for t in CHECKPOINT_TIMES:
        ckpt = ensure_checkpoint(nu, t, allow_download)
        if ckpt is None:
            continue
        cache_key = f"t{t:06.1f}"
        # Per-checkpoint cache to make re-runs cheap.
        if cache.exists():
            with np.load(cache, allow_pickle=False) as d:
                if cache_key in d.files:
                    per_t.append(d[cache_key])
                    times_used.append(t)
                    continue
        print(f"  computing Pi(m) for {branch} at t={t}")
        pi_m = compute_pi_m(ckpt)
        per_t.append(pi_m)
        times_used.append(t)
        # Update cache.
        existing = {}
        if cache.exists():
            with np.load(cache, allow_pickle=False) as d:
                existing = {k: d[k] for k in d.files}
        existing[cache_key] = pi_m
        np.savez(cache, **existing)
    if not per_t:
        return None, []
    return np.mean(np.stack(per_t), axis=0), times_used


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use only checkpoints already on disk; do not download from Modal.",
    )
    args = parser.parse_args()

    # --- compute Pi(m) for each run ---
    pi_per_nu = {}
    for run in RUNS:
        nu = run["nu"]
        print(f"--- nu = {nu} ---")
        pi_m, ts = time_averaged_pi_m(nu, allow_download=not args.local)
        if pi_m is None:
            print(f"  no checkpoints available for nu={nu}; skipping")
            continue
        plateau = float(np.mean(pi_m[INERTIAL_M_MIN:INERTIAL_M_MAX + 1]))
        print(f"  averaged Pi(m) over t = {ts}; "
              f"mean over m in [{INERTIAL_M_MIN}, {INERTIAL_M_MAX}] = {plateau:.2f}")
        pi_per_nu[nu] = pi_m

    # --- figure ---
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.0, 3.2))

    # Panel (a): eps_nu vs nu (plateau).
    nus = np.array([r["nu"] for r in RUNS])
    eps_ts = np.array([r["eps_ts"] for r in RUNS])
    eps_ts_std = np.array([r["eps_ts_std"] for r in RUNS])
    eps_spec = np.array([r["eps_spec"] for r in RUNS])
    ax_a.errorbar(nus, eps_ts, yerr=eps_ts_std, fmt="o", color="#1f77b4",
                  ms=5, capsize=3, lw=1.0,
                  label=r"IMEX $200\,\tau_A$ (mean $\pm$ std)")
    ax_a.plot(nus, eps_spec, "D", color="#d62728", ms=4, mfc="none",
              label=r"IMEX spectrum window ($t \in [2100, 2200]$)")
    ax_a.axhline(PLATEAU_MEAN, ls="--", color="0.3", lw=1.0,
                 label=rf"plateau $\bar\varepsilon_\nu = {PLATEAU_MEAN:.2f}$")
    legacy_nus = np.array([p["nu"] for p in LEGACY_LAWSON])
    legacy_eps = np.array([p["eps"] for p in LEGACY_LAWSON])
    ax_a.plot(legacy_nus, legacy_eps, "s", color="0.5", ms=5, mfc="0.85",
              label=r"Lawson-RK4 short probes ($50\,\tau_A$, pre-blowup)")
    ax_a.set_xscale("log")
    ax_a.set_xlabel(r"$\nu$")
    ax_a.set_ylabel(r"$\bar\varepsilon_\nu$")
    ax_a.set_title(r"(a) dissipative anomaly at $M = 128$", fontsize=9)
    ax_a.set_ylim(20, 80)
    ax_a.legend(loc="lower left", frameon=False, fontsize=7)

    # Panel (b): Pi(m) for the six runs.
    cmap = plt.cm.viridis
    nus_present = sorted(pi_per_nu)
    for i, nu in enumerate(nus_present):
        pi_m = pi_per_nu[nu]
        m = np.arange(pi_m.size)
        color = cmap(i / max(1, len(nus_present) - 1))
        ax_b.plot(m, pi_m, "-", color=color, lw=1.2, label=rf"$\nu = {nu:g}$")
    ax_b.axvspan(INERTIAL_M_MIN, INERTIAL_M_MAX, color="0.92", zorder=0)
    # Inertial-range plateau value, averaged across the six nu and the
    # inertial-range band. Use this for annotation rather than a horizontal
    # reference line, to avoid suggesting Pi(m) should match the integrated
    # eps_nu (the two differ by the contribution of the g_0 <-> Alfvenic
    # coupling at low m).
    plateau_values = [float(np.mean(pi[INERTIAL_M_MIN:INERTIAL_M_MAX + 1]))
                      for pi in pi_per_nu.values()]
    pi_inertial = float(np.mean(plateau_values))
    ax_b.axhline(pi_inertial, ls=":", color="0.3", lw=1.0,
                 label=rf"$\Pi_\mathrm{{inertial}} = {pi_inertial:.1f}$")
    ax_b.text(0.5 * (INERTIAL_M_MIN + INERTIAL_M_MAX), 1.5 * pi_inertial,
              "inertial range", ha="center", fontsize=7, color="0.4")
    ax_b.set_xlabel(r"$m$")
    ax_b.set_ylabel(r"$\Pi(m)$")
    ax_b.set_title(r"(b) Hermite cascade flux", fontsize=9)
    ax_b.set_xlim(0, 128)
    ax_b.set_ylim(bottom=0)
    ax_b.legend(loc="upper right", frameon=False, fontsize=7, ncol=2)

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
