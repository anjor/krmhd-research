"""Analysis: dissipation rate vs collisionality for Study 02.

Produces:
    Figure 1 (headline): epsilon_nu vs nu (log-log), showing plateau.
    Figure 2: Hermite spectra W(m) at selected nu values.
    Figure 3: Dissipation spectrum D(m) at each nu.
    Figure 4: Energy balance check -- injection vs dissipation at each nu.

Usage:
    uv run python studies/02-collisionality-scan/analysis/dissipation_plateau.py [--data-dir PATH]

Reads all *_diagnostics.npz files from the data directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.dissipation import compute_collisional_dissipation, compute_resistive_dissipation

STUDY_DIR = Path(__file__).resolve().parents[1]
FIGURES_DIR = STUDY_DIR / "figures"

# JPP figure standards
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

# Qualitative palette for line plots
COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]


def load_all_runs(data_dir: Path) -> list[dict]:
    """Load diagnostics from all completed runs, sorted by nu (descending)."""
    runs = []
    for npz_path in sorted(data_dir.glob("*_diagnostics.npz")):
        d = np.load(npz_path)
        run = {k: d[k] for k in d.files}
        run["run_id"] = npz_path.stem.replace("_diagnostics", "")
        runs.append(run)

    # Sort by nu descending (highest first)
    runs.sort(key=lambda r: float(r["nu"]), reverse=True)
    return runs


def compute_steady_state_dissipation(run: dict, frac: float = 0.5) -> dict:
    """Compute time-averaged dissipation over final fraction of run.

    Returns dict with nu, epsilon_nu_mean, epsilon_nu_std, epsilon_eta.
    """
    eps_hist = run["epsilon_nu_history"]
    n = len(eps_hist)
    start = max(1, int(n * (1 - frac)))  # skip initial transient
    eps_ss = eps_hist[start:]

    # Detect blowup: find the last index before eps_nu exceeds 100x
    # the value at the 25% mark (baseline from early steady state).
    eps_full = run["epsilon_nu_history"]
    n_full = len(eps_full)
    quarter = max(1, n_full // 4)
    baseline = eps_full[quarter] if np.isfinite(eps_full[quarter]) and eps_full[quarter] > 0 else np.nan

    if np.isfinite(baseline):
        blowup_indices = np.where(
            (eps_full > 100 * baseline) & np.isfinite(eps_full)
        )[0]
        blowup_idx = blowup_indices[0] if len(blowup_indices) > 0 else n_full
    else:
        blowup_idx = n_full

    # Use stable window: second half of pre-blowup region
    stable_end = blowup_idx
    stable_start = max(1, stable_end // 2)
    eps_ss = eps_full[stable_start:stable_end]

    valid = np.isfinite(eps_ss) & (eps_ss > 0)
    if np.sum(valid) == 0:
        return {
            "nu": float(run["nu"]),
            "epsilon_nu_mean": np.nan,
            "epsilon_nu_std": np.nan,
            "epsilon_eta": np.nan,
            "run_id": run["run_id"],
        }

    eps_ss_valid = eps_ss[valid]

    # Resistive dissipation from final spectrum
    eps_eta = compute_resistive_dissipation(
        run["E_kperp"], run["k_perp"], float(run["eta"]), int(run["hyper_r"])
    )

    return {
        "nu": float(run["nu"]),
        "epsilon_nu_mean": float(np.mean(eps_ss_valid)),
        "epsilon_nu_std": float(np.std(eps_ss_valid)),
        "epsilon_eta": float(eps_eta),
        "run_id": run["run_id"],
    }


def plot_dissipation_plateau(results: list[dict]) -> None:
    """Figure 1: epsilon_nu vs nu (log-log). The headline result."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    nus = [r["nu"] for r in results if np.isfinite(r["epsilon_nu_mean"])]
    eps = [r["epsilon_nu_mean"] for r in results if np.isfinite(r["epsilon_nu_mean"])]
    errs = [r["epsilon_nu_std"] for r in results if np.isfinite(r["epsilon_nu_mean"])]

    if not nus:
        print("WARNING: No valid dissipation data to plot")
        return

    ax.errorbar(nus, eps, yerr=errs, fmt="o-", color="k", capsize=3, markersize=4)

    # Reference: horizontal line at mean of low-nu values (plateau)
    if len(eps) >= 3:
        plateau_val = np.mean(eps[-3:])
        ax.axhline(plateau_val, color="gray", ls="--", lw=0.8, alpha=0.7)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\varepsilon_\nu$")
    ax.set_title(r"Collisional dissipation rate vs.\ $\nu$")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "epsilon_vs_nu.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "epsilon_vs_nu.png", dpi=150)
    plt.close(fig)
    print("Saved: epsilon_vs_nu.pdf/png")


def plot_hermite_spectra(runs: list[dict], nu_select: list[float] | None = None) -> None:
    """Figure 2: W(m) at selected nu values."""
    if nu_select is None:
        # Pick ~5 representative values spread across the range
        all_nus = sorted(set(float(r["nu"]) for r in runs), reverse=True)
        indices = np.linspace(0, len(all_nus) - 1, min(5, len(all_nus)), dtype=int)
        nu_select = [all_nus[i] for i in indices]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for i, nu_target in enumerate(sorted(nu_select, reverse=True)):
        # Find closest run
        run = min(runs, key=lambda r: abs(float(r["nu"]) - nu_target))
        # Use time-averaged W(m) from second half
        W_hist = run["W_m_history"]
        n = len(W_hist)
        start = max(1, n // 2)
        valid_rows = [W_hist[j] for j in range(start, n) if np.all(np.isfinite(W_hist[j]))]
        if not valid_rows:
            continue
        W_mean = np.mean(valid_rows, axis=0)

        m = np.arange(len(W_mean))
        mask = (m >= 2) & (W_mean > 0)
        color = COLORS[i % len(COLORS)]
        ax.loglog(m[mask], W_mean[mask], "o-", color=color, markersize=3,
                  label=rf"$\nu = {float(run['nu']):.0e}$")

    # Reference slopes
    m_ref = np.arange(3, 25)
    ax.loglog(m_ref, 0.5 * m_ref ** (-0.5), "k--", lw=0.7, alpha=0.5, label=r"$m^{-1/2}$")
    ax.loglog(m_ref, 0.1 * m_ref ** (-1.5), "k:", lw=0.7, alpha=0.5, label=r"$m^{-3/2}$")

    ax.set_xlabel(r"Hermite moment $m$")
    ax.set_ylabel(r"$W(m)$")
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "hermite_spectra_nu_scan.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "hermite_spectra_nu_scan.png", dpi=150)
    plt.close(fig)
    print("Saved: hermite_spectra_nu_scan.pdf/png")


def plot_dissipation_spectrum(runs: list[dict], nu_select: list[float] | None = None) -> None:
    """Figure 3: D(m) = 2*nu*(m/M)^n * E_m at each nu."""
    if nu_select is None:
        all_nus = sorted(set(float(r["nu"]) for r in runs), reverse=True)
        indices = np.linspace(0, len(all_nus) - 1, min(5, len(all_nus)), dtype=int)
        nu_select = [all_nus[i] for i in indices]

    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    for i, nu_target in enumerate(sorted(nu_select, reverse=True)):
        run = min(runs, key=lambda r: abs(float(r["nu"]) - nu_target))
        nu = float(run["nu"])
        M = int(run["M"])
        hyper_n = int(run["hyper_n"])

        # Time-averaged W(m)
        W_hist = run["W_m_history"]
        n = len(W_hist)
        start = max(1, n // 2)
        valid_rows = [W_hist[j] for j in range(start, n) if np.all(np.isfinite(W_hist[j]))]
        if not valid_rows:
            continue
        W_mean = np.mean(valid_rows, axis=0)

        m = np.arange(len(W_mean))
        D_m = 2.0 * nu * (m / M) ** hyper_n * W_mean
        D_m[:2] = 0.0  # m=0,1 exempt

        mask = (m >= 2) & (D_m > 0)
        color = COLORS[i % len(COLORS)]
        ax.semilogy(m[mask], D_m[mask], "o-", color=color, markersize=3,
                    label=rf"$\nu = {nu:.0e}$")

    ax.set_xlabel(r"Hermite moment $m$")
    ax.set_ylabel(r"$D(m) = 2\nu (m/M)^n W(m)$")
    ax.legend(loc="best", frameon=False)

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "dissipation_spectrum.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "dissipation_spectrum.png", dpi=150)
    plt.close(fig)
    print("Saved: dissipation_spectrum.pdf/png")


def plot_energy_balance(results: list[dict], runs: list[dict]) -> None:
    """Figure 4: injection rate vs total dissipation at each nu."""
    fig, ax = plt.subplots(figsize=(3.4, 2.8))

    nus = []
    ratios = []
    for res, run in zip(results, runs):
        if not np.isfinite(res["epsilon_nu_mean"]):
            continue
        eps_total = res["epsilon_nu_mean"] + res["epsilon_eta"]
        total_inj = float(run["total_injection"])
        sim_time = float(run["times"][-1])
        if sim_time > 0 and total_inj > 0:
            mean_inj = total_inj / sim_time
            ratio = eps_total / mean_inj
            nus.append(res["nu"])
            ratios.append(ratio)

    if not nus:
        print("WARNING: No valid energy balance data to plot")
        return

    ax.semilogx(nus, ratios, "ko-", markersize=4)
    ax.axhline(1.0, color="gray", ls="-", lw=0.5)
    ax.axhline(0.95, color="gray", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(1.05, color="gray", ls="--", lw=0.5, alpha=0.5)

    ax.set_xlabel(r"$\nu$")
    ax.set_ylabel(r"$\varepsilon_{\rm total} / \varepsilon_{\rm inj}$")
    ax.set_title("Energy balance check")

    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / "energy_balance_check.pdf", dpi=300)
    fig.savefig(FIGURES_DIR / "energy_balance_check.png", dpi=150)
    plt.close(fig)
    print("Saved: energy_balance_check.pdf/png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Study 02 dissipation analysis")
    parser.add_argument("--data-dir", type=str, default=str(STUDY_DIR / "data"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print(f"Loading runs from: {data_dir}")

    runs = load_all_runs(data_dir)
    if not runs:
        print("ERROR: No diagnostic files found")
        sys.exit(1)

    print(f"Found {len(runs)} runs:")
    for r in runs:
        print(f"  nu={float(r['nu']):.1e}  run_id={r['run_id']}")

    # Compute steady-state dissipation for each run
    results = [compute_steady_state_dissipation(r) for r in runs]

    print("\nSteady-state dissipation summary:")
    print(f"{'nu':>10s}  {'eps_nu':>12s}  {'eps_eta':>12s}  {'run_id'}")
    for res in results:
        print(
            f"{res['nu']:10.1e}  {res['epsilon_nu_mean']:12.4e}  "
            f"{res['epsilon_eta']:12.4e}  {res['run_id']}"
        )

    # Generate all figures
    plot_dissipation_plateau(results)
    plot_hermite_spectra(runs)
    plot_dissipation_spectrum(runs)
    plot_energy_balance(results, runs)

    print("\nDone. Figures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
