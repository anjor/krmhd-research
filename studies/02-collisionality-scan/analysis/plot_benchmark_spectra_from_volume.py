#!/usr/bin/env python3
"""Plot E(k⊥) or E(n⊥) spectra from Modal volume benchmark data.

Downloads spectra from the Modal volume and generates publication-quality
plots. Supports plotting individual snapshots, time-averaged spectra,
and comparing multiple branches.

Usage:
    # Plot snapshots for a single branch (mode number):
    uv run python analysis/plot_benchmark_spectra_from_volume.py \\
        --branch alfven128_lowkz_f0p02_eta50 --mode-number

    # Compare latest spectra across branches:
    uv run python analysis/plot_benchmark_spectra_from_volume.py \\
        --branch alfven128_lowkz_f0p02_eta20 alfven128_lowkz_f0p02_eta50 \\
        --compare --mode-number

    # Average over a time window:
    uv run python analysis/plot_benchmark_spectra_from_volume.py \\
        --branch alfven128_lowkz_f0p02_eta50 --avg-start 1500 --avg-end 1800

    # Use already-downloaded local data (skip Modal download):
    uv run python analysis/plot_benchmark_spectra_from_volume.py \\
        --branch alfven128_lowkz_f0p02_eta50 --local
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = PROJECT_ROOT / "studies/02-collisionality-scan"
DATA_DIR = STUDY_DIR / "data/benchmark_128"
FIG_DIR = STUDY_DIR / "figures"
VOLUME_NAME = "krmhd-benchmark-vol"
L = 1.0  # box size


def download_spectra(branch: str) -> Path:
    """Download spectra from Modal volume, return local spectra dir."""
    local_dir = DATA_DIR / branch / "spectra"
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME,
         f"/{branch}/spectra/", str(local_dir)],
        check=True, capture_output=True,
    )
    # Fix nested dir from modal download
    nested = local_dir / "spectra"
    if nested.exists():
        for f in nested.glob("*.npz"):
            f.rename(local_dir / f.name)
        nested.rmdir()
    return local_dir


def load_spectra(
    spectra_dir: Path,
    t_min: float | None = None,
    t_max: float | None = None,
) -> list[dict]:
    """Load spectra files, optionally filtered by time range."""
    files = sorted(spectra_dir.glob("*.npz"))
    spectra = []
    for f in files:
        d = dict(np.load(f))
        t = float(d["time"])
        if t_min is not None and t < t_min:
            continue
        if t_max is not None and t > t_max:
            continue
        spectra.append(d)
    return spectra


def plot_snapshots(
    branch: str,
    spectra: list[dict],
    mode_number: bool = False,
    n_show: int = 8,
    output: Path | None = None,
) -> None:
    """Plot evenly-spaced spectral snapshots for a single branch."""
    if not spectra:
        print(f"  No spectra for {branch}")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    indices = np.linspace(0, len(spectra) - 1, min(n_show, len(spectra)), dtype=int)
    cmap = plt.cm.viridis

    for i, idx in enumerate(indices):
        s = spectra[idx]
        k = s["k_perp"]
        E = s["E_total"]
        t = float(s["time"])
        x = k * L / (2 * np.pi) if mode_number else k
        mask = E > 0
        color = cmap(i / max(len(indices) - 1, 1))
        ax.loglog(x[mask], E[mask], color=color, linewidth=1.2, label=f"t={t:.0f}")

    # Reference slopes from latest spectrum
    s_last = spectra[-1]
    k = s_last["k_perp"]
    E = s_last["E_total"]
    x_all = k * L / (2 * np.pi) if mode_number else k
    valid = E > 0
    x_v, E_v = x_all[valid], E[valid]
    if len(E_v) > 5:
        x_ref = np.array([2, 20]) if mode_number else np.array([8, 60])
        ax.loglog(x_ref, E_v[3] * (x_ref / x_v[3]) ** (-5.0 / 3.0),
                  "k--", lw=2.5, alpha=0.6, label="$-5/3$")
        ax.loglog(x_ref, E_v[3] * (x_ref / x_v[3]) ** (-3.0 / 2.0),
                  "k:", lw=2.5, alpha=0.6, label="$-3/2$")

    if mode_number:
        ax.axvspan(1, 2, alpha=0.1, color="blue")
        ax.axvline(42, color="red", ls=":", alpha=0.4)
        ax.set_xlabel("Mode number $n_\\perp$")
        ax.set_xlim(1, 60)
    else:
        ax.set_xlabel("$k_\\perp$")
        ax.set_xlim(5, 300)

    ax.set_ylabel("$E(k_\\perp)$")
    t_range = f"t={float(spectra[0]['time']):.0f}→{float(spectra[-1]['time']):.0f}"
    ax.set_title(f"{branch} — Snapshots ({t_range})")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if output is None:
        suffix = "_n" if mode_number else "_k"
        output = FIG_DIR / f"{branch}_snapshots{suffix}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_averaged(
    branch: str,
    spectra: list[dict],
    mode_number: bool = False,
    output: Path | None = None,
) -> None:
    """Plot time-averaged spectrum."""
    if not spectra:
        print(f"  No spectra for {branch}")
        return

    k = spectra[0]["k_perp"]
    E_avg = np.mean([s["E_total"] for s in spectra], axis=0)
    x = k * L / (2 * np.pi) if mode_number else k
    mask = E_avg > 0

    fig, ax = plt.subplots(figsize=(8, 6))
    t0 = float(spectra[0]["time"])
    t1 = float(spectra[-1]["time"])
    ax.loglog(x[mask], E_avg[mask], "k-", linewidth=2.5,
              label=f"Average (t={t0:.0f}→{t1:.0f}, {len(spectra)} snapshots)")

    x_v, E_v = x[mask], E_avg[mask]
    if len(E_v) > 5:
        x_ref = np.array([2, 20]) if mode_number else np.array([8, 60])
        ax.loglog(x_ref, E_v[3] * (x_ref / x_v[3]) ** (-5.0 / 3.0),
                  "k--", lw=2.5, alpha=0.6, label="$-5/3$")
        ax.loglog(x_ref, E_v[3] * (x_ref / x_v[3]) ** (-3.0 / 2.0),
                  "k:", lw=2.5, alpha=0.6, label="$-3/2$")

    if mode_number:
        ax.axvspan(1, 2, alpha=0.1, color="blue")
        ax.axvline(42, color="red", ls=":", alpha=0.4)
        ax.set_xlabel("Mode number $n_\\perp$")
        ax.set_xlim(1, 60)
    else:
        ax.set_xlabel("$k_\\perp$")
        ax.set_xlim(5, 300)

    ax.set_ylabel("$E(k_\\perp)$")
    ax.set_title(f"{branch} — Time-Averaged Spectrum")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if output is None:
        suffix = "_n" if mode_number else "_k"
        output = FIG_DIR / f"{branch}_averaged{suffix}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def plot_compare(
    branches: list[str],
    all_spectra: dict[str, list[dict]],
    mode_number: bool = False,
    output: Path | None = None,
) -> None:
    """Compare latest spectra across multiple branches."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

    for branch, color in zip(branches, colors):
        spectra = all_spectra.get(branch, [])
        if not spectra:
            continue
        s = spectra[-1]
        k = s["k_perp"]
        E = s["E_total"]
        t = float(s["time"])
        x = k * L / (2 * np.pi) if mode_number else k
        mask = E > 0
        ax.loglog(x[mask], E[mask], color=color, linewidth=2.5,
                  label=f"{branch} (t={t:.0f})")

    # Reference from last branch
    if spectra:
        x_all = k * L / (2 * np.pi) if mode_number else k
        valid = E > 0
        x_v, E_v = x_all[valid], E[valid]
        if len(E_v) > 5:
            x_ref = np.array([2, 20]) if mode_number else np.array([8, 60])
            ax.loglog(x_ref, E_v[3] * (x_ref / x_v[3]) ** (-5.0 / 3.0),
                      "k--", lw=2.5, alpha=0.6, label="$-5/3$")
            ax.loglog(x_ref, E_v[3] * (x_ref / x_v[3]) ** (-3.0 / 2.0),
                      "k:", lw=2.5, alpha=0.6, label="$-3/2$")

    if mode_number:
        ax.set_xlabel("Mode number $n_\\perp$")
        ax.set_xlim(1, 60)
    else:
        ax.set_xlabel("$k_\\perp$")
        ax.set_xlim(5, 300)

    ax.set_ylabel("$E(k_\\perp)$")
    ax.set_title("Branch Comparison — Latest Spectra")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    if output is None:
        suffix = "_n" if mode_number else "_k"
        output = FIG_DIR / f"branch_comparison{suffix}.png"
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--branch", nargs="+", required=True,
                        help="Branch name(s) on Modal volume")
    parser.add_argument("--mode-number", action="store_true",
                        help="Plot against mode number n instead of k_perp")
    parser.add_argument("--compare", action="store_true",
                        help="Compare latest spectra across branches")
    parser.add_argument("--avg-start", type=float, default=None,
                        help="Start time for averaging window")
    parser.add_argument("--avg-end", type=float, default=None,
                        help="End time for averaging window")
    parser.add_argument("--local", action="store_true",
                        help="Use local data (skip Modal download)")
    parser.add_argument("--n-show", type=int, default=8,
                        help="Number of snapshots to show (default: 8)")
    args = parser.parse_args()

    all_spectra: dict[str, list[dict]] = {}

    for branch in args.branch:
        if not args.local:
            print(f"Downloading {branch}...")
            download_spectra(branch)

        spectra_dir = DATA_DIR / branch / "spectra"
        spectra = load_spectra(spectra_dir, args.avg_start, args.avg_end)
        all_spectra[branch] = spectra
        print(f"  {branch}: {len(spectra)} spectra loaded")

    if args.compare:
        plot_compare(args.branch, all_spectra, mode_number=args.mode_number)
    elif args.avg_start is not None or args.avg_end is not None:
        for branch in args.branch:
            plot_averaged(branch, all_spectra[branch],
                          mode_number=args.mode_number)
    else:
        for branch in args.branch:
            plot_snapshots(branch, all_spectra[branch],
                           mode_number=args.mode_number, n_show=args.n_show)


if __name__ == "__main__":
    main()
