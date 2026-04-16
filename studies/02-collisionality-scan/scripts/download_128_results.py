#!/usr/bin/env python3
"""Download 128³ benchmark results from Modal volume and generate summary plots.

Usage:
    # Download all default branches (full):
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py

    # Download a specific branch:
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py --only alfven128_eta4_f0p003

    # Download only spectra/ and diagnostics_timeseries.npz (skip checkpoints):
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py --only hermite128_nu3_long --spectra-only

    # Download one specific checkpoint:
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py --only hermite128_nu3_long --checkpoint t2120.0

    # Just list what's on the volume:
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py --list
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOCAL_OUTPUT = PROJECT_ROOT / "studies/02-collisionality-scan/data/benchmark_128"

VOLUME_NAME = "krmhd-benchmark-vol"

BRANCHES = [
    "alfven128_lowkz_f0p001",
    "alfven128_lowkz_f0p002",
    "alfven128_lowkz_f0p005",
    "alfven128_lowkz_f0p01",
]

# Blown-up nonlinear Hermite runs — used by analysis/diagnose_hermite_blowup.py.
HERMITE_BLOWUP_BRANCHES = [
    "hermite128_nu3_long",
    "hermite128_nu5_long",
    "hermite128_nu10_long",
    "hermite128_nu1p0_v3",
]


def list_volume() -> None:
    subprocess.run(["modal", "volume", "ls", VOLUME_NAME, "/"], check=True)


def download_branch(label: str) -> None:
    local_dir = LOCAL_OUTPUT / label
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} → {local_dir}")
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, f"/{label}", str(local_dir)],
        check=True,
    )


def download_spectra_only(label: str) -> None:
    """Pull spectra/ subdir + diagnostics_timeseries.npz for a branch."""
    local_dir = LOCAL_OUTPUT / label
    (local_dir / "spectra").mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label}/spectra/ → {local_dir}/spectra/")
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME,
         f"/{label}/spectra/", str(local_dir / "spectra")],
        check=True,
    )
    # Also pull the single-file diagnostics timeseries if it exists.
    diag_path = f"/{label}/diagnostics_timeseries.npz"
    try:
        subprocess.run(
            ["modal", "volume", "get", VOLUME_NAME, diag_path, str(local_dir)],
            check=True,
        )
    except subprocess.CalledProcessError:
        print(f"  (no diagnostics_timeseries.npz for {label})")


def download_checkpoint(label: str, ckpt_tag: str) -> None:
    """Pull a single checkpoint file, e.g. ckpt_tag='t2120.0' → checkpoint_t2120.0.h5."""
    local_dir = LOCAL_OUTPUT / label / "checkpoints"
    local_dir.mkdir(parents=True, exist_ok=True)
    remote_path = f"/{label}/checkpoints/checkpoint_{ckpt_tag}.h5"
    print(f"Downloading {remote_path} → {local_dir}/")
    subprocess.run(
        ["modal", "volume", "get", VOLUME_NAME, remote_path, str(local_dir)],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List volume contents only.")
    parser.add_argument("--only", nargs="+", default=None, help="Download only these branches.")
    parser.add_argument(
        "--hermite-blowup", action="store_true",
        help="Download the four blown-up hermite128 runs (spectra-only by default).",
    )
    parser.add_argument(
        "--spectra-only", action="store_true",
        help="Download only spectra/ + diagnostics_timeseries.npz (skip checkpoints).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="Download a specific checkpoint tag (e.g. 't2120.0') for each --only branch.",
    )
    args = parser.parse_args()

    if args.list:
        list_volume()
        return

    if args.hermite_blowup:
        branches = HERMITE_BLOWUP_BRANCHES
        if args.checkpoint is None:
            args.spectra_only = True
    else:
        branches = args.only if args.only else BRANCHES

    for label in branches:
        if args.checkpoint:
            download_checkpoint(label, args.checkpoint)
        elif args.spectra_only:
            download_spectra_only(label)
        else:
            download_branch(label)

    print("\nDone.")


if __name__ == "__main__":
    main()
