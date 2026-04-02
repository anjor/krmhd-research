#!/usr/bin/env python3
"""Download 128³ benchmark results from Modal volume and generate summary plots.

Usage:
    # Download all branches:
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py

    # Download a specific branch:
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py --only alfven128_eta4_f0p003

    # Just list what's on the volume:
    uv run python studies/02-collisionality-scan/scripts/download_128_results.py --list
"""
from __future__ import annotations

import argparse
import subprocess
import sys
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List volume contents only.")
    parser.add_argument("--only", nargs="+", default=None, help="Download only these branches.")
    args = parser.parse_args()

    if args.list:
        list_volume()
        return

    branches = args.only if args.only else BRANCHES
    for label in branches:
        download_branch(label)

    print("\nDone. Generate plots with:")
    print("  uv run python studies/02-collisionality-scan/analysis/plot_128_benchmark.py")


if __name__ == "__main__":
    main()
