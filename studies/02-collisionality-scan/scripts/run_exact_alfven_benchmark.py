#!/usr/bin/env python3
"""Wrapper for the exact GANDALF Alfvén benchmark with local compatibility fixes."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

import numpy as np


DEFAULT_BENCHMARK = Path("/tmp/alfvenic_cascade_benchmark.py")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--benchmark-script",
        default=str(DEFAULT_BENCHMARK),
        help="Path to the upstream exact benchmark script to execute.",
    )
    args, remaining = parser.parse_known_args()

    # NumPy 2 removed np.trapz; the exact benchmark still calls it during
    # final diagnostics. Alias it locally so completed evolutions do not
    # get marked as failures by the calibration loop.
    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid  # type: ignore[attr-defined]

    benchmark_path = Path(args.benchmark_script)
    sys.argv = [str(benchmark_path), *remaining]
    runpy.run_path(str(benchmark_path), run_name="__main__")


if __name__ == "__main__":
    main()
