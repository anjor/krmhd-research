"""Sweep all nu configs for Study 02 sequentially.

Usage:
    uv run python studies/02-collisionality-scan/scripts/sweep_nu.py [--dev] [--configs config1.yaml config2.yaml ...]

Options:
    --dev       Run only the dev config for smoke testing.
    --configs   Run specific configs (filenames relative to configs/).
                If omitted, runs all production configs in order of decreasing nu.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = STUDY_DIR / "scripts"
CONFIGS_DIR = STUDY_DIR / "configs"

# Ordered from highest nu (fastest) to lowest (slowest)
PRODUCTION_CONFIGS = [
    "nu1e-1.yaml",
    "nu3e-2.yaml",
    "nu1e-2.yaml",
    "nu3e-3.yaml",
    "nu1e-3.yaml",
    "nu3e-4.yaml",
    "nu1e-4.yaml",
    "nu3e-5.yaml",
    "nu1e-5.yaml",
]

DEV_CONFIG = "nu1e-3_dev.yaml"


def run_config(config_name: str) -> tuple[str, float, bool]:
    """Run a single config via subprocess.

    Returns (config_name, wall_time, success).
    """
    config_path = CONFIGS_DIR / config_name
    if not config_path.exists():
        print(f"  ERROR: Config not found: {config_path}")
        return config_name, 0.0, False

    cmd = [
        sys.executable, str(SCRIPTS_DIR / "run_local.py"),
        f"configs/{config_name}",
    ]

    print(f"\n{'='*60}")
    print(f"Running: {config_name}")
    print(f"{'='*60}")

    start = time.time()
    result = subprocess.run(cmd, cwd=str(STUDY_DIR))
    wall_time = time.time() - start

    success = result.returncode == 0
    status = "OK" if success else f"FAILED (exit {result.returncode})"
    print(f"\n  {config_name}: {status} ({wall_time:.1f}s)")

    return config_name, wall_time, success


def main() -> None:
    args = sys.argv[1:]

    if "--dev" in args:
        configs = [DEV_CONFIG]
    elif "--configs" in args:
        idx = args.index("--configs")
        configs = args[idx + 1:]
        if not configs:
            print("Error: --configs requires at least one config filename")
            sys.exit(1)
    else:
        configs = PRODUCTION_CONFIGS

    print(f"Study 02 Collisionality Sweep")
    print(f"Configs to run: {len(configs)}")
    for c in configs:
        print(f"  - {c}")

    results: list[tuple[str, float, bool]] = []
    sweep_start = time.time()

    for config_name in configs:
        result = run_config(config_name)
        results.append(result)

    sweep_time = time.time() - sweep_start

    # Summary table
    print(f"\n{'='*60}")
    print(f"SWEEP SUMMARY")
    print(f"{'='*60}")
    print(f"{'Config':<20} {'Wall Time':>12} {'Status':>10}")
    print(f"{'-'*20} {'-'*12} {'-'*10}")

    n_pass = 0
    for name, wall_time, success in results:
        status = "PASS" if success else "FAIL"
        if success:
            n_pass += 1
        print(f"{name:<20} {wall_time:>10.1f}s {status:>10}")

    print(f"\n{n_pass}/{len(results)} configs completed successfully")
    print(f"Total sweep time: {sweep_time:.1f}s")

    if n_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
