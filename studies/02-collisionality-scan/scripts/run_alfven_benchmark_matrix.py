#!/usr/bin/env python3
"""Launch exact Alfvén benchmark calibration experiments from a YAML manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "studies/02-collisionality-scan/configs/alfven_benchmark_matrix.yaml"


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def build_command(benchmark_script: str, common: dict, experiment: dict) -> list[str]:
    cmd = [sys.executable, benchmark_script]

    merged = {**common, **experiment}
    for internal_key in ("name", "description"):
        merged.pop(internal_key, None)

    for key, value in merged.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
            continue
        if value is None:
            continue
        cmd.extend([flag, str(value)])

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Experiment manifest YAML.")
    parser.add_argument(
        "--benchmark-script",
        type=str,
        default=None,
        help="Path to the exact benchmark script. Defaults to manifest benchmark_script.",
    )
    parser.add_argument("--list", action="store_true", help="List experiment names and exit.")
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run or print only the named experiments.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    benchmark_script = args.benchmark_script or cfg["benchmark_script"]
    common = cfg.get("common", {})
    experiments = cfg["experiments"]

    if args.only is not None:
        wanted = set(args.only)
        experiments = [exp for exp in experiments if exp["name"] in wanted]

    if not experiments:
        raise SystemExit("No experiments selected.")

    if args.list:
        for exp in experiments:
            print(f"{exp['name']}: {exp.get('description', '')}")
        return

    for exp in experiments:
        cmd = build_command(benchmark_script, common, exp)
        print(f"\n=== {exp['name']} ===")
        if exp.get("description"):
            print(exp["description"])
        print("Command:")
        print(" ".join(cmd))
        if args.dry_run:
            continue
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


if __name__ == "__main__":
    main()
