#!/usr/bin/env python3
"""Sequential overnight runner for Alfvén benchmark calibration branches."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = PROJECT_ROOT / "studies/02-collisionality-scan/configs/alfven_benchmark_matrix.yaml"
DEFAULT_LOG = PROJECT_ROOT / "studies/02-collisionality-scan/benchmark_output/overnight_loop_log.csv"
SCORE_SCRIPT = PROJECT_ROOT / "studies/02-collisionality-scan/analysis/score_benchmark_snapshots.py"


def load_score_run():
    spec = importlib.util.spec_from_file_location("score_benchmark_snapshots", SCORE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.score_run


def load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def build_command(python_bin: str, benchmark_script: str, common: dict, experiment: dict) -> list[str]:
    merged = {**common, **experiment}
    for internal_key in ("name", "description"):
        merged.pop(internal_key, None)

    cmd = [python_bin, benchmark_script]
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


def latest_snapshot_dir(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("snapshots_*"))
    return candidates[-1] if candidates else None


def best_score(snapshot_dir: Path) -> tuple[float, float]:
    score_run = load_score_run()
    scores = score_run(snapshot_dir, slope_n_min=3.5, slope_n_max=10.0, shoulder_start=12.0)
    if not scores:
        return float("-inf"), float("nan")
    best = max(scores, key=lambda item: item.score)
    return best.score, best.time


def append_log_row(log_path: Path, row: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Restrict the overnight loop to these experiment names.",
    )
    parser.add_argument(
        "--stop-score",
        type=float,
        default=-6.0,
        help="Stop once a branch achieves a score above this threshold.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    common = cfg.get("common", {})
    benchmark_script = cfg["benchmark_script"]
    python_bin = sys.executable
    experiments = cfg["experiments"]
    if args.only is not None:
        wanted = set(args.only)
        experiments = [exp for exp in experiments if exp["name"] in wanted]

    for exp in experiments:
        cmd = build_command(python_bin, benchmark_script, common, exp)
        print(f"\n=== Running {exp['name']} ===")
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)

        output_dir = PROJECT_ROOT / exp["output_dir"]
        snapshot_dir = latest_snapshot_dir(output_dir)
        score = float("-inf")
        best_time = float("nan")
        if snapshot_dir is not None:
            score, best_time = best_score(snapshot_dir)

        row = {
            "name": exp["name"],
            "output_dir": exp["output_dir"],
            "best_score": score,
            "best_time": best_time,
        }
        append_log_row(args.log, row)
        print(f"Best score for {exp['name']}: {score:.3f} at t={best_time:.1f}")

        if score >= args.stop_score:
            print(f"Stopping early: score {score:.3f} exceeded threshold {args.stop_score:.3f}")
            break


if __name__ == "__main__":
    main()
