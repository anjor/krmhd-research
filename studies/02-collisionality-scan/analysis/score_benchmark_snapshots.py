#!/usr/bin/env python3
"""Score benchmark snapshot spectra for inertial-range quality and tail pileup."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


TARGET_SLOPE = -5.0 / 3.0


@dataclass
class SnapshotScore:
    time: float
    mag_slope: float
    kin_slope: float
    mag_tail_ratio: float
    kin_tail_ratio: float
    score: float


def read_spectrum(csv_path: Path) -> tuple[list[float], list[float]]:
    n_vals: list[float] = []
    e_vals: list[float] = []
    with csv_path.open() as fh:
        reader = csv.reader(fh)
        next(reader)
        next(reader)
        for row in reader:
            if not row:
                continue
            e_vals.append(float(row[1]))
            n_vals.append(float(row[2]))
    return n_vals, e_vals


def fit_slope(n_vals: list[float], e_vals: list[float], n_min: float, n_max: float) -> float:
    xs: list[float] = []
    ys: list[float] = []
    for n, e in zip(n_vals, e_vals):
        if n_min <= n <= n_max and e > 0:
            xs.append(math.log(n))
            ys.append(math.log(e))
    if len(xs) < 2:
        return float("nan")
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den == 0:
        return float("nan")
    return num / den


def tail_ratio(n_vals: list[float], e_vals: list[float], shoulder_start: float, ref_n: float = 10.0) -> float:
    ref_candidates = [(abs(n - ref_n), e) for n, e in zip(n_vals, e_vals) if e > 0]
    if not ref_candidates:
        return float("inf")
    _, ref_e = min(ref_candidates, key=lambda item: item[0])
    tail_values = [e for n, e in zip(n_vals, e_vals) if n >= shoulder_start and e > 0]
    if not tail_values or ref_e <= 0:
        return float("inf")
    return max(tail_values) / ref_e


def score_snapshot(kinetic_csv: Path, magnetic_csv: Path, slope_n_min: float, slope_n_max: float, shoulder_start: float) -> SnapshotScore:
    time = float(kinetic_csv.stem.split("_t")[-1])
    n_kin, e_kin = read_spectrum(kinetic_csv)
    n_mag, e_mag = read_spectrum(magnetic_csv)

    kin_slope = fit_slope(n_kin, e_kin, slope_n_min, slope_n_max)
    mag_slope = fit_slope(n_mag, e_mag, slope_n_min, slope_n_max)
    kin_tail = tail_ratio(n_kin, e_kin, shoulder_start)
    mag_tail = tail_ratio(n_mag, e_mag, shoulder_start)

    slope_penalty = abs(mag_slope - TARGET_SLOPE) + 0.5 * abs(kin_slope - TARGET_SLOPE)
    tail_penalty = max(0.0, math.log10(max(mag_tail, 1.0))) + 0.5 * max(0.0, math.log10(max(kin_tail, 1.0)))
    score = -(slope_penalty + tail_penalty)

    return SnapshotScore(
        time=time,
        mag_slope=mag_slope,
        kin_slope=kin_slope,
        mag_tail_ratio=mag_tail,
        kin_tail_ratio=kin_tail,
        score=score,
    )


def score_run(snapshot_dir: Path, slope_n_min: float, slope_n_max: float, shoulder_start: float) -> list[SnapshotScore]:
    kinetic_files = {path.stem.split("_t")[-1]: path for path in snapshot_dir.glob("kinetic_t*.csv")}
    magnetic_files = {path.stem.split("_t")[-1]: path for path in snapshot_dir.glob("magnetic_t*.csv")}
    times = sorted(set(kinetic_files) & set(magnetic_files), key=lambda value: float(value))
    return [
        score_snapshot(kinetic_files[t], magnetic_files[t], slope_n_min, slope_n_max, shoulder_start)
        for t in times
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot_dirs", nargs="+", type=Path)
    parser.add_argument("--slope-n-min", type=float, default=3.5)
    parser.add_argument("--slope-n-max", type=float, default=10.0)
    parser.add_argument("--shoulder-start", type=float, default=12.0)
    args = parser.parse_args()

    for snapshot_dir in args.snapshot_dirs:
        scores = score_run(snapshot_dir, args.slope_n_min, args.slope_n_max, args.shoulder_start)
        if not scores:
            print(f"{snapshot_dir}: no snapshot CSVs found")
            continue
        best = max(scores, key=lambda item: item.score)
        latest = scores[-1]
        print(f"\n{snapshot_dir}")
        print(
            f"  best t={best.time:.1f} score={best.score:.3f} "
            f"mag_slope={best.mag_slope:.3f} kin_slope={best.kin_slope:.3f} "
            f"mag_tail={best.mag_tail_ratio:.2f} kin_tail={best.kin_tail_ratio:.2f}"
        )
        print(
            f"  latest t={latest.time:.1f} score={latest.score:.3f} "
            f"mag_slope={latest.mag_slope:.3f} kin_slope={latest.kin_slope:.3f} "
            f"mag_tail={latest.mag_tail_ratio:.2f} kin_tail={latest.kin_tail_ratio:.2f}"
        )


if __name__ == "__main__":
    main()
