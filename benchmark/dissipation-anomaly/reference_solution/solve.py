"""Reference solution for the dissipation-anomaly benchmark.

Solves tasks 01-04 from the sanitized runs/*.npz data alone (no access to
ground_truth.json) and writes answers/taskNN.json. Task 05 answers are the
author's answer key. Used to validate the grader; a benchmark run by an
agent under test must not be given this directory.

Usage:
    uv run python benchmark/dissipation-anomaly/reference_solution/solve.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BENCH_DIR / "runs"
ANSWERS_DIR = Path(__file__).resolve().parent / "answers"

NU_SCAN = ["nu1", "nu3", "nu5", "nu10", "nu20", "nu50"]
M_SCAN = ["M64_nu3", "nu3", "M256_nu3"]


def load(label: str) -> dict:
    d = np.load(RUNS_DIR / f"{label}.npz")
    return {k: d[k] for k in d.files}


def eps_nu_series(r: dict) -> np.ndarray:
    M = int(r["M"])
    nu = float(r["nu"])
    m = np.arange(M + 1, dtype=np.float64)
    w = (m / M) ** int(r["hyper_n"])
    w[:2] = 0.0
    return 2.0 * nu * r["W_m"] @ w


def main() -> None:
    ANSWERS_DIR.mkdir(exist_ok=True)
    runs = {label: load(label) for label in set(NU_SCAN + M_SCAN)}

    # Task 01
    eps_mean, herm_mean = {}, {}
    for label in NU_SCAN:
        r = runs[label]
        eps_mean[label] = float(np.mean(eps_nu_series(r)))
        sel = r["ts_time"] - float(r["t0"]) > 30.0
        herm_mean[label] = float(np.mean(r["ts_hermite_energy"][sel]))
    vals = np.array(list(eps_mean.values()))
    task01 = {
        "eps_nu_mean": eps_mean,
        "hermite_energy_mean": herm_mean,
        "is_plateau": bool((vals.max() - vals.min()) / vals.mean() < 0.05),
    }

    # Task 02
    slopes = {}
    for label in NU_SCAN:
        r = runs[label]
        W_avg = np.mean(r["W_m"], axis=0)
        m = np.arange(int(r["M"]) + 1)
        sel = (m >= 4) & (m <= 40)
        slopes[label] = float(np.polyfit(np.log(m[sel]), np.log(W_avg[sel]), 1)[0])
    task02 = {"slope": slopes, "best_theory": "m^-1/2"}

    # Task 03
    totals, peaks = {}, {}
    for label in NU_SCAN:
        r = runs[label]
        M = int(r["M"])
        m = np.arange(M + 1, dtype=np.float64)
        w = (m / M) ** int(r["hyper_n"])
        w[:2] = 0.0
        D_m = 2.0 * float(r["nu"]) * w * np.mean(r["W_m"], axis=0)
        totals[label] = float(np.sum(D_m))
        peaks[label] = int(np.argmax(D_m))
    tot = np.array([totals[lbl] for lbl in NU_SCAN])
    pk = [peaks[lbl] for lbl in NU_SCAN]
    task03 = {
        "total_dissipation": totals,
        "peak_m": peaks,
        "totals_match_within_5pct": bool((tot.max() - tot.min()) / tot.mean() < 0.05),
        "peak_m_trend": "decreases" if all(np.diff(pk) <= 0) else "other",
    }

    # Task 04
    eps_m = {label: float(np.mean(eps_nu_series(runs[label]))) for label in M_SCAN}
    em = np.array(list(eps_m.values()))
    task04 = {
        "eps_nu_mean": eps_m,
        "truncation_independent": bool((em.max() - em.min()) / em.mean() < 0.075),
    }

    # Task 05 — author's answer key (reasoning task; no computation involved)
    task05 = {
        "q1": "B",
        "q2": {"A": True, "B": False, "C": True, "D": True, "E": False},
        "q3": "C",
    }

    for name, ans in [("task01", task01), ("task02", task02), ("task03", task03),
                      ("task04", task04), ("task05", task05)]:
        with open(ANSWERS_DIR / f"{name}.json", "w") as f:
            json.dump(ans, f, indent=2)
        print(f"Wrote {ANSWERS_DIR.relative_to(BENCH_DIR)}/{name}.json")


if __name__ == "__main__":
    main()
