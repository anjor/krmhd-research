"""Prepare sanitized benchmark data from raw GANDALF diagnostics.

Reads the raw Modal-volume downloads (diagnostics_timeseries.npz plus
spectra/spectrum_t*.npz per run), strips every field that directly encodes
an answer (the precomputed eps_nu scalar and time series), and writes:

    runs/<label>.npz          -- sanitized per-run data given to the agent
    grading/ground_truth.json -- answers computed from the same data

Ground truth is cross-checked against Table 1 of the dissipative-anomaly
paper draft; a mismatch beyond tolerance aborts data prep.

Usage:
    uv run python benchmark/dissipation-anomaly/prepare_data.py --raw-dir /tmp/krmhd_bench_raw
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent

# label -> (raw run directory name, nu, M)
RUNS: dict[str, tuple[str, float, int]] = {
    "nu1": ("hermite128_nu1_imex", 1.0, 128),
    "nu3": ("hermite128_nu3_imex", 3.0, 128),
    "nu5": ("hermite128_nu5_imex", 5.0, 128),
    "nu10": ("hermite128_nu10_imex", 10.0, 128),
    "nu20": ("hermite128_nu20_imex", 20.0, 128),
    "nu50": ("hermite128_nu50_imex", 50.0, 128),
    "M64_nu3": ("hermite_M64_nu3_imex", 3.0, 64),
    "M256_nu3": ("hermite_M256_nu3_imex", 3.0, 256),
}

NU_SCAN = ["nu1", "nu3", "nu5", "nu10", "nu20", "nu50"]
M_SCAN = ["M64_nu3", "nu3", "M256_nu3"]

HYPER_N = 6  # hyper-collisional exponent in nu * (m/M)^6
T0 = 2000.0  # tau_A at which g0 forcing was activated (resume point)
SLOPE_RANGE = (4, 40)  # inertial-range fit window in m

# Spectrum-window means from Table 1 of paper/dissipative-anomaly/main.tex.
PAPER_EPS_SPEC = {"nu1": 48.56, "nu3": 48.15, "nu5": 47.99, "nu10": 47.99,
                  "nu20": 48.26, "nu50": 48.79}
PAPER_SLOPES = {"nu1": -0.472, "nu3": -0.484, "nu5": -0.489, "nu10": -0.493,
                "nu20": -0.494, "nu50": -0.496}
# Time-series means for the M-convergence check (different averaging window
# than the spectrum snapshots, hence the looser tolerance below).
PAPER_EPS_M_SCAN = {"M64_nu3": 49.11, "nu3": 49.21, "M256_nu3": 50.28}

# Task 5 answer key (see tasks/task05_blowup_diagnosis/task.md).
TASK05_KEY = {
    "q1": "B",
    "q2": {"A": True, "B": False, "C": True, "D": True, "E": False},
    "q3": "C",
}


def eps_nu_from_spectrum(W_m: np.ndarray, nu: float, M: int) -> np.ndarray:
    """Collisional dissipation eps_nu = 2 nu sum_{m>=2} (m/M)^6 W(m) per snapshot.

    W_m has shape (n_snapshots, M+1); returns shape (n_snapshots,).
    """
    m = np.arange(M + 1, dtype=np.float64)
    weights = (m / M) ** HYPER_N
    weights[:2] = 0.0  # m = 0, 1 exempt from collisions
    return 2.0 * nu * W_m @ weights


def load_run(raw_dir: Path, label: str) -> dict:
    """Load raw diagnostics and spectra for one run into float64 arrays."""
    run_name, nu, M = RUNS[label]
    ts = np.load(raw_dir / run_name / "diagnostics_timeseries.npz")
    snap_paths = sorted(glob.glob(str(raw_dir / run_name / "spectra" / "spectrum_t*.npz")))
    if not snap_paths:
        raise FileNotFoundError(f"No spectra snapshots for {label} under {raw_dir}")

    spec_time, W_m, E_kperp = [], [], []
    k_perp = None
    for p in snap_paths:
        d = np.load(p)
        spec_time.append(float(d["time"]))
        W_m.append(d["E_m"].astype(np.float64))
        E_kperp.append(d["E_total"].astype(np.float64))
        k_perp = d["k_perp"].astype(np.float64)

    order = np.argsort(spec_time)
    return {
        "nu": nu,
        "M": M,
        "ts_time": ts["time"].astype(np.float64),
        "ts_E_total": ts["E_total"].astype(np.float64),
        "ts_hermite_energy": ts["hermite_energy"].astype(np.float64),
        "spec_time": np.array(spec_time)[order],
        "W_m": np.array(W_m)[order],
        "k_perp": k_perp,
        "E_kperp": np.array(E_kperp)[order],
    }


def compute_ground_truth(runs: dict[str, dict]) -> dict:
    """Compute all graded quantities from the sanitized arrays only."""
    gt: dict = {"task01": {}, "task02": {}, "task03": {}, "task04": {}, "task05": TASK05_KEY}

    eps_mean, hermite_mean = {}, {}
    slopes = {}
    totals, peak_m = {}, {}
    for label in NU_SCAN:
        r = runs[label]
        eps = eps_nu_from_spectrum(r["W_m"], r["nu"], r["M"])
        eps_mean[label] = float(np.mean(eps))

        in_window = r["ts_time"] - T0 > 30.0
        hermite_mean[label] = float(np.mean(r["ts_hermite_energy"][in_window]))

        W_avg = np.mean(r["W_m"], axis=0)
        m = np.arange(r["M"] + 1)
        sel = (m >= SLOPE_RANGE[0]) & (m <= SLOPE_RANGE[1])
        slopes[label] = float(np.polyfit(np.log(m[sel]), np.log(W_avg[sel]), 1)[0])

        weights = (m / r["M"]) ** HYPER_N
        weights[:2] = 0.0
        D_m = 2.0 * r["nu"] * weights * W_avg
        totals[label] = float(np.sum(D_m))
        peak_m[label] = int(np.argmax(D_m))

    vals = np.array(list(eps_mean.values()))
    gt["task01"] = {
        "eps_nu_mean": eps_mean,
        "hermite_energy_mean": hermite_mean,
        "is_plateau": bool((vals.max() - vals.min()) / vals.mean() < 0.05),
    }
    gt["task02"] = {"slope": slopes, "best_theory": "m^-1/2"}

    peaks = [peak_m[lbl] for lbl in NU_SCAN]
    tot = np.array([totals[lbl] for lbl in NU_SCAN])
    gt["task03"] = {
        "total_dissipation": totals,
        "peak_m": peak_m,
        "totals_match_within_5pct": bool((tot.max() - tot.min()) / tot.mean() < 0.05),
        "peak_m_trend": "decreases" if all(np.diff(peaks) <= 0) else "other",
    }

    eps_m = {}
    for label in M_SCAN:
        r = runs[label]
        eps_m[label] = float(np.mean(eps_nu_from_spectrum(r["W_m"], r["nu"], r["M"])))
    em = np.array(list(eps_m.values()))
    # Threshold matches the explicit 7.5% criterion stated in task04's task.md;
    # the spectrum-window spread across M is ~5.4% (the paper's 2.4% figure
    # uses the longer time-series window, which is stripped from this data).
    gt["task04"] = {
        "eps_nu_mean": eps_m,
        "truncation_independent": bool((em.max() - em.min()) / em.mean() < 0.075),
    }
    return gt


def cross_check_paper(gt: dict) -> None:
    """Abort if computed ground truth disagrees with the paper draft."""
    for label, paper in PAPER_EPS_SPEC.items():
        got = gt["task01"]["eps_nu_mean"][label]
        assert abs(got - paper) / paper < 0.01, f"eps_nu({label}): {got:.2f} vs paper {paper}"
    for label, paper in PAPER_SLOPES.items():
        got = gt["task02"]["slope"][label]
        assert abs(got - paper) < 0.01, f"slope({label}): {got:.3f} vs paper {paper}"
    for label, paper in PAPER_EPS_M_SCAN.items():
        got = gt["task04"]["eps_nu_mean"][label]
        assert abs(got - paper) / paper < 0.05, f"M-scan eps({label}): {got:.2f} vs paper {paper}"
    print("Cross-check against paper Table 1: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("/tmp/krmhd_bench_raw"))
    args = parser.parse_args()

    runs = {label: load_run(args.raw_dir, label) for label in RUNS}

    out_dir = BENCH_DIR / "runs"
    out_dir.mkdir(exist_ok=True)
    for label, r in runs.items():
        np.savez_compressed(
            out_dir / f"{label}.npz",
            nu=r["nu"], M=r["M"], hyper_n=HYPER_N, t0=T0,
            ts_time=r["ts_time"], ts_E_total=r["ts_E_total"],
            ts_hermite_energy=r["ts_hermite_energy"],
            spec_time=r["spec_time"], W_m=r["W_m"],
            k_perp=r["k_perp"], E_kperp=r["E_kperp"],
        )
        print(f"Wrote runs/{label}.npz  ({r['W_m'].shape[0]} snapshots, M={r['M']})")

    gt = compute_ground_truth(runs)
    cross_check_paper(gt)

    grading_dir = BENCH_DIR / "grading"
    grading_dir.mkdir(exist_ok=True)
    with open(grading_dir / "ground_truth.json", "w") as f:
        json.dump(gt, f, indent=2)
    print("Wrote grading/ground_truth.json")


if __name__ == "__main__":
    main()
