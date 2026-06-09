"""Deterministic grader for the dissipation-anomaly LLM benchmark.

Scores an answers directory (answers/task01.json ... task05.json) against
grading/ground_truth.json. Pure stdlib; no numpy required.

Usage:
    python grading/grade.py path/to/answers [--threshold 0.9]

Exit code 0 if the total score fraction is >= threshold, 1 otherwise.

Scoring (45 points total):
    task01: 6 x eps_nu_mean (2% rel), 6 x hermite_energy_mean (2% rel),
            1 x is_plateau                                          = 13
    task02: 6 x slope (+/- 0.05 abs), 1 x best_theory               =  7
    task03: 6 x total_dissipation (2% rel), 6 x peak_m (+/- 3),
            1 x totals_match_within_5pct, 1 x peak_m_trend          = 14
    task04: 3 x eps_nu_mean (2% rel), 1 x truncation_independent    =  4
    task05: 1 x q1, 5 x q2 items, 1 x q3                            =  7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GRADING_DIR = Path(__file__).resolve().parent

REL_TOL = 0.02      # relative tolerance on dissipation rates and energies
SLOPE_TOL = 0.05    # absolute tolerance on fitted spectral exponents
PEAK_M_TOL = 3      # absolute tolerance on integrand peak location


def load_answers(answers_dir: Path, name: str) -> dict | None:
    path = answers_dir / f"{name}.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {name}: unreadable ({e})")
        return None


def check_rel(ans: dict | None, truth: dict, key: str, label: str, tol: float) -> tuple[int, int, list[str]]:
    """Score one dict-of-floats field at relative tolerance tol."""
    earned, possible, notes = 0, len(truth[key]), []
    for run, true_val in truth[key].items():
        try:
            got = float(ans[key][run])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            notes.append(f"{label}[{run}]: missing")
            continue
        if abs(got - true_val) <= tol * abs(true_val):
            earned += 1
        else:
            notes.append(f"{label}[{run}]: {got:.4g} (expected {true_val:.4g} +/- {tol:.0%})")
    return earned, possible, notes


def check_abs(ans: dict | None, truth: dict, key: str, label: str, tol: float) -> tuple[int, int, list[str]]:
    """Score one dict-of-numbers field at absolute tolerance tol."""
    earned, possible, notes = 0, len(truth[key]), []
    for run, true_val in truth[key].items():
        try:
            got = float(ans[key][run])  # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            notes.append(f"{label}[{run}]: missing")
            continue
        if abs(got - true_val) <= tol:
            earned += 1
        else:
            notes.append(f"{label}[{run}]: {got:.4g} (expected {true_val:.4g} +/- {tol})")
    return earned, possible, notes


def check_eq(ans: dict | None, truth: dict, key: str, label: str) -> tuple[int, int, list[str]]:
    """Score one exact-match field (bool or string)."""
    true_val = truth[key]
    try:
        got = ans[key]  # type: ignore[index]
    except (KeyError, TypeError):
        return 0, 1, [f"{label}: missing"]
    if got == true_val:
        return 1, 1, []
    return 0, 1, [f"{label}: {got!r} (expected {true_val!r})"]


def grade_task(name: str, checks: list[tuple[int, int, list[str]]]) -> tuple[int, int]:
    earned = sum(c[0] for c in checks)
    possible = sum(c[1] for c in checks)
    print(f"{name}: {earned}/{possible}")
    for _, _, notes in checks:
        for note in notes:
            print(f"    - {note}")
    return earned, possible


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers_dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="passing fraction (default 0.9)")
    args = parser.parse_args()

    with open(GRADING_DIR / "ground_truth.json") as f:
        gt = json.load(f)

    total_earned, total_possible = 0, 0

    a = load_answers(args.answers_dir, "task01")
    t = gt["task01"]
    e, p = grade_task("task01_dissipation_plateau", [
        check_rel(a, t, "eps_nu_mean", "eps_nu_mean", REL_TOL),
        check_rel(a, t, "hermite_energy_mean", "hermite_energy_mean", REL_TOL),
        check_eq(a, t, "is_plateau", "is_plateau"),
    ])
    total_earned += e; total_possible += p

    a = load_answers(args.answers_dir, "task02")
    t = gt["task02"]
    e, p = grade_task("task02_hermite_spectrum", [
        check_abs(a, t, "slope", "slope", SLOPE_TOL),
        check_eq(a, t, "best_theory", "best_theory"),
    ])
    total_earned += e; total_possible += p

    a = load_answers(args.answers_dir, "task03")
    t = gt["task03"]
    e, p = grade_task("task03_dissipation_integrand", [
        check_rel(a, t, "total_dissipation", "total_dissipation", REL_TOL),
        check_abs(a, t, "peak_m", "peak_m", PEAK_M_TOL),
        check_eq(a, t, "totals_match_within_5pct", "totals_match_within_5pct"),
        check_eq(a, t, "peak_m_trend", "peak_m_trend"),
    ])
    total_earned += e; total_possible += p

    a = load_answers(args.answers_dir, "task04")
    t = gt["task04"]
    e, p = grade_task("task04_M_convergence", [
        check_rel(a, t, "eps_nu_mean", "eps_nu_mean", REL_TOL),
        check_eq(a, t, "truncation_independent", "truncation_independent"),
    ])
    total_earned += e; total_possible += p

    a = load_answers(args.answers_dir, "task05")
    t = gt["task05"]
    q2_checks = []
    for item, true_val in t["q2"].items():
        got = None
        if a is not None and isinstance(a.get("q2"), dict):
            got = a["q2"].get(item)
        if got == true_val:
            q2_checks.append((1, 1, []))
        else:
            q2_checks.append((0, 1, [f"q2[{item}]: {got!r} (expected {true_val!r})"]))
    e, p = grade_task("task05_blowup_diagnosis", [
        check_eq(a, t, "q1", "q1"),
        *q2_checks,
        check_eq(a, t, "q3", "q3"),
    ])
    total_earned += e; total_possible += p

    frac = total_earned / total_possible
    print(f"\nTOTAL: {total_earned}/{total_possible} ({frac:.1%})  "
          f"[pass threshold {args.threshold:.0%}]")
    sys.exit(0 if frac >= args.threshold else 1)


if __name__ == "__main__":
    main()
