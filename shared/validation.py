"""Validation gates for KRMHD simulation diagnostics.

Implements the 4 gates from CLAUDE.md:
1. Energy conservation: relative error < 1%
2. Energy balance: injection ≈ dissipation within 5%
3. Steady state: fluctuations < 10% of mean over final 50 τ_A
4. Spectral sanity: E(k⊥) shows inertial range
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from krmhd.diagnostics import EnergyHistory
    from krmhd.physics import KRMHDState


@dataclass
class GateResult:
    """Result of a single validation gate."""
    name: str
    passed: bool
    message: str
    value: float | None = None
    threshold: float | None = None


def check_energy_conservation(history: EnergyHistory) -> GateResult:
    """Check relative energy error < 1% at steady state.

    Compares max deviation of total energy from its time-mean
    over the second half of the simulation.
    """
    E = np.array(history.E_total)
    if len(E) < 4:
        return GateResult("energy_conservation", False, "Too few data points", None, 0.01)

    # Use second half of simulation for steady-state check
    E_ss = E[len(E) // 2:]
    E_mean = np.mean(E_ss)
    if E_mean == 0:
        return GateResult("energy_conservation", False, "Zero mean energy", 0.0, 0.01)

    rel_error = np.max(np.abs(E_ss - E_mean)) / E_mean
    passed = rel_error < 0.01
    return GateResult(
        "energy_conservation", passed,
        f"Relative energy error: {rel_error:.4f} (threshold: 0.01)",
        float(rel_error), 0.01,
    )


def check_energy_balance(
    injection_rate: float, dissipation_rate: float,
) -> GateResult:
    """Check injection rate ≈ dissipation rate within 5%.

    Parameters
    ----------
    injection_rate : float
        Mean energy injection rate (from forcing).
    dissipation_rate : float
        Mean energy dissipation rate (from viscosity/resistivity).
    """
    if injection_rate == 0:
        return GateResult(
            "energy_balance", False,
            "Zero injection rate — forcing may be disabled", 0.0, 0.05,
        )

    rel_diff = abs(injection_rate - dissipation_rate) / injection_rate
    passed = rel_diff < 0.05
    return GateResult(
        "energy_balance", passed,
        f"|(inj - diss)/inj| = {rel_diff:.4f} (threshold: 0.05)",
        float(rel_diff), 0.05,
    )


def check_steady_state(history: EnergyHistory, tau_A: float = 1.0) -> GateResult:
    """Check total energy fluctuations < 10% of mean over final 50 τ_A.

    Parameters
    ----------
    tau_A : float
        Alfvén crossing time. Used to select final 50 τ_A window.
    """
    times = np.array(history.times)
    E = np.array(history.E_total)

    if len(E) < 4:
        return GateResult("steady_state", False, "Too few data points", None, 0.10)

    t_final = times[-1]
    window_start = t_final - 50.0 * tau_A

    if window_start < times[0]:
        # Simulation shorter than 50 τ_A — use full run
        mask = np.ones(len(times), dtype=bool)
        msg_extra = " (run shorter than 50 τ_A, using full run)"
    else:
        mask = times >= window_start
        msg_extra = ""

    E_window = E[mask]
    E_mean = np.mean(E_window)
    if E_mean == 0:
        return GateResult("steady_state", False, "Zero mean energy in window", 0.0, 0.10)

    fluct = np.std(E_window) / E_mean
    passed = fluct < 0.10
    return GateResult(
        "steady_state", passed,
        f"Energy fluctuation: {fluct:.4f} (threshold: 0.10){msg_extra}",
        float(fluct), 0.10,
    )


def check_spectral_sanity(
    k_perp: np.ndarray, E_kperp: np.ndarray,
) -> GateResult:
    """Check that E(k⊥) shows an inertial range (power-law behavior).

    Fits log-log slope in the mid-k range. A reasonable inertial range
    has slope between -5 and -1 (roughly k^{-3/2} to k^{-5/3} expected).
    """
    # Filter to positive, finite values
    valid = (k_perp > 0) & (E_kperp > 0) & np.isfinite(E_kperp)
    if np.sum(valid) < 4:
        return GateResult("spectral_sanity", False, "Too few valid spectral bins", None, None)

    lk = np.log10(k_perp[valid])
    lE = np.log10(E_kperp[valid])

    # Use middle 60% of k-range for fit (avoid forcing and dissipation scales)
    n = len(lk)
    i_lo = n // 5
    i_hi = 4 * n // 5
    if i_hi - i_lo < 3:
        i_lo, i_hi = 0, n

    coeffs = np.polyfit(lk[i_lo:i_hi], lE[i_lo:i_hi], 1)
    slope = coeffs[0]

    # Inertial range: slope should be negative (energy decreasing with k)
    # and not too steep (bottleneck) or too shallow (under-resolved)
    passed = -5.0 < slope < -1.0
    return GateResult(
        "spectral_sanity", passed,
        f"Spectral slope: {slope:.2f} (expected between -5 and -1)",
        float(slope), None,
    )


def run_all_gates(
    history: EnergyHistory,
    state: KRMHDState,
    injection_rate: float | None = None,
    dissipation_rate: float | None = None,
    tau_A: float = 1.0,
) -> list[GateResult]:
    """Run all validation gates and return results.

    Parameters
    ----------
    history : EnergyHistory
        Time series of energy components.
    state : KRMHDState
        Final simulation state (for spectral analysis).
    injection_rate : float, optional
        Mean forcing injection rate. If None, energy balance gate is skipped.
    dissipation_rate : float, optional
        Mean dissipation rate. If None, energy balance gate is skipped.
    tau_A : float
        Alfvén crossing time for steady-state window.
    """
    from krmhd.diagnostics import energy_spectrum_perpendicular

    results: list[GateResult] = []

    results.append(check_energy_conservation(history))

    if injection_rate is not None and dissipation_rate is not None:
        results.append(check_energy_balance(injection_rate, dissipation_rate))
    else:
        results.append(GateResult(
            "energy_balance", False,
            "Skipped — injection/dissipation rates not provided", None, 0.05,
        ))

    results.append(check_steady_state(history, tau_A))

    k_perp, E_kperp = energy_spectrum_perpendicular(state)
    results.append(check_spectral_sanity(np.array(k_perp), np.array(E_kperp)))

    return results


def print_gate_results(results: list[GateResult]) -> None:
    """Print validation gate results to stdout."""
    print("\n=== Validation Gates ===")
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name}: {r.message}")
        if not r.passed:
            all_passed = False
    print(f"\nOverall: {'ALL PASSED' if all_passed else 'SOME GATES FAILED'}")
    print()
