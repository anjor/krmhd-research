"""Dissipation rate computation from Hermite and spatial spectra.

Physics: In steady-state driven turbulence, the total dissipation rate
epsilon = epsilon_nu + epsilon_eta must balance the injection rate.
The dissipative anomaly predicts epsilon -> const as nu -> 0.

The collision operator in GANDALF uses normalized moment indices:
    C[g_m] = -nu * (m/M)^hyper_n * g_m    for m >= 2
    C[g_m] = 0                             for m = 0, 1

The resistive operator uses normalized perpendicular wavenumbers:
    R[g_k] = -eta * (k_perp^2 / k_perp_max^2)^hyper_r * g_k

References:
    - Nastac, Tatsuno & Schekochihin (2024) -- dissipative anomaly theory
    - GANDALF timestepping.py lines 460-487 (operator implementation)
"""

from __future__ import annotations

import numpy as np


def compute_collisional_dissipation(
    E_m: np.ndarray,
    nu: float,
    M: int,
    hyper_n: int = 1,
) -> float:
    """Compute collisional dissipation rate from Hermite moment spectrum.

    Parameters
    ----------
    E_m : np.ndarray
        Energy in each Hermite moment, shape [M+1].
        E_m[m] = sum_k |g_{m,k}|^2.
        Typically from krmhd.diagnostics.hermite_moment_energy(state).
    nu : float
        Collision frequency coefficient.
    M : int
        Number of Hermite moments (truncation).
    hyper_n : int
        Hyper-collision order. Default 1 (standard Lenard-Bernstein).

    Returns
    -------
    float
        Collisional dissipation rate:
        epsilon_nu = 2 * nu * sum_{m=2}^{M} (m/M)^hyper_n * E_m[m]

        Factor of 2: GANDALF damps amplitudes as g -> g*exp(-rate*dt),
        so energy E=|g|^2 decays at dE/dt = -2*rate*E.
        Moments m=0 (density) and m=1 (momentum) are exempt.
    """
    if M <= 1 or len(E_m) <= 2 or nu == 0.0:
        return 0.0

    m_indices = np.arange(len(E_m))
    rates = nu * (m_indices / M) ** hyper_n
    # m=0,1 exempt from collisions
    rates[:2] = 0.0
    return float(2.0 * np.sum(rates * E_m))


def compute_resistive_dissipation(
    E_kperp: np.ndarray,
    k_perp: np.ndarray,
    eta: float,
    hyper_r: int = 1,
) -> float:
    """Compute resistive dissipation rate from perpendicular energy spectrum.

    Parameters
    ----------
    E_kperp : np.ndarray
        Energy spectrum E(k_perp), shape [n_bins].
    k_perp : np.ndarray
        Perpendicular wavenumber bins, shape [n_bins].
    eta : float
        Resistivity coefficient.
    hyper_r : int
        Hyper-resistivity order. Default 1 (standard Laplacian).

    Returns
    -------
    float
        Resistive dissipation rate:
        epsilon_eta = 2 * eta * sum_k (k_perp^2 / k_max^2)^hyper_r * E(k_perp)

        Uses GANDALF's normalized k_perp (k_perp_max from the grid).
    """
    k_max = k_perp[-1] if len(k_perp) > 0 else 1.0
    if k_max == 0:
        return 0.0
    rates = eta * (k_perp**2 / k_max**2) ** hyper_r
    return float(2.0 * np.sum(rates * E_kperp))


def compute_total_dissipation(
    E_m: np.ndarray,
    E_kperp: np.ndarray,
    k_perp: np.ndarray,
    nu: float,
    eta: float,
    M: int,
    hyper_n: int = 1,
    hyper_r: int = 1,
) -> dict[str, float]:
    """Compute total dissipation rate (collisional + resistive).

    Returns
    -------
    dict
        'collisional': epsilon_nu
        'resistive': epsilon_eta
        'total': epsilon_nu + epsilon_eta
    """
    eps_nu = compute_collisional_dissipation(E_m, nu, M, hyper_n)
    eps_eta = compute_resistive_dissipation(E_kperp, k_perp, eta, hyper_r)
    return {
        "collisional": eps_nu,
        "resistive": eps_eta,
        "total": eps_nu + eps_eta,
    }
