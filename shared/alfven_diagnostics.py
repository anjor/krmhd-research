"""Alfvén-sector diagnostics used by the Study 3 forcing gate.

These diagnostics intentionally contain no Hermite quantities: cross-helicity
belongs to the RMHD advector, not to the passive compressive distribution.
"""

from __future__ import annotations

import numpy as np

from krmhd.physics import KRMHDState


def _rfft_doubling(nx: int) -> np.ndarray:
    """Return Parseval weights for the stored x-rFFT half-plane."""
    weights = np.full(nx // 2 + 1, 2.0)
    weights[0] = 1.0
    if nx % 2 == 0:
        weights[-1] = 1.0
    return weights


def elsasser_energies(state: KRMHDState) -> tuple[float, float]:
    """Return perpendicular-gradient energies carried by z⁺ and z⁻.

    Common FFT-normalisation factors cancel in sigma_c, but are included so the
    values scale conventionally with the number of perpendicular grid points.
    """
    grid = state.grid
    kperp2 = np.asarray(grid.kx)[None, None, :] ** 2 + np.asarray(grid.ky)[None, :, None] ** 2
    weights = _rfft_doubling(grid.Nx)[None, None, :]
    normalisation = 1.0 / (grid.Nx * grid.Ny) ** 2
    eplus = 0.25 * normalisation * np.sum(weights * kperp2 * np.abs(np.asarray(state.z_plus)) ** 2)
    eminus = 0.25 * normalisation * np.sum(weights * kperp2 * np.abs(np.asarray(state.z_minus)) ** 2)
    return float(eplus), float(eminus)


def sigma_c(state: KRMHDState) -> float:
    """Volume-integrated Elsasser cross-helicity normalised to [-1, 1]."""
    eplus, eminus = elsasser_energies(state)
    denominator = eplus + eminus
    return float((eplus - eminus) / denominator) if denominator > 0.0 else float("nan")


def elsasser_perpendicular_spectra(state: KRMHDState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shell-integrated E⁺(n_perp) and E⁻(n_perp) spectra.

    Shells are indexed by integer perpendicular mode number rather than by
    physical k.  This makes the task-1 pile-up test independent of box size.
    """
    grid = state.grid
    nx_modes = np.rint(np.abs(np.asarray(grid.kx)) * grid.Lx / (2.0 * np.pi))
    ny_modes = np.rint(np.abs(np.asarray(grid.ky)) * grid.Ly / (2.0 * np.pi))
    nperp = np.rint(np.sqrt(nx_modes[None, :] ** 2 + ny_modes[:, None] ** 2)).astype(int)
    nmax = int(nperp.max())
    kperp2 = np.asarray(grid.kx)[None, :] ** 2 + np.asarray(grid.ky)[:, None] ** 2
    weights = _rfft_doubling(grid.Nx)[None, :]
    normalisation = 0.25 / (grid.Nx * grid.Ny) ** 2

    def shell_sum(field: np.ndarray) -> np.ndarray:
        density = normalisation * weights * kperp2 * np.sum(np.abs(field) ** 2, axis=0)
        return np.bincount(nperp.ravel(), weights=density.ravel(), minlength=nmax + 1)

    return np.arange(nmax + 1), shell_sum(np.asarray(state.z_plus)), shell_sum(np.asarray(state.z_minus))


def high_k_fraction(nperp: np.ndarray, eplus: np.ndarray, eminus: np.ndarray, dealias_cutoff: int) -> float:
    """Fraction of Elsasser energy in the final 20% of resolved shells."""
    total = float(np.sum(eplus) + np.sum(eminus))
    if total == 0.0:
        return float("nan")
    tail = nperp >= int(np.ceil(0.8 * dealias_cutoff))
    return float((np.sum(eplus[tail]) + np.sum(eminus[tail])) / total)
