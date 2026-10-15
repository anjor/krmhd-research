"""Field-line tracing and field-aligned (parallel) spectra for the H1 pilot: is the parallel structure of the
compressive field g frozen at the injected field-aligned wavenumber (H1; S09 section 6.3) or critically balanced like
the Alfvenic fields (Meyrand et al. 2019, who traced field lines with RK4 and cubic splines)?

Field lines of b = zhat + b_perp, b_perp = zhat x grad Psi = (-d_y Psi, d_x Psi) (so that b_perp . grad f = {Psi, f},
p1lib's grad_par), parametrised by z (RMHD: arc length = z to leading order): dx/dz = b_x(x, y, z), dy/dz = b_y.
Fields are put on a grid refined by `pad` by exact Fourier interpolation (p1lib.Box.to_real), represented by periodic
cubic B-splines (scipy.ndimage, mode "grid-wrap"), and the lines are integrated with classical RK4 in z.

Along each line of length Z (several box lengths; the lines wander, so successive passes sample new field) a sampled
field f(z_k) gives
  E_par(kappa): the Hann-windowed periodogram averaged over lines, kappa = 2 pi n / Z (kappa = 0 holds the
                structure longer than a line), normalised so that
                sum_n E_par = <f^2> along the lines (one-sided in |kappa|);
  C_par(l): the normalised autocorrelation <f(z) f(z + l)>/<f^2> (unbiased, averaged over lines);
  kappa_1 = sum |kappa| E_par / sum E_par (first moment: the energy-weighted |kappa| that sets the phase-mixing flux);
  kappa_half = pi / (3 l_half), l_half the first lag with C_par = 1/2 (exact for a single wavenumber, cos(kappa l)).
The same measures of the Eulerian k_z structure (along z at fixed x, y) come from the Fourier coefficients directly
(eulerian_measures), so that field-aligned and k_z-based wavenumbers are compared like with like.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage

__all__ = ["Tracer", "bandpass", "line_measures", "eulerian_measures", "kappa_measures_from_spectrum"]


def bandpass(box, f: np.ndarray, n_lo: float, n_hi: float) -> np.ndarray:
    """f restricted to k_perp L/2pi in (n_lo, n_hi] (n_lo = 0 keeps k_perp = 0 out as well)."""
    kp = np.sqrt(box.KP2) / box.k1
    sel = (kp > n_lo + 1e-9) & (kp <= n_hi + 1e-9)
    return np.asarray(f) * sel


class Tracer:
    """Field lines of one snapshot.  psi: rfft coefficients of Psi on box (p1lib.Box).  pad: refinement of the
    interpolation grid.  trace(...) integrates lines; sample(f) interpolates a field (rfft coefficients) along the last
    traced lines."""

    def __init__(self, box, psi: np.ndarray, pad: int = 2) -> None:
        self.box = box
        self.pad = int(pad)
        self.n = box.N * self.pad
        self.h = box.L / self.n
        bx = box.to_real(-1j * box.KY * psi, self.pad)
        by = box.to_real(1j * box.KX * psi, self.pad)
        self.cbx = ndimage.spline_filter(bx, order=3, mode="grid-wrap")
        self.cby = ndimage.spline_filter(by, order=3, mode="grid-wrap")
        self.b_rms = float(np.sqrt(np.mean(bx ** 2 + by ** 2)))
        self.X = self.Y = None
        self.z = None

    def _interp(self, coef: np.ndarray, x: np.ndarray, y: np.ndarray, z: float | np.ndarray) -> np.ndarray:
        n = self.n
        zz = np.broadcast_to(np.mod(np.asarray(z, dtype=float) / self.h, n), x.shape)
        coords = np.array([zz.ravel(), np.mod(y / self.h, n).ravel(), np.mod(x / self.h, n).ravel()])
        return ndimage.map_coordinates(coef, coords, order=3, mode="grid-wrap", prefilter=False).reshape(x.shape)

    def b(self, x: np.ndarray, y: np.ndarray, z: float) -> tuple[np.ndarray, np.ndarray]:
        return self._interp(self.cbx, x, y, z), self._interp(self.cby, x, y, z)

    def trace(self, x0: np.ndarray, y0: np.ndarray, z_len: float, dz: float, z0: float = 0.0) -> tuple[np.ndarray, ...]:
        """RK4 in z from (x0, y0, z0) over z_len with step dz; returns X, Y [n_steps + 1, n_lines] (unwrapped) and z."""
        ns = int(round(z_len / dz))
        X = np.empty((ns + 1, len(x0)))
        Y = np.empty_like(X)
        X[0], Y[0] = x0, y0
        x, y = np.array(x0, dtype=float), np.array(y0, dtype=float)
        z = z0
        for k in range(ns):
            k1x, k1y = self.b(x, y, z)
            k2x, k2y = self.b(x + 0.5 * dz * k1x, y + 0.5 * dz * k1y, z + 0.5 * dz)
            k3x, k3y = self.b(x + 0.5 * dz * k2x, y + 0.5 * dz * k2y, z + 0.5 * dz)
            k4x, k4y = self.b(x + dz * k3x, y + dz * k3y, z + dz)
            x = x + dz / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
            y = y + dz / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
            z = z0 + (k + 1) * dz
            X[k + 1], Y[k + 1] = x, y
        self.X, self.Y = X, Y
        self.z = z0 + dz * np.arange(ns + 1)
        return X, Y, self.z

    def sample(self, f: np.ndarray) -> np.ndarray:
        """Values of the field f (rfft coefficients on the box) along the traced lines, [n_steps + 1, n_lines]."""
        fr = self.box.to_real(np.asarray(f), self.pad)
        c = ndimage.spline_filter(fr, order=3, mode="grid-wrap")
        Z = np.broadcast_to(self.z[:, None], self.X.shape)
        return self._interp(c, self.X, self.Y, Z)


def kappa_measures_from_spectrum(kappa: np.ndarray, E: np.ndarray, lags: np.ndarray, C: np.ndarray) -> dict[str, float]:
    """kappa_1 (first moment of |kappa| under E), kappa_rms (second moment), kappa_half = pi/(3 l_half) from the
    first crossing of C = 1/2 (linear interpolation; nan if C stays above 1/2)."""
    E = np.asarray(E, dtype=float)
    tot = E.sum()
    k1 = float(np.sum(np.abs(kappa) * E) / tot) if tot > 0 else float("nan")
    krms = float(math.sqrt(np.sum(kappa ** 2 * E) / tot)) if tot > 0 else float("nan")
    idx = np.where(C < 0.5)[0]
    if len(idx) == 0 or idx[0] == 0:
        lh = float("nan")
    else:
        i = idx[0]
        lh = float(lags[i - 1] + (C[i - 1] - 0.5) / (C[i - 1] - C[i]) * (lags[i] - lags[i - 1]))
    return dict(kappa_1=k1, kappa_rms=krms, l_half=lh, kappa_half=math.pi / (3.0 * lh) if lh == lh else float("nan"))


def line_measures(samples: np.ndarray, dz: float, max_lag: int | None = None) -> dict[str, Any]:
    """Parallel spectrum, autocorrelation and wavenumber measures of samples [K, n_lines] taken every dz along lines.
    The periodogram uses a Hann window; the global mean (over all lines and samples) is removed."""
    f = np.asarray(samples, dtype=float)
    K = f.shape[0]
    f = f - f.mean()                        # global mean only: structures longer than a line stay at kappa ~ 0
    var = float(np.mean(f ** 2))
    win = np.hanning(K)
    fw = f * win[:, None] / math.sqrt(np.mean(win ** 2))
    F = np.fft.rfft(fw, axis=0)
    P = np.mean(np.abs(F) ** 2, axis=1) / K ** 2
    P[1:] *= 2.0
    if K % 2 == 0:
        P[-1] /= 2.0
    kappa = 2.0 * np.pi * np.fft.rfftfreq(K, d=dz)
    E = P * var / P.sum() if P.sum() > 0 else P
    L = max_lag if max_lag is not None else K // 2
    G = np.fft.rfft(f, n=2 * K, axis=0)
    ac = np.fft.irfft(np.abs(G) ** 2, axis=0)[:L + 1].mean(axis=1) / (K - np.arange(L + 1))
    C = ac / ac[0] if ac[0] > 0 else ac
    lags = dz * np.arange(L + 1)
    out = dict(kappa=kappa, E_par=E, lags=lags, C_par=C, var=var)
    out.update(kappa_measures_from_spectrum(kappa, E, lags, C))
    return out


def eulerian_measures(box, f: np.ndarray, lags: np.ndarray) -> dict[str, Any]:
    """The same measures for the Eulerian z structure of f (rfft coefficients): E_z(|k_z|) from the Fourier
    coefficients and C_z(l) = sum_kz E_z cos(k_z l) / sum E_z."""
    f = np.asarray(f)
    dens = (box.w * np.abs(f) ** 2).sum(axis=(1, 2)) / box.Ntot ** 2           # per k_z row
    nzb = int(box.nz.max()) + 1
    Ez = np.bincount(box.nz, weights=dens, minlength=nzb)
    kz = box.k1 * np.arange(nzb)
    C = np.array([np.sum(dens * np.cos(box.kz * l)) for l in lags]) / dens.sum()
    out = dict(kappa=kz, E_z=Ez, C_z=C)
    out.update(kappa_measures_from_spectrum(kz, Ez, lags, C))
    return out
