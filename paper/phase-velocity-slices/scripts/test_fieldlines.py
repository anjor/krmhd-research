#!/usr/bin/env python
"""Self-test of fieldlines.py (32^3 grids, under a minute).  Exits with status 1 if a check fails.

F1  Psi = a sin(k1 x + k1 z): b = (0, a k1 cos(k1 x + k1 z)); the exact line is x = x0, y = y0 + a [sin(k1 x0 + k1 z)
    - sin(k1 x0)].  RK4 + cubic splines reproduce it.
F2  A z-independent random Psi: lines lie on Psi contours, so f = Psi is constant along every line; the along-line
    variance (each line's mean removed) is a tiny fraction of <Psi^2>.
F3  Straight lines (Psi = 0) and f = cos(k_z z) with |n_z| = 2: kappa_1 and kappa_half equal 2 k1 along lines, and
    the Eulerian measures of the same field agree.
F4  Random turbulent-like Psi and f: the derivative of f along the traced lines (4th-order finite differences of the
    samples) equals grad_par f = d_z f + {Psi, f} computed spectrally and sampled on the same lines (fields chosen so
    that the product b . grad f lies inside the 2/3 band, where GANDALF's dealiased bracket is exact).
"""
from __future__ import annotations

import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1lib as P  # noqa: E402
import fieldlines as FL  # noqa: E402
import numpy as np  # noqa: E402

FAILED: list[str] = []
T0 = time.time()


def check(label: str, ok: bool, detail: str = "") -> bool:
    ok = bool(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)
    return ok


def main() -> int:
    N = 32
    box = P.Box(N)
    k1 = box.k1
    rng = np.random.default_rng(3)
    x0 = rng.uniform(0, 1, 256)
    y0 = rng.uniform(0, 1, 256)

    a = 0.05
    xr = box.coords()
    X3 = xr[None, None, :]
    Z3 = xr[:, None, None]
    psi = box.from_real(np.broadcast_to(a * np.sin(k1 * X3 + k1 * Z3), (N, N, N)))
    tr = FL.Tracer(box, psi, pad=2)
    X, Y, z = tr.trace(x0, y0, 2.0, 1.0 / 128)
    Yex = y0[None, :] + a * (np.sin(k1 * x0[None, :] + k1 * z[:, None]) - np.sin(k1 * x0[None, :]))
    err = float(max(np.abs(X - x0[None, :]).max(), np.abs(Y - Yex).max()))
    check("F1 analytic line, 2 box lengths", err < 1e-5, f"max position error {err:.1e} (excursion {2 * a:.2f})")

    psi2 = box.random_field(rng, (1, 6), 0)
    psi2 = P.normalise_rms_grad(box, psi2, 0.5)
    tr2 = FL.Tracer(box, psi2, pad=2)
    tr2.trace(x0, y0, 4.0, 1.0 / 128)
    s = tr2.sample(psi2)
    frac = float(np.var(s, axis=0).mean()) / box.avg2(psi2)
    lm = FL.line_measures(s, 1.0 / 128)
    floor = 2 * np.pi / 4.0 / 3.0 * 1.05                     # Hann leakage of a constant into the first bin
    check("F2 Psi constant along 2D field lines (kappa_1 at the window floor (2 pi/Z)/3)",
          frac < 1e-4 and lm["kappa_1"] < floor,
          f"along-line variance / <Psi^2> = {frac:.1e}; kappa_1 = {lm['kappa_1'] / k1:.1e} k1")

    f3 = np.zeros((N, N, N // 2 + 1), complex)
    f3 = box.from_real(np.broadcast_to(np.cos(2 * k1 * Z3 + 0.3), (N, N, N)))
    tr3 = FL.Tracer(box, np.zeros_like(f3), pad=2)
    tr3.trace(x0, y0, 4.0, 1.0 / 128)
    lm3 = FL.line_measures(tr3.sample(f3), 1.0 / 128)
    eu3 = FL.eulerian_measures(box, f3, lm3["lags"])
    check("F3 single wavenumber: kappa_1, kappa_half (lines) and Eulerian = 2 k1",
          abs(lm3["kappa_1"] / (2 * k1) - 1) < 0.03 and abs(lm3["kappa_half"] / (2 * k1) - 1) < 0.02
          and abs(eu3["kappa_1"] / (2 * k1) - 1) < 1e-12 and abs(eu3["kappa_half"] / (2 * k1) - 1) < 0.01,
          f"lines {lm3['kappa_1'] / k1:.3f}, {lm3['kappa_half'] / k1:.3f}; Eulerian {eu3['kappa_1'] / k1:.3f}, "
          f"{eu3['kappa_half'] / k1:.3f} (units of k1)")

    psi4 = P.normalise_rms_grad(box, box.random_field(rng, (1, 4), 2), 0.6)      # product b.grad f stays inside
    f4 = box.random_field(rng, (1, 4), 2)                                           # the 2/3 band (|n| <= 8 < 10)
    tr4 = FL.Tracer(box, psi4, pad=2)
    dz = 1.0 / 256
    tr4.trace(x0, y0, 1.0, dz)
    s4 = tr4.sample(f4)
    g4 = tr4.sample(box.gradpar(psi4, f4))
    fd = (-s4[4:] + 8 * s4[3:-1] - 8 * s4[1:-3] + s4[:-4]) / (12 * dz)
    e4 = float(np.sqrt(np.mean((fd - g4[2:-2]) ** 2) / np.mean(g4[2:-2] ** 2)))
    check("F4 d f/dz along lines = grad_par f (spectral)", e4 < 1e-2, f"relative rms difference {e4:.1e}")

    print(f"{'ALL PASS' if not FAILED else 'FAILED: ' + ', '.join(FAILED)} ({time.time() - T0:.0f} s)")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
