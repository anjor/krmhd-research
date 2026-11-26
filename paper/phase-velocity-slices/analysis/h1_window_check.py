#!/usr/bin/env python
"""Leakage check of the field-aligned spectra: the shares of a field's variance beyond 2, 4 and 8 k_1 along the
traced lines with the estimator of scripts/fieldlines.py (Hann-windowed periodogram of each non-periodic 4L segment,
global mean removed) against three others on the same samples: Hann after removing a straight line from each segment,
a 4-term Blackman-Harris window (side lobes 92 dB down, against 31 dB for Hann) and no window.

If the Hann shares are set by spectral leakage from the bulk of the spectrum near k_1, a window with lower side lobes
gives smaller shares; if they are the field's own structure, the windowed estimates agree.  The fields are those of
the tails quoted in section 7: the free energy (sum over m of the spectra of g_m, as in h1_kpar_spectra.py), g_0, the
whole E x B velocity u_perp (h1_alfven_lines.py) and the frozen copy of g_0 (h1_frozen_copy.py).

Reads <run dir>/snap_XXX.npz (local snapshots; the lines are traced with the options and seed of h1_fieldlines.py, so
the Hann spectra reproduce the stored fieldlines_/alfven_lines_/frozen_copy_ spectra) and writes
data/h1_window_check.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_window_check.py <run> <snaps> [<run> <snaps> ...]
e.g. P1_forced-beta0p3-m64_20261018_144656 7,12 P1_forced-beta0p3-n96_20261019_200712 3,5 (as run on 2026-11-26)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import p1lib as P  # noqa: E402  (first: float64 JAX on the CPU)
import fieldlines as FL  # noqa: E402
import numpy as np  # noqa: E402
from h1_fieldlines import DEFAULTS  # noqa: E402
from h1_frozen_copy import frozen_copy  # noqa: E402

SHARES = (2, 4, 8)          # thresholds in units of k_1


def blackman_harris(K: int) -> np.ndarray:
    n = np.arange(K)
    a = (0.35875, 0.48829, 0.14128, 0.01168)
    return sum((-1) ** i * c * np.cos(2 * np.pi * i * n / (K - 1)) for i, c in enumerate(a))


def periodogram(samples: np.ndarray, window: str, detrend: bool) -> tuple[np.ndarray, float]:
    """Line-averaged one-sided periodogram of samples [K, n_lines], normalised to the variance of the samples that
    enter it (after the line detrend if any), and that variance."""
    f = np.asarray(samples, dtype=float)
    K = f.shape[0]
    if detrend:
        x = np.arange(K) - (K - 1) / 2.0
        a, b = f.mean(0), (x[:, None] * f).sum(0) / (x ** 2).sum()
        f = f - a[None, :] - b[None, :] * x[:, None]
    else:
        f = f - f.mean()
    w = {"hann": np.hanning(K), "blackman_harris": blackman_harris(K), "none": np.ones(K)}[window]
    P_ = np.mean(np.abs(np.fft.rfft(f * w[:, None], axis=0)) ** 2, axis=1)
    P_[1:] *= 2.0
    if K % 2 == 0:
        P_[-1] /= 2.0
    var = float(np.mean(f ** 2))
    return P_ * var / P_.sum(), var


ESTIMATORS = {"hann (fieldlines.py)": ("hann", False), "hann, line detrended": ("hann", True),
              "blackman-harris": ("blackman_harris", False), "no window": ("none", False)}


def shares(kap: np.ndarray, E: np.ndarray, total: float) -> dict[str, float]:
    """Power beyond k k_1 over the variance of the field along the lines (global mean removed), so that an estimator
    that removes the per-line mean and trend is not credited with a smaller total."""
    return {f"above_{k}": float(E[kap > k + 1e-9].sum() / total) for k in SHARES}


def analyse(run: str, snap: int) -> dict:
    rd = P.DATA / run
    cfg = json.load(open(rd / f"{run}.json"))["config"]
    opts = dict(DEFAULTS)
    opts.update(cfg.get("analysis", {}).get("fieldlines", {}))
    box = P.Box(int(cfg["box"]["N"]), float(cfg["box"]["L"]))
    z = np.load(rd / f"snap_{snap:03d}.npz")
    zp, zm, g = (np.asarray(z[k], dtype=complex) for k in ("zp", "zm", "g"))
    phi, psi = P.phi_psi(zp, zm)
    t0 = time.time()
    tr = FL.Tracer(box, psi, pad=int(opts["pad"]))
    dz = float(opts["dz"])
    f_frozen = frozen_copy(box, tr, g[..., 0], 4, dz)          # traces from the grid points; the lines come after
    rng = np.random.default_rng(int(opts["seed"]))
    n = int(opts["n_lines"])
    x0, y0 = rng.uniform(0, box.L, n), rng.uniform(0, box.L, n)
    _, _, zz = tr.trace(x0, y0, float(opts["z_len"]), dz)
    kap = 2 * np.pi * np.fft.rfftfreq(len(zz), d=dz) / box.k1
    fields = {"free energy": [g[..., m] for m in range(g.shape[-1])], "g0": [g[..., 0]],
              "u_perp": [-1j * box.KY * phi, 1j * box.KX * phi], "frozen copy": [f_frozen]}
    out: dict = dict(run=run, N=box.N, snapshot=snap, t=float(z["t"]))
    hann_check = 0.0
    for name, comps in fields.items():
        E = {k: 0.0 for k in ESTIMATORS}
        total = 0.0
        for c in comps:                                            # a sum of fields: spectra in units of variance
            s = tr.sample(c)
            total += float(np.mean((s - s.mean()) ** 2))
            for k, (w, dt) in ESTIMATORS.items():
                E[k] = E[k] + periodogram(s, w, dt)[0]
            ref = FL.line_measures(s, dz)["E_par"]                 # the stored estimator, for the reproduction check
            hann_check = max(hann_check, float(np.abs(periodogram(s, "hann", False)[0] - ref).max() / ref.sum()))
        out[name] = {k: shares(kap, e, total) for k, e in E.items()}
    out["hann_vs_fieldlines_max_abs_diff"] = hann_check
    out["wall_s"] = time.time() - t0
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("items", nargs="+", help="pairs <run> <comma-separated snapshot indices>, e.g. RUN1 7,12 RUN2 3,5")
    args = ap.parse_args()
    if len(args.items) % 2:
        raise SystemExit("give <run> <snaps> pairs")
    pairs = list(zip(args.items[0::2], args.items[1::2]))
    res = []
    for run, snaps in pairs:
        for sn in (int(s) for s in snaps.split(",")):
            r = analyse(run, sn)
            res.append(r)
            print(f"{run} snap {sn} (N = {r['N']}, {r['wall_s']:.0f} s; Hann reproduces fieldlines.py to "
                  f"{r['hann_vs_fieldlines_max_abs_diff']:.1e}):", flush=True)
            for name in ("free energy", "g0", "u_perp", "frozen copy"):
                print(f"   {name:12s} " + "; ".join(
                    f"{k}: " + "/".join(f"{100 * v:.3g}" for v in r[name][k].values()) for k in ESTIMATORS)
                    + "  (% beyond 2/4/8 k1)", flush=True)
    (P.DATA / "h1_window_check.json").write_text(json.dumps(P._jsonable(res), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
