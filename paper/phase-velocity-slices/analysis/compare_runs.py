#!/usr/bin/env python
"""Compare two runs of scripts/run_forced_turbulence.py with the same set-up, sample by sample: the check that a run on
other hardware (another GPU, a CPU) reproduces an earlier run to round-off over a short window
(configs/check_beta0p3_m64.yaml, configs/check_beta0p3_n128.yaml).

The two configs must agree except for the label and the output settings (time.t_end, print_every, snapshot_every,
restart_every).  Both runs then start from the same state with the same forcing seeds and differ only by round-off
(float64, ~1e-16 per operation), which the turbulence amplifies roughly exponentially in time, fastest at the
smallest scales.  For each sample time the runs share, the script gives the largest relative difference over all
sampled diagnostics (the ts__* arrays of the run files: energies, spectra, fluxes, dissipation, chi, ...), each
normalised by the largest magnitude of the reference diagnostic at that sample, and the largest relative difference
of the per-step time steps (dt is set from the fields every step, so it is a sensitive check of the state).  Two kinds
of diagnostic are round-off by construction and are treated so: the per-m mean squared wavenumbers (kfa2, kz2) are
ratios, compared only at the m whose free energy is at least W_FLOOR of the largest W(m) (moments that hold 1e-26 of
it right after the start have ratios of round-off; check of 2026-11-06: agreement 1e-11 or better above 1e-18); and the
time-stepping error accumulators (acc_err_*), of order 1e-14, are compared relative to the energy injected over the
same interval.  Values that are NaN in both runs (0/0 where g = 0 at t = 0) count as equal.  A sample
is shared when the times agree to 1e-9; the shorter run's final sample, taken after a step shortened to hit t_end, is
not shared.  PASS when every relative difference is at most --rtol (default 1e-8): a set-up difference (config,
GANDALF version, forcing) shows up at 1e-3 or more from the first sample on.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/compare_runs.py <run folder> <reference run folder> [--rtol R]
Writes compare_<reference run ID>.json into the first run folder; exit status 1 on FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Config keys that do not change the dynamics
OUTPUT_KEYS = {("label",), ("time", "t_end"), ("time", "print_every"), ("time", "snapshot_every"),
               ("time", "restart_every")}
ENV_KEYS = ("krmhd_version", "jax_version", "numpy_version", "scipy_version", "python_version", "precision",
            "jax_backend", "hardware")
RATIO_PER_M = ("kfa2", "kz2")          # per-m <k^2> weighted by |g_m|^2: ratios, meaningless where g_m is negligible
W_FLOOR = 1e-16                        # ... so compared only where W(m) >= W_FLOOR max_m W(m)
ERR_SCALE = {"acc_err_A": "acc_inj_A", "acc_err_W": "acc_inj_g", "acc_err_Wm": "acc_inj_g"}


def load(run_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The time series (ts__*), the per-step time steps and the metadata of a finished run folder."""
    npz, js = run_dir / f"{run_dir.name}.npz", run_dir / f"{run_dir.name}.json"
    if not npz.exists() or not js.exists():
        raise SystemExit(f"{run_dir}: no {npz.name} and {js.name} (unfinished run?)")
    with np.load(npz) as z:
        arrays = {k: z[k] for k in z.files if k.startswith("ts__") or k == "dt_steps"}
    return arrays, json.loads(js.read_text())


def config_differences(a: Any, b: Any, path: tuple[str, ...] = ()) -> list[str]:
    """Differences between two parsed configs outside OUTPUT_KEYS, as 'key.path: a vs b'."""
    if path in OUTPUT_KEYS:
        return []
    if isinstance(a, dict) and isinstance(b, dict):
        out: list[str] = []
        for k in sorted(set(a) | set(b)):
            out += config_differences(a.get(k), b.get(k), path + (str(k),))
        return out
    return [] if a == b else [f"{'.'.join(path)}: {a!r} vs {b!r}"]


def relative_differences(a: dict[str, np.ndarray], b: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    """Per diagnostic, max|a - b| / max|b| at each of the first n samples (absolute where the reference is zero), with
    the treatment of RATIO_PER_M and ERR_SCALE described in the module docstring."""
    W = b["ts__pWm"][:n]
    Wmax = W.max(axis=1, keepdims=True)
    keep_m = W >= W_FLOOR * np.where(Wmax > 0, Wmax, np.inf)       # (n, M+1); nothing kept while g = 0
    out = {}
    for k in sorted(set(a) & set(b)):
        if not k.startswith("ts__") or a[k].dtype.kind not in "fc" or a[k].shape[1:] != b[k].shape[1:]:
            continue
        name = k[4:]
        A, B = a[k][:n], b[k][:n]
        D = np.abs(A - B)
        D[np.isnan(A) & np.isnan(B)] = 0.0
        D[np.isnan(A) ^ np.isnan(B)] = np.inf
        Bm = np.where(np.isnan(B), 0.0, np.abs(B))
        if name in RATIO_PER_M:
            D, Bm = np.where(keep_m, D, 0.0), np.where(keep_m, Bm, 0.0)
        D, Bm = D.reshape(n, -1), Bm.reshape(n, -1)
        if D.shape[1] == 0:
            continue
        scale = np.abs(b["ts__" + ERR_SCALE[name]][:n]) if name in ERR_SCALE else Bm.max(axis=1)
        diff = D.max(axis=1)
        out[name] = np.where(scale > 0, diff / np.where(scale > 0, scale, 1.0), diff)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run", type=Path, help="run folder to check")
    ap.add_argument("reference", type=Path, help="run folder of the earlier run")
    ap.add_argument("--rtol", type=float, default=1e-8, help="largest relative difference that passes")
    args = ap.parse_args()
    a, ma = load(args.run)
    b, mb = load(args.reference)

    print(f"run       {ma['run_id']}: " + ", ".join(f"{k} {ma.get(k)}" for k in ENV_KEYS))
    print(f"reference {mb['run_id']}: " + ", ".join(f"{k} {mb.get(k)}" for k in ENV_KEYS))
    cdiff = config_differences(ma["config"], mb["config"])
    for d in cdiff:
        print(f"config differs: {d}")

    ta, tb = a["ts__t"], b["ts__t"]
    n = 0
    while n < min(len(ta), len(tb)) and abs(float(ta[n]) - float(tb[n])) <= 1e-9:
        n += 1
    rel = relative_differences(a, b, n) if n else {}
    worst = np.zeros(n)
    worst_key = [""] * n
    for k, r in rel.items():
        for i in range(n):
            if r[i] > worst[i]:
                worst[i], worst_key[i] = float(r[i]), k
    print(f"\n{n} shared samples, {len(rel)} diagnostics compared at each")
    print(f"{'sample':>6} {'t':>10} {'max rel. diff':>14}  largest in")
    for i in range(n):
        print(f"{i:6d} {float(ta[i]):10.6f} {worst[i]:14.3e}  {worst_key[i]}")

    # per-step time steps up to the last shared sample (the shorter run's final steps are shortened to hit t_end)
    da, db = a["dt_steps"], b["dt_steps"]
    m = 0
    if n:
        m = int(np.searchsorted(np.cumsum(db), float(tb[n - 1]) * (1 + 1e-12), side="right"))
        m = min(m, len(da), len(db))
    dt_rel = float(np.max(np.abs(da[:m] - db[:m]) / db[:m])) if m else float("nan")
    print(f"\ntime steps: {m} shared, max rel. diff {dt_rel:.3e}")

    ok = not cdiff and n >= 3 and m > 0 and float(worst.max()) <= args.rtol and dt_rel <= args.rtol
    sa = ma.get("extra", {}).get("s_per_step")
    sb = mb.get("extra", {}).get("s_per_step")
    print(f"s/step (whole run, compilation included): run {sa:.4g}, reference {sb:.4g}" if sa and sb else "")
    print(f"\n{'PASS' if ok else 'FAIL'}: rtol {args.rtol:g}"
          + ("" if n >= 3 else f"; only {n} shared samples (need 3)") + ("; config differs" if cdiff else ""))
    out = dict(run=ma["run_id"], reference=mb["run_id"], rtol=args.rtol, passed=bool(ok), config_differences=cdiff,
               environment={"run": {k: ma.get(k) for k in ENV_KEYS}, "reference": {k: mb.get(k) for k in ENV_KEYS}},
               shared_samples=n, t=[float(x) for x in ta[:n]], max_rel_diff=[float(x) for x in worst],
               largest_in=worst_key, per_diagnostic_max={k: float(r.max()) for k, r in rel.items()},
               shared_steps=m, dt_max_rel_diff=dt_rel, s_per_step={"run": sa, "reference": sb})
    path = args.run / f"compare_{mb['run_id']}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
