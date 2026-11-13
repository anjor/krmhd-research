#!/usr/bin/env python
"""Field-aligned spectra of the whole (unfiltered) Alfvenic velocity, magnetic field and density moment on the same
traced field lines, for the comparison with the parallel spectra of Meyrand et al. (2019, figure 3), who found that
the parallel spectra of the density and field-strength fluctuations measured along the perturbed field lines track
that of the Alfvenic velocity.

For each chosen snapshot (snap_XXX.npz: z+-, g) the script traces the same lines as h1_fieldlines.py (same options and
seed), samples along them the E x B velocity u_perp = zhat x grad Phi, the perpendicular magnetic field
b_perp = zhat x grad Psi (both components of each) and the density moment g_0, all unfiltered, and computes their
spectra in the arc length (fieldlines.line_measures; the spectrum of a vector is the sum over its two components).
A passive field frozen to the field lines keeps the field-aligned spectrum of its injection, whereas a field that
inherits the parallel cascade of the Alfvenic fields has a spectrum along the lines like theirs.

Reads <run dir>/snap_XXX.npz and writes <run dir>/alfven_lines_snap_XXX.npz (+ .json summary); re-runnable, does not
touch the simulation data.  By default it uses the snapshots that have a fieldlines_ file.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_alfven_lines.py paper/phase-velocity-slices/data/<run ID> [--snaps 3,5]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import p1lib as P  # noqa: E402  (first: float64 JAX on the CPU)
import fieldlines as FL  # noqa: E402
import numpy as np  # noqa: E402
from h1_fieldlines import DEFAULTS  # noqa: E402  (the tracing options of the field-line analysis)


def analyse_snapshot(box: P.Box, snap: Path, opts: dict) -> tuple[dict, dict]:
    """Spectra along the field lines of u_perp, b_perp and g0 (all unfiltered); returns (arrays, summary)."""
    z = np.load(snap)
    zp, zm = (np.asarray(z[k], dtype=complex) for k in ("zp", "zm"))
    g0 = np.asarray(z["g"][..., 0], dtype=complex)
    t = float(z["t"])
    phi, psi = P.phi_psi(zp, zm)
    t0 = time.time()
    tr = FL.Tracer(box, psi, pad=int(opts["pad"]))
    rng = np.random.default_rng(int(opts["seed"]))
    n = int(opts["n_lines"])
    x0, y0 = rng.uniform(0, box.L, n), rng.uniform(0, box.L, n)
    dz = float(opts["dz"])
    _, _, zz = tr.trace(x0, y0, float(opts["z_len"]), dz)
    arrays: dict = dict(t=t, kappa_par=2 * np.pi * np.fft.rfftfreq(len(zz), d=dz))
    summ: dict = dict(snapshot=snap.name, t=t)
    fields = {"u": (-1j * box.KY * phi, 1j * box.KX * phi),      # zhat x grad Phi
              "b": (-1j * box.KY * psi, 1j * box.KX * psi),      # zhat x grad Psi
              "g0": (g0,)}
    for nm, comps in fields.items():
        E = None
        var = 0.0
        for c in comps:
            lm = FL.line_measures(tr.sample(c), dz)
            E = lm["E_par"] if E is None else E + lm["E_par"]
            var += float(lm["var"])
        arrays[f"{nm}__E_par"] = E
        kap = arrays["kappa_par"]
        summ[nm] = dict(var=var, kappa_1=float(np.sum(kap * E) / E.sum()))
    summ["wall_s"] = time.time() - t0
    return arrays, summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--snaps", default=None, help="comma-separated snapshot indices (default: those with a fieldlines_ file)")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    if not rd.is_absolute():
        rd = P.REPO / rd
    meta = json.load(open(rd / f"{rd.name}.json"))
    cfg = meta["config"]
    opts = dict(DEFAULTS)
    opts.update(cfg.get("analysis", {}).get("fieldlines", {}))
    box = P.Box(int(cfg["box"]["N"]), float(cfg["box"]["L"]))
    if args.snaps:
        snaps = [rd / f"snap_{int(i):03d}.npz" for i in args.snaps.split(",")]
    else:
        snaps = [rd / f.name.replace("fieldlines_", "") for f in sorted(rd.glob("fieldlines_snap_*.npz"))]
    for sp in snaps:
        arrays, summ = analyse_snapshot(box, sp, opts)
        stem = rd / f"alfven_lines_{sp.stem}"
        np.savez_compressed(f"{stem}.npz", **arrays)
        summ["options"] = opts
        with open(f"{stem}.json", "w") as fh:
            json.dump(P._jsonable(summ), fh, indent=1)
        print(f"{sp.name} (t = {summ['t']:.1f}): {summ['wall_s']:.0f} s; kappa_1/k1 along lines: "
              + ", ".join(f"{nm} {summ[nm]['kappa_1'] / box.k1:.2f}" for nm in ("u", "b", "g0")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
