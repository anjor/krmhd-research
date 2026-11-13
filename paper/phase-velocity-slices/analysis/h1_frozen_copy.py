#!/usr/bin/env python
"""Field-aligned structure of k_perp-filtered fields: g_0 against a copy of g_0 that is frozen to the field lines by
construction (control for the band-resolved field-aligned wavenumbers of h1_fieldlines.py).

Filtering a field into k_perp bands and then sampling it along field lines mixes neighbouring lines, which separate
along the field, so a filtered field can show field-aligned structure that the full field does not have.  To measure
this effect, the script builds from each snapshot a field f that equals g_0 on the planes z_n = n L/n_planes and,
between two neighbouring planes, is the blend

    f(x, z) = cos^2(pi d / 2a) g_0(X_lo, z_lo) + sin^2(pi d / 2a) g_0(X_hi, z_hi),   a = L/n_planes,  d = z - z_lo,

where X_lo and X_hi are the points at which the field line through (x, z) crosses the planes z_lo <= z < z_hi.  Along
any field line f is a smooth interpolation of the values of g_0 at the plane crossings, so it has no field-aligned
structure on scales shorter than the plane spacing (field-aligned wavenumbers above about pi/a) except that of the
cos^2 blend, while its perpendicular structure is that of g_0.  The script then traces the same lines as
h1_fieldlines.py (same options and seed), samples f, g_0 and Phi along them, whole and filtered into the same k_perp
bands, and stores the spectra along the lines.  If the band-resolved wavenumbers of f grow with k_perp as those of g_0
do, that growth comes from the filtering and not from structure of g_0 along the field.

Reads <run dir>/snap_XXX.npz (z+-, g in complex64) and writes <run dir>/frozen_copy_snap_XXX.npz (+ .json summary);
re-runnable, does not touch the simulation data.  By default it uses the snapshots that have a fieldlines_ file.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_frozen_copy.py paper/phase-velocity-slices/data/<run ID> [--snaps 3,5]
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
from scipy import ndimage  # noqa: E402
from h1_fieldlines import DEFAULTS  # noqa: E402  (the tracing options of the field-line analysis)


def frozen_copy(box: P.Box, tr: FL.Tracer, g0: np.ndarray, n_planes: int, dz: float) -> np.ndarray:
    """rfft coefficients of the frozen copy f of g0 (rfft coefficients of the density moment) on box.

    Each grid point is traced along b = zhat + zhat x grad Psi (tr) to the planes z_lo = n a <= z and z_hi = z_lo + a
    (a = L / n_planes) with RK4 steps of at most dz, and f is the cos^2 blend of g0 at the two crossing points."""
    N, L = box.N, box.L
    a = L / n_planes
    if N % n_planes:
        raise ValueError(f"N = {N} is not divisible by n_planes = {n_planes}")
    coef = ndimage.spline_filter(box.to_real(g0, tr.pad), order=3, mode="grid-wrap")
    xs = np.arange(N) * L / N
    X0, Y0 = np.meshgrid(xs, xs, indexing="xy")          # [y, x], as the real-space arrays [z, y, x]
    x0, y0 = X0.ravel(), Y0.ravel()
    f = np.empty((N, N, N))
    for k in range(N):
        zk = k * L / N
        zlo = math.floor(zk / a + 1e-9) * a
        vals = []
        for zt in (zlo, zlo + a):
            dist = zt - zk
            if abs(dist) < 1e-12:
                xe, ye = x0, y0
            else:
                ns = math.ceil(abs(dist) / dz - 1e-9)
                X, Y, _ = tr.trace(x0, y0, dist, dist / ns, z0=zk)
                xe, ye = X[-1], Y[-1]
            vals.append(tr._interp(coef, xe, ye, zt))
        w = math.cos(math.pi * (zk - zlo) / (2 * a)) ** 2
        f[k] = (w * vals[0] + (1.0 - w) * vals[1]).reshape(N, N)
    return box.from_real(f)


def analyse_snapshot(box: P.Box, snap: Path, opts: dict, n_planes: int) -> tuple[dict, dict]:
    """Spectra along the field lines of f, g0 and Phi, whole and in the k_perp bands of opts; returns (arrays, summary)."""
    z = np.load(snap)
    zp, zm = (np.asarray(z[k], dtype=complex) for k in ("zp", "zm"))
    g0 = np.asarray(z["g"][..., 0], dtype=complex)
    t = float(z["t"])
    phi, psi = P.phi_psi(zp, zm)
    t0 = time.time()
    tr = FL.Tracer(box, psi, pad=int(opts["pad"]))
    dz = float(opts["dz"])
    f = frozen_copy(box, tr, g0, n_planes, dz)
    t_build = time.time() - t0
    rng = np.random.default_rng(int(opts["seed"]))
    n = int(opts["n_lines"])
    x0, y0 = rng.uniform(0, box.L, n), rng.uniform(0, box.L, n)
    _, _, zz = tr.trace(x0, y0, float(opts["z_len"]), dz)
    arrays: dict = dict(t=t, kappa_par=2 * np.pi * np.fft.rfftfreq(len(zz), d=dz))
    summ: dict = dict(snapshot=snap.name, t=t, n_planes=n_planes, var_f_over_var_g0=box.avg2(f) / box.avg2(g0),
                      build_s=t_build)
    bands = [tuple(b) for b in opts["kperp_bands"]]
    for nm, fld in (("f", f), ("g0", g0), ("phi", phi)):
        lm = FL.line_measures(tr.sample(fld), dz)
        arrays[f"{nm}__E_par"] = lm["E_par"]
        summ[nm] = dict(kappa_1=float(lm["kappa_1"]), var=float(lm["var"]))
        for i, (lo, hi) in enumerate(bands):
            lb = FL.line_measures(tr.sample(FL.bandpass(box, fld, lo, hi)), dz)
            arrays[f"{nm}_band{i}__E_par"] = lb["E_par"]
            summ[f"{nm}_band{i}"] = dict(kappa_1=float(lb["kappa_1"]), var=float(lb["var"]))
    summ["bands"] = bands
    summ["wall_s"] = time.time() - t0
    return arrays, summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--snaps", default=None, help="comma-separated snapshot indices (default: those with a fieldlines_ file)")
    ap.add_argument("--planes", type=int, default=4, help="number of planes z_n per box length (default 4)")
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
    k1 = box.k1
    for sp in snaps:
        arrays, summ = analyse_snapshot(box, sp, opts, args.planes)
        stem = rd / f"frozen_copy_{sp.stem}"
        np.savez_compressed(f"{stem}.npz", **arrays)
        summ["options"] = opts
        with open(f"{stem}.json", "w") as fh:
            json.dump(P._jsonable(summ), fh, indent=1)
        print(f"{sp.name} (t = {summ['t']:.1f}): build {summ['build_s']:.0f} s, total {summ['wall_s']:.0f} s, "
              f"var(f)/var(g0) = {summ['var_f_over_var_g0']:.3f}", flush=True)
        for nm in ("f", "g0", "phi"):
            print(f"   {nm:4s} kappa_1/k1 whole {summ[nm]['kappa_1'] / k1:.2f}; bands " +
                  " ".join(f"{summ[f'{nm}_band{i}']['kappa_1'] / k1:.2f}" for i in range(len(summ['bands']))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
