#!/usr/bin/env python
"""Field-aligned (parallel) structure of the compressive and Alfvenic fields in saved snapshots of a forced run
(scripts/run_forced_turbulence.py): the discriminating measurement between H1 (the parallel structure of g is frozen
at the injected field-aligned wavenumber; S09 section 6.3) and a critically balanced parallel cascade of g inherited
from the Alfvenic field (Meyrand et al. 2019).

For each chosen snapshot (snap_XXX.npz: z+-, g in complex64): trace n_lines field lines of b = zhat + zhat x grad Psi
over z_len box lengths with RK4 (scripts/fieldlines.py; cubic splines on a grid refined by 2), sample along them and
compute the parallel spectrum E_par(kappa), the autocorrelation C_par(l) and the wavenumbers kappa_1 (first moment of
|kappa|), kappa_half (pi/(3 l_half)) for
  * g_0, Phi and Psi band-passed in k_perp L/2pi shells (config analysis.kperp_bands): kappa(k_perp) of g against
    the Alfvenic kappa(k_perp) ~ k_perp^(2/3) of critical balance;
  * every moment g_m (unfiltered): kappa_1(m), used for the no-echo flux s sqrt(2(m+1)) kappa_1(m) W(m);
  * selected slices a_j (u_j nearest analysis.slices_u);
  * the injected pattern: fresh draws of the run's density forcing (GANDALF's generator) on the same lines, whose
    field-aligned spectrum is the kappa distribution of the two-channel model.
The same measures of the Eulerian k_z structure come from the Fourier coefficients.  Results go to
<run dir>/fieldlines_<snap>.npz (+ .json summary); the script is re-runnable and does not touch the simulation data.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_fieldlines.py paper/phase-velocity-slices/data/<run ID> [--snaps 7,8,9]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
import p1lib as P  # noqa: E402  (first: float64 JAX on the CPU)
import fieldlines as FL  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
from krmhd.forcing import gaussian_white_noise_fourier_perp_lowkz, gaussian_white_noise_fourier  # noqa: E402

DEFAULTS = dict(n_lines=4096, z_len=4.0, dz=1.0 / 128, pad=2, seed=11,
                kperp_bands=[[0, 2], [2, 4], [4, 8], [8, 16]], slices_u=[0.0, 1.0, -1.0, 2.0, -2.0, 3.0],
                n_forcing_draws=4)


def summary(meas: dict) -> dict:
    return {k: float(meas[k]) for k in ("kappa_1", "kappa_half", "kappa_rms", "var") if k in meas}


def forcing_draws(box: P.Box, fcfg: dict, n: int, seed: int) -> list[np.ndarray]:
    """Fresh realisations of the run's density forcing pattern (unit amplitude, GANDALF's generator)."""
    d = fcfg["density"]
    key = jax.random.PRNGKey(seed)
    out = []
    for _ in range(n):
        if d.get("generator", "perp_lowkz") == "perp_lowkz":
            F, key = gaussian_white_noise_fourier_perp_lowkz(box.grid, 1.0, int(d["n_min"]), int(d["n_max"]),
                                                             int(d.get("max_nz", 1)), bool(d.get("include_nz0", False)),
                                                             1.0, key)
        else:
            F, key = gaussian_white_noise_fourier(box.grid, 1.0, int(d["n_min"]), int(d["n_max"]), 1.0, key)
        out.append(np.asarray(F, dtype=complex))
    return out


def analyse_snapshot(box: P.Box, sl: P.Slices, cfg: dict, snap: Path, opts: dict) -> tuple[dict, dict]:
    z = np.load(snap)
    zp, zm, g = (np.asarray(z[k], dtype=complex) for k in ("zp", "zm", "g"))
    t = float(z["t"])
    phi, psi = P.phi_psi(zp, zm)
    t0 = time.time()
    tr = FL.Tracer(box, psi, pad=int(opts["pad"]))
    rng = np.random.default_rng(int(opts["seed"]))
    n = int(opts["n_lines"])
    x0, y0 = rng.uniform(0, box.L, n), rng.uniform(0, box.L, n)
    dz = float(opts["dz"])
    X, Y, zz = tr.trace(x0, y0, float(opts["z_len"]), dz)
    t_trace = time.time() - t0
    arrays: dict = dict(t=t, x0=x0, y0=y0, wander_rms=np.sqrt(np.mean((X[-1] - x0) ** 2 + (Y[-1] - y0) ** 2)))
    summ: dict = dict(snapshot=snap.name, t=t, n_lines=n, z_len=float(opts["z_len"]), dz=dz, b_rms=tr.b_rms,
                      trace_s=t_trace)

    def do(name: str, f: np.ndarray) -> dict:
        lm = FL.line_measures(tr.sample(f), dz)
        eu = FL.eulerian_measures(box, f, lm["lags"])
        arrays[f"{name}__E_par"] = lm["E_par"]
        arrays[f"{name}__C_par"] = lm["C_par"]
        arrays[f"{name}__E_z"] = eu["E_z"]
        arrays[f"{name}__C_z"] = eu["C_z"]
        fa = box.gradpar(psi, f)
        krms_fa = math.sqrt(box.avg2(fa) / box.avg2(f)) if box.avg2(f) > 0 else float("nan")
        out = dict(par=summary(lm), z=summary(eu), kappa_rms_spectral_fa=krms_fa)
        summ[name] = out
        return out

    arrays["kappa_par"] = 2 * np.pi * np.fft.rfftfreq(len(zz), d=dz)
    arrays["lags"] = dz * np.arange(len(zz) // 2 + 1)
    arrays["kz"] = box.k1 * np.arange(int(box.nz.max()) + 1)
    bands = [tuple(b) for b in opts["kperp_bands"]]
    for i, (lo, hi) in enumerate(bands):
        for nm, f in (("g0", g[..., 0]), ("phi", phi), ("psi", psi)):
            do(f"{nm}_band{i}", FL.bandpass(box, f, lo, hi))
    M = sl.M
    k1m = np.zeros(M + 1)
    kzm = np.zeros(M + 1)
    for m in range(M + 1):
        r = do(f"g{m}", g[..., m])
        k1m[m] = r["par"]["kappa_1"]
        kzm[m] = r["z"]["kappa_1"]
    arrays["kappa1_m_par"] = k1m
    arrays["kappa1_m_z"] = kzm
    a = sl.amplitudes(g)
    js = sorted(set(sl.nearest(u) for u in opts["slices_u"]))
    arrays["slice_j"] = np.array(js)
    for j in js:
        do(f"slice{j}", a[..., j])
    # injected pattern: average spectrum over fresh draws
    Es, Ez = [], []
    for F in forcing_draws(box, cfg["forcing"], int(opts["n_forcing_draws"]), int(opts["seed"]) + 1):
        lm = FL.line_measures(tr.sample(F), dz)
        Es.append(lm["E_par"] / lm["E_par"].sum())
        Ez.append(FL.eulerian_measures(box, F, lm["lags"])["E_z"])
    EF = np.mean(Es, axis=0)
    arrays["F__E_par"] = EF
    arrays["F__E_z"] = np.mean(Ez, axis=0)
    kap = arrays["kappa_par"]
    summ["F"] = dict(par=dict(kappa_1=float(np.sum(kap * EF) / EF.sum()),
                              kappa_rms=float(math.sqrt(np.sum(kap ** 2 * EF) / EF.sum()))))
    summ["bands"] = bands
    summ["slices"] = [dict(j=int(j), u=float(sl.u[j])) for j in js]
    summ["wall_s"] = time.time() - t0
    return arrays, summ


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--snaps", default=None, help="comma-separated snapshot indices (default: the last three)")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    if not rd.is_absolute():
        rd = P.REPO / rd
    meta = json.load(open(rd / f"{rd.name}.json"))
    cfg = meta["config"]
    opts = dict(DEFAULTS)
    opts.update(cfg.get("analysis", {}).get("fieldlines", {}))
    v = cfg["velocity"]
    sl = P.make_slices(int(v["M"]), P.parse_number(v["Lambda"]), v["closure"])
    box = P.Box(int(cfg["box"]["N"]), float(cfg["box"]["L"]))
    snaps = sorted(rd.glob("snap_*.npz"))
    if args.snaps:
        snaps = [rd / f"snap_{int(i):03d}.npz" for i in args.snaps.split(",")]
    else:
        snaps = snaps[-3:]
    for sp in snaps:
        arrays, summ = analyse_snapshot(box, sl, cfg, sp, opts)
        stem = rd / f"fieldlines_{sp.stem}"
        np.savez_compressed(f"{stem}.npz", **arrays)
        summ["options"] = opts
        with open(f"{stem}.json", "w") as fh:
            json.dump(P._jsonable(summ), fh, indent=1)
        k1 = box.k1
        print(f"{sp.name} (t = {summ['t']:.3f}): {summ['n_lines']} lines x {summ['z_len']} L, b_rms = "
              f"{summ['b_rms']:.3f}, trace {summ['trace_s']:.0f} s, total {summ['wall_s']:.0f} s", flush=True)
        for i, b in enumerate(summ["bands"]):
            row = "  ".join(f"{nm}: par {summ[f'{nm}_band{i}']['par']['kappa_1'] / k1:.2f}/"
                            f"{summ[f'{nm}_band{i}']['par']['kappa_half'] / k1:.2f}, z "
                            f"{summ[f'{nm}_band{i}']['z']['kappa_1'] / k1:.2f}" for nm in ("g0", "phi", "psi"))
            print(f"   k_perp L/2pi in {b}: kappa_1/kappa_half (units of k1)  {row}")
        print(f"   injected pattern: kappa_1 = {summ['F']['par']['kappa_1'] / k1:.2f} k1 along lines")
        print("   kappa_1(m)/k1 along lines: " + " ".join(f"{x / k1:.2f}" for x in arrays["kappa1_m_par"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
