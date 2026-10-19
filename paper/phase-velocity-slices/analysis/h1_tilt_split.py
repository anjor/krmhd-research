#!/usr/bin/env python
"""Prediction (v): the backward ("unmixing") flux that a split of the k_z-only Hermite flux by the sign of k_z reports
when phase mixing follows the field-aligned wavenumber and nothing returns.

If the free energy at moment m phase mixes at field-aligned wavenumber kappa while the split sorts it by the Eulerian
k_z = kappa - delta, with delta = b_perp . k_perp the tilt term, the part with sign(k_z) != sign(kappa) is counted as
unmixing, with weight |k_z|.  For kappa at the rms field-aligned wavenumber of g_m and delta Gaussian with variance
k_z,rms^2 - kappa^2 (k_z,rms the rms Eulerian wavenumber of g_m), the expected ratio of the unmixing to the
phase-mixing part is
    R = [sigma phi(z) - kappa (1 - Phi(z))] / [kappa Phi(z) + sigma phi(z)],   z = kappa/sigma,
with phi and Phi the standard normal density and distribution.  The script compares R(m) with the measured ratio
-Pi_minus/Pi_plus (h1_analysis.json, P_e) for each run given, prints the comparison and writes
figures/h1_tilt_split.{pdf,png} and data/h1_tilt_split.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_tilt_split.py <run> [<run> ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import figstyle as FS  # noqa: E402
import p1lib as P  # noqa: E402


def ratio(kappa: float, sigma: float) -> float:
    """E[|k|; k < 0] / E[|k|; k > 0] for k = kappa - delta, delta ~ N(0, sigma^2)."""
    if sigma <= 0:
        return 0.0
    z = kappa / sigma
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return (sigma * phi - kappa * (1 - Phi)) / (kappa * Phi + sigma * phi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    out = {}
    for r in args.runs:
        d = json.load(open(P.DATA / r / "h1_analysis.json"))
        meta = json.load(open(P.DATA / r / f"{r}.json"))
        kfa = np.array(d["kappa"]["kfa_rms_m_over_k1"])
        kz = np.array(d["kappa"]["kz_rms_m_over_k1"])
        ms = sorted(int(k) for k in d["P_e"] if k.isdigit() and int(k) >= 1)
        pred = [ratio(kfa[m], math.sqrt(max(kz[m] ** 2 - kfa[m] ** 2, 0.0))) for m in ms]
        meas = [float(d["P_e"][str(m)]["minus_over_plus"]) for m in ms]
        cfg = meta["config"]
        out[r] = dict(beta=P.parse_number(cfg["velocity"]["beta_i"]), M=int(cfg["velocity"]["M"]),
                      N=int(cfg["box"]["N"]), m=ms, predicted=pred, measured=meas,
                      kfa_rms=[float(kfa[m]) for m in ms], kz_rms=[float(kz[m]) for m in ms])
        print(f"{r}: " + "; ".join(f"m={m} pred {p:.3f} meas {q:.3f}" for m, p, q in zip(ms, pred, meas)))
    (P.DATA / "h1_tilt_split.json").write_text(json.dumps(P._jsonable(out), indent=1))

    fcfg = FS.style_config()
    plt = FS.setup_style(fcfg, usetex=True)
    pal = fcfg["palette"]
    fig, a = plt.subplots(1, 1, figsize=(fcfg["single_width_in"], 2.7))
    for i, (r, d) in enumerate(out.items()):
        c = pal[i % len(pal)]
        lab = rf"$\beta_i={d['beta']:g}$, ${d['N']}^3$, $M={d['M']}$"
        a.plot(d["m"], d["measured"], "o", ms=3, color=c, label=lab)
        a.plot(d["m"], d["predicted"], "-", lw=0.9, color=c)
    a.set_xscale("log")
    a.set_xlabel(r"$m$")
    a.set_ylabel(r"$-\Pi^-_{m+1/2}/\Pi^+_{m+1/2}$")
    a.set_ylim(0, None)
    a.legend(frameon=False, fontsize=6)
    outdir = Path(args.outdir) if args.outdir else FS.FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "h1_tilt_split.pdf", bbox_inches="tight")
    fig.savefig(outdir / "h1_tilt_split.png", dpi=fcfg["png_dpi"], bbox_inches="tight")
    print(f"wrote {outdir / 'h1_tilt_split'}.pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
