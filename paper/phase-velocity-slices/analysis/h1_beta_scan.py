#!/usr/bin/env python
"""The beta_i scan at fixed chi: Hermite spectra against the two-channel model, and the perpendicular sink against the
time since injection.

Reads <run>/<run>.npz, .json and h1_analysis.json (analysis/h1_pilot.py) of each run given; writes
figures/h1_beta.{pdf,png} and data/h1_beta_scan.json.
  (a) W(m)/eps (lines) and the two-channel model with the traced P(kappa) of the injected pattern and the measured
      T (dashed), for the runs given with --spectrum;
  (b) D_perp(m)/W(m) against the age a = sqrt(2m)/(s kappa_1) of the free energy at m, with kappa_1 the mean
      field-aligned wavenumber of the injected pattern, for every run given with --spectrum or --age, over
      1 <= m <= M/2 (below the collisional range).  A joint fit of the saturating form q = g1 - (g1 - g0) exp(-a/a_c),
      the form of the age-dependent sink of h1_age_model.py, is drawn and reported with its rms relative residual, and
      the same form against m for comparison.  The age a is in units of tau_A.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_beta_scan.py --spectrum <run> [...] [--age <run> ...]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import figstyle as FS  # noqa: E402
import p1lib as P  # noqa: E402
import h1lib as H  # noqa: E402

MODEL = "traced P(kappa) of the injected pattern"


def saturating(x: np.ndarray, g1: float, g0: float, xc: float) -> np.ndarray:
    """Sink rate D_perp/W (per tau_A) rising from g0 at x = 0 to g1 over the scale xc: g1 - (g1 - g0) exp(-x/xc)."""
    return g1 - (g1 - g0) * np.exp(-x / xc)


def fit_saturating(x: np.ndarray, q: np.ndarray) -> dict:
    """Least-squares fit of saturating(x) to q, with the rms relative residual (q - fit)/q."""
    p0 = (float(q.max()), float(q.min()), float(np.median(x)))
    (g1, g0, xc), _ = curve_fit(saturating, x, q, p0=p0, maxfev=20000)
    rel = (q - saturating(x, g1, g0, xc)) / q
    return dict(g1=float(g1), g0=float(g0), xc=float(xc), rms_rel_resid=float(np.sqrt(np.mean(rel ** 2))))


def load(run_id: str, acc: dict) -> dict:
    rd = P.DATA / run_id
    meta = json.load(open(rd / f"{run_id}.json"))
    an = json.load(open(rd / "h1_analysis.json"))
    with np.load(rd / f"{run_id}.npz") as z:
        ts = {k[4:]: z[k] for k in z.files if k.startswith("ts__")}
    t = ts["t"]
    width = float((acc.get("runs") or {}).get(run_id, {}).get("window", meta["config"]["gates"]["steady_window"]))
    first = int(np.where(H.window_slice(t, width))[0][0])
    dtk = ts["acc_sum_dt"]
    Tw = float(dtk[first + 1:].sum())
    eps = float(ts["acc_inj_g"][first + 1:].sum() / Tw)
    W = H.time_average(ts["pWm"], dtk, first)
    beta = P.parse_number(meta["config"]["velocity"]["beta_i"])
    M = len(W) - 1
    k1 = 2 * math.pi / float(meta["config"]["box"]["L"])
    kap1 = float(an["fieldlines"]["kappa1_F_over_k1"]) * k1
    ratio = np.array(an["P_b"][MODEL]["W_over_model_m"])
    return dict(run_id=run_id, beta=beta, M=M, eps=eps, W=W, model=W / ratio, q=np.array(an["P_a"]["Dperp_over_W"]),
                inv_T=float(an["P_a"]["inv_T"]), s=math.sqrt(beta), kappa1=kap1, k1=k1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spectrum", nargs="+", required=True, help="runs shown in (a) and (b)")
    ap.add_argument("--age", nargs="*", default=[], help="further runs shown in (b) only")
    ap.add_argument("--acceptance", default="h1_acceptance.yaml")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    acc = P.load_config(args.acceptance)
    spec = [load(r, acc) for r in args.spectrum]
    allruns = spec + [load(r, acc) for r in args.age]

    # collapse of the sink on the age of the free energy
    rows = []
    for d in allruns:
        for m in range(1, d["M"] // 2 + 1):
            rows.append((math.sqrt(2 * m) / (d["s"] * d["kappa1"]), d["q"][m], m))
    rows = np.array(rows)
    out = {}
    for name, x in (("age", rows[:, 0]), ("m", rows[:, 2])):
        f = out[f"fit_sat_{name}"] = fit_saturating(x, rows[:, 1])
        print(f"D_perp/W = {f['g1']:.3f} - ({f['g1']:.3f} - {f['g0']:.3f}) exp(-{name}/{f['xc']:.3f}): "
              f"rms relative residual {f['rms_rel_resid']:.3f}")
    out["runs"] = [dict(run_id=d["run_id"], beta=d["beta"], M=d["M"], inv_T=d["inv_T"], kappa1_over_k1=d["kappa1"] / d["k1"],
                        W_over_model_max_dev_m1_to_half=float(np.abs(np.log(d["W"][1:d["M"] // 2 + 1] /
                                                                             d["model"][1:d["M"] // 2 + 1])).max()))
                   for d in allruns]
    for r in out["runs"]:
        print(f"{r['run_id']}: beta {r['beta']}, M {r['M']}, 1/T {r['inv_T']:.3f}, kappa_1/k1 {r['kappa1_over_k1']:.3f}, "
              f"max |ln W/model| (m <= M/2) {r['W_over_model_max_dev_m1_to_half']:.3f}")
    (P.DATA / "h1_beta_scan.json").write_text(json.dumps(P._jsonable(out), indent=1))

    fcfg = FS.style_config()
    plt = FS.setup_style(fcfg, usetex=True)
    pal = fcfg["palette"]
    fig, ax = plt.subplots(1, 2, figsize=(fcfg["double_width_in"], 2.6))
    betas = sorted({d["beta"] for d in allruns}, reverse=True)
    col = {b: pal[i % len(pal)] for i, b in enumerate(betas)}
    for d in spec:
        m = np.arange(d["M"] + 1)
        c = col[d["beta"]]
        ax[0].loglog(m[1:], d["W"][1:] / d["eps"], "-", color=c, lw=1.1,
                     label=rf"$\beta_i = {d['beta']:g}$, $M = {d['M']}$")
        ax[0].loglog(m[1:], d["model"][1:] / d["eps"], "--", color=c, lw=0.9)
    ax[0].loglog([], [], "k--", lw=0.9, label="two-channel model")
    ax[0].set_xlabel(r"$m$")
    ax[0].set_ylabel(r"$W(m)/\varepsilon$")
    ax[0].legend(fontsize=7, frameon=False)
    for d in allruns:
        m = np.arange(1, d["M"] // 2 + 1)
        a = np.sqrt(2 * m) / (d["s"] * d["kappa1"])
        ax[1].semilogx(a, d["q"][m], "-" if d["M"] > 32 else ":", color=col[d["beta"]], lw=1.0,
                       label=rf"$\beta_i = {d['beta']:g}$, $M = {d['M']}$")
    f = out["fit_sat_age"]
    aa = np.geomspace(rows[:, 0].min(), rows[:, 0].max(), 50)
    ax[1].semilogx(aa, saturating(aa, f["g1"], f["g0"], f["xc"]), "k-", lw=0.6)
    ax[1].set_xlabel(r"$\sqrt{2m}/(s\kappa_1)$ $(\tau_A)$")
    ax[1].set_ylabel(r"$D_\perp(m)/W(m)$")
    ax[1].set_ylim(0, 0.9)
    ax[1].legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.subplots_adjust(wspace=0.32)
    for a_, lab in zip(ax, "ab"):
        a_.text(0.0, 1.02, f"({lab})", transform=a_.transAxes)
    outdir = Path(args.outdir) if args.outdir else FS.FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "h1_beta.pdf", bbox_inches="tight")
    fig.savefig(outdir / "h1_beta.png", dpi=fcfg["png_dpi"], bbox_inches="tight")
    print(f"wrote {outdir / 'h1_beta'}.pdf/.png and {P.DATA / 'h1_beta_scan.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
