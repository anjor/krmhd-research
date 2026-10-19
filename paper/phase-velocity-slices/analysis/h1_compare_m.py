#!/usr/bin/env python
"""Resolution: forced runs of the beta_i scan against higher-resolution counterparts, in Hermite index (--by M: M = 32
against M = 64, hypercollision coefficient scaled so that the total attenuation is the same and the collisional range
moves from m > 16 to m > 32) or in k_perp (--by N: 64^3 against 128^3 at the same M, hyper-resistivity normalised to
the grid).

For each pair (or series of runs at increasing resolution, --pair beta:<low>:<mid>:...:<high>): the free-energy spectrum W(m)/eps and the perpendicular sink D_perp(m)/W(m), time-averaged over each
run's accepted window (configs/h1_acceptance.yaml, default the final 50 tau_A), with eps the measured injection rate;
the ratio W_64/W_32 over m <= 16 (collision-free in both), 1/T (median of D_perp/W over 2 <= m <= 16), the collisional
share of the free-energy dissipation and the free energy at m > 3M/4.  Writes figures/<name>.{pdf,png} and
data/<name>.json (default name h1_resolution_m).  Reads only saved data.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_compare_m.py --pair 1:<run M=32>:<run M=64> [--pair ...]
    uv run python paper/phase-velocity-slices/analysis/h1_compare_m.py --by N --name h1_resolution_n --pair 0.3:<64^3>:<96^3>:<128^3>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import figstyle as FS  # noqa: E402
import p1lib as P  # noqa: E402
import h1lib as H  # noqa: E402


def averages(run_id: str, acc: dict) -> dict:
    """Window averages of one forced run: W(m)/eps, D_perp(m)/W(m), 1/T, collisional share, W fraction at m > 3M/4."""
    rd = P.DATA / run_id
    with np.load(rd / f"{run_id}.npz") as z:
        ts = {k[4:]: z[k] for k in z.files if k.startswith("ts__")}
    meta = json.load(open(rd / f"{run_id}.json"))
    width = float((acc.get("runs") or {}).get(run_id, {}).get("window", meta["config"]["gates"]["steady_window"]))
    t = ts["t"]
    first = int(np.where(H.window_slice(t, width))[0][0])
    dtk = ts["acc_sum_dt"]
    Tw = float(dtk[first + 1:].sum())
    eps = float(ts["acc_inj_g"][first + 1:].sum() / Tw)
    W = H.time_average(ts["pWm"], dtk, first)
    Dp = ts["acc_loss_eta"][first + 1:].sum(0) / Tw
    Dn = ts["acc_loss_nu"][first + 1:].sum(0) / Tw
    M = len(W) - 1
    q = Dp / W
    m = np.arange(M + 1)
    inv_T = float(np.median(q[(m >= 2) & (m <= 16)]))
    tau_nl = None
    if (rd / "h1_analysis.json").exists():
        tau_nl = json.load(open(rd / "h1_analysis.json")).get("outer", {}).get("tau_nl")
    return dict(run_id=run_id, M=M, N=int(meta["config"]["box"]["N"]), tau_nl=tau_nl,
                T_over_tau_nl=(1.0 / inv_T / tau_nl) if tau_nl else None, window=[float(t[first]), float(t[-1])], eps=eps, W_over_eps=(W / eps).tolist(),
                Dperp_over_W=q.tolist(), inv_T=inv_T,
                nu_share=float(Dn.sum() / (Dn.sum() + Dp.sum())), W_frac_high=float(W[3 * M // 4 + 1:].sum() / W.sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pair", action="append", required=True,
                    help="beta_i:<run ID at low resolution>[:<intermediate> ...]:<run ID at high resolution>")
    ap.add_argument("--acceptance", default="h1_acceptance.yaml")
    ap.add_argument("--outdir", default=None, help="figure folder (default paper/phase-velocity-slices/figures)")
    ap.add_argument("--by", choices=("M", "N"), default="M", help="what the pair differs in (labels)")
    ap.add_argument("--name", default="h1_resolution_m", help="basename of the figure and of the JSON summary")
    args = ap.parse_args()
    acc = P.load_config(args.acceptance)
    out = {}
    for spec in args.pair:
        b, *ids = spec.split(":")
        if len(ids) < 2:
            raise SystemExit(f"--pair {spec}: give at least two runs")
        series = [averages(r, acc) for r in ids]
        a32, a64 = series[0], series[-1]
        w32, w64 = np.array(a32["W_over_eps"]), np.array(a64["W_over_eps"])
        ratio = w64[1:17] / w32[1:17]
        out[b] = dict(low=a32, high=a64, series=series, W_high_over_low_m1_to_16=ratio.tolist(),
                      max_abs_log_ratio_m1_to_16=float(np.abs(np.log(ratio)).max()))
        inv_t = " -> ".join(f"{a['inv_T']:.3f}" for a in series)
        t_tau = " -> ".join("-" if a["T_over_tau_nl"] is None else f"{a['T_over_tau_nl']:.2f}" for a in series)
        share = " -> ".join(f"{a['nu_share']:.3f}" for a in series)
        print(f"beta_i = {b}: W_high/W_low (m = 1..16) {ratio.min():.3f}..{ratio.max():.3f}; 1/T {inv_t}; "
              f"T/tau_nl {t_tau}; collisional share {share}; "
              f"W at m > 3M/4 {a32['W_frac_high']:.4f} -> {a64['W_frac_high']:.4f}")
    (P.DATA / f"{args.name}.json").write_text(json.dumps(P._jsonable(out), indent=1))

    fcfg = FS.style_config()
    plt = FS.setup_style(fcfg, usetex=True)
    pal = fcfg["palette"]
    fig, ax = plt.subplots(1, 2, figsize=(fcfg["double_width_in"], 2.6))
    for i, (b, d) in enumerate(sorted(out.items(), key=lambda kv: -float(kv[0]))):
        c = pal[i % len(pal)]
        series = d["series"]
        styles = ["--", ":", "-."][:len(series) - 1] + ["-"]
        for a, ls in zip(series, styles):
            lab = rf"$M = {a['M']}$" if args.by == "M" else rf"${a['N']}^3$"
            m = np.arange(a["M"] + 1)
            ax[0].loglog(m[1:], np.array(a["W_over_eps"])[1:], ls, color=c, lw=1.0,
                         label=rf"$\beta_i = {b}$, {lab}")
            ax[1].semilogx(m[1:], np.array(a["Dperp_over_W"])[1:], ls, color=c, lw=1.0)
    ax[0].set_xlabel(r"$m$")
    ax[0].set_ylabel(r"$W(m)/\varepsilon$")
    ax[0].legend(fontsize=7, frameon=False)
    ax[1].set_xlabel(r"$m$")
    ax[1].set_ylabel(r"$D_\perp(m)/W(m)$")
    ax[1].set_ylim(0, 1.5)
    fig.subplots_adjust(wspace=0.32)
    for a, lab in zip(ax, "ab"):
        a.text(0.0, 1.02, f"({lab})", transform=a.transAxes)
    outdir = Path(args.outdir) if args.outdir else FS.FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{args.name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{args.name}.png", dpi=fcfg["png_dpi"], bbox_inches="tight")
    print(f"wrote {outdir / args.name}.pdf/.png and {P.DATA / (args.name + '.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
