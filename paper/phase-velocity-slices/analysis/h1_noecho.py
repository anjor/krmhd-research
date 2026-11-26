#!/usr/bin/env python
"""Result 1 for every run: the field-aligned flux over the no-echo flux, Pi_{m+1/2}/(s c_m kappa_1(m) W(m)), against m
(figure h1_noecho, second referee read item 2.2).

The ratio is 1 if all the free energy at m phase mixes at its measured field-aligned wavenumber kappa_1(m) with no
return flux (c_m: the exact coefficient of linear phase mixing, h1lib.phase_mixing_coefficient).  The uniform-sink
model with no return flux gives the same ratio a little below 1, because the sink removes free energy while it phase
mixes; it is drawn as a thin line of the run's colour.  A return flux would make the measured ratio fall as m grows.
The ratio drawn is "paired": Pi and W are taken at the snapshot times at which kappa_1(m) is measured, so that the
three come from the same instants.  The ratio with Pi and W averaged over the whole window and kappa_1(m) from the
snapshots mixes a long time average with a few snapshots; it is drawn faint for comparison, and it is low where W(m)
is intermittent (the dip at m = 35 at beta_i = 0.05).
Each curve runs from the moment above the forced one to m = M/2.  It is drawn solid up to m_c, the top of the
collision-free range of the rule of h1_mrange.py (the largest m with 2 nu_m <= 0.13/T, T the median of D_perp/W over
2 <= m <= m_c), and faded from m_c to M/2, where the hypercollisions act.
(a) The runs of table 2 at 64^3; (b) the runs at beta_i = 0.3 and 1 against N (64^3 to 256^3); (c) the runs of the
m_* scan (table 3).

Reads <run>/h1_analysis.json (no_echo.Pi_fa_over_noecho, no_echo.model_Pi_over_noecho, no_echo.coefficient_c_m,
P_a.Dperp_over_W, outer.chi_out), <run>/<run>.json (config: velocity.M, velocity.beta_i, box.N, stepper.nu,
stepper.hyper_n, forcing.density.moment), <run>/<run>.npz (ts__t, ts__Pi_fa, ts__pWm) and
<run>/fieldlines_snap_*.npz (t, kappa1_m_par).
Writes figures/h1_noecho.{pdf,png} and data/h1_noecho.json (the ratios, m_c and their ranges over 2 <= m <= m_c and
8 <= m <= M/2).

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_noecho.py
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
from h1_block_spreads import RUNS  # noqa: E402
from h1_mrange import m_c_rule  # noqa: E402

PANELS = {  # panel: [(label, colour index, marker filled)], with the legend text built from the run
    "a": [("b1", 0, False), ("b03", 1, False), ("b01", 2, False), ("b1_M64", 0, True), ("b03_M64", 1, True),
          ("b01_M64", 2, True), ("b1_mf1", 3, True)],
    "b": [("b03_M64", 0, True), ("b03_N96", 1, True), ("b03_N128", 2, True), ("b03_N256", 3, True),
          ("b1_M64", 0, False), ("b1_N128", 2, False)],
    "c": [("b005", 0, True), ("b01_M64", 1, True), ("b03_M192", 2, True), ("chi075", 3, True), ("chi056", 4, True),
          ("chi040", -1, True)],
}


def load(label: str) -> dict:
    rid = RUNS[label]
    an = json.load(open(P.DATA / rid / "h1_analysis.json"))
    cfg = json.load(open(P.DATA / rid / f"{rid}.json"))["config"]
    v, st = cfg["velocity"], cfg["stepper"]
    M = int(v["M"])
    q = np.array(an["P_a"]["Dperp_over_W"], dtype=float)
    rule = m_c_rule(q, M, P.parse_number(st["nu"]), int(st["hyper_n"]))
    ratio = np.array(an["no_echo"]["Pi_fa_over_noecho"], dtype=float)
    model = np.array(an["no_echo"]["model_Pi_over_noecho"], dtype=float)
    mf = int(cfg["forcing"]["density"].get("moment", 0))
    m = np.arange(M)

    def rng(lo: int, hi: int, x: np.ndarray) -> list[float]:
        s = x[(m >= lo) & (m <= hi)]
        return [float(s.min()), float(s.max())]
    # the same ratio with Pi and W taken at the snapshot times of the traced lines (nearest sample), so that the
    # flux, W and kappa_1(m) come from the same instants: sum_s Pi(t_s) / (s c_m sum_s kappa_1(m; t_s) W(t_s))
    with np.load(P.DATA / rid / f"{rid}.npz") as z:
        t, Pi_t, W_t = z["ts__t"], z["ts__Pi_fa"], z["ts__pWm"]
    c_m = np.array(an["no_echo"]["coefficient_c_m"], dtype=float)
    snaps = [np.load(f) for f in sorted((P.DATA / rid).glob("fieldlines_snap_*.npz"))]
    idx = [int(np.argmin(np.abs(t - float(z["t"])))) for z in snaps]
    num = sum(Pi_t[i] for i in idx)
    den = sum(np.sqrt(P.parse_number(v["beta_i"])) * c_m * z["kappa1_m_par"][:-1] * W_t[i][:-1] for i, z in zip(idx, snaps))
    with np.errstate(divide="ignore", invalid="ignore"):
        paired = num / den
    mc = rule["m_c"]
    return dict(label=label, run_id=rid, M=M, N=int(cfg["box"]["N"]), beta=P.parse_number(v["beta_i"]), mf=mf,
                chi=float(an["outer"]["chi_out"]), m_c=mc, ratio=ratio, model=model,
                ratio_2_mc=rng(2, mc, ratio), model_2_mc=rng(2, mc, model),
                ratio_8_M2=rng(8, M // 2, ratio), model_8_M2=rng(8, M // 2, model), paired=paired,
                paired_2_mc=rng(2, mc, paired), paired_8_M2=rng(8, M // 2, paired), n_snapshots=len(snaps),
                argmin_8_M2=int(8 + np.argmin(ratio[8:M // 2 + 1])))


def legend_text(d: dict, panel: str) -> str:
    if panel == "a":
        return rf"${d['beta']:g}$, ${d['M']}$" + (r", $g_1$" if d["mf"] == 1 else "")
    if panel == "b":
        return rf"${d['beta']:g}$, ${d['N']}^3$"
    return rf"${d['beta']:g}$, ${d['chi']:.2f}$"


TITLES = dict(a=r"$\beta_i$, $M$ ($64^3$)", b=r"$\beta_i$, $N^3$", c=r"$\beta_i$, $\chi$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    labels = sorted({lab for rows in PANELS.values() for lab, _, _ in rows}, key=list(RUNS).index)
    D = {lab: load(lab) for lab in labels}
    for lab in labels:
        d = D[lab]
        print(f"{lab:9s} N {d['N']:3d} M {d['M']:3d} m_c {d['m_c']:3d}: ratio 2..m_c {d['ratio_2_mc'][0]:.3f}-"
              f"{d['ratio_2_mc'][1]:.3f} (model {d['model_2_mc'][0]:.3f}-{d['model_2_mc'][1]:.3f}); 8..M/2 "
              f"{d['ratio_8_M2'][0]:.3f}-{d['ratio_8_M2'][1]:.3f} (model {d['model_8_M2'][0]:.3f}-{d['model_8_M2'][1]:.3f}, "
              f"min at m = {d['argmin_8_M2']}); paired at the {d['n_snapshots']} snapshot times 2..m_c "
              f"{d['paired_2_mc'][0]:.3f}-{d['paired_2_mc'][1]:.3f}, 8..M/2 {d['paired_8_M2'][0]:.3f}-{d['paired_8_M2'][1]:.3f}")
    out = {lab: {k: v for k, v in d.items() if k not in ("ratio", "model", "paired")}
           | dict(ratio=d["ratio"].tolist(), model=d["model"].tolist(), paired=d["paired"].tolist()) for lab, d in D.items()}
    (P.DATA / "h1_noecho.json").write_text(json.dumps(P._jsonable(out), indent=1))

    style = FS.style_config()
    pal = list(style["palette"])
    blk = style["reference_color"]

    def draw(plt):
        from matplotlib.lines import Line2D
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        fig, ax = plt.subplots(1, 3, figsize=(style["double_width_in"], 2.6), sharey=True)
        for a, (panel, rows) in zip(ax, PANELS.items()):
            handles = []
            for lab, ci, filled in rows:
                d = D[lab]
                c = blk if ci < 0 else pal[ci]
                m = np.arange(d["M"])
                ok = (m > d["mf"]) & (m <= d["M"] // 2)      # from the moment above the forced one to M/2
                inside, beyond = ok & (m <= d["m_c"]), ok & (m >= d["m_c"])
                mk = dict(marker="o" if panel != "b" or d["beta"] < 1 else "s", ms=1.8, mfc=c if filled else "white",
                          mec=c, mew=0.5)
                ls = "-" if filled or panel == "c" else "--"
                a.plot(m[inside], d["paired"][inside], ls, color=c, lw=0.9, **mk)
                a.plot(m[beyond], d["paired"][beyond], ls, color=c, lw=0.6, alpha=0.3)
                a.plot(m[ok], d["ratio"][ok], ":", color=c, lw=0.6, alpha=0.45)
                a.plot(m[ok], d["model"][ok], "-", color=c, lw=0.5, alpha=0.8)
                handles.append(Line2D([], [], color=c, ls=ls, lw=0.9, **mk, label=legend_text(d, panel)))
            a.axhline(1.0, color=blk, lw=0.5, ls=":")
            a.set_xscale("log")
            a.set_xlim(0.8, 300 if panel == "c" else 36)
            a.set_ylim(0.70, 1.04)
            a.xaxis.set_major_formatter(ScalarFormatter())
            a.xaxis.set_minor_formatter(NullFormatter())
            a.set_xticks([1, 10, 100] if panel == "c" else [1, 2, 4, 8, 16, 32])
            a.set_xlabel(r"$m$")
            a.legend(handles=handles, fontsize=5.5, loc="lower left", ncol=2, handlelength=2.0, columnspacing=0.8,
                     borderaxespad=0.3, title=TITLES[panel], title_fontsize=5.5)
            FS.panel_label(a, f"({panel})")
        ax[0].set_ylabel(r"$\Pi_{m+1/2}/[s\,c_m\,\kappa_1(m)\,W(m)]$")
        fig.tight_layout()
        return fig

    if args.outdir:
        FS.FIG_DIR = Path(args.outdir)
    mode = FS.render(draw, "h1_noecho", style)
    print(f"wrote {FS.FIG_DIR / 'h1_noecho'}.pdf/.png ({mode}) and data/h1_noecho.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
