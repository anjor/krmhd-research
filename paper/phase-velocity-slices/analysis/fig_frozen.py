#!/usr/bin/env python
"""Figure fig:frozen and table figures/frozen_table.tex: the frozen-line theorem tested with GANDALF (run family C,
scripts/run_frozen_line.py, configs/frozen_line.yaml).

Physics.  Under ideal induction and without spatial dissipation of g, g(t) = X_t exp(-t s grad_par(0) A) g(0)
(paper thm:frozen), so every volume-integrated single-point statistic of g(t), taken with grad_par(t), equals that of
the linear problem along the field lines frozen at t = 0 with no flow (cor:singlepoint).  The runs integrate, from the
same physical initial fields at every N of the config (16, 20, 32, 40, 64):
  NL  ideal KRMHD with gandalf_step (Lawson-RK4);  FR  the frozen-line prediction;  LZ  the k_z-only prediction.
For each statistic X the relative difference NL - FR is
  sum |X_NL - X_FR| / sum |X_FR|     for W(m) = p_m <g_m^2>, <g_m^4> (m = 0, 1, 2), W(k_perp), W(|k_z|);
  max |X_NL - X_FR| / max |X_FR|     for the slice correlation matrix C_jl = <a_j a_l> and the Hermite fluxes,
the field-aligned flux Pi_(m+1/2) = 2 s p_m A_(m,m+1) <g_m grad_par g_(m+1)> (grad_par(t) for NL, grad_par(0) for FR)
and the k_z-only flux (d_z in place of grad_par).  The single-point statistics must converge with N (the departure is
the Fourier truncation error of the relabelling), the others must not.

Panel (a): W(m) at t = 0.125 for NL, FR and LZ on the finest grid, with W(m) at t = 0.  Panel (b): the relative differences
NL - FR at t = 0.125 against N.  The table gives the same differences at t = 0.062 and 0.125, and the NL - LZ
difference of W(m) and of the field-aligned flux.

Reads only the saved .npz files of the latest runs P1_frozen-N<N>_* (and P1_frozen-N<N>-dt<k>_* for the time-step
check on the finest grid); writes figures/frozen.pdf, figures/frozen.png, figures/frozen_table.tex and data/computed/fig_frozen.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/fig_frozen.py [--config paper/phase-velocity-slices/configs/frozen_line.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import (CACHE_DIR, CONFIGS, DATA, FIG_DIR, environment, panel_label, render,  # noqa: E402
                      style_config)

NAME = "frozen"


def latest_run(label: str) -> str:
    """The newest data folder P1_<label>_<YYYYMMDD_HHMMSS>."""
    import re
    pat = re.compile(rf"^P1_{re.escape(label)}_(\d{{8}}_\d{{6}})$")
    found = sorted((m.group(1), p) for p in os.listdir(DATA) if (m := pat.match(p)))
    if not found:
        raise FileNotFoundError(f"no run folder P1_{label}_<timestamp> in {DATA}")
    return os.path.join(DATA, found[-1][1])


def load(run_dir: str) -> tuple[dict, dict]:
    """Arrays (.npz) and metadata (.json) of one run."""
    base = os.path.join(run_dir, os.path.basename(run_dir))
    with np.load(base + ".npz") as z:
        arrays = {k: z[k] for k in z.files}
    with open(base + ".json") as fh:
        meta = json.load(fh)
    return arrays, meta


def rel_l1(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).sum() / np.abs(b).sum())


def rel_max(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.abs(a - b).max() / np.abs(b).max())


# statistic key -> (array key, metric, single-point?, LaTeX label)
STATS = {
    "W": ("pW_m", rel_l1, True, r"$W(m)$"),
    "C": ("C_slice", rel_max, True, r"$C_{jl}=\langle a_ja_l\rangle$"),
    "Pi": ("Pi_fa", rel_max, True, r"$\Pi_{m+1/2}$ with $\nabla_\parallel$"),
    "g4": ("g4", rel_l1, True, r"$\langle g_m^4\rangle$, $m\le2$"),
    "Wkp": ("W_kperp", rel_l1, False, r"$W(k_\perp)$"),
    "Wkz": ("W_kz", rel_l1, False, r"$W(|k_z|)$"),
    "Pz": ("Pi_kz", rel_max, False, r"$\Pi_{m+1/2}$ with $\partial_z$"),
}


def metrics(d: dict, i: int, ref: str = "FR") -> dict:
    """Relative differences NL - ref at output index i for every statistic in STATS."""
    return {k: f(d[f"NL__{a}"][i], d[f"{ref}__{a}"][i]) for k, (a, f, _, _) in STATS.items()}


def t_index(d: dict, t: float) -> int:
    i = int(np.argmin(np.abs(d["t_out"] - t)))
    if abs(d["t_out"][i] - t) > 1e-9:
        raise ValueError(f"time {t} not among the outputs {d['t_out']}")
    return i


def sci(x: float, digits: int = 2) -> str:
    """LaTeX scientific notation a\\times10^{b} with `digits` significant digits."""
    if x == 0 or not np.isfinite(x):
        return f"${x:g}$"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, digits - 1) >= 10:
        m, e = m / 10, e + 1
    if -2 <= e <= 0:
        return f"${x:.{digits - 1 - e}f}$"
    return f"${m:.{digits - 1}f}\\times10^{{{e}}}$"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIGS / "frozen_line.yaml"))
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    style = style_config()
    Ns = [int(n) for n in cfg["box"]["N"]]
    t_fig = 0.125
    t_tab = [0.062, 0.125]
    runs, metas = {}, {}
    for N in Ns:
        runs[N], metas[N] = load(latest_run(f"{cfg['label']}-N{N}"))
        if metas[N]["outcome"] != "pass":
            print(f"run {metas[N]['run_id']} did not pass its gates: stop")
            return 1
    out: dict = dict(runs={N: metas[N]["run_id"] for N in Ns}, environment=environment(), metrics={},
                     metrics_unweighted_W={}, nl_vs_lz={}, dt={N: float(runs[N]["dt"]) for N in Ns})
    for t in sorted(set(t_tab + [t_fig, 0.25])):
        out["metrics"][t] = {N: metrics(runs[N], t_index(runs[N], t)) for N in Ns}
        out["metrics_unweighted_W"][t] = {N: rel_l1(runs[N]["NL__W_m"][t_index(runs[N], t)],
                                                    runs[N]["FR__W_m"][t_index(runs[N], t)]) for N in Ns}
        out["nl_vs_lz"][t] = {N: metrics(runs[N], t_index(runs[N], t), ref="LZ") for N in Ns}
    chk = cfg["time"].get("dt_check")
    if chk:
        lab = f"{cfg['label']}-N{int(chk['N'])}-dt{int(chk['divide'])}"
        try:
            dd, mm = load(latest_run(lab))
            out["dt_check"] = dict(run=mm["run_id"], N=int(chk["N"]), dt=float(dd["dt"]),
                                   metrics={t: metrics(dd, t_index(dd, t)) for t in t_tab + [0.25]})
        except FileNotFoundError:
            print(f"no time-step check run {lab}")
    for N in Ns:
        g = metas[N]["gates"]["energy_conservation"]
        out.setdefault("gates", {})[N] = dict(free_energy_rel_drift=g["free_energy_rel_drift"],
                                             alfvenic_energy_rel_drift=g["alfvenic_energy_rel_drift"],
                                             rho_dt_max=metas[N]["extra"]["rho_dt_max"])
    out["chi"] = metas[Ns[-1]]["extra"]["chi"]
    out["bracket_over_streaming_g0"] = metas[Ns[-1]]["extra"]["bracket_over_streaming_g0"]

    # ---------------------------------------------------------------- figure
    pal = style["palette"]
    black = style["reference_color"]
    d48 = runs[Ns[-1]]
    i_t = t_index(d48, t_fig)
    m = np.arange(d48["NL__pW_m"].shape[-1])

    short = {"W": r"$W(m)$", "C": r"$C_{jl}$", "Pi": r"$\Pi_{m+1/2}$, $\nabla_\parallel$",
             "g4": r"$\langle g_m^4\rangle$", "Wkp": r"$W(k_\perp)$", "Wkz": r"$W(|k_z|)$",
             "Pz": r"$\Pi_{m+1/2}$, $\partial_z$"}

    def draw(plt):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(style["double_width_in"], 2.75), gridspec_kw=dict(wspace=0.3))
        w0 = d48["NL__pW_m"][0]
        nz0 = w0 > 0
        axa.plot(m[nz0], w0[nz0], ls="none", color="0.55", marker="x", ms=4.5, mew=1.0, label=r"$t=0$")
        axa.plot(m, d48["FR__pW_m"][i_t], color=pal[0], lw=1.3, label=r"frozen lines")
        axa.plot(m, d48["LZ__pW_m"][i_t], color=pal[1], lw=1.1, ls=(0, (4, 2)), label=r"$k_z$ only")
        axa.plot(m, d48["NL__pW_m"][i_t], ls="none", marker="o", ms=4.2, mfc="white", mec=black, mew=1.0,
                 label=r"GANDALF")
        axa.set_yscale("log")
        axa.set_xlabel(r"$m$")
        axa.set_ylabel(r"$W(m)$")
        axa.set_xticks(m)
        axa.legend(loc="lower left", handlelength=1.8)
        sp = [k for k, v in STATS.items() if v[2]]
        nsp = [k for k, v in STATS.items() if not v[2]]
        mk = {"W": "o", "C": "s", "Pi": "^", "g4": "D", "Wkp": "o", "Wkz": "s", "Pz": "^"}
        col = {"W": pal[0], "C": pal[1], "Pi": pal[2], "g4": pal[3], "Wkp": pal[0], "Wkz": pal[4], "Pz": pal[2]}
        hs, hn = [], []
        for k in sp:
            y = [out["metrics"][t_fig][N][k] for N in Ns]
            hs += axb.plot(Ns, y, color=col[k], lw=1.1, marker=mk[k], ms=3.8, label=short[k])
        for k in nsp:
            y = [out["metrics"][t_fig][N][k] for N in Ns]
            hn += axb.plot(Ns, y, color=col[k], lw=1.0, ls=(0, (4, 2)), marker=mk[k], ms=3.8, mfc="white",
                           label=short[k])
        axb.set_xscale("log")
        axb.set_yscale("log")
        axb.set_xticks(Ns)
        axb.set_xticks([], minor=True)
        axb.set_xticklabels([str(N) for N in Ns])
        axb.set_xlim(Ns[0] / 1.1, Ns[-1] * 1.1)
        axb.set_ylim(2e-7, 2.0)
        axb.set_xlabel(r"$N$")
        axb.set_ylabel(r"relative difference")
        axb.legend(handles=hs + hn + [plt.Line2D([], [], ls="none")], ncol=2, loc="lower left",
                   fontsize=style["legend_font_size_pt"] - 0.5, handlelength=1.6, columnspacing=0.9,
                   labelspacing=0.25, handletextpad=0.5, borderaxespad=0.3)
        panel_label(axa, "(a)")
        panel_label(axb, "(b)")
        return fig

    out["rendering"] = render(draw, NAME, style)

    # ---------------------------------------------------------------- table
    run_dates = sorted({out["runs"][N].split("_")[-2] for N in Ns})          # YYYYMMDD of each run ID
    n_list = ", ".join(str(N) for N in Ns[:-1]) + f" and {Ns[-1]}"
    lines = [r"% Generated by analysis/fig_frozen.py from the frozen-line runs at N = " + n_list + " ("
             + ", ".join(f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in run_dates) + ").",
             r"% Relative differences between GANDALF (gandalf_step, Lawson-RK4) and the frozen-line prediction"
             r" (last two rows: the k_z-only prediction).",
             r"\begin{tabular}{l" + "c" * len(Ns) + "}",
             r"\toprule",
             r"statistic & " + " & ".join(f"$N={N}$" for N in Ns) + r" \\"]
    for t in t_tab:
        lines += [r"\midrule", rf"\multicolumn{{{1 + len(Ns)}}}{{l}}{{$t={t:g}$}} \\", r"\midrule"]
        for k, (a, f, single, lab) in STATS.items():
            if k == "Wkp":
                lines.append(r"\addlinespace")
            lines.append(lab + " & " + " & ".join(sci(out["metrics"][t][N][k]) for N in Ns) + r" \\")
        lines.append(r"\addlinespace")
        for k, lab in (("W", r"$W(m)$, $k_z$-only prediction"), ("Pi", r"$\Pi_{m+1/2}$, $k_z$-only prediction")):
            lines.append(lab + " & " + " & ".join(sci(out["nl_vs_lz"][t][N][k]) for N in Ns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(FIG_DIR / "frozen_table.tex", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / "fig_frozen.json", "w") as fh:
        json.dump(out, fh, indent=1, default=float)
        fh.write("\n")
    for t in t_tab + [0.25]:
        print(f"t = {t}: NL - FR")
        for k in STATS:
            print(f"   {k:4s} " + "  ".join(f"N={N}: {out['metrics'][t][N][k]:.2e}" for N in Ns)
                  + (f"   [dt/2 at N={out['dt_check']['N']}: {out['dt_check']['metrics'][t][k]:.2e}]"
                     if "dt_check" in out else ""))
        print("   W unweighted <g_m^2>: " + "  ".join(f"N={N}: {out['metrics_unweighted_W'][t][N]:.2e}" for N in Ns))
        print("   NL - LZ: W " + "  ".join(f"{out['nl_vs_lz'][t][N]['W']:.2e}" for N in Ns) + ";  Pi "
              + "  ".join(f"{out['nl_vs_lz'][t][N]['Pi']:.2e}" for N in Ns))
    print(f"wrote figures/{NAME}.pdf, .png ({out['rendering']}) and figures/frozen_table.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
