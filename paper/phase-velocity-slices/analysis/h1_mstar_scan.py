#!/usr/bin/env python
"""The break m_* of the Hermite spectrum against beta_i at fixed chi and against chi at fixed beta_i (m_* scan,
Anjor 2026-11-10).

For each run (window averages of its accepted window, configs/h1_acceptance.yaml, and its h1_analysis.json from
h1_pilot.py) the numbers come from h1_run_numbers.numbers:
  * measured m_* = ell_*^2/2, with ell_* the decay length of W(m) sqrt(2m) ~ exp(-sqrt(2m)/ell_*) fitted over 1.5
    e-folds of the run's own decay from m = 2 (h1lib.fit_decay_window), its uncertainty from the standard error of
    the fit to the four block averages of W(m) (sigma of m_* = ell_* sigma of ell_*);
  * the status of that measurement: measured (the 1.5 e-fold window lies below the collisional range m > M/2; filled
    marker), marginal (1 <= n_e < 1.5 e-folds over the whole collision-free range; open marker), unresolved (n_e < 1;
    not plotted);
  * the model value (s kappa_1 T)^2/2 with the run's measured T (median of D_perp/W over 2 <= m <= 16) and the mean
    field-aligned wavenumber kappa_1 of the injected pattern (traced field lines), and the model spectrum with the
    traced P(kappa) fitted over the same window as the data.
A power law m_* ~ beta_i^a (panel a) is fitted to the measured points by weighted least squares in ln m_*.  Against chi
no single power law is drawn (Anjor 2026-11-12, option A: the chi points steepen from about chi^-2.5 to chi^-4 below
chi ~ 0.5); the local exponents -d ln m_*/d ln chi between neighbouring points are written to the .json, as is the
weighted single power law for reference.

Writes figures/h1_mstar.{pdf,png} (JPP double column) and data/h1_mstar_scan.json.
  (a) m_* against beta_i, runs given with --beta (chi_out ~ 0.95);
  (b) m_* against chi_out, runs given with --chi (one beta_i), with reference slopes chi^-2 and chi^-4 through the
      run of largest chi.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_mstar_scan.py --beta <run> [...] --chi <run> [...]
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
from h1_run_numbers import numbers  # noqa: E402

KEYS = ("beta", "N", "M", "chi_out", "tau_nl", "T_over_tau_nl", "kappa1_injected", "ell_win", "ell_win_se",
        "m_star_win", "fit_status", "fit_n_e", "fit_m_max", "coll_over_sink_at_m_max", "s_kappa1_T", "m_star_model",
        "ell_model_win", "m_star_model_win", "ell_star", "m_star_fit")


def point(run_id: str, acc: dict) -> dict:
    """The numbers of one run used here (see the module docstring), with sigma of m_* = ell_* sigma(ell_*)."""
    d = numbers(run_id, acc)
    out = {k: d[k] for k in KEYS}
    out["run_id"] = run_id
    out["m_star_win_sigma"] = d["ell_win"] * d["ell_win_se"]
    return out


def power_law(x: list[float], y: list[float], sy: list[float]) -> dict:
    """Weighted least squares of ln y = c + a ln x with sigma(ln y) = sy/y: exponent a, its standard error, c, chi^2
    and the number of points.  With two points the fit is exact and the error comes from the two sigmas alone."""
    lx, ly = np.log(x), np.log(y)
    w = (np.asarray(y) / np.asarray(sy)) ** 2
    A = np.vstack([np.ones_like(lx), lx]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    c, a = cov @ (A.T @ (w * ly))
    chi2 = float(np.sum(w * (ly - c - a * lx) ** 2))
    return dict(exponent=float(a), exponent_se=float(math.sqrt(cov[1, 1])), log_prefactor=float(c), chi2=chi2,
                n=len(x))


def measured(points: list[dict]) -> list[dict]:
    return [p for p in points if p["fit_status"] == "measured"]


def plain_log_ticks(axis, candidates: list[float], lo: float, hi: float) -> None:
    """Major ticks of a log axis at the candidate values inside [lo, hi], labelled as plain numbers."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    ticks = [c for c in candidates if lo <= c <= hi]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def draw_panel(ax, points: list[dict], xkey: str, colors: list[str]) -> None:
    """m_* against points[xkey]: data (filled if measured, open if marginal, error bars from the blocks) and the
    model value (s kappa_1 T)^2/2 (crosses)."""
    labelled = set()
    for p in points:
        if p["fit_status"] == "unresolved":
            continue
        filled = p["fit_status"] == "measured"
        lab = "measured" if filled else "marginal"
        ax.errorbar(p[xkey], p["m_star_win"], yerr=p["m_star_win_sigma"], fmt="o", ms=5, color=colors[0],
                    mfc=colors[0] if filled else "white", capsize=2, zorder=3,
                    label=None if lab in labelled else rf"$m_*$ {lab}")
        ax.plot(p[xkey], p["m_star_model"], "x", ms=6, color=colors[1], zorder=4,
                label=None if "model" in labelled else r"$(s\kappa_1 T)^2/2$")
        labelled |= {lab, "model"}
    ax.set_xscale("log")
    ax.set_yscale("log")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--beta", nargs="+", required=True, help="runs at chi_out ~ 0.95 for panel (a)")
    ap.add_argument("--chi", nargs="+", required=True, help="runs at one beta_i for panel (b)")
    ap.add_argument("--acceptance", default="h1_acceptance.yaml")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    acc = P.load_config(args.acceptance)
    pa = [point(r, acc) for r in args.beta]
    pb = [point(r, acc) for r in args.chi]
    betas_b = {round(p["beta"], 6) for p in pb}
    if len(betas_b) != 1:
        raise SystemExit(f"--chi runs must share one beta_i, got {sorted(betas_b)}")

    out: dict = dict(beta_scan=pa, chi_scan=pb)
    ma, mb = measured(pa), measured(pb)
    if len(ma) >= 2:
        out["fit_beta"] = power_law([p["beta"] for p in ma], [p["m_star_win"] for p in ma],
                                    [p["m_star_win_sigma"] for p in ma])
        out["fit_beta_model"] = power_law([p["beta"] for p in ma], [p["m_star_model"] for p in ma],
                                          [p["m_star_win_sigma"] for p in ma])
    if len(mb) >= 2:
        f = power_law([p["chi_out"] for p in mb], [p["m_star_win"] for p in mb], [p["m_star_win_sigma"] for p in mb])
        out["fit_chi"] = dict(f, exponent=-f["exponent"])           # m_* ~ chi^-b: b reported
        g = power_law([p["chi_out"] for p in mb], [p["m_star_model"] for p in mb], [p["m_star_win_sigma"] for p in mb])
        out["fit_chi_model"] = dict(g, exponent=-g["exponent"])
        t = power_law([p["chi_out"] for p in mb], [p["T_over_tau_nl"] for p in mb], [0.03 * p["T_over_tau_nl"] for p in mb])
        out["fit_T_over_tau_nl_chi"] = t                             # constant T/tau_nl is the chi^-2 law
        s = sorted(mb, key=lambda p: -p["chi_out"])
        out["local_exponents_chi"] = [
            dict(chi=[a["chi_out"], b["chi_out"]],
                 m_star=-math.log(b["m_star_win"] / a["m_star_win"]) / math.log(b["chi_out"] / a["chi_out"]),
                 m_star_model=-math.log(b["m_star_model"] / a["m_star_model"]) / math.log(b["chi_out"] / a["chi_out"]))
            for a, b in zip(s[:-1], s[1:])]

    fcfg = FS.style_config()
    plt = FS.setup_style(fcfg, usetex=True)
    pal = fcfg["palette"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(fcfg["double_width_in"], 0.42 * fcfg["double_width_in"]))

    draw_panel(axa, pa, "beta", pal)
    if "fit_beta" in out:
        f = out["fit_beta"]
        xs = np.geomspace(min(p["beta"] for p in pa) / 1.3, max(p["beta"] for p in pa) * 1.3, 50)
        ax_ref = math.exp(f["log_prefactor"]) * np.exp(f["exponent"] * np.log(np.median([p["beta"] for p in ma])))
        b0 = np.median([p["beta"] for p in ma])
        axa.plot(xs, math.exp(f["log_prefactor"]) * xs ** f["exponent"], "-", color=pal[0], lw=0.8,
                 label=rf"fit $\propto\beta_i^{{{f['exponent']:.2f}}}$")
        axa.plot(xs, ax_ref * (xs / b0), ":", color=fcfg["reference_color"], lw=0.8, label=r"$\propto\beta_i$")
    axa.set_xlabel(r"$\beta_i$")
    axa.set_ylabel(r"$m_*$")
    if axa.get_legend_handles_labels()[0]:
        axa.legend(loc="upper left", frameon=False)
    FS.panel_label(axa, "(a)")

    draw_panel(axb, pb, "chi_out", pal)
    top = max(pb, key=lambda p: p["chi_out"])
    xs = np.geomspace(min(p["chi_out"] for p in pb) / 1.15, top["chi_out"] * 1.15, 50)
    for k, ls in ((2, ":"), (4, "--")):
        axb.plot(xs, top["m_star_win"] * (xs / top["chi_out"]) ** (-k), ls, color=fcfg["reference_color"], lw=0.8,
                 label=rf"$\propto\chi^{{-{k}}}$")
    axb.set_xlabel(r"$\chi$")
    axb.set_ylabel(r"$m_*$")
    axb.legend(loc="upper right", frameon=False)
    FS.panel_label(axb, "(b)")
    mticks = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 300, 500]
    for ax, xt in ((axa, [0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]), (axb, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])):
        plain_log_ticks(ax.xaxis, xt, *ax.get_xlim())
        plain_log_ticks(ax.yaxis, mticks, *ax.get_ylim())
    fig.tight_layout()

    outdir = Path(args.outdir) if args.outdir else FS.FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "h1_mstar.pdf", bbox_inches="tight")
    fig.savefig(outdir / "h1_mstar.png", dpi=fcfg["png_dpi"], bbox_inches="tight")
    (P.DATA / "h1_mstar_scan.json").write_text(json.dumps(P._jsonable(out), indent=1))
    for name, pts in (("beta", pa), ("chi", pb)):
        for p in pts:
            print(f"{name}: {p['run_id']}: beta {p['beta']:g}, chi {p['chi_out']:.2f}, M {p['M']}, {p['fit_status']} "
                  f"(n_e {p['fit_n_e']:.2f}, m <= {p['fit_m_max']}): m_* {p['m_star_win']:.1f} +- "
                  f"{p['m_star_win_sigma']:.1f}; model (s kappa_1 T)^2/2 {p['m_star_model']:.1f}, model spectrum "
                  f"{p['m_star_model_win']:.1f}; T/tau_nl {p['T_over_tau_nl']:.1f}, kappa_1 {p['kappa1_injected']:.2f}")
    for k in ("fit_beta", "fit_beta_model", "fit_chi", "fit_chi_model", "fit_T_over_tau_nl_chi"):
        if k in out:
            f = out[k]
            print(f"{k}: exponent {f['exponent']:.3f} +- {f['exponent_se']:.3f} (chi^2 {f['chi2']:.2f}, {f['n']} points)")
    for e in out.get("local_exponents_chi", []):
        print(f"local exponent -dln m_*/dln chi over chi {e['chi'][0]:.2f}-{e['chi'][1]:.2f}: measured {e['m_star']:.2f}, "
              f"model {e['m_star_model']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
