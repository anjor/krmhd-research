#!/usr/bin/env python
"""Overview of one forced run (figure fig:overview): energies against time with the averaging blocks, the
perpendicular spectra, W(m) per block and the dissipation per Hermite moment.

Reads <run>/<run>.npz and .json (scripts/run_forced_turbulence.py) and the run's accepted window
(configs/h1_acceptance.yaml); writes figures/h1_overview.{pdf,png}.
  (a) E_A and W against t; the dashed lines mark the four averaging blocks, the shading the excluded start;
  (b) E_A(k_perp) and W(k_perp) averaged over the window, with k_perp^{-5/3} for reference;
  (c) W(m)/eps for each block, each divided by its own injection rate;
  (d) D_perp(m)/eps and D_nu(m)/eps, the free energy removed from moment m by the hyper-resistivity and by the
      hypercollisions, as fractions of the injection rate.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/fig_overview.py <run ID>
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id")
    ap.add_argument("--acceptance", default="h1_acceptance.yaml")
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--name", default="h1_overview")
    args = ap.parse_args()
    rd = P.DATA / args.run_id
    meta = json.load(open(rd / f"{args.run_id}.json"))
    with np.load(rd / f"{args.run_id}.npz") as z:
        ts = {k[4:]: z[k] for k in z.files if k.startswith("ts__")}
        nmax = int(z["nmax"])
    acc = P.load_config(args.acceptance)
    width = float((acc.get("runs") or {}).get(args.run_id, {}).get("window", meta["config"]["gates"]["steady_window"]))
    t, dt = ts["t"], ts["acc_sum_dt"]
    first = int(np.where(H.window_slice(t, width))[0][0])
    b = H.block_averages(t, dt, first, 4, dict(W=ts["pWm"]), dict(eps=ts["acc_inj_g"]))
    sel = np.arange(len(t)) > first
    Tw = dt[sel].sum()
    eps = ts["acc_inj_g"][sel].sum() / Tw
    EAk = H.time_average(ts["EAk"], dt, first)
    Wk = H.time_average(ts["Wmk"].sum(2), dt, first)
    Dp = ts["acc_loss_eta"][sel].sum(0) / Tw
    Dn = ts["acc_loss_nu"][sel].sum(0) / Tw
    M = len(Dp) - 1
    m = np.arange(M + 1)

    fcfg = FS.style_config()
    plt = FS.setup_style(fcfg, usetex=True)
    pal = fcfg["palette"]
    fig, ax = plt.subplots(2, 2, figsize=(fcfg["double_width_in"], 5.0))
    a = ax[0, 0]
    a.plot(t, ts["E"], color=pal[0], lw=0.6, label=r"$E_A$")
    a.plot(t, ts["pWm"].sum(1), color=pal[1], lw=0.6, label=r"$W$")
    for e in b["edges"]:
        a.axvline(e, color="0.6", lw=0.5, ls="--")
    a.axvspan(t[0], t[first], color="0.92", lw=0)
    a.set_xlabel(r"$t/\tau_A$")
    a.set_ylabel(r"energy")
    a.legend(frameon=False, fontsize=8)
    a = ax[0, 1]
    n = np.arange(len(EAk))
    v = (n >= 1) & (n <= nmax + 8)
    a.loglog(n[v], EAk[v], "-", color=pal[0], lw=1.0, label=r"$E_A(k_\perp)$")
    a.loglog(n[v], Wk[v], "-", color=pal[1], lw=1.0, label=r"$W(k_\perp)$")
    ref = np.arange(2, 9)
    a.loglog(ref, 1.5 * EAk[2] * (ref / 2.0) ** (-5 / 3), "k--", lw=0.7, label=r"$k_\perp^{-5/3}$")
    a.axvline(nmax, color="0.6", lw=0.5, ls=":")
    a.set_xlabel(r"$k_\perp L/2\pi$")
    a.set_ylabel(r"spectrum")
    a.legend(frameon=False, fontsize=8)
    a = ax[1, 0]
    for i in range(4):
        a.loglog(m[1:], b["W"][i][1:] / b["eps"][i], lw=0.8, color=pal[(i + 2) % len(pal)],
                 label=rf"$t={b['edges'][i]:.0f}$--${b['edges'][i + 1]:.0f}$")
    a.axvspan(M / 2, M, color="0.92", lw=0, zorder=0)
    a.set_xlabel(r"$m$")
    a.set_ylabel(r"$W(m)/\varepsilon$")
    a.legend(frameon=False, fontsize=7)
    a = ax[1, 1]
    a.semilogy(m, Dp / eps, "o-", ms=2.5, color=pal[0], lw=0.8, label=rf"$D_\perp(m)/\varepsilon$ (total {Dp.sum() / eps:.2f})")
    a.semilogy(m[2:], np.maximum(Dn[2:], 1e-12) / eps, "s-", ms=2.5, color=pal[1], lw=0.8,
               label=rf"$D_\nu(m)/\varepsilon$ (total {Dn.sum() / eps:.2f})")
    a.set_ylim(1e-5, 1)
    a.set_xlabel(r"$m$")
    a.set_ylabel(r"dissipation / injection")
    a.legend(frameon=False, fontsize=7)
    for a_, lab in zip(ax.flat, "abcd"):
        a_.text(0.0, 1.02, f"({lab})", transform=a_.transAxes)
    fig.tight_layout()
    outdir = Path(args.outdir) if args.outdir else FS.FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{args.name}.pdf", bbox_inches="tight")
    fig.savefig(outdir / f"{args.name}.png", dpi=fcfg["png_dpi"], bbox_inches="tight")
    print(f"wrote {outdir / args.name}.pdf/.png; eps = {eps:.3f}, D_perp share {Dp.sum() / (Dp.sum() + Dn.sum()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
