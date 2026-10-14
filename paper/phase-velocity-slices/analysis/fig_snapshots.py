#!/usr/bin/env python
"""Figure fig:snapshots: phase-velocity slices of a density blob in decaying Alfvenic turbulence (run family A,
scripts/run_slices_turbulence.py, configs/slices_turbulence.yaml).

Physics.  A density perturbation g = f e_0 of one compressive branch loads every slice with the same pattern,
a_j(0) = p_0 R_0j f (paper eq:sliceforcing with F = f e_0).  Each slice is then a passive scalar carried by
zhat x grad(Phi + s u_j Psi) and streaming along z at s u_j (thm:slice, thm:trunc): the slice with u_j = 0 is
advected by u_E alone, and the slices with s u_j = -1 and +1 by z- and z+ alone (prop:elsasser), so the three
patterns differ although they start identical.  The density moment g_0 = sum_j R_0j a_j is their superposition.

Panels, on the plane z = z0 of the run: (a) g_0 at t = 0; (b) g_0 at the last output (1.5 outer eddy times);
(c)-(e) the slices with s u_j nearest to -1, 0 and +1 at the last output, each normalised by p_0 R_0j so that all
three start from the same f.  viridis, symmetric colour scale shared within each group ((a)-(b) and (c)-(e)).

Reads only the saved .npz of the latest run P1_slices-turb_*; writes figures/snapshots.pdf and .png and
data/computed/fig_snapshots.json (the numbers quoted in the text: chi, tau, the u_j shown, slice energies, the
decoupling check).

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/fig_snapshots.py [--config paper/phase-velocity-slices/configs/slices_turbulence.yaml]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import CACHE_DIR, CONFIGS, DATA, environment, render, style_config  # noqa: E402

NAME = "snapshots"


def latest_run(label: str) -> str:
    """The newest data folder P1_<label>_<YYYYMMDD_HHMMSS>."""
    pat = re.compile(rf"^P1_{re.escape(label)}_(\d{{8}}_\d{{6}})$")
    found = sorted((m.group(1), p) for p in os.listdir(DATA) if (m := pat.match(p)))
    if not found:
        raise FileNotFoundError(f"no run folder P1_{label}_<timestamp> in {DATA}")
    return os.path.join(DATA, found[-1][1])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIGS / "slices_turbulence.yaml"))
    ap.add_argument("--run-dir", default=None, help="a specific run folder (default: the latest run of the label)")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    style = style_config()
    rd = args.run_dir or latest_run(cfg["label"])
    base = os.path.join(rd, os.path.basename(rd))
    with np.load(base + ".npz") as z:
        d = {k: z[k] for k in z.files}
    with open(base + ".json") as fh:
        meta = json.load(fh)
    if meta["outcome"] != "pass":
        print(f"run {meta['run_id']} did not pass its gates: stop")
        return 1
    s = float(np.sqrt(float(meta["config"]["velocity"]["beta_i"])))
    u = d["u"]
    j_show = [int(np.argmin(np.abs(s * u - float(x)))) for x in cfg["output"]["su_shown"]]
    R0 = d["R"][0]
    p0 = d["p"][0]
    t_out, tau = d["t_out"], float(d["tau"])
    a_last = d["snap__a_plane"][-1]                       # [y, x, j] on the refined plane
    g0 = d["snap__g0_plane"]                               # [time, y, x]
    x = d["coords"]
    L = float(meta["config"]["box"]["L"])
    ext = [0.0, L, 0.0, L]
    slices_norm = [a_last[..., j] / (p0 * R0[j]) for j in j_show]

    numbers = dict(run=meta["run_id"], environment=environment(), chi=meta["extra"]["chi"], tau=tau,
                   t_out=t_out.tolist(), t_out_over_tau=(t_out / tau).tolist(), j_shown=j_show,
                   u_shown=[float(u[j]) for j in j_show], su_shown=[float(s * u[j]) for j in j_show],
                   u_nodes=u.tolist(), g0_max=[float(np.abs(g).max()) for g in g0],
                   slice_max_shown=[float(np.abs(a).max()) for a in slices_norm],
                   E_slice=d["snap__E_slice"].tolist(),
                   E_slice_ratio_last=(d["snap__E_slice"][-1] / d["snap__E_slice"][0]).tolist(),
                   budget_residual_last=((d["E_slice_series"][-1] + d["D_slice_series"][-1])
                                         / d["E_slice_series"][0] - 1.0).tolist(),
                   decoupling=dict(err_l2_per_slice_max=d["snap__dec_err_l2"].max(axis=1).tolist(),
                                   err_l2_global_max=d["snap__dec_err_l2_global"].max(axis=1).tolist(),
                                   err_max_max=d["snap__dec_err_max"].max(axis=1).tolist(),
                                   control_l2_min=[float(np.min(c[np.abs(u) > 1e-12]))
                                                   for c in d["snap__dec_control_l2"]],
                                   bracket_over_streaming=[[float(np.min(r[np.isfinite(r)])),
                                                            float(np.max(r[np.isfinite(r)]))]
                                                           for r in d["snap__dec_ratio"]]),
                   gates=meta["gates"])

    def draw(plt):
        from matplotlib import gridspec
        W = style["double_width_in"]
        fig = plt.figure(figsize=(W, 1.68))
        gs = gridspec.GridSpec(1, 9, figure=fig, width_ratios=[1, 1, 0.06, 0.5, 1, 1, 1, 0.06, 0.02],
                               wspace=0.16, left=0.065, right=0.93, bottom=0.21, top=0.86)
        axes = [fig.add_subplot(gs[0, i]) for i in (0, 1, 4, 5, 6)]
        cax1, cax2 = fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[0, 7])
        v1 = float(max(np.abs(g0[0]).max(), np.abs(g0[-1]).max()))
        v2 = float(max(np.abs(a).max() for a in slices_norm))
        imgs = [g0[0], g0[-1]] + slices_norm
        vm = [v1, v1, v2, v2, v2]
        labels = [r"(a) $t=0$", rf"(b) $t={t_out[-1] / tau:.1f}\,t_{{\rm nl}}$"] + \
                 [(rf"({c}) $su_j={s * u[j]:+.2f}$" if abs(s * u[j]) > 1e-9 else rf"({c}) $u_j=0$")
                  for c, j in zip("cde", j_show)]
        for k, (ax, im, v, lab) in enumerate(zip(axes, imgs, vm, labels)):
            h = ax.imshow(im, origin="lower", extent=ext, cmap="viridis", vmin=-v, vmax=v, interpolation="bilinear")
            ax.set_xticks([0, 0.5 * L, L])
            ax.set_yticks([0, 0.5 * L, L])
            ax.set_xticklabels(["0", "0.5", "1"])
            ax.set_yticklabels(["0", "0.5", "1"] if k == 0 else [])
            ax.tick_params(direction="out", top=False, right=False, length=2.5)
            ax.set_xlabel(r"$x/L$", labelpad=1)
            if k == 0:
                ax.set_ylabel(r"$y/L$", labelpad=1)
            ax.text(0.0, 1.03, lab, transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=style["legend_font_size_pt"] + 1)
            if k == 0:
                cb1 = fig.colorbar(h, cax=cax1)
            if k == 2:
                cb2 = fig.colorbar(h, cax=cax2)
        for cb in (cb1, cb2):
            cb.ax.tick_params(labelsize=style["legend_font_size_pt"], length=2, direction="out")
        cb1.ax.set_title(r"$g_0$", fontsize=style["legend_font_size_pt"] + 1, pad=4)
        cb2.set_label(r"$a_j/(p_0R_{0j})$", labelpad=2)
        return fig

    numbers["rendering"] = render(draw, NAME, style)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_DIR / f"fig_{NAME}.json", "w") as fh:
        json.dump(numbers, fh, indent=1, default=float)
        fh.write("\n")
    print(f"run {meta['run_id']}: chi = {numbers['chi']['chi_z']:.3f}, tau = {tau:.4f}, t_out/tau = "
          f"{np.round(t_out / tau, 3).tolist()}; slices shown u_j = {np.round(numbers['u_shown'], 4).tolist()} "
          f"(s u_j = {np.round(numbers['su_shown'], 4).tolist()})")
    print("decoupling (per output time): max rel. L2 per slice", np.array(numbers["decoupling"]["err_l2_per_slice_max"]),
          "; of the largest slice rate", np.array(numbers["decoupling"]["err_l2_global_max"]),
          "; control min", np.round(numbers["decoupling"]["control_l2_min"], 3))
    print("E_j(t_last)/E_j(0):", np.round(numbers["E_slice_ratio_last"], 4).tolist())
    print(f"wrote figures/{NAME}.pdf and .png ({numbers['rendering']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
