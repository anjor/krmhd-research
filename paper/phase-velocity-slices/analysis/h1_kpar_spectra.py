#!/usr/bin/env python
"""Field-aligned spectra of the free energy against the perpendicular resolution, and the band-resolved
field-aligned wavenumbers of g_0 against those of its frozen copy (figure h1_kpar_spectra, section 7).

(a) The spectrum of the free energy along the traced field lines, sum_m E_par[g_m](kappa), and along z,
sum_m E_z[g_m](k_z), each normalised to unit integral, for the runs given with --nseries (one per resolution), with
the spectrum of the injected pattern traced along the same lines.  A parallel cascade of the free energy would extend
the field-aligned spectrum as the dissipation scale moves to smaller scales with the resolution.
(b) The spectra along the same lines of the whole Alfvenic velocity u_perp and of the density moment g_0
(h1_alfven_lines.py), for the runs given with --alfven: the comparison with Meyrand et al. (2019, figure 3), who
found that the parallel spectra of the compressive fluctuations along the perturbed field lines track that of u_perp.
(c) The mean field-aligned wavenumber kappa_1 in k_perp bands of g_0, of its frozen copy (h1_frozen_copy.py) and of
Phi, for the runs given with --frozen.
(d) The kappa-weighted spectra kappa W(kappa) of (a), each normalised to unit integral: kappa W is the phase-mixing
flux s kappa W per unit c_m, so (d) shows at which kappa the flux is carried.  The dash-dotted line is the same for
the frozen-kappa model of the last --nseries run (frozen_model: every kappa of the traced injection phase mixes at its
own kappa against 1/T and the hypercollisions), the spectrum with no parallel cascade.
Every share is also quoted at one threshold, K_ONE = 4 k1, for every field and direction, as a free-energy share and
as a kappa-weighted share, with the range over snapshots.

Reads <run dir>/fieldlines_snap_*.npz (h1_fieldlines.py: kappa_par, kz, g<m>__E_par, g<m>__E_z, F__E_par),
<run dir>/frozen_copy_snap_*.npz/.json (h1_frozen_copy.py), <run dir>/alfven_lines_snap_*.npz (h1_alfven_lines.py:
kappa_par, u__E_par, b__E_par, g0__E_par), <run dir>/h1_analysis.json (no_echo.coefficient_c_m, P_a.inv_T) and
<run dir>/<run>.json (config), averaging over the snapshots of each run.  Prints the numbers quoted in the paper and writes
figures/h1_kpar_spectra.{pdf,png} and data/h1_kpar_spectra.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_kpar_spectra.py \
        --nseries <run 64^3> <run 96^3> <run 128^3> <run 256^3> --frozen <run 64^3> <run 128^3> \
        --alfven <run 64^3> <run 96^3> <run 128^3> <run 256^3>
The 256^3 alfven_lines_ files were made on Modal next to the snapshots (scripts/modal_analysis.py running
h1_alfven_lines.py, 2026-11-26); their g0 spectra equal those of the fieldlines_ files of the same snapshots exactly.
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

K1 = 2.0 * np.pi        # box wavenumber 2 pi / L with L = 1 in every run
K_ONE = 4.0             # the one threshold, in units of k1, of the shares quoted for every field and direction


def run_info(run: str) -> dict:
    cfg = json.load(open(P.DATA / run / f"{run}.json"))["config"]
    return dict(N=int(cfg["box"]["N"]), M=int(cfg["velocity"]["M"]), beta=P.parse_number(cfg["velocity"]["beta_i"]))


def free_energy_spectra(run: str) -> dict:
    """Snapshot means of the free-energy spectra along the lines (sum over m of E_par[g_m]) and along z, of the
    injected pattern's spectrum along the lines, and of the flux spectrum along the lines sum_m c_m kappa E_par[g_m]
    (c_m of h1_analysis.json, m < M), with the per-snapshot spectra (key "snap") for the spread; kappa and k_z in units
    of k1."""
    files = sorted((P.DATA / run).glob("fieldlines_snap_*.npz"))
    if not files:
        raise FileNotFoundError(f"no fieldlines_ files in {run}")
    zs = [np.load(f) for f in files]
    M = run_info(run)["M"]
    c_m = np.array(json.load(open(P.DATA / run / "h1_analysis.json"))["no_echo"]["coefficient_c_m"], dtype=float)
    kap = zs[0]["kappa_par"] / K1
    snap = dict(W_par=[sum(z[f"g{m}__E_par"] for m in range(M + 1)) for z in zs],
                W_z=[sum(z[f"g{m}__E_z"] for m in range(M + 1)) for z in zs],
                F_par=[z["F__E_par"] for z in zs],
                Pi_par=[kap * sum(c_m[m] * z[f"g{m}__E_par"] for m in range(M)) for z in zs],
                **{f"g{m}_par": [z[f"g{m}__E_par"] for z in zs] for m in (2, 8, 16, 32) if m <= M})
    out = {k: np.mean(v, axis=0) for k, v in snap.items()}
    return dict(kap=kap, kz=zs[0]["kz"] / K1, n=len(zs), snap=snap, **out)


def frozen_model(run: str, kap: np.ndarray, PF: np.ndarray) -> dict:
    """Field-aligned spectra the free energy would have with no parallel cascade: each kappa of the injected pattern's
    traced spectrum PF phase mixes at its own kappa against the uniform sink 1/T and the run's hypercollisions
    (h1_pilot.lyapunov_mode), so W(kappa) = PF(kappa) sum_m w_m(kappa).  T is that of h1_analysis.json (P_a.inv_T).
    Returns W(kappa) and kappa W(kappa) on the kappa grid (units of k1)."""
    from h1_pilot import lyapunov_mode
    cfg = json.load(open(P.DATA / run / f"{run}.json"))["config"]
    v, st = cfg["velocity"], cfg["stepper"]
    sl = P.make_slices(int(v["M"]), P.parse_number(v["Lambda"]), v["closure"])
    s = float(np.sqrt(P.parse_number(v["beta_i"])))
    T = 1.0 / json.load(open(P.DATA / run / "h1_analysis.json"))["P_a"]["inv_T"]
    nu, hn = P.parse_number(st["nu"]), int(st["hyper_n"])
    W = np.zeros_like(kap)
    for i, (k, w) in enumerate(zip(kap, PF)):
        if w > 0:
            W[i] = w * (T if k <= 0 else lyapunov_mode(sl, s, float(k * K1), T, nu, hn)["W"].sum())
    return dict(W=W, T=T)


def frozen_bands(run: str) -> dict:
    """Snapshot means of the band-resolved and whole-field spectra along the lines of f, g0 and Phi."""
    files = sorted((P.DATA / run).glob("frozen_copy_snap_*.npz"))
    if not files:
        raise FileNotFoundError(f"no frozen_copy_ files in {run}")
    zs = [np.load(f) for f in files]
    js = [json.load(open(f.with_suffix(".json"))) for f in files]
    bands = js[0]["bands"]
    out: dict = dict(kap=zs[0]["kappa_par"] / K1, bands=bands, n=len(zs),
                     var_ratio=float(np.mean([j["var_f_over_var_g0"] for j in js])))
    for nm in ("f", "g0", "phi"):
        out[nm] = np.mean([z[f"{nm}__E_par"] for z in zs], axis=0)
        for i in range(len(bands)):
            out[f"{nm}_band{i}"] = np.mean([z[f"{nm}_band{i}__E_par"] for z in zs], axis=0)
    return out


def alfven_lines(run: str) -> dict:
    """Snapshot means of the spectra along the lines of the whole u_perp, b_perp and g0 (h1_alfven_lines.py)."""
    files = sorted((P.DATA / run).glob("alfven_lines_snap_*.npz"))
    if not files:
        raise FileNotFoundError(f"no alfven_lines_ files in {run}")
    zs = [np.load(f) for f in files]
    out: dict = dict(kap=zs[0]["kappa_par"] / K1, n=len(zs), snap={})
    for nm in ("u", "b", "g0"):
        out["snap"][nm] = [z[f"{nm}__E_par"] for z in zs]
        out[nm] = np.mean(out["snap"][nm], axis=0)
    return out


def share_above(k: np.ndarray, E: np.ndarray, k0: float) -> float:
    return float(E[k > k0 + 1e-9].sum() / E.sum())


def kshare_above(k: np.ndarray, E: np.ndarray, k0: float) -> float:
    """Share of sum kappa E(kappa) beyond k0: for the free energy, the share of its phase-mixing flux s kappa W that
    is carried beyond k0 (up to the m-dependent c_m)."""
    return share_above(k, k * E, k0)


def snap_range(f, k: np.ndarray, Es: list, k0: float) -> list[float]:
    vals = [f(k, E, k0) for E in Es]
    return [float(min(vals)), float(max(vals))]


def mean_k(k: np.ndarray, E: np.ndarray) -> float:
    return float((k * E).sum() / E.sum())


def density(k: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Spectral density per unit kappa/k1 with unit integral (bins of constant width); empty bins (beyond the
    dealiasing edge) are NaN so that they are not drawn on logarithmic axes."""
    d = E / E.sum() / (k[2] - k[1])
    return np.where(d > 0, d, np.nan)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nseries", nargs="+", required=True, help="runs at increasing N (fieldlines_ files)")
    ap.add_argument("--frozen", nargs="+", required=True, help="runs with frozen_copy_ files")
    ap.add_argument("--alfven", nargs="+", required=True, help="runs with alfven_lines_ files")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    out: dict = dict(nseries={}, frozen={}, alfven={})
    spec = {}
    for r in args.nseries:
        s = free_energy_spectra(r)
        info = run_info(r)
        spec[r] = s
        row = dict(N=info["N"], M=info["M"], beta_i=info["beta"], snapshots=s["n"],
                   mean_kappa_par=mean_k(s["kap"], s["W_par"]), mean_kz=mean_k(s["kz"], s["W_z"]),
                   injected_mean_kappa=mean_k(s["kap"], s["F_par"]))
        for k0 in (2, 4, 8):
            row[f"share_par_above_{k0}"] = share_above(s["kap"], s["W_par"], k0)
            row[f"share_z_above_{k0}"] = share_above(s["kz"], s["W_z"], k0)
            row[f"injected_share_above_{k0}"] = share_above(s["kap"], s["F_par"], k0)
        # one threshold, K_ONE, for every field and direction: the free-energy share and the kappa-weighted share
        # (the share of the phase-mixing flux), with the range over snapshots
        sn = s["snap"]
        row["at_k0"] = K_ONE
        row["W_par"] = dict(share=share_above(s["kap"], s["W_par"], K_ONE), kw_share=kshare_above(s["kap"], s["W_par"], K_ONE),
                            share_snaps=snap_range(share_above, s["kap"], sn["W_par"], K_ONE),
                            kw_share_snaps=snap_range(kshare_above, s["kap"], sn["W_par"], K_ONE))
        row["W_z"] = dict(share=share_above(s["kz"], s["W_z"], K_ONE), kw_share=kshare_above(s["kz"], s["W_z"], K_ONE),
                          share_snaps=snap_range(share_above, s["kz"], sn["W_z"], K_ONE),
                          kw_share_snaps=snap_range(kshare_above, s["kz"], sn["W_z"], K_ONE))
        row["F_par"] = dict(share=share_above(s["kap"], s["F_par"], K_ONE), kw_share=kshare_above(s["kap"], s["F_par"], K_ONE),
                            share_snaps=snap_range(share_above, s["kap"], sn["F_par"], K_ONE))
        row["Pi_par"] = dict(share=share_above(s["kap"], s["Pi_par"], K_ONE),
                             share_snaps=snap_range(share_above, s["kap"], sn["Pi_par"], K_ONE))
        row["kw_share_par_m"] = {m[1:-4]: kshare_above(s["kap"], s[m], K_ONE) for m in s if m.startswith("g") and m.endswith("_par")}
        # where the kappa-weighted tail sits: 4 < kappa <= 8 against kappa > 8
        kw = s["kap"] * s["W_par"]
        row["W_par"]["kw_share_4_to_8"] = float(kw[(s["kap"] > K_ONE + 1e-9) & (s["kap"] <= 8 + 1e-9)].sum() / kw.sum())
        row["W_par"]["kw_share_above_8"] = kshare_above(s["kap"], s["W_par"], 8)
        fm = frozen_model(r, s["kap"], s["F_par"])
        s["model_W"] = fm["W"]
        row["frozen_model"] = dict(T=fm["T"], share=share_above(s["kap"], fm["W"], K_ONE),
                                   kw_share=kshare_above(s["kap"], fm["W"], K_ONE),
                                   kw_share_above_8=kshare_above(s["kap"], fm["W"], 8),
                                   mean_kappa=mean_k(s["kap"], fm["W"]))
        out["nseries"][r] = row
        print(f"N = {info['N']:3d} (beta_i = {info['beta']}, M = {info['M']}, {s['n']} snapshots): free energy "
              f"<kappa>/k1 along lines {row['mean_kappa_par']:.2f}, <k_z>/k1 {row['mean_kz']:.2f}; share beyond "
              + ", ".join(f"{k0} k1: {100 * row[f'share_par_above_{k0}']:.3g}% along lines, "
                          f"{100 * row[f'share_z_above_{k0}']:.3g}% along z (injected {100 * row[f'injected_share_above_{k0}']:.3g}%)"
                          for k0 in (2, 4, 8)))
    fz = {}
    for r in args.frozen:
        d = frozen_bands(r)
        info = run_info(r)
        fz[r] = d
        nb = len(d["bands"])
        row = dict(N=info["N"], beta_i=info["beta"], snapshots=d["n"], bands=d["bands"], var_f_over_var_g0=d["var_ratio"])
        for nm in ("f", "g0", "phi"):
            row[f"{nm}_kappa1_bands"] = [mean_k(d["kap"], d[f"{nm}_band{i}"]) for i in range(nb)]
            row[f"{nm}_kappa1_whole"] = mean_k(d["kap"], d[nm])
            row[f"{nm}_share_above_4_whole"] = share_above(d["kap"], d[nm], 4)
        out["frozen"][r] = row
        print(f"N = {info['N']:3d} frozen copy ({d['n']} snapshots, var(f)/var(g0) = {d['var_ratio']:.2f}): kappa_1/k1 in "
              f"bands {d['bands']}")
        for nm in ("g0", "f", "phi"):
            print(f"   {nm:4s} " + " ".join(f"{x:.2f}" for x in row[f"{nm}_kappa1_bands"])
                  + f"   whole {row[f'{nm}_kappa1_whole']:.2f}, share beyond 4 k1 {100 * row[f'{nm}_share_above_4_whole']:.3g}%")
    alf = {}
    for r in args.alfven:
        d = alfven_lines(r)
        info = run_info(r)
        alf[r] = d
        row = dict(N=info["N"], beta_i=info["beta"], snapshots=d["n"])
        for nm in ("u", "b", "g0"):
            row[f"{nm}_mean_kappa"] = mean_k(d["kap"], d[nm])
            for k0 in (2, 4, 8):
                row[f"{nm}_share_above_{k0}"] = share_above(d["kap"], d[nm], k0)
            row[nm] = dict(share=share_above(d["kap"], d[nm], K_ONE), kw_share=kshare_above(d["kap"], d[nm], K_ONE),
                           share_snaps=snap_range(share_above, d["kap"], d["snap"][nm], K_ONE),
                           kw_share_snaps=snap_range(kshare_above, d["kap"], d["snap"][nm], K_ONE))
        out["alfven"][r] = row
        print(f"N = {info['N']:3d} whole fields along lines ({d['n']} snapshots): " + "; ".join(
            f"{nm} <kappa>/k1 {row[f'{nm}_mean_kappa']:.2f}, beyond 2/4/8 k1 " + "/".join(
                f"{100 * row[f'{nm}_share_above_{k0}']:.3g}%" for k0 in (2, 4, 8)) for nm in ("u", "b", "g0")))
    with open(P.DATA / "h1_kpar_spectra.json", "w") as fh:
        json.dump(P._jsonable(out), fh, indent=1)

    def pc(d: dict, k: str) -> str:
        lo, hi = d[f"{k}_snaps"]
        return f"{100 * d[k]:.3g}% ({100 * lo:.3g}-{100 * hi:.3g})"
    print(f"\nshares beyond {K_ONE:g} k1 (free-energy share | kappa-weighted share; range over snapshots in brackets)")
    for r in args.nseries:
        row = out["nseries"][r]
        print(f"N = {row['N']:3d}: free energy along lines {pc(row['W_par'], 'share')} | {pc(row['W_par'], 'kw_share')};"
              f" along z {pc(row['W_z'], 'share')} | {pc(row['W_z'], 'kw_share')}; injected pattern "
              f"{pc(row['F_par'], 'share')}; flux sum_m c_m kappa W along lines {pc(row['Pi_par'], 'share')}; "
              "kappa-weighted at m = " + ", ".join(f"{m}: {100 * v:.3g}%" for m, v in row["kw_share_par_m"].items()))
        fm = row["frozen_model"]
        print(f"        kappa-weighted along lines in 4-8 k1 {100 * row['W_par']['kw_share_4_to_8']:.3g}%, beyond 8 k1 "
              f"{100 * row['W_par']['kw_share_above_8']:.3g}%; frozen-kappa model (T = {fm['T']:.2f}): free energy "
              f"{100 * fm['share']:.3g}%, kappa-weighted {100 * fm['kw_share']:.3g}% (beyond 8 k1 "
              f"{100 * fm['kw_share_above_8']:.3g}%), <kappa>/k1 {fm['mean_kappa']:.2f}")
    for r in args.alfven:
        row = out["alfven"][r]
        print(f"N = {row['N']:3d}: u_perp along lines {pc(row['u'], 'share')} | {pc(row['u'], 'kw_share')}; "
              f"g0 {pc(row['g0'], 'share')} | {pc(row['g0'], 'kw_share')}")

    style = FS.style_config()
    pal = style["palette"]
    blk = style["reference_color"]

    def draw(plt):
        from matplotlib.lines import Line2D
        from matplotlib.ticker import NullFormatter, ScalarFormatter
        fig, axs = plt.subplots(2, 2, figsize=(style["double_width_in"], 4.9))
        ax = [axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]]
        a = ax[0]
        handles = []
        for r, c in zip(args.nseries, pal):
            s = spec[r]
            a.loglog(s["kap"][1:], density(s["kap"], s["W_par"])[1:], "-", color=c, lw=1.0)
            a.loglog(s["kz"][1:], density(s["kz"], s["W_z"])[1:], ":", color=c, lw=1.0)
            handles.append(Line2D([], [], color=c, lw=1.0, label=rf"${out['nseries'][r]['N']}^3$"))
        s0 = spec[args.nseries[0]]                   # injected pattern traced on the lines of the first run (the
        a.loglog(s0["kap"][1:], density(s0["kap"], s0["F_par"])[1:], "--", color=blk, lw=0.8)   # step differs at 256^3)
        handles += [Line2D([], [], color=blk, ls="-", lw=1.0, label="along field lines"),
                    Line2D([], [], color=blk, ls=":", lw=1.0, label=r"along $z$"),
                    Line2D([], [], color=blk, ls="--", lw=0.8, label="injected pattern")]
        a.set_xlim(0.25, 90)
        a.set_ylim(1e-8, 5)
        a.set_xlabel(r"$\kappa/k_1$, $k_z/k_1$")
        a.set_ylabel(r"free-energy spectrum")
        a.legend(handles=handles, fontsize=5.5, loc="lower left", ncol=1)
        FS.panel_label(a, "(a)")
        a = ax[2]
        handles = []
        for r, ls in zip(args.frozen, ("--", "-")):
            d = out["frozen"][r]
            kc = np.array([0.5 * (max(b[0], 1) + b[1]) for b in d["bands"]])
            for nm, c, mk in (("g0", pal[0], "o"), ("f", pal[1], "s"), ("phi", pal[2], "^")):
                a.loglog(kc, d[f"{nm}_kappa1_bands"], ls, marker=mk, color=c, ms=3, lw=1.0,
                         mfc=c if ls == "-" else "none")
            handles.append(Line2D([], [], color=blk, ls=ls, lw=1.0, label=rf"${d['N']}^3$"))
        handles = [Line2D([], [], color=pal[0], marker="o", ms=3, lw=1.0, label=r"$g_0$"),
                   Line2D([], [], color=pal[1], marker="s", ms=3, lw=1.0, label=r"frozen copy of $g_0$"),
                   Line2D([], [], color=pal[2], marker="^", ms=3, lw=1.0, label=r"$\Phi$")] + handles
        kin = np.mean([out["nseries"][r]["injected_mean_kappa"] for r in args.nseries])
        a.axhline(kin, color="0.5", lw=0.6, ls="--")
        handles.append(Line2D([], [], color="0.5", ls="--", lw=0.6, label="injected"))
        a.set_xlabel(r"$k_\perp/k_1$ (band centre)")
        a.set_ylabel(r"$\kappa_1/k_1$ along field lines")
        for axis in (a.xaxis, a.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        a.set_xticks([1.5, 3, 6, 12])
        a.set_yticks([1, 2, 3])
        a.set_ylim(0.8, 6.5)
        a.legend(handles=handles, fontsize=5.5, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.6)
        FS.panel_label(a, "(c)")
        a = ax[1]
        handles = []
        styles = ("--", ":", "-.", "-")[-len(args.alfven):] if len(args.alfven) == 4 else ("--", ":", "-")[-len(args.alfven):]
        for r, ls in zip(args.alfven, styles):
            d = alf[r]
            for nm, c in (("u", pal[2]), ("g0", pal[0])):
                a.loglog(d["kap"][1:], density(d["kap"], d[nm])[1:], ls, color=c, lw=1.0)
            handles.append(Line2D([], [], color=blk, ls=ls, lw=1.0, label=rf"${out['alfven'][r]['N']}^3$"))
        handles = [Line2D([], [], color=pal[2], lw=1.0, label=r"$\boldsymbol u_\perp$"),
                   Line2D([], [], color=pal[0], lw=1.0, label=r"$g_0$")] + handles
        a.set_xlim(0.25, 60)
        a.set_ylim(1e-7, 5)
        a.set_xlabel(r"$\kappa/k_1$")
        a.set_ylabel(r"spectrum along field lines")
        a.legend(handles=handles, fontsize=6.5, loc="lower left", ncol=1)
        FS.panel_label(a, "(b)")
        # (d) the kappa-weighted spectra: the share of the phase-mixing flux s kappa W carried at each kappa
        a = ax[3]
        handles = []
        for r, c in zip(args.nseries, pal):
            s = spec[r]
            a.loglog(s["kap"][1:], density(s["kap"], s["kap"] * s["W_par"])[1:], "-", color=c, lw=1.0)
            a.loglog(s["kz"][1:], density(s["kz"], s["kz"] * s["W_z"])[1:], ":", color=c, lw=1.0)
            handles.append(Line2D([], [], color=c, lw=1.0, label=rf"${out['nseries'][r]['N']}^3$"))
        s = spec[args.nseries[-1]]
        a.loglog(s["kap"][1:], density(s["kap"], s["kap"] * s["model_W"])[1:], "-.", color="0.45", lw=0.9)
        a.axvline(K_ONE, color="0.6", lw=0.5, ls="--")
        handles += [Line2D([], [], color=blk, ls="-", lw=1.0, label="along field lines"),
                    Line2D([], [], color=blk, ls=":", lw=1.0, label=r"along $z$"),
                    Line2D([], [], color="0.45", ls="-.", lw=0.9,
                           label=rf"frozen $\kappa$, ${out['nseries'][args.nseries[-1]]['N']}^3$")]
        a.set_xlim(0.25, 90)
        a.set_ylim(1e-6, 5)
        a.set_xlabel(r"$\kappa/k_1$, $k_z/k_1$")
        a.set_ylabel(r"$\kappa$-weighted free-energy spectrum")
        a.legend(handles=handles, fontsize=5.5, loc="lower left", ncol=1)
        FS.panel_label(a, "(d)")
        fig.tight_layout()
        return fig

    if args.outdir:
        FS.FIG_DIR = Path(args.outdir)
    mode = FS.render(draw, "h1_kpar_spectra", style)
    print(f"wrote figures/h1_kpar_spectra.pdf/.png ({mode}) and data/h1_kpar_spectra.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
