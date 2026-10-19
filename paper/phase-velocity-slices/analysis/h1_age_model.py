#!/usr/bin/env python
"""The two-channel model with a perpendicular sink that depends on the age of the free energy.

Each injection into g_0 at field-aligned wavenumber kappa evolves by the linear hierarchy (streaming i s kappa A and
the run's hypercollisions), and the perpendicular cascade removes it at a rate gamma(t) that depends only on the time
t since injection, the same at every m.  With E_m(t; kappa) the free energy at moment m of a unit injection and
S(t) = exp(-int_0^t gamma), the steady state is
    W(m) = eps sum_kappa P(kappa) int_0^inf E_m(t; kappa) S(t) dt,
    D_perp(m) = eps sum_kappa P(kappa) int_0^inf E_m(t; kappa) gamma(t) S(t) dt.
For gamma = 1/T this is the uniform-sink model of h1_pilot.py (its Lyapunov solution).  The script fits
gamma(t) = g0 + (g1 - g0)(1 - exp(-t/t_c)) to D_perp(m)/W(m), 1 <= m <= M/2, of one run, then predicts W(m) and
D_perp(m)/W(m) of the other runs with that gamma(t) (the Alfvenic turbulence is the same in all of them) and their own
s, Lambda, hypercollisions and traced P(kappa) of the injected pattern; no parameter is fitted to them.  The uniform
sink with each run's own T is evaluated in the same way for comparison.  Writes data/h1_age_model.json and
figures/h1_age_model.{pdf,png}.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_age_model.py --fit <run> --predict <run> [<run> ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import figstyle as FS  # noqa: E402
import p1lib as P  # noqa: E402
import h1lib as H  # noqa: E402

T_MAX, DT = 40.0, 0.004          # time grid of the Green's functions, in tau_A


def load(run_id: str, acc: dict) -> dict:
    """Measured window averages and the traced P(kappa) of the injected pattern of one run."""
    rd = P.DATA / run_id
    meta = json.load(open(rd / f"{run_id}.json"))
    with np.load(rd / f"{run_id}.npz") as z:
        ts = {k[4:]: z[k] for k in z.files if k.startswith("ts__")}
    cfg = meta["config"]
    width = float((acc.get("runs") or {}).get(run_id, {}).get("window", cfg["gates"]["steady_window"]))
    t, dtk = ts["t"], ts["acc_sum_dt"]
    first = int(np.where(H.window_slice(t, width))[0][0])
    Tw = float(dtk[first + 1:].sum())
    W = H.time_average(ts["pWm"], dtk, first)
    eps = float(ts["acc_inj_g"][first + 1:].sum() / Tw)
    Dp = ts["acc_loss_eta"][first + 1:].sum(0) / Tw
    fl = sorted(glob.glob(str(rd / "fieldlines_snap_*.npz")))
    zs = [np.load(f) for f in fl]
    kap = zs[0]["kappa_par"]
    PF = np.mean([z["F__E_par"] for z in zs], axis=0)
    keep = (kap > 0) & (PF > 1e-4 * PF.max())
    v, st = cfg["velocity"], cfg["stepper"]
    return dict(run_id=run_id, beta=P.parse_number(v["beta_i"]), M=int(v["M"]), N=int(cfg["box"]["N"]),
                Lam=P.parse_number(v["Lambda"]), closure=v["closure"], nu=P.parse_number(st["nu"]),
                hyper_n=int(st["hyper_n"]), W=W, eps=eps, q=Dp / W, kap=kap[keep], P=PF[keep] / PF[keep].sum())


def green(d: dict) -> tuple[np.ndarray, np.ndarray]:
    """Ebar[m, t] = sum_kappa P(kappa) E_m(t; kappa) for a unit injection into g_0, on the time grid."""
    sl = P.make_slices(d["M"], d["Lam"], d["closure"])
    s = math.sqrt(d["beta"])
    coll = P.coll_rates(d["M"], d["nu"], d["hyper_n"])
    t = np.arange(0.0, T_MAX + DT / 2, DT)
    g0 = np.zeros(d["M"] + 1, complex)
    g0[0] = 1.0 / math.sqrt(sl.p[0])
    Ebar = np.zeros((d["M"] + 1, len(t)))
    for kap, w in zip(d["kap"], d["P"]):
        L = 1j * s * kap * sl.A + np.diag(coll)
        lam, V = np.linalg.eig(L)
        c = np.linalg.solve(V, g0)
        G = V @ (np.exp(-np.outer(lam, t)) * c[:, None])
        Ebar += w * sl.p[:, None] * np.abs(G) ** 2
    return t, Ebar


def gamma_of(t: np.ndarray, prm) -> np.ndarray:
    g0, g1, tc = prm
    return g0 + (g1 - g0) * (1.0 - np.exp(-t / tc))


def steady(t: np.ndarray, Ebar: np.ndarray, gam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """W(m)/eps and D_perp(m)/eps for the sink rate gam(t)."""
    S = np.exp(-np.concatenate([[0.0], np.cumsum(0.5 * (gam[1:] + gam[:-1]) * np.diff(t))]))
    wts = np.full(len(t), DT)
    wts[0] = wts[-1] = DT / 2
    return Ebar @ (S * wts), Ebar @ (gam * S * wts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fit", required=True)
    ap.add_argument("--predict", nargs="+", required=True)
    ap.add_argument("--acceptance", default="h1_acceptance.yaml")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()
    acc = P.load_config(args.acceptance)
    runs = [load(r, acc) for r in [args.fit] + args.predict]
    gr = {d["run_id"]: green(d) for d in runs}

    f = runs[0]
    t, Eb = gr[f["run_id"]]
    mf = np.arange(1, f["M"] // 2 + 1)

    def resid(prm):
        W, D = steady(t, Eb, gamma_of(t, prm))
        return np.log(D[mf] / W[mf]) - np.log(f["q"][mf])

    fit = least_squares(resid, x0=[0.35, 0.65, 1.0], bounds=([0.0, 0.0, 0.01], [3.0, 3.0, 20.0]))
    prm = fit.x
    print(f"fit to {f['run_id']}: gamma(t) = {prm[0]:.3f} + ({prm[1]:.3f} - {prm[0]:.3f})(1 - exp(-t/{prm[2]:.3f})); "
          f"rms residual in ln(D_perp/W) {np.sqrt(np.mean(fit.fun ** 2)):.3f}")
    out = dict(fit=f["run_id"], gamma_params=dict(g0=prm[0], g1=prm[1], t_c=prm[2]), runs={})
    curves = []
    for d in runs:
        t, Eb = gr[d["run_id"]]
        Wa, Da = steady(t, Eb, gamma_of(t, prm))
        invT = float(np.median(d["q"][2:17]))
        Wu, _ = steady(t, Eb, np.full(len(t), invT))
        ms = np.arange(1, d["M"] // 2 + 1)
        ra, ru = d["W"][ms] / (d["eps"] * Wa[ms]), d["W"][ms] / (d["eps"] * Wu[ms])
        qa = Da[ms] / Wa[ms]
        out["runs"][d["run_id"]] = dict(beta=d["beta"], N=d["N"], M=d["M"], m=ms.tolist(), W_over_age=ra.tolist(),
                                        W_over_uniform=ru.tolist(), Dperp_over_W_model=qa.tolist(),
                                        Dperp_over_W_meas=d["q"][ms].tolist())
        print(f"{d['run_id']} (beta {d['beta']:g}, {d['N']}^3, M {d['M']}{', FIT' if d is f else ''}): "
              f"W/age-model {ra.min():.3f}-{ra.max():.3f}; W/uniform {ru.min():.3f}-{ru.max():.3f}; "
              f"D_perp/W model/meas {np.min(qa / d['q'][ms]):.3f}-{np.max(qa / d['q'][ms]):.3f}")
        curves.append((d, ms, ra, ru, d is f))
    (P.DATA / "h1_age_model.json").write_text(json.dumps(P._jsonable(out), indent=1))

    fcfg = FS.style_config()
    plt = FS.setup_style(fcfg, usetex=True)
    pal = fcfg["palette"]
    fig, a = plt.subplots(1, 1, figsize=(fcfg["single_width_in"], 2.7))
    for i, (d, ms, ra, ru, isfit) in enumerate(curves):
        c = pal[i % len(pal)]
        a.plot(ms, ra, "-", color=c, lw=1.1, label=rf"$\beta_i={d['beta']:g}$" + (" (fit)" if isfit else ""))
        a.plot(ms, ru, ":", color=c, lw=0.9)
    a.axhline(1.0, color="0.5", lw=0.5)
    a.set_xscale("log")
    a.set_xlabel(r"$m$")
    a.set_ylabel(r"$W(m)/W_{\rm model}(m)$")
    a.legend(frameon=False, fontsize=7)
    outdir = Path(args.outdir) if args.outdir else FS.FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "h1_age_model.pdf", bbox_inches="tight")
    fig.savefig(outdir / "h1_age_model.png", dpi=fcfg["png_dpi"], bbox_inches="tight")
    print(f"wrote {outdir / 'h1_age_model'}.pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
