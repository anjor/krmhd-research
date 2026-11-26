#!/usr/bin/env python
"""The s = 0 control of H1: the residence time of a passive scalar in the same Alfvenic turbulence.

With beta_i = 1e-8 the streaming speed s = beta_i^{1/2} is 1e-4, so g_0 does not phase mix: it is advected by the
E x B flow and removed only by the perpendicular hyper-diffusion.  Its residence time T_ps = <W_0>/<D_perp,0> (free
energy over its dissipation rate, in tau_A = L_z/v_A) is the time in which the perpendicular cascade removes a passive
scalar.  H1 says the streaming runs lose free energy at every m to the same cascade.  The registered predictions are:
  - the age-dependent sink of analysis/h1_age_model.py (data/h1_age_model.json, fitted to the beta_i = 0.1, M = 64
    run): a passive scalar holds every age, so T_ps = int_0^inf exp(-int_0^theta gamma) dtheta;
  - the uniform-sink T of the paired streaming run (beta_i = 1, 64^3, M = 32: same forcing, seeds, eta, nu and
    hand-off), the median of W(m)/D_perp(m) over 2 <= m <= 16 (h1_pilot.py).

Reads only saved data: <run dir>/<run ID>.npz/.json (scripts/run_forced_turbulence.py), data/h1_age_model.json and
the paired run's h1_analysis.json.  Writes data/h1_s0.json.  It reports the gates the run recorded and the block
check of the streaming runs (configs/h1_acceptance.yaml); it does not decide acceptance.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_s0.py P1_forced-s0dt_20261126_195653
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "scripts"))
import h1lib as H  # noqa: E402
import p1lib as P  # noqa: E402

PAIRED = "P1_forced-beta1_20261015_122613"


def residence_time(g0: float, g1: float, t_c: float) -> float:
    """T_ps = int_0^inf S(theta) dtheta for the sink gamma(theta) = g0 + (g1 - g0)(1 - exp(-theta/t_c)) [1/tau_A],
    with S = exp(-int_0^theta gamma) the fraction of free energy of age theta still present."""
    th = np.linspace(0.0, 60.0 / g1, 600001)
    S = np.exp(-(g1 * th - (g1 - g0) * t_c * (1.0 - np.exp(-th / t_c))))
    return float(np.trapezoid(S, th))


def correlation_time(t: np.ndarray, x: np.ndarray) -> float:
    """Integrated autocorrelation time of x(t) [tau_A], on a uniform grid, integrated to the first zero."""
    tu = np.linspace(t[0], t[-1], len(t))
    y = np.interp(tu, t, x) - np.interp(tu, t, x).mean()
    ac = np.correlate(y, y, mode="full")[len(y) - 1:]
    ac = ac / ac[0]
    stop = int(np.argmax(ac <= 0)) or len(ac)
    return float(np.trapezoid(ac[:stop], dx=tu[1] - tu[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_id")
    ap.add_argument("--n-blocks", type=int, default=4)
    args = ap.parse_args()
    rd = ROOT / "data" / args.run_id
    meta = json.load(open(rd / f"{args.run_id}.json"))
    with np.load(rd / f"{args.run_id}.npz") as z:
        ts = {k[4:]: z[k] for k in z.files if k.startswith("ts__")}
    acc = P.load_config(ROOT / "configs" / "h1_acceptance.yaml")

    t, dtk = ts["t"], ts["acc_sum_dt"]
    first = int(np.where(H.window_slice(t, float(meta["config"]["gates"]["steady_window"])))[0][0])
    sel = slice(first + 1, None)
    Tw = float(dtk[sel].sum())
    W = H.time_average(ts["pWm"], dtk, first)
    Dperp = ts["acc_loss_eta"][sel].sum(0) / Tw
    Dnu = ts["acc_loss_nu"][sel].sum(0) / Tw
    eps = float(ts["acc_inj_g"][sel].sum() / Tw)
    T_ps = float(W[0] / Dperp[0])

    b = H.block_averages(t, dtk, first, args.n_blocks, dict(W=ts["pWm"], E_A=ts["E"]),
                         dict(eps=ts["acc_inj_g"], D_perp=ts["acc_loss_eta"]))
    Tb = b["W"][:, 0] / b["D_perp"][:, 0]
    W0_eps = b["W"][:, 0] / b["eps"]
    block_tol = float(acc["steady_state"]["block_tol"])

    gp = json.load(open(ROOT / "data" / "h1_age_model.json"))["gamma_params"]
    pa = json.load(open(ROOT / "data" / PAIRED / "h1_analysis.json"))["P_a"]
    pred = residence_time(gp["g0"], gp["g1"], gp["t_c"])

    out = dict(
        run_id=args.run_id, s=float(np.sqrt(float(meta["config"]["velocity"]["beta_i"]))),
        window=[float(t[first]), float(t[-1])], window_length=Tw,
        gates_recorded={k: v["pass"] if isinstance(v, dict) else v for k, v in meta["gates"].items()},
        steady_fluct=dict(E_A=meta["gates"]["steady_state"]["E_A_fluct"], W=meta["gates"]["steady_state"]["W_fluct"]),
        eps=eps, W=W.tolist(), D_perp=Dperp.tolist(), D_nu=Dnu.tolist(),
        W0_share=float(W[0] / W.sum()),
        T_ps=T_ps,
        blocks=dict(edges=b["edges"], T_ps=Tb.tolist(), W0_over_eps=W0_eps.tolist(), E_A=b["E_A"].tolist(),
                    eps=b["eps"].tolist(),
                    spread_T_ps=float(H.rel_spread(Tb)), spread_W0_over_eps=float(H.rel_spread(W0_eps)),
                    spread_E_A=float(H.rel_spread(b["E_A"])), block_tol=block_tol,
                    T_ps_se=float(Tb.std(ddof=1) / np.sqrt(len(Tb)))),
        correlation_time_W0=correlation_time(t[first:], ts["pWm"][first:, 0]),
        correlation_time_E_A=correlation_time(t[first:], ts["E"][first:]),
        predicted=dict(age_sink_T_ps=pred, age_sink_params=gp, inv_g1=1.0 / gp["g1"],
                       uniform_T_paired=float(pa["T"]), paired_run=PAIRED,
                       paired_T_at_m0=float(1.0 / pa["Dperp_over_W"][0])),
    )
    out["ratio_to_age_sink"] = T_ps / pred
    out["ratio_to_uniform_T"] = T_ps / out["predicted"]["uniform_T_paired"]
    dest = ROOT / "data" / "h1_s0.json"
    dest.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("W", "D_perp", "D_nu")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
