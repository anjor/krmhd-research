#!/usr/bin/env python
"""Numbers quoted in section 7 for each analysed forced run, from saved data only.

For each run (window averages over its accepted window, configs/h1_acceptance.yaml, and its h1_analysis.json from
h1_pilot.py):
  * the decay length ell_* of the Hermite spectrum, from a least-squares fit of ln[W(m) sqrt(2m)] = c - sqrt(2m)/ell_*
    over 2 <= m <= M/2 (below the collisional range), against the model's s kappa_1 T, with s = beta_i^{1/2}, kappa_1
    the mean field-aligned wavenumber of the injected pattern (traced field lines) and 1/T the median of D_perp/W over
    2 <= m <= 16; the breaks m_* = ell_*^2/2 and (s kappa_1 T)^2/2;
  * W(m) and the field-aligned flux divided by the two-channel model with the traced P(kappa), over 1 <= m <= M/2;
  * the outer scale (tau_nl, chi), 1/T, T/tau_nl, the collisional share of the free-energy dissipation, the E_A
    spectral slope, the per-m budget residual, D_perp/W at m <= 1 and m = 16;
  * 1/T in each of four equal blocks of the window, their spread and the standard error of their mean (the
    uncertainty quoted for T), and the local logarithmic slope of W(m) over 2-8, 4-16 and 8-32;
  * the collisional share of the dissipation in the two-channel model with the traced P(kappa), sum_m 2 nu_m W_model(m)
    / eps with nu_m the hypercollisional amplitude damping rate, against the measured share;
  * field-aligned and Eulerian wavenumbers: kappa_1 of g_0 and Phi per k_perp band, the rms field-aligned and k_z
    wavenumbers of g_m over 1 <= m <= M/2, and the unmixing/phase-mixing ratio of the k_z-split flux for m >= 1.
Writes data/h1_run_numbers.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_run_numbers.py <run> [<run> ...]
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
import p1lib as P  # noqa: E402
import h1lib as H  # noqa: E402
from h1_compare_m import averages  # noqa: E402


def block_inv_T(run_id: str, window: list[float], n_blocks: int = 4) -> list[float]:
    """1/T (median of D_perp(m)/W(m) over 2 <= m <= 16) in each of n_blocks equal parts of the window."""
    rd = P.DATA / run_id
    with np.load(rd / f"{run_id}.npz") as z:
        ts = {k[4:]: z[k] for k in z.files if k.startswith("ts__")}
    t, dtk = ts["t"], ts["acc_sum_dt"]
    edges = np.linspace(window[0], window[1], n_blocks + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        first = int(np.searchsorted(t, a - 1e-9))
        last = int(np.searchsorted(t, b + 1e-9))          # samples first..last-1
        W = H.time_average(ts["pWm"][:last], dtk[:last], first)
        Dp = ts["acc_loss_eta"][first + 1:last].sum(0) / dtk[first + 1:last].sum()
        m = np.arange(len(W))
        out.append(float(np.median((Dp / W)[(m >= 2) & (m <= 16)])))
    return out


def numbers(run_id: str, acc: dict) -> dict:
    """The quoted numbers of one run (see the module docstring); wavenumbers in units of k_1 = 2 pi / L."""
    rd = P.DATA / run_id
    av = averages(run_id, acc)
    an = json.load(open(rd / "h1_analysis.json"))
    cfg = json.load(open(rd / f"{run_id}.json"))["config"]
    M = av["M"]
    beta = P.parse_number(cfg["velocity"]["beta_i"])
    k1 = 2 * math.pi / float(cfg["box"]["L"])
    m = np.arange(M + 1)
    W = np.array(av["W_over_eps"])
    q = np.array(av["Dperp_over_W"])
    fit = (m >= 2) & (m <= M // 2)
    x = np.sqrt(2.0 * m[fit])
    slope = np.polyfit(x, np.log(W[fit] * x), 1)[0]
    ell = -1.0 / slope
    fl = an["fieldlines"]
    kap1 = fl["kappa1_F_over_k1"]
    T = 1.0 / av["inv_T"]
    skT = math.sqrt(beta) * kap1 * k1 * T
    tr = an["P_b"]["traced P(kappa) of the injected pattern"]
    rW = np.array(tr["W_over_model_m"])
    rP = np.array(tr["Pi_over_model_m"])
    mm = (m >= 1) & (m <= M // 2)
    kap = an["kappa"]
    pe = {k: v for k, v in an["P_e"].items() if k.isdigit() and int(k) >= 1}
    ratio = [v["minus_over_plus"] for v in pe.values()]
    slopes = {f"{a}-{b}": float(np.polyfit(np.log(m[a:b + 1]), np.log(W[a:b + 1]), 1)[0]) for a, b in ((2, 8), (4, 16), (8, 32))}
    binv = block_inv_T(run_id, av["window"])
    st = cfg["stepper"]
    coll = P.coll_rates(M, P.parse_number(st["nu"]), int(st["hyper_n"]))
    nu_share_model = float(np.sum(2.0 * coll[:len(rW)] * W[:len(rW)] / rW))      # W is W/eps, rW = W/W_model
    return dict(
        run_id=run_id, beta=beta, N=av["N"], M=M, window=av["window"], tau_nl=av["tau_nl"],
        chi_out=an["outer"]["chi_out"], inv_T=av["inv_T"], T_over_tau_nl=av["T_over_tau_nl"], nu_share=av["nu_share"],
        slope_EA=an["gates_applied"]["gates"]["spectral_sanity"]["slope_EA"],
        budget_resid_fa=an["P_a"]["budget_resid_fa_max_over_eps"],
        Dperp_over_W_m01=[float(q[0]), float(q[1])], Dperp_over_W_m16=float(q[16]),
        ell_star=ell, s_kappa1_T=skT, ell_over_skT=ell / skT, m_star_fit=ell ** 2 / 2, m_star_model=skT ** 2 / 2,
        kappa1_injected=kap1,
        W_over_model_range=[float(rW[mm[:len(rW)]].min()), float(rW[mm[:len(rW)]].max())],
        Pi_over_model_range=[float(rP[: M // 2].min()), float(rP[: M // 2].max())],
        bands=fl["bands"], kappa1_g0_bands=fl["kappa1_par_over_k1"]["g0"], kappa1_phi_bands=fl["kappa1_par_over_k1"]["phi"],
        kappa1_z_g0_bands=fl["kappa1_z_over_k1"]["g0"],
        kfa_rms_gm_range=[float(np.min(kap["kfa_rms_m_over_k1"][1:M // 2 + 1])), float(np.max(kap["kfa_rms_m_over_k1"][1:M // 2 + 1]))],
        kz_rms_gm_range=[float(np.min(kap["kz_rms_m_over_k1"][1:M // 2 + 1])), float(np.max(kap["kz_rms_m_over_k1"][1:M // 2 + 1]))],
        split_ratio_m_ge1=dict(zip(pe.keys(), ratio)),
        inv_T_blocks=binv, inv_T_block_spread=(max(binv) - min(binv)) / av["inv_T"],
        inv_T_block_se=float(np.std(binv, ddof=1) / math.sqrt(len(binv)) / np.mean(binv)), W_slopes=slopes,
        nu_share_model=nu_share_model,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--acceptance", default="h1_acceptance.yaml")
    args = ap.parse_args()
    acc = P.load_config(args.acceptance)
    out = {r: numbers(r, acc) for r in args.runs}
    for r, d in out.items():
        print(f"{r}: beta {d['beta']:g}, {d['N']}^3, M {d['M']}: tau_nl {d['tau_nl']:.3f}, chi {d['chi_out']:.2f}, "
              f"1/T {d['inv_T']:.3f}, T/tau_nl {d['T_over_tau_nl']:.1f}, nu share {d['nu_share']:.3f}, slope E_A {d['slope_EA']:.2f}, "
              f"budget {d['budget_resid_fa']:.1e}")
        print(f"    ell_* {d['ell_star']:.2f} vs s kappa_1 T {d['s_kappa1_T']:.2f} (ratio {d['ell_over_skT']:.3f}); "
              f"m_* {d['m_star_fit']:.0f} (fit) {d['m_star_model']:.0f} (model); kappa_1 {d['kappa1_injected']:.2f}; "
              f"D_perp/W m=0,1 {d['Dperp_over_W_m01'][0]:.2f},{d['Dperp_over_W_m01'][1]:.2f}, m=16 {d['Dperp_over_W_m16']:.2f}")
        print(f"    W/model {d['W_over_model_range'][0]:.3f}-{d['W_over_model_range'][1]:.3f}; "
              f"Pi/model {d['Pi_over_model_range'][0]:.3f}-{d['Pi_over_model_range'][1]:.3f}; "
              f"g0 kappa_1 per band {', '.join(f'{v:.2f}' for v in d['kappa1_g0_bands'])}; "
              f"phi {', '.join(f'{v:.2f}' for v in d['kappa1_phi_bands'])}; g0 k_z {', '.join(f'{v:.2f}' for v in d['kappa1_z_g0_bands'])}")
        print(f"    k_fa rms g_m {d['kfa_rms_gm_range'][0]:.2f}-{d['kfa_rms_gm_range'][1]:.2f}; "
              f"k_z rms {d['kz_rms_gm_range'][0]:.2f}-{d['kz_rms_gm_range'][1]:.2f}; split ratio "
              + ", ".join(f"m={k}: {v:.3f}" for k, v in d["split_ratio_m_ge1"].items()))
        print(f"    1/T blocks {', '.join(f'{v:.3f}' for v in d['inv_T_blocks'])} (spread {d['inv_T_block_spread']:.3f}, "
              f"standard error {d['inv_T_block_se']:.3f}); W slopes " + ", ".join(f"{k}: {v:.2f}" for k, v in d["W_slopes"].items())
              + f"; collisional share measured {d['nu_share']:.3f}, model {d['nu_share_model']:.3f}")
    (P.DATA / "h1_run_numbers.json").write_text(json.dumps(P._jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
