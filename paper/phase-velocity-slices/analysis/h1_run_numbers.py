#!/usr/bin/env python
"""Numbers quoted in section 7 for each analysed forced run, from saved data only.

For each run (window averages over its accepted window, configs/h1_acceptance.yaml, and its h1_analysis.json from
h1_pilot.py):
  * the decay length ell_* of the Hermite spectrum, from a least-squares fit of ln[W(m) sqrt(2m)] = c - sqrt(2m)/ell_*
    over 2 <= m <= M/2 (below the collisional range), against the model's s kappa_1 T, with s = beta_i^{1/2}, kappa_1
    the mean field-aligned wavenumber of the injected pattern (traced field lines) and 1/T the median of D_perp/W over
    2 <= m <= 16; the breaks m_* = ell_*^2/2 and (s kappa_1 T)^2/2;
  * the same fit over 1.5 e-folds of the run's own decay (h1lib.fit_decay_window; m_* scan, 2026-11-10): ell_* with
    its standard error from the four block averages of W(m) on that window, the e-fold count n_e over the whole
    collision-free range, the status (measured, marginal, unresolved), and the same windowed fit applied to the model
    spectrum with the traced P(kappa);
  * W(m) and the field-aligned flux divided by the two-channel model with the traced P(kappa), over 1 <= m <= M/2;
  * the outer scale (tau_nl, chi), 1/T, T/tau_nl, the collisional share of the free-energy dissipation, the E_A
    spectral slope, the per-m budget residual, D_perp/W at m <= 1 and m = 16;
  * 1/T in each of four equal blocks of the window, their spread and the standard error of their mean (the
    uncertainty quoted for T), and the local logarithmic slope of W(m) over 2-8, 4-16 and 8-32;
  * the collisional share of the dissipation in the two-channel model with the traced P(kappa), sum_m 2 nu_m W_model(m)
    / eps with nu_m the hypercollisional amplitude damping rate, against the measured share;
  * field-aligned and Eulerian wavenumbers: kappa_1 of g_0 and Phi per k_perp band, the rms field-aligned and k_z
    wavenumbers of g_m over 1 <= m <= M/2, and the unmixing/phase-mixing ratio of the k_z-split flux for m >= 1;
  * for the m_* scan (section 7, the break against beta_i and chi): the perpendicular rate D_perp/W averaged over the
    windowed fit range uniformly in sqrt(2m) and the decay length s kappa_1 T it gives against the fitted ell_*; the
    max/min of D_perp/W over 1 <= m <= M/2; the sink D_perp/W x tau_nl at ages sqrt(2m)/(s kappa_1 k_1) of 4..30
    tau_nl and the ages of m = 2 and 16 in units of tau_nl; eta and the shells at which the cumulative hyper-resistive
    dissipation of E_A reaches 25, 50 and 75%.
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
from h1_compare_m import averages, block_inv_T, inv_T_se  # noqa: E402


def block_W(run_id: str, window: list[float], n_blocks: int = 4) -> list[np.ndarray]:
    """W(m) (dt-weighted mean of the sampled p_m <g_m^2>) in each of n_blocks equal parts of the averaging window, as
    h1_compare_m.block_inv_T divides it; for the block standard error of the decay length ell_*."""
    rd = P.DATA / run_id
    with np.load(rd / f"{run_id}.npz") as z:
        t, dtk, pW = z["ts__t"], z["ts__acc_sum_dt"], z["ts__pWm"]
    edges = np.linspace(window[0], window[1], n_blocks + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        first = int(np.searchsorted(t, a - 1e-9))
        last = int(np.searchsorted(t, b + 1e-9))
        out.append(H.time_average(pW[:last], dtk[:last], first))
    return out


def dissipation_median_shell(run_id: str, window: list[float]) -> list[int]:
    """Shells n_perp = k_perp/k_1 at which the cumulative hyper-resistive dissipation of E_A, eta (k_perp/k_max)^(2r)
    E_A(k_perp) averaged over the window (dt-weighted), reaches 25%, 50% and 75%; the shape only, so k_max is taken
    as the largest recorded shell.  Locates the dissipation range when eta changes between runs (m_* scan)."""
    rd = P.DATA / run_id
    cfg = json.load(open(rd / f"{run_id}.json"))["config"]
    with np.load(rd / f"{run_id}.npz") as z:
        t, dt, EAk, n = z["ts__t"], z["ts__acc_sum_dt"], z["ts__EAk"], z["nperp"]
    h = (t >= window[0] - 1e-9) & (t <= window[1] + 1e-9)
    Ek = (EAk[h] * dt[h][:, None]).sum(0) / dt[h].sum()
    r = int(cfg["stepper"]["hyper_r"])
    D = (n / n.max()) ** (2 * r) * Ek
    cum = np.cumsum(D) / D.sum()
    return [int(round(float(n[np.searchsorted(cum, f)]))) for f in (0.25, 0.5, 0.75)]


AGES_EDDY = (4, 8, 12, 16, 20, 25, 30)


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
    ell = H.fit_decay(W, 2, M // 2)                     # whole collision-free range (the paper's fit up to 2026-11-10)
    win = H.fit_decay_window(W, M)                      # 1.5 e-folds of the run's own decay (m_* scan spec)
    ell_blocks = [H.fit_decay(Wb, 2, win["m_max"]) for Wb in block_W(run_id, av["window"])]
    fl = an["fieldlines"]
    kap1 = fl["kappa1_F_over_k1"]
    T = 1.0 / av["inv_T"]
    skT = math.sqrt(beta) * kap1 * k1 * T
    tr = an["P_b"]["traced P(kappa) of the injected pattern"]
    rW = np.array(tr["W_over_model_m"])
    rP = np.array(tr["Pi_over_model_m"])
    m_hi_model = min(win["m_max"], len(rW) - 1)         # the model spectrum W/(W/model) on the same window
    ell_model_win = H.fit_decay(W[:len(rW)] / rW, 2, m_hi_model)
    mm = (m >= 1) & (m <= M // 2)
    kap = an["kappa"]
    pe = {k: v for k, v in an["P_e"].items() if k.isdigit() and int(k) >= 1}
    ratio = [v["minus_over_plus"] for v in pe.values()]
    slopes = {f"{a}-{b}": float(np.polyfit(np.log(m[a:b + 1]), np.log(W[a:b + 1]), 1)[0]) for a, b in ((2, 8), (4, 16), (8, 32))}
    binv = block_inv_T(run_id, av["window"])
    # the perpendicular rate averaged over the windowed fit range uniformly in sqrt(2m), as the fit weights it: with an
    # age-dependent sink (eq. Wage) the decay over that range is s kappa_1 / <D_perp/W>
    x = np.sqrt(2.0 * m[2:win["m_max"] + 1])
    inv_T_fit = float(np.trapezoid(q[2:win["m_max"] + 1], x) / (x[-1] - x[0]))
    skT_fit = math.sqrt(beta) * kap1 * k1 / inv_T_fit
    # the sink against the age theta = sqrt(2m)/(s kappa_1 k_1) of the free energy, both in units of tau_nl
    m1 = m[1:M // 2 + 1]
    age_eddy = np.sqrt(2.0 * m1) / (math.sqrt(beta) * kap1 * k1) / av["tau_nl"]
    rate_eddy = q[1:M // 2 + 1] * av["tau_nl"]
    a2_16 = np.sqrt(2.0 * np.array([2, 16])) / (math.sqrt(beta) * kap1 * k1) / av["tau_nl"]
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
        ell_win=win["ell"], ell_win_blocks=ell_blocks,
        ell_win_se=float(np.std(ell_blocks, ddof=1) / math.sqrt(len(ell_blocks))),
        m_star_win=win["ell"] ** 2 / 2, fit_m_max=win["m_max"], fit_n_e=win["n_e"], fit_status=win["status"],
        ell_model_win=ell_model_win, m_star_model_win=ell_model_win ** 2 / 2, model_fit_m_max=m_hi_model,
        coll_over_sink_at_m_max=float(2.0 * coll[win["m_max"]] / av["inv_T"]),     # hypercollisional / perpendicular rate
        kappa1_injected=kap1,
        W_over_model_range=[float(rW[mm[:len(rW)]].min()), float(rW[mm[:len(rW)]].max())],
        Pi_over_model_range=[float(rP[: M // 2].min()), float(rP[: M // 2].max())],
        W_over_model_at_M2=float(rW[min(M // 2, len(rW) - 1)]), Pi_over_model_at_M2=float(rP[min(M // 2, len(rP) - 1)]),
        noecho_range_m8=noecho_range(an, M),
        bands=fl["bands"], kappa1_g0_bands=fl["kappa1_par_over_k1"]["g0"], kappa1_phi_bands=fl["kappa1_par_over_k1"]["phi"],
        kappa1_z_g0_bands=fl["kappa1_z_over_k1"]["g0"],
        kfa_rms_gm_range=[float(np.min(kap["kfa_rms_m_over_k1"][1:M // 2 + 1])), float(np.max(kap["kfa_rms_m_over_k1"][1:M // 2 + 1]))],
        kz_rms_gm_range=[float(np.min(kap["kz_rms_m_over_k1"][1:M // 2 + 1])), float(np.max(kap["kz_rms_m_over_k1"][1:M // 2 + 1]))],
        split_ratio_m_ge1=dict(zip(pe.keys(), ratio)),
        inv_T_blocks=binv, inv_T_block_spread=(max(binv) - min(binv)) / av["inv_T"],
        inv_T_block_se=inv_T_se(binv), W_slopes=slopes,
        nu_share_model=nu_share_model,
        inv_T_fit_range=inv_T_fit, s_kappa1_T_fit_range=skT_fit, ell_win_over_skT_fit_range=win["ell"] / skT_fit,
        Dperp_over_W_max_over_min=an["P_a"]["spread_max_over_min_m1_to_M2"],
        sink_eddy_at_age={a: float(np.interp(a, age_eddy, rate_eddy, left=np.nan, right=np.nan)) for a in AGES_EDDY},
        age_eddy_m2_m16=[float(a2_16[0]), float(a2_16[1])],
        eta=P.parse_number(st["eta"]), dissipation_shells_EA=dissipation_median_shell(run_id, av["window"]),
    )


def noecho_range(an: dict, M: int) -> list[float] | None:
    """Range over 8 <= m <= M/2 of the field-aligned flux divided by the no-echo flux s sqrt(2(m+1)) kappa_1(m) W(m),
    the flux W(m) carries if all of it phase mixes forward at its measured field-aligned wavenumber (h1_pilot.py); a
    return flux would make the ratio fall below one increasingly at large m."""
    r = (an.get("no_echo") or {}).get("Pi_fa_over_noecho")
    if not r:
        return None
    v = np.array(r[8:M // 2 + 1], dtype=float)
    v = v[np.isfinite(v)]
    return [float(v.min()), float(v.max())] if v.size else None


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
        print(f"    windowed fit ({d['fit_status']}, n_e {d['fit_n_e']:.2f}, 2 <= m <= {d['fit_m_max']}): ell_* "
              f"{d['ell_win']:.2f} +- {d['ell_win_se']:.2f}, m_* {d['m_star_win']:.1f}; model spectrum on the same "
              f"window (m <= {d['model_fit_m_max']}) ell_* {d['ell_model_win']:.2f}, m_* {d['m_star_model_win']:.1f}; "
              f"2 nu_m T at the window's top {d['coll_over_sink_at_m_max']:.3f}")
        print(f"    W/model {d['W_over_model_range'][0]:.3f}-{d['W_over_model_range'][1]:.3f}; "
              f"Pi/model {d['Pi_over_model_range'][0]:.3f}-{d['Pi_over_model_range'][1]:.3f}; "
              f"at m = M/2: W/model {d['W_over_model_at_M2']:.3f}, Pi/model {d['Pi_over_model_at_M2']:.3f}; "
              + (f"Pi/no-echo over 8 <= m <= M/2 {d['noecho_range_m8'][0]:.3f}-{d['noecho_range_m8'][1]:.3f}; "
                 if d['noecho_range_m8'] else "") +
              f"g0 kappa_1 per band {', '.join(f'{v:.2f}' for v in d['kappa1_g0_bands'])}; "
              f"phi {', '.join(f'{v:.2f}' for v in d['kappa1_phi_bands'])}; g0 k_z {', '.join(f'{v:.2f}' for v in d['kappa1_z_g0_bands'])}")
        print(f"    k_fa rms g_m {d['kfa_rms_gm_range'][0]:.2f}-{d['kfa_rms_gm_range'][1]:.2f}; "
              f"k_z rms {d['kz_rms_gm_range'][0]:.2f}-{d['kz_rms_gm_range'][1]:.2f}; split ratio "
              + ", ".join(f"m={k}: {v:.3f}" for k, v in d["split_ratio_m_ge1"].items()))
        print(f"    1/T blocks {', '.join(f'{v:.3f}' for v in d['inv_T_blocks'])} (spread {d['inv_T_block_spread']:.3f}, "
              f"standard error {d['inv_T_block_se']:.3f}); W slopes " + ", ".join(f"{k}: {v:.2f}" for k, v in d["W_slopes"].items())
              + f"; collisional share measured {d['nu_share']:.3f}, model {d['nu_share_model']:.3f}")
        print(f"    rate over the fit range {d['inv_T_fit_range']:.3f} (s kappa_1 T {d['s_kappa1_T_fit_range']:.2f}, ell_win/that "
              f"{d['ell_win_over_skT_fit_range']:.3f}); D_perp/W max/min over 1..M/2 {d['Dperp_over_W_max_over_min']:.2f}; "
              f"sink x tau_nl at age/tau_nl " + ", ".join(f"{a}: {v:.3f}" for a, v in d["sink_eddy_at_age"].items())
              + f"; m = 2..16 at age {d['age_eddy_m2_m16'][0]:.1f}-{d['age_eddy_m2_m16'][1]:.1f} tau_nl; eta {d['eta']:g}, "
              f"E_A hyper-resistive dissipation shells (25/50/75%) {d['dissipation_shells_EA']}")
    (P.DATA / "h1_run_numbers.json").write_text(json.dumps(P._jsonable(out), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
