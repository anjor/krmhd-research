#!/usr/bin/env python
"""Analysis of a forced run for the test of H1 (.loop/specs/rework-H1.md): time averages over the steady-state window,
predictions P-a, P-b, P-e, P-f, the two-channel (no-echo) model with measured kappa and T, and three figures.

Reads only saved data: <run dir>/<run ID>.npz/.json (scripts/run_forced_turbulence.py) and, if present,
<run dir>/fieldlines_snap_*.npz/.json (analysis/h1_fieldlines.py).  Writes <run dir>/h1_analysis.json and the figures
figures/h1_wm, h1_flux, h1_kpar (.pdf and .png; --outdir to write elsewhere, e.g. for pipeline checks).  Refuses to
analyse a run whose gates fail (CLAUDE.md) unless --check (pipeline check on a short run; nothing is interpreted).  The
gates are re-evaluated with Anjor's decisions in configs/h1_acceptance.yaml (steady-state threshold and block-average
check for every run; per-run tolerances where he accepted a deviation); h1_analysis.json records the run's own gates
and the re-evaluated ones.

Two-channel model.  Each field-aligned wavenumber kappa of the injected free energy is a linear Hermite problem with
an m-independent sink of free energy at rate 1/T (the k_perp passive-scalar cascade, H1; D_perp(m) = W(m)/T) plus
GANDALF's hypercollisions:
    d g/dt = -(i s kappa A + D) g + xi e_0,   D = diag(1/(2T) + nu (m/M)^n [m >= 2]),   <xi xi*> = q delta(t),
whose stationary covariance C solves the Lyapunov equation L C + C L^H = q e_0 e_0^T (L = i s kappa A + D).  Then
W(m) = p_m C_mm, the field-aligned flux Pi_{m+1/2} = -2 s p_m A_{m,m+1} kappa Im C_{m+1,m}, the injection is p_0 q,
and the slice covariance is Wl^T C Wl.  Summing over a kappa distribution P(kappa) normalised to the measured
injection rate eps gives W(m), Pi(m) and C(u, u') with no free parameter once T (from D_perp(m)/W(m)) and P(kappa)
(measured) are fixed.  Its large-m form is the formula of the spec, W(m) = eps/(s kappa sqrt(2m))
exp(-sqrt(2m)/(s kappa T)).  The no-echo flux s sqrt(2(m+1)) kappa_1(m) W(m), with kappa_1(m) the first moment of
|kappa| of g_m measured along traced field lines, is the flux if all free energy at m were phase mixing (no return).

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_pilot.py paper/phase-velocity-slices/data/<run ID>
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from scipy.optimize import curve_fit

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import figstyle as FS  # noqa: E402
import p1lib as P  # noqa: E402
import h1lib as H  # noqa: E402


# ------------------------------------------------------------------------------------------------------------------
# two-channel model
# ------------------------------------------------------------------------------------------------------------------
def lyapunov_mode(sl: P.Slices, s: float, kappa: float, T: float, nu: float, hyper_n: int, mf: int = 0) -> dict:
    """Stationary statistics of one field-aligned wavenumber with unit free-energy injection into moment mf
    (p_mf q = 1)."""
    M = sl.M
    D = np.full(M + 1, 0.5 / T) + P.coll_rates(M, nu, hyper_n)      # amplitude rates: free energy decays at 1/T
    L = 1j * s * kappa * sl.A + np.diag(D)
    Q = np.zeros((M + 1, M + 1), complex)
    Q[mf, mf] = 1.0 / sl.p[mf]
    C = scipy.linalg.solve_continuous_lyapunov(-L, -Q)
    W = sl.p * np.real(np.diag(C))
    Pi = np.array([-2.0 * s * sl.p[m] * sl.A[m, m + 1] * kappa * np.imag(C[m + 1, m]) for m in range(M)])
    S = np.real(sl.Wl.T @ C @ sl.Wl)
    return dict(W=W, Pi=Pi, S=S)


def two_channel(sl: P.Slices, s: float, kappas: np.ndarray, weights: np.ndarray, T: float, nu: float, hyper_n: int,
                eps: float, mf: int = 0) -> dict:
    """Sum of lyapunov_mode over a kappa distribution (weights normalised to 1), scaled to injection eps into
    moment mf."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    W = np.zeros(sl.M + 1)
    Pi = np.zeros(sl.M)
    S = np.zeros((sl.M + 1, sl.M + 1))
    for k, wk in zip(kappas, w):
        if wk <= 0 or k <= 0:
            if wk > 0:                                   # kappa = 0: no phase mixing, all of it decays at m = mf
                W[mf] += eps * wk * T
                S += eps * wk * T * np.outer(sl.Wl[mf], sl.Wl[mf]) / sl.p[mf]
            continue
        r = lyapunov_mode(sl, s, float(k), T, nu, hyper_n, mf)
        W += eps * wk * r["W"]
        Pi += eps * wk * r["Pi"]
        S += eps * wk * r["S"]
    return dict(W=W, Pi=Pi, S=S)


def formula(m: np.ndarray, eps: float, s: float, kappa: float, T: float) -> np.ndarray:
    """The spec's continuum formula W(m) = eps/(s kappa sqrt(2m)) exp(-sqrt(2m)/(s kappa T))."""
    m = np.asarray(m, dtype=float)
    return eps / (s * kappa * np.sqrt(2 * m)) * np.exp(-np.sqrt(2 * m) / (s * kappa * T))


def corr_norm(S: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(S), 1e-300, None))
    return S / np.outer(d, d)


# ------------------------------------------------------------------------------------------------------------------
def apply_gate_decisions(meta: dict, ts: dict, first: int, M: int, acc: dict) -> dict:
    """Re-evaluate a forced run's gates with Anjor's decisions (configs/h1_acceptance.yaml): energy conservation and
    energy balance as recorded; steady state with acc.steady_state.tol on the recorded std/mean plus the block-average
    check (h1lib.gate_block_convergence); spectral sanity with the run's own tolerances.  acc.runs.<run ID> may override
    m_tol, steady_tol and block_tol, and set the averaging window (window, tau_A, ending at the end of the run; read in
    main): the std/mean of E_A and W is then recomputed (dt-weighted) over the window that starts at sample `first`.
    Returns the gates, the decisions applied and an overall pass."""
    g = meta["gates"]
    gc = meta["config"]["gates"]
    ss = acc["steady_state"]
    run_acc = (acc.get("runs") or {}).get(meta["run_id"], {})
    tol = float(run_acc.get("steady_tol", ss["tol"]))
    flucts = {k: float(v) for k, v in g["steady_state"].items() if k.endswith("_fluct")}
    width = float(run_acc.get("window", gc["steady_window"]))
    if "window" in run_acc:
        dtw = ts["acc_sum_dt"][first + 1:]
        def fl(x: np.ndarray) -> float:
            mu = np.average(x[first + 1:], weights=dtw)
            return float(np.sqrt(np.average((x[first + 1:] - mu) ** 2, weights=dtw)) / mu)
        flucts = dict(E_A_fluct=fl(ts["E"]), W_fluct=fl(ts["pWm"].sum(1)),
                      recorded={k: v for k, v in flucts.items()})
    tested = {k: v for k, v in flucts.items() if k.endswith("_fluct")}
    steady = dict(applies=True, criterion=f"std/mean < {tol} over the final {width} tau_A, each series",
                  tol=tol, window=width, **flucts, **{"pass": bool(all(f < tol for f in tested.values()))})
    b = H.block_averages(ts["t"], ts["acc_sum_dt"], first, int(ss["n_blocks"]),
                         dict(W=ts["pWm"], Pi_fa=ts["Pi_fa"], E_A=ts["E"]),
                         dict(eps=ts["acc_inj_g"], D_perp=ts["acc_loss_eta"], D_nu=ts["acc_loss_nu"]))
    b["nu_share"] = b["D_nu"].sum(1) / (b["D_nu"].sum(1) + b["D_perp"].sum(1))
    blocks = H.gate_block_convergence(b, "eps", {"W(m), m <= 3M/4": ("W", slice(0, 3 * M // 4 + 1), True),
                                                 "Pi_fa, m+1/2 <= 3M/4": ("Pi_fa", slice(0, 3 * M // 4), True),
                                                 "D_perp(m), m <= 3M/4": ("D_perp", slice(0, 3 * M // 4 + 1), True),
                                                 "D_nu(m), m >= M/2": ("D_nu", slice(M // 2, M + 1), True),
                                                 "collisional share of the dissipation": ("nu_share", None, False)},
                                      float(run_acc.get("block_tol", ss["block_tol"])))
    blocks["E_A_blocks"] = b["E_A"].tolist()
    blocks["W_total_blocks"] = b["W"].sum(1).tolist()
    steady["blocks"] = blocks
    steady["pass"] = bool(steady["pass"] and blocks["pass"])
    m_tol = float(run_acc.get("m_tol", gc["m_tol"]))
    spec = H.regate_spectral_sanity(g["spectral_sanity"], float(gc["resolution_tol"]), m_tol)
    gates = dict(energy_conservation=bool(g["energy_conservation"]["pass"]), energy_balance=bool(g["energy_balance"]["pass"]),
                 steady_state=steady, spectral_sanity=spec)
    decisions = [ss["decision"]] + ([run_acc["decision"]] if "decision" in run_acc else [])
    ok = gates["energy_conservation"] and gates["energy_balance"] and steady["pass"] and spec["pass"]
    return dict(gates=gates, decisions=decisions, **{"pass": bool(ok)})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--window", type=float, default=None, help="averaging window in tau_A (default: the steady window)")
    ap.add_argument("--outdir", default=None, help="figure folder (default paper/phase-velocity-slices/figures)")
    ap.add_argument("--check", action="store_true", help="pipeline check on a short run (no gate requirement)")
    ap.add_argument("--acceptance", default="h1_acceptance.yaml", help="gate decisions (default configs/h1_acceptance.yaml)")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    if not rd.is_absolute():
        rd = P.REPO / rd
    with np.load(rd / f"{rd.name}.npz") as z:
        arr = {k: z[k] for k in z.files}
    with open(rd / f"{rd.name}.json") as fh:
        meta = json.load(fh)
    if not args.check and not meta["gates"]["steady_state"].get("applies", False):
        raise SystemExit(f"{meta['run_id']}: the steady-state gate does not apply (run too short); use --check")
    cfg = meta["config"]
    v = cfg["velocity"]
    M, Lam, beta = int(v["M"]), P.parse_number(v["Lambda"]), P.parse_number(v["beta_i"])
    s = math.sqrt(beta)
    sl = P.make_slices(M, Lam, v["closure"])
    st = cfg["stepper"]
    nu, hn = P.parse_number(st["nu"]), int(st["hyper_n"])
    ts = {k[4:]: a for k, a in arr.items() if k.startswith("ts__")}
    t = ts["t"]
    acc = P.load_config(args.acceptance)
    run_acc = (acc.get("runs") or {}).get(meta["run_id"], {})
    width = args.window if args.window is not None else float(run_acc.get("window", cfg["gates"]["steady_window"]))
    win = H.window_slice(t, width) if t[-1] - t[0] >= width else np.ones_like(t, dtype=bool)
    first = int(np.where(win)[0][0])
    sel = np.zeros_like(win)
    sel[first + 1:] = True
    dtk = ts["acc_sum_dt"]
    Tw = float(dtk[sel].sum())
    applied = apply_gate_decisions(meta, ts, first, M, acc)
    if not applied["pass"] and not args.check:
        raise SystemExit(f"{meta['run_id']}: gates fail under configs/{Path(args.acceptance).name}; CLAUDE.md: stop and "
                         f"report (no analysis)\n{json.dumps(P._jsonable(applied), indent=1)}")

    def tav(key: str) -> np.ndarray:
        return H.time_average(ts[key], dtk, first)

    k1 = 2 * np.pi / float(cfg["box"]["L"])
    eps = float(ts["acc_inj_g"][sel].sum() / Tw)
    eps_target = P.parse_number(cfg["forcing"]["density"]["eps"])
    mf = int(cfg["forcing"]["density"].get("moment", 0))   # the forced Hermite moment
    W = tav("pWm")
    Dperp_exact = ts["acc_loss_eta"][sel].sum(0) / Tw
    Dnu_exact = ts["acc_loss_nu"][sel].sum(0) / Tw
    Dperp = tav("Dperp")
    Pfa, Pkz, Pp, Pm = tav("Pi_fa"), tav("Pi_kz"), tav("Pi_plus"), tav("Pi_minus")
    m = np.arange(M + 1)
    mh = np.arange(M) + 0.5
    out: dict = dict(run_id=meta["run_id"], window=[float(t[first]), float(t[-1])], window_length=Tw, check=args.check,
                     eps_measured=eps, eps_target=eps_target, gates_recorded=meta["gates"], gates_applied=applied)

    # ---- P-a: uniform sink
    r = Dperp_exact / W
    rng_a = (m >= 2) & (m <= M // 2)
    invT = float(np.median(r[rng_a]))
    T = 1.0 / invT
    rr = r[(m >= 1) & (m <= M // 2)]
    out["P_a"] = dict(inv_T=invT, T=T, Dperp_over_W=r.tolist(), spread_max_over_min_m1_to_M2=float(rr.max() / rr.min()),
                      cv_m1_to_M2=float(rr.std() / rr.mean()), Dperp_sampled_vs_exact=float(np.abs(Dperp - Dperp_exact).max()
                                                                                          / Dperp_exact.max()))
    Pe = lambda Pi: np.concatenate([[0.0], Pi, [0.0]])  # noqa: E731
    dWdt = (ts["pWm"][-1] - ts["pWm"][first]) / Tw
    inj = np.zeros(M + 1)
    inj[mf] = eps
    for name, Pi in (("fa", Pfa), ("kz", Pkz)):
        pe = Pe(Pi)
        res = (pe[:-1] - pe[1:]) + inj - Dperp_exact - Dnu_exact - dWdt
        out["P_a"][f"budget_resid_{name}_max_over_eps"] = float(np.abs(res).max() / eps)
    # ---- outer-scale numbers, chi, tau_nl
    chi = {k: float(tav(k)) for k in ("chi_u", "chi_z", "chi_out", "chi_out_fa", "u_rms", "b_rms", "u_out", "kperp_out",
                                     "kpar_out", "kpar_out_fa", "E")}
    tau_nl = 1.0 / (chi["kperp_out"] * chi["u_out"])
    out["outer"] = dict(chi, tau_nl=tau_nl, C=T / tau_nl)
    # ---- kappa
    kF_rms = float(math.sqrt(tav("kF_fa2")))
    kF_z = float(math.sqrt(tav("kF_z2")))
    sig_tilt = float(math.sqrt(tav("tiltF2")))
    kap_eff = Pfa / (s * np.sqrt(2.0 * (m[:-1] + 1)) * W[:-1])
    out["kappa"] = dict(kF_z_over_k1=kF_z / k1, kF_fa_rms_over_k1=kF_rms / k1, tilt_rms_over_k1=sig_tilt / k1,
                        kappa_eff_m_over_k1=(kap_eff / k1).tolist(),
                        kfa_rms_m_over_k1=(np.sqrt(tav("kfa2")) / k1).tolist(),
                        kz_rms_m_over_k1=(np.sqrt(tav("kz2")) / k1).tolist())
    # field-line files (optional)
    fls = sorted(rd.glob("fieldlines_snap_*.npz"))
    fl = None
    if fls:
        zs = [np.load(f) for f in fls]
        js = [json.load(open(f.with_suffix(".json"))) for f in fls]
        kap = zs[0]["kappa_par"]
        PF = np.mean([z["F__E_par"] for z in zs], axis=0)
        Pg0 = np.mean([z["g0__E_par"] / z["g0__E_par"].sum() for z in zs], axis=0)
        k1m = np.mean([z["kappa1_m_par"] for z in zs], axis=0)
        bands = js[0]["bands"]
        kb = {nm: np.array([[j[f"{nm}_band{i}"]["par"]["kappa_1"] for i in range(len(bands))] for j in js]).mean(0)
              for nm in ("g0", "phi", "psi")}
        kbz = {nm: np.array([[j[f"{nm}_band{i}"]["z"]["kappa_1"] for i in range(len(bands))] for j in js]).mean(0)
               for nm in ("g0", "phi", "psi")}
        kbh = {nm: np.array([[j[f"{nm}_band{i}"]["par"]["kappa_half"] for i in range(len(bands))] for j in js]).mean(0)
               for nm in ("g0", "phi", "psi")}
        fl = dict(kap=kap, PF=PF, Pg0=Pg0, k1m=k1m, bands=bands, kb=kb, kbz=kbz, kbh=kbh, n=len(fls),
                  times=[j["t"] for j in js])
        out["fieldlines"] = dict(snapshots=[f.name for f in fls], times=fl["times"], bands=bands,
                                 kappa1_par_over_k1={k: (x / k1).tolist() for k, x in kb.items()},
                                 kappa_half_par_over_k1={k: (x / k1).tolist() for k, x in kbh.items()},
                                 kappa1_z_over_k1={k: (x / k1).tolist() for k, x in kbz.items()},
                                 kappa1_F_over_k1=float(np.sum(kap * PF) / PF.sum() / k1),
                                 kappa1_g_m_over_k1=(k1m / k1).tolist())
    # ---- models
    models: dict = {}
    models["single kappa = k_z(forcing)"] = two_channel(sl, s, np.array([kF_z]), np.array([1.0]), T, nu, hn, eps, mf)
    xi = np.linspace(-4, 4, 81)
    gk = np.abs(kF_z + sig_tilt * xi)
    models["kappa = k_z + tilt (Gaussian)"] = two_channel(sl, s, gk, np.exp(-xi ** 2 / 2), T, nu, hn, eps, mf)
    if fl is not None:
        models["traced P(kappa) of the injected pattern"] = two_channel(sl, s, fl["kap"], fl["PF"], T, nu, hn, eps, mf)
        models["traced P(kappa) of g_0"] = two_channel(sl, s, fl["kap"], fl["Pg0"], T, nu, hn, eps, mf)
        noecho = s * np.sqrt(2.0 * (m[:-1] + 1)) * fl["k1m"][:-1] * W[:-1]
    else:
        noecho = None
    rng_b = (m >= 2) & (m <= M // 2)
    out["P_b"] = {}
    for name, md in models.items():
        lr = np.log(W[rng_b] / md["W"][rng_b])
        out["P_b"][name] = dict(W_over_model_m=(W / md["W"]).tolist(), log_rms_dev_m2_to_M2=float(np.sqrt(np.mean(lr ** 2))),
                                mean_log_ratio=float(lr.mean()),
                                Pi_over_model_m=(Pfa / np.where(np.abs(md["Pi"]) > 0, md["Pi"], np.nan)).tolist())
    mstar = {k: (s * kk * T) ** 2 / 2 for k, kk in (("kF_z", kF_z), ("kF_fa_rms", kF_rms))}
    out["m_star"] = dict(mstar, prediction_C2_beta_over_2chi2=(T / tau_nl) ** 2 * beta / (2 * chi["chi_out"] ** 2))
    # ---- P-e
    pick = [mm for mm in (0, 1, 2, 4, 8, 16) if mm < M]
    out["P_e"] = {str(mm): dict(Pi_fa=float(Pfa[mm] / eps), Pi_kz=float(Pkz[mm] / eps), Pi_plus=float(Pp[mm] / eps),
                                Pi_minus=float(Pm[mm] / eps), minus_over_plus=float(-Pm[mm] / Pp[mm]),
                                tilt_part=float((Pfa[mm] - Pkz[mm]) / eps)) for mm in pick}
    out["P_e"]["tilt_over_kz_forcing_squared"] = (sig_tilt / kF_z) ** 2
    if noecho is not None:
        out["no_echo"] = dict(Pi_fa_over_noecho=(Pfa / noecho).tolist())
    # ---- P-f
    C = tav("C_slice")
    Cn = corr_norm(C)
    E_sl = np.diag(C)
    u = sl.u
    ucut = 2.5
    I, J = np.where(np.triu(np.ones_like(Cn, dtype=bool), 1) & (np.abs(u)[:, None] < ucut) & (np.abs(u)[None, :] < ucut))
    du = np.abs(u[I] - u[J])
    cv = Cn[I, J]
    try:
        popt, _ = curve_fit(lambda x, d: 1.0 / (1.0 + (x / d) ** 2), du, cv, p0=[1.0])
        du_fit = float(abs(popt[0]))
    except RuntimeError:
        du_fit = float("nan")
    out["P_f"] = dict(du_c_fit=du_fit, du_c_pred_kF_z=1.0 / (s * kF_z * T), du_c_pred_kF_fa=1.0 / (s * kF_rms * T),
                      u_cut=ucut)
    mod_key = "traced P(kappa) of the injected pattern" if fl is not None else "kappa = k_z + tilt (Gaussian)"
    Cm = corr_norm(models[mod_key]["S"])
    out["P_f"]["model_used"] = mod_key
    out["P_f"]["rms_dev_measured_vs_model"] = float(np.sqrt(np.mean((Cm[I, J] - cv) ** 2)))

    with open(rd / "h1_analysis.json", "w") as fh:
        json.dump(P._jsonable(out), fh, indent=1)
    print(json.dumps(P._jsonable({k: out[k] for k in ("window", "eps_measured", "outer", "m_star", "P_f")}), indent=1))
    print("P-a:", {k: v for k, v in out["P_a"].items() if k != "Dperp_over_W"})
    print("P-b:", {k: (v["log_rms_dev_m2_to_M2"], v["mean_log_ratio"]) for k, v in out["P_b"].items()})
    print("P-e:", out["P_e"])

    # ------------------------------------------------------------------------------------------------ figures
    fcfg = FS.style_config()
    outdir = Path(args.outdir) if args.outdir else FS.FIG_DIR
    outdir.mkdir(parents=True, exist_ok=True)
    pal = fcfg["palette"]
    blk = fcfg["reference_color"]
    mm = m[1:]

    def save(fig, name: str) -> None:
        fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight")
        fig.savefig(outdir / f"{name}.png", dpi=fcfg["png_dpi"], bbox_inches="tight")

    tex = {"single kappa = k_z(forcing)": r"single $\kappa = k_z^{F}$",
           "kappa = k_z + tilt (Gaussian)": r"$\kappa = k_z^{F} + \mathbf{b}_\perp\cdot\mathbf{k}_\perp$ (Gaussian)",
           "traced P(kappa) of the injected pattern": r"traced $P(\kappa)$, injected pattern",
           "traced P(kappa) of g_0": r"traced $P(\kappa)$ of $g_0$"}
    plt = FS.setup_style(fcfg, usetex=True)
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    try:
        # h1_wm
        fig, ax = plt.subplots(1, 2, figsize=(fcfg["double_width_in"], 2.6))
        a = ax[0]
        a.loglog(mm, W[1:] / eps, "o", color=blk, ms=3, label="measured")
        for (name, md), c in zip(models.items(), pal):
            a.loglog(mm, md["W"][1:] / eps, "-", color=c, label=tex.get(name, name))
        a.loglog(mm, formula(mm, 1.0, s, kF_z, T), ":", color=pal[0], lw=0.9, label=r"formula, $\kappa = k_{z}$")
        a.axvspan(M // 2, M, color="0.9", zorder=0)
        a.set_xlabel(r"$m$")
        a.set_ylabel(r"$W(m)/\varepsilon$")
        a.legend(fontsize=6, loc="lower left")
        FS.panel_label(a, "(a)")
        a = ax[1]
        a.semilogx(m[1:], (Dperp_exact / W)[1:] * T, "o-", color=pal[0], ms=3, label=r"$D_\perp(m)\,T/W(m)$")
        a.semilogx(m[2:], (Dnu_exact / W)[2:] * T, "s-", color=pal[1], ms=3, label=r"$D_\nu(m)\,T/W(m)$")
        a.axhline(1.0, color=blk, lw=0.6)
        a.axvspan(M // 2, M, color="0.9", zorder=0)
        a.set_ylim(0, 3)
        a.set_xlabel(r"$m$")
        a.set_ylabel(r"rate $\times\,T$")
        a.legend(fontsize=7)
        FS.panel_label(a, "(b)")
        fig.tight_layout()
        save(fig, "h1_wm")
        plt.close(fig)
        # h1_flux
        fig, ax = plt.subplots(1, 2, figsize=(fcfg["double_width_in"], 2.6))
        a = ax[0]
        a.semilogx(mh, Pfa / eps, "o-", color=blk, ms=3, label=r"$\Pi_{m+1/2}$ field-aligned")
        a.semilogx(mh, Pkz / eps, "-", color=pal[0], label=r"$k_z$ only")
        a.semilogx(mh, Pp / eps, "--", color=pal[1], label=r"$k_z$ split: $+$")
        a.semilogx(mh, -Pm / eps, ":", color=pal[1], label=r"$k_z$ split: $-$ (sign flipped)")
        a.semilogx(mh, models[mod_key]["Pi"] / eps, "-", color=pal[2], label="two-channel model")
        if noecho is not None:                       # asymptotic (phase-mixing) relation: shown for m >= 2
            a.semilogx(mh[2:], noecho[2:] / eps, "-.", color=pal[3],
                       label=r"no-echo $s\sqrt{2(m+1)}\,\kappa_1(m)W(m)$")
        a.set_xlabel(r"$m + 1/2$")
        a.set_ylabel(r"flux$/\varepsilon$")
        a.legend(fontsize=6, loc="upper right")
        FS.panel_label(a, "(a)")
        a = ax[1]
        a.plot(du, cv, ".", color=blk, ms=2, label=r"measured, $|u|<%.1f$" % ucut)
        a.plot(du, Cm[I, J], ".", color=pal[2], ms=2, label="two-channel model")
        x = np.linspace(0, du.max(), 200)
        a.plot(x, 1 / (1 + (x * s * kF_z * T) ** 2), "-", color=pal[0], lw=0.9,
               label=r"$[1+(\Delta u\, s k_z T)^2]^{-1}$")
        a.set_xlabel(r"$|u_j - u_l|$")
        a.set_ylabel(r"$C_{jl}/(C_{jj}C_{ll})^{1/2}$")
        a.legend(fontsize=6)
        FS.panel_label(a, "(b)")
        fig.tight_layout()
        save(fig, "h1_flux")
        plt.close(fig)
        # h1_kpar
        if fl is not None:
            fig, a = plt.subplots(1, 1, figsize=(fcfg["single_width_in"], 2.6))
            kc = np.array([0.5 * (max(b[0], 1) + b[1]) for b in fl["bands"]])
            for (nm, lab), c in zip((("g0", r"$g_0$"), ("phi", r"$\Phi$"), ("psi", r"$\Psi$")), pal):
                a.loglog(kc, fl["kb"][nm] / k1, "o-", color=c, ms=3, label=lab + " along lines")
                a.loglog(kc, fl["kbz"][nm] / k1, "x:", color=c, ms=3, lw=0.8, label=lab + r" $k_z$")
            a.loglog(kc, fl["kb"]["phi"][0] / k1 * (kc / kc[0]) ** (2 / 3), "-", color=blk, lw=0.6,
                     label=r"$\propto k_\perp^{2/3}$")
            a.axhline(float(np.sum(fl["kap"] * fl["PF"]) / fl["PF"].sum()) / k1, color="0.5", lw=0.6, ls="--",
                      label="injected pattern")
            a.set_xlabel(r"$k_\perp L/2\pi$")
            a.set_ylabel(r"$\kappa_1 L/2\pi$")
            for axis in (a.xaxis, a.yaxis):
                axis.set_major_formatter(ScalarFormatter())
                axis.set_minor_formatter(NullFormatter())
            a.set_xticks([float(x) for x in kc])
            a.set_yticks([0.5, 1, 2, 3])
            a.legend(fontsize=5.5, ncol=2, loc="upper left")
            fig.tight_layout()
            save(fig, "h1_kpar")
            plt.close(fig)
    except RuntimeError as exc:                      # LaTeX unavailable: the numbers are already written
        print(f"figure rendering failed: {exc}")
    print(f"wrote {rd / 'h1_analysis.json'} and figures in {outdir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
