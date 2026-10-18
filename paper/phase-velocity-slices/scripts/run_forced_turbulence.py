#!/usr/bin/env python
"""Forced KRMHD turbulence for the test of hypothesis H1 (.loop/specs/rework-H1.md): balanced white-in-time forcing of
Phi at the outer scale (GANDALF's force_alfven_modes) and of the density moment g_0 of one compressive branch, both at
fixed injection rates (the amplitude is rescaled with dt every step), hyper-resistivity on z+- and every g_m,
hypercollisions at high m, GANDALF's gandalf_step (Lawson-RK4) with dt from the explicit slice-advection limit.

Per step (exact, h1lib.make_accounting): injected Alfvenic energy and free energy, the hyper-resistive and
hypercollisional losses per Hermite index, and the time-stepping error of the ideal part (the change of the energy in
the step before the dissipative factors).  Every `sample_every` steps (h1lib.make_sampler): W(m), W(m, k_perp),
W(m, k_z), the field-aligned and k_z-only Hermite fluxes, the sign(k_z) split, D_perp(m), D_nu(m), the slice energies
and correlation matrix, the field-aligned and Eulerian parallel wavenumbers per m and of the injected pattern, the
Alfvenic energies, spectra, dissipation, chi (energy-weighted and outer-scale).  Periodic snapshots of z+- and g
(complex64) and a restart file (complex128, with the forcing PRNG keys) are written to the run folder, together with
resume.npz, the complete state of the loop (fields, keys, accumulators, time-step state, the records so far).  With
--resume the script continues the latest unfinished run of the config's label in the data folder in place (same run
ID and folder, one time series), so that a run interrupted after the start of the averaging window (a preemption, a
timeout, a retry on Modal) need not be restarted or stitched; without an unfinished run it starts a new one.

Gates (CLAUDE.md; tau_A = L_z/v_A = 1 as in Studies 02 and 04): energy conservation of the ideal part (time-stepping
error below 1% of the injected energy, Alfvenic and free energy); energy balance (injection = dissipation within 5%,
both energies); steady state (std/mean < 10% over the final 50 tau_A, Alfvenic energy and free energy); spectral
sanity and resolution.  The last three apply only to runs that cover the steady-state window.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/scripts/run_forced_turbulence.py paper/phase-velocity-slices/configs/forced_beta1.yaml [--resume]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1lib as H  # noqa: E402  (imports p1lib first: float64 JAX on the CPU)
import p1lib as P  # noqa: E402
import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from krmhd.timestepping import gandalf_step  # noqa: E402

ACC_KEYS = ("steps", "sum_dt", "inj_A", "inj_g", "loss_A", "err_A", "err_W")


def initial_fields(box: P.Box, M: int, cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict]:
    """Initial z+-, g and time: from rest, or from a restart file of an earlier run (`init.path`, relative to the
    repository or absolute).  With keep_g false (or a different M) g starts at zero; the time restarts at 0 unless
    keep_time.  Returns the restart file's forcing keys too (used only with keep_keys)."""
    ini = cfg.get("init", {"type": "rest"})
    shape = (box.N, box.N, box.N // 2 + 1)
    if ini["type"] == "rest":
        return np.zeros(shape, complex), np.zeros(shape, complex), np.zeros(shape + (M + 1,), complex), 0.0, {}
    path = ini["path"]
    if not os.path.isabs(path):
        path = str(P.REPO / path)
    with np.load(path) as z:
        zp, zm, g0 = z["zp"], z["zm"], z["g"]
        t0 = float(z["t"])
        keys = dict(key_A=z["key_A"], key_g=z["key_g"])
    if zp.shape != shape:
        raise ValueError(f"restart grid {zp.shape} differs from {shape}")
    g = np.zeros(shape + (M + 1,), complex)
    if ini.get("keep_g", False):
        mm = min(M, g0.shape[-1] - 1)
        g[..., :mm + 1] = g0[..., :mm + 1]
    return zp, zm, g, (t0 if ini.get("keep_time", False) else 0.0), (keys if ini.get("keep_keys", False) else {})


RESUME_FILE = "resume.npz"


def find_resumable(base: str | os.PathLike, label: str) -> P.Path | None:
    """The latest unfinished run folder of `label` under `base`: it holds resume.npz but not the final <run ID>.json."""
    runs = sorted(P.Path(base).glob(f"{P.RUN_PREFIX}_{label}_*"))
    for d in reversed(runs):
        if (d / RESUME_FILE).exists() and not (d / f"{d.name}.json").exists():
            return d
    return None


def config_for_resume(cfg: dict) -> str:
    """The config as compared on resume: everything except time.t_end, which may be extended."""
    c = json.loads(json.dumps(cfg))
    c.get("time", {}).pop("t_end", None)
    return json.dumps(c, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--base", default=None, help="data folder (default paper/phase-velocity-slices/data)")
    ap.add_argument("--resume", action="store_true",
                    help="continue the latest unfinished run of this config's label in the data folder, in place")
    args = ap.parse_args()
    cfg = P.load_config(args.config)
    v = cfg["velocity"]
    M, Lam, beta = int(v["M"]), P.parse_number(v["Lambda"]), P.parse_number(v["beta_i"])
    s = math.sqrt(beta)
    closure = str(v["closure"])
    sl = P.make_slices(M, Lam, closure)
    tc = cfg["time"]
    M_dt = int(tc.get("dt_umax_M", M))
    umax = P.make_slices(M_dt, Lam, closure).umax
    box = P.Box(int(cfg["box"]["N"]), float(cfg["box"]["L"]))
    sp = P.StepParams.from_config(cfg["stepper"])
    if sp.closure != closure:
        raise ValueError("stepper.closure and velocity.closure differ")
    base = args.base or P.DATA
    fo = H.WhiteForcing(box, cfg["forcing"], sl.p[0])
    resume_dir = find_resumable(base, cfg["label"]) if args.resume else None
    rs: dict = {}
    if resume_dir is not None:
        with np.load(resume_dir / RESUME_FILE) as z:
            rs = {k: z[k] for k in z.files}
        if str(rs["config_cmp"]) != config_for_resume(cfg):
            raise SystemExit(f"{resume_dir.name}: its config differs from {args.config} (other than time.t_end); "
                             "not resuming")
        run = P.Run.reopen(resume_dir, cfg, args.config, str(rs["started_local"]), float(rs["elapsed"]))
        zp, zm, g, t = rs["zp"], rs["zm"], rs["g"], float(rs["t"])
        fo.set_keys(rs["key_A"], rs["key_g"])
    else:
        run = P.Run(cfg["label"], cfg, args.config, base=base)
        zp, zm, g, t, keys = initial_fields(box, M, cfg)
        if keys:
            fo.set_keys(keys["key_A"], keys["key_g"])
    state = P.make_state(box, zp, zm, g, Lam, beta, t=t)
    post, energies = H.make_accounting(box, sl, sp)
    sample = H.make_sampler(box, sl, s, sp, n_outer=int(cfg.get("diagnostics", {}).get("n_outer", 2)))
    N6 = float(box.Ntot) ** 2
    wj = jnp.asarray(box.w)
    kp2j = jnp.asarray(box.KP2)
    p0 = float(sl.p[0])

    @jax.jit
    def ea_w0(zp_, zm_, g0_):
        return (jnp.sum(0.25 * wj * kp2j * (jnp.abs(zp_) ** 2 + jnp.abs(zm_) ** 2)) / N6,
                p0 * jnp.sum(wj * jnp.abs(g0_) ** 2) / N6)

    t_end = float(tc["t_end"])
    dt_max = float(tc["dt_max"])
    safety = float(tc["safety"])
    dt_rule = str(tc.get("dt_rule", "bound"))          # "bound": k_perp,max max|v|; "exact": Lanczos spectral radius
    if dt_rule not in ("bound", "exact"):
        raise ValueError(f"time.dt_rule must be 'bound' or 'exact', got {dt_rule!r}")
    every = int(tc["recompute_every"])
    safety_A = float(tc.get("safety_alfven", safety))  # safety on the Alfvenic CFL limit (default: safety)
    fast_every = int(tc.get("limit_every", every))     # steps between cheap rescalings of both limits (H.make_fast_limits)
    fast = H.make_fast_limits(box, s, umax) if fast_every < every else None
    ref: tuple[float, float, float, float] | None = None
    s_every = int(tc["sample_every"])
    p_every = int(tc.get("print_every", 200))
    snap_dt = float(tc.get("snapshot_every", math.inf))
    rst_dt = float(tc.get("restart_every", math.inf))
    t_start = float(rs["t_start"]) if rs else t
    if rs:
        print(f"[{run.run_id}] resuming in place from t = {t:.4f}, step {int(rs['n'])} ({resume_dir})", flush=True)
    print(f"[{run.run_id}] N = {box.N}, M = {M}, Lambda = {Lam:+.4f}, beta_i = {beta}, closure {closure}; "
          f"eta = {sp.eta} (r = {sp.hyper_r}), nu = {sp.nu} (n = {sp.hyper_n}), scheme {sp.scheme}; dt from u_max = "
          f"{umax:.4f} (M = {M_dt}), rule {dt_rule}, safety {safety} (Alfvenic {safety_A}), limits every {every} steps "
          f"(rescaled every {fast_every}), dt_max {dt_max}; t = {t:.3f} -> {t_end}", flush=True)
    print(f"    forcing: {json.dumps(fo.describe())}", flush=True)

    EA_s, pW_s = (np.asarray(x) for x in energies(state.z_plus, state.z_minus, state.g))
    acc = {k: 0.0 for k in ACC_KEYS}
    acc_le = np.zeros(M + 1)
    acc_ln = np.zeros(M + 1)
    acc_errm = np.zeros(M + 1)
    rec: list[dict] = []
    tser: list[float] = []
    dts_all: list[float] = []
    lims: list[dict] = []
    n = 0
    dt = dt_max
    next_snap = (math.floor(t / snap_dt) + 1) * snap_dt if math.isfinite(snap_dt) else math.inf
    next_rst = t + rst_dt
    n_snap = 0
    wall0 = time.time()
    wall_steps = 0.0
    if rs:
        for k in ACC_KEYS:
            acc[k] = float(rs[f"acc_{k}"])
        acc_le, acc_ln, acc_errm = rs["acc_le"].copy(), rs["acc_ln"].copy(), rs["acc_errm"].copy()
        recs = {k[5:]: rs[k] for k in rs if k.startswith("rec__")}
        rec = [{k: v[i] for k, v in recs.items()} for i in range(len(recs["t"]))]
        dts_all = [float(x) for x in rs["dts_all"]]
        lims = json.loads(str(rs["lims_json"]))
        n, dt = int(rs["n"]), float(rs["dt"])
        EA_s, pW_s = np.asarray(rs["EA_s"]), np.asarray(rs["pW_s"]).copy()   # as accumulated, not recomputed
        ref = tuple(float(x) for x in rs["ref"]) if rs["ref"].size else None
        next_snap, next_rst, n_snap = float(rs["next_snap"]), float(rs["next_rst"]), int(rs["n_snap"])
        wall0 = time.time() - float(rs["elapsed"])
        wall_steps = float(rs["wall_steps"])

    def record(d: dict) -> None:
        d = {k: np.asarray(x) for k, x in d.items()}
        d.update(t=np.asarray(t), dt=np.asarray(dt))
        for k in ACC_KEYS:
            d[f"acc_{k}"] = np.asarray(acc[k])
        d["acc_loss_eta"] = acc_le.copy()
        d["acc_loss_nu"] = acc_ln.copy()
        d["acc_err_Wm"] = acc_errm.copy()
        rec.append(d)

    def save_restart(path: str) -> None:
        np.savez(path, zp=np.asarray(state.z_plus), zm=np.asarray(state.z_minus), g=np.asarray(state.g),
                 t=np.asarray(t), n=np.asarray(n), **fo.keys())

    def save_resume() -> None:
        """The complete loop state, written atomically (resume.npz), for --resume."""
        tmp = run.dir / "resume.tmp.npz"
        np.savez(tmp, zp=np.asarray(state.z_plus), zm=np.asarray(state.z_minus), g=np.asarray(state.g),
                 t=np.asarray(t), n=np.asarray(n), dt=np.asarray(dt), t_start=np.asarray(t_start),
                 ref=np.asarray(ref if ref is not None else [], dtype=float),
                 next_snap=np.asarray(next_snap), next_rst=np.asarray(next_rst), n_snap=np.asarray(n_snap),
                 elapsed=np.asarray(time.time() - wall0), wall_steps=np.asarray(wall_steps),
                 acc_le=acc_le, acc_ln=acc_ln, acc_errm=acc_errm, dts_all=np.asarray(dts_all),
                 EA_s=np.asarray(EA_s), pW_s=np.asarray(pW_s),
                 lims_json=np.asarray(json.dumps([{k: float(v) for k, v in x.items()} for x in lims])),
                 config_cmp=np.asarray(config_for_resume(cfg)), started_local=np.asarray(run.started_local),
                 **{f"acc_{k}": np.asarray(acc[k]) for k in ACC_KEYS}, **fo.keys(),
                 **{f"rec__{k}": v for k, v in P.stack_records(rec).items()})
        os.replace(tmp, run.dir / RESUME_FILE)

    if not rs:
        record(sample(state.z_plus, state.z_minus, state.g, fo.F_g_last))
    while t < t_end - 1e-12:
        if n % every == 0:
            lim = H.dt_limits(box, state.z_plus, state.z_minus, s, umax)
            if dt_rule == "exact" and lim["rho_bound"] > 0:
                rho = H.exact_slice_radius(box, state.z_plus, state.z_minus, s, umax, tol=1e-3)
                lim.update(rho_exact=rho, dt_slice_exact=P.LAWSON_RK4_LIMIT / rho)
                dts_lim = lim["dt_slice_exact"]
            else:
                dts_lim = lim["dt_slice"]
            dt = min(dt_max, safety * dts_lim, safety_A * lim["dt_alfven"])
            lim.update(t=t, dt=dt)
            lims.append(lim)
            if fast is not None:
                V0, vp0 = (float(x) for x in fast(state.z_plus, state.z_minus))
                ref = (dts_lim, lim["v_perp_max"], V0, vp0)
        elif fast is not None and ref is not None and n % fast_every == 0:
            # between full evaluations: scale both limits by the change of the (cheap) maxima since the last one
            dts_ref, vp_ref, V0, vp0 = ref
            V1, vp1 = (float(x) for x in fast(state.z_plus, state.z_minus))
            dt_A = (box.L / box.N) / max(sp.v_A, vp_ref * vp1 / max(vp0, 1e-300))
            dt = min(dt_max, safety * dts_ref * V0 / max(V1, 1e-300), safety_A * dt_A)
        dt_step = min(dt, t_end - t) if t_end - t < 1.5 * dt else dt
        tw = time.time()
        state = gandalf_step(state, dt_step, eta=sp.eta, v_A=sp.v_A, nu=sp.nu, hyper_r=sp.hyper_r,
                             hyper_n=sp.hyper_n, scheme=sp.scheme, eta_z=sp.eta_z, hyper_rz=sp.hyper_rz,
                             closure=sp.closure)
        EA1, pW1, lA, le, ln = (np.asarray(x) for x in post(state.z_plus, state.z_minus, state.g, dt_step))
        wall_steps += time.time() - tw
        acc["err_A"] += float(EA1 + lA - EA_s)
        errm = pW1 + le + ln - pW_s
        acc_errm += errm
        acc["err_W"] += float(errm.sum())
        acc["loss_A"] += float(lA)
        acc_le += le
        acc_ln += ln
        state = fo.apply(state, dt_step)
        EA2, w02 = (float(x) for x in ea_w0(state.z_plus, state.z_minus, state.g[..., 0]))
        acc["inj_A"] += EA2 - float(EA1)
        acc["inj_g"] += w02 - float(pW1[0])
        EA_s = EA2
        pW_s = pW1.copy()
        pW_s[0] = w02
        t += dt_step
        n += 1
        acc["steps"] += 1
        acc["sum_dt"] += dt_step
        dts_all.append(dt_step)
        if not math.isfinite(EA2) or not np.all(np.isfinite(pW_s)):
            raise RuntimeError(f"non-finite energies at t = {t}, step {n}")
        if n % s_every == 0 or t >= t_end - 1e-12:
            record(sample(state.z_plus, state.z_minus, state.g, fo.F_g_last))
            acc = {k: 0.0 for k in ACC_KEYS}
            acc_le[:] = 0.0
            acc_ln[:] = 0.0
            acc_errm[:] = 0.0
        if n % p_every == 0:
            r = rec[-1]
            el = time.time() - wall0
            print(f"    t = {t:8.4f}  step {n:6d}  dt = {dt:.3e}  E_A = {float(r['E']):.4e}  W = "
                  f"{float(r['pWm'].sum()):.4e}  chi_u = {float(r['chi_u']):.3f}  chi_out = {float(r['chi_out']):.3f} "
                  f"(fa {float(r['chi_out_fa']):.3f})  u_rms = {float(r['u_rms']):.3f}  W(M)/W(0) = "
                  f"{float(r['pWm'][-1] / max(r['pWm'][0], 1e-300)):.1e}  res A/g "
                  f"{float(r['EAk'][box.nmax:].sum() / r['EAk'].sum()):.1e}/"
                  f"{float(r['Wmk'][box.nmax:].sum() / max(r['Wmk'].sum(), 1e-300)):.1e}  wall {el:.0f} s "
                  f"({wall_steps / n:.3f} s/step)", flush=True)
        if t >= next_snap - 1e-12:
            np.savez_compressed(run.dir / f"snap_{n_snap:03d}.npz", zp=np.asarray(state.z_plus, np.complex64),
                                zm=np.asarray(state.z_minus, np.complex64), g=np.asarray(state.g, np.complex64),
                                t=np.asarray(t))
            n_snap += 1
            next_snap += snap_dt
        if t >= next_rst - 1e-12:
            save_restart(str(run.dir / "restart.npz"))
            np.savez_compressed(run.dir / "partial_timeseries.npz", **P.stack_records(rec))
            next_rst += rst_dt
            save_resume()
    save_restart(str(run.dir / "restart.npz"))
    if (run.dir / "partial_timeseries.npz").exists():
        os.remove(run.dir / "partial_timeseries.npz")
    wall = time.time() - wall0

    # ---------------------------------------------------------------- gates
    ts = P.stack_records(rec)
    T = ts["t"]
    gc = cfg["gates"]
    width = float(gc["steady_window"])
    covers = T[-1] - t_start >= width
    win = H.window_slice(T, width) if covers else np.ones_like(T, dtype=bool)
    # interval sums are attached to the sample that closes the interval: use those strictly after the window start
    iw = np.where(win)[0]
    first = iw[0]
    sel = np.zeros_like(win)
    sel[first + 1:] = True
    Tw = float(ts["acc_sum_dt"][sel].sum())
    injA, injg = float(ts["acc_inj_A"][sel].sum()), float(ts["acc_inj_g"][sel].sum())
    lossA = float(ts["acc_loss_A"][sel].sum())
    lossg = float(ts["acc_loss_eta"][sel].sum() + ts["acc_loss_nu"][sel].sum())
    errA, errW = float(ts["acc_err_A"][sel].sum()), float(ts["acc_err_W"][sel].sum())
    dEA = float(ts["E"][-1] - ts["E"][first])
    dW = float(ts["pWm"][-1].sum() - ts["pWm"][first].sum())
    closure_A = (dEA - (injA - lossA + errA)) / max(injA, 1e-300)
    closure_W = (dW - (injg - lossg + errW)) / max(injg, 1e-300)
    ctol = float(gc["conservation_tol"])
    g_cons = dict(applies=True, criterion=("ideal-part (time-stepping) error of the energy below "
                                           f"{ctol} of the injected energy over the window, Alfvenic and free energy; "
                                           "forcing and dissipation accounted exactly per step"), tol=ctol,
                  alfvenic=H.gate_budget_closure("E_A", errA, injA, ctol),
                  free_energy=H.gate_budget_closure("W", errW, injg, ctol),
                  accounting_identity_residual_A=closure_A, accounting_identity_residual_W=closure_W)
    g_cons["pass"] = bool(all(x.get("pass", False) for x in (g_cons["alfvenic"], g_cons["free_energy"])
                              if x.get("applies", True)))
    btol = float(gc["balance_tol"])
    bal_A = H.gate_energy_balance("E_A", injA, lossA, dEA, Tw, btol)
    bal_W = H.gate_energy_balance("W", injg, lossg, dW, Tw, btol)
    g_bal: dict = dict(applies=bool(covers), criterion=f"|<inj> - <diss>|/<inj> < {btol} over the final "
                       f"{width} tau_A, Alfvenic and free energy", alfvenic=bal_A, free_energy=bal_W)
    g_bal["pass"] = bool(all(x["pass"] for x in (bal_A, bal_W) if x.get("applies", True)))
    if not covers:
        g_bal["note"] = f"not applicable: run covers {T[-1] - t_start:.2f} < {width} tau_A (values over the whole run)"
    g_ss = H.gate_steady_state(T, {"E_A": ts["E"], "W": ts["pWm"].sum(1)}, width, float(gc["steady_tol"]),
                               t_first=t_start)
    dtk = ts["acc_sum_dt"]
    EAk = H.time_average(ts["EAk"], dtk, first)
    Wk = H.time_average(ts["Wmk"], dtk, first).sum(1)
    pWm = H.time_average(ts["pWm"], dtk, first)
    Dnu = H.time_average(ts["Dnu"], dtk, first)
    nper = np.arange(len(EAk))
    g_spec = H.gate_spectral_sanity(nper, EAk, Wk, box.nmax, float(EAk[box.nmax:].sum() / EAk.sum()),
                                    float(Wk[box.nmax:].sum() / max(Wk.sum(), 1e-300)), float(gc["resolution_tol"]),
                                    pWm, Dnu, float(gc["m_tol"]))
    if not covers:
        g_spec["applies"] = False
        g_spec["note"] = "not applicable: run shorter than the steady-state window (values recorded)"
    gates = P.make_gates(g_cons, energy_balance=g_bal, steady_state=g_ss, spectral_sanity=g_spec)

    # per-m budget over the window with the sampled fluxes (field-aligned vs k_z-only)
    Pfa = H.time_average(ts["Pi_fa"], dtk, first)
    Pkz = H.time_average(ts["Pi_kz"], dtk, first)
    Dm = (ts["acc_loss_eta"][sel].sum(0) + ts["acc_loss_nu"][sel].sum(0)) / Tw
    dWm = (ts["pWm"][-1] - ts["pWm"][first]) / Tw
    inj_m = np.zeros(M + 1)
    inj_m[0] = injg / Tw

    def resid(Pi: np.ndarray) -> np.ndarray:
        Pe = np.concatenate([[0.0], Pi, [0.0]])
        return (Pe[:-1] - Pe[1:]) + inj_m - Dm - dWm

    res_fa, res_kz = resid(Pfa), resid(Pkz)
    eps_g = injg / Tw
    def wmean(key: str) -> float:
        return float(H.time_average(ts[key], dtk, first))

    sm = wmean("chi_u")
    extra = dict(n_steps=n, wall_s=wall, s_per_step=wall_steps / n, t_start=t_start, t_end=float(T[-1]),
                 steps_per_tauA=n / (T[-1] - t_start), dt_mean=float(np.mean(dts_all)),
                 dt_min=float(np.min(dts_all[:-1] if n > 1 else dts_all)), dt_max=float(np.max(dts_all)), umax_dt=umax, M_dt=M_dt, forcing=fo.describe(),
                 window=[float(T[first]), float(T[-1])], window_length=Tw,
                 eps_A_measured=injA / Tw, eps_g_measured=eps_g, diss_A=lossA / Tw, diss_W=lossg / Tw,
                 per_m_budget_max_resid_fa_over_eps=float(np.abs(res_fa).max() / eps_g) if eps_g > 0 else None,
                 per_m_budget_max_resid_kz_over_eps=float(np.abs(res_kz).max() / eps_g) if eps_g > 0 else None,
                 chi_u_window_mean=sm, chi_out_window_mean=wmean("chi_out"), chi_out_fa_window_mean=wmean("chi_out_fa"),
                 u_rms_window_mean=wmean("u_rms"), b_rms_window_mean=wmean("b_rms"), E_A_window_mean=wmean("E"),
                 dt_limits_last=lims[-1] if lims else {}, n_snapshots=n_snap)
    arrays = dict(ts=ts, u=sl.u, p=sl.p, R=sl.R, Wl=sl.Wl, A=sl.A, nperp=nper, nz=np.arange(ts["EAkz"].shape[1]),
                  dt_steps=np.array(dts_all), lim_t=np.array([x["t"] for x in lims]),
                  lim_dt_slice=np.array([x["dt_slice"] for x in lims]),
                  lim_dt_alfven=np.array([x["dt_alfven"] for x in lims]),
                  lim_V_slice=np.array([x["V_slice"] for x in lims]),
                  lim_dt_slice_exact=np.array([x.get("dt_slice_exact", np.nan) for x in lims]),
                  res_fa=res_fa, res_kz=res_kz,
                  N=np.array(box.N), M=np.array(M), k1=np.array(box.k1), nmax=np.array(box.nmax))
    notes = (f"forced KRMHD, {box.N}^3, M = {M}, Lambda = {Lam:.4f}, beta_i = {beta:g}, {sp.scheme}, eta = {sp.eta:g} "
             f"(r = {sp.hyper_r}), nu = {sp.nu:g} (n = {sp.hyper_n}); eps_A = {fo.eps_A:g}, eps_g = {fo.eps_g:g} "
             f"({fo.gen_g}); t = {t_start:.2f}..{T[-1]:.2f} tau_A, {n} steps, {wall_steps / n:.3f} s/step; "
             f"window chi_u = {sm:.2f}, chi_out = {extra['chi_out_window_mean']:.2f}; "
             f"time-stepping error / injection: A {errA / max(injA, 1e-300):.1e}, W {errW / max(injg, 1e-300):.1e}; "
             f"inj vs diss: A {bal_A.get('rel_diff', float('nan')):.3f}, W {bal_W.get('rel_diff', float('nan')):.3f}")
    meta = run.save(arrays, gates, notes=notes, extra=extra)
    if (run.dir / RESUME_FILE).exists():
        os.remove(run.dir / RESUME_FILE)
    print(run.summary(meta), flush=True)
    print("gates:", json.dumps({k: (gv.get("pass") if isinstance(gv, dict) else gv) for k, gv in gates.items()}))
    print("extra:", json.dumps(P._jsonable(extra))[:3000])
    print("run-log row:", P.run_log_row(meta), sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
