#!/usr/bin/env python
"""Run family A: phase-velocity slices of one compressive branch in decaying random Alfvenic fields (paper section 3,
thm:slice, and section 7, thm:trunc), for the figure fig:snapshots and the decoupling check of section 8.

Setup (configs/slices_turbulence.yaml): N^3 periodic box of side L = 1, M + 1 Hermite moments (zero closure),
Lambda = sqrt5, beta_i = 1; balanced random Gaussian Elsasser fields z+- with equal energies, k_perp L/2 pi in
[1, 3], |k_z L/2 pi| <= 2, amplitude set by chi = kbar_perp z_rms/(kbar_par v_A); g(0) = f e_0 with f a localised,
band-limited Gaussian density blob; hyper-resistivity exp(-eta (k_perp^2/k_perp,max^2)^r dt) on z+- and on every g_m
(hence on every slice alike); no forcing, no hypercollisions; GANDALF's gandalf_step (Lawson-RK4) with
dt = safety x min(dt_A, 2 sqrt2/rho), rho = k_perp,max max_j max_x |grad_perp(Phi + s u_j Psi)|, recomputed every few
steps and shortened to land on the output times (0, 0.5 and 1.5 outer eddy times tau = 1/(kbar_perp z_rms)).

Saved at each output time: the slice amplitudes a_j(x, y, z0) of every slice, g_0, Phi and Psi on the plane z = z0
(exact Fourier interpolation to a grid refined by `pad`), the slice energies <a_j^2>, W(m) = p_m <g_m^2>, the k_perp
spectra of the free energy and of the Alfvenic energy, and the decoupling check: the slice projection w_j . dg/dt of
GANDALF's right-hand side compared with -{Phi + s u_j Psi, a_j} - s u_j d_z a_j computed independently (numpy
brackets with 2x padded products), with the control that advects every slice by Phi alone.  The energy of each slice
and of the Alfvenic fields is tracked with its exact accumulated hyper-resistive loss (from the multiplicative damping
factor of each step), so E_j(t) + D_j(t) = E_j(0) up to the time-stepping error of the ideal part.

Gates: energy budget (free energy and Alfvenic energy including the accumulated hyper-resistive loss) within 1%;
resolution (energy fraction in the outermost k_perp shells at the last output) below the configured tolerance; energy
balance, steady state and spectral sanity not applicable (short verification run).

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/scripts/run_slices_turbulence.py paper/phase-velocity-slices/configs/slices_turbulence.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p1lib as P  # noqa: E402  (first: float64 JAX on the CPU)
import numpy as np  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def alfvenic_fields(box: P.Box, cfg: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Balanced z+-: independent Gaussian fields with equal rms gradients, scaled to the target chi."""
    a = cfg["alfvenic"]
    rng = np.random.default_rng(int(a["seed"]))
    nperp, nz = tuple(float(x) for x in a["nperp"]), float(a["nzmax"])
    zp = P.normalise_rms_grad(box, box.random_field(rng, nperp, nz), 1.0)
    zm = P.normalise_rms_grad(box, box.random_field(rng, nperp, nz), 1.0)
    chi1 = P.nonlinearity_parameter(box, zp, zm)["chi_z"]
    scale = P.parse_number(a["chi"]) / chi1
    zp, zm = zp * scale, zm * scale
    return zp, zm, P.nonlinearity_parameter(box, zp, zm)


def damping_exponent(box: P.Box, sp: P.StepParams) -> np.ndarray:
    """eta (k_perp^2/k_perp,max^2)^r per mode with GANDALF's normalisation (k_perp,max^2 = k_x[(N-1)//3]^2 +
    k_y[(N-1)//3]^2): the hyper-resistive damping rate applied to z+- and to every g_m."""
    return sp.eta * (box.KP2 / box.kperp_max2_gandalf) ** sp.hyper_r


def step_losses(box: P.Box, a: np.ndarray, zp: np.ndarray, zm: np.ndarray, rate: np.ndarray, dt: float):
    """Energy removed by the multiplicative damping of the last step, from the post-step fields: for a field f with
    damping factor D = exp(-rate dt), the pre-damping energy exceeds the post-damping one by
    sum_k w (D^-2 - 1)|f_k|^2/N^6.  Returns (per-slice losses [M+1], Alfvenic loss)."""
    fac = np.expm1(2.0 * rate * dt) * box.w / box.Ntot ** 2           # [Nz, Ny, Nkx]
    dE = np.einsum("zyx,zyxj->j", fac, np.abs(a) ** 2)
    dA = 0.25 * float(np.sum(fac * box.KP2 * (np.abs(zp) ** 2 + np.abs(zm) ** 2)))
    return dE, dA


def decoupling_check(box: P.Box, sl: P.Slices, s: float, rhs_g, zp, zm, g) -> dict:
    """Compare the slice projection of GANDALF's dg/dt with the slice equation computed independently.

    For every slice j: err_l2 = ||w_j.dg/dt - pred_j||/||pred_j|| (volume L2 norms) and err_max (max |.| over
    modes), pred_j = -{Phi + s u_j Psi, a_j} - s u_j d_z a_j with numpy brackets (2x padded products); err_l2_global =
    ||w_j.dg/dt - pred_j||/max_l ||pred_l|| (round-off relative to the largest slice rate: the tail slices carry
    little energy, so their own relative error includes cancellation among the O(1) moments g_m); control_l2 the
    same as err_l2 with -{Phi, a_j} in place of -{Phi + s u_j Psi, a_j}; ratio = ||{Phi + s u_j Psi, a_j}|| /
    ||s u_j d_z a_j||; rate_l2 = ||pred_j||."""
    phi, psi = P.phi_psi(zp, zm)
    dg = np.asarray(rhs_g(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g)))
    a = sl.amplitudes(g)
    da = sl.amplitudes(dg)
    out = {k: np.zeros(sl.M + 1) for k in ("err_l2", "err_max", "control_l2", "ratio", "rate_l2", "abs_l2")}
    for j in range(sl.M + 1):
        adv = box.bracket_independent(phi + s * sl.u[j] * psi, a[..., j])
        stream = s * sl.u[j] * box.dz(a[..., j])
        pred = -adv - stream
        nrm = math.sqrt(box.avg2(pred))
        out["abs_l2"][j] = math.sqrt(box.avg2(da[..., j] - pred))
        out["rate_l2"][j] = nrm
        out["err_l2"][j] = out["abs_l2"][j] / nrm
        out["err_max"][j] = float(np.abs(da[..., j] - pred).max() / np.abs(pred).max())
        ctrl = -box.bracket_independent(phi, a[..., j]) - stream
        out["control_l2"][j] = math.sqrt(box.avg2(da[..., j] - ctrl)) / nrm
        out["ratio"][j] = math.sqrt(box.avg2(adv) / box.avg2(stream)) if abs(sl.u[j]) > 1e-12 else np.inf
    out["err_l2_global"] = out["abs_l2"] / out["rate_l2"].max()
    return out


def snapshot(box: P.Box, sl: P.Slices, s: float, rhs_g, zp, zm, g, cfg: dict) -> dict:
    """Everything saved at one output time."""
    o = cfg["output"]
    z0, pad = float(o["z0"]), int(o["pad"])
    phi, psi = P.phi_psi(zp, zm)
    a = sl.amplitudes(g)
    rec = dict(a_plane=box.plane_z(a, z0, pad), g0_plane=box.plane_z(g[..., 0], z0, pad),
               phi_plane=box.plane_z(phi, z0, pad), psi_plane=box.plane_z(psi, z0, pad),
               E_slice=P.slice_energies(box, a), pW_m=sl.p * P.W_m(box, g), W_kperp=P.W_kperp(box, g, sl.p))
    ea = P.alfvenic_energy_density_k(box, zp, zm)
    shell = np.broadcast_to(box.nperp, ea.shape)
    rec["EA_kperp"] = np.bincount(shell.ravel(), weights=ea.ravel(), minlength=box.nperp_band + 1)[:box.nperp_band + 1]
    rec.update({f"res_{k}": v for k, v in P.resolution_check(box, sl, zp, zm, g).items() if k != "nmax"})
    rec.update({f"dec_{k}": v for k, v in decoupling_check(box, sl, s, rhs_g, zp, zm, g).items()})
    rec.update(P.alfvenic_energy(box, zp, zm))
    rec["Q_1"] = float(np.sum(rec["E_slice"]))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--base", default=None, help="data folder (default paper/phase-velocity-slices/data; tests only)")
    ap.add_argument("--N", type=int, default=None, help="override the grid (tests only)")
    args = ap.parse_args()
    cfg = P.load_config(args.config)
    if args.N:
        cfg["box"]["N"] = args.N
    v = cfg["velocity"]
    M, Lam, beta = int(v["M"]), P.parse_number(v["Lambda"]), P.parse_number(v["beta_i"])
    s = math.sqrt(beta)
    sl = P.make_slices(M, Lam, v["closure"])
    run = P.Run(cfg["label"], cfg, args.config, base=args.base or P.DATA)
    box = P.Box(int(cfg["box"]["N"]), float(cfg["box"]["L"]))
    sp = P.StepParams.from_config(cfg["stepper"])
    zp, zm, chi = alfvenic_fields(box, cfg)
    b = cfg["blob"]
    g = np.zeros(zp.shape + (M + 1,), complex)
    g[..., 0] = P.parse_number(b["amplitude"]) * box.gaussian_blob([float(x) for x in b["sigma"]],
                                                                   [float(x) for x in b["center"]])
    tau = 1.0 / (chi["kperp_mean"] * chi["z_rms"])           # outer eddy time 1/(kbar_perp z_rms) at t = 0
    t_out = [0.0] + [float(x) * tau for x in cfg["time"]["t_out_eddy"]]
    _, rhs_g = P.make_rhs(box, M, Lam, beta, v["closure"])
    rate = damping_exponent(box, sp)
    print(f"[{run.run_id}] N = {box.N}, M = {M}, Lambda = {Lam:+.4f}, beta_i = {beta}; chi = {chi['chi_z']:.3f} "
          f"(chi_u = {chi['chi_u']:.3f}), z_rms = {chi['z_rms']:.4f}, u_rms = {chi['u_rms']:.4f}, kbar_perp/k1 = "
          f"{chi['kperp_mean'] / box.k1:.3f}, kbar_par/k1 = {chi['kpar_mean'] / box.k1:.3f}; tau = {tau:.4f}; "
          f"t_out = {[round(t, 5) for t in t_out]}; eta = {sp.eta}, r = {sp.hyper_r}", flush=True)

    state = P.make_state(box, zp, zm, g, Lam, beta)
    snaps = [snapshot(box, sl, s, rhs_g, zp, zm, g, cfg)]
    print(f"    t = 0: decoupling max err_l2 {snaps[0]['dec_err_l2'].max():.1e}, control min "
          f"{snaps[0]['dec_control_l2'][np.abs(sl.u) > 1e-12].min():.2f}", flush=True)
    E0 = snaps[0]["E_slice"].copy()
    EA0 = snaps[0]["E"]
    D = np.zeros(M + 1)
    DA = 0.0
    ts, Es, Ds, EAs, DAs, dts, rhos, rho_dt, binds = [0.0], [E0], [D.copy()], [EA0], [0.0], [], [], [], []
    t, n, k_out = 0.0, 0, 1
    every = int(cfg["time"]["recompute_every"])
    dt = None
    while k_out < len(t_out):
        if n % every == 0 or dt is None:
            zp_, zm_, _ = P.fields_of(state)
            rec = P.recommended_dt(box, zp_, zm_, sl, s, safety=float(cfg["time"]["safety"]))
            dt_base = rec["dt"]
            rhos.append(rec["rate"])
            rho_dt.append(rec["rate"] * dt_base)
            binds.append(rec["binding"])
        dt = min(dt_base, t_out[k_out] - t)
        landing = abs(t + dt - t_out[k_out]) < 1e-12
        state = P.gandalf_advance(state, dt, 1, sp)
        t = t_out[k_out] if landing else t + dt
        n += 1
        zp_, zm_, g_ = P.fields_of(state)
        a_ = sl.amplitudes(g_)
        dE, dA = step_losses(box, a_, zp_, zm_, rate, dt)
        D += dE
        DA += dA
        dts.append(dt)
        if n % every == 0 or landing:
            ts.append(t)
            Es.append(P.slice_energies(box, a_))
            Ds.append(D.copy())
            EAs.append(P.alfvenic_energy(box, zp_, zm_)["E"])
            DAs.append(DA)
            if not np.all(np.isfinite(Es[-1])):
                raise RuntimeError(f"non-finite state at t = {t}")
        if landing:
            snaps.append(snapshot(box, sl, s, rhs_g, zp_, zm_, g_, cfg))
            bud = (Es[-1] + Ds[-1]) / E0 - 1.0
            print(f"    t = {t:.5f} ({t / tau:.2f} tau, step {n}, dt = {dt_base:.2e}): decoupling max err_l2 "
                  f"{snaps[-1]['dec_err_l2'].max():.1e} (global {snaps[-1]['dec_err_l2_global'].max():.1e}; "
                  f"err_max {snaps[-1]['dec_err_max'].max():.1e}), control min "
                  f"{snaps[-1]['dec_control_l2'][np.abs(sl.u) > 1e-12].min():.2f}; free-energy budget "
                  f"{(Es[-1].sum() + Ds[-1].sum()) / E0.sum() - 1:.1e}, per-slice max {np.abs(bud).max():.1e}; "
                  f"E_j/E_j(0) {np.round(Es[-1] / E0, 4).tolist()}; resolution g {snaps[-1]['res_g_beyond']:.1e}, "
                  f"Alfvenic {snaps[-1]['res_alfven_beyond']:.1e}", flush=True)
            k_out += 1

    Es, Ds = np.array(Es), np.array(Ds)
    EAs, DAs = np.array(EAs), np.array(DAs)
    tol = float(cfg["gates"]["budget_tol"])
    budget_W = float(np.max(np.abs((Es.sum(1) + Ds.sum(1)) / E0.sum() - 1.0)))
    budget_A = float(np.max(np.abs((EAs + DAs) / EA0 - 1.0)))
    g_budget = dict(applies=True, criterion=f"|E(t) + accumulated hyper-resistive loss - E(0)|/E(0) < {tol} for the "
                    "free energy sum_j <a_j^2> and the Alfvenic energy (hyper-resistive run: plain conservation does "
                    "not apply; the damping loss of every step is computed exactly from its multiplicative factor)",
                    tol=tol, free_energy_budget_residual=budget_W, alfvenic_energy_budget_residual=budget_A,
                    free_energy_dissipated_fraction=float(Ds[-1].sum() / E0.sum()),
                    alfvenic_energy_dissipated_fraction=float(DAs[-1] / EA0))
    g_budget["pass"] = budget_W < tol and budget_A < tol
    rtol = float(cfg["gates"]["resolution_tol"])
    res = dict(applies=True, criterion=f"energy fraction in k_perp shells n >= n_max = {box.nmax} (band corners "
               f"included) < {rtol} at the last output, free energy of g and Alfvenic energy", tol=rtol,
               g_beyond=float(snaps[-1]["res_g_beyond"]), alfven_beyond=float(snaps[-1]["res_alfven_beyond"]),
               g_shell=float(snaps[-1]["res_g_shell"]), alfven_shell=float(snaps[-1]["res_alfven_shell"]))
    res["pass"] = res["g_beyond"] < rtol and res["alfven_beyond"] < rtol
    gates = P.make_gates(g_budget, resolution=res)
    arrays = dict(t_out=np.array(t_out), tau=np.array(tau), u=sl.u, p=sl.p, R=sl.R, N=np.array(box.N),
                  coords=box.coords(int(cfg["output"]["pad"])), t_series=np.array(ts), E_slice_series=Es,
                  D_slice_series=Ds, EA_series=EAs, DA_series=DAs, dt_steps=np.array(dts), rho_series=np.array(rhos),
                  snap=P.stack_records(snaps))
    rel = (Es[-1] + Ds[-1]) / E0 - 1.0
    dec = np.array([sn["dec_err_l2"].max() for sn in snaps])
    decg = np.array([sn["dec_err_l2_global"].max() for sn in snaps])
    ctl = np.array([sn["dec_control_l2"][np.abs(sl.u) > 1e-12].min() for sn in snaps])
    extra = dict(chi=chi, tau=tau, t_out=t_out, n_steps=n, dt_min=float(min(dts)), dt_max=float(max(dts)),
                 rho_dt_max=float(max(rho_dt)),
                 binding=sorted(set(binds)), decoupling_err_l2_max_per_time=dec.tolist(),
                 decoupling_err_l2_global_max_per_time=decg.tolist(),
                 decoupling_err_max_max_per_time=[float(sn["dec_err_max"].max()) for sn in snaps],
                 control_l2_min_per_time=ctl.tolist(), slice_budget_residual_final=rel.tolist(),
                 slice_energy_ratio_final=(Es[-1] / E0).tolist(), u=sl.u.tolist())
    notes = (f"family A (slices in decaying turbulence), {box.N}^3, M = {M}, Lambda = sqrt5, beta_i = {beta:g}, "
             f"balanced z+-, chi = {chi['chi_z']:.2f}, hyper-resistivity eta = {sp.eta:g} (r = {sp.hyper_r}), "
             f"Lawson-RK4, {n} steps to t = {t_out[-1]:.4f} (1.5 tau); decoupling vs GANDALF RHS max rel. L2 "
             f"{decg.max():.1e} of the largest slice rate, {dec.max():.1e} per slice (control {ctl.min():.2f}); free-energy budget {budget_W:.1e}, Alfvenic {budget_A:.1e}; "
             f"outer-shell fractions g {res['g_beyond']:.1e}, Alfvenic {res['alfven_beyond']:.1e}")
    meta = run.save(arrays, gates, notes=notes, extra=extra)
    print(run.summary(meta), flush=True)
    print("run-log row:", P.run_log_row(meta), sep="\n")
    P.require_gates(gates)
    return 0


if __name__ == "__main__":
    sys.exit(main())
