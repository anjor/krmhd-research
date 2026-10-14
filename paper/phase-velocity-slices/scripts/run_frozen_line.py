#!/usr/bin/env python
"""Run family C: numerical test of the frozen-line theorem (paper section 6, thm:frozen and cor:singlepoint).

For each resolution N in the config, from the same physical initial fields (drawn on an N0^3 grid and embedded):
  NL  ideal KRMHD advanced with GANDALF's gandalf_step (Lawson-RK4): z+- evolve by RMHD, g is passive;
  FR  frozen-line prediction h(t) = exp(-t s grad_par(0) A) g(0): GANDALF's right-hand side for g with
      (z+, z-) = (Psi0, -Psi0), i.e. Phi = 0 and Psi = Psi(0), classical RK4 at the same dt;
  LZ  k_z-only linear prediction (Phi = Psi = 0), exact per k_z.
By the theorem (ideal induction, no spatial dissipation of g), every volume-integrated single-point statistic of g(t)
taken with grad_par(t) equals that of h(t) taken with grad_par(0); the differences NL - FR must therefore fall with N
for W(m) = p_m <g_m^2>, the slice correlation matrix, the field-aligned Hermite flux and <g_m^4>, and must not for
W(k_perp), W(|k_z|) and the k_z-only flux, which involve Fourier filters.

dt: the largest candidate dt of the config not above safety x min(dt_A, 2 sqrt2/rho) on the finest grid at t = 0
(dt_A the Alfvenic CFL limit, rho = k_perp,max max_j max_x |grad_perp(Phi + s u_j Psi)| the slice-advection rate), with
every output time a multiple of dt; the same dt for every N.  rho(t) dt <= 2 sqrt2 is checked during the run.  The
finest grid is repeated at dt/divide to show that the time-step error is below the spatial error.

Each integration at one N (and the dt check) is a run with ID P1_frozen-N<N>_<YYYYMMDD_HHMMSS> and writes .npz and
.json to paper/phase-velocity-slices/data/<run ID>/.  Gate: the free energy sum_m p_m <g_m^2> and the Alfvenic energy of
NL drift by less than 1% (ideal run); the other gates are recorded as not applicable (short verification run).

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/scripts/run_frozen_line.py paper/phase-velocity-slices/configs/frozen_line.yaml [--only N]
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


def initial_fields(cfg: dict, M: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Phi, Psi and g on the N0^3 grid (rfft coefficients), drawn in the order Phi, Psi, g_0, g_1."""
    ic = cfg["initial"]
    N0 = int(cfg["box"]["N0"])
    b0 = P.Box(N0, float(cfg["box"]["L"]))
    rng = np.random.default_rng(int(ic["seed"]))
    nperp, nz = tuple(float(x) for x in ic["nperp"]), float(ic["nzmax"])
    phi = b0.random_field(rng, nperp, nz)
    psi = b0.random_field(rng, nperp, nz)
    g = np.zeros((N0, N0, N0 // 2 + 1, M + 1), complex)
    g[..., 0] = b0.random_field(rng, nperp, nz)
    g[..., 1] = float(ic["g1_amplitude"]) * b0.random_field(rng, nperp, nz)
    return phi, psi, g, N0


def fields_on(box: P.Box, cfg: dict, sl: P.Slices) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The initial z+, z-, g on this box: embedded and normalised as in the config."""
    ic = cfg["initial"]
    phi0, psi0, g0, N0 = initial_fields(cfg, sl.M)
    phi = P.normalise_rms_grad(box, box.embed(phi0, N0), P.parse_number(ic["rms_grad_phi"]))
    psi = P.normalise_rms_grad(box, box.embed(psi0, N0), P.parse_number(ic["rms_grad_psi"]))
    g = box.embed(g0, N0)
    g = g * math.sqrt(P.parse_number(ic["free_energy"]) / P.free_energy(box, g, sl.p))
    zp, zm = P.elsasser(phi, psi)
    return zp, zm, g


def choose_dt(cfg: dict, sl: P.Slices, s: float) -> tuple[float, dict]:
    """Largest candidate dt below safety x min(dt_A, 2 sqrt2/rho) on the finest grid, with t_out multiples of dt."""
    tc = cfg["time"]
    box = P.Box(max(int(n) for n in cfg["box"]["N"]), float(cfg["box"]["L"]))
    zp, zm, _ = fields_on(box, cfg, sl)
    rec = P.recommended_dt(box, zp, zm, sl, s, safety=float(tc["safety"]))
    for dt in sorted((float(x) for x in tc["dt_candidates"]), reverse=True):
        if dt > rec["dt"]:
            continue
        try:
            P.steps_for_times(tc["t_out"], dt)
        except ValueError:
            continue
        return dt, rec
    raise ValueError(f"no candidate dt satisfies dt <= {rec['dt']:.3e} with the output times as multiples")


def diag(box: P.Box, sl: P.Slices, s: float, zp, zm, g, psi_for_flux, cfg: dict) -> dict:
    """Diagnostics of one state (z+- the advecting fields of that integration); the field-aligned flux uses
    grad_par = d_z + {psi_for_flux, .}: Psi(t) for NL, Psi(0) for FR, 0 for LZ."""
    d = cfg["diagnostics"]
    return P.diagnostics(box, sl, s, zp, zm, g, quartic_ms=tuple(int(m) for m in d["quartic_ms"]),
                         pad=int(d["pad"]), psi_flux=psi_for_flux)


def run_one(cfg: dict, cfg_path: str, N: int, dt: float, dt_info: dict, label: str, base: str | None = None) -> dict:
    """NL, FR and LZ at resolution N with time step dt; saves the run and returns its metadata."""
    v = cfg["velocity"]
    M, Lam, beta = int(v["M"]), P.parse_number(v["Lambda"]), P.parse_number(v["beta_i"])
    s = math.sqrt(beta)
    sl = P.make_slices(M, Lam, v["closure"])
    run = P.Run(label, cfg, cfg_path, base=base or P.DATA)
    box = P.Box(N, float(cfg["box"]["L"]))
    zp0, zm0, g0 = fields_on(box, cfg, sl)
    _, psi0 = P.phi_psi(zp0, zm0)
    sp = P.StepParams.from_config(cfg["stepper"])
    tc = cfg["time"]
    t_out = [float(t) for t in tc["t_out"]]
    n_out = P.steps_for_times(t_out, dt)
    every = int(tc["check_every"])
    chi = P.nonlinearity_parameter(box, zp0, zm0)
    ratio_bs = math.sqrt(box.avg2(box.bracket(P.phi_psi(zp0, zm0)[0], g0[..., 0]))
                         / box.avg2(s * box.dz((g0 @ sl.A.T)[..., 0])))
    print(f"[{run.run_id}] N = {N}, M = {M}, Lambda = {Lam:+.4f}, beta_i = {beta}, dt = {dt}, steps = {n_out[-1]}; "
          f"chi_z = {chi['chi_z']:.2f}; ||{{Phi, g_0}}||/||s d_z (A g)_0|| = {ratio_bs:.2f}", flush=True)

    state = P.make_state(box, zp0, zm0, g0, Lam, beta)
    fr = P.FrozenRK4(box, M, Lam, beta, dt, closure=v["closure"])
    zpF, zmF = jnp.asarray(psi0), jnp.asarray(-psi0)
    gF = jnp.asarray(g0)
    recs = {k: [diag(box, sl, s, zp0, zm0, g0, psi0, cfg)] for k in ("NL", "FR")}
    recs["LZ"] = [diag(box, sl, s, np.zeros_like(zp0), np.zeros_like(zm0), g0, np.zeros_like(psi0), cfg)]
    t_series, W_series, EA_series, rho_series = [0.0], [P.free_energy(box, g0, sl.p)], \
        [P.alfvenic_energy(box, zp0, zm0)["E"]], [P.slice_advection_rate(box, zp0, zm0, s, sl.u)["rate"]]
    n = 0
    for n_target in n_out:
        while n < n_target:
            k = min(every - n % every, n_target - n)
            state = P.gandalf_advance(state, dt, k, sp)
            gF = fr.advance(zpF, zmF, gF, k)
            n += k
            if n % every == 0 or n == n_target:
                zp, zm, g = P.fields_of(state)
                t_series.append(n * dt)
                W_series.append(P.free_energy(box, g, sl.p))
                EA_series.append(P.alfvenic_energy(box, zp, zm)["E"])
                rho_series.append(P.slice_advection_rate(box, zp, zm, s, sl.u)["rate"])
                if not np.isfinite(W_series[-1]) or rho_series[-1] * dt > P.LAWSON_RK4_LIMIT:
                    raise RuntimeError(f"step {n}: rho dt = {rho_series[-1] * dt:.2f} > 2 sqrt2 or non-finite state")
        zp, zm, g = P.fields_of(state)
        _, psi = P.phi_psi(zp, zm)
        t = n * dt
        recs["NL"].append(diag(box, sl, s, zp, zm, g, psi, cfg))
        recs["FR"].append(diag(box, sl, s, zpF, zmF, np.asarray(gF), psi0, cfg))
        gZ = P.kz_only_evolve(box, g0, sl, s, t)
        recs["LZ"].append(diag(box, sl, s, np.zeros_like(zp0), np.zeros_like(zm0), gZ, np.zeros_like(psi0), cfg))
        dW = abs(P.rel_l1(recs["NL"][-1]["pW_m"], recs["FR"][-1]["pW_m"]))
        print(f"    t = {t:.4f}: W(m) NL vs FR {dW:.2e}, NL vs LZ "
              f"{P.rel_l1(recs['NL'][-1]['pW_m'], recs['LZ'][-1]['pW_m']):.2e}; free energy drift "
              f"{abs(W_series[-1] / W_series[0] - 1):.1e}; rho dt = {rho_series[-1] * dt:.2f}", flush=True)

    gate = P.gate_energy_conservation(W_series, EA_series, tol=float(cfg["gates"]["conservation_tol"]))
    gate_fr = dict(applies=True, criterion="frozen-line prediction: slice energies conserved (relative drift < tol)",
                   tol=float(cfg["gates"]["conservation_tol"]),
                   max_slice_energy_drift=float(np.max(np.abs(recs["FR"][-1]["E_slice"] / recs["FR"][0]["E_slice"]
                                                              - 1.0))))
    gate_fr["pass"] = gate_fr["max_slice_energy_drift"] < gate_fr["tol"]
    res = P.resolution_check(box, sl, *P.fields_of(state))
    gates = P.make_gates(gate, resolution=res, frozen_prediction_conservation=gate_fr)
    arrays = dict(t_out=np.array([0.0] + [n_ * dt for n_ in n_out]), t_series=np.array(t_series),
                  W_series=np.array(W_series), EA_series=np.array(EA_series), rho_series=np.array(rho_series),
                  u=sl.u, p=sl.p, A=sl.A, N=np.array(N), dt=np.array(dt), M=np.array(M), Lambda=np.array(Lam),
                  s=np.array(s), kperp_shell_k1=np.arange(box.nperp_band + 1), kz_shell_k1=np.arange(box.nz_band + 1))
    for case, rr in recs.items():
        arrays[case] = P.stack_records(rr)
    extra = dict(N=N, dt=dt, n_steps=n_out[-1], dt_choice=dict(
        rule="largest candidate <= safety x min(dt_A, 2 sqrt2/rho) at t = 0 on the finest grid; t_out multiples of dt",
        safety=dt_info["safety"], dt_alfven=dt_info["dt_alfven"], dt_slice=dt_info["dt_slice"],
        binding=dt_info["binding"], rho_finest=dt_info["rate"]), rho_dt_max=float(np.max(rho_series) * dt),
        chi=chi, bracket_over_streaming_g0=ratio_bs, krmhd_rhs_scheme="NL: gandalf_step lawson_rk4; FR: classical "
        "RK4 on GANDALF's right-hand side for g; LZ: exact matrix exponential")
    notes = (f"family C (frozen-line), N = {N}^3, M = {M}, Lambda = sqrt5, beta_i = {beta:g}, ideal, Lawson-RK4, "
             f"dt = {dt:g}, t = 0..{t_out[-1]:g}; free-energy drift {gate['free_energy_rel_drift']:.1e}, Alfvenic "
             f"energy drift {gate['alfvenic_energy_rel_drift']:.1e}; W(m) NL vs frozen at t = "
             + ", ".join(f"{t_:g}: {P.rel_l1(recs['NL'][i + 1]['pW_m'], recs['FR'][i + 1]['pW_m']):.1e}"
                         for i, t_ in enumerate(t_out)))
    meta = run.save(arrays, gates, notes=notes, extra=extra)
    print(run.summary(meta), flush=True)
    P.require_gates(gates)
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config")
    ap.add_argument("--only", type=int, default=None, help="run only this N (and its dt check if configured)")
    ap.add_argument("--skip-dt-check", action="store_true")
    ap.add_argument("--base", default=None, help="data folder (default paper/phase-velocity-slices/data; tests only)")
    args = ap.parse_args()
    cfg = P.load_config(args.config)
    v = cfg["velocity"]
    sl = P.make_slices(int(v["M"]), P.parse_number(v["Lambda"]), v["closure"])
    s = math.sqrt(P.parse_number(v["beta_i"]))
    dt, info = choose_dt(cfg, sl, s)
    print(f"dt = {dt} (safety x min(dt_A, 2 sqrt2/rho) = {info['dt']:.3e}; dt_A = {info['dt_alfven']:.3e}, "
          f"2 sqrt2/rho = {info['dt_slice']:.3e}, binding: {info['binding']})", flush=True)
    rows = []
    for N in (int(n) for n in cfg["box"]["N"]):
        if args.only is not None and N != args.only:
            continue
        rows.append(P.run_log_row(run_one(cfg, args.config, N, dt, info, f"{cfg['label']}-N{N}", args.base)))
    chk = cfg["time"].get("dt_check")
    if chk and not args.skip_dt_check and (args.only is None or args.only == int(chk["N"])):
        k = int(chk["divide"])
        rows.append(P.run_log_row(run_one(cfg, args.config, int(chk["N"]), dt / k, info,
                                          f"{cfg['label']}-N{int(chk['N'])}-dt{k}", args.base)))
    print("run-log rows:", *rows, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
