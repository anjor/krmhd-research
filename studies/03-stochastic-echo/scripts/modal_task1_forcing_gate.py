#!/usr/bin/env python3
"""Run Study 3 Task 1's config-driven fluid forcing gate on Modal."""
from __future__ import annotations
import argparse, json, shutil, sys, time
from pathlib import Path
import modal
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from shared.run_utils import detect_hardware, generate_run_id, log_run

app = modal.App("krmhd-study03-task1-forcing")
image = (modal.Image.debian_slim(python_version="3.12").apt_install("git")
 .pip_install("jax[cuda12]", "gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@v0.5.0", "numpy", "pyyaml")
 .add_local_dir(PROJECT_ROOT, remote_path="/root/krmhd-research", copy=True))

@app.function(image=image, gpu="A100", timeout=14400)
def execute(config_yaml: str) -> dict:
    import sys, time
    import jax, numpy as np, yaml
    sys.path.insert(0, "/root/krmhd-research")
    from krmhd.config import SimulationConfig
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step
    from shared.alfven_diagnostics import elsasser_energies, elsasser_perpendicular_spectra, high_k_fraction, sigma_c
    from shared.alfven_forcing import apply_alfven_forcing, pop_alfven_forcing_options
    raw = yaml.safe_load(config_yaml); gate = raw.pop("study3_forcing_gate"); options = pop_alfven_forcing_options(raw); config = SimulationConfig(**raw)
    if config.initial_condition.M != 0: raise ValueError("Task 1 requires M=0")
    grid = config.create_grid(); state = config.create_initial_state(grid); physics, forcing = config.physics, config.forcing
    tau = grid.Lz / physics.v_A; target = gate["target_outer_times"] * tau; interval = gate["sample_interval_outer_times"] * tau
    next_sample = 0.; step = 0; records = []; key = jax.random.PRNGKey(forcing.seed); started = time.time()
    print(f"GPU device: {jax.devices()}")
    while float(state.time) < target:
        dt = compute_cfl_timestep(state, physics.v_A, config.time_integration.cfl_safety)
        state = gandalf_step(state, dt, physics.eta, physics.v_A, nu=0., hyper_r=physics.hyper_r, hyper_n=physics.hyper_n)
        key, force_key = jax.random.split(key); state, _ = apply_alfven_forcing(state, forcing, dt, force_key, options); step += 1
        if float(state.time) >= next_sample:
            ep, em = elsasser_energies(state); n, sp, sm = elsasser_perpendicular_spectra(state)
            records.append({"time":float(state.time),"outer_times":float(state.time/tau),"E_plus":ep,"E_minus":em,"sigma_c":sigma_c(state),"high_k_fraction":high_k_fraction(n,sp,sm,min(grid.Nx,grid.Ny)//3)})
            next_sample += interval
    n, sp, sm = elsasser_perpendicular_spectra(state); steady = np.array([r["outer_times"] for r in records]) >= gate["target_outer_times"]-gate["averaging_outer_times"]
    sigmas = np.array([r["sigma_c"] for r in records])[steady]; tails = np.array([r["high_k_fraction"] for r in records])[steady]; mean = float(np.mean(sigmas)); tail = float(np.max(tails))
    return {"records":records,"steps":step,"wall_time_s":time.time()-started,"mean_sigma_c":mean,"std_sigma_c":float(np.std(sigmas)),"absolute_sigma_error":abs(mean-options.target_sigma_c),"max_high_k_fraction":tail,"nperp":n.tolist(),"E_plus_spectrum":sp.tolist(),"E_minus_spectrum":sm.tolist()}

@app.local_entrypoint()
def main(config: str = "studies/03-stochastic-echo/configs/task1_sigma06_fluid.yaml") -> None:
    path = (PROJECT_ROOT / config).resolve(); raw = yaml.safe_load(path.read_text()); gate, forcing = raw["study3_forcing_gate"], raw["forcing"]
    run_id = generate_run_id("03", f"forcing_sig{forcing['target_sigma_c']:g}"); output = PROJECT_ROOT / raw["io"]["output_dir"] / run_id; output.mkdir(parents=True, exist_ok=False); shutil.copy2(path, output / "config.yaml")
    (output / "prediction.md").write_text(f"# Pre-run prediction\n\nTarget injection sigma_c: {forcing['target_sigma_c']:+.3f}. The final {gate['averaging_outer_times']} outer times should agree within {gate['sigma_tolerance']:.1%}; high-k energy fraction should remain below {gate['max_high_k_fraction']:.1%}.\n")
    result = execute.remote(path.read_text()); passed = result["absolute_sigma_error"] <= gate["sigma_tolerance"] and result["max_high_k_fraction"] <= gate["max_high_k_fraction"]
    summary = {**result,"run_id":run_id,"target_sigma_c":forcing["target_sigma_c"],"pass":passed}; (output / "summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez(output / "elsasser_spectra.npz", nperp=result["nperp"], E_plus=result["E_plus_spectrum"], E_minus=result["E_minus_spectrum"])
    (output / "result.md").write_text(f"# Task 1 forcing-gate result\n\nOutcome: **{'PASS' if passed else 'FAIL'}**\n\nMean sigma_c: {summary['mean_sigma_c']:+.4f} ± {summary['std_sigma_c']:.4f}; target {forcing['target_sigma_c']:+.4f}.\n\nMaximum high-k fraction: {summary['max_high_k_fraction']:.3e}.\n")
    log_run(run_id,str(path.relative_to(PROJECT_ROOT)),detect_hardware()+"; Modal A100",result["wall_time_s"],"pass" if passed else "fail",f"Task 3 forcing gate target={forcing['target_sigma_c']:+.3f}, mean={summary['mean_sigma_c']:+.3f}, tail={summary['max_high_k_fraction']:.2e}")
    print(json.dumps(summary,indent=2))
