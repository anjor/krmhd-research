"""Run 128³ Alfvénic cascade benchmark on Modal A100 with persistent volume.

Goal: achieve fully developed turbulence at 128³ resolution, with enough scale
separation for a clean k⊥^{-5/3} inertial range. Runs three parameter branches
in parallel based on 64³ calibration results.

Output is saved to a Modal volume so checkpoints can be used for:
  - Resuming runs that need more time
  - Downloading spectra for local plotting
  - Starting the Hermite-forcing phase from a known Alfvénic steady state

Usage:
    # Run all three branches in parallel:
    uv run modal run studies/02-collisionality-scan/scripts/modal_128_benchmark.py

    # List output files on the volume:
    modal volume ls krmhd-benchmark-vol /

    # Download results locally:
    modal volume get krmhd-benchmark-vol /alfven128_eta4_f0p003 ./data/benchmark_128/
"""
from __future__ import annotations

import modal

app = modal.App("krmhd-alfven-128")

volume = modal.Volume.from_name("krmhd-benchmark-vol", create_if_missing=True)

krmhd_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]",
        "gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@v0.4.4",
        "numpy",
        "h5py",
        "pyyaml",
        "matplotlib",
    )
)

# Run 7: Low-k_z forcing (balanced Elsasser) with fixed dissipation.
# All Gaussian shell forcing runs blow up at 128³ due to energy pileup
# at high k_perp — regardless of hyper_r (2 or 4) or eta (2 or 4).
# The problem is the forcing exciting high-k_z modes that feed the pileup.
#
# Switch to balanced_elsasser_lowkz: restricts forcing to |n_z| <= 1,
# excludes k_z=0, and uses Gaussian white noise (no 1/k_perp singularity).
# Hold dissipation fixed (eta=2, hyper_r=2) and scan forcing amplitude.
FORCING_MODE = "balanced_lowkz"  # Used in the time-stepping loop

# Run 11: Extend η=20 and η=50 to t=2000. Both survived to t=1000
# with spectra still developing. η=10 blew up at t=990.
BRANCHES = [
    {
        "label": "alfven128_lowkz_f0p02_eta20",
        "eta": 20.0,
        "force_amplitude": 0.02,
        "total_time": 2000,
        "averaging_start": 1500,
        "resume_from": "alfven128_lowkz_f0p02_eta20/checkpoints/checkpoint_t1000.0.h5",
    },
    {
        "label": "alfven128_lowkz_f0p02_eta50",
        "eta": 50.0,
        "force_amplitude": 0.02,
        "total_time": 2000,
        "averaging_start": 1500,
        "resume_from": "alfven128_lowkz_f0p02_eta50/checkpoints/checkpoint_t1000.0.h5",
    },
]

VOL_MOUNT = "/data"


@app.function(
    image=krmhd_image,
    gpu="A100",
    timeout=36000,  # 10 hours
    volumes={VOL_MOUNT: volume},
)
def run_branch(
    label: str,
    eta: float,
    force_amplitude: float,
    total_time: float,
    averaging_start: float,
    hyper_r: int = 2,
    resume_from: str | None = None,
) -> dict:
    """Run a single 128³ Alfvénic cascade branch with checkpointing."""
    import time as _time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

    from krmhd.config import SimulationConfig
    from krmhd.diagnostics import (
        compute_energy,
        energy_spectrum_perpendicular,
        hermite_moment_energy,
    )
    from krmhd.forcing import force_alfven_modes, force_alfven_modes_balanced
    from krmhd.io import save_checkpoint, load_checkpoint
    from krmhd.physics import KRMHDState, initialize_random_spectrum
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step

    # ---- Output directory on the volume ----
    out_dir = Path(VOL_MOUNT) / label
    ckpt_dir = out_dir / "checkpoints"
    spectra_dir = out_dir / "spectra"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fixed physics (matching upstream benchmark) ----
    resolution = 128
    Lx = Ly = Lz = 1.0
    v_A = 1.0
    beta_i = 1.0
    nu = 0.0  # Alfvén-only, no collisions
    # hyper_r passed as function parameter (default=2, now testing 4)
    hyper_n = 2
    M = 10
    cfl_safety = 0.3
    n_force_min = 1
    n_force_max = 2

    # ---- Initial condition or resume ----
    initial_step = 0
    if resume_from is not None:
        ckpt_path = Path(VOL_MOUNT) / resume_from
        state, grid, metadata = load_checkpoint(str(ckpt_path))
        initial_step = int(metadata.get("step", 0))
        print(f"  Resumed from {ckpt_path.name}: t={state.time:.2f}, step={initial_step}")
    else:
        grid = SpectralGrid3D.create(
            Nx=resolution, Ny=resolution, Nz=resolution,
            Lx=Lx, Ly=Ly, Lz=Lz,
        )
        state = initialize_random_spectrum(
            grid, M=M, alpha=5.0 / 3.0, amplitude=0.05,
            k_min=1.0, k_max=15.0, v_th=1.0, beta_i=beta_i, seed=42,
        )

    # ---- Compute fixed dt (benchmark convention) ----
    dt = compute_cfl_timestep(state, v_A, cfl_safety)
    tau_A = Lz / v_A
    total_steps = int(total_time * tau_A / dt)
    checkpoint_interval_time = 10.0  # τ_A between checkpoints
    diagnostic_interval = 100  # steps between energy diagnostics
    snapshot_interval = 500  # steps between spectral snapshots (during averaging)

    print(f"\n{'='*70}")
    print(f"128³ Alfvénic Cascade: {label}")
    print(f"{'='*70}")
    print(f"  eta={eta}, force_amplitude={force_amplitude}")
    print(f"  hyper_r={hyper_r}, hyper_n={hyper_n}, nu={nu}")
    print(f"  dt={dt:.6f}, total_steps≈{total_steps}")
    print(f"  total_time={total_time} τ_A, averaging_start={averaging_start} τ_A")
    print(f"  Device: {jax.devices()}")
    print(f"  Output: {out_dir}")

    # ---- Tracking arrays ----
    etotal_history: list[float] = []
    ekin_history: list[float] = []
    emag_history: list[float] = []
    time_history: list[float] = []
    last_checkpoint_time = float(state.time) if resume_from else 0.0
    wall_start = _time.time()

    # ---- Helper: save perpendicular energy spectrum ----
    def save_spectrum(state: KRMHDState, step: int) -> None:
        k_perp_bins, E_perp = energy_spectrum_perpendicular(state)
        t = float(state.time)
        np.savez(
            spectra_dir / f"spectrum_t{t:06.1f}_step{step:07d}.npz",
            k_perp=np.array(k_perp_bins),
            E_total=np.array(E_perp),
            time=t,
            step=step,
        )

    # ---- Helper: save checkpoint ----
    def do_checkpoint(state: KRMHDState, step: int, desc: str) -> None:
        t = float(state.time)
        path = ckpt_dir / f"checkpoint_t{t:06.1f}.h5"
        metadata = {
            "step": step,
            "eta": eta,
            "force_amplitude": force_amplitude,
            "hyper_r": hyper_r,
            "hyper_n": hyper_n,
            "nu": nu,
            "v_A": v_A,
            "n_force_min": n_force_min,
            "n_force_max": n_force_max,
            "resolution": resolution,
            "label": label,
            "description": desc,
        }
        save_checkpoint(state, str(path), metadata=metadata, overwrite=True)
        volume.commit()  # Persist to volume immediately
        print(f"  💾 Checkpoint: {path.name} ({desc})")

    # ---- Main time-stepping loop ----
    rng_key = jax.random.PRNGKey(42)
    step = initial_step

    while state.time < total_time * tau_A:
        step += 1

        # Apply forcing FIRST (matching upstream benchmark ordering)
        rng_key, subkey = jax.random.split(rng_key)
        if FORCING_MODE == "balanced_lowkz":
            state, rng_key = force_alfven_modes_balanced(
                state,
                amplitude=force_amplitude,
                n_min=n_force_min,
                n_max=n_force_max,
                dt=dt,
                key=subkey,
                max_nz=1,
                include_nz0=False,
                correlation=0.0,
            )
        else:
            state, rng_key = force_alfven_modes(
                state,
                amplitude=force_amplitude,
                n_min=n_force_min,
                n_max=n_force_max,
                dt=dt,
                key=subkey,
            )

        # Advance state (includes dissipation)
        state = gandalf_step(
            state, dt, eta, v_A,
            nu=nu, hyper_r=hyper_r, hyper_n=hyper_n,
        )

        # Fix JAX scalar time
        if not isinstance(state.time, float):
            state = KRMHDState(
                z_plus=state.z_plus, z_minus=state.z_minus,
                B_parallel=state.B_parallel, g=state.g,
                M=state.M, beta_i=state.beta_i, v_th=state.v_th,
                nu=state.nu, Lambda=state.Lambda,
                time=float(state.time), grid=state.grid,
            )

        # ---- Diagnostics ----
        if step % diagnostic_interval == 0:
            energy = compute_energy(state)
            etotal = float(energy["total"])
            ekin = float(energy["kinetic"])
            emag = float(energy["magnetic"])
            t = float(state.time)
            etotal_history.append(etotal)
            ekin_history.append(ekin)
            emag_history.append(emag)
            time_history.append(t)

            # NaN check
            if np.isnan(etotal):
                print(f"  ❌ NaN at step {step}, t={t:.2f} — aborting.")
                do_checkpoint(state, step, "NaN abort")
                break

            if step % (diagnostic_interval * 10) == 0:
                wall_elapsed = _time.time() - wall_start
                print(
                    f"  step {step:7d}  t={t:7.2f}  "
                    f"E_tot={etotal:.4e}  E_kin={ekin:.4e}  E_mag={emag:.4e}  "
                    f"wall={wall_elapsed:.0f}s"
                )

        # ---- Periodic checkpoints (by simulation time) ----
        if state.time - last_checkpoint_time >= checkpoint_interval_time * tau_A:
            do_checkpoint(state, step, f"periodic t={state.time:.1f}")
            last_checkpoint_time = state.time

        # ---- Spectral snapshots during averaging window ----
        if (
            state.time >= averaging_start * tau_A
            and step % snapshot_interval == 0
        ):
            save_spectrum(state, step)

    # ---- Final checkpoint and spectrum ----
    do_checkpoint(state, step, "final")
    save_spectrum(state, step)

    wall_time = _time.time() - wall_start

    # ---- Steady-state assessment on second half ----
    if len(etotal_history) >= 4:
        half = len(etotal_history) // 2
        second_half = etotal_history[half:]
        mean_e = sum(second_half) / len(second_half)
        max_dev = max(abs(e - mean_e) for e in second_half)
        rel_var = max_dev / mean_e if mean_e > 0 else float("inf")
    else:
        rel_var = float("inf")

    # ---- Save time-series to volume ----
    np.savez(
        out_dir / "energy_timeseries.npz",
        time=np.array(time_history),
        E_total=np.array(etotal_history),
        E_kinetic=np.array(ekin_history),
        E_magnetic=np.array(emag_history),
    )
    volume.commit()

    verdict = "STEADY" if rel_var < 0.3 else "NOT_STEADY"
    print(f"\n{'='*70}")
    print(f"  {label}: {verdict}")
    print(f"  E_var={rel_var:.1%}, E_final={etotal_history[-1]:.4e}")
    print(f"  wall_time={wall_time:.0f}s ({wall_time/3600:.1f}h)")
    print(f"{'='*70}")

    return {
        "label": label,
        "eta": float(eta),
        "force_amplitude": float(force_amplitude),
        "verdict": verdict,
        "e_variation": float(rel_var),
        "e_final": float(etotal_history[-1]) if etotal_history else float("nan"),
        "final_time": float(state.time),
        "wall_time": float(wall_time),
        "n_steps": step,
        "n_checkpoints": len(list((Path(VOL_MOUNT) / label / "checkpoints").glob("*.h5"))),
        "n_spectra": len(list((Path(VOL_MOUNT) / label / "spectra").glob("*.npz"))),
    }


@app.local_entrypoint()
def main(resume: bool = False):
    """Submit all 128³ branches in parallel on A100 GPUs."""
    print(f"Submitting {len(BRANCHES)} 128³ benchmark branches...")
    for b in BRANCHES:
        print(f"  {b['label']}: eta={b['eta']}, f={b['force_amplitude']}")
    if resume:
        print("  (resuming from latest checkpoints)")

    futures = []
    for b in BRANCHES:
        resume_path = b.get("resume_from")
        if resume and not resume_path:
            resume_path = f"{b['label']}/checkpoints/checkpoint_t0080.0.h5"
        futures.append(
            run_branch.spawn(
                label=b["label"],
                eta=b["eta"],
                force_amplitude=b["force_amplitude"],
                total_time=b["total_time"],
                averaging_start=b["averaging_start"],
                hyper_r=b.get("hyper_r", 2),
                resume_from=resume_path,
            )
        )

    results = [f.get() for f in futures]

    print(f"\n{'='*80}")
    print("128³ ALFVÉNIC CASCADE BENCHMARK RESULTS")
    print(f"{'='*80}")
    print(f"{'Label':<30s}  {'Verdict':>10s}  {'E_var':>8s}  {'E_final':>12s}  {'t_final':>8s}  {'Wall':>8s}")
    print("-" * 80)
    for r in results:
        print(
            f"{r['label']:<30s}  {r['verdict']:>10s}  "
            f"{r['e_variation']:>7.1%}  {r['e_final']:>12.4e}  "
            f"{r['final_time']:>7.1f}  {r['wall_time']/3600:>7.1f}h"
        )

    print(f"\nResults saved to Modal volume 'krmhd-benchmark-vol'.")
    print("Download with:")
    for r in results:
        print(f"  modal volume get krmhd-benchmark-vol /{r['label']} ./data/benchmark_128/{r['label']}")
