"""Run 128³ Hermite calibration on Modal — add g₀ forcing to steady Alfvénic state.

Resume from the η=100 steady Alfvénic checkpoint, expand M from 10 to 128,
add Hermite forcing on g₀, and monitor the collisional dissipation rate ε_ν.

v2: Fixed Lambda=√5 (standard coupling for β_i=1). v1 inherited Lambda=1 from
the Alfvénic checkpoint, which killed the Hermite cascade entirely.

Usage:
    uv run modal run studies/02-collisionality-scan/scripts/modal_128_hermite.py
"""
from __future__ import annotations

import modal

app = modal.App("krmhd-hermite-128")

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

VOL_MOUNT = "/data"

# Hermite calibration: ν=1.0 (ν=0.01 blew up — too weak for M=128)
# v3: Lambda=√5 fix + ν=1.0
BRANCHES = [
    {
        "label": "hermite128_nu1p0_v3",
        "nu": 1.0,
        "hermite_amplitude": 0.0035,
        "total_time": 2500,  # 500 τ_A of Hermite evolution
        "averaging_start": 2400,
        "resume_from": "alfven128_lowkz_f0p02_eta100/checkpoints/checkpoint_t2000.0.h5",
    },
]


@app.function(
    image=krmhd_image,
    gpu="A100",
    timeout=36000,  # 10 hours
    volumes={VOL_MOUNT: volume},
)
def run_hermite_branch(
    label: str,
    nu: float,
    hermite_amplitude: float,
    total_time: float,
    averaging_start: float,
    resume_from: str,
) -> dict:
    """Run 128³ Alfvénic + Hermite cascade."""
    import time as _time
    from pathlib import Path

    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

    from krmhd.diagnostics import (
        compute_energy,
        energy_spectrum_perpendicular,
        hermite_moment_energy,
    )
    from functools import partial
    from krmhd.forcing import force_alfven_modes_balanced
    from krmhd.io import save_checkpoint, load_checkpoint
    from krmhd.physics import KRMHDState, initialize_hermite_moments
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step

    # ---- Inlined low-k_z Hermite forcing (from shared/hermite_forcing.py) ----
    @partial(jax.jit, static_argnums=(10,))
    def _hermite_forcing_lowkz_jit(
        kx, ky, kz, amplitude, kperp_min, kperp_max,
        kz_allowed, dt, real_part, imag_part, nx_full,
    ):
        kx_3d = kx[jnp.newaxis, jnp.newaxis, :]
        ky_3d = ky[jnp.newaxis, :, jnp.newaxis]
        k_perp = jnp.sqrt(kx_3d**2 + ky_3d**2)
        perp_mask = (k_perp >= kperp_min) & (k_perp <= kperp_max)
        kz_mask = kz_allowed[:, jnp.newaxis, jnp.newaxis]
        mask = perp_mask & kz_mask
        scale = amplitude / jnp.sqrt(dt)
        noise = (real_part + 1j * imag_part) * scale
        forced = noise * mask.astype(noise.dtype)
        forced = forced.at[:, :, 0].set(forced[:, :, 0].real.astype(forced.dtype))
        if nx_full % 2 == 0:
            nq = forced.shape[2] - 1
            forced = forced.at[:, :, nq].set(forced[:, :, nq].real.astype(forced.dtype))
        forced = forced.at[0, 0, 0].set(0.0 + 0.0j)
        return forced

    def _hermite_forcing_lowkz(grid, amplitude, n_min, n_max, max_nz, include_nz0, dt, key):
        kperp_min = 2.0 * jnp.pi * n_min / grid.Lx
        kperp_max = 2.0 * jnp.pi * n_max / grid.Lx
        k1 = 2.0 * jnp.pi / grid.Lz
        nz_int = jnp.round(jnp.abs(grid.kz) / k1).astype(jnp.int32)
        kz_allowed = nz_int <= max_nz
        if not include_nz0:
            kz_allowed = kz_allowed & (nz_int != 0)
        key, sk1, sk2 = jax.random.split(key, 3)
        shape = (grid.Nz, grid.Ny, grid.Nx // 2 + 1)
        real_part = jax.random.normal(sk1, shape=shape, dtype=jnp.float32)
        imag_part = jax.random.normal(sk2, shape=shape, dtype=jnp.float32)
        forcing = _hermite_forcing_lowkz_jit(
            grid.kx, grid.ky, grid.kz, amplitude,
            float(kperp_min), float(kperp_max), kz_allowed,
            float(dt), real_part, imag_part, grid.Nx,
        )
        return forcing, key

    # ---- Output directory on the volume ----
    out_dir = Path(VOL_MOUNT) / label
    ckpt_dir = out_dir / "checkpoints"
    spectra_dir = out_dir / "spectra"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    # ---- Fixed physics ----
    eta = 100.0
    hyper_r = 2
    hyper_n = 6  # Hermite hyper-collisions
    M_new = 128  # Hermite resolution
    cfl_safety = 0.3
    v_A = 1.0
    Lz = 1.0
    tau_A = Lz / v_A

    # Alfvén forcing params (same as calibration)
    force_amplitude = 0.02
    n_force_min = 1
    n_force_max = 2

    # Hermite forcing params
    forced_moments = (0,)  # Force g₀ only

    # ---- Load checkpoint and expand Hermite sector ----
    ckpt_path = Path(VOL_MOUNT) / resume_from
    state, grid, metadata = load_checkpoint(str(ckpt_path))
    initial_step = int(metadata.get("step", 0))
    print(f"  Resumed from {ckpt_path.name}: t={state.time:.2f}, step={initial_step}")
    print(f"  Original M={state.M}, expanding to M={M_new}")

    # Expand g from M=10 to M=128: keep existing moments, zero-pad the rest
    old_g = jnp.array(state.g)  # shape (Nz, Ny, Nkx, M_old+1)
    M_old = state.M
    new_g = jnp.zeros(
        (old_g.shape[0], old_g.shape[1], old_g.shape[2], M_new + 1),
        dtype=old_g.dtype,
    )
    new_g = new_g.at[:, :, :, :M_old + 1].set(old_g)

    # Add small Hermite seed to avoid g=0 fixed point
    seed_g = initialize_hermite_moments(
        grid, M_new, perturbation_amplitude=1e-3, seed=137,
    )
    # Only seed moments beyond M_old (don't disturb existing)
    new_g = new_g.at[:, :, :, M_old + 1:].set(seed_g[:, :, :, M_old + 1:])

    # Create new state with expanded Hermite sector
    # Lambda=√5 for β_i=1 — Lambda=1 kills the Hermite cascade entirely
    Lambda_correct = float(jnp.sqrt(5.0))
    print(f"  Setting Lambda={Lambda_correct:.4f} (was {state.Lambda})")
    state = KRMHDState(
        z_plus=state.z_plus,
        z_minus=state.z_minus,
        B_parallel=state.B_parallel,
        g=new_g,
        M=M_new,
        beta_i=state.beta_i,
        v_th=state.v_th,
        nu=nu,
        Lambda=Lambda_correct,
        time=state.time,
        grid=state.grid,
    )

    # ---- Compute fixed dt ----
    dt = compute_cfl_timestep(state, v_A, cfl_safety)
    total_steps = int(total_time * tau_A / dt)
    checkpoint_interval_time = 10.0
    diagnostic_interval = 100
    snapshot_interval = 500

    print(f"\n{'='*70}")
    print(f"128³ Alfvénic + Hermite Cascade: {label}")
    print(f"{'='*70}")
    print(f"  eta={eta}, nu={nu}, hyper_r={hyper_r}, hyper_n={hyper_n}")
    print(f"  M={M_new}, Alfvén f={force_amplitude}, Hermite f={hermite_amplitude}")
    print(f"  forced_moments={forced_moments}")
    print(f"  dt={dt:.6f}, total_time={total_time} τ_A")
    print(f"  Device: {jax.devices()}")
    print(f"  Output: {out_dir}")

    # ---- Tracking ----
    etotal_history: list[float] = []
    eps_nu_history: list[float] = []
    hermite_energy_history: list[float] = []
    time_history: list[float] = []
    last_checkpoint_time = float(state.time)
    wall_start = _time.time()

    # ---- Helper: compute ε_ν ----
    def compute_eps_nu(E_m: np.ndarray) -> float:
        """Collisional dissipation: ε_ν = 2ν Σ_{m≥2} (m/M)^hyper_n E_m."""
        m_idx = np.arange(len(E_m))
        rates = nu * (m_idx / M_new) ** hyper_n
        rates[:2] = 0.0  # m=0,1 exempt
        return float(2.0 * np.sum(rates * E_m))

    # ---- Helper: save spectrum + Hermite diagnostics ----
    def save_spectrum(state: KRMHDState, step: int) -> None:
        k_perp_bins, E_perp = energy_spectrum_perpendicular(state)
        E_m = np.array(hermite_moment_energy(state))
        t = float(state.time)
        np.savez(
            spectra_dir / f"spectrum_t{t:06.1f}_step{step:07d}.npz",
            k_perp=np.array(k_perp_bins),
            E_total=np.array(E_perp),
            E_m=E_m,
            eps_nu=compute_eps_nu(E_m),
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
            "nu": nu,
            "force_amplitude": force_amplitude,
            "hermite_amplitude": hermite_amplitude,
            "hyper_r": hyper_r,
            "hyper_n": hyper_n,
            "v_A": v_A,
            "n_force_min": n_force_min,
            "n_force_max": n_force_max,
            "resolution": 128,
            "M": M_new,
            "label": label,
            "description": desc,
        }
        save_checkpoint(state, str(path), metadata=metadata, overwrite=True)
        volume.commit()
        print(f"  💾 Checkpoint: {path.name} ({desc})")

    # ---- Main loop ----
    rng_key = jax.random.PRNGKey(42)
    step = initial_step

    while state.time < total_time * tau_A:
        step += 1

        # Apply Alfvén forcing FIRST
        rng_key, subkey = jax.random.split(rng_key)
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

        # Apply Hermite forcing (low-k_z, same mask as Alfvén forcing)
        # Hermite cascade rate ~ k_∥, so full-shell forcing would inject
        # energy at high k_z that cascades to high m too fast → pileup.
        rng_key, subkey = jax.random.split(rng_key)
        hermite_forcing_field, rng_key = _hermite_forcing_lowkz(
            state.grid, hermite_amplitude,
            n_force_min, n_force_max,
            1, False,  # max_nz=1, include_nz0=False
            dt, subkey,
        )
        g_new = jnp.array(state.g)
        for m in forced_moments:
            g_new = g_new.at[:, :, :, m].add(hermite_forcing_field)
        state = KRMHDState(
            z_plus=state.z_plus, z_minus=state.z_minus,
            B_parallel=state.B_parallel, g=g_new,
            M=state.M, beta_i=state.beta_i, v_th=state.v_th,
            nu=state.nu, Lambda=state.Lambda,
            time=state.time, grid=state.grid,
        )

        # Advance state
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
            t = float(state.time)

            E_m = np.array(hermite_moment_energy(state))
            eps_nu = compute_eps_nu(E_m)
            hermite_total = float(np.sum(E_m))

            etotal_history.append(etotal)
            eps_nu_history.append(eps_nu)
            hermite_energy_history.append(hermite_total)
            time_history.append(t)

            if np.isnan(etotal) or np.isnan(eps_nu):
                print(f"  ❌ NaN at step {step}, t={t:.2f} — aborting.")
                do_checkpoint(state, step, "NaN abort")
                break

            if step % (diagnostic_interval * 10) == 0:
                wall_elapsed = _time.time() - wall_start
                print(
                    f"  step {step:7d}  t={t:7.2f}  "
                    f"E_tot={etotal:.4e}  ε_ν={eps_nu:.4e}  "
                    f"ΣW_m={hermite_total:.4e}  wall={wall_elapsed:.0f}s"
                )

        # ---- Periodic checkpoints ----
        if state.time - last_checkpoint_time >= checkpoint_interval_time * tau_A:
            do_checkpoint(state, step, f"periodic t={state.time:.1f}")
            last_checkpoint_time = state.time

        # ---- Spectral snapshots during averaging ----
        if state.time >= averaging_start * tau_A and step % snapshot_interval == 0:
            save_spectrum(state, step)

    # ---- Final save ----
    do_checkpoint(state, step, "final")
    save_spectrum(state, step)

    wall_time = _time.time() - wall_start

    # ---- Steady-state check on ε_ν ----
    if len(eps_nu_history) >= 4:
        half = len(eps_nu_history) // 2
        second_half = eps_nu_history[half:]
        valid = [e for e in second_half if np.isfinite(e)]
        if valid:
            mean_eps = sum(valid) / len(valid)
            max_dev = max(abs(e - mean_eps) for e in valid)
            rel_var = max_dev / mean_eps if mean_eps > 0 else float("inf")
        else:
            rel_var = float("inf")
    else:
        rel_var = float("inf")

    # ---- Save time series ----
    np.savez(
        out_dir / "diagnostics_timeseries.npz",
        time=np.array(time_history),
        E_total=np.array(etotal_history),
        eps_nu=np.array(eps_nu_history),
        hermite_energy=np.array(hermite_energy_history),
    )
    volume.commit()

    verdict = "STEADY" if rel_var < 0.3 else "NOT_STEADY"
    print(f"\n{'='*70}")
    print(f"  {label}: {verdict}")
    print(f"  ε_ν var={rel_var:.1%}")
    if eps_nu_history:
        print(f"  ε_ν final={eps_nu_history[-1]:.4e}")
    print(f"  wall_time={wall_time:.0f}s ({wall_time/3600:.1f}h)")
    print(f"{'='*70}")

    return {
        "label": label,
        "nu": float(nu),
        "verdict": verdict,
        "eps_nu_variation": float(rel_var),
        "eps_nu_final": float(eps_nu_history[-1]) if eps_nu_history else float("nan"),
        "e_final": float(etotal_history[-1]) if etotal_history else float("nan"),
        "final_time": float(state.time),
        "wall_time": float(wall_time),
        "n_steps": step,
    }


@app.local_entrypoint()
def main():
    """Submit Hermite calibration runs."""
    print(f"Submitting {len(BRANCHES)} Hermite calibration branches...")
    for b in BRANCHES:
        print(f"  {b['label']}: ν={b['nu']}, hermite_f={b['hermite_amplitude']}")

    futures = []
    for b in BRANCHES:
        futures.append(
            run_hermite_branch.spawn(
                label=b["label"],
                nu=b["nu"],
                hermite_amplitude=b["hermite_amplitude"],
                total_time=b["total_time"],
                averaging_start=b["averaging_start"],
                resume_from=b["resume_from"],
            )
        )

    results = [f.get() for f in futures]

    print(f"\n{'='*80}")
    print("HERMITE CALIBRATION RESULTS")
    print(f"{'='*80}")
    for r in results:
        print(f"  {r['label']}: {r['verdict']}, ε_ν={r['eps_nu_final']:.4e}, "
              f"var={r['eps_nu_variation']:.1%}, t={r['final_time']:.0f}, "
              f"wall={r['wall_time']/3600:.1f}h")
