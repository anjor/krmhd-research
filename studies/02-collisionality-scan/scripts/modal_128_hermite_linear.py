"""Run 128³ LINEAR Hermite benchmark on Modal — phase mixing only, no z± forcing.

This is the control experiment for the nonlinear Hermite calibration. With no
Alfvén forcing, the z± fields decay and only the Hermite sector evolves via
the parallel streaming term v_th * k_∥. Expected result:
  - W(m) ~ m^{-1/2} (thesis prediction for phase mixing)
  - ε_ν = 2ν Σ_{m≥2} (m/M)^n W_m → const as cascade fills in

Same Lambda=√5, same resolution, same Hermite forcing as the nonlinear run.
Provides a clean comparison: linear phase mixing vs nonlinear turbulent cascade.

Usage:
    uv run modal run studies/02-collisionality-scan/scripts/modal_128_hermite_linear.py
"""
from __future__ import annotations

import modal

app = modal.App("krmhd-hermite-linear-128")

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

BRANCHES = [
    {
        "label": "hermite128_linear_nu0p01",
        "nu": 0.01,
        "hermite_amplitude": 0.0035,
        "total_time": 500,   # 500 τ_A from scratch
        "averaging_start": 400,
    },
]


@app.function(
    image=krmhd_image,
    gpu="A100",
    timeout=36000,  # 10 hours
    volumes={VOL_MOUNT: volume},
)
def run_linear_hermite(
    label: str,
    nu: float,
    hermite_amplitude: float,
    total_time: float,
    averaging_start: float,
) -> dict:
    """Run 128³ LINEAR Hermite benchmark — no Alfvén forcing."""
    import time as _time
    from functools import partial
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
    from krmhd.io import save_checkpoint
    from krmhd.physics import KRMHDState, initialize_random_spectrum
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step

    # ---- Inlined low-k_z Hermite forcing ----
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

    # ---- Output directory ----
    out_dir = Path(VOL_MOUNT) / label
    ckpt_dir = out_dir / "checkpoints"
    spectra_dir = out_dir / "spectra"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    spectra_dir.mkdir(parents=True, exist_ok=True)

    # ---- Physics ----
    resolution = 128
    Lx = Ly = Lz = 1.0
    v_A = 1.0
    beta_i = 1.0
    eta = 100.0       # Same as nonlinear run
    hyper_r = 2
    hyper_n = 6        # Hermite hyper-collisions
    M = 128
    cfl_safety = 0.3
    tau_A = Lz / v_A

    # Hermite forcing params
    n_force_min = 1
    n_force_max = 2
    forced_moments = (0,)

    # Lambda = √5 for β_i = 1
    Lambda = float(jnp.sqrt(5.0))

    # ---- Initial condition: small random Alfvén + Hermite seed ----
    grid = SpectralGrid3D.create(
        Nx=resolution, Ny=resolution, Nz=resolution,
        Lx=Lx, Ly=Ly, Lz=Lz,
    )
    # Start with very small z± (will decay without forcing)
    state = initialize_random_spectrum(
        grid, M=M, alpha=5.0 / 3.0, amplitude=0.01,
        k_min=1.0, k_max=15.0, v_th=1.0, beta_i=beta_i, seed=42,
    )

    # Override Lambda and nu
    state = KRMHDState(
        z_plus=state.z_plus,
        z_minus=state.z_minus,
        B_parallel=state.B_parallel,
        g=state.g,
        M=M,
        beta_i=beta_i,
        v_th=state.v_th,
        nu=nu,
        Lambda=Lambda,
        time=0.0,
        grid=grid,
    )

    # ---- Compute fixed dt ----
    dt = compute_cfl_timestep(state, v_A, cfl_safety)
    total_steps = int(total_time * tau_A / dt)
    checkpoint_interval_time = 10.0
    diagnostic_interval = 100
    snapshot_interval = 500

    print(f"\n{'='*70}")
    print(f"128³ LINEAR Hermite Benchmark: {label}")
    print(f"{'='*70}")
    print(f"  eta={eta}, nu={nu}, hyper_r={hyper_r}, hyper_n={hyper_n}")
    print(f"  M={M}, Lambda={Lambda:.4f}")
    print(f"  Hermite f={hermite_amplitude}, NO Alfvén forcing")
    print(f"  forced_moments={forced_moments}")
    print(f"  dt={dt:.6f}, total_time={total_time} τ_A")
    print(f"  Device: {jax.devices()}")
    print(f"  Output: {out_dir}")

    # ---- Tracking ----
    etotal_history: list[float] = []
    eps_nu_history: list[float] = []
    hermite_energy_history: list[float] = []
    time_history: list[float] = []
    last_checkpoint_time = 0.0
    wall_start = _time.time()

    def compute_eps_nu(E_m: np.ndarray) -> float:
        m_idx = np.arange(len(E_m))
        rates = nu * (m_idx / M) ** hyper_n
        rates[:2] = 0.0
        return float(2.0 * np.sum(rates * E_m))

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

    def do_checkpoint(state: KRMHDState, step: int, desc: str) -> None:
        t = float(state.time)
        path = ckpt_dir / f"checkpoint_t{t:06.1f}.h5"
        metadata = {
            "step": step,
            "eta": eta,
            "nu": nu,
            "hermite_amplitude": hermite_amplitude,
            "hyper_r": hyper_r,
            "hyper_n": hyper_n,
            "v_A": v_A,
            "resolution": resolution,
            "M": M,
            "Lambda": Lambda,
            "label": label,
            "description": desc,
            "linear_benchmark": True,
        }
        save_checkpoint(state, str(path), metadata=metadata, overwrite=True)
        volume.commit()
        print(f"  💾 Checkpoint: {path.name} ({desc})")

    # ---- Main loop: Hermite forcing only, NO Alfvén forcing ----
    rng_key = jax.random.PRNGKey(42)
    step = 0

    while state.time < total_time * tau_A:
        step += 1

        # Apply Hermite forcing ONLY (no Alfvén forcing — linear benchmark)
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

        # ---- Spectral snapshots ----
        if state.time >= averaging_start * tau_A and step % snapshot_interval == 0:
            save_spectrum(state, step)

    # ---- Final save ----
    do_checkpoint(state, step, "final")
    save_spectrum(state, step)

    wall_time = _time.time() - wall_start

    # ---- Steady-state check ----
    if len(eps_nu_history) >= 4:
        half = len(eps_nu_history) // 2
        second_half = eps_nu_history[half:]
        valid = [e for e in second_half if np.isfinite(e) and e > 0]
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
    """Submit linear Hermite benchmark."""
    print(f"Submitting {len(BRANCHES)} linear Hermite benchmark branches...")
    for b in BRANCHES:
        print(f"  {b['label']}: ν={b['nu']}, hermite_f={b['hermite_amplitude']}")

    futures = []
    for b in BRANCHES:
        futures.append(
            run_linear_hermite.spawn(
                label=b["label"],
                nu=b["nu"],
                hermite_amplitude=b["hermite_amplitude"],
                total_time=b["total_time"],
                averaging_start=b["averaging_start"],
            )
        )

    results = [f.get() for f in futures]

    print(f"\n{'='*80}")
    print("LINEAR HERMITE BENCHMARK RESULTS")
    print(f"{'='*80}")
    for r in results:
        print(f"  {r['label']}: {r['verdict']}, ε_ν={r['eps_nu_final']:.4e}, "
              f"var={r['eps_nu_variation']:.1%}, t={r['final_time']:.0f}, "
              f"wall={r['wall_time']/3600:.1f}h")
