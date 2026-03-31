"""Find Alfvénic steady state: scan eta and fampl on Modal.

The Alfvénic cascade must reach steady state (E_total ≈ const) before
we can study the Hermite cascade. Current params (eta=2, fampl=0.005)
show linear energy growth → nonlinear blowup at t~130 τ_A.

Tests Alfvénic cascade ONLY (no Hermite forcing) at various (eta, fampl)
combinations to find parameters that give steady state.
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
import modal

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

app = modal.App("krmhd-alfven-scan")

krmhd_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "jax[cuda12]",
        "gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@v0.4.4",
        "numpy", "h5py", "pyyaml",
    )
)

# (eta, fampl) combinations to test
PARAMS = [
    # Current params (baseline — known to blow up)
    (2.0, 0.005),
    # Higher eta (more dissipation)
    (5.0, 0.005),
    (10.0, 0.005),
    (20.0, 0.005),
    # Lower fampl (less injection)
    (2.0, 0.001),
    (2.0, 0.002),
    # Combined
    (5.0, 0.002),
    (10.0, 0.002),
]


@app.function(image=krmhd_image, gpu="A100", timeout=7200)
def run_alfven_test(eta: float, fampl: float) -> dict:
    """Run Alfvénic cascade test — NO Hermite forcing."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache_krmhd")

    from krmhd.config import SimulationConfig
    from krmhd.diagnostics import compute_energy
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step
    from krmhd.physics import KRMHDState
    from shared.alfven_forcing import AlfvenForcingOptions, apply_alfven_forcing

    config = SimulationConfig(
        name=f"alfven_eta{eta}_fampl{fampl}",
        description="Alfvénic steady state test",
        grid={"Nx": 64, "Ny": 64, "Nz": 32, "Lx": 1.0, "Ly": 1.0, "Lz": 1.0},
        physics={"v_A": 1.0, "eta": eta, "nu": 1.0, "beta_i": 1.0,
                 "Lambda": 2.236, "hyper_r": 2, "hyper_n": 2},
        initial_condition={"type": "random_spectrum", "amplitude": 0.05,
                          "alpha": 1.667, "k_min": 1.0, "k_max": 10.0,
                          "k_wave": [0.0, 0.0, 1.0], "M": 4},
        forcing={"enabled": True, "amplitude": fampl, "k_min": 1.0, "k_max": 2.0},
        time_integration={"n_steps": 50000, "cfl_safety": 0.3, "save_interval": 2500},
        io={"output_dir": "data", "save_spectra": True,
            "save_energy_history": True, "save_fields": False,
            "save_final_state": False, "overwrite": True},
    )

    grid = config.create_grid()
    state = config.create_initial_state(grid)
    physics = config.physics
    ti = config.time_integration
    forcing_cfg = config.forcing
    alfven_forcing_options = AlfvenForcingOptions(
        mode="gandalf_perp_lowkz",
        max_nz=1,
        include_nz0=False,
    )
    rng_key = jax.random.PRNGKey(42)

    etotal_history = []
    time_history = []

    print(f"\n=== eta={eta}, fampl={fampl}, 50k steps, NO Hermite forcing ===")
    print(f"Device: {jax.devices()}")

    wall_start = time.time()

    for step in range(1, ti.n_steps + 1):
        dt = compute_cfl_timestep(state, physics.v_A, ti.cfl_safety)
        state = gandalf_step(
            state, dt, physics.eta, physics.v_A,
            nu=physics.nu, hyper_r=physics.hyper_r, hyper_n=physics.hyper_n,
        )
        if not isinstance(state.time, float):
            state = KRMHDState(
                z_plus=state.z_plus, z_minus=state.z_minus,
                B_parallel=state.B_parallel, g=state.g,
                M=state.M, beta_i=state.beta_i, v_th=state.v_th,
                nu=state.nu, Lambda=state.Lambda,
                time=float(state.time), grid=state.grid,
            )

        if forcing_cfg.enabled:
            rng_key, subkey = jax.random.split(rng_key)
            state, _ = apply_alfven_forcing(
                state,
                forcing_cfg=forcing_cfg,
                dt=dt,
                key=subkey,
                options=alfven_forcing_options,
            )

        if step % ti.save_interval == 0:
            energy = compute_energy(state)
            etotal = float(energy["total"])
            etotal_history.append(etotal)
            time_history.append(float(state.time))
            print(f"  step {step:5d}  t={state.time:.2f}  E_total={etotal:.6f}")

    wall_time = time.time() - wall_start

    # Check steady state: is E_total stable in the second half?
    if len(etotal_history) >= 4:
        half = len(etotal_history) // 2
        second_half = etotal_history[half:]
        mean_e = sum(second_half) / len(second_half)
        max_dev = max(abs(e - mean_e) for e in second_half)
        rel_var = max_dev / mean_e if mean_e > 0 else float('inf')
        steady = rel_var < 0.3  # 30% variation = not steady
    else:
        rel_var = float('inf')
        steady = False

    verdict = "STEADY" if steady else "NOT_STEADY"
    print(f"\n  eta={eta}, fampl={fampl}: {verdict} (var={rel_var:.1%}, "
          f"E_final={etotal_history[-1]:.4f}, wall={wall_time:.0f}s)")

    return {
        "eta": float(eta),
        "fampl": float(fampl),
        "etotal_history": [float(e) for e in etotal_history],
        "time_history": [float(t) for t in time_history],
        "rel_variation": float(rel_var),
        "steady": bool(steady),
        "wall_time": float(wall_time),
    }


@app.local_entrypoint()
def main():
    """Submit all (eta, fampl) tests in parallel."""
    print(f"Submitting {len(PARAMS)} Alfvénic stability tests...")
    for eta, fampl in PARAMS:
        print(f"  eta={eta}, fampl={fampl}")

    results = list(run_alfven_test.starmap(PARAMS))

    print("\n" + "=" * 70)
    print("ALFVÉNIC STEADY STATE SCAN")
    print("=" * 70)
    print(f"{'eta':>6s}  {'fampl':>8s}  {'verdict':>12s}  {'variation':>10s}  {'E_final':>12s}")
    print("-" * 70)
    for r in results:
        print(f"{r['eta']:6.1f}  {r['fampl']:8.3f}  "
              f"{'STEADY' if r['steady'] else 'NOT_STEADY':>12s}  "
              f"{r['rel_variation']:10.1%}  {r['etotal_history'][-1]:12.4f}")
