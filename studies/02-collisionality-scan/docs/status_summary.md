# Study 02 — Current Status Summary (2026-03-30)

## Goal

Demonstrate the KRMHD **dissipative anomaly**: ε_ν → const as ν → 0.

ε_ν = 2ν Σ_{m≥2} (m/M)^n W_m is the collisional dissipation rate in Hermite
velocity space. The anomaly predicts this is independent of ν at steady state.

## Physics setup

- **Alfvénic cascade** (z± modes): forced externally with `force_alfven_modes_gandalf`,
  fampl=0.005, k_min=1, k_max=2. These evolve independently of g.
- **Hermite cascade** (g_m modes): forced on g₀ with `force_hermite_moments`,
  amplitude=0.0035. Energy cascades from m=0 to higher m via parallel streaming,
  and is dissipated by collisions ν(m/M)^n at m≥2.
- z± advects g via Poisson bracket {Φ, g_m} (mixes but doesn't inject net energy).
- Both cascades must independently reach steady state.

## What we've tried and what happened

### Attempt 1: hyper_n=2, M=32, amplitude=0.15
- **Result**: Blowup at t≈135 τ_A (ν=0.1). eps_nu grows exponentially.
- **Diagnosis**: amplitude too large (ε_inj ≈ 1.1/τ_A, 900× too strong).

### Attempt 2: hyper_n=2, M=32, amplitude=0.005
- **Result**: Also blows up, just slower (eps_nu doubles every 3-4 τ_A).
- **Diagnosis**: Not an amplitude problem. Cascade rate (~14/τ_A) >> damping rate
  (ν=0.01/τ_A) at truncation m=M. Energy reflects off the zero closure at m=M.

### Attempt 3: hyper_n=6, M=128, amplitude=0.0035 (GANDALF benchmark params)
- **GANDALF benchmark** (ν=0.25, Lambda=-1.0, 0.84 τ_A): Appeared to work —
  clean m^{-1/2} spectrum from m=2-20, 98% forward flux. But the run was too
  short to reveal the instability.
- **Study 02 run** (ν=0.01, Lambda=2.236, 94+ τ_A): Blows up identically.
  eps_nu exponential from t≈25 τ_A onward.
- **Diagnosis**: hyper_n=6 doesn't fix the fundamental problem. With GANDALF's
  normalization ν(m/M)^6, the damping at m=M is just ν — still orders of magnitude
  below the cascade rate.

### Attempt 4: ν stability scan (ν = 0.25 to 100, M=32, hyper_n=6)
- **Status**: Submitted to Modal, partial results before crash.
- ν=0.5 confirmed BLOWUP (27× growth in 20k steps).
- Higher ν runs (10, 25, 50, 100) may be more stable but results lost to
  numpy version mismatch in deserialization. Need to rerun.

## The core problem

The Hermite cascade rate at mode m scales as √β_i · k_∥ · √(m/2). At the
truncation m=M, this is roughly 2π·√(M/2) ≈ 25-50 per τ_A (for M=32-128).

The collisional damping at m=M is ν·(M/M)^n = ν, regardless of n.

For steady state, damping must balance cascade flux at or before the truncation.
This requires **ν ≫ 1** (potentially ν > 25-50) — far above our intended scan
range of ν = 0.01 to 0.1.

The zero closure (g_{M+1} = 0) acts as a reflecting boundary. Energy that reaches
m=M bounces back, piling up and growing without bound.

## What's left to try

1. **Very high ν** (ν = 10-100) to confirm stability exists, then find the
   minimum stable ν. The scan is interrupted — needs rerun with fixed deserialization.

2. **Absorbing closure** instead of zero closure at m=M — GANDALF supports
   `closure_symmetric()`. This might absorb energy at the truncation rather than
   reflecting it.

3. **Hermite-only runs** (disable Alfvénic forcing) to check if the {Φ, g_m}
   mixing term is contributing to the instability by concentrating g energy at
   high-k_∥ modes where the cascade is faster.

4. **Much larger M** (M=512 or 1024) to give the spectrum room to decay before
   the truncation. Expensive but might be necessary.

5. **Different hyper_n convention** — the thesis uses νm^n (not ν(m/M)^n). Could
   use standard collisions (hyper_n=1) with higher ν to match thesis behavior.

## Code state

- Branch: `collisionality-scan-benchmark-params`
- GANDALF: pinned to v0.4.4
- All 9 configs: M=128, hyper_n=6, amplitude=0.0035, forced_moments=[0]
- `modal_app.py` and `run_local.py`: hermite forcing is config-driven
- `modal_nu_scan.py`: parallel ν scan script (needs numpy fix for deserialization)

## Key files

- `infrastructure/modal_app.py` — Modal GPU runner
- `studies/02-collisionality-scan/configs/*.yaml` — simulation configs
- `studies/02-collisionality-scan/scripts/modal_nu_scan.py` — parallel ν scan
- `studies/02-collisionality-scan/analysis/dissipation_plateau.py` — analysis/plots
- GANDALF source: `.venv/lib/python3.14/site-packages/krmhd/`
  - `forcing.py:1120-1239` — force_hermite_moments
  - `timestepping.py:549-585` — collision operator
  - `physics.py:876-1127` — g_m RHS equations (includes {Φ, g_m} coupling)
