# Task 01 — Collisional dissipation plateau

## Background

You are given diagnostic data from six driven kinetic reduced MHD (KRMHD)
turbulence simulations that differ only in the collision frequency
ν ∈ {1, 3, 5, 10, 20, 50}. Each simulation evolves the Elsasser fields and
M + 1 = 129 Hermite moments g_m of the ion distribution function on a 128³
grid. Energy is injected by forcing at low wavenumbers and removed by
hyper-collisions acting on the Hermite moments.

The collisional dissipation rate is

    ε_ν(t) = 2 ν Σ_{m=2}^{M} (m/M)^6 W(m, t)

where W(m, t) is the Hermite spectrum (moments m = 0, 1 are exempt from
collisions). Each run was restarted from a common Alfvénic steady state at
t₀ = 2000 τ_A, at which point the Hermite forcing was switched on, and run
for 200 τ_A.

## Data

`runs/nu{1,3,5,10,20,50}.npz`, each containing:

- `nu`, `M`, `hyper_n`, `t0` — run parameters
- `spec_time` (n,) — snapshot times in τ_A
- `W_m` (n, M+1) — Hermite spectrum W(m, t) at each snapshot
- `ts_time`, `ts_E_total`, `ts_hermite_energy` — scalar time series sampled
  every 100 steps: total energy and total Hermite free energy Σ_m W(m, t)
- `k_perp`, `E_kperp` — perpendicular energy spectrum at each snapshot

The spectrum snapshots span the statistically stationary window
t ∈ [2100, 2200] τ_A; the scalar time series covers the full 200 τ_A
including the initial fill-in transient.

## Questions

1. For each run, compute the time-averaged collisional dissipation rate
   ⟨ε_ν⟩ over all spectrum snapshots.
2. For each run, compute the time-averaged total Hermite free energy
   ⟨Σ_m W(m)⟩ from the scalar time series, restricted to t − t₀ > 30 τ_A
   (to exclude the fill-in transient).
3. Does ⟨ε_ν⟩ depend on ν? Report `true` for `is_plateau` if the spread
   (max − min)/mean across the six runs is below 5%, else `false`.

## Answer format

Write `answers/task01.json`:

```json
{
  "eps_nu_mean": {"nu1": 0.0, "nu3": 0.0, "nu5": 0.0, "nu10": 0.0, "nu20": 0.0, "nu50": 0.0},
  "hermite_energy_mean": {"nu1": 0.0, "nu3": 0.0, "nu5": 0.0, "nu10": 0.0, "nu20": 0.0, "nu50": 0.0},
  "is_plateau": true
}
```
