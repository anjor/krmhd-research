# Task 04 — Hermite-resolution convergence

## Background

A ν-independent dissipation rate at fixed Hermite truncation M leaves open
whether the value is set by the cascade or by the truncation itself. To
check, the ν = 3 run was repeated at M = 64 and M = 256 (the latter for
100 τ_A instead of 200 τ_A).

## Data

`runs/M64_nu3.npz`, `runs/nu3.npz`, `runs/M256_nu3.npz` — same format as
Task 01. Note that `M` differs per file, and the M = 256 run has fewer
snapshots.

## Questions

1. Compute the snapshot-averaged ⟨ε_ν⟩ for each of the three runs, using
   the formula from Task 01 with each run's own M.
2. Is the dissipation rate independent of the truncation? Report `true`
   for `truncation_independent` if the spread (max − min)/mean across the
   three values is below 7.5%, else `false`.

## Answer format

Write `answers/task04.json`:

```json
{
  "eps_nu_mean": {"M64_nu3": 0.0, "nu3": 0.0, "M256_nu3": 0.0},
  "truncation_independent": true
}
```
