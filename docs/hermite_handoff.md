# Hermite Cascade Handoff — April 2026

## Summary

The 128^3 Hermite cascade campaign (April 4-6, 2026) achieved a working
**linear phase-mixing benchmark** but revealed that **all nonlinear runs
(with z+/- turbulence) eventually blow up** regardless of collisionality.
This is the key finding: the Alfvenic turbulent background drives the
Hermite cascade faster than (m/M)^6 hyper-dissipation can absorb at M=128.

## What was accomplished

### 1. Lambda=sqrt(5) fix (critical bug)

The Alfvenic checkpoint stores Lambda=1.0, which completely kills the
Hermite cascade (the (1-1/Lambda) coupling factor becomes zero). All
Hermite runs must explicitly override Lambda=sqrt(5) for beta_i=1.

**Files:** `modal_128_hermite.py` line 175, `modal_128_hermite_linear.py` line 147.

### 2. Linear Hermite benchmark (WORKING)

- **Script:** `studies/02-collisionality-scan/scripts/modal_128_hermite_linear.py`
- **Label on volume:** `hermite128_linear_16cube_nu1p0`
- **Parameters:** 16^3 spatial (no k-coupling in linear case), M=128, nu=1.0, Lambda=sqrt(5), hyper_n=6, T4 GPU
- **Result:** 500 tau_A completed in 37 minutes. W(m) spectrum monotonically decreasing. epsilon_nu oscillating around 0.23 (94% variation — noisy but stable).
- **Plot:** `figures/hermite_linear_16cube_results.png`

### 3. Linear nu-scan (quick calibration)

Short 2 tau_A runs at nu=5,10,20,50,100. At this duration, higher nu
truncates the inertial range before it can develop. The nu=1 long run
(500 tau_A) gave the best-developed spectrum.

- **Plot:** `figures/hermite_linear_nu_scan.png`

### 4. Nonlinear Hermite campaign (ALL BLOW UP)

Every nonlinear run eventually blows up. Higher nu delays the blowup
but does not prevent it:

| nu  | Blowup time (tau_A after Hermite start) | Last sane epsilon_nu |
|-----|----------------------------------------|---------------------|
| 1   | ~80                                    | --                  |
| 3   | ~122                                   | 342                 |
| 5   | ~167                                   | 2,350               |
| 10  | ~184                                   | 1,000               |
| 20  | >50 (probe only)                       | 60.5                |
| 50  | >50 (probe only)                       | 53.2                |
| 100 | >50 (probe only)                       | 46.0                |

The 50 tau_A probes for nu=10-100 gave epsilon_nu in the range 46-63,
which looked like a dissipative anomaly signal. But the long runs showed
these were transients — epsilon_nu drifts upward before blowup.

- **Plot:** `figures/hermite_nonlinear_blowup_summary.png`

### 5. Modal volume data inventory

All data lives on `krmhd-benchmark-vol`. Key entries:

**Alfvenic steady state (good):**
- `alfven128_lowkz_f0p02_eta100/` — steady at t=2000, 8% E variation

**Linear Hermite (good):**
- `hermite128_linear_16cube_nu1p0/` — 500 tau_A, stable

**Nonlinear Hermite (all blown up):**
- `hermite128_nu1p0_v3/` — NaN at t~2185
- `hermite128_nu10_long/` — blew up at t~2184
- `hermite128_nu5_long/` — blew up at t~2167
- `hermite128_nu3_long/` — blew up at t~2122
- `hermite128_nu2/` — 50 tau_A probe only
- `hermite128_nu100/`, `hermite128_nu50/`, `hermite128_nu20/`, `hermite128_nu10/` — 50 tau_A probes
- `hermite128_nu0p01/`, `hermite128_nu0p01_v2/` — early failed runs (Lambda=1 or nu too low)

## Root cause analysis

The z+/- turbulent spectrum has power at k_z values beyond the forced
n_z=1. The parallel streaming rate is v_th * k_z * sqrt(m/2) / Lambda.
Even at moderate k_z (say n_z=3-5), the cascade rate at m=M=128 is:

  rate ~ 1 * (2*pi*3) * sqrt(64) / sqrt(5) ~ 67 per tau_A

The hyper-collisional damping rate at m=M is just 2*nu per tau_A (since
(m/M)^6 = 1). Even at nu=100, the damping rate (200/tau_A) exceeds the
n_z=3 cascade rate, but power at higher k_z drives even faster cascades.

The fundamental mismatch: the Alfvenic turbulence populates a broad
range of k_z, and the highest k_z modes drive Hermite cascade rates
that outstrip any finite hyper-dissipation at m=M.

## Options for next steps

1. **Lower M (e.g., M=32 or M=64):** Reduces the maximum cascade rate
   (proportional to sqrt(M)) and increases the dissipation at the
   truncation scale. M=32 was stable in earlier dev-grid tests.

2. **Higher hyper_n:** Steeper dissipation profile. But since blowup is
   at m=M where (m/M)^n = 1, higher hyper_n doesn't help at the
   truncation scale — it only helps the inertial range.

3. **Filter high-k_z from z+/-:** If the Alfvenic spectrum is truncated
   to |n_z| <= 1-2, the cascade rate would be manageable. This is
   artificial but may be justified if the physics of interest is in the
   perpendicular cascade.

4. **Implicit or semi-implicit Hermite dissipation:** Instead of the
   current explicit exponential damping, use an implicit scheme that
   can handle arbitrarily stiff dissipation. This would require GANDALF
   changes.

5. **Accept quasi-steady transient:** The runs ARE stable for 50-100 tau_A
   with meaningful epsilon_nu values. If the physics question only needs
   a quasi-steady window, the existing data may suffice — especially the
   50 tau_A probes at nu=3-100 which showed epsilon_nu ~ 46-63.

6. **Reduce spatial resolution for the Hermite sector:** If z+/- is kept
   at 128^3 but g is on a coarser grid, this would reduce the k_z range
   available for cascade driving. Requires GANDALF changes.

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/modal_128_hermite.py` | Nonlinear Hermite (z+/- + g forcing) |
| `scripts/modal_128_hermite_linear.py` | Linear Hermite benchmark (g forcing only) |
| `scripts/modal_128_benchmark.py` | Alfvenic cascade (z+/- only) |
| `analysis/plot_benchmark_spectra_from_volume.py` | Reusable spectrum plotting |

## Key learnings for future sessions

1. **Always check Lambda when loading checkpoints.** Alfvenic checkpoints have Lambda=1.
2. **50 tau_A probes give false stability signals.** Need 150+ tau_A to catch the nonlinear blowup.
3. **Linear and nonlinear are fundamentally different.** Linear phase mixing is well-behaved; nonlinear has the k_z-driven instability.
4. **16^3 is sufficient for linear Hermite.** No k-coupling in the linear case.
5. **The nu=0.01 thesis value is for much lower M.** At M=128 with nonlinear coupling, nu >> 1 is needed even for transient stability.
