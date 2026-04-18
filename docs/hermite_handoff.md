# Hermite Cascade Handoff — April 2026

## Current status (2026-04-18)

**Unblocked.** GANDALF v0.5.0 (PR anjor/gandalf#138) ships an IMEX-RK222
Hermite integrator that treats streaming + hyper-collisional damping
implicitly and leaves only Poisson-bracket nonlinearities explicit.
`gandalf_step` default is now `scheme="imex_rk222"`. PR #143 adds
checkpoint scheme metadata.

**Acceptance test passed (ν=3, M=128, 128³, 200 τ_A).** Run
`hermite128_nu3_imex` completed the full 200 τ_A window from the
Alfvénic steady-state checkpoint at t=2000 without NaN or blowup —
the first M=128 nonlinear run to survive past ~122 τ_A. ε_ν settled
into a noisy plateau within ~20 τ_A:

- ε_ν = 49.2 ± 10.7 (mean ± std, after skipping initial 30 τ_A),
  rel std 21.6%.
- E_total drifts upward (26.8k → 40.7k, ~1.5×) — the Hermite sector
  was empty at t=2000 and is still filling in; full energy balance
  not yet reached, but the dissipation rate is already steady.
- Wall time 7.2h on A100.
- Plot: `figures/hermite128_nu3_imex_timeseries.{png,pdf}`.

The ε_ν ≈ 50 plateau is consistent with the old ν=50 and ν=100 short
(50 τ_A) Lawson probes (53 and 46) that happened to survive before
the numerical instability kicked in. That ν-independence across ~30×
in ν is the dissipative-anomaly signature we were after — but it
needs the ν=5, 10 IMEX long runs to confirm.

**Next:** run ν=5 and ν=10 at 200 τ_A (same script, extend BRANCHES)
and plot ε_ν(ν).

## Summary

The 128^3 Hermite cascade campaign (April 4-6, 2026) achieved a working
**linear phase-mixing benchmark** but revealed that **all nonlinear runs
(with z+/- turbulence) eventually blow up** regardless of collisionality.
The original interpretation — Alfvénic turbulence driving the Hermite
cascade faster than (m/M)^6 hyper-dissipation can absorb at M=128 — is
superseded by the numerical-instability diagnosis further down; the
blowup is a scheme-level issue, not physics.

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

**Nonlinear Hermite — IMEX-RK222 (v0.5.0):**
- `hermite128_nu3_imex/` — 200 τ_A completed clean, ε_ν = 49 ± 11 (2026-04-17)

**Nonlinear Hermite — Lawson-RK4 (superseded, all blown up):**
- `hermite128_nu1p0_v3/` — NaN at t~2185
- `hermite128_nu10_long/` — blew up at t~2184
- `hermite128_nu5_long/` — blew up at t~2167
- `hermite128_nu3_long/` — blew up at t~2122
- `hermite128_nu2/` — 50 tau_A probe only
- `hermite128_nu100/`, `hermite128_nu50/`, `hermite128_nu20/`, `hermite128_nu10/` — 50 tau_A probes
- `hermite128_nu0p01/`, `hermite128_nu0p01_v2/` — early failed runs (Lambda=1 or nu too low)

## Root cause analysis

### Original hypothesis (superseded by diagnosis below)

The z+/- turbulent spectrum has power at k_z values beyond the forced
n_z=1. The parallel streaming rate is v_th * k_z * sqrt(m/2) / Lambda.
Even at moderate k_z (say n_z=3-5), the cascade rate at m=M=128 is:

  rate ~ 1 * (2*pi*3) * sqrt(64) / sqrt(5) ~ 67 per tau_A

The hyper-collisional damping rate at m=M is just 2*nu per tau_A (since
(m/M)^6 = 1). Even at nu=100, the damping rate (200/tau_A) exceeds the
n_z=3 cascade rate, but power at higher k_z drives even faster cascades.

The fundamental mismatch [was claimed to be]: the Alfvenic turbulence
populates a broad range of k_z, and the highest k_z modes drive Hermite
cascade rates that outstrip any finite hyper-dissipation at m=M.

### Diagnosis update — April 16, 2026

A direct look at the saved data contradicts the physical-pileup narrative
above. See `analysis/diagnose_hermite_blowup.py` and figures
`hermite_blowup_Wmt.png`, `hermite_blowup_Wm_threshold_comparison.png`,
`hermite_blowup_kz_spectrum_{zpm,g}.png`, `hermite_blowup_eps_nu_vs_t.png`.

Key observations at the ν=3, 5, 10 long runs:

1. **No pileup before blowup.** W(m=M) sits at O(1) noise level (nu=5: 0.5,
   nu=10: 0.3) for the entire 100–180 τ_A before blowup. No monotonic
   accumulation toward the truncation scale.

2. **g has essentially zero power at m=M pre-blowup.** The k_z spectrum of
   g(m=M) at the last pre-blowup checkpoint sits at ~10⁻¹⁴ (numerical
   noise floor), while Σ_m g power is ~10⁻¹. So no cascade flux is
   actually arriving at m=M in the first place.

3. **Blowup is simultaneous across all m.** W(m=M), W(m=M-5), and
   W(m=M-20) all grow at roughly the same rate and reach similar
   amplitudes during the final exponential explosion. Physical pileup
   would be localised at m≈M.

4. **Blowup is ~30 orders in ~20–30 τ_A** — exponential growth with rate
   of order 1–2 per τ_A, independent of ν once active.

5. **z± k_z spectrum is unchanged** from the Alfvénic-only reference at
   every checkpoint. Confirms g is passive as expected.

6. **ν delays onset but does not change the growth rate.** Higher ν pushes
   the blowup time out (122 → 167 → 184 τ_A for ν = 3 → 5 → 10) without
   suppressing the final growth rate. Consistent with ν damping reducing
   a numerical instability's effective growth rate until damping slightly
   exceeds it, shifting onset; growth once excited is set by the scheme,
   not physics.

**Conclusion:** the blowup is a **numerical instability in the Hermite
integrator**, not a physical-cascade pileup. The handoff's
"cascade-rate-exceeds-damping" picture was an incorrect inference from
the ν-dependence of onset times; the actual signatures — bounded W(m=M)
before blowup, m=M numerical noise in g's k_z spectrum, all-m
simultaneous explosion — point to a scheme-level stability problem.

Likely mechanism: the Lawson-RK4 treatment of the Hermite subsystem has
a hidden stability boundary at large m·k_z that the current CFL check
(`krmhd.timestepping.compute_cfl_timestep`) does not enforce. The
comment in that function claims no Hermite streaming CFL is needed
because the integrating factor is exact for linear streaming — which is
true in isolation, but once combined with the nonlinear advection
N(z±, g) inside Lawson-RK4, stability requires dt × max(v_th·k_z·√m/Λ)
to be bounded.

## GANDALF issue to file

**Title:** Numerical instability in Hermite integrator at high M × k_z

**Summary:** The Lawson-RK4 step for the passive g evolution becomes
unstable at 128³ with M=128, producing simultaneous all-m exponential
growth starting at a ν-dependent onset time. See this repo's
`docs/hermite_handoff.md` "Diagnosis update" section and the four
`hermite_blowup_*.png` figures for evidence.

**Reproducer:**
- `studies/02-collisionality-scan/scripts/modal_128_hermite.py` with
  ν ∈ {3, 5, 10}, hyper_n=6, M=128, 128³ grid, Λ=√5, cfl_safety=0.3.

**Candidate fixes** (pick the cleanest — I have no strong preference
on implementation):
1. **Add Hermite streaming term to `compute_cfl_timestep`.** Simplest
   change: enforce dt ≤ C · Λ / (v_th · k_z_max · √M). Costs a ~20×
   timestep reduction at M=128, so may force option 2 or 3 for
   throughput.
2. **Switch Hermite sector to an IMEX integrator.** Treat streaming +
   hyper-collisional damping implicitly; only nonlinear advection by
   z± stays explicit. Removes the stability constraint from linear
   terms.
3. **Implicit hyper-collisional operator alone.** Strong enough ν at m=M
   can damp the numerical mode directly. Less general than (2) but
   smaller change.

**Acceptance test:** ν=3, M=128, 128³, 200 τ_A run completes without
blowup and with ε_ν reaching a statistically steady value.

## What to run once GANDALF ships a fix

After the GANDALF fix lands, repeat the nu-scan (ν=3, 5, 10 for 200 τ_A)
using `modal_128_hermite.py`. If ε_ν saturates to a ν-independent value
at sufficiently small ν, that's the dissipative-anomaly signature we
were originally looking for. The existing Alfvénic steady state
checkpoint and linear Hermite benchmark remain valid baselines.

## Options for next steps (historical — see GANDALF issue above)

These options were listed before the numerical-instability diagnosis.
Retained for context; most are now moot.

1. ~~**Lower M (e.g., M=32 or M=64):**~~ Not scientifically valid —
   M is the phase-space resolution whose cascade we are studying.
   Reducing M does not diagnose or fix the underlying scheme issue.

2. **Higher hyper_n:** Doesn't help in any interpretation (already noted).

3. **Filter high-k_z from z+/-:** Still artificial. No longer needed —
   the k_z spectrum of z± is not the cause of blowup.

4. **Implicit or semi-implicit Hermite dissipation:** Still viable — see
   candidate fix (2) or (3) in the GANDALF issue above.

5. **Accept quasi-steady transient:** Rejected — the 50 τ_A ε_ν values
   are transients of a numerically unstable run, not physically
   meaningful steady states.

6. **Reduce spatial resolution for the Hermite sector:** Not needed —
   high z± k_z is not driving a physical cascade. Would mask, not fix,
   the scheme issue.

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
3. **Linear and nonlinear are fundamentally different.** Linear phase mixing is well-behaved; nonlinear has a numerical-scheme instability at high m·k_z.
4. **16^3 is sufficient for linear Hermite.** No k-coupling in the linear case.
5. **The nu=0.01 thesis value is for much lower M.** At M=128 the high ν required to delay the numerical blowup is not a physics statement; it's a scheme crutch.
6. **Look at the W(m,t) pcolormesh before trusting a "pileup" story.** A physical cascade pileup localises at m≈M and grows monotonically; simultaneous all-m exponential explosion is a numerical signature.
