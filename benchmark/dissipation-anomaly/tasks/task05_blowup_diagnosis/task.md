# Task 05 — Diagnosing a late-time blowup

## Background

Before the simulations in Tasks 01–04 succeeded, every nonlinear Hermite run
attempted with the solver's original time integrator failed with a late-time
blowup. That integrator composed an exact integrating factor
exp(L_stream · dt) for the linear parallel-streaming operator with explicit
Runge–Kutta stages for the nonlinear Poisson-bracket advection; the timestep
was set by the perpendicular-advection CFL with a safety factor of 0.3. The
runs used M = 128 Hermite moments on a 128³ grid with hyper-collisional
damping ν(m/M)^6 and low-k_z forcing.

Two hypotheses were on the table:

- **Pileup**: the cascade in m carries energy to the truncation m = M faster
  than collisions can absorb it; the zero closure g_{M+1} = 0 reflects it,
  and energy accumulates at high m until the run diverges. This is a
  physical failure: the modeled dissipation is too weak for the modeled
  cascade.
- **Numerical instability**: the time-integration scheme is unstable for
  this operator combination, independent of where energy sits in m.

## Evidence

The following observations were collected before any fix was attempted:

- **(A)** W(m = M) remained bounded at the O(1) noise level until the moment
  of blowup, with no gradual accumulation beforehand.
- **(B)** The blowup onset time increased monotonically with collisionality:
  ≈ 80 τ_A at ν = 1, 122 τ_A at ν = 3, 167 τ_A at ν = 5, 184 τ_A at ν = 10.
- **(C)** The parallel-wavenumber spectrum of g_m at m = M sat at the 10⁻¹⁴
  floor throughout the run — no measurable flux was arriving at the
  truncation.
- **(D)** At blowup, all Hermite moments from m = 0 to m = M grew
  exponentially simultaneously, at a rate independent of ν.
- **(E)** A linear benchmark with the same integrator (no Alfvénic forcing,
  streaming and collisions only) ran stably for 500 τ_A and reproduced the
  expected phase-mixing spectrum.

## Questions

**q1.** Which explanation is most consistent with the evidence?

- (A) Physical pileup: cascade energy reflecting off the zero closure at m = M
- (B) A numerical instability of the time-integration scheme, triggered in
  the nonlinear runs
- (C) The forcing amplitude is too large, overdriving the cascade
- (D) The perpendicular grid is under-resolved, causing spectral blocking

**q2.** For each observation A–E, state whether it is *inconsistent* with
the pileup hypothesis (i.e., evidence against pileup). Answer `true`
(inconsistent with pileup) or `false` (consistent with, or neutral toward,
pileup) for each.

**q3.** Which change to the solver should fix the failure while preserving
the physics?

- (A) Increase M so the cascade has more room before the truncation
- (B) Raise ν so collisional damping absorbs the cascade flux at lower m
- (C) Treat the linear streaming and collisional damping terms implicitly,
  keeping the nonlinear advection explicit
- (D) Refine the perpendicular grid to relax the advection CFL limit

## Answer format

Write `answers/task05.json`:

```json
{
  "q1": "A",
  "q2": {"A": false, "B": false, "C": false, "D": false, "E": false},
  "q3": "A"
}
```

(The values above show the format only, not the answers.)
