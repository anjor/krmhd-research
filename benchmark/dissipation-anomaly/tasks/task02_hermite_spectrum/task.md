# Task 02 — Hermite spectrum scaling

## Background

Theory makes two competing predictions for the steady-state Hermite spectrum
W(m) of KRMHD turbulence in the inertial range of moment space:

- Linear phase mixing (Zocco & Schekochihin 2011): W(m) ∝ m^{-1/2}
- Stochastic-echo-dominated transfer (Adkins & Schekochihin 2018):
  W(m) ∝ m^{-3/2}

## Data

Same files as Task 01: `runs/nu{1,3,5,10,20,50}.npz`.

## Questions

1. For each run, compute the time-averaged Hermite spectrum ⟨W(m)⟩ over all
   spectrum snapshots, then fit a power law W(m) ∝ m^α by least squares in
   log-log space over the inertial range m ∈ [4, 40] (inclusive). Report the
   exponent α for each run.
2. Which theoretical prediction do the measured exponents match? Report
   `"m^-1/2"` or `"m^-3/2"`.

## Answer format

Write `answers/task02.json`:

```json
{
  "slope": {"nu1": 0.0, "nu3": 0.0, "nu5": 0.0, "nu10": 0.0, "nu20": 0.0, "nu50": 0.0},
  "best_theory": "m^-1/2"
}
```
