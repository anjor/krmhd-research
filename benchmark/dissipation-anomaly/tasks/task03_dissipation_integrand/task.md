# Task 03 — Self-adjustment of the dissipation range

## Background

If the time-averaged dissipation ⟨ε_ν⟩ is independent of ν (Task 01), the
spectrum must compensate: as ν decreases, the dissipation range retreats to
higher m and W(m) grows in the damped tail so that the dissipation integral
stays fixed. This task examines that compensation directly through the
per-m dissipation integrand

    D(m) = 2 ν (m/M)^6 ⟨W(m)⟩,    m ≥ 2

computed from the time-averaged Hermite spectrum ⟨W(m)⟩.

## Data

Same files as Task 01: `runs/nu{1,3,5,10,20,50}.npz`.

## Questions

1. For each run, compute D(m) from the snapshot-averaged spectrum and report
   the total Σ_{m=2}^{M} D(m).
2. For each run, report the moment number m at which D(m) peaks.
3. Do the totals agree across the six runs to within 5% (max − min)/mean?
   Report as `totals_match_within_5pct`.
4. How does the peak location move as ν increases from 1 to 50? Report
   `"decreases"` if it moves to lower m monotonically (ties allowed),
   else `"other"`.

## Answer format

Write `answers/task03.json`:

```json
{
  "total_dissipation": {"nu1": 0.0, "nu3": 0.0, "nu5": 0.0, "nu10": 0.0, "nu20": 0.0, "nu50": 0.0},
  "peak_m": {"nu1": 0, "nu3": 0, "nu5": 0, "nu10": 0, "nu20": 0, "nu50": 0},
  "totals_match_within_5pct": true,
  "peak_m_trend": "decreases"
}
```
