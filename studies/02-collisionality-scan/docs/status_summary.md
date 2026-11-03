# Study 02 — Closed (2026-11-03)

## Goal

Study 02 set out to show the KRMHD dissipative anomaly: a collisional
dissipation rate ε_ν in Hermite space that stays finite and independent of ν
as ν → 0 in driven, nonlinear turbulence.

## What was done

After the March–April calibration (see `run_log.md` here, `docs/run_log.md`
and `docs/collaboration_report.md`), the production scan ran six values of ν
(1, 3, 5, 10, 20, 50) at 128³ and M = 128 on GANDALF v0.5.0 with the
IMEX-RK222 integrator. Each run lasted 200 τ_A from a common Alfvénic
checkpoint. A Hermite-resolution check at ν = 3 used M = 64 and M = 256. The
time-averaged ε_ν came out the same at every ν, and W(m) followed m^{-1/2}.
A paper draft was written in `paper/dissipative-anomaly/`.

## Outcome

The Alfvénic base state turned out to be weakly turbulent, and the Hermite
sector evolved almost entirely by linear phase mixing. The flat ε_ν and the
m^{-1/2} spectrum are the linear phase-mixing result already derived by Zocco
& Schekochihin (2011, Phys. Plasmas 18, 102309) and Kanekar, Schekochihin,
Dorland & Loureiro (2015, J. Plasma Phys. 81, 305810104). They are not
evidence of a dissipative anomaly in nonlinear turbulence. The forcing on g_0
is white noise, so it fixes the injected power, and in steady state ε_ν equals
that power whatever the value of ν.

## Decision

On 2026-11-03 Anjor decided not to submit the paper and to close the study. A
paper limited to the linear problem would mostly repeat Kanekar et al. (2015).
A rerun of the scan at χ ~ 1 was considered and not taken up.

## What is kept

- `configs/`, `scripts/` and `analysis/` are unchanged. Study 04's `SPEC.md`
  was written from `scripts/modal_128_hermite.py` and from §2 of the paper
  draft, so both stay.
- `paper/dissipative-anomaly/` keeps the draft as a record, with a README
  saying it will not be submitted.
- Data: `studies/02-collisionality-scan/data/` on Anjor's laptop (6.7 GB,
  gitignored) and the Modal volume `krmhd-benchmark-vol`.
- The run records in `run_log.md` and `docs/run_log.md` are left as written.
  Their "dissipative anomaly" notes predate this outcome.

The previous version of this file (status on 2026-03-30) is in git history:
`git show c89540a:studies/02-collisionality-scan/docs/status_summary.md`.
