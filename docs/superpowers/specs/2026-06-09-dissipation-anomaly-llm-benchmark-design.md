# Dissipation-anomaly LLM benchmark — design

Date: 2026-06-09
Status: implemented autonomously under a /goal directive; review welcome.

## Purpose

Convert the dissipative-anomaly study (Study 02, `paper/dissipative-anomaly/`)
into a standalone benchmark that measures how well an LLM agent can perform
real plasma-turbulence data analysis and numerical-methods reasoning. The
ground truth comes from eight real GANDALF simulations (six-point ν-scan at
M=128 plus M ∈ {64, 256} checks at ν=3); the paper draft is unpublished, so
the answers are not in any training corpus.

## Approaches considered

1. **Full replication** — the agent re-runs the simulations and reproduces the
   paper. Faithful, but each evaluation needs Modal credentials and GPU-hours.
   Rejected as the core (noted as a possible future "tier 3").
2. **Agentic data analysis** (chosen) — ship sanitized diagnostic data from the
   real runs; the agent writes analysis code and reports numerical answers in a
   fixed JSON schema; a deterministic grader scores them against ground truth.
3. **Q&A only** — physics questions with an answer key. Cheap but weakly
   discriminating and contaminable by general physics knowledge. One reasoning
   task of this kind is kept (the blowup diagnosis), structured rather than
   free-form.

## Layout

Self-contained directory `benchmark/dissipation-anomaly/` with no imports from
the rest of the repo:

```
benchmark/dissipation-anomaly/
  README.md                # what it tests, how to run an agent, how to grade
  tasks/
    task01_dissipation_plateau/task.md
    task02_hermite_spectrum/task.md
    task03_dissipation_integrand/task.md
    task04_M_convergence/task.md
    task05_blowup_diagnosis/task.md
  runs/                    # sanitized npz, one per simulation run (committed)
    nu1.npz ... nu50.npz, M64_nu3.npz, M256_nu3.npz
  grading/
    ground_truth.json      # produced by prepare_data.py, checked against paper
    grade.py               # deterministic grader; no third-party deps beyond numpy
  prepare_data.py          # raw Modal data -> runs/*.npz + ground_truth.json
  reference_solution/      # solves tasks from runs/ only; used to validate grader
```

The directory named `runs/` (not `data/`) because `**/data/` is gitignored.
Raw Modal downloads stay in a local scratch dir and are not committed.

## Data sanitization

From each `diagnostics_timeseries.npz` keep: `times`, `W_m_history`, `nu`,
`M`, `hyper_n`, and the perpendicular spectrum (`k_perp`, `E_kperp`) where
present. Strip `epsilon_nu_history` and any other field that directly encodes
an answer. The agent must compute ε_ν(t) = 2ν Σ_{m≥2} (m/M)^6 W(m,t) itself.

## Tasks and grading

Agents write `answers/taskNN.json` per a schema stated in each task.md.
`grade.py answers/` scores all tasks, prints a per-task and total score, and
exits nonzero below a configurable threshold.

1. **Dissipation plateau** — per-run ε̄_ν over the stated averaging window
   (t − t₀ > 30 τ_A, t₀ = 2000) and a plateau verdict. Score: each value
   within 2% of ground truth; verdict correct.
2. **Hermite spectrum** — log-log slope of ⟨W(m)⟩ over m ∈ [4, 40] per run,
   plus which theoretical scaling (m^{-1/2} vs m^{-3/2}) matches. Tolerance
   ±0.05 absolute on slopes.
3. **Dissipation integrand** — per-run cumulative dissipation from the
   time-averaged spectrum, peak-m of 2ν(m/M)^6 W(m), and the invariance
   statement (spread < a few %, peak moves down as ν rises).
4. **M-convergence** — ε̄_ν at M ∈ {64, 128, 256}, ν = 3; verdict that the
   plateau is truncation-independent (spread ≲ 3%).
5. **Blowup diagnosis** (reasoning) — the Appendix-A evidence (onset times vs
   ν, bounded W(m=M), flat k_z spectrum at m=M, simultaneous all-m exponential
   growth) presented without the conclusion; structured questions distinguish
   "physical pileup at the truncation" from "scheme-level numerical
   instability" and ask which observation rules out pileup. Multiple-choice +
   short structured fields; graded deterministically.

Ground truth is computed by `prepare_data.py` from the raw data with the
paper's exact windows, then cross-checked against Table 1 of `main.tex`
(ε̄_ν = 49.09 ± 0.30; slopes −0.472…−0.496; M-scan 49.11/49.21/50.28). A
mismatch beyond tolerance fails data prep.

## Validation

- Reference solution, written against `runs/` only, must score 100%.
- A deliberately wrong answers set must fail.
- Ground truth must match the paper numbers within stated tolerances.

## Out of scope

- Running new simulations; LLM-judge grading; leaderboard tooling.
- Hiding `grading/` from the agent is the harness operator's job (as with
  test files in SWE-bench); the README states this.
