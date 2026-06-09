# Dissipation-anomaly benchmark for LLM agents

A standalone benchmark that tests whether an LLM agent can carry out real
plasma-turbulence data analysis and numerical-methods reasoning. The data
comes from eight kinetic reduced MHD (KRMHD) turbulence simulations run with
the GANDALF spectral solver: a six-point scan in collision frequency
ν ∈ {1, 3, 5, 10, 20, 50} at Hermite resolution M = 128, plus M = 64 and
M = 256 checks at ν = 3. The underlying scientific result — a dissipative
anomaly in the Hermite cascade — is from an unpublished paper draft, so the
graded answers are not in any model's training data.

## What is being tested

| Task | Skill | Points |
|------|-------|--------|
| 01 dissipation plateau | compute ε_ν(t) from W(m,t), windowed time averages, plateau detection | 13 |
| 02 Hermite spectrum | time-averaged spectra, log-log power-law fits, matching theory to measurement | 7 |
| 03 dissipation integrand | per-m dissipation, invariance of the integral, peak tracking across ν | 14 |
| 04 M-convergence | same analysis across runs with different array shapes and run lengths | 4 |
| 05 blowup diagnosis | discriminating numerical instability from physical pileup; choosing a remedy | 7 |

Tasks 01–04 are agentic data-analysis tasks: the agent must read npz files,
implement the formulas stated in the task, and report numbers. Task 05 is a
structured reasoning task built from the real diagnostic evidence that
preceded the solver fix (an IMEX scheme replacing a Lawson integrator).

## Directory layout

```
tasks/taskNN_*/task.md   task statements given to the agent
runs/*.npz               sanitized simulation diagnostics given to the agent
grading/ground_truth.json + grade.py   WITHHELD from the agent
prepare_data.py          regenerates runs/ + ground truth from raw Modal data
reference_solution/      validates the grader; WITHHELD from the agent
```

## Running an agent

1. Give the agent a working directory containing only `tasks/` and `runs/`,
   with this instruction:

   > Solve the five tasks under `tasks/`, in order. Each `task.md` states the
   > questions and the exact JSON schema for the answer. Write your answers
   > to `answers/task01.json` through `answers/task05.json`. The simulation
   > data is under `runs/`. You may write and run any analysis code you like
   > (Python with numpy is sufficient).

2. Do **not** expose `grading/` or `reference_solution/` to the agent — they
   contain the answers, in the same way that held-out tests do in coding
   benchmarks.

3. Grade:

   ```bash
   python grading/grade.py path/to/agent/workdir/answers
   ```

   Prints a per-task breakdown and a total out of 45; exits 0 if the score
   is at or above the pass threshold (default 90%).

The grader is pure-stdlib Python; the agent environment needs numpy.

## Validating the benchmark

```bash
uv run python reference_solution/solve.py        # solves from runs/ only
python grading/grade.py reference_solution/answers   # must score 45/45
```

Ground truth is computed from the shipped data by `prepare_data.py` and
cross-checked against Table 1 of the paper draft
(`paper/dissipative-anomaly/main.tex`): the spectrum-window means
ε̄_ν = 48.0–48.8 across the ν scan, inertial-range slopes −0.472 to −0.496,
and the M-scan values. Data prep aborts on any disagreement.

## Data provenance and sanitization

Raw inputs are the per-run `diagnostics_timeseries.npz` and
`spectra/spectrum_t*.npz` files on the Modal volume `krmhd-benchmark-vol`
(runs `hermite128_nu{1,3,5,10,20,50}_imex`, `hermite_M{64,256}_nu3_imex`).
`prepare_data.py` strips every field that directly encodes an answer — in
particular the precomputed `eps_nu` scalar carried in each snapshot and the
`eps_nu` time series — so the agent must implement the dissipation formula
itself. To regenerate:

```bash
modal volume get krmhd-benchmark-vol /<run>/diagnostics_timeseries.npz <raw-dir>/<run>/
modal volume get krmhd-benchmark-vol /<run>/spectra <raw-dir>/<run>/
uv run python prepare_data.py --raw-dir <raw-dir>
```

## Scoring notes

- Numerical tolerances: 2% relative on dissipation rates and energies,
  ±0.05 absolute on fitted exponents, ±3 on the integrand peak location.
  These are wide enough to admit any defensible analysis choice (e.g.
  weighted vs unweighted least squares) but reject formula errors such as
  using the wrong hyper-collisional exponent, dropping the factor of 2, or
  fitting outside the stated inertial range. (Dropping the m ≥ 2 exemption
  is numerically harmless here — the (m/M)^6 weights at m = 0, 1 are
  ~10⁻¹³ — so it is deliberately not a graded distinction.)
- Boolean verdicts use explicit thresholds stated in the task text, so they
  test whether the agent computed the right quantity, not its judgment about
  what "close" means.
