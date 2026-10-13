# CLAUDE.md — Instructions for AI-Assisted Research

## Setup

```bash
uv sync
```

This installs all dependencies (GANDALF, JAX, NumPy, etc.) into a local `.venv/`. Use `uv run` to execute scripts, e.g. `uv run python scripts/run.py`.

## Project Context

This repo orchestrates numerical experiments using the GANDALF KRMHD spectral solver. You are helping a plasma physicist run simulations, analyze data, and write papers. The physicist provides physics judgment; you handle implementation.

## Research context

### Study 04 / GANDALF facts
- GANDALF can be changed, through issues or PRs on https://github.com/anjor/gandalf (Critical Rule 1). "Read-only" means never edit the installed package from this repo. It does not mean GANDALF is frozen.
- The plan has no status emails. Ignore any older plan text that says otherwise. Questions for Anjor go in `QUESTIONS.md`.
- The G± hierarchies mix density and δB∥. Never describe them as δB∥-specific.
- Before resuming any study, re-read its current `SPEC.md` and journal (`STATUS.md`, `decisions.md`, newest `log/` entries). Flag any plan assumption that conflicts with them before acting.

## Critical Rules

### 1. GANDALF is read-only
GANDALF is installed as a Python package (`import krmhd` or `from krmhd import ...`). **Never modify GANDALF source code from this repo.** If GANDALF needs changes, file an issue on https://github.com/anjor/gandalf. Study 04's loop changes GANDALF through PRs on the GANDALF repo, under `studies/04-phase-space-helicity/LOOP.md`.

### 2. Physics validation before science
Every simulation must pass these gates before extracting any scientific result:
- **Energy conservation**: relative error < 1% at steady state
- **Energy balance**: injection rate ≈ dissipation rate (within 5%)
- **Steady state**: total energy fluctuations < 10% of mean over final 50 τ_A
- **Spectral sanity**: E(k⊥) shows inertial range (not bottleneck-dominated or under-resolved)

If a gate fails, **stop and report**. Do not proceed to analysis.

### 3. Config-driven runs
All simulation parameters live in YAML config files under `studies/XX/configs/`. No hardcoded physics values in Python scripts. Scripts read configs and produce outputs deterministically.

### 4. Run identification
Every run gets a unique ID: `{study_number}_{parameter_label}_{YYYYMMDD_HHMMSS}`
Example: `01_M032_20260301_143022`

Log every run to `docs/run_log.md` with: ID, config path, hardware, wall time, outcome (pass/fail), notes.

### 5. Idempotent analysis
Analysis scripts read saved diagnostic data and produce figures. They must be re-runnable without re-running simulations. Save diagnostic data in standard formats (HDF5 or NumPy .npz).

## Autonomous workflows

Study 04's loop follows `studies/04-phase-space-helicity/LOOP.md`. Its checkpoint is `STATUS.md`, and where LOOP.md is more specific than the rules below, LOOP.md wins. These rules are the defaults for every other loop or workflow (e.g. paper drafting).

### Long-running loops & workflows
- Keep a checkpoint file (e.g. `.loop/STATE.md`) with: current phase/gate, last completed step, next step, open questions with Anjor's answers (e.g. Q9, Q10), and the exact relaunch command.
- Update the checkpoint after each step, so a usage-limit cutoff can be resumed without reloading scripts.
- Wrap API or critic calls in timeouts of 15 minutes or less, never 2 hours. On timeout, log it and retry or skip; don't hang.
- Keep critic review rounds bounded (max 2 per artifact) so gates don't stall.

### Status reports
When asked for status, answer in this fixed format:
1. Running / stopped
2. Current phase & gate
3. Last completed step
4. Blockers
5. What I need from you
6. Next action

Read it from the checkpoint file (`STATUS.md` for Study 04) instead of reconstructing it from logs.

## Environment

### Environment & permissions
- The shell is zsh. Unquoted variables do NOT word-split, so use arrays or `${=var}` explicitly.
- These actions are blocked by the permission classifier:
  - deleting branches or secrets
  - merging PRs without review
  - changing branch protection/repo settings
  - bulk issue triage
- Don't attempt them. Collect them into a "Needs your action" checklist with exact commands or URLs at the end of the task.
- For stuck git operations, prefer `git am --abort` / `git rebase --abort` over deleting `.git` internals.

## Code Style

- Python 3.10+
- Type hints on all functions
- Docstrings with physics context (what quantity, what units, what equation)
- JAX idioms: prefer `jax.vmap`, `jax.lax.scan` over Python loops
- NumPy for post-processing, JAX for anything touching GANDALF

## Figure Standards

Target: Journal of Plasma Physics
- Single column: 3.4 inches wide
- Double column: 7.0 inches wide
- Font: 10pt, matching LaTeX document
- Use `matplotlib.pyplot` with `text.usetex = True`
- Colormap: viridis for 2D data, qualitative palette for line plots
- Always label axes with physics notation (e.g., $k_\perp \rho_i$, $W(m)$)
- Save as both PDF (for paper) and PNG (for quick inspection)

## GANDALF API Quick Reference

```python
from krmhd.config import SimulationConfig
from krmhd.timestepping import gandalf_step, compute_cfl_timestep
from krmhd.diagnostics import (
    EnergyHistory,            # Energy time series tracker
    compute_energy,           # Dict with 'magnetic', 'kinetic', 'compressive', 'total'
    hermite_moment_energy,    # W(m) spectrum
    hermite_flux,             # Forward/backward Hermite flux
    energy_spectrum_perpendicular,  # E(k_perp)
)
from krmhd.forcing import force_alfven_modes_gandalf
from krmhd.io import save_checkpoint, save_timeseries, load_timeseries

# Load config from YAML
config = SimulationConfig.from_yaml("configs/M032.yaml")

# Create grid and initial state
grid = config.create_grid()
state = config.create_initial_state(grid)

# Time stepping (no single run_simulation — loop with gandalf_step)
dt = compute_cfl_timestep(state, config.physics.v_A)
state = gandalf_step(state, dt, config.physics.eta, config.physics.v_A)
```

## Directory Conventions

- `configs/` — YAML parameter files (committed)
- `scripts/` — Run orchestration and sweeps (committed)
- `analysis/` — Post-processing and plotting (committed)
- `figures/` — Output plots (committed, PDF + PNG)
- `data/` — Simulation outputs (**.gitignored**, stored on Modal volumes or S3)

## Modal Usage

```python
from infrastructure.modal_runner import submit_run, fetch_results

# Submit a single run
run_id = submit_run("studies/01-hermite-convergence/configs/M032.yaml")

# Fetch diagnostic data when complete
data = fetch_results(run_id)
```

## What NOT to do

- Don't modify GANDALF
- Don't hardcode parameters in scripts
- Don't skip validation gates
- Don't start Study N+1 before Study N validation is complete
- Don't chase the helicity barrier — it's out of scope
- Don't over-engineer infrastructure — simple scripts that work > elegant frameworks that don't
- Don't generate synthetic data. Always run simulations.
