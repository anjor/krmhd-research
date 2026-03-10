# CLAUDE.md — Instructions for AI-Assisted Research

## Project Context

This repo orchestrates numerical experiments using the GANDALF KRMHD spectral solver. You are helping a plasma physicist run simulations, analyze data, and write papers. The physicist provides physics judgment; you handle implementation.

## Critical Rules

### 1. GANDALF is read-only
GANDALF is installed as a Python package (`import gandalf` or `from gandalf.krmhd import ...`). **Never modify GANDALF source code from this repo.** If GANDALF needs changes, file an issue on https://github.com/anjor/gandalf.

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
from gandalf.krmhd.config import SimulationConfig
from gandalf.krmhd.simulation import run_simulation
from gandalf.krmhd.diagnostics import (
    hermite_moment_energy,    # W(m) spectrum
    hermite_flux,             # Forward/backward Hermite flux
    energy_spectrum,          # E(k_perp)
    total_energy,             # Scalar energy
    check_hermite_convergence # Recurrence check
)

# Load config from YAML
config = SimulationConfig.from_yaml("configs/M032.yaml")

# Run
state, diagnostics = run_simulation(config)
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
