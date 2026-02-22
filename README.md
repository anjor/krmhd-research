# KRMHD Research

Research simulations using [GANDALF](https://github.com/anjor/gandalf) for kinetic reduced MHD turbulence.

## Research Problems

### Problem 1: Hermite Convergence Study
**Goal:** Determine M_crit(ν) — minimum Hermite moments for converged turbulence statistics.

- **Status:** Phase 1 (coarse survey) ready to launch
- **Location:** `problem1-hermite-convergence/`

### Problem 2: Collisionality Scan (planned)
Heat flux Q(ν) across collisionality regimes.

### Problem 3: Phase-Space Echo Efficiency (planned)
Echo damping vs imbalance σ_c.

## Structure

```
problem1-hermite-convergence/
├── scripts/     # Execution & analysis scripts
├── configs/     # GANDALF config files
├── results/     # Simulation outputs (checkpoints, diagnostics)
├── analysis/    # Processed data, figures
└── logs/        # Run logs, progress tracking
```

## Commit Policy
Commit early, commit often. Every meaningful step gets versioned.
