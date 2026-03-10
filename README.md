# krmhd-research

Numerical experiments on phase-space cascades in kinetic plasma turbulence, using the [GANDALF](https://github.com/anjor/gandalf) KRMHD spectral solver.

## Studies

1. **Hermite convergence in nonlinear driven turbulence** — How many velocity-space moments are needed? What is the Hermite spectrum scaling?
2. **Collisionality scan** — Verifying the dissipative anomaly (heating rate independent of collision frequency)
3. **Echo efficiency in imbalanced turbulence** — How does cross-helicity suppress plasma echo?

See [RESEARCH_PLAN.md](RESEARCH_PLAN.md) for full details.

## Setup

```bash
# Clone
git clone https://github.com/anjor/krmhd-research.git
cd krmhd-research

# Install GANDALF
pip install -e "git+https://github.com/anjor/gandalf.git#egg=gandalf"

# Install analysis dependencies
pip install -r requirements.txt
```

## Running

```bash
# Local (M1 Pro) — single run
python studies/01-hermite-convergence/scripts/run_local.py configs/M016.yaml

# Modal (cloud GPU) — parameter sweep
python studies/01-hermite-convergence/scripts/sweep_M.py
```

## Project Structure

```
studies/           # One directory per study, each with configs/scripts/analysis/figures
shared/            # Reusable diagnostics, plotting, validation
infrastructure/    # Modal integration
paper/             # LaTeX drafts
docs/              # Physics notes, run log
```
