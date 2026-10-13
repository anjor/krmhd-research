# Phase-velocity slices

Analytical paper: the compressive sector of kinetic reduced MHD as a continuum of passive scalars, one per parallel phase velocity, and its consequences (invariants, Lagrangian linearity, phase mixing as decorrelation of slices, mirror symmetry, truncated Hermite hierarchies). Target: Journal of Plasma Physics.

Layout:

- `main.tex`, `refs.bib`, `Makefile`: the draft (`make` builds `main.pdf`).
- `derivations/`: self-checking scripts, one per group of equations. Each exits non-zero if a check fails.
- `configs/`: YAML parameters for the GANDALF verification runs.
- `scripts/`: run scripts. Each run gets an ID `P1_<label>_<YYYYMMDD_HHMMSS>` and a row in `docs/run_log.md`.
- `analysis/`: scripts that read saved run data and make the figures.
- `figures/`: PDF and PNG output.
- `data/`: run output (git-ignored).

Run everything from the repository root with `uv run python paper/phase-velocity-slices/<path>`.
