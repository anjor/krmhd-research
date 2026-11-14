# Phase mixing against a passive-scalar cascade (P1)

Paper P1: the Hermite spectrum of kinetic reduced MHD turbulence. The frozen-line theorem, a two-channel model (phase mixing at the injected field-aligned wavenumber against a uniform perpendicular sink) and its test in forced simulations. Target: Journal of Plasma Physics. The exact decomposition of the compressive sector into passive scalars, one per parallel phase velocity, and its invariants are in the companion paper in `../compressive-slices/`; the self-checking scripts for both papers stay in `derivations/` here.

Layout:

- `main.tex`, `refs.bib`, `Makefile`: the draft (`make` builds `main.pdf`).
- `derivations/`: self-checking scripts, one per group of equations. Each exits non-zero if a check fails.
- `configs/`: YAML parameters for the GANDALF verification runs.
- `scripts/`: run scripts. Each run gets an ID `P1_<label>_<YYYYMMDD_HHMMSS>` and a row in `docs/run_log.md`.
- `analysis/`: scripts that read saved run data and make the figures.
- `figures/`: PDF and PNG output.
- `data/`: run output (git-ignored).

Run everything from the repository root with `uv run python paper/phase-velocity-slices/<path>`.
