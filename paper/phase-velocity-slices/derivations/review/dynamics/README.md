# Dynamics: independent checks

These scripts check the main dynamical results again by routes that share no code with the scripts in `../../dynamics`: a pointwise Lagrangian test of the frozen-line theorem, a forced test in which the g forcing is correlated with the Alfvenic forcing at the same instant, direct 2D quadrature for the tilt results, a symbolic and numerical analysis of GANDALF's IMEX-RK222 step, and algebraic checks of the two-slice law, the Hermite flux and the slices at s u = +-1. The one exception is `c05_imex.py`, which rebuilds the initial fields of d03 with `../../dynamics/dcore.py` in order to continue that run. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails.

| script (arguments) | what it checks | paper section | run time |
|---|---|---|---|
| `c01_lagrangian_pointwise.py` | the frozen-line theorem and Cauchy's solution pointwise along Lagrangian markers, converging with N = 16 to 48 | Section 6 | 37 s |
| `c01_lagrangian_pointwise.py 32,48,64 1.5 0.25,0.5 strong` | the same test in a strong flow that reaches the grid scale (convergence printed, not asserted; output in `out/c01_lagrangian_pointwise_strong.out`) | Section 6 | 167 s |
| `c02_forcing_same_instant.py 16,24,32,48` | the corollary for white forcing when the g injections and O(1) Phi kicks are correlated at the same instant, with an anticipating control | Section 6 | 198 s |
| `c04_tilt_landau.py` | the tilt results by direct 2D quadrature, and the sign of the field-aligned flux from g_0 at M = 5000 | Section 5 (k_z-based versus field-aligned split); the tilt statistics of an earlier draft's appendix are not in the current paper | 18 s |
| `c05_imex.py` | the stability function and order of GANDALF's IMEX-RK222 step for the slice advection, and the blow-up of the d03 run at dt = 0.002 | not in the current paper (its runs use Lawson-RK4) | 34 s |
| `c06_algebra.py` | the Christoffel-Darboux identity, the normalisation of hermite_flux, the fourth-order statement (not in the current paper), the absence of a Fjortoft-type constraint, the range of Lambda and the s u = -1 slice | Sections 3, 5, 6 and 8 | 4 s |

`c00_core.py` holds the helpers shared by these scripts. A log of the GANDALF time integrations goes to `paper/phase-velocity-slices/data/derivations/review/dynamics/`, which is git-ignored. There is no c03: that check concerns a question outside the scope of this paper. Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.7.0); the whole folder took 458 s.

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/review/dynamics/run_all.sh`, which runs each script with `uv run python` (with the arguments in the table), writes its output to `out/<script>.out` and exits non-zero if any run fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/review/dynamics/<script>.py [arguments]`.
