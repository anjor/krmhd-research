# Dynamics: self-checking scripts

These scripts check the dynamical results of the paper: the frozen-line theorem, its corollaries and what breaks it, the k_z-based phase-mixing decomposition under field-line tilt, the two-slice law and the Hermite flux, the absence of a Fjortoft-type constraint, the mirror symmetry and the parity lemma, and the slices advected by a single Elsasser field. Most checks use GANDALF's own right-hand side or its time stepper `gandalf_step`. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails.

| script | what it checks | paper section | run time |
|---|---|---|---|
| `d00_sympy_identities.py` | the algebra behind the frozen-line theorem, the parity lemma, the two-slice law and the Elsasser slices, in exact arithmetic | Section 6 (also Sections 5 and 7) | 2 s |
| `d01_gandalf_conventions.py` | that GANDALF's induction equation is ideal, its slice equation, the Elsasser form and direction of z+, and slice-energy conservation | Section 6 (also Section 7) | 2 s |
| `d02_frozen_line.py` | that single-point statistics of nonlinear runs converge with N to the frozen-line prediction while W(k_perp), W(k_z) and the k_z-only flux do not | Section 6 | 143 s |
| `d03_frozen_gandalf_step.py` | the frozen-line prediction through `gandalf_step` (Lawson-RK4 and IMEX-RK222) | Section 6, Appendix D | 70 s |
| `d04_forcing_superposition.py` | the corollary for white forcing, in Phi-forced ideal turbulence | Section 6 | 120 s |
| `d05_breakers.py` | which added terms break the frozen-line theorem and which do not | Section 6 | 171 s |
| `d06_tilt_kz_decomposition.py` | the k_z-based phase-mixing decomposition under a uniform field-line tilt | Section 5, Appendix E | 17 s |
| `d07_two_slice_law.py` | the two-slice law with and without the k-sum, and the Hermite flux as a Christoffel-Darboux sum of two-slice rates | Section 5 | 5 s |
| `d08_no_fjortoft.py` | that no Fjortoft-type constraint from Gamma, H or any Q_phi acts on the Hermite transfer | Section 5 | 2 s |
| `d09_mirror.py` | the mirror map, its action on slices and statistics, pair forcing, the parity-lemma algebra and the fourth-order statement | Section 7 | 9 s |
| `d09b_parity_lemma_gandalf.py` | the parity lemma in GANDALF runs, and the cases it does not cover | Section 7 | 62 s |
| `d10_elsasser_slices.py` | that the slices at s u = -1 and s u = +1 are each advected by one Elsasser field | Section 7 | 4 s |

`dcore.py` holds the GANDALF wrappers and diagnostics shared by the scripts. Numerical output (the results table of d02 and a log of the GANDALF time integrations) goes to `paper/phase-velocity-slices/data/derivations/dynamics/`, which is git-ignored. Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.6.0).

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/dynamics/run_all.sh`, which runs each script with `uv run python`, writes its output to `out/<script>.out` and exits non-zero if any script fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/dynamics/<script>.py`.
