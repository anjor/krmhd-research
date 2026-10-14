# Dynamics: self-checking scripts

These scripts check the dynamical results of the paper: the frozen-line theorem, its corollaries and what breaks it, the k_z-based phase-mixing decomposition under field-line tilt, the two-slice law and the Hermite flux, the absence of a Fjortoft-type constraint, the mirror symmetry, and the slices advected by a single Elsasser field. Most checks use GANDALF's own right-hand side or its time stepper `gandalf_step`. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails.

| script | what it checks | paper section | run time |
|---|---|---|---|
| `d00_sympy_identities.py` | the algebra behind the frozen-line theorem (ideal induction, Jacobi commutator, Cauchy's solution, the frozen-in shear), the two-slice law and the Elsasser slices, in exact arithmetic; part E7 (fourth-order algebra) is not in the current paper | Sections 3, 5 and 6 | 2 s |
| `d01_gandalf_conventions.py` | that GANDALF's induction equation is ideal, its slice equation, the Elsasser form and direction of z+, and slice-energy conservation | Sections 2, 3 and 6 | 2 s |
| `d02_frozen_line.py` | that single-point statistics of nonlinear runs (W(m), slice correlations, the field-aligned flux, fourth moments, quantiles, and the hypercollisional dissipation through its spectrum eps_nu(m)) converge with N to the frozen-line prediction while W(k_perp), W(k_z) and the k_z-only flux do not | Sections 6 and 8 | 119 s |
| `d03_frozen_gandalf_step.py` | the frozen-line prediction through `gandalf_step` (Lawson-RK4; the IMEX-RK222 part, with its second-order time error, concerns the time-stepping appendix of an earlier draft, not in the current paper) | Section 6 | 51 s |
| `d04_forcing_superposition.py` | the corollary for white forcing, in Phi-forced ideal turbulence | Sections 6 and 8 | 105 s |
| `d05_breakers.py` | which added terms break the frozen-line theorem and which do not | Sections 6 and 8 | 153 s |
| `d06_tilt_kz_decomposition.py` | that a split of the Hermite flux by the sign of k_z exchanges phase mixing and unmixing on modes whose k_z and field-aligned wavenumber have opposite signs; the tilt statistics (T3, T4) are not in the current paper | Sections 5 and 9 | 16 s |
| `d07_two_slice_law.py` | the two-slice law with and without the k-sum, the Hermite flux as a Christoffel-Darboux sum of two-slice rates, and the frozen-line form of the slice correlation | Sections 5, 6 and 8 | 5 s |
| `d08_no_fjortoft.py` | that no Fjortoft-type constraint from Gamma, H or any Q_phi acts on the Hermite transfer, Q_{u^2}/Q_1 = m + 1/2, and Gamma_m = 0 in mirror-symmetric states | Section 5 | 1 s |
| `d09_mirror.py` | the mirror map and its action on slices and statistics; the pair-forcing, parity-lemma and fourth-order parts are not in the current paper | Sections 3 and 5 | 7 s |
| `d09b_parity_lemma_gandalf.py` | the parity lemma in GANDALF runs, and the cases it does not cover | not in the current paper | 53 s |
| `d10_elsasser_slices.py` | that the slices at s u = -1 and s u = +1 are each advected by one Elsasser field | Sections 3 and 8 | 4 s |

`dcore.py` holds the GANDALF wrappers and diagnostics shared by the scripts. Numerical output (the results table of d02 and a log of the GANDALF time integrations) goes to `paper/phase-velocity-slices/data/derivations/dynamics/`, which is git-ignored. Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.7.0); the whole folder took 518 s.

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/dynamics/run_all.sh`, which runs each script with `uv run python`, writes its output to `out/<script>.out` and exits non-zero if any script fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/dynamics/<script>.py`.
