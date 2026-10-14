# Truncated Hermite hierarchy: independent checks

These scripts check the results of `../../truncated/` again by independent routes: GANDALF's own eigensystem in place of the symmetrised construction, high-precision quadrature and root-finding in place of eigenvalue solvers, other parameters and configurations, and GANDALF's time stepper where the first checks used a matrix exponential. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails. `cc.py` holds the shared helpers and was written independently of `../../truncated/tcore.py`. CMM is Chandran, Mallet & Meyrand (2026, arXiv:2607.27981v1).

| script | what it checks | paper section | run time |
|---|---|---|---|
| `c00_cmm_numbering.py` | the numbers of the equations in CMM's Appendix B, from its LaTeX source, against the (B n) numbers quoted in the paper | Sections 2, 4 and 7 (CMM equation numbers) | <1 s |
| `c01_conventions.py` | the conventions shared with CMM and GANDALF: streaming matrix as the matrix of L in CMM's basis, Hermite basis, bracket sign, s = sqrt(beta_i), direction of z+, Alfvenic equations | Sections 2 and 7 | 38 s |
| `c02_slices.py` | the slice equation, the symmetriser and the two-slice law with GANDALF's own eigenvectors at M = 64 and 65, and slice leakage through `gandalf_step` | Sections 5, 7 and 8 | 39 s |
| `c03_invariants.py` | the conserved forms by a 60-digit SVD, k-weighted forms, CMM's integrals by quadrature, the copy closure, Casimirs, the Galerkin matrix of P(1/v), the even-M helicity error to M = 2048 | Sections 7 and 8, Appendix C | 70 s |
| `c04_dictionary.py` | the delta-correlated spectrum by Gauss-Hermite quadrature, Kanekar et al. (2015) (4.10) and (4.15) at finite Lambda, the large-m dictionary by 2-D quadrature | Section 5 | 30 s |
| `c05_flux.py` | the flux budget, the Christoffel-Darboux identity and the slice form of the flux with GANDALF's eigenvectors; the sine kernel at M = 1024; the pair weight of the flux | Sections 5 and 8 | 5 s |
| `c06_cfl.py` | the time-step limit of the fastest slices on a 3-D random state through `gandalf_step`, and the IMEX hyper-resistive threshold for r = 1, 2, 4 | Sections 7 and 8 (the time-step limit of the fastest slices); the IMEX threshold part is not in the current paper | 48 s |
| `c07_imex.py` | the stage times of ARS(2,2,2) on a scalar test, and the second-order g(T) (krmhd >= 0.7.0, gandalf#158; first order in <= 0.6.0) and second-order W drift through `gandalf_step` | not in the current paper | 23 s |
| `c08_spectrum.py` | u_max by root-finding, the size of its Lambda shift, the u_max asymptotics to M = 2000, the determinant identity and sum rules to 50 digits, the spectral-shift law to M = 8192 | Section 7, Appendix C | 18 s |
| `c09_reflection.py` | the revival ratio through `gandalf_step`, the revival time at finite Lambda, and the forced steady state against two open references | not in the current paper | 33 s |
| `c11_coll_damping.py` | the first-order hypercollisional damping of each slice: the u_max slices are damped most | not in the current paper | 1 s |
| `c12_zero_slice.py` | the left null vector psi_m(0) at even M for any Lambda; the u = 0 slice is advected by Phi alone and conserves Im H | Section 7, Appendix C | 6 s |

Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.6.0, CPU, float64) with other jobs running; the whole folder took 311 s. `c01_conventions.py`, `c02_slices.py`, `c06_cfl.py`, `c07_imex.py` and `c09_reflection.py` integrate in time with `gandalf_step` and list each integration (scheme, grid, M, steps, wall time) in their output.

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/review/truncated/run_all.sh`, which runs each script with `uv run python`, writes its output to `out/<script>.out`, records the start time and wall time of each script in `out/run_all.out` and exits non-zero if any script fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/review/truncated/<script>.py`, except that `c00_cmm_numbering.py` reads downloaded LaTeX and is run with `uv run python -I`; it fetches the source into the git-ignored `paper/phase-velocity-slices/data/cmm/` with `../../fetch_cmm.py` when needed and prints SKIPPED if it cannot.
