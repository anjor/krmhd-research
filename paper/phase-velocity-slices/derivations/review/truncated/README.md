# Truncated Hermite hierarchy: independent review checks

These scripts check the results of `../../truncated/` by independent routes: GANDALF's own eigensystem in place of the symmetrised construction, high-precision quadrature and root-finding in place of eigenvalue solvers, other parameters and configurations, and GANDALF's time stepper where the first checks used a matrix exponential. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails. `cc.py` holds the shared helpers and was written independently of `../../truncated/tcore.py`.

| script | what it checks | paper section | run time |
|---|---|---|---|
| `c00_cmm_numbering.py` | Numbers the equations of CMM's Appendix B from its LaTeX source and checks the (B n) numbers quoted in the paper | Sections 2, 4 and 8 (CMM equation numbers) | < 1 s |
| `c01_conventions.py` | Conventions shared with CMM and GANDALF: streaming matrix as the matrix of L in CMM's basis, Hermite basis, bracket sign, s = sqrt(beta_i), direction of z+, Alfvenic equations | Section 2, Section 8 | 34 s |
| `c02_slices.py` | Slice equation, symmetriser and two-slice law with GANDALF's own eigenvectors at M = 64 and 65; slice leakage through `gandalf_step` | Section 8; Section 5 | 41 s |
| `c03_invariants.py` | Conserved forms by a 60-digit SVD, k-weighted forms, CMM's integrals by quadrature, copy closure, Casimirs, Galerkin matrix of P(1/v), even-M helicity error to M = 2048 | Section 8, Appendix C | 88 s |
| `c04_dictionary.py` | Delta-correlated spectrum by Gauss-Hermite quadrature, Kanekar et al. (2015) (4.10) and (4.15) at finite Lambda, large-m dictionary by 2-D quadrature | Section 5 | 39 s |
| `c05_flux.py` | Flux budget, Christoffel-Darboux identity and slice form of the flux with GANDALF's eigenvectors; sine kernel at M = 1024; pair weight of the flux | Section 5; Section 8 | 11 s |
| `c06_cfl.py` | Time-step limit of the fastest slices on a 3-D random state through `gandalf_step`; IMEX hyper-resistive threshold for r = 1, 2, 4 | Appendix D | 71 s |
| `c07_imex.py` | Stage times of ARS(2,2,2) on a scalar test; first-order g(T) and second-order W drift through `gandalf_step` | Appendix D | 26 s |
| `c08_spectrum.py` | u_max by root-finding, size of its Lambda shift, u_max asymptotics to M = 2000, determinant identity and sum rules to 50 digits, spectral-shift law to M = 8192 | Section 8, Appendix C | 20 s |
| `c09_reflection.py` | Revival ratio through `gandalf_step`, revival time at finite Lambda, forced steady state against two open references | Section 8, Appendix C | 40 s |
| `c11_coll_damping.py` | First-order hypercollisional damping of each slice: the u_max slices are damped most | Section 8, Appendix D | 1 s |
| `c12_zero_slice.py` | Left null vector psi_m(0) at even M for any Lambda; the u = 0 slice is advected by Phi alone and conserves Im H | Section 8 | 9 s |

Run times are wall-clock times of one run of `run_all.sh` on an Apple M1 Pro (CPU only, float64); they vary with the load on the machine.

Run the whole folder from the repository root with `sh paper/phase-velocity-slices/derivations/review/truncated/run_all.sh`, which writes each script's output to `out/<script>.out`, prints one summary line per script and exits non-zero if any script fails. `c00_cmm_numbering.py` reads CMM's LaTeX source, which `paper/phase-velocity-slices/derivations/fetch_cmm.py` downloads once into the git-ignored `paper/phase-velocity-slices/data/cmm/`; it runs with `python -I` and prints SKIPPED if the source cannot be obtained.
