# Truncated Hermite hierarchy: self-checking scripts

These scripts check the results on GANDALF's truncated Hermite hierarchy against the installed `krmhd` package (0.6.0, read-only): its exact decomposition into M+1 passive slices, the phase velocities, the conserved forms, the dictionary between slice correlations and the Hermite spectrum, the Hermite flux, hypercollisional reflection at the closure, and what the slices imply for time stepping. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails. `tcore.py` holds the shared helpers. Independent checks of the same results are in `../review/truncated/`.

| script | what it checks | paper section | run time |
|---|---|---|---|
| `t1_hierarchy.py` | GANDALF's hierarchy is exactly M+1 passive slices: streaming matrix, symmetrising weight, slice equation on `krmhd_rhs`, no slice leakage through `gandalf_step` except by hypercollisions | Section 8, Appendix C | 21 s |
| `t2_eigenvalues.py` | Phase velocities u_j: Gauss-Hermite limit, determinant identity and sum rules, spectral-shift law, zero eigenvalue at even M, u_max and its asymptotics, conventions | Section 8, Appendix C | 12 s |
| `t3_invariants.py` | Conserved quadratic forms Q_phi, identification with CMM's W, Gamma and H, helicity at odd and even M | Section 8, Appendix C | 6 s |
| `t4_dictionary.py` | Dictionary between slice correlations and W(m), in the continuum and for the truncated hierarchy | Section 5; Section 8, Appendix C | 10 s |
| `t5_flux_cd.py` | Hermite flux as Christoffel-Darboux-filtered dephasing of slice pairs, and its sine-kernel limit | Section 5; Section 8, Appendix C | 3 s |
| `t6_reflection.py` | Hypercollisional reflection law R = exp(-X), and the forced steady state, where W_zero/W_open approaches coth X at low m | Section 8, Appendix C | 42 s |
| `t7a_cfl_slices.py` | Explicit advection of slice j by Phi + s u_j Psi, the CFL estimate that omits u_j, and the stability of the explicit stages | Appendix D | 27 s |
| `t7b_imex_order.py` | First-order coupling of g to z+- in GANDALF's IMEX-RK222 step and the second-order drift of the invariants | Appendix D | 18 s |

Run times are wall-clock times of one run of `run_all.sh` on an Apple M1 Pro (CPU only, float64).

Run the whole folder from the repository root with `sh paper/phase-velocity-slices/derivations/truncated/run_all.sh`, which writes each script's output to `out/<script>.out`, prints one summary line per script and exits non-zero if any script fails. A single script runs with `uv run python paper/phase-velocity-slices/derivations/truncated/<script>.py`.
