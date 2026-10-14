# Truncated Hermite hierarchy: self-checking scripts

These scripts check the results on GANDALF's truncated Hermite hierarchy against the installed `krmhd` package (read-only): its exact decomposition into M+1 passive slices, the phase velocities, the conserved forms, the dictionary between slice correlations and the Hermite spectrum, the Hermite flux, hypercollisional reflection at the closure, and what the slices imply for time stepping. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails. CMM is Chandran, Mallet & Meyrand (2026, arXiv:2607.27981v1). Independent checks of the same results are in `../review/truncated/`.

| script | what it checks | paper section | run time |
|---|---|---|---|
| `t1_hierarchy.py` | GANDALF's hierarchy is exactly M+1 passive slices: streaming matrix, symmetrising weight, slice equation on `krmhd_rhs`, no leakage between slices through `gandalf_step` except by hypercollisions | Section 8, Appendix C | 22 s |
| `t2_eigenvalues.py` | the phase velocities u_j: Gauss-Hermite limit, determinant identity and sum rules, spectral-shift law, zero eigenvalue at even M, u_max and its asymptotics, conventions | Section 8, Appendix C | 12 s |
| `t3_invariants.py` | the conserved quadratic forms Q_phi, their identification with CMM's W, Gamma and H, and the helicity at odd and even M | Section 8, Appendix C | 5 s |
| `t4_dictionary.py` | the dictionary between slice correlations and W(m), in the continuum and for the truncated hierarchy | Section 5; Section 8, Appendix C | 11 s |
| `t5_flux_cd.py` | the Hermite flux as Christoffel-Darboux-filtered dephasing of slice pairs, and its sine-kernel limit | Section 5; Section 8, Appendix C | 4 s |
| `t6_reflection.py` | the hypercollisional reflection law R = exp(-X), and the forced steady state, where W_zero/W_open tends to coth X at low m | Section 8, Appendix C | 43 s |
| `t7a_cfl_slices.py` | the explicit advection of slice j by Phi + s u_j Psi, GANDALF's CFL estimate (includes u_j in krmhd >= 0.7.0, gandalf#157; omitted it in <= 0.6.0, where the u_max slices blow up at dt_cfl), and the stability of the explicit stages | Appendix D | 27 s |
| `t7b_imex_order.py` | the coupling of g to z+- in GANDALF's IMEX-RK222 step (second order in krmhd >= 0.7.0, gandalf#158; first order with the t + dt/2 stage time of <= 0.6.0) and the second-order drift of the invariants | Appendix D | 17 s |

`tcore.py` holds the helpers shared by the scripts. Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.6.0, CPU, float64) with other jobs running; the whole folder took 141 s. `t1_hierarchy.py`, `t7a_cfl_slices.py` and `t7b_imex_order.py` integrate in time with `gandalf_step` and list each integration (scheme, grid, M, steps, wall time) in their output.

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/truncated/run_all.sh`, which runs each script with `uv run python`, writes its output to `out/<script>.out`, records the start time and wall time of each script in `out/run_all.out` and exits non-zero if any script fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/truncated/<script>.py`.
