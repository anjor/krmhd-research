# Continuum theory: independent checks

These scripts check the main continuum results again by routes that share no code with the scripts in `../../continuum`: quadrature instead of closed forms, a Volterra equation instead of the spectral formulas, exact arithmetic, and GANDALF's own right-hand side and time stepper. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails. CMM is Chandran, Mallet & Meyrand (2026, arXiv:2607.27981v1).

| script | what it checks | paper section | run time |
|---|---|---|---|
| `c01_gandalf_conventions.py` | GANDALF's right-hand side against an independent evaluation of the KRMHD equations, its slices, and the direction of z+ under `gandalf_step` | Section 2, Section 3 | 7 s |
| `c02_branches.py` | the two branches, Lambda+-, the free-energy and H splits and the moments, by velocity quadrature | Section 2, Appendix A | 1 s |
| `c03_spectral_volterra.py` | the spectral theorem through time evolution, Landau decay, the H kernel and the inverse transform | Section 3, Appendix B | 16 s |
| `c04_cmm_invariants_flux.py` | CMM's W, Gamma and H as members of Q_phi, their Hermite forms and the local flux law | Section 4 | 5 s |
| `c05_lemmas_finite_krho.py` | the finite k_perp rho_i advection lemmas and the role of the self-consistent field | Section 4 | 3 s |
| `c06_zero_eigenvalue.py` | the zero eigenvalue of the ladder matrix at even M and the u = 0 slice in GANDALF | Section 4, Section 8 | 23 s |
| `c07_fluid_and_elsasser_slices.py` | the fluid slow-wave and entropy slices, and the slices at s u = +-1 in GANDALF | Section 7 | 3 s |

Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.6.0).

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/review/continuum/run_all.sh`, which runs each script with `uv run python`, writes its output to `out/<script>.out` and exits non-zero if any script fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/review/continuum/<script>.py`.

One further check is not included. It compared the equation tags printed in the compiled arXiv PDF of CMM with the numbers used in the paper, and it relied on a text extraction of the PDF by poppler's `pdftotext -layout`, an external tool whose layout output cannot be reproduced cleanly from the project environment. The same equation numbers are checked from the LaTeX source by `../../continuum/s00_cmm_labels.py`.
