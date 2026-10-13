# Continuum theory: self-checking scripts

These scripts check the continuum theory of the paper: the two compressive branches of KRMHD, the branch operator and its spectrum, the Case-Van Kampen (slice) transform, the slice equation, the invariants and the finite k_perp rho_i lemmas. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails. CMM is Chandran, Mallet & Meyrand (2026, arXiv:2607.27981v1).

| script | what it checks | paper section | run time |
|---|---|---|---|
| `s00_cmm_labels.py` | equation numbers of CMM cited in the paper, recomputed from their LaTeX source | Section 2, Appendix A | <1 s |
| `s01_branches.py` | the two compressive branches, Lambda+-, the moments and the free-energy split (with the (B16) typo), in exact arithmetic | Section 2, Appendix A | 20 s |
| `s02_operator.py` | the branch operator L, the free-energy product, self-adjointness and the Hermite ladder matrix | Section 3, Appendix B | 6 s |
| `s03_dispersion.py` | the dielectric function of one branch, the count of unstable roots, Lambda = 1 and the Landau roots | Section 3, Appendix B | 45 s |
| `s04_cvk_transform.py` | Case-Van Kampen modes, the slice transform, Parseval, completeness and Stone's formula | Section 3, Appendix B | 3 s |
| `s05_ladder_ops.py` | the truncated ladder matrix as the Gauss rule of the spectral measure, and convergence in M | Section 3, Appendix B | 28 s |
| `s06_slice_equation.py` | the slice equation, with own NumPy code and with GANDALF's right-hand side | Section 3, Appendix B | 4 s |
| `s07_invariants.py` | the invariant family Q_phi, CMM's W, Gamma and H, and the local flux law | Section 4 | 2 s |
| `s08_fluid_slices.py` | the RMHD conventions, the fluid slow-wave and entropy slices, and the slices at s u = +-1 | Section 2, Section 7 | 3 s |
| `s09_finite_krho.py` | the advection lemmas at finite k_perp rho_i | Section 4 | <1 s |

`cc_common.py` holds the numerics shared by s02-s06. Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.6.0).

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/continuum/run_all.sh`, which runs each script with `uv run python`, writes its output to `out/<script>.out` and exits non-zero if any script fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/continuum/<script>.py`, except that `s00_cmm_labels.py` reads downloaded LaTeX and is run with `uv run python -I`; it fetches the source into `paper/phase-velocity-slices/data/cmm/` with `../fetch_cmm.py` when needed and prints SKIPPED if it cannot.
