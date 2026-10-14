# Continuum theory: self-checking scripts

These scripts check the continuum theory of the paper: the two compressive branches of KRMHD, the branch operator and its spectrum, the Case-Van Kampen (slice) transform, the slice equation, the invariants, the finite k_perp rho_i lemmas, the Gauss rule of the truncated ladder matrix, and the case 0 < Lambda < 1. Each script prints PASS or FAIL for every check and exits with status 1 if any check fails. CMM is Chandran, Mallet & Meyrand (2026, arXiv:2607.27981v1).

| script | what it checks | paper section | run time |
|---|---|---|---|
| `s00_cmm_labels.py` | equation numbers of CMM cited in the paper, recomputed from their LaTeX source and compared with the numbers the paper quotes | Sections 2, 3, 4 and 7, Appendix A | <1 s |
| `s01_branches.py` | the two compressive branches, Lambda+-, the moments and the free-energy split (with the (B16) typo), in exact arithmetic | Section 2, Appendix A | 20 s |
| `s02_operator.py` | the branch operator L, the free-energy product, L = UP, self-adjointness and the Hermite ladder matrix | Sections 2 and 3, Appendix B | 6 s |
| `s03_dispersion.py` | the dielectric function of one branch, the count of unstable roots, Lambda = 1, the Landau roots (with the root at Lambda = 1 + delta to second order) and the Vlasov-Poisson correspondence | Sections 2 and 3, Appendix B | 43 s |
| `s04_cvk_transform.py` | Case-Van Kampen modes, the slice transform, Parseval, completeness, Stone's formula, the spectral density and the slice at u = 0 | Sections 3 and 4, Appendix B | 3 s |
| `s05_ladder_ops.py` | the truncated ladder matrix as the Gauss rule of the spectral measure, and convergence in M | Sections 7 and 8, Appendix C | 28 s |
| `s06_slice_equation.py` | the slice equation, with own NumPy code and with GANDALF's right-hand side, and the coupling of slices by velocity-dependent collisions | Section 3 | 4 s |
| `s07_invariants.py` | the invariant family Q_phi, CMM's W, Gamma and H, and the local flux law | Section 4 (also Section 5) | 2 s |
| `s08_fluid_slices.py` | the RMHD conventions, the fluid slow-wave and entropy slices, and the slices at s u = +-1 | Sections 2 and 3 | 3 s |
| `s09_finite_krho.py` | the advection lemmas at finite k_perp rho_i | Section 4 | <1 s |
| `s10_unstable_pair.py` | the case 0 < Lambda < 1: the expansion over the Case-Van Kampen modes and the two discrete modes F_+-, the coefficient alpha_+ = -pi res_G = [F_+,G]/[F_+,F_+], the orthogonality of F_+- and the norm [F_+,F_+] | Appendix B | 1 s |

`cc_common.py` holds the numerics shared by s02-s06 and s10. Run times are wall times of one run on an Apple M1 Pro (Python 3.14, krmhd 0.6.0).

Run the folder from the repository root with `bash paper/phase-velocity-slices/derivations/continuum/run_all.sh`, which runs each script `s??_*.py` with `uv run python`, writes its output to `out/<script>.out` and exits non-zero if any script fails. A single script runs as `uv run python paper/phase-velocity-slices/derivations/continuum/<script>.py`, except that `s00_cmm_labels.py` reads downloaded LaTeX and is run with `uv run python -I`; it fetches the source into `paper/phase-velocity-slices/data/cmm/` with `../fetch_cmm.py` when needed and prints SKIPPED if it cannot.

Part L6 of `s05_ladder_ops.py` (the density response as a sum over Landau roots) checks material that is not in the current paper.
