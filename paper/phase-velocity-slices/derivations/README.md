# Self-checking scripts for the paper

Every equation and number in the paper is checked by a script in this folder. Each script prints PASS or FAIL for each check and exits with status 1 if any check fails. The scripts use GANDALF (the installed `krmhd` package, version 0.7.0) read-only, in float64 on the CPU.

| folder | what it covers | paper sections |
|---|---|---|
| `continuum/` | the two compressive branches, the branch operator and its spectrum, the Case-Van Kampen (slice) transform, the slice equation, the invariants, the finite k_perp rho_i lemmas, the Gauss rule of the ladder matrix, the case 0 < Lambda < 1, and the CMM equation numbers | 2, 3, 4, 7, 8; appendices A, B and C |
| `truncated/` | GANDALF's truncated Hermite hierarchy as M+1 passive slices, the phase velocities, the conserved forms, the Hermite-spectrum dictionary and Kanekar et al.'s forced spectrum, the Hermite flux and its sine-kernel form, the time-step limit of the fastest slices | 2, 5, 7, 8; appendix C |
| `dynamics/` | the frozen-line theorem and its corollaries, the k_z-based split of the Hermite flux, the two-slice law, the absence of a Fjortoft-type constraint, the mirror symmetry, the slices advected by one Elsasser field | 2, 3, 5, 6, 8, 9 |
| `review/continuum/`, `review/truncated/`, `review/dynamics/` | independent checks of the same results by routes that share no code with the first set (quadrature instead of closed forms, GANDALF's own eigensystem, other parameters, pointwise Lagrangian tests) | as above |

Each folder has a README with one line per script, its paper section and its run time, and a `run_all.sh` that runs the folder from the repository root and writes the output of each script to `out/<script>.out`:

```bash
bash paper/phase-velocity-slices/derivations/continuum/run_all.sh
```

The whole set takes about 35 minutes on an Apple M1 Pro. Numerical output that is not text (for example `.npy` tables) goes to `paper/phase-velocity-slices/data/derivations/`, which is git-ignored. `fetch_cmm.py` downloads the LaTeX source of Chandran, Mallet & Meyrand (2026, arXiv:2607.27981) into the git-ignored `data/cmm/` for the two scripts that check CMM's equation numbers; they print SKIPPED if the source cannot be obtained.

Some scripts, or parts of them, check results of an earlier, longer draft that are not in the current paper: hypercollisional reflection (`truncated/t6_reflection.py`, `review/truncated/c09_reflection.py`), IMEX-RK222 time stepping (`truncated/t7b_imex_order.py`, `review/truncated/c07_imex.py`, `review/dynamics/c05_imex.py`, parts of `t7a`, `d03` and `review/truncated/c06_cfl.py`), the hypercollisional damping of slices (`review/truncated/c11_coll_damping.py`), the tilt statistics of the k_z-based split (parts of `dynamics/d06_tilt_kz_decomposition.py` and `review/dynamics/c04_tilt_landau.py`), the parity lemma, pair forcing and fourth-order statistics (`dynamics/d09b_parity_lemma_gandalf.py`, parts of `d09` and `review/dynamics/c06_algebra.py`), the truncated dictionary (part E of `truncated/t4_dictionary.py`) and the Landau-root sum of the density response (part L6 of `continuum/s05_ladder_ops.py`). They are kept because they run in the same suites; the per-folder READMEs mark them "not in the current paper". These scripts are verification checks with fixed test parameters, in the manner of unit tests. The simulations behind the figures of section 8 are config-driven and live in `../configs/`, `../scripts/` and `../analysis/`.
