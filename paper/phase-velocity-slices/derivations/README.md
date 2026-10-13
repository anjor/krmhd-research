# Self-checking scripts for the paper

Every equation and number in the paper is checked by a script in this folder. Each script prints PASS or FAIL for each check and exits with status 1 if any check fails. The scripts use GANDALF (the installed `krmhd` package, version 0.6.0) read-only, in float64 on the CPU.

| folder | what it covers | paper sections |
|---|---|---|
| `continuum/` | the two compressive branches, the branch operator and its spectrum, the Case-Van Kampen (slice) transform, the slice equation, the invariants, the finite k_perp rho_i lemmas | 2, 3, 4, 7; appendices A and B |
| `truncated/` | GANDALF's truncated Hermite hierarchy as M+1 passive slices, the phase velocities, the conserved forms, the Hermite-spectrum dictionary, the Hermite flux, hypercollisional reflection, time stepping | 5, 8; appendices C and D |
| `dynamics/` | the frozen-line theorem and its corollaries, field-line tilt, the two-slice law, the absence of a Fjortoft-type constraint, the mirror symmetry and parity lemma, the slices advected by one Elsasser field | 5, 6, 7; appendices D and E |
| `review/continuum/`, `review/truncated/`, `review/dynamics/` | independent checks of the same results by routes that share no code with the first set (quadrature instead of closed forms, GANDALF's own eigensystem, other parameters, pointwise Lagrangian tests) | as above |

Each folder has a README with one line per script, its paper section and its run time, and a `run_all.sh` that runs the folder from the repository root and writes the output of each script to `out/<script>.out`:

```bash
bash paper/phase-velocity-slices/derivations/continuum/run_all.sh
```

The whole set takes about 30 minutes on an Apple M1 Pro. Numerical output that is not text (for example `.npy` tables) goes to `paper/phase-velocity-slices/data/derivations/`, which is git-ignored. `fetch_cmm.py` downloads the LaTeX source of Chandran, Mallet & Meyrand (2026, arXiv:2607.27981) into the git-ignored `data/cmm/` for the two scripts that check CMM's equation numbers; they print SKIPPED if the source cannot be obtained.

These scripts are verification checks with fixed test parameters, in the manner of unit tests. The simulations behind the figures of section 9 are config-driven and live in `../configs/`, `../scripts/` and `../analysis/`.
