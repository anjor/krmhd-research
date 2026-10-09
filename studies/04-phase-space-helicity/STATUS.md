# Study 04 — status

Front page of the autonomous loop. `LOOP.md` is the protocol. The log is in `log/`.

## Next block

**Hard stop (it-20261009-2146): waits for Anjor's answer to QUESTIONS.md Q9.** Gate 1 criteria part (a), the report machinery, was refuted for the third time since the approach changed: `critic_it-20261005-2234_1.md`, `critic_it-20261009-2116_1.md` and `critic_it-20261009-2146_1.md` (claim C15). LOOP.md §4 then makes it a hard stop.
What remains after Anjor's answer, for options (a) and (b) of Q9, against the code at a64d2e2:
1. Run git with every `GIT_*` variable removed (and `-c core.fsmonitor=false`) in `git_head`, `tree_is_clean`, `main` and `report_history_problems`, and check that `rev-parse --show-toplevel` and `--git-dir` are the repo's.
2. Make the writer refuse any head line that the runner's case-insensitive readers (guards.py `RESULT_PASS_RE`, `CRITIC_LABEL_RE`, `claims_pass`) would take as a verdict or Critic line, including through Unicode case folding.
3. State these limits in the docstrings: a stale `.pyc` in `analysis/__pycache__/`; the scripts' shared scratch folder on `sys.path`; JAX and XLA variables that `-E` keeps; `finish` does not check the gate code against the named commit; a forged saved critic file is caught only by the runner; and the shared, ignored scratch outputs that the re-judge reads.
4. Tests: the git environment, the runner's case-folding reading, the history refusal and `gate1_rejudge` through `finish_report`, and `main finish` reaching `REJUDGE`.
Questions 2 to 6 of the review held up. If part (a) passes: part (b), then (c), then the claim reviews, then the Gate 1 evaluation.
The changed approach (it-20261005-1446) stands:
1. Split the review into three narrower critic reviews, each needing SUPPORTED before the next: (a) the shared report machinery (now `analysis/gate_report.py`, plus the environment checks and the command line in `gates.py`); (b) the script rules and D03; (c) the claim rules (`judge_claim`, `GATE1_CLAIMS`). Name Gate 1 in each request.
2. Take the numeric `as2018` thresholds out of the coded check: keep its exit code and section 1 (algebraic antisymmetry). The instability goes into a successor of C16 (C19) judged by its own claim review. C19 must say that `critic_as2018.py` tests the plain Λ → ∞ ∂_v, and that D02's operator is J = A_low − A_up with the (1, 0) entry c_Λ, antisymmetric only with the weight P (check D02's own check of that). The dt comparison in `critic_as2018.py` also changes run length and fit window (T = 3.0 vs 1.2), so it does not isolate dt; say so in C19 or fix the script.
3. Mark in SPEC.md §1–§3 and §6 which statements D01–D03 or a claim check and which are accepted as written. Open from the third review: bracket sign and Φ/Ψ definitions (C14 depends on them), k = 0 zeroing, z± stepper, c_Λ values, η_z factor, order of the resistive step, (N−1)//3 vs N//3 (D03 uses N = 16, where they agree), the §6 k = 0 exclusion, the §2 profile numbers against the npz, a D2 negative control in D03. Extend D03 where cheap (an N where (N−1)//3 ≠ N//3; a krmhd_rhs check of Φ/Ψ and the bracket sign; k = 0 zeroing).
4. C15 cannot be in `GATE1_CLAIMS` (claim-review titles may not name a gate); the `decisions.md` row of a SUPPORTED criteria review is its record. C10's falsifier cannot test its content (the k-summed Γ is zero by symmetry); supersede C10 or record why not.
After the criteria are SUPPORTED: claim reviews for C2, C4, C6, C10, C14, C17, C18 and C19, then evaluate Gate 1.

## Queue

1. Phase 1 (PLAN.md §4): `analysis/helicity.py`, `analysis/budget.py`, the local conservation tests under both closures, the sign test, then the ν-scan checkpoint analysis. The v0.5.0 checkpoints test the pipeline only (decision 5). The checkpoint analysis waits for QUESTIONS.md Q8 (the Study 2 data link is outside the session's readable folders).
2. The H_ph-sp derivation (decision 4): the matrix of 1/v∥ in the Hermite basis, conservation of H_ph-sp in the GANDALF hierarchy and its truncation term at m = M, as a self-checking script in `derivations/` with a critic pass. Then the post-processing diagnostic, labelled exploratory.
3. The new base state (decision 5): regenerate the Alfvénic base state on the pinned GANDALF, recalibrate the forcing, then the ν = 3 Hermite branch that set A starts from. Configs, run code with resume from checkpoint, a local smoke test, a `RUNS.md` row, then a launch through `loop/modal_launch.py`.

## In flight

Nothing. No Modal runs, no GANDALF PRs.

## Compute

Compute cap 200.0 A100-h: used 0.00, reserved 0.00, left 200.00 (launches 1, runs in flight 0)

## Iterations, newest first

- it-20261009-2146 — hard stop: Gate 1 criteria part (a), third submission. Self-probe fixes first (a64d2e2: symlinks refused at finish, an install location check, Gate 1 re-judge rebuilds every head field except the commit and the time). Critic REFUTED (`critic_it-20261009-2146_1.md`: git runs with the inherited environment; the runner's case-folding reader of heads). That is three failed entries on the changed approach, so STOP was written and Q9 asked.
- it-20261009-2116 — WIP: Gate 1 criteria part (a), second submission. Recovered the stash of it-20261005-2234 and handled Anjor's answer to the WIP-streak stop. Fixed the findings of `critic_it-20261005-2234_1.md`. Critic REFUTED again (`critic_it-20261009-2116_1.md`: a consistently edited head still finished as PASS). Fixed after the review: re-judge from saved outputs, head hash binding, Frozen lines, the runner's gate names, isolated install check, and the machinery split into `gate_report.py`. Not yet reviewed.
- it-20261006-1454 — stub (Anjor ended the session); the runner wrote STOP for the WIP streak, and Anjor resumed the loop on 2026-10-09.
- it-20261005-2234 — stub (Anjor ended the session after 14.5 min): Gate 1 criteria part (a), first submission; critic REFUTED (`critic_it-20261005-2234_1.md`); its fix was recovered from its stash by it-20261009-2116.
- it-20261005-1519 — stub (Anjor ended the session after 0.8 min).
- it-20261005-1446 — WIP: Gate 1 criteria, third try: C11 → C16, C1 → C17, C18 added, D03 checks SPEC §1, §3, §6 against GANDALF; critic REFUTED (`critic_it-20261005-1446_1.md`: review not bound to the evaluation, C16 wrong about D02's operator, dt rule). Cheap findings fixed; the next try changes approach.
- it-20261005-1416 — WIP: Gate 1 criteria revised after the REFUTED review of the dead iteration; the critic's second review is also REFUTED (`critic_it-20261005-1416_1.md`). Cheap findings fixed; five remain (Next block). C9 → C12 → C14, C13 and C15 added. Q8 raised.
- it-20261002-0916 — stub (killed by a signal after 13 min): Gate 1 criteria written; critic REFUTED; work recovered from its stash by it-20261005-1416.
- it-20261001-2139 — done: loop setup, run interactively from Anjor's brief `loop/SETUP_PROMPT.md`. Recorded decisions 1 to 5 and the loop design; built `LOOP.md`, the runner, the launcher and the two reviewer subagents; tested them; smoke launch L001 round trip at 0 A100-h. No physics. Next: Gate 1.

## Before the loop

- 2026-10-01 — Phase 0 complete, on `main`. `SPEC.md` written. Blind rediscovery test and critic (`docs/rediscovery.md`, `derivations/blind_invariants.py`, `critic_invariants.py`): the fresh context found the paper's Γ^± (CMM B32) exactly, plus the closure results. D01 proves W and Γ symbolically and against the GANDALF v0.6.0 RHS (round-off). D02 gives the two reference profiles; the literal AS2018 Hermite equation is unstable as a truncated system (C11, `critic_as2018.py`), so reference (b) uses the antisymmetrised velocity derivative. Claims C1–C11 in `claims.md`. GANDALF issue #155 filed for runtime closure selection. Questions 1 and 2 answered from the repo; Gate 1 questions and the collaborator paragraph in `QUESTIONS.md`.
- 2026-10-01 — Decisions: no status emails; GANDALF changes via issues; work directly on `main`.
- 2026-09-19 — Plan and scaffold added. No code yet.
