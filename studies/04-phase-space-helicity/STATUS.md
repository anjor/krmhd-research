# Study 04 — status

Front page of the autonomous loop. `LOOP.md` is the protocol. The log is in `log/`.

## Next block

**Gate 1 criteria, third submission to the critic** (PLAN.md §4 Phase 0; LOOP.md §5). `analysis/gates.py` has the Gate 1 coded check and the shared report writer. Two critic reviews of the criteria returned REFUTED (`critic_it-20261002-0916_1.md`, `critic_it-20261005-1416_1.md`; claim C15). it-20261005-1416 fixed: NaN handling in the `critic_as2018.py` rule, c_Λ values in SPEC §1, the m = 0 correlator sentence in SPEC §2, SPEC §3 GANDALF defaults (`hyper_r`, `hyper_n`, `eta_z`), C12 superseded by C14 (forcing condition), claim reviews must be titled `claim <ID>` and gate reviews must name the gate, HEAD read before the scripts run and the tree re-checked after. Still open from `critic_it-20261005-1416_1.md`, to fix before the third review:
- Finding 2–3: the AS2018 growth band [0.25, 1.0] cannot fail from above, because M S is the analytic ceiling of the Itô rate, and the Heun raising-only rows are judged only by γ > 0. C11's "≈ ½ M S" does not match D02's own docstring ("about M S"). Recommended: supersede C11 with a claim that does not quote a coefficient (unstable, 0 < γ ≤ M S for Itô by the critic's bound, dt-independent under both interpretations), and set the criteria from that text; state in the intent that the C11 numbers (γ/(MS) 0.36–0.93) are in view.
- Finding 5: C1's "in-phase partner of the `hermite_flux` correlator" is false at m = 0. Supersede C1 with corrected text; the new claim needs a new critic verdict (its Phase 0 verdict does not carry over).
- Finding 6: the D02 docstring (`02_reference_profiles.py` L132–134) says the k-odd part of Γ is not constrained by a symmetry; the critic found an exact symmetry that makes it zero. Check it, correct the docstring and SPEC §2.
- Finding 9: PLAN.md §4 has Gate 1 accept the whole of SPEC.md, but the check covers §1–§2. Decide: add coded checks of SPEC §3–§6 against GANDALF's signatures and defaults, or argue the narrower scope in the criteria.
- Finding 11: `finish_report` checks only "Gate 4" for Gate 4; LOOP.md §6 also needs `set <S>`.
After a SUPPORTED review of the criteria (row in `decisions.md`), get claim verdicts for C2, C4, C6, C10, C11 or its successor, C14 (and C1's successor), then evaluate Gate 1. This block has now been tried twice; if the third review fails, change the approach (for example split the criteria review into the script rules and the claim rules).

## Queue

1. Phase 1 (PLAN.md §4): `analysis/helicity.py`, `analysis/budget.py`, the local conservation tests under both closures, the sign test, then the ν-scan checkpoint analysis. The v0.5.0 checkpoints test the pipeline only (decision 5). The checkpoint analysis waits for QUESTIONS.md Q8 (the Study 2 data link is outside the session's readable folders).
2. The H_ph-sp derivation (decision 4): the matrix of 1/v∥ in the Hermite basis, conservation of H_ph-sp in the GANDALF hierarchy and its truncation term at m = M, as a self-checking script in `derivations/` with a critic pass. Then the post-processing diagnostic, labelled exploratory.
3. The new base state (decision 5): regenerate the Alfvénic base state on the pinned GANDALF, recalibrate the forcing, then the ν = 3 Hermite branch that set A starts from. Configs, run code with resume from checkpoint, a local smoke test, a `RUNS.md` row, then a launch through `loop/modal_launch.py`.

## In flight

Nothing. No Modal runs, no GANDALF PRs.

## Compute

Compute cap 200.0 A100-h: used 0.00, reserved 0.00, left 200.00 (launches 1, runs in flight 0)

## Iterations, newest first

- it-20261005-1416 — WIP: Gate 1 criteria revised after the REFUTED review of the dead iteration; the critic's second review is also REFUTED (`critic_it-20261005-1416_1.md`). Cheap findings fixed; five remain (Next block). C9 → C12 → C14, C13 and C15 added. Q8 raised.
- it-20261002-0916 — stub (killed by a signal after 13 min): Gate 1 criteria written; critic REFUTED; work recovered from its stash by it-20261005-1416.
- it-20261001-2139 — done: loop setup, run interactively from Anjor's brief `loop/SETUP_PROMPT.md`. Recorded decisions 1 to 5 and the loop design; built `LOOP.md`, the runner, the launcher and the two reviewer subagents; tested them; smoke launch L001 round trip at 0 A100-h. No physics. Next: Gate 1.

## Before the loop

- 2026-10-01 — Phase 0 complete, on `main`. `SPEC.md` written. Blind rediscovery test and critic (`docs/rediscovery.md`, `derivations/blind_invariants.py`, `critic_invariants.py`): the fresh context found the paper's Γ^± (CMM B32) exactly, plus the closure results. D01 proves W and Γ symbolically and against the GANDALF v0.6.0 RHS (round-off). D02 gives the two reference profiles; the literal AS2018 Hermite equation is unstable as a truncated system (C11, `critic_as2018.py`), so reference (b) uses the antisymmetrised velocity derivative. Claims C1–C11 in `claims.md`. GANDALF issue #155 filed for runtime closure selection. Questions 1 and 2 answered from the repo; Gate 1 questions and the collaborator paragraph in `QUESTIONS.md`.
- 2026-10-01 — Decisions: no status emails; GANDALF changes via issues; work directly on `main`.
- 2026-09-19 — Plan and scaffold added. No code yet.
