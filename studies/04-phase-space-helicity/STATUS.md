# Study 04 — status

Front page of the autonomous loop. `LOOP.md` is the protocol. The log is in `log/`.

## Next block

**Gate 1, under the new rules** (decision 3; `LOOP.md`, "Quality gates"). Gate 1 has no coded check yet. Write one in `analysis/gates.py`: each derivation script (`derivations/01_invariant_mapping.py`, `02_reference_profiles.py`, `blind_invariants.py`, `critic_invariants.py`, `critic_as2018.py`) must run on the pinned GANDALF and pass its own checks. Get `study04-critic` verdicts for the claims marked supported that have no critic verdict: C2, C4 (blind SymPy only), C6, C9 (the model part) and C10. Then pass or fail the gate, write the report in `gate_reports/`, and record it in `decisions.md`, `claims.md` and `QUESTIONS.md` (For review).

## Queue

1. Phase 1 (PLAN.md §4): `analysis/helicity.py`, `analysis/budget.py`, the local conservation tests under both closures, the sign test, then the ν-scan checkpoint analysis. The v0.5.0 checkpoints test the pipeline only (decision 5).
2. The H_ph-sp derivation (decision 4): the matrix of 1/v∥ in the Hermite basis, conservation of H_ph-sp in the GANDALF hierarchy and its truncation term at m = M, as a self-checking script in `derivations/` with a critic pass. Then the post-processing diagnostic, labelled exploratory.
3. The new base state (decision 5): regenerate the Alfvénic base state on the pinned GANDALF, recalibrate the forcing, then the ν = 3 Hermite branch that set A starts from. Configs, run code with resume from checkpoint, a local smoke test, a `RUNS.md` row, then a launch through `loop/modal_launch.py`.

## In flight

Nothing. No Modal runs, no GANDALF PRs.

## Compute

Compute cap 200.0 A100-h: used 0.00, reserved 0.00, left 200.00 (launches 1, runs in flight 0)

## Iterations, newest first

- it-20261001-2139 — done: loop setup, run interactively from Anjor's brief `loop/SETUP_PROMPT.md`. Recorded decisions 1 to 5 and the loop design; built `LOOP.md`, the runner, the launcher and the two reviewer subagents; tested them; smoke launch L001 round trip at 0 A100-h. No physics. Next: Gate 1.

## Before the loop

- 2026-10-01 — Phase 0 complete, on `main`. `SPEC.md` written. Blind rediscovery test and critic (`docs/rediscovery.md`, `derivations/blind_invariants.py`, `critic_invariants.py`): the fresh context found the paper's Γ^± (CMM B32) exactly, plus the closure results. D01 proves W and Γ symbolically and against the GANDALF v0.6.0 RHS (round-off). D02 gives the two reference profiles; the literal AS2018 Hermite equation is unstable as a truncated system (C11, `critic_as2018.py`), so reference (b) uses the antisymmetrised velocity derivative. Claims C1–C11 in `claims.md`. GANDALF issue #155 filed for runtime closure selection. Questions 1 and 2 answered from the repo; Gate 1 questions and the collaborator paragraph in `QUESTIONS.md`.
- 2026-10-01 — Decisions: no status emails; GANDALF changes via issues; work directly on `main`.
- 2026-09-19 — Plan and scaffold added. No code yet.
