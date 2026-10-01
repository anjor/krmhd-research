# Study 04 — status

Next open block: **Gate 1 decision (Anjor)**. See `QUESTIONS.md` items 1–4. Phase 1 does not start until Gate 1 is passed.

## Log

- 2026-10-01 — Phase 0 complete, on `main`. `SPEC.md` written. Blind rediscovery test and critic (`docs/rediscovery.md`, `derivations/blind_invariants.py`, `critic_invariants.py`): the fresh context found the paper's Γ^± (CMM B32) exactly, plus the closure results. D01 proves W and Γ symbolically and against the GANDALF v0.6.0 RHS (round-off). D02 gives the two reference profiles; the literal AS2018 Hermite equation is unstable as a truncated system (C11, `critic_as2018.py`), so reference (b) uses the antisymmetrised velocity derivative. Claims C1–C11 in `claims.md`. GANDALF issue #155 filed for runtime closure selection. Questions 1 and 2 answered from the repo; Gate 1 questions and the collaborator paragraph in `QUESTIONS.md`.
- 2026-10-01 — Decisions: no status emails; GANDALF changes via issues; work directly on `main`.
- 2026-09-19 — Plan and scaffold added. No code yet.
