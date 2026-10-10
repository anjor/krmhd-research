# Questions for Anjor

Four sections: Blocking, Open (not blocking), For review, Answered. Every question has a permanent ID (Q1, Q2, ...). An ID is never reused or renumbered. Q1 to Q5 keep the numbers of the Gate 1 and solver-version questions of 1 October 2026; Q6 and Q7 are questions 1 and 2 of PLAN.md §9. Items under For review get IDs R1, R2, ...

How Anjor answers: add a line that starts with `ANSWER:` under a question, commit and push. How Anjor vetoes a passed gate or a decision: add a line that starts with `VETO:` under its item in For review, say what is wrong, commit and push. The loop reads only pushed commits. Uncommitted edits are never its input. The next iteration handles a veto before any other work.

## Blocking

Questions the loop cannot get past without Anjor. While this section has an item, `STOP` exists in the study folder and the loop does not run. Answer the question, delete `STOP` (or run `loop/resume.sh`), commit and push.

- **STOP 2026-10-09 (agent)**: Gate 1 criteria part (a), the shared report machinery, was refuted a third time since the approach changed (critic_it-20261009-2146_1.md: the git checks run with the inherited environment, so git environment variables can make a report name another repository's HEAD and pass the clean-tree check; the runner reads head lines with case folding that the writer does not mirror). LOOP.md section 4: ... See `STOP`.

### Q9. Gate 1 report machinery refuted three times since the approach changed: how should Gate 1 go on? (it-20261009-2146)

Since it-20261005-1446 split the Gate 1 criteria review into parts (a), (b) and (c), part (a) has been refuted three times: `critic_it-20261005-2234_1.md`, `critic_it-20261009-2116_1.md` and `critic_it-20261009-2146_1.md` (claim C15). LOOP.md §4 makes three failed entries on one approach a hard stop.

The third review found that most of the machinery now holds. Questions 2 to 6 held up: re-judge from saved outputs, binding to this evaluation, agreement between the writer, the launcher and the runner on the verdict, the Result rule, and importability by a frozen evaluation. The launcher's form check passed on writer-made reports for every gate. What it refuted:
1. `git_head`, `tree_is_clean` and `report_history_problems` run git with the inherited environment, so `GIT_DIR` and `GIT_WORK_TREE` (or a global git config) can make a report name another repository's HEAD and pass the clean-tree check.
2. Limits that are true but not stated in the docstrings: a stale `.pyc` in the ignored `analysis/__pycache__/`; the derivation scripts' shared folder on `sys.path`; JAX and XLA variables that `-E` keeps; and `finish` does not check that the gate code at finish equals the code at the named commit.
3. The runner's reader of a head uses case-insensitive regexes on keys it does not lower-case, so Unicode case folding (ſ, ı, the Kelvin sign) lets the writer emit a table row that the runner reads as `Result: PASS` or as the Critic line. The review found no path to a false PASS; the risk is a spurious runner STOP.
4. Test gaps: none for the git environment or the runner's case-insensitive reading, and the history refusal and the re-judge are never exercised through `finish_report`.

Each review has found new edge cases. They are less about whether the report is right and more about an adversary who controls the environment, which no review can close for good.

Options:
- (a) The loop fixes 1 to 4 and asks the critic once more, as a fresh allowance: git with `GIT_*` removed and the toplevel checked against the repo, the writer refusing anything the runner's case-insensitive reading would take as a verdict or Critic line, the limits stated, and the tests added.
- (b) As (a), but you also fix a threat model for the gate code, for example: "the machinery guards against mistakes and against a head edited after the fact, not against a session that sets git or Python environment variables or plants ignored files". The critic then judges against it, so the review can converge.
- (c) You review the report machinery yourself and accept it, recorded as your decision, and the loop goes on to part (b).

Recommendation: (b). The fixes in (a) are cheap, about one iteration. Without a stated threat model, a fourth review is likely to find a fifth class of environmental edge case. The runner already checks the session for the threats in it (T1 and the git config guards).

ANSWER: (b). Fix 1 to 4, then ask the critic once more, as a fresh allowance. Threat model for the gate code, for parts (a), (b) and (c): the machinery must catch honest mistakes (a report naming the wrong commit, a dirty tree, a check that did not run or did not finish, a verdict or Result line that does not follow from the saved outputs, a criterion changed after review) and a report head edited after the fact. It does not need to defend against a session that sets git, Python, JAX or XLA environment variables, plants or edits ignored files (a stale `.pyc`, scratch outputs), or forges a saved critic file. The runner's guards cover a hostile session. Weaknesses outside this model go in the docstrings as stated limits and are not grounds for REFUTED. Give the critic this paragraph with each criteria review. If the fourth review of part (a) is refuted on findings inside the model, stop and ask me again.

## Open, not blocking

The loop proceeds on its recommendation. If Anjor answers differently, the loop adapts and logs the rework.

### Q8. The Study 2 data link is outside the loop session's readable folders (it-20261005-1416)

`studies/02-collisionality-scan/data` in the loop's clone links to the data in Anjor's checkout `~/repos/anjor/krmhd-research`. A loop session may only list and read inside the clone and the GANDALF worktree, so listing `hermite128_nu3_imex/checkpoints/` through the link was denied in it-20261005-1416. Phase 1 task 5 (the ν-scan checkpoint analysis, a pipeline test under decision 5) and the Gate 2 checkpoint budgets need to read those files. Options: (a) Anjor adds the Study 2 data folder to the session's readable folders (a change to the loop's settings, which only he makes); (b) the loop reads the checkpoints from the cloud volume instead, through the launcher's `fetch`, which today fetches only the loop's own runs; (c) Anjor copies the ν = 3 checkpoints (t = 2180, 2190, 2200) and its `diagnostics_timeseries.npz` into `studies/04-phase-space-helicity/data/v050_nu3/` in the loop's clone, which git ignores. Recommendation: (c), because it touches no guard and the pipeline test needs only ν = 3. Not blocking now: Gate 1 and the Phase 1 local tests do not need the checkpoints. The loop will not read through the link until this is settled.

### Q4. Collaborator paragraph (Gate 1, 2026-10-01)

**Collaborator paragraph** (draft below). To Chandran, Mallet and Meyrand, or to Alex first?

This stays with Anjor and does not block (brief, 1 October 2026): sending it leaves the two repos, so the loop never sends it. PLAN.md §9 question 3 asked the same thing. Recommendation from the loop: none on the addressee, which is Anjor's call. The draft below stands as written.

#### Draft collaborator paragraph

> I am setting up a numerical test of whether conservation of your Γ^± (Appendix B, eqs B25 and B32) changes the Hermite cascade in driven KRMHD turbulence. The solver is GANDALF (JAX, Fourier–Hermite, 128³ with M = 128), which evolves one compressive Hermite hierarchy g with a (1 − 1/Λ) g_0 coupling in the g_1 equation; with Λ = Λ^+ = √5 at β_i = τ = Z = 1 this is your G^+ branch, and a second run at Λ^− would give G^−. I have checked symbolically and against the solver's right-hand side that Σ_m √(m+1) g_m g_{m+1} − g_0 g_1/Λ is exactly conserved at finite M under zero truncation, with no boundary term, and that it is the in-phase part of the same neighbour correlator whose quadrature part is the Hermite flux. The plan is to inject Γ^± in a controlled way by forcing g_0 and g_1 with a fixed relative phase, at equal free-energy injection, and compare W(m), the forward and backward Hermite fluxes and the Γ^± spectrum between opposite signs of injection and a zero-injection control. The balanced base state has ⟨Γ^±⟩ = 0 by the z → −z, v∥ → −v∥ symmetry, so the forcing breaks that symmetry by hand. Two questions I would value your view on: whether you expect Γ^± or H_ph-sp to be the more telling diagnostic of a velocity-space bottleneck, given that H_ph-sp is nonlocal in Hermite space; and whether running the two G^± branches separately (two runs at Λ^±) loses anything for this question, since the branches are decoupled in KRMHD.

## For review

Gates the loop has passed and decisions it has made, newest first, for Anjor's veto. Each item names its gate report or `decisions.md` row and the commit.

None yet.

## Answered

- **STOP 2026-10-06 (runner)**: WIP streak (runner checks after iteration it-20261006-1454): streak: 6 iterations in a row ended WIP or as a stub (limit 6): it-20261006-1454 it-20261005-2234 it-20261005-1519 it-20261005-1446 it-20261005-1416 it-20261002-0916 See `STOP`.

ANSWER: Four of the six were not failures: I cancelled those sessions myself (Ctrl-C) because I needed my Claude usage limits for other work. Only it-20261005-1416 and it-20261005-1446 ended WIP on their own. Continue Gate 1 with the split critic review planned in STATUS.md. From now on the runner stubs a session I end with a signal as 'interrupted', and the streak does not count it.
Handled (it-20261009-2116): continued Gate 1 criteria part (a) of the split review, fixing the findings of the REFUTED review critic_it-20261005-2234_1.md, with a new review.

### Q2. Which invariant is the study about? (Gate 1, 2026-10-01)

**Which invariant is the study about?** PLAN §2 calls Γ "the phase-space helicity". The paper's phase-space helicity H_ph-sp (their B18) has a 1/v∥ weight and is dense in Hermite space. The quantity the study is set up to measure, Γ = Σ √(2(m+1)) Re[g_{m+1}g_m*] with the Λ correction at m = 0, is their *additional* invariant Γ^± (B25, B32), the v∥-weighted free energy. Their closing paragraph of App. B is a prediction about exactly this Γ^±: not sign-definite, so no Fjørtoft constraint, so they expect it to cascade forward with W. Options: (a) keep Γ^± as the target (local in m, directly tied to the flux correlator, cheap); (b) add H_ph-sp as a second diagnostic (dense M×M form per k, fine in post-processing, but its truncation boundary term is nonlocal and needs its own derivation); (c) both. Recommendation: (a) for Phases 1–3, with H_ph-sp as a Phase 1 stretch item if the matrix elements of P(1/v∥) in the Hermite basis can be computed cleanly.

Answered 1 October 2026 by decision 4 (`decisions.md`, from Anjor's brief `loop/SETUP_PROMPT.md`): The Γ of the plan is the paper's Γ^± (their B25 and B32). It stays the forced quantity. H0, H1 and H2 stay as written and refer to Γ^±. The loop also derives and measures the paper's phase-space helicity H_ph-sp (their B18), in post-processing. That diagnostic is exploratory, and every result from it says so. Recorded as a dated clarification under PLAN.md §2.

### Q5. Regenerate the base state on GANDALF v0.6.0? (solver version, 2026-10-01)

**Regenerate the base state on GANDALF v0.6.0?** The ν-scan checkpoints (Phase 1 task 5) and the "ν-scan base parameters" for Set A were produced with v0.5.0. Two fixes since then change the Alfvénic sector:
   - gandalf#144: the Elsasser nonlinearity in v0.5.0 was 2× too large. The fix touched only `z_plus_rhs` and `z_minus_rhs`. The g equations (advection by Φ, the field-line term in Ψ) were already correct. So in the v0.5.0 base state the turbulence advected itself at twice the rate at which it advected g. That ratio sets phase unmixing and the backward Hermite flux, which is what this study measures.
   - gandalf#148: the integrating factor was applied twice to the Elsasser nonlinear increment, so the z± stepper was first order (measured 1.13 → 2.03 after the fix).

   The g right-hand side is the same in v0.5.0 and v0.6.0, so D01 and the Gate 1 mapping are unaffected. Options: (a) regenerate `alfven128_lowkz_f0p02_eta100` to saturation on v0.6.0, plus at least ν = 3, and use that for Phase 1 task 5 and Set A; (b) run Phase 1 task 5 on the v0.5.0 checkpoints as a pipeline check only, and run Set A on a regenerated v0.6.0 base; (c) stay on v0.5.0 for consistency with Study 2. Recommendation: (b). It does not block Phases 1–2, and Set A should not be launched on a base state with the 2× nonlinearity. The forcing amplitude f = 0.02 was calibrated under the 2× nonlinearity and may need recalibrating against the target δB/B₀. The same issue affects the Study 2 dissipative-anomaly result, which used this base state.

Answered 1 October 2026 by decision 5 (`decisions.md`, from Anjor's brief `loop/SETUP_PROMPT.md`): option (b). Phase 1 uses the v0.5.0 checkpoints only to test the pipeline. No science claim rests on them. The loop regenerates the Alfvénic base state on the pinned GANDALF and recalibrates the forcing. Set A starts from the new base state.

### Q1. Accept the invariant mapping and `SPEC.md`? (Gate 1, 2026-10-01)

**Accept the invariant mapping and `SPEC.md`?** D01 proves the two invariants and maps them to Chandran, Mallet & Meyrand App. B (claims C1, C3–C6). One correction to PLAN §3 came out of the blind test: the copy closure keeps Γ exact and only breaks W (C5).

Superseded by decision 3 (1 October 2026): the agent passes gates itself. Gate 1 is now evaluated by the loop under `LOOP.md`, with a coded check and a critic verdict, and the pass is listed under For review for Anjor's veto.

### Q3. Phase 1 closure test (anjor/gandalf#155)

*Phase 1 closure test (anjor/gandalf#155).* — Shipped in GANDALF v0.6.0 (anjor/gandalf#153; #155 closed): `gandalf_step(..., closure="symmetric")`, also on `krmhd_rhs` and `PhysicsConfig.closure`. It applies g_{M+1} = g_{M−1} in the explicit RHS (streaming and the {Ψ, ·} term at m = M), the Lawson eigensystem and the IMEX operator; the default `"zero"` is unchanged. This repo is pinned to v0.6.0 (PR #9), so the copy-closure test in PLAN Phase 1 task 3 can run now. (2026-10-01)

### Q6. Where are the ν-scan checkpoints, and which configs produced them? (PLAN.md §9.1)

*Where are the ν-scan checkpoints, and which configs produced them?* — Local: `studies/02-collisionality-scan/data/hermite128_nu{1,3,5,10,20,50}_imex/checkpoints/checkpoint_t{2180,2190,2200}.0.h5` (float32, g shape [128,128,65,129]). Same labels on Modal volume `krmhd-benchmark-vol`. Produced by `studies/02-collisionality-scan/scripts/modal_128_hermite.py`, resumed from `alfven128_lowkz_f0p02_eta100/checkpoints/checkpoint_t2000.0.h5`, not from a YAML config. Only ν = 3 has `spectra/` and `diagnostics_timeseries.npz` locally. (Answered from the repo, 2026-10-01.)

### Q7. Which Λ did the ν-scan use, Λ⁺ or Λ⁻? (PLAN.md §9.2)

*Which Λ did the ν-scan use, Λ⁺ or Λ⁻?* — Λ = 2.2360677 = +√5 = Λ⁺ of CMM (B7) at β_i = τ = Z = 1 (checkpoint attr `state/Lambda`; D01 C1–C3). Λ⁻ = −√5 has never been run. (Answered from the repo, 2026-10-01.)
