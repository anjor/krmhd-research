# Study 04 — Phase-space helicity and the Hermite cascade in KRMHD

Amended 1 October 2026, from Anjor's brief `loop/SETUP_PROMPT.md` and his decisions of that date (`decisions.md`, decisions 1 to 5 and the loop design). The study now runs as an autonomous loop under `LOOP.md`. §1, §5 and §6 are rewritten. In §4 only who passes each gate and who launches runs has changed. §2 has a dated clarification (decision 4). The rest of §2, §3, §7 and the tasks in §4 are as written on 19 September. §2 and §7 are frozen: a change to them is a hard stop. Where the paragraph below says that Anjor launches production runs, decision 1 replaces it.

Plan, 19 September 2026. Source: the 16 September research memo. This file is the operating document for an AI-led study. Anjor supplies physics judgement and launches production runs. The agent does everything else.

## 1. How to work on this study

Rewritten 1 October 2026 (decisions 1 to 5 and the loop design).

The study runs as an autonomous loop. Each iteration is a fresh agent session that reads `LOOP.md` first and follows it. `LOOP.md` is the full protocol: how one iteration runs, how gates pass, how runs are launched, how GANDALF is changed, and the hard stops at which the loop stops and waits for Anjor. `STATUS.md` names the next block.

Working rules that still hold, in addition to the repo `CLAUDE.md` and `LOOP.md`:

- Production runs go through `loop/modal_launch.py` and nothing else. Every run set has a row in `RUNS.md`: configs, purpose, gate served, estimated and actual A100-hours.
- Runs that fit on the Mac may be run directly: grids up to 32³ with M ≤ 64, under 20 minutes wall time. Use `uv run`.
- Questions and points for Anjor go into `QUESTIONS.md`.
- Every claim goes into `claims.md`: claim, origin (human, agent, literature), evidence, falsifier, status.
- Every derivation is a script in `derivations/` that checks itself with SymPy or a numerical test.
- Gates are code in `analysis/gates.py`. The success signal is `validate_run(run_id)`, never a narrative.
- Derivations, gate evaluations and interpretations get a second pass from a critic context (the `study04-critic` subagent). The critic sees `SPEC.md`, the scripts and the raw outputs. It does not see the actor's conclusion. Record its verdict in `claims.md`.
- GANDALF changes go through issues and pull requests on anjor/gandalf, in the loop's GANDALF worktree, under `LOOP.md`. Never edit the installed package. Study diagnostics and forcing live in `shared/` or this study's `analysis/`.

## 2. Question

Does conservation of the KRMHD phase-space helicity Γ (Chandran, Mallet & Meyrand 2026) change how compressive free energy moves to high Hermite number?

Pre-registered outcomes:

- H0: controlled Γ injection changes neither W(m) nor the Hermite flux beyond seed and time scatter.
- H1: Γ does not cascade. Injected Γ stays at m ≲ 5 and reshapes the low-m spectrum.
- H2: Γ cascades with the free energy and changes the ratio of forward to backward Hermite flux.

Any of the three is a result if the flux budget closes and the effect is resolved.

Clarification, 1 October 2026 (decision 4; answers Gate 1 question Q2). The Γ of this plan is the paper's Γ^± (Chandran, Mallet & Meyrand, their B25 and B32). It stays the forced quantity. H0, H1 and H2 stay as written and refer to Γ^±. The loop also derives and measures the paper's phase-space helicity H_ph-sp (their B18), in post-processing. That diagnostic is exploratory, and every result from it says so.

## 3. The invariant in GANDALF variables

GANDALF evolves one Hermite hierarchy g_m with closure parameter Λ. The g_1 equation couples to g_0 through (1 − 1/Λ)/√2 · g_0 (`physics.py`, thesis Eq. 2.8). In these variables the invariant has the form

    Γ = Σ_k Σ_{m=0}^{M−1} c_m Re[g_{m+1}(k) g_m*(k)]

with c_m ∝ √(m+1) for m ≥ 1 and c_0 ∝ (1 − 1/Λ).

The existing diagnostic `hermite_flux` computes −k∥ √(2(m+1)) Im[g_{m+1} g_m*]. So the invariant is the in-phase part of the same neighbour correlator whose quadrature part is the free-energy flux. Two consequences to check in Phase 0:

- In the linear phase-mixing solution, neighbouring moments are in quadrature. So Γ(m) = 0 for m ≥ 1. The m^(−1/2) base state should show this. Γ(m) is then a direct measure of departure from linear phase mixing.
- A single GANDALF run tests one Γ^σ. Both signs need runs at Λ⁺ and Λ⁻, derived from β_i and Z/τ.

The exact coefficients, the √2 conventions and the Λ term are Gate 1's job. Do not trust this section until `derivations/01_invariant_mapping.py` reproduces conservation symbolically and numerically.

## 4. Phases

### Phase 0 — theory and conventions (week 1)

1. Write `SPEC.md`: equations, normalisation, domain, dissipation ν(m/M)^n with n = 6, the m = 0, 1 exemptions, closure, observables, tolerances. Every default the solver resolves goes in it.
2. Blind rediscovery test. Give a fresh model context the collisionless g_m equations and a quadratic ansatz Σ a_m g_m g_{m+1} + Σ b_m g_m². Ask it to find conserved combinations. Solve the coefficient constraints with SymPy. Hand the result to a critic context to falsify on random spectral states. Only then compare with the paper's Appendix B. Record the outcome in `docs/rediscovery.md`. This is a methodology result whatever happens.
3. `derivations/01_invariant_mapping.py`: map GANDALF's g and Λ to the paper's G^± and Λ^±. Prove dΓ/dt = 0 for the discrete collisionless hierarchy with the closure used, or state the boundary term at m = M.
4. `derivations/02_reference_profiles.py`: compute Γ(m) and the flux for the linear phase-mixing solution and for the Adkins–Schekochihin echo model. These are the two reference profiles.
5. Draft the paragraph for Chandran, Mallet and Meyrand, or for Alex. Put it in `QUESTIONS.md`. Anjor sends it.

Gate 1: the mapping and `SPEC.md` are accepted. The agent passes the gate under `LOOP.md` (coded check and critic) and logs the pass for Anjor's veto.

### Phase 1 — diagnostics and conservation (weeks 2–3)

1. `analysis/helicity.py`: `hermite_helicity(state)` returning c_m Re[g_{m+1} g_m*] on the [Nz, Ny, Nx//2+1, M] grid, parallel to `hermite_flux`. Reductions: Γ(m), Γ(k⊥), total Γ.
2. `analysis/budget.py`: dW/dt and dΓ/dt budgets: injection, collisional sink, truncation term at m = M, residual.
3. Local conservation tests at 32³, M = 32 and 64, ν = 0, no forcing, seeded g. The residuals of Γ and W must fall with dt and with M. Test both closures, `closure_zero` and `closure_symmetric`.
4. Sign test: flip the sign of g_1 in the initial condition and confirm Γ flips sign.
5. Apply the diagnostics to the ν-scan checkpoints. Report Γ(m), Γ(k⊥), the flux and the collisional Γ sink. Compare with the two reference profiles from Phase 0.

Gate 2: residuals converge, and budgets close within tolerance on the checkpoints. The agent declares the gate passed under `LOOP.md` and logs the pass for Anjor's veto. First result: where Γ lives in m in the base state.

Runs: local only, plus the existing checkpoints. No new Modal runs.

### Phase 2 — helicity forcing (week 4)

1. `shared/hermite_pair_forcing.py`: force g_0 and g_1 on the same k-shell with f_1 = r e^{iφ} f_0. φ = 0 and φ = π give opposite Γ injection. φ = π/2 gives zero. Use the same k-support and amplitude as the existing `perp_lowkz` Hermite forcing.
2. Injection diagnostics: measured ε_W and ε_Γ per step, and accumulated. Injection is measured from the state response, not inferred from the forcing definition.
3. Two short local runs at 32³, M = 64, φ = 0 and π, with equal measured ε_W.

Gate 3: ε_Γ flips sign between φ = 0 and π at equal ε_W, within 5 percent. The agent declares the gate passed under `LOOP.md` and logs the pass for Anjor's veto.

Runs: local.

### Phase 3 — the experiment (weeks 5–7)

Run set A, launched by the agent through `loop/modal_launch.py` and listed in `RUNS.md`: 128³, M = 128, the ν-scan base parameters, three forcings (φ = π/2, 0, π), two seeds each. Six runs.

Measurements per run: W(m), forward and backward Hermite flux, Γ(m), Γ(k⊥), transfer time from injection to the collisional range, low-m accumulation, ε_ν.

Gate 4: the difference between φ = 0 and φ = π exceeds the seed and time scatter of φ = π/2. If it does, the agent launches run set B: the clearest pair at M = 64 and 256, and at two values of ν. If it does not, it launches nothing and goes to Phase 4 with H0.

The agent sets the convergence criteria before the launch (decision 3) and decides whether set B is worth the compute, within the cap.

### Phase 4 — interpretation and write-up (weeks 8–10)

The actor writes the interpretation in `docs/interpretation.md`. The critic attacks it with alternative explanations: forcing artefact, truncation, hypercollision order, Λ sign, averaging window. Anjor chooses between interpretations.

Output: a short paper in `paper/phase-space-helicity/`, or a negative-result memo. Either way, add Γ to the Study 3 diagnostics.

## 5. Runs and compute

Rewritten 1 October 2026 (decisions 1 and 5).

| Set | Where | Size | Purpose |
|---|---|---|---|
| Conservation tests | Mac | 32³, M ≤ 64, ν = 0 | Gate 2 |
| Forcing tests | Mac | 32³, M = 64 | Gate 3 |
| Checkpoint analysis | Mac | existing v0.5.0 ν-scan states | pipeline test only (decision 5) |
| New base state | Modal, the agent launches | 128³: Alfvénic sector to saturation on the pinned GANDALF, forcing recalibrated, then the ν = 3 Hermite branch at M = 128 | base state for set A (decision 5) |
| Set A | Modal, the agent launches | 128³, M = 128, six runs | Gate 4 |
| Set B | Modal, the agent launches | 128³ at M = 64 and 256, two ν | convergence |

The agent launches Modal runs only through `loop/modal_launch.py`. The cap is 200 A100-hours for the whole study, set in `loop/config.env` and enforced by the launcher, which reserves hours in `compute_ledger.json` before any Modal call. Expected spend is about 140 to 160 A100-hours: about 42 for set A, 65 for set B and 25 to 40 for the new base state. Estimate each set's hours from measured wall times before launching it (`docs/run_log.md`: the v0.5.0 ν-scan runs at 128³, M = 128, 200 τ_A took 24,228 to 26,058 s on one A100). Record the estimate and the actual cost in `RUNS.md`.

## 6. Anjor's decisions

Rewritten 1 October 2026 (decision 3 and the loop design).

The agent decides everything that is not a hard stop and records each decision and its reason in `decisions.md`. It passes gates itself, with a coded check and a critic verdict, and lists each pass in `QUESTIONS.md` under "For review". Anjor reviews after the fact. A `VETO:` line under a passed gate or a decision reopens it, and an `ANSWER:` line answers a question. He pauses and resumes the loop with `loop/pause.sh` and `loop/resume.sh`, and he alone changes the compute cap.

The loop stops and waits for Anjor only at the hard stops in `LOOP.md`: a launch would exceed the compute cap; the first or second kill criterion of §7 is met; a change is needed to a frozen section (§2 and §7 here, `SPEC.md` §7); a change is needed to the loop's own guards (`loop/`, `.claude/`, `LOOP.md`); something would leave the two repos (an email, a message to a collaborator, an arXiv submission, a post, a release); the study is complete; the loop cannot make progress.

The decision points of 19 September change accordingly. Gates 1 to 4, the set B convergence criteria and the launches of the base state and sets A and B are the agent's. Where §4 says that Anjor chooses between interpretations, the agent records its choice in `decisions.md` and lists it for review. Sending the collaborator paragraph, submitting a paper and any external claim stay with Anjor, because they leave the two repos.

## 7. Kill criteria

- After Phase 1: the Γ residual does not converge with dt and M under either closure. Report and stop.
- After Phase 2: ε_Γ cannot be controlled independently of ε_W. Report and stop.
- After Phase 3: the effect is below scatter at two resolutions. Write up the null with the closed budget.

## 8. Layout

```
studies/04-phase-space-helicity/
  PLAN.md          this file
  SPEC.md          equations, conventions, tolerances
  LOOP.md          the loop's protocol (added 1 October 2026)
  STATUS.md        front page: next block, queue, in flight, compute, one line per iteration
  QUESTIONS.md     Blocking, Open not blocking, For review, Answered
  RUNS.md          run sets: proposed → launched → done
  claims.md
  decisions.md
  compute_ledger.json  A100-hour reservations and charges, written only by the launcher
  log/             one file per day, one entry per iteration, append-only
  gate_reports/    one committed report per gate evaluation
  loop/            runner, launcher, settings, tests (guarded; only Anjor changes it)
  derivations/
  configs/
  analysis/        helicity.py, budget.py, gates.py, plots
  figures/
  docs/            rediscovery.md, interpretation.md
```

## 9. Open questions at the start

All three are answered or moved; see `QUESTIONS.md` (Q6, Q7 and Q4).

1. Where are the ν-scan checkpoints (Modal volume path), and which configs produced them?
2. Which Λ did the ν-scan use, and is it Λ⁺ or Λ⁻? The Study 2 configs have Λ = 2.236.
3. Does the collaborator paragraph go to Chandran first, or to Alex first?

## 10. References

Chandran, Mallet & Meyrand 2026, arXiv:2607.27981. Adkins & Schekochihin 2018, JPP 84, arXiv:1709.03203. Meyrand, Kanekar, Dorland & Schekochihin 2019, PNAS 116, 1185. Kanekar 2015, PhD thesis (GANDALF equations). Study 2 dissipative-anomaly paper, this repo.
