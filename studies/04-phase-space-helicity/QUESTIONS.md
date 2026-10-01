# Questions for Anjor

Newest first. When a question is answered, move it to the bottom with the answer.

## Open

### Gate 1 (Phase 0 complete, 2026-10-01)

1. **Accept the invariant mapping and `SPEC.md`?** D01 proves the two invariants and maps them to Chandran, Mallet & Meyrand App. B (claims C1, C3–C6). One correction to PLAN §3 came out of the blind test: the copy closure keeps Γ exact and only breaks W (C5).

2. **Which invariant is the study about?** PLAN §2 calls Γ "the phase-space helicity". The paper's phase-space helicity H_ph-sp (their B18) has a 1/v∥ weight and is dense in Hermite space. The quantity the study is set up to measure, Γ = Σ √(2(m+1)) Re[g_{m+1}g_m*] with the Λ correction at m = 0, is their *additional* invariant Γ^± (B25, B32), the v∥-weighted free energy. Their closing paragraph of App. B is a prediction about exactly this Γ^±: not sign-definite, so no Fjørtoft constraint, so they expect it to cascade forward with W. Options: (a) keep Γ^± as the target (local in m, directly tied to the flux correlator, cheap); (b) add H_ph-sp as a second diagnostic (dense M×M form per k, fine in post-processing, but its truncation boundary term is nonlocal and needs its own derivation); (c) both. Recommendation: (a) for Phases 1–3, with H_ph-sp as a Phase 1 stretch item if the matrix elements of P(1/v∥) in the Hermite basis can be computed cleanly.

4. **Collaborator paragraph** (draft below). To Chandran, Mallet and Meyrand, or to Alex first?

### Solver version (2026-10-01, from the GANDALF repo audit)

5. **Regenerate the base state on GANDALF v0.6.0?** The ν-scan checkpoints (Phase 1 task 5) and the "ν-scan base parameters" for Set A were produced with v0.5.0. Two fixes since then change the Alfvénic sector:
   - gandalf#144: the Elsasser nonlinearity in v0.5.0 was 2× too large. The fix touched only `z_plus_rhs` and `z_minus_rhs`. The g equations (advection by Φ, the field-line term in Ψ) were already correct. So in the v0.5.0 base state the turbulence advected itself at twice the rate at which it advected g. That ratio sets phase unmixing and the backward Hermite flux, which is what this study measures.
   - gandalf#148: the integrating factor was applied twice to the Elsasser nonlinear increment, so the z± stepper was first order (measured 1.13 → 2.03 after the fix).

   The g right-hand side is the same in v0.5.0 and v0.6.0, so D01 and the Gate 1 mapping are unaffected. Options: (a) regenerate `alfven128_lowkz_f0p02_eta100` to saturation on v0.6.0, plus at least ν = 3, and use that for Phase 1 task 5 and Set A; (b) run Phase 1 task 5 on the v0.5.0 checkpoints as a pipeline check only, and run Set A on a regenerated v0.6.0 base; (c) stay on v0.5.0 for consistency with Study 2. Recommendation: (b). It does not block Phases 1–2, and Set A should not be launched on a base state with the 2× nonlinearity. The forcing amplitude f = 0.02 was calibrated under the 2× nonlinearity and may need recalibrating against the target δB/B₀. The same issue affects the Study 2 dissipative-anomaly result, which used this base state.

### Draft collaborator paragraph

> I am setting up a numerical test of whether conservation of your Γ^± (Appendix B, eqs B25 and B32) changes the Hermite cascade in driven KRMHD turbulence. The solver is GANDALF (JAX, Fourier–Hermite, 128³ with M = 128), which evolves one compressive Hermite hierarchy g with a (1 − 1/Λ) g_0 coupling in the g_1 equation; with Λ = Λ^+ = √5 at β_i = τ = Z = 1 this is your G^+ branch, and a second run at Λ^− would give G^−. I have checked symbolically and against the solver's right-hand side that Σ_m √(m+1) g_m g_{m+1} − g_0 g_1/Λ is exactly conserved at finite M under zero truncation, with no boundary term, and that it is the in-phase part of the same neighbour correlator whose quadrature part is the Hermite flux. The plan is to inject Γ^± in a controlled way by forcing g_0 and g_1 with a fixed relative phase, at equal free-energy injection, and compare W(m), the forward and backward Hermite fluxes and the Γ^± spectrum between opposite signs of injection and a zero-injection control. The balanced base state has ⟨Γ^±⟩ = 0 by the z → −z, v∥ → −v∥ symmetry, so the forcing breaks that symmetry by hand. Two questions I would value your view on: whether you expect Γ^± or H_ph-sp to be the more telling diagnostic of a velocity-space bottleneck, given that H_ph-sp is nonlocal in Hermite space; and whether running the two G^± branches separately (two runs at Λ^±) loses anything for this question, since the branches are decoupled in KRMHD.

## Answered

3. *Phase 1 closure test (anjor/gandalf#155).* — Shipped in GANDALF v0.6.0 (anjor/gandalf#153; #155 closed): `gandalf_step(..., closure="symmetric")`, also on `krmhd_rhs` and `PhysicsConfig.closure`. It applies g_{M+1} = g_{M−1} in the explicit RHS (streaming and the {Ψ, ·} term at m = M), the Lawson eigensystem and the IMEX operator; the default `"zero"` is unchanged. This repo is pinned to v0.6.0 (PR #9), so the copy-closure test in PLAN Phase 1 task 3 can run now. (2026-10-01)

1. *Where are the ν-scan checkpoints, and which configs produced them?* — Local: `studies/02-collisionality-scan/data/hermite128_nu{1,3,5,10,20,50}_imex/checkpoints/checkpoint_t{2180,2190,2200}.0.h5` (float32, g shape [128,128,65,129]). Same labels on Modal volume `krmhd-benchmark-vol`. Produced by `studies/02-collisionality-scan/scripts/modal_128_hermite.py`, resumed from `alfven128_lowkz_f0p02_eta100/checkpoints/checkpoint_t2000.0.h5`, not from a YAML config. Only ν = 3 has `spectra/` and `diagnostics_timeseries.npz` locally. (Answered from the repo, 2026-10-01.)
2. *Which Λ did the ν-scan use, Λ⁺ or Λ⁻?* — Λ = 2.2360677 = +√5 = Λ⁺ of CMM (B7) at β_i = τ = Z = 1 (checkpoint attr `state/Lambda`; D01 C1–C3). Λ⁻ = −√5 has never been run. (Answered from the repo, 2026-10-01.)
