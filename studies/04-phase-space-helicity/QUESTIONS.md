# Questions for Anjor

Newest first. When a question is answered, move it to the bottom with the answer.

## Open

### Gate 1 (Phase 0 complete, 2026-10-01)

1. **Accept the invariant mapping and `SPEC.md`?** D01 proves the two invariants and maps them to Chandran, Mallet & Meyrand App. B (claims C1, C3–C6). One correction to PLAN §3 came out of the blind test: the copy closure keeps Γ exact and only breaks W (C5).

2. **Which invariant is the study about?** PLAN §2 calls Γ "the phase-space helicity". The paper's phase-space helicity H_ph-sp (their B18) has a 1/v∥ weight and is dense in Hermite space. The quantity the study is set up to measure, Γ = Σ √(2(m+1)) Re[g_{m+1}g_m*] with the Λ correction at m = 0, is their *additional* invariant Γ^± (B25, B32), the v∥-weighted free energy. Their closing paragraph of App. B is a prediction about exactly this Γ^±: not sign-definite, so no Fjørtoft constraint, so they expect it to cascade forward with W. Options: (a) keep Γ^± as the target (local in m, directly tied to the flux correlator, cheap); (b) add H_ph-sp as a second diagnostic (dense M×M form per k, fine in post-processing, but its truncation boundary term is nonlocal and needs its own derivation); (c) both. Recommendation: (a) for Phases 1–3, with H_ph-sp as a Phase 1 stretch item if the matrix elements of P(1/v∥) in the Hermite basis can be computed cleanly.

3. **Phase 1 closure test.** `gm_rhs` hardcodes zero truncation. Since Γ is exact under both closures (C5), the two-closure test in PLAN Phase 1 task 3 only matters for W. Options: file a GANDALF issue for runtime closure selection, or drop the copy-closure test and keep the dt- and M-convergence tests. Recommendation: drop it; note the result in the paper instead.

4. **Collaborator paragraph** (draft below). To Chandran, Mallet and Meyrand, or to Alex first?

5. **Email.** Per `decisions.md` questions also go by email. A Gmail draft of this block's summary is prepared; it is not sent.

### Draft collaborator paragraph

> I am setting up a numerical test of whether conservation of your Γ^± (Appendix B, eqs B25 and B32) changes the Hermite cascade in driven KRMHD turbulence. The solver is GANDALF (JAX, Fourier–Hermite, 128³ with M = 128), which evolves a single compressive field g with a (1 − 1/Λ) g_0 coupling in the g_1 equation; I read this as your G^+ with Λ^+ = √5 at β_i = τ = Z = 1. I have checked symbolically and against the solver's right-hand side that Σ_m √(m+1) g_m g_{m+1} − g_0 g_1/Λ is exactly conserved at finite M under zero truncation, with no boundary term, and that it is the in-phase part of the same neighbour correlator whose quadrature part is the Hermite flux. The plan is to inject Γ^± in a controlled way by forcing g_0 and g_1 with a fixed relative phase, at equal free-energy injection, and compare W(m), the forward and backward Hermite fluxes and the Γ^± spectrum between opposite signs of injection and a zero-injection control. The balanced base state has ⟨Γ^±⟩ = 0 by the z → −z, v∥ → −v∥ symmetry, so the forcing breaks that symmetry by hand. Two questions I would value your view on: whether you expect Γ^± or H_ph-sp to be the more telling diagnostic of a velocity-space bottleneck, given that H_ph-sp is nonlocal in Hermite space; and whether the identification of GANDALF's single-Λ hierarchy with one of your G^± branches is the right reading.

## Answered

1. *Where are the ν-scan checkpoints, and which configs produced them?* — Local: `studies/02-collisionality-scan/data/hermite128_nu{1,3,5,10,20,50}_imex/checkpoints/checkpoint_t{2180,2190,2200}.0.h5` (float32, g shape [128,128,65,129]). Same labels on Modal volume `krmhd-benchmark-vol`. Produced by `studies/02-collisionality-scan/scripts/modal_128_hermite.py`, resumed from `alfven128_lowkz_f0p02_eta100/checkpoints/checkpoint_t2000.0.h5`, not from a YAML config. Only ν = 3 has `spectra/` and `diagnostics_timeseries.npz` locally. (Answered from the repo, 2026-10-01.)
2. *Which Λ did the ν-scan use, Λ⁺ or Λ⁻?* — Λ = 2.2360677 = +√5 = Λ⁺ of CMM (B7) at β_i = τ = Z = 1 (checkpoint attr `state/Lambda`; D01 C1–C3). Λ⁻ = −√5 has never been run. (Answered from the repo, 2026-10-01.)
