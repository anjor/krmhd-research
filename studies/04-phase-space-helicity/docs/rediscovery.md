# Blind rediscovery test of the second KRMHD invariant

Phase 0, task 2. Run on 1 October 2026. Protocol from `PLAN.md` §4: a fresh model context was given the collisionless g_m equations and a quadratic ansatz, with no access to the repository, the study documents, or the Chandran, Mallet & Meyrand paper, and asked to find all conserved combinations. An independent critic context, given only the equations and the claimed coefficients, tried to falsify them on random spectral states. Only after both had reported was the result compared with the paper's Appendix B.

## What the blind context was given

The hierarchy of `SPEC.md` §1 (g_0, g_1 with the c = (1 − 1/Λ)/√2 coupling, g_m for m ≥ 2, g_{M+1} = 0), the definition of the Poisson bracket and ∇∥ = ∂_z + {Ψ, ·}, the antisymmetry identities, and the ansatz Q = ∫ [Σ a_m Re(g_{m+1} g_m*) + Σ b_m |g_m|²]. Nothing about helicity, the flux, or the paper.

## What it found

Exactly two invariants, for every M ≥ 2 and every real c (SymPy nullspace of the condition PA = AᵀP, dimension 2 at M = 2 … 12, symbolic c and c ∈ {0, 1/√2, −3/7}):

- Q1: b_0 = √2 c = 1 − 1/Λ, b_m = 1 (m ≥ 1), a_m = 0. The free energy with the Λ weight on g_0.
- Q2: a_0 = 2c = √2 (1 − 1/Λ), a_m = √(2(m+1)) (m ≥ 1), b_m = 0.

It gave the reduction argument (the {Φ, ·} term conserves every bilinear for every P; the streaming and {Ψ, ·} terms reduce to the single-k condition PA symmetric), the two recursions b_{m+1} = b_m, p_{m+1}/p_m = √((m+2)/(m+1)) for m ≥ 1 with the m = 0 boundary conditions b_0 = √2 c b_1 and p_0 = c p_1, and three things that were not asked for:

1. The physical identification: for c = 1/√2 (Λ → ∞) Q2 = g†Ag with A the matrix of v∥ in the orthonormal Hermite basis, so Q2 is the Hermite image of ∫ v∥ δf² dv.
2. Zero truncation is exact for both (no boundary term): the condition only involves A_{m,m+1} for m ≤ M − 1.
3. Under the copy closure g_{M+1} = g_{M−1}, Q2 is unchanged and still exact, while Q1 survives only with b_M = √M/(√M + √(M+1)).

Item 3 corrected a statement in the first draft of `SPEC.md` §2, which had both invariants broken by the copy closure. Script: `derivations/blind_invariants.py`, all checks pass (symbolic at M = 6, numerical to 1e-18 at M = 6 and 20 with random antisymmetric operators standing in for the brackets).

## Critic verdict

`derivations/critic_invariants.py`: 16³ pseudo-spectral grid, fields band-limited to |n| ≤ 4 so cubic products are alias-free, float64, 90 cases (M ∈ {5, 9}, Λ ∈ {+√5, −√5, 3}, three seeds, Φ and Ψ on or off, three closures). Worst-case relative residuals: W under zero truncation 1.4e-17, Γ under zero truncation 7.5e-18, Γ under copy closure 8.0e-18, W under copy closure with the modified last weight 1.3e-17. Negative controls (unweighted energy, Γ with p_0 = √2) at 1e-3, fourteen orders of magnitude above. Under a generic closure g_{M+1} = 2.7 g_{M−1} + 0.3 g_M neither invariant survives. Verdict on every claim: SUPPORTED, none falsified. The critic did not test uniqueness (claim C4), only the two named counter-examples.

## Comparison with Chandran, Mallet & Meyrand 2026, Appendix B

Read only after the above. Their (B4) is the kinetic equation for the two compressive fields G^± with Λ^± of (B7); at β_i = τ = Z = 1, Λ^± = ±√5. Expanded in Hermite polynomials with the √(2^m m!) normalisation (B27–B30), G̃^±_m obeys the same ladder as GANDALF's g_m with the same (1 − 1/Λ) coupling at m = 1, so GANDALF's single hierarchy with Λ = +√5 is their G^+ branch, and with Λ = −√5 it would be G^−. Each G^± is a fixed mixture of the density-like and δB∥-like kinetic fields (B5); GANDALF evolves one such mixture per run (claim C6).

- Their energy (B31), Σ G̃_m² − G̃_0²/Λ, is Q1 term for term.
- Their additional invariant (B25), (B32), Γ^± = (v_th/√2) ∫ [Σ √(m+1) G̃_m G̃_{m+1} − G̃_0 G̃_1/Λ], is Q2 term for term: the m = 0 coefficient is 1 − 1/Λ and the m ≥ 1 coefficients are √(m+1), ratio √2 to ours for all m (`derivations/01_invariant_mapping.py` Part C).
- Their (B25) identifies Γ^± as ∫ v∥ (G^±)²/(2F_M) dv − M_0 M_1/Λ^±, the first v∥ moment of the free-energy density with a correction from the Λ coupling. This is the blind context's item 1 with the finite-Λ correction.
- They do not discuss the truncated hierarchy or closures. Items 2 and 3 are new.

What the blind context could not have found: the paper's headline phase-space helicity H_ph-sp (B17–B19), P ∫ (G^±)²/(2 v∥ F_M) dv, has a 1/v∥ weight whose Hermite matrix is dense. It is outside the nearest-neighbour ansatz by construction. The test was therefore scoped to Γ^±, and found it in full. Whether the study should also measure H_ph-sp is Gate 1 question 2 in `QUESTIONS.md`.

Their closing paragraph of Appendix B is a prediction about this Γ^±: because it is not sign-definite, the Fjørtoft argument does not apply and they expect Γ^± and W to cascade forward together (claim C8). Their §5 names "whether gyrokinetic helicity conservation leads to a cascade bottleneck in velocity space" as open. That is this study's question.

## Methodology result

A model context with the equations and a quadratic ansatz, and nothing else, recovered the paper's Γ^± with the exact Λ-dependent coefficient, gave the physical interpretation, and added two closure results the paper does not contain. An independent context confirmed all of it to round-off and found a counter-example closure. The cost was two short agent runs. The limitation is the ansatz: the result is only as general as the form the context was told to search, and the 1/v∥ invariant was invisible to it.
