# Study 04 — SPEC

Equations, conventions, defaults and tolerances. Written 1 October 2026 from the installed GANDALF (`krmhd`, v0.5.0 IMEX path), the Study 2 ν-scan runner `studies/02-collisionality-scan/scripts/modal_128_hermite.py`, and the Study 2 paper `paper/dissipative-anomaly/main.tex` §2. Every statement here was read from code or checkpoint metadata, not from memory. Items marked **[D01]** are claims that `derivations/01_invariant_mapping.py` must prove before they are used.

## 1. Equations in GANDALF variables

GANDALF evolves, in Fourier space on a triply periodic box, the Elsasser potentials z⁺, z⁻ and M+1 Hermite moments g_m, m = 0…M, of one compressive kinetic field along v∥. In KRMHD the compressive sector is two decoupled hierarchies G^± (Schekochihin et al. 2009; CMM 2026 App. B, eqs. B4–B7), each a fixed linear combination of the density-like and δB∥-like v⊥-moments of the ion distribution (B5), each with its own coupling constant Λ^±. GANDALF's g with a given Λ is G^σ for the σ with Λ^σ = Λ. Its zeroth moment is therefore the corresponding mixture of δn/n and δB∥/B, not δn alone (the `g0_rhs` docstring's "δn_e/n₀" is loose). δn/n and δB∥/B are recovered from the zeroth moments of both hierarchies, so a single GANDALF run carries one of the two; the other needs a run at the other Λ. As of GANDALF v0.6.0 (gandalf#149, this repo PR #9) there is no separate B∥ field.

Potentials: Φ = (z⁺ + z⁻)/2 (stream function), Ψ = (z⁺ − z⁻)/2 (parallel vector potential A∥).

Poisson bracket: {f, h} = ∂_x f ∂_y h − ∂_y f ∂_x h, computed pseudo-spectrally with 2/3 dealiasing.

Parallel gradient along the perturbed field line: ∇∥ = ∂_z + {Ψ, ·}.

Alfvénic sector (`physics.z_plus_rhs`, `z_minus_rhs`): the energy-conserving GANDALF form of RMHD with τ_A = L_z/v_A = 1. z± do not depend on g. Integrating-factor RK2 midpoint, linear term ±ik_z z±.

Hermite sector (`physics.g0_rhs`, `g1_rhs`, `gm_rhs`, thesis Eqs. 2.7–2.9), with s ≡ √β_i and c_Λ ≡ (1 − 1/Λ)/√2:

    ∂_t g_0 = −{Φ, g_0} − s ∇∥ ( g_1/√2 )
    ∂_t g_1 = −{Φ, g_1} − s ∇∥ ( g_2 + c_Λ g_0 )
    ∂_t g_m = −{Φ, g_m} − s ∇∥ ( √((m+1)/2) g_{m+1} + √(m/2) g_{m−1} ),   2 ≤ m ≤ M
    g_{M+1} = 0   (zero-truncation closure, hardcoded in gm_rhs)

Collisions and resistivity are not in these RHS functions. They are applied by the timestepper (§3). The k = 0 mode of every RHS is zeroed (`zero_k0_mode`).

Compact form. Write g = (g_0, …, g_M) at one k. Then ∂_t g = −{Φ, g} − s ∇∥ (A g) with A the real (M+1)×(M+1) matrix

    A_{01} = 1/√2,  A_{10} = c_Λ,  A_{m,m+1} = √((m+1)/2) (m ≥ 1),  A_{m,m−1} = √(m/2) (m ≥ 2),  zero elsewhere.

A is symmetric for m ≥ 1 and asymmetric only in the (0,1) block. Both ∂_z and {Ψ, ·} are antisymmetric under the volume integral. {Φ, ·} is a common incompressible advection of every moment.

Hermite basis: orthonormal Hermite functions ψ_m(v) = N_m H_m(v) e^{−v²/2}, N_m = (2^m m! √π)^{−1/2}, v = v∥/v_th (`hermite.py`). So ∫ g² dv = Σ_m g_m² and Σ_k |g_m(k)|² is the free energy in moment m up to the g_0 weight in §2.

Parameters: β_i = 1, v_th = 1, v_A = 1, τ = T_i/T_e = 1, Z = 1. Λ enters only through c_Λ. Λ± = −τ/Z + 1/β_i ± √((1 + τ/Z)² + 1/β_i²) = ±√5 at these parameters. The ν-scan used Λ = +√5 = 2.2360677 (checkpoint attr `state/Lambda`), so c_Λ = (1 − 1/√5)/√2 = 0.39100. For Λ⁻ = −√5, c_Λ = (1 + 1/√5)/√2 = 1.02315. Λ = 1 gives c_Λ = 0 and decouples g_0 from the hierarchy (the Study 2 lost run). **The Λ sign selects which of the two decoupled compressive fields G^± is evolved; it does not set the sign of the invariant (D01 Part C).**

## 2. Quadratic invariants of the collisionless, unforced, ideal system **[D01]**

With ν = η = 0, no forcing, and the zero-truncation closure, the following are exact invariants at every M (claims to prove in D01, symbolically from the matrix A and numerically from `krmhd.physics` RHS evaluations in float64):

Free energy

    W = Σ_{k≠0} [ (1 − 1/Λ) |g_0|² + Σ_{m=1}^{M} |g_m|² ]

Note the weight (1 − 1/Λ) on g_0. GANDALF's `hermite_moment_energy` returns the unweighted Σ_k |g_m|² per m; W as defined here is the quantity the collisionless dynamics conserve. The Study 2 ε_ν used m ≥ 2 only and is unaffected.

Neighbour correlator invariant (the "Γ" of PLAN.md §3)

    Γ = Σ_{k≠0} Σ_{m=0}^{M−1} c_m Re[ g_{m+1}(k) g_m*(k) ],   c_0 = √2 (1 − 1/Λ),   c_m = √(2(m+1)) for m ≥ 1

Normalisation chosen so that c_m for m ≥ 1 equals the prefactor in `krmhd.diagnostics.hermite_flux`, Π_m(k) = −k∥ √(2(m+1)) Im[g_{m+1} g_m*]. Then Γ_m and Π_m/(−k∥) are the real and imaginary parts of one complex correlator C_m(k) = c_m g_{m+1} g_m*.

Expected structure of the proof: a quadratic form g†Pg is conserved by ∂_t g = −s D (A g) with D antisymmetric iff PA is symmetric. P = diag((1 − 1/Λ), 1, …, 1) gives W. P tridiagonal with off-diagonal p_m gives p_{m+1}/p_m = A_{m+1,m}/A_{m+1,m+2} = √((m+1)/(m+2)) for m ≥ 1 and p_0 = c_Λ p_1, which is Γ. The {Φ, ·} advection conserves every bilinear ∫ g_m g_n separately. The {Ψ, ·} term has the same ladder and antisymmetry as ∂_z. Zero truncation removes row and column M+1 and leaves the symmetry condition intact, so **no boundary term at m = M under zero truncation**. The copy closure g_{M+1} = g_{M−1} changes A_{M,M−1}. The Γ condition never involves A_{M,M−1}, so **Γ is exactly conserved under both closures**; W is conserved under the copy closure only if the m = M weight is changed from 1 to √M/(√M + √(M+1)), otherwise it has a boundary term on the (M−1, M) block (D01, blind rediscovery test).

Physical identification (blind rediscovery test, D01): A is the matrix of multiplication by v∥ in the orthonormal Hermite basis, so for Λ → ∞ Γ = Σ_k g†Ag is the Hermite image of ∫ dv∥ v∥ g². Γ is the first v∥ moment of the free-energy density; finite Λ only changes the m = 0 weight.

Reference profiles (`derivations/02_reference_profiles.py`, `reference_profiles.npz`): (a) linear phase mixing at one k with white-noise forcing on g_0: Γ_m = 0 exactly for every m including m = 0 (parity of the propagator), W(m) ∝ m^{−0.49}, constant Π; (b) an echo-type stochastic model with Kraichnan white-in-time advection and the antisymmetrised Hermite velocity derivative (the literal Adkins–Schekochihin 3.10 is unstable as a truncated system, claim C11): W(m) ∝ m^{−1.0}, Π⁻/Π⁺ = 0.24, and both the k-summed Γ(m) and its k-odd part indistinguishable from zero. Consequence for the experiment: neither reference produces Γ; only the symmetry-breaking pair forcing of Phase 2 does.

## 3. Dissipation and time stepping (as applied by `timestepping.gandalf_step`, scheme `imex_rk222`)

- Hyper-collisions: rate ν (m/M)^n with n = `hyper_n` = 6, applied for m ≥ 2 only; m = 0, 1 exempt. Folded into the implicit operator L on the Hermite axis together with linear streaming (ARS(2,2,2), per-k_z batched LU). Unconditionally stable.
- Resistivity: factor exp(−η (k⊥²/k⊥,max²)^r dt) with η = 100, r = `hyper_r` = 2, k⊥,max from the 2/3 dealias index, applied after each step to z⁺, z⁻ **and to every g_m**. So g has a resistive sink at high k⊥ in addition to the collisional sink. Both must appear in the W and Γ budgets.
- Nonlinear brackets explicit. Poisson brackets dealiased individually; g is re-masked after each step.
- Timestep: fixed, `compute_cfl_timestep(state, v_A, cfl_safety=0.3)` evaluated once after the checkpoint load. ν-scan value dt = 2.34 × 10⁻³ τ_A at 128³.
- Scheme is a per-call argument. Every loop must pass `scheme="imex_rk222"` explicitly.
- Precision: production checkpoints are float32. Conservation tests in Phase 1 run with `jax_enable_x64` so that residuals are limited by the scheme, not by the dtype.

## 4. Domain and run classes

| Class | Grid | M | ν | η | Box | Where | Purpose |
|---|---|---|---|---|---|---|---|
| Conservation | 32³ | 32, 64 | 0 | 0 | L = 1 | Mac, `uv run` | Gate 2 residuals vs dt and M |
| Forcing test | 32³ | 64 | 3 | 100 | L = 1 | Mac | Gate 3 |
| Checkpoint analysis | 128³ | 128 | 1–50 | 100 | L = 1 | Mac, existing data | Gate 2 budgets, base-state Γ; with the v0.5.0 checkpoints a pipeline test only (decision 5) |
| Base state | 128³ | 128 (Hermite branch) | 3 (Hermite branch) | 100 | L = 1 | Modal, the agent | decision 5: the Alfvénic base state regenerated to saturation on the pinned GANDALF with the forcing recalibrated, then the ν = 3 Hermite branch that set A starts from |
| Set A | 128³ | 128 | 3 | 100 | L = 1 | Modal, the agent | Gate 4 |
| Set B | 128³ | 64, 256 | two values | 100 | L = 1 | Modal, the agent | convergence |

Box: L_x = L_y = L_z = 1 (checkpoint attrs). k_⊥ and k_z in units of 2π/L. τ_A = 1. Local runs must stay under 20 minutes wall time. Modal runs are launched by the agent through `loop/modal_launch.py` only (decision 1, 1 October 2026).

## 5. Forcing

Alfvénic (`krmhd.forcing.force_alfven_modes_balanced`): balanced white-noise Elsasser forcing, amplitude f = 0.02, n_⊥ ∈ [1, 2], |n_z| = 1 (n_z = 0 excluded), correlation 0. Applied before the step every step.

Hermite base drive (inlined `perp_lowkz` forcing in the ν-scan runner, same as `shared/hermite_forcing.py`): Gaussian white noise, amplitude f_H = 0.0035, scaled by 1/√dt, same k-mask as the Alfvénic drive, added to g_0 only, before the step. The rfft reality conditions are enforced on the k_x = 0 and Nyquist planes; k = 0 zeroed.

Phase 2 pair forcing (`shared/hermite_pair_forcing.py`, to be written): one noise realisation f_0 per step on the same mask, added to g_0 and to g_1 as f_1 = r e^{iφ} f_0. φ ∈ {0, π/2, π}. r is a free amplitude ratio fixed by the Gate 3 test. ε_W and ε_Γ are measured from the state response: ε_X = [X(after forcing) − X(before forcing)]/dt per step, accumulated. They are not inferred from the forcing definition.

## 6. Observables

All k-sums use the rfft weighting of `hermite_moment_energy` (k_x = 0 and Nyquist planes once, other k_x planes twice) and exclude k = 0.

| Symbol | Definition | Code |
|---|---|---|
| W(m) | Σ_k |g_m|² (unweighted) | `krmhd.diagnostics.hermite_moment_energy` |
| W | §2, with (1 − 1/Λ) on m = 0 | `analysis/helicity.py` |
| E(k⊥) | perpendicular energy spectrum of z± | `energy_spectrum_perpendicular` |
| Π_m(k) | −k∥ √(2(m+1)) Im[g_{m+1} g_m*] | `krmhd.diagnostics.hermite_flux` |
| Π(m), Π±(m) | Σ_k Π_m(k); forward (Π_m > 0) and backward (Π_m < 0) parts summed separately | `analysis/helicity.py` |
| Γ_m(k) | c_m Re[g_{m+1} g_m*] | `analysis/helicity.py: hermite_helicity` |
| Γ(m), Γ(k⊥), Γ | reductions of Γ_m(k) | same |
| ε_ν | 2ν Σ_{m≥2} (m/M)⁶ W(m) | as in Study 2 |
| ε_η^g | resistive sink of W: Σ_k 2η (k⊥²/k⊥,max²)^r × (weighted |g_m|²) | `analysis/budget.py` |
| ε_W, ε_Γ | measured injection rates, §5 | `analysis/budget.py` |
| T_M | truncation term in the Γ budget at m = M, if any | `analysis/budget.py` |

Budgets: dW/dt = ε_W − ε_ν − ε_η^g + R_W, dΓ/dt = ε_Γ − (collisional Γ sink) − (resistive Γ sink) − T_M + R_Γ. The residuals R are the gate quantities.

## 7. Tolerances

Gate 2 (conservation and budgets)
- Unforced, ν = η = 0, seeded g, 32³: relative residual r_X = |X(t) − X(0)| / X_scale over 10 τ_A, with W_scale = W(0) and Γ_scale = Σ_k Σ_m |c_m| |g_{m+1}| |g_m| (so a zero Γ does not give 0/0). r_X must fall as dt² (IMEX-RK222 is second order) over at least a factor 4 in dt, and must not grow with M between 32 and 64.
- Sign test: flipping the sign of g_1 in the initial condition flips the sign of Γ to within r_Γ.
- Checkpoint budgets: |R_W| ≤ 5 % of ε_W and |R_Γ| ≤ 5 % of max(|ε_Γ|, collisional Γ sink) over the 100 τ_A averaging window.

Gate 3 (forcing)
- At φ = 0 and φ = π with the same noise seed: measured ε_W equal within 5 %, measured ε_Γ equal and opposite within 5 %. At φ = π/2, |ε_Γ| ≤ 5 % of the φ = 0 value.

Gate 4 (experiment)
- Effect = difference of a measured quantity between φ = 0 and φ = π runs. Scatter = standard deviation across the two φ = π/2 seeds and across 50 τ_A sub-windows. Effect must exceed 2 × scatter. Anjor sets the convergence criteria for set B.

Study-wide (from repo CLAUDE.md): energy balance within 5 %, total energy fluctuations < 10 % of mean over the final 50 τ_A, E(k⊥) shows an inertial range.

## 7a. Gate 4 criteria per run set

Added 1 October 2026 (decision 3, `LOOP.md`). Before set A or set B is launched, the agent writes here, in a subsection `### 7a.<set>`, the quantities, the averaging windows and the criteria that will judge that set, consistent with §7, and puts the same criteria in code in `analysis/gate4_<set>.py`, with a critic pass. For set B this includes the convergence criteria. The new base state (decision 5) is accepted the same way: its criteria go in `### 7a.base` and its evaluation in `analysis/gate_base.py` (`LOOP.md` §5). The launch freezes the subsection and the code (`modal_launch.py launch --freeze`). After the launch, changing them is a hard stop.

## 8. Base-state inventory (Study 2 ν-scan, IMEX, all from the same t₀ = 2000 τ_A Alfvénic checkpoint)

| ν | Local path (`studies/02-collisionality-scan/data/`) | Checkpoints | Spectra | ε_ν |
|---|---|---|---|---|
| 1 | `hermite128_nu1_imex/` | t = 2180, 2190, 2200 | no | 48.4 |
| 3 | `hermite128_nu3_imex/` | t = 2180, 2190, 2200 | yes, plus `diagnostics_timeseries.npz` | 49.2 |
| 5 | `hermite128_nu5_imex/` | t = 2180, 2190, 2200 | no | 49.3 |
| 10 | `hermite128_nu10_imex/` | t = 2180, 2190, 2200 | no | 49.2 |
| 20 | `hermite128_nu20_imex/` | t = 2180, 2190, 2200 | no | 49.2 |
| 50 | `hermite128_nu50_imex/` | t = 2180, 2190, 2200 | no | 49.2 |

Same labels on Modal volume `krmhd-benchmark-vol`, each with `checkpoints/` every 10 τ_A and `spectra/` every 500 steps in the averaging window. Fetch with `studies/02-collisionality-scan/scripts/download_128_results.py --only <label> --spectra-only`. Checkpoint schema: `grid` attrs (L, N), `metadata` attrs (ν, η, orders, forcing, scheme, step), `state` attrs (Λ, M, β_i, ν, v_th, time), datasets `state/{z_plus,z_minus,g}_{real,imag}` float32 plus legacy `state/B_parallel_{real,imag}` (ignored by GANDALF ≥ 0.6.0 `load_checkpoint`), g shape [128, 128, 65, 129].

## 9. Known gaps and deviations

1. ~~Only zero truncation is selectable in `gm_rhs`.~~ Resolved in GANDALF v0.6.0 (gandalf#153, closes #155): `closure="zero" | "symmetric"` on `gandalf_step`, `krmhd_rhs` and `PhysicsConfig`, applied consistently to the explicit RHS, the Lawson eigensystem and the IMEX operator.
2. The IMEX operator L and the explicit bracket are split; conservation residuals at finite dt are second order, not round-off. The Gate 2 criterion is convergence with dt, not machine zero.
3. g carries a resistive sink over all m. This is in GANDALF by design (both schemes). It is a term in the budgets, not an error.
4. Hermite forcing noise in the ν-scan runner was float32. Local tests use float64.
5. One run evolves one compressive hierarchy G^σ. Both σ need two runs (Λ = ±√5). There is no separate B∥ field in GANDALF v0.6.0.
6. The ν-scan checkpoints of §8 were made on GANDALF v0.5.0, before gandalf#144 (Elsasser nonlinearity 2× too large) and #148 (first-order z± stepper). Under decision 5 (1 October 2026) they serve only to test the Phase 1 pipeline, and no science claim rests on them. The agent regenerates the base state on the pinned GANDALF (run class "Base state" in §4) and set A starts from it.
