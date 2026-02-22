# Problem 1: Hermite Convergence Study Plan

**Goal:** Determine M_crit(ν) — the minimum Hermite moments needed for converged turbulence statistics as a function of collisionality.

**Context:** This study will establish the computational requirements for the multi-ion species research direction, providing the foundation for Problems 2-5.

## 1. Convergence Metric Design

### Recommendation: Multi-tier convergence assessment

**Primary metric: Hermite moment energy fraction**
- Definition: `E_M / E_total < threshold` where E_M is energy in highest moment
- Threshold: `1e-3` (established pattern from `hermite_convergence.py`)
- Physics rationale: Measures actual moment truncation error directly

**Secondary metrics for validation:**
- **Energy spectrum E(k⊥)**: k⊥^(-5/3) slope in inertial range
- **Heat flux Q**: Parallel electron heat conduction
- **Cross-helicity σ_c**: z⁺/z⁻ energy ratio (measures Alfvénic balance)

**Trade-off analysis:**
- **E_M/E_total**: 
  - ✅ Direct measure of truncation error
  - ✅ Cheap to compute during runs
  - ✅ Physics-agnostic (pure numerical)
  - ⚠️ May not detect subtle changes in turbulent transport
  
- **E(k⊥) slope**: 
  - ✅ Measures cascade physics directly
  - ⚠️ Requires spectral analysis (more expensive)
  - ⚠️ May be noisy in finite simulations
  
- **Heat flux Q**:
  - ✅ Astrophysically relevant quantity
  - ✅ Sensitive to kinetic physics
  - ⚠️ Harder to interpret convergence threshold
  - ⚠️ May require longer statistical averaging

**Implementation**: Use E_M/E_total as primary gate, validate with secondary metrics on subset of runs.

## 2. Parameter Range Design

### Hermite moment values (M)
**Recommended range: M ∈ {8, 16, 24, 32, 48, 64}**

**Rationale:**
- **M = 8**: Baseline (likely insufficient for weakly collisional)
- **M = 16-24**: Expected convergence for moderate collisionality
- **M = 32-48**: High-resolution test cases  
- **M = 64**: Upper bound (computational limit check)

**Trade-off:** 6 M values provide good coverage without excessive compute cost.

### Collisionality values (ν)
**Recommended range: ν ∈ {0.001, 0.003, 0.01, 0.03, 0.1, 0.3}**

**Physics reasoning:**
- **ν = 0.001**: Weakly collisional (expect high M_crit, approaching collisionless limit)
- **ν = 0.01**: Moderate (thesis baseline)
- **ν = 0.1**: Strongly collisional (expect low M_crit) 
- **ν = 0.3**: Very strong (M_crit should saturate at high-ν)

**Critical physics**: Anjor expects **ν-independent dissipation** at low ν. The range must capture the transition from:
- **High ν**: Collisional regime (Landau damping dominant, M_crit ~ ν^(-α))
- **Low ν**: Collisionless regime (other kinetic effects, M_crit saturates)

**Coverage:** Factor-of-3 spacing provides good logarithmic coverage. May need to extend to lower ν (e.g., 0.0003) if saturation isn't reached at ν = 0.001.

**Expected scaling**: 
- **High ν**: M_crit ~ ν^(-α) where α ≈ 0.3-0.5 based on collision damping γ_m ~ ν m²
- **Low ν**: M_crit → constant (collisionless saturation)

## 3. Simulation Setup Design

### Base resolution: 64³
**Justification:**
- Captures turbulent cascade k⊥ ∈ [1, 20] (inertial range)
- Computationally manageable for parameter survey
- Matches successful benchmarks in existing examples

### Box size: L = 1.0 (unit cube)
**Reasoning:** 
- Standard GANDALF normalization
- k_min = 2π (drives large-scale dynamics)
- k_max ≈ 42 (with 2/3 dealiasing)

### Forcing configuration
**Balanced forcing**: σ_c ≈ 0 (equal z⁺, z⁻ injection)
- k_force = [1, 2] (large-scale energy injection)  
- ε_force = 1.0 (normalized energy injection rate)
- **Rationale**: Balanced turbulence tests both wave directions equally

### Runtime strategy
**Two phases:**
1. **Spin-up**: 2 τ_A (develop turbulent cascade)
2. **Statistics**: 3 τ_A (steady-state sampling)
3. **Total**: 5 τ_A per run

**Time step**: dt = 0.005 (CFL constraint for given resolution)

## 4. Compute Budget Analysis

### Cost estimates (per run)
**Base case** (M=16, 64³, 5 τ_A):
- ~0.8 GPU-hours based on existing benchmarks
- Scales as M × N³ × T

**Total parameter matrix**: 6 M values × 6 ν values = 36 runs
**Base cost**: 36 × 0.8 = 29 GPU-hours

**Scaling by M**:
- M=8: 0.4 GPU-h/run
- M=16: 0.8 GPU-h/run  
- M=32: 1.6 GPU-h/run
- M=64: 3.2 GPU-h/run

**Estimated total**: ~65 GPU-hours (within 10³ target with margin)

### Staged execution strategy
**Phase 1** (25% budget): Coarse survey
- M ∈ {8, 16, 32} × ν ∈ {0.001, 0.01, 0.1}
- 9 runs, ~12 GPU-hours
- Identify rough M_crit(ν) scaling

**Phase 2** (75% budget): Detailed convergence
- Fill in parameter matrix around identified boundaries
- Focus compute on transition region
- ~53 GPU-hours remaining

## 5. Hypercollision Order Selection

### Recommendation: hyper_n = 2 (standard quadratic)

**Trade-off analysis:**
- **hyper_n = 1** (linear):
  - ✅ Gentle damping, preserves more physics
  - ⚠️ May allow pile-up at high-m
  - ⚠️ Less selective damping

- **hyper_n = 2** (quadratic):
  - ✅ **Standard choice in original GANDALF**
  - ✅ Good balance: selective but not too steep
  - ✅ Well-tested physics preservation
  - ⚠️ None identified

- **hyper_n = 4,6** (high-order):
  - ✅ Very selective (preserves low-m physics)
  - ⚠️ May create artificial barriers
  - ⚠️ Less tested in turbulent context
  - ⚠️ Potential numerical instabilities

**Physics reasoning**: 
- Collision operator γ_m ~ ν m² suggests natural quadratic scaling
- Matches Landau damping dispersion ω_d ~ k∥²v_th²
- Preserves established GANDALF physics fidelity

## 6. Analysis Plan

### Real-time monitoring
- Track E_M/E_total every 0.1 τ_A
- Alert if convergence degrades during run
- Early termination if E_M/E_total > 0.1 (divergence)

### Post-processing pipeline
1. **Convergence analysis**: Fit M_crit(ν) scaling law
2. **Physics validation**: Compare energy spectra, heat flux
3. **Computational metrics**: GPU-hours vs. M, accuracy trade-offs
4. **Visualization**: Convergence phase diagram (M vs ν)

### Deliverables
- **Database**: HDF5 with all runs, diagnostics, metadata
- **Analysis notebook**: Interactive convergence analysis
- **Figure set**: Publication-quality convergence plots
- **Scaling formula**: M_crit(ν) = A × ν^(-α) + M_min

## 7. Risk Assessment & Contingencies

### Identified risks
1. **Insufficient M range**: M_crit may exceed M=64
   - **Mitigation**: Start with coarse survey, extend range if needed
   
2. **Turbulence non-stationarity**: Statistics may not converge in 3 τ_A
   - **Mitigation**: Monitor energy history, extend runtime for outliers
   
3. **Resolution effects**: 64³ may be insufficient for accurate physics
   - **Mitigation**: Cross-check subset at 96³ resolution

### Physics uncertainties
1. **Closure sensitivity**: Results may depend on closure_zero vs closure_symmetric
   - **Investigation**: Compare closures for converged cases
   
2. **Forcing effects**: Balanced vs imbalanced forcing may affect M_crit
   - **Investigation**: Test σ_c = ±0.5 for few parameter points

## 8. Anjor's Direction (Updated 2026-02-22)

**Resolved decisions:**

1. **Convergence threshold**: Start with **10⁻³**, discover empirically. The threshold itself is part of what we're learning — that's good science.

2. **Validation metrics**: **Track all three** (E(k⊥) slope, Q, σ_c) and let the data reveal which is most sensitive. We don't know a priori which will be most informative.

3. **High-ν saturation**: **YES** — Anjor expects dissipation to become **ν-independent at low ν**. This is **key physics**: collisionless regime has different dissipation mechanism than collisional regime. The ν range must capture this transition.

4. **Forcing**: **Single ion, balanced forcing only.** No imbalanced cases (σ_c ≠ 0) for this study.

5. **Resolution**: **64³ as planned** (no explicit concern raised).

**Updated physics focus:**
- The transition to ν-independent dissipation at low ν is a **critical physical signature**
- This represents the transition from collisional (Landau damping) to collisionless (other kinetic effects) dissipation
- Parameter survey must extend to low enough ν to capture this saturation

**Scientific philosophy:**
- Unknown convergence thresholds and validation metrics are **part of the discovery** — we're mapping the numerical-physical phase space
- Empirical determination of these quantities is legitimate and valuable science

---

## 9. Execution Status (Updated 2026-02-22)

**✅ PHASE 1 READY FOR LAUNCH**

### Scripts Created (in `~/.openclaw/workspace-physics/scripts/`)
1. **`launch_phase1.sh`** — Main execution launcher
2. **`phase1_launch.py`** — Parameter sweep engine (9 configurations)
3. **`monitor_phase1.py`** — Progress monitoring (auto-generated)
4. **`analyze_phase1.py`** — Comprehensive post-analysis
5. **`README.md`** — Complete documentation

### Execution Command
```bash
cd ~/.openclaw/workspace-physics
./scripts/launch_phase1.sh
```

### Expected Phase 1 Output
- **9 simulation runs**: M ∈ {8, 16, 32} × ν ∈ {0.001, 0.01, 0.1}
- **~12 GPU-hours**: Individual runs 0.4-1.6 hours
- **M_crit(ν) scaling law**: Capture ν-independent saturation
- **Validation metrics**: Secondary convergence indicators

### Key Physics to Validate
- **Low ν saturation**: M_crit → constant (collisionless regime)
- **High ν scaling**: M_crit ~ ν^(-α) (collisional regime)  
- **Transition physics**: Different dissipation mechanisms

**Next steps**: 
1. ✅ Execute Phase 1: `./scripts/launch_phase1.sh`
2. Monitor progress: `python scripts/monitor_phase1.py`
3. Analyze results: `python scripts/analyze_phase1.py`
4. Design Phase 2 based on M_crit(ν) findings

**Estimated timeline**: 1-2 days execution + 0.5 days analysis = Phase 1 complete in ~3 days.