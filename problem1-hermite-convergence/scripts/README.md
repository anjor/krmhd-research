# Problem 1 Execution Scripts - Hermite Convergence Study

Scripts for executing and analyzing the Hermite convergence parameter survey based on Anjor's direction (2026-02-22).

## Phase 1: Coarse Survey

**Objective**: M ∈ {8, 16, 32} × ν ∈ {0.001, 0.01, 0.1} — 9 runs, ~12 GPU-hours

### Execution Scripts

1. **`launch_phase1.sh`** — Main launcher
   ```bash
   ./launch_phase1.sh
   ```
   - Checks GANDALF environment 
   - Sets up dependencies
   - Launches all Phase 1 runs

2. **`phase1_launch.py`** — Python parameter sweep engine
   - Generates 9 configuration files
   - Estimates GPU time requirements
   - Launches simulations sequentially
   - Creates monitoring infrastructure

3. **`monitor_phase1.py`** (auto-generated) — Progress monitoring
   ```bash
   python scripts/monitor_phase1.py
   ```
   - Checks completion status
   - Generates preliminary convergence plots
   - Updates as runs complete

### Analysis Scripts

4. **`analyze_phase1.py`** — Comprehensive analysis (post-completion)
   ```bash
   python scripts/analyze_phase1.py
   ```
   - Extracts convergence data (E_M/E_total)
   - Validates with secondary metrics (E(k⊥), Q, σ_c)
   - Fits M_crit(ν) scaling law
   - Generates publication-quality plots
   - Creates detailed analysis report

## Key Physics Goals

Based on Anjor's direction:

1. **Convergence threshold**: 10⁻³ (empirical discovery)
2. **Validation metrics**: Track all three, let data reveal sensitivity
3. **ν-independent saturation**: Capture transition to collisionless dissipation
4. **Forcing**: Single ion, balanced (σ_c ≈ 0)
5. **Resolution**: 64³

## Expected Outcomes

### Primary: M_crit(ν) scaling law
- **High ν**: M_crit ~ ν^(-α) (collisional regime)
- **Low ν**: M_crit → constant (collisionless regime)  
- **Transition**: Captures different dissipation mechanisms

### Secondary: Validation metric sensitivity
- Which metric (spectrum slope, heat flux, cross-helicity) is most sensitive to convergence?
- How do they correlate with primary E_M/E_total criterion?

### Physics insight: ν-independent dissipation
- Confirms different kinetic mechanisms at low ν
- Guides multi-ion species studies (Problems 2-5)

## Files Generated

### Configurations
- `~/.openclaw/workspace-physics/configs/phase1/` — YAML configs for each run

### Results  
- `gandalf/results/problem1/phase1/` — Simulation outputs
- `results/phase1_analysis/` — Analysis plots
- `results/phase1_report.md` — Comprehensive analysis report

### Monitoring
- `gandalf/logs/phase1_*.log` — Individual run logs

## Usage Workflow

1. **Launch**: `./launch_phase1.sh`
2. **Monitor**: `python scripts/monitor_phase1.py` (periodic)
3. **Analyze**: `python scripts/analyze_phase1.py` (when complete)
4. **Interpret**: Review `phase1_report.md` and plots
5. **Plan Phase 2**: Use results to refine parameter matrix

## Compute Requirements

- **Total**: ~12 GPU-hours estimated
- **Individual runs**: 0.4-1.6 GPU-hours (scales with M)
- **Backend**: JAX Metal (Apple Silicon optimized)

## Next Steps

After Phase 1 completion:
- Use M_crit(ν) scaling to guide Phase 2 parameter selection
- Focus compute on transition regions
- Extend ν range if saturation not captured
- Validate physics with secondary metrics