# Human-AI Collaboration Report: KRMHD Collisionality Scan

**Project:** Kinetic Reduced MHD Phase-Space Cascade Studies
**Study:** 02 -- Collisionality Scan
**Date:** March 2026
**Human researcher:** Anjor Kanekar (independent plasma physicist, GANDALF author)
**AI agent:** Claude Code (Anthropic), handling implementation, debugging, and simulation execution

---

## 1. Objective

The goal of Study 02 is to demonstrate the **dissipative anomaly** in kinetic reduced MHD: the prediction that collisional dissipation of Hermite moment energy $\varepsilon_\nu$ plateaus at a finite, nonzero value as the collision frequency $\nu \to 0$. This is the phase-space analogue of the classical hydrodynamic dissipative anomaly and a key prediction of KRMHD turbulence theory.

The experiment design is a 9-point scan over $\nu$ from $10^{-1}$ to $10^{-5}$ (half-decade spacing), each run using $M = 32$ Hermite moments on a $64^2 \times 32$ spectral grid, driven by Alfvenic forcing.

---

## 2. Timeline

### Phase 1: Infrastructure (days 1--2)

The AI built the complete simulation infrastructure for Study 02, following the project's config-driven conventions (`CLAUDE.md`):

- **9 production YAML configs** plus 1 dev config, parameterized over $\nu$ with all other physics held constant
- **`shared/dissipation.py`** -- computes collisional dissipation rates from Hermite moment spectra: $\varepsilon_\nu = 2\nu \sum_m m \int d^2 k_\perp\, |g_m(k_\perp)|^2$
- **`scripts/run_local.py`** -- simulation runner with extended diagnostics (energy history, spectra, Hermite moment profiles)
- **`scripts/sweep_nu.py`** -- orchestrator to run all $\nu$ values sequentially
- **`analysis/dissipation_plateau.py`** -- 4-panel analysis figure (energy time series, perpendicular spectra, $\varepsilon_\nu$ vs $\nu$, Hermite energy spectra)
- **`shared/validation.py`** and **`shared/run_utils.py`** -- physics validation gates and run utilities

All parameters live in YAML; no hardcoded physics values in scripts.

### Phase 2: GANDALF bug -- Hermite integrating factor (issue #120)

The first simulation attempts immediately blew up: the $g$ field (Hermite moments representing the perturbed distribution function) diverged within a few timesteps.

**Diagnosis by AI:** GANDALF used plain RK2 for the Hermite moment hierarchy, but the linear phase-mixing terms have purely imaginary eigenvalues (oscillatory modes). The RK2 stability function for a pure oscillation $\dot{y} = i\omega y$ is:

$$|R(i\omega\Delta t)|^2 = 1 + \frac{(\omega\Delta t)^4}{4} > 1 \quad \text{for all } \omega \neq 0$$

This is unconditionally unstable -- every oscillatory mode grows at every timestep regardless of $\Delta t$. By contrast, the Elsasser fields $z^\pm$ already used integrating factors for their oscillatory Alfven wave propagation terms.

**Resolution:** Filed issue #120 with full stability analysis. The human implemented the fix in GANDALF v0.4.0 (PR #121): eigendecomposition-based integrating factors for the Hermite streaming matrix. The same PR also fixed a collision operator double-counting bug.

### Phase 3: Parameter regime struggles

With the integrating factor fix, simulations ran without immediate blowup but failed to reach physically meaningful steady states:

- **$\eta = 0.01$ (original plan):** MHD turbulence energy grew without bound on the $64^2$ grid. The dissipation scale was far below the grid resolution.
- **$\eta = 1.0$ (increased for stability):** MHD reached steady state but the cascade was overdamped -- spectral slope of $-6.5$ instead of the expected $-5/3$ Kolmogorov-like scaling.
- **$g$ field stayed at zero:** The `initialize_random_spectrum` function sets $g = 0$, and the $g$ RHS terms $\{\Phi, g_m\}$ require $g \neq 0$ to be nonzero. Added explicit $g$ seeding as a workaround.

### Phase 4: $\Lambda = 1$ kills the cascade

The human identified that the collisional dissipation $\varepsilon_\nu$ was scaling linearly with $\nu$ (no plateau), and asked: "shouldn't the dissipative anomaly happen at low $\nu$?"

**AI investigation:** With $\Lambda = 1.0$, the factor $(1 - 1/\Lambda)$ in the $g_1$ equation vanishes, killing the $g_0 \to g_1$ coupling that drives the Hermite cascade. Without this coupling, energy cannot flow from the MHD fields into the Hermite hierarchy.

The correct value of $\Lambda$ is derived from the KRMHD dispersion relation:
$$\Lambda_\pm = -\frac{\tau}{Z} + \frac{1}{\beta_i} \pm \sqrt{\left(1 + \frac{\tau}{Z}\right)^2 + \frac{1}{\beta_i^2}}$$

For $\beta_i = 1$, $\tau = 1$, $Z = 1$: $\Lambda = \sqrt{5} \approx 2.236$.

The human suggested $\Lambda = -1$ for maximal coupling but the team settled on $\sqrt{5}$ as the physically motivated value.

### Phase 5: Modal cloud GPU infrastructure

To accelerate the iteration cycle, the AI built `infrastructure/modal_app.py` for running simulations on Modal T4 GPUs:

- **10x speedup** over laptop for production configs (233s vs ~2400s)
- **JIT compilation** also much faster on GPU (~2 min vs ~45 min on laptop)
- Automatic result upload and retrieval

This was critical for making the debugging loop tractable.

### Phase 6: GANDALF bug -- missing dealiasing (issue #122)

Even with correct $\Lambda$, $g$ still blew up at low $\nu$ after ~12,000 timesteps.

**AI diagnosis:** Deep comparison of the $z^\pm$ and $g$ code paths revealed that $g$ was missing dealiasing in two places:
1. No dealiasing mask applied after the RK2 substep (the $z^\pm$ fields get this implicitly through their dissipation step)
2. The $g$ RHS assembly lacked the defensive dealiasing that $z^\pm$ RHS had

Filed issue #122 with detailed code path comparison. The human fixed it in GANDALF (PR #123).

### Phase 7: GANDALF bug -- RK2 fundamentally unstable for advection (issue #124)

The dealiasing fix delayed blowup from step 12,000 to step 15,000 but did not eliminate it.

**AI analysis:** The problem is fundamental to RK2. For pure advection $\partial_t g = i k \cdot v\, g$, the RK2 amplification factor is:

$$|R(i\omega\Delta t)|^2 = 1 + \frac{(\omega\Delta t)^4}{4} > 1$$

The $z^\pm$ fields survive because their advection is self-limiting (they advect by each other); the $g$ field is advected by the external potential $\Phi$ from the MHD turbulence, which is a parametric drive that does not self-regulate.

Filed issue #124 recommending RK4, which has a stability region on the imaginary axis for $|\omega\Delta t| < 2.83$. The human implemented Lawson-form RK4 for Hermite moments in GANDALF v0.4.2 (PR #125), keeping the Elsasser fields on the existing midpoint scheme.

### Phase 8: Benchmark parameters and final sweep

The AI found GANDALF's own benchmark parameters (from `alfvenic_cascade_benchmark.py`) that produce verified $k_\perp^{-5/3}$ spectra:

| Parameter | Value |
|-----------|-------|
| $L_x$ | 1.0 |
| $\eta$ | 2.0 |
| hyper_r | 2 |
| hyper_n | 2 |
| forcing amplitude | 0.005 |

Updated all 9 configs to these benchmark parameters and ran the full sweep on Modal with GANDALF v0.4.2. **All 9 configurations ran to completion with zero blowups** -- the first fully stable sweep.

### Phase 9: Current status

**Result:** $\varepsilon_\nu \propto \nu$ (linear scaling, no plateau). The Hermite cascade is not yet self-sustaining from MHD turbulence. The $g$ energy decays from its initial seed rather than being replenished by the MHD cascade.

**Interpretation:** The dissipative anomaly requires the MHD turbulence to actively drive energy into the Hermite hierarchy via the coupling terms $\{z^\pm, g_m\}$. The coupling mechanism exists ($\Lambda = \sqrt{5}$) but either:
- The runs need to be longer for the cascade to establish itself
- Stronger turbulence amplitude is needed to overcome the initial transient
- Higher Hermite resolution ($M > 32$) may be required to resolve the cascade front

---

## 3. GANDALF Issues Filed

| Issue | Title | Status |
|-------|-------|--------|
| #118 | Native `compute_dissipation_rate` diagnostic + $g$ initialization | Open |
| #120 | Hermite time integration unconditionally unstable | Fixed in v0.4.0 (PR #121) |
| #122 | Missing dealiasing in RK2 step and RHS assembly for $g$ | Fixed in v0.4.2 (PR #123) |
| #124 | RK2 fundamentally unstable for Hermite advection | Fixed in v0.4.2 (PR #125) |
| #126 | RK4 returns time as JAX array instead of float | Workaround in `modal_app.py` |

The three stability/dealiasing bugs (#120, #122, #124) were all discovered through the process of trying to run the collisionality scan. They represent genuine numerical analysis insights: the mismatch between the time-integration scheme and the mathematical character of the equations (oscillatory and advective terms needing integrating factors and higher-order methods, respectively).

---

## 4. Collaboration Dynamics

### Division of labor

The collaboration followed a clear division:

- **Human** provided physics direction: identified the $\Lambda = 1$ problem, suggested parameter values, interpreted whether $\varepsilon_\nu \propto \nu$ was physical or numerical, pointed to benchmark configs, and implemented all GANDALF fixes (PRs #121, #123, #125).
- **AI** handled implementation and debugging: wrote all study infrastructure (configs, scripts, analysis, Modal app), performed stability analysis of time-integration schemes, compared code paths between $z^\pm$ and $g$ to find missing dealiasing, ran simulations autonomously, and filed detailed bug reports with reproduction cases and fix suggestions.

### What worked well

1. **Systematic bug discovery.** Each failed simulation run produced diagnostic information that the AI used to identify the next issue. The progression from integrating factor (immediate blowup) to dealiasing (blowup at step 12,000) to RK4 (blowup at step 15,000) was methodical -- each fix resolved one class of instability and revealed the next.

2. **Detailed issue reports.** The AI's bug reports included stability analysis, code path comparisons, and concrete fix suggestions, which made it straightforward for the human to implement the changes in GANDALF.

3. **Autonomous long-running work.** The AI ran multi-hour simulation sweeps on Modal while the human was away, tracking progress through git commits. This made effective use of asynchronous time.

4. **Config-driven reproducibility.** Every run was defined by a YAML config and logged to `docs/run_log.md` with its outcome. Failed runs were as valuable as successful ones because they were fully reproducible.

### What was challenging

1. **Parameter space navigation.** Finding parameters where both the MHD cascade and the Hermite cascade work simultaneously on a $64^2$ grid proved difficult. The MHD cascade needs low enough $\eta$ for an inertial range, but the Hermite cascade needs numerical stability at low $\nu$. The benchmark parameters ($\eta = 2.0$ with hyper-diffusion) were the eventual solution.

2. **Long feedback loops.** JAX JIT compilation on the laptop took ~45 minutes per config before any physics timesteps ran. This made local iteration impractical until the Modal GPU infrastructure was built (reducing JIT to ~2 minutes).

3. **Distinguishing numerical artifacts from physics.** The $g$ blowup could have been physical (the Hermite cascade developing finite-amplitude oscillations) or numerical (time-integration instability). Determining which required careful analysis of the amplification factor and comparison with the $z^\pm$ fields, which used a different integration scheme.

4. **The unsolved physics problem.** After fixing all numerical issues, the result ($\varepsilon_\nu \propto \nu$, no plateau) may simply mean that the parameter regime or run duration is not sufficient to observe the dissipative anomaly. The AI cannot determine whether this is a matter of running longer, changing parameters, or whether the $64^2$ grid is fundamentally too coarse -- that requires physics judgment.

### Key takeaway

The collaboration was most productive when operating in a tight loop: the AI runs a simulation, something fails, the AI diagnoses the failure with quantitative analysis, the human validates the diagnosis and implements the fix in GANDALF, and the cycle repeats. Three genuine solver bugs were found and fixed through this process, which is a concrete contribution to the GANDALF codebase independent of whether the dissipative anomaly is eventually observed in the simulations.

---

## 5. Artifacts Produced

### Code (committed to this repo)

| Path | Description |
|------|-------------|
| `studies/02-collisionality-scan/configs/*.yaml` | 9 production + 1 dev config for $\nu$ scan |
| `studies/02-collisionality-scan/scripts/run_local.py` | Simulation runner with extended diagnostics |
| `studies/02-collisionality-scan/scripts/sweep_nu.py` | Sweep orchestrator |
| `studies/02-collisionality-scan/analysis/dissipation_plateau.py` | 4-panel analysis figure |
| `shared/dissipation.py` | Collisional dissipation rate computation |
| `shared/validation.py` | Physics validation gates |
| `shared/run_utils.py` | Run identification and logging utilities |
| `infrastructure/modal_app.py` | Modal cloud GPU runner |

### Data (in `data/`, gitignored)

9-point sweep results from GANDALF v0.4.2 with benchmark parameters: energy time series, perpendicular spectra, and Hermite moment profiles for each $\nu$ value.

### GANDALF improvements (merged upstream)

- v0.4.0: Eigendecomposition-based integrating factors for Hermite streaming
- v0.4.2: Post-RK2 dealiasing for $g$, Lawson-form RK4 for Hermite advection

---

## 6. Next Steps

1. **Longer runs** at low $\nu$ to determine whether the Hermite cascade needs more time to establish itself from MHD driving
2. **Higher forcing amplitude** to increase the energy flux into the Hermite hierarchy
3. **Hermite resolution study** ($M = 64, 128$) to check whether the cascade front is resolved
4. **Comparison with theory** for the expected scaling of the cascade equilibration time with $\nu$
