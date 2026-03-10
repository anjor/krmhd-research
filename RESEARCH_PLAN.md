# KRMHD Research Plan: Phase-Space Cascades in Kinetic Turbulence

## Overview

Three sequenced numerical studies using [GANDALF](https://github.com/anjor/gandalf) to address open questions in phase-space turbulence in the KRMHD (Kinetic Reduced MHD) regime. Each study builds on the previous, producing 2–3 papers targeting Journal of Plasma Physics.

**Code**: GANDALF — JAX-based KRMHD spectral solver. Fourier in (x, y), Hermite-Laguerre in velocity space. Single ion species (kinetic ions, fluid electrons), strong guide field ordering, k⊥ρ_i ≪ 1.

**Hardware**: MacBook M1 Pro (development, small runs), [Modal](https://modal.com) (production parameter scans on cloud GPUs).

**Orchestration**: Claude Code / Codex drives simulation runs and data analysis from this repo. GANDALF is an installed dependency — never modified from here.

---

## Study 1: Hermite Convergence in Nonlinear Driven KRMHD Turbulence

### Motivation

Every KRMHD/gyrokinetic simulation must choose a Hermite truncation number M, yet no systematic convergence study exists for fully nonlinear, driven 3D KRMHD turbulence where both the perpendicular spatial cascade and the velocity-space (Hermite) cascade operate simultaneously.

Theory makes clear predictions. Linear phase mixing produces W(m) ∝ m^{−1/2} (Kanekar et al. 2015). In the presence of nonlinear perpendicular advection, stochastic echo cancels phase mixing on average, steepening the spectrum. The Adkins & Schekochihin (2018) analytic solution of the 1D Vlasov-Kraichnan model predicts W(m) ∝ m^{−3/2}. This has never been confirmed in 3D KRMHD simulations.

### Gap in the literature

The research confirms this is genuinely open:

- **Meyrand et al. (2019, PNAS)**: The only published nonlinear driven 3D KRMHD turbulence simulation. Shows the spectrum is steeper than m^{−1}, confirming fluidization, but presents results at a single resolution without varying M. No convergence study.
- **Zhou, Liu & Loureiro (2023, PNAS)**: Driven 3D KREHM (electron kinetics, sub-ion scales). Finds m^{−1/2} — echo is *not* effective because free energy concentrates near intermittent current sheets. Compares M = 30 and M = 60 but not a systematic scan.
- **Adkins, Meyrand & Squire (2025, ApJ)**: Most systematic existing check — 32 simulations spanning M = 4–32 across four parameter regimes. But this is KREHM (not KRMHD), and convergence serves as appendix-level validation rather than the paper's focus.
- **Hoffmann, Frei & Ricci (2023)** and **Mandell, Dorland & Landreman (2018)**: Systematic Hermite convergence studies, but for tokamak gyrokinetics (ITG/TEM turbulence in curved geometry with strong collisionality) — fundamentally different physics.
- **Pezzi et al. (2018)**: Hybrid-Vlasov-Maxwell, finds ∼ m^{−2} at β = 0.5.
- **Servidio et al. (2017)**: MMS spacecraft observations show m^{−3/2} in magnetosheath — the only measurement consistent with the Adkins & Schekochihin prediction, but observational not numerical.

The measured Hermite exponents range from −1/2 to −2 across different models. Nobody has measured it systematically in KRMHD with controlled resolution.

### Scientific questions

1. What is W(m) in fully nonlinear, driven 3D KRMHD turbulence? Is it m^{−3/2} as predicted by Adkins & Schekochihin (2018), or something different?
2. At what M do macroscopic quantities (E(k⊥), total heating rate Q_tot) converge?
3. At what M does the Hermite spectrum W(m) itself converge?
4. Does convergence of bulk quantities occur at much lower M than convergence of the Hermite spectrum (as the fluidization picture implies)?
5. At what M does the echo mechanism "turn on" — i.e., the backward Hermite flux becomes significant?

### Parameter space

| Parameter | Values | Notes |
|-----------|--------|-------|
| **M** (Hermite modes) | 4, 8, 16, 32, 64, 128 | Main scan variable |
| N⊥ | 64² | Fixed; sufficient for perpendicular inertial range |
| N∥ | 16 | Enough parallel modes for 3D driving |
| β_i | 1.0 | Standard value |
| ν | 0.01 | Weak collisions; hypercollisional (hyper_n = 6) |
| Forcing | Balanced Alfvénic (σ_c = 0) | Simplest case; echoes maximally active |
| Duration | ~200 τ_A after saturation | Ensure steady state + good statistics |

### Key diagnostics

1. **Hermite spectrum W(m)** at steady state — fit power law exponent, compare to m^{−1/2}, m^{−1}, m^{−3/2}, m^{−2}
2. **Convergence of E(k⊥)** — perpendicular energy spectrum must become independent of M above some threshold
3. **Total dissipation rate Q_tot** vs M — primary convergence metric
4. **Hermite flux decomposition** — forward (phase-mixing) and backward (echo) components as function of M
5. **Recurrence diagnostic** — energy pileup at m = M signals insufficient resolution

### Run matrix

6 runs at different M. Low-M runs (4, 8) are cheap. M = 128 is the ground truth reference.

### Expected outcome

- Macroscopic quantities converge at M ∼ 16–32
- The Hermite spectrum itself requires M ∼ 64+ to measure the scaling cleanly
- The measured exponent is steeper than −1 (confirming Meyrand et al.) but may differ from the −3/2 of the 1D Kraichnan model due to finite correlation times and anisotropy
- The main figure: W(m) at multiple M overlaid, showing the physical spectrum emerging above the truncation artifact

### Estimated compute

∼10–50 GPU-hours on Modal. M = 128 run dominates cost.

### Validation gates (must pass before extracting science)

- [ ] Energy conservation to < 1% relative error at steady state
- [ ] E(k⊥) ∝ k^{−5/3} reproduced in the perpendicular inertial range
- [ ] Injection rate = dissipation rate at steady state (energy balance)
- [ ] M = 4 run shows clear recurrence artifacts (expected failure = good sanity check)

### Deliverable

**Paper 1**: "Hermite convergence and the velocity-space cascade in nonlinear KRMHD turbulence" → Journal of Plasma Physics

---

## Study 2: Collisionality Scan — The Dissipative Anomaly

### Motivation

The "dissipative anomaly" — the collisionless analogue of the zeroth law of turbulence — predicts that the total turbulent heating rate Q_tot becomes independent of collision frequency ν as ν → 0. This is a foundational result: it guarantees efficient heating in extremely collisionless plasmas (solar wind, accretion flows, ICM). Nastac, Tatsuno & Schekochihin (2024) extended the theoretical proof to the phase-space entropy cascade. Numerical confirmation in driven 3D KRMHD turbulence has never been published.

### Scientific questions

1. Does Q_tot plateau as ν → 0 across 2+ decades of collisionality?
2. How does the dissipation spectrum D(m) shift in Hermite space as ν decreases?
3. Does the velocity-space dissipation scale follow m_d ∝ ν^{−α} for some α?
4. Is the result sensitive to the form of the collision operator (hyper_n)?

### Prerequisites

Study 1 determines the minimum M for converged macroscopic quantities. All Study 2 runs use that M (or slightly above, for safety).

### Parameter space

| Parameter | Values | Notes |
|-----------|--------|-------|
| **ν** | 1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001 | 3 decades |
| hyper_n | 2, 4, 6 | Sensitivity check (select ν values only) |
| M | From Study 1 (likely 32 or 64) | Fixed |
| N⊥ | 64² (possibly 128² at low ν) | May need higher resolution at low ν |
| β_i | 1.0 | Same as Study 1 |
| σ_c | 0.0 | Balanced |

### Key diagnostics

1. **Q_tot(ν)** — the headline plot: should plateau
2. **D(m, ν)** — Hermite-resolved dissipation spectrum at each ν
3. **m_d(ν)** — velocity-space dissipation scale
4. **k_d(ν)** — spatial dissipation scale (coupled to m_d via k⊥ ∼ m)
5. **Energy balance** — injection = dissipation at every ν

### Run matrix

7 collisionalities × 1 base hyper_n = 7 primary runs, plus ∼6 supplementary runs for hyper_n sensitivity at 3 selected ν values. Total: ∼13 runs.

### Expected outcome

Q_tot flat to within ∼10% across 2+ decades of ν. Dissipation shifts to higher m and k⊥ as ν decreases, but the integrated rate is invariant. Main figure: Q_tot vs ν plateau, with insets showing D(m) at different ν.

### Estimated compute

∼50–200 GPU-hours. Low-ν runs need longer integration and possibly higher spatial resolution.

### Validation gates

- [ ] Study 1 convergence results reproduced at the chosen M
- [ ] Energy balance holds at every ν to < 1%
- [ ] High-ν limit recovers expected collisional damping behavior

### Deliverable

**Paper 2**: "Numerical verification of the dissipative anomaly in kinetic plasma turbulence" → Journal of Plasma Physics or Physical Review E

---

## Study 3: Echo Efficiency in Imbalanced Turbulence

### Motivation

Meyrand et al. (2019) showed plasma echoes "fluidize" balanced KRMHD turbulence — the echo returns energy from velocity space, making collisionless turbulence behave more fluid-like. But the solar wind is imbalanced (|σ_c| ∼ 0.5–0.8), and theory predicts imbalance suppresses echoes because the counter-propagating Elsässer field needed for echo reformation is depleted.

This connects directly to the helicity barrier (Squire et al. 2022, Nature Astronomy), which forms in imbalanced turbulence at ion scales. Understanding whether echoes are already suppressed at MHD scales (where GANDALF operates) constrains whether the barrier can form. Nobody has mapped the (σ_c, amplitude) parameter space systematically.

### Scientific questions

1. How does echo efficiency η = Γ_backward / Γ_forward depend on cross-helicity σ_c?
2. Is there a sharp threshold in σ_c above which echoes are negligible?
3. How does turbulence amplitude (δB/B_0) affect echo efficiency?
4. Does the Hermite spectrum exponent change with σ_c (from ∼ m^{−3/2} toward m^{−1/2} as imbalance increases)?

### Prerequisites

Studies 1 and 2 establish resolution requirements and confirm the dissipative anomaly. We know how to set up a converged, steady-state driven turbulence run.

### Parameter space

| Parameter | Values | Notes |
|-----------|--------|-------|
| **σ_c** | 0.0, 0.3, 0.5, 0.7, 0.9 | Balanced → strongly imbalanced |
| **Forcing amplitude** | 3 levels (weak/moderate/strong) | Controls δB/B_0 |
| M | From Study 1 | Fixed |
| ν | 0.01 (in plateau from Study 2) | Fixed |
| β_i | 0.3, 1.0 | Low-β (solar wind–like) + moderate |
| N⊥ | 128² | Higher resolution for echo diagnostics |

### Key diagnostics

1. **Echo efficiency** η(k⊥) = Γ_backward(k⊥) / Γ_forward(k⊥)
2. **Hermite spectrum** W(m) at each σ_c — does the exponent change?
3. **Effective damping rate** γ_eff vs linear γ_Landau — net damping after echo cancellation
4. **Elsässer spectra** E±(k⊥) — energy in z⁺ vs z⁻
5. **2D phase-space flux** in (k⊥, m) — the full picture

### Run matrix

5 σ_c × 3 amplitudes × 2 β = 30 runs. Most expensive study.

### Expected outcome

Echo efficiency drops sharply for |σ_c| > 0.3, near-complete suppression at |σ_c| > 0.7. Higher amplitudes also suppress echoes (nonlinear cascade overwhelms echo coherence). The Hermite spectrum transitions from m^{−3/2} (balanced, echo-active) toward m^{−1/2} (imbalanced, echo-suppressed, linear-like phase mixing). Main figure: η(σ_c) at different amplitudes showing suppression threshold.

### Estimated compute

∼200–500 GPU-hours. Higher resolution, longer runs, larger parameter space.

### Validation gates

- [ ] σ_c = 0 run reproduces Study 1 results (consistency check)
- [ ] Elsässer ratio z⁺/z⁻ matches prescribed σ_c
- [ ] Energy balance holds across all runs

### Deliverable

**Paper 3**: "Echo suppression in imbalanced Alfvénic turbulence" → Journal of Plasma Physics (or PRL if the result is crisp)

---

## Repo Structure

```
krmhd-research/
├── README.md                         # Project overview (public-facing)
├── RESEARCH_PLAN.md                  # This document
├── CLAUDE.md                         # Instructions for Claude Code / Codex
│
├── studies/
│   ├── 01-hermite-convergence/
│   │   ├── configs/                  # YAML configs for each M value
│   │   ├── scripts/
│   │   │   ├── run_local.py          # Single run on M1 Pro
│   │   │   ├── run_modal.py          # Single run on Modal GPU
│   │   │   ├── sweep_M.py            # Orchestrate full M scan
│   │   │   └── check_steady_state.py # Verify saturation before analysis
│   │   ├── analysis/
│   │   │   ├── hermite_spectrum.py   # Measure W(m), fit power law
│   │   │   ├── convergence_plot.py   # Q_tot, E(k⊥) vs M
│   │   │   └── echo_flux.py          # Forward/backward Hermite flux
│   │   ├── figures/                  # Publication-quality plots
│   │   ├── STUDY.md                  # Problem statement, run log, results
│   │   └── data/                     # Simulation outputs (.gitignored)
│   │
│   ├── 02-collisionality-scan/
│   │   ├── configs/
│   │   ├── scripts/
│   │   │   ├── sweep_nu.py           # Orchestrate ν scan
│   │   │   └── sweep_hyper_n.py      # Collision operator sensitivity
│   │   ├── analysis/
│   │   │   ├── dissipative_anomaly.py # Q_tot vs ν plateau plot
│   │   │   └── dissipation_spectrum.py # D(m) at each ν
│   │   ├── figures/
│   │   ├── STUDY.md
│   │   └── data/
│   │
│   └── 03-echo-imbalance/
│       ├── configs/
│       ├── scripts/
│       │   └── sweep_sigma_c.py      # Orchestrate (σ_c, amplitude) scan
│       ├── analysis/
│       │   ├── echo_efficiency.py    # η vs σ_c
│       │   └── phase_space_flux.py   # 2D (k⊥, m) flux maps
│       ├── figures/
│       ├── STUDY.md
│       └── data/
│
├── shared/
│   ├── diagnostics.py                # Reusable analysis routines
│   ├── plotting.py                   # JPP figure style, shared colormaps
│   ├── validation.py                 # Energy balance, steady-state checks
│   └── run_utils.py                  # Config loading, run ID generation, logging
│
├── infrastructure/
│   ├── modal_app.py                  # Modal app definition for GANDALF runs
│   ├── modal_runner.py               # Submit config → get diagnostics back
│   └── storage.py                    # Upload/download simulation data (S3 or Modal volumes)
│
├── paper/
│   ├── hermite-convergence/          # LaTeX for Paper 1
│   ├── dissipative-anomaly/          # LaTeX for Paper 2
│   └── echo-imbalance/              # LaTeX for Paper 3
│
└── docs/
    ├── physics_notes.md              # Derivations, key equations, references
    └── run_log.md                    # Chronological log of all runs
```

## CLAUDE.md Key Directives

The `CLAUDE.md` file instructs Claude Code / Codex on:

1. **GANDALF is a dependency** — `import gandalf`. Never modify the solver from this repo.
2. **Physics validation first** — every run checks energy conservation, spectral scaling, steady state before science extraction. Fail loudly if validation gates don't pass.
3. **Config-driven** — all parameters in YAML. No hardcoded physics values in scripts.
4. **Idempotent analysis** — all analysis scripts read from saved data and can be re-run.
5. **Figure standards** — JPP-compatible (single-column: 3.4", double-column: 7"), LaTeX labels via matplotlib's `text.usetex`, consistent colormap (viridis for 2D, qualitative palette for line plots).
6. **Run logging** — every simulation gets a unique ID (`{study}_{param}_{timestamp}`), parameters and outcome logged to `run_log.md`.
7. **No scope creep** — these three studies only. The helicity barrier is a separate project.

---

## Timeline

| Phase | Weeks | Activities |
|-------|-------|------------|
| **Setup** | 1 | Repo structure, Modal integration, verify GANDALF driven turbulence end-to-end (locally + Modal) |
| **Study 1: runs** | 2–3 | M scan (6 runs), monitor convergence |
| **Study 1: analysis + draft** | 4–5 | Hermite spectra, convergence plots, draft Paper 1 |
| **Study 2: runs** | 5–7 | ν scan (13 runs) |
| **Study 2: analysis + draft** | 7–9 | Dissipative anomaly plots, draft Paper 2 |
| **Study 3: runs** | 9–12 | (σ_c, amplitude, β) scan (30 runs) |
| **Study 3: analysis + draft** | 12–14 | Echo efficiency maps, draft Paper 3 |
| **Finalize + submit** | 14–16 | Polish all papers, submit |

Total: ∼4 months. Studies overlap with writing.

---

## Budget

| Item | Estimate |
|------|----------|
| Study 1 (Modal GPU) | $30–80 |
| Study 2 (Modal GPU) | $80–200 |
| Study 3 (Modal GPU) | $200–400 |
| **Total compute** | **$300–700** |

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| GANDALF bugs at high M (> 64) | Low | High | Existing M = 128 linear benchmark validates. If issues arise, M = 64 likely sufficient. |
| Dissipative anomaly doesn't converge | Low | Medium | Would itself be interesting — implies KRMHD needs modification at very low ν. Document and publish regardless. |
| Hermite exponent ≠ −3/2 | Medium | Low | This is a *measurement*, not a prediction. Any well-measured exponent in KRMHD is a result. |
| Echo diagnostics are noisy | Medium | Medium | Increase run duration, ensemble-average realizations. JAX `vmap` makes this easy. |
| Modal costs exceed budget | Low | Medium | Low-M runs pilot first. Abort expensive runs early if diagnostics show problems. |
| Scope creep into helicity barrier | Medium | High | Hard boundary: these three studies only. Barrier requires k⊥ρ_i ∼ 1 physics beyond GANDALF's ordering. |

---

## Key References

- Schekochihin et al. (2009), ApJS 182:310 — KRMHD formulation
- Kanekar et al. (2015), JPP 81 — Linear Hermite cascade, m^{−1/2}
- Schekochihin et al. (2016), JPP 82 — Phase mixing vs nonlinear advection
- Adkins & Schekochihin (2018), JPP 84 — Solvable model, m^{−3/2} prediction
- Meyrand, Kanekar, Dorland & Schekochihin (2019), PNAS 116:1185 — Fluidization of KRMHD turbulence
- Meyrand, Squire, Schekochihin & Dorland (2021), JPP 87 — Helicity barrier theory
- Squire et al. (2022), Nature Astronomy — Helicity barrier + PSP
- Zhou, Liu & Loureiro (2023), PNAS — Electron heating in KREHM, m^{−1/2}
- Nastac, Tatsuno & Schekochihin (2024) — Phase-space entropy cascade and dissipative anomaly
- Adkins, Meyrand & Squire (2025), ApJ — KREHM turbulent heating, M convergence check
- Hoffmann, Frei & Ricci (2023), JPP — Gyromoment convergence in tokamak turbulence
- Mandell, Dorland & Landreman (2018), JPP — Laguerre-Hermite formulation
- Kanekar (2025), arXiv:2511.21891 — GANDALF code paper
