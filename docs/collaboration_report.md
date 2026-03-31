# Human-AI Collaboration Report: KRMHD Collisionality Scan

**Project:** Kinetic Reduced MHD Phase-Space Cascade Studies
**Study:** 02 -- Collisionality Scan
**Date:** March 2026
**Human researcher:** Anjor Kanekar (independent plasma physicist, GANDALF author)
**AI agent:** Claude Code (Anthropic) initially, then Codex (OpenAI) during the March 30, 2026 forcing/diagnostic phase

---

## 0. Rolling Notes (March 30, 2026)

These are living notes added after the main report was written, to preserve
research context for the eventual write-up.

- **Tooling transition.** The active AI agent switched from Claude Code to Codex during the late-March forcing/diagnostic debugging phase. This is worth capturing in the final write-up because it marks a workflow transition, not a physics transition.
- **Narrative drift warning.** The main body of this report describes the earlier `M=32`, `hyper_n=2`, GANDALF v0.4.2 benchmark-parameter campaign. The current repo state has moved on to `M=128`, `hyper_n=6`, GANDALF v0.4.4, so parts of the report below are historically accurate but no longer describe the exact live configuration.
- **New forcing diagnosis.** Study 02 was not accidentally running with `k_z=0` only. Direct inspection of the current setup showed the old `force_alfven_modes_gandalf` path was forcing `k_z` planes `0, ±1, ±2` in mode-number units. That means the Hermite coupling was present, but the Alfvénic drive was broader in parallel structure than intended for a clean RMHD-style low-`k_z` forcing test.
- **Important nuance.** The old GANDALF shell forcing uses a full `|k|` mask but scales amplitude like `1/k_perp`. In practice that means it can over-weight modes with very small `k_perp`, including effectively `k_z`-dominated modes. The concern is therefore not "missing parallel structure" but "parallel forcing that is too aggressive / not sufficiently RMHD-restricted."
- **Codex implementation on March 30, 2026.** A new config-driven Alfvén forcing mode, `gandalf_perp_lowkz`, was added in `shared/alfven_forcing.py`. It preserves the current study semantics by applying the same forcing field to `z^+` and `z^-` so only `phi` is driven, but it restricts the support to a perpendicular band with low `|n_z|`.
- **Current Study 02 default.** The live Study 02 configs now select low-`k_z` forcing with `mode: gandalf_perp_lowkz`, `max_nz: 1`, and `include_nz0: false`. A direct validation check confirmed that the new forcing populates only the `k_z = ±2π/L_z` planes, not `k_z = 0` or `±4π/L_z`.
- **Local A/B result: Alfvén-only.** A clean local fluid-only probe on the Study 02 dev grid (`32^3`, `eta=2`, `fampl=0.01`, `5000` steps, `M=4`, no Hermite seed, no Hermite forcing) showed that the old shell forcing still drives strong secular growth, but the new low-`k_z` forcing reduces it substantially. At `t = 46.88`, the old shell forcing reached `E_total = 71.95`, while the low-`k_z` forcing reached `E_total = 47.63`. So broad `k_z` forcing was a real contributor to the Alfvén-side overdrive, but it was not the only cause of non-steady behavior.
- **Local A/B result: coupled run.** Shorter local probes with direct `g_0` forcing active (`nu = 10^-3`) showed a different picture: changing the Alfvén forcing support changed the fluid-energy history, but the Hermite diagnostics (`eps_nu`, total `W_m`, and the last-five-moment tail fraction) were nearly identical through the tested windows. For example, at `M=128` and `t = 9.38`, the old shell forcing reached `E_total = 18.36` while the low-`k_z` forcing reached `E_total = 9.73`, yet `eps_nu` and the Hermite tail fraction agreed to the printed precision. In other words, once direct Hermite forcing is turned on, the current short-time Hermite behavior is dominated by that drive and not strongly sensitive to the Alfvén forcing selector.
- **Updated working diagnosis.** Restricting the Alfvén forcing to low `|n_z|` is still the right default, but it should be treated as a partial fix. It helps clean up the forcing physics and reduces fluid-energy growth, yet a true steady-state turbulence run will still require separate work on the energy budget / forcing amplitude and the Hermite-side validation criteria.
- **First promising dev-grid branch.** After the forcing cleanup, the best local dev-grid candidate so far is `eta = 2`, low-`k_z` forcing, and Alfvén forcing amplitude `0.001`. In a fluid-only test (`M=4`, no Hermite seed/forcing), this branch stayed in the narrow band `E_total ≈ 4.5–5.8` from `t ≈ 37` to `t ≈ 94`, which is the first local result in this phase that looks plausibly near a steady Alfvénic state rather than linearly overdriven growth.
- **Longer coupled dev-grid result.** The same low-drive branch with direct `g_0` forcing restored (`M=32`, `nu = 10^-3`) was extended to `6000` steps (`t = 56.25`) and remained well behaved: `E_total = 5.49`, `eps_nu = 2.70e-02`, total Hermite energy `ΣW_m = 3.12e+02`, and the last-five-moment fraction stayed at only `4.2%`. This is the strongest local evidence so far that the branch is not simply delaying the old truncation blowup.
- **Promotion to `M=128` also held.** The same branch was then rerun at `M=128` on the dev grid through `3000` steps (`t = 28.12`). It remained comparably clean: `E_total = 3.29`, `eps_nu = 1.21e-02`, `ΣW_m = 1.92e+02`, and the last-five-moment fraction was only `0.88%`. That means the apparent improvement is not just a low-`M` artifact.
- **First formal saved run on the candidate branch.** A dedicated config, `configs/nu1e-3_dev_lowdrive.yaml`, was added so this branch can be rerun without CLI overrides. Running it through the actual Study 02 local runner produced run `02_nu1e-3_20260330_160135` with outputs written to `studies/02-collisionality-scan/data_lowdrive/`. The saved diagnostics at `t = 9.375` gave `E_total = 0.973`, `eps_nu = 1.29e-02`, `ΣW_m = 1.04e+02`, and a last-five-moment fraction of `1.48%`, which is consistent with the probe-based picture.
- **Long saved dev-grid artifact also matched.** A second dedicated config, `configs/nu1e-3_dev_lowdrive_long.yaml`, pushed the same branch through the Study 02 local runner to `3000` steps (`run 02_nu1e-3_20260330_161023`). The saved diagnostics at `t = 28.124` gave `E_total = 3.286`, `eps_nu = 1.20e-02`, `ΣW_m = 1.92e+02`, a final last-five-moment fraction of `0.88%`, and a maximum last-five-moment fraction of only `5.13%` across the whole run. This formal artifact agrees with the probe-based conclusion that the branch remains well away from truncation.
- **First production-grid smoke test survived.** The same low-drive branch was then promoted to a `64^2 x 32` smoke config, `configs/nu1e-3_lowdrive_smoke.yaml`, and run through the full local runner as `02_nu1e-3_20260330_163006`. It reached `t = 4.688` without numerical failure and even passed the crude `spectral_sanity` gate, with saved diagnostics `E_total = 0.544`, `eps_nu = 3.16e-01`, `ΣW_m = 1.69e+03`, and a final last-five-moment fraction of only `1.18%` (maximum `5.01%`). The main new lesson is that the production grid drives a much stronger Hermite cascade at the same nominal parameters, even though the cascade still appears resolved.
- **Interpretation of the production-grid smoke.** This is encouraging but not yet “steady state.” The production-grid branch is not blowing up, and the Hermite tail is still small, but the dissipation level is roughly an order of magnitude larger than on the dev grid. So the next honest step is a longer production-grid continuation on this same low-drive branch, not a jump to the collisionality scan yet.
- **Comparison figure generated.** A dedicated analysis script, `analysis/plot_lowdrive_candidate.py`, now produces `figures/lowdrive_candidate_comparison.png` / `.pdf`, comparing the dev-grid long run and the production-grid smoke run in terms of `E_total(t)`, `eps_nu(t)`, truncation-tail fraction, `E(k_perp)`, `W(m)`, and `D(m)`.
- **Upstream docs issue filed.** The documentation / getting-started pain points from this exercise were filed upstream as GANDALF issue `#129`: "Docs: add a kinetic-turbulence getting-started guide and clarify forcing/diagnostic semantics."
- **Hermite forcing asymmetry fixed locally.** A remaining study-level mismatch was that Alfvén forcing had already been restricted to low `|n_z|`, but Hermite forcing was still using the upstream full low-`|k|` shell. A new study-local helper, `shared/hermite_forcing.py`, now lets Study 02 force Hermite moments on the same low-`|n_z|` support (`mode: perp_lowkz`) as the Alfvén drive.
- **Important correction:** that asymmetry was not the whole problem. Short reruns on the production grid showed that switching Hermite forcing from the broad shell to low-`|n_z|` made essentially no difference to the early `eps_nu` history, and even setting the ongoing Hermite forcing amplitude to zero after the initial seed left the early `eps_nu` values nearly unchanged. The practical conclusion is that the strong early production-grid Hermite cascade is being driven mainly by phase mixing of the seeded passive field by the Alfvén flow, not by the direct Hermite forcing path.
- **Probe-tooling caveat.** The quick local script `test_nu_stability.py` defaults to `M=32` unless `--m-override` is passed. That is useful for cheap A/B iteration, but it is not the same as the actual promoted `M=128` study branch, so any production-grid claim now needs the `M` value stated explicitly.
- **Clean `M=128` isolation result.** Once the probes were rerun with `--m-override 128`, the source of the early production-grid Hermite activity became much clearer. With the same Alfvén drive and no Hermite seed, direct low-`k_z` Hermite forcing produced only `ΣW_m ≈ 2.1e+01` and `eps_nu ≈ 1.0e-08` by `t = 0.94`. With the seed restored but ongoing Hermite forcing set to zero, the run returned to `ΣW_m ≈ 3.3e+03` and `eps_nu ≈ 6.6e-01`. Restoring the ongoing Hermite forcing on top of that changed almost nothing over the same window.
- **Seed amplitude is now the dominant startup knob.** A final `M=128` rerun with ongoing Hermite forcing still disabled but a smaller seed amplitude (`1e-4` instead of `1e-3`) reduced `ΣW_m` from `≈ 3.3e+03` to `≈ 3.3e+01` and `eps_nu` from `≈ 6.6e-01` to `≈ 6.6e-03`, while leaving the tail fraction essentially unchanged. That quadratic scaling is exactly what one expects if the early Hermite cascade is inherited from the seeded passive field rather than the ongoing forcing path.
- **Alfvén-amplitude check.** Doubling the Alfvén forcing amplitude from `0.001` to `0.002` at `M=128` approximately doubled the fluid `E_total` over the first `t ≈ 0.94`, but left the early Hermite diagnostics unchanged to the printed precision when the seed amplitude was held fixed. So increasing the Alfvén drive is not, by itself, a lever for reducing the startup Hermite burst.
- **Practical workflow change.** The Hermite seed is no longer treated as a hidden runner detail. A new study-local helper, `shared/hermite_seed.py`, makes `enabled`, `amplitude`, and `seed` config-driven, and the Study 02 YAMLs now record those values explicitly under `hermite_seed:`. That matters both for reproducibility and for the eventual paper trail.
- **First passive-smoke artifact.** The new config `configs/nu1e-3_lowdrive_passive_smoke.yaml` keeps the same low-drive Alfvén branch but turns ongoing Hermite forcing off and reduces the startup Hermite seed to `1e-4`. The first saved production-grid artifact from this branch, `02_nu1e-3_20260330_214808`, reached `t = 4.688` with `E_total = 0.568`, `eps_nu = 3.06e-03`, `ΣW_m = 1.61e+01`, and a final last-five-moment fraction of `1.14%` (maximum `5.05%`). Compared to the earlier coupled smoke on the same grid (`eps_nu = 3.16e-01`, `ΣW_m = 1.69e+03`), this is roughly a two-order-of-magnitude reduction in the Hermite-sector startup burst at essentially the same fluid energy level.
- **First passive long continuation.** The corresponding continuation config, `configs/nu1e-3_lowdrive_passive_long.yaml`, was then run through `3000` steps as `02_nu1e-3_20260331_074724`. It reached `t = 14.062` with `E_total = 1.58`, `eps_nu = 1.11e-03`, `ΣW_m = 9.34`, a final last-five-moment fraction of `1.01%`, and a maximum tail fraction of `6.63%` over the whole run. The mean Hermite dissipation over the second half of the saved history was only `1.71e-03`. This is not yet a textbook steady state, but it is the cleanest production-grid branch so far and it remains far away from the earlier Hermite startup pathology.
- **Upstream forcing-helper bug filed.** While making the study-local low-`|n_z|` Hermite forcing path work, a separate upstream bug was found in GANDALF's `gaussian_white_noise_fourier_perp_lowkz()` helper: the JIT wrapper appears to use the wrong `static_argnums`. That was filed upstream as issue `#131`, with the local workaround noted in the issue body.
- **Pure-fluid check first.** Before pushing further on the passive-Hermite branch, the study was simplified all the way down to Alfvénic turbulence only: no Hermite seed, no Hermite forcing, `nu = 0`. One practical wrinkle showed up immediately: although parts of the package docs describe `M=0` as the pure-fluid limit, the current `gandalf_step()` still rejects `M < 2` because of the collision-operator normalization. For now the safe local workaround is `M=2` with `g ≡ 0`, which is dynamically equivalent for these no-Hermite tests.
- **Benchmark forcing does not directly transfer to the current low-`k_z` production setup.** Using the benchmark values previously noted from GANDALF (`eta = 2`, `hyper_r = 2`, `hyper_n = 2`, forcing amplitude `0.005`) in a fluid-only production-grid smoke run still produced strong secular energy growth: the saved run `02_alfven_20260331_094512` reached `E_total = 2.84` by `t = 4.688` with no sign of saturation.
- **Lower-drive fluid-only probes still grow.** Local `5000`-step pure-fluid probes at the same production resolution showed that `fampl = 0.001` and `fampl = 0.002` are both cleaner than `0.005`, but neither is actually steady by `t = 23.44`: `E_total` reached `2.65` and `5.30`, respectively. Extending the `fampl = 0.001`, `eta = 2` branch to `20000` steps (`t = 93.74`) still gave continued growth, up to `E_total = 7.83`.
- **Changing eta alone is not fixing the fluid branch.** Additional pure-fluid probes at `(eta, fampl) = (10, 0.002)` and `(20, 0.005)` closely tracked the lower-eta branches over the same `t = 23.44` window. So the current production-grid Alfvénic problem is not mainly a Hermite problem and not obviously solved by raising `eta` alone.
- **Updated practical diagnosis.** Under the current study-local low-`k_z` forcing path, the production-grid Alfvénic cascade is still not reaching an honest steady state. That means the next parameter-design step should happen in the pure-fluid problem first, before resuming any passive-Hermite or collisionality work.
- **Upstream RMHD-only request filed.** The inconsistency between the documented `M=0` fluid limit and the current timestepper restriction `M >= 2` was filed upstream as issue `#132`, requesting a true RMHD-only / no-Hermite execution path.
- **Fast upstream turnaround.** GANDALF `v0.4.4` was released immediately afterward and includes the RMHD-only fix: true `M=0`, `nu=0` runs now execute cleanly without the old `M=2, g=0` workaround. The local workspace was bumped from `v0.4.3` to `v0.4.4`, and the rerun `02_alfven_20260331_114340` confirmed that the pure-fluid benchmark config now works exactly as intended.
- **Important distinction.** The `v0.4.4` fix solves the execution-path / API inconsistency for RMHD-only runs, but it does not by itself solve the fluid-branch physics. The true `M=0` benchmark rerun reproduced the same secular growth previously seen with the `M=2, g=0` workaround, so the remaining problem is still the Alfvénic parameter/forcing branch rather than the Hermite plumbing.
- **Probe/benchmark ordering check.** One possible explanation for the mismatch was that the local Study 02 probe scripts were applying forcing after the timestep, whereas the upstream `alfvenic_cascade_benchmark.py` applies forcing before `gandalf_step()`. The probe scripts were corrected to match the upstream ordering, and the fluid-only forcing-family comparison was rerun. The result was effectively unchanged: `balanced_elsasser_lowkz` is still the least explosive of the tested forcing families, but none of the current branches at `64^2 x 32` yet shows a proper steady inertial range.
- **First fluid-only milestone made explicit.** The campaign goal is now staged more sharply than before: get a convincing steady-state Alfvénic inertial range first, then reintroduce passive/Hermite physics, and only after both sectors behave independently should the collisionality scan resume.
- **Balanced low-`k_z` fluid-only sweep.** A dedicated production-grid `M=0` sweep (`analysis/compare_balanced_lowkz_fluid_branches.py`, figure `figures/balanced_lowkz_fluid_branches.png`) compared five branches through `t = 46.88`: amplitudes `0.002`, `0.005`, and `0.010` with forcing shells `n = 1-2`, plus amplitudes `0.002` and `0.005` with a broader `n = 1-3` band. The result was clean:
  - Raising the forcing amplitude simply scales up the secular growth. For the `n = 1-2` branch, final fluid energy grew from `E_total = 4.12` (`A = 0.002`) to `25.8` (`A = 0.005`) to `103` (`A = 0.010`) by the same final time.
  - Widening the forced band from `n = 1-2` to `n = 1-3` does broaden the perpendicular spectrum, but at the tested amplitudes it also makes the energy-growth problem much worse: `A = 0.002, n = 1-3` still reached `E_total = 21.0`, and `A = 0.005, n = 1-3` reached `131.6`.
  - So the immediate lesson is: "stronger forcing" is not the path to the first milestone. The current best direction is the opposite one -- keep the balanced low-`k_z` forcing family, widen the forced band only cautiously, and lower the amplitude if extra spectral breadth is needed.
- **Refined fluid-only branch search.** A second focused comparison (`analysis/compare_balanced_lowkz_fluid_refined.py`, figure `figures/balanced_lowkz_fluid_refined.png`) tested whether the broader `n = 1-3` band could be rescued by lowering the forcing amplitude further. This was more encouraging:
  - `A = 0.001, n = 1-3` reached only `E_total = 5.25` by `t = 46.88`, much lower than the clearly overdriven `A = 0.002, n = 1-3` branch (`E_total = 21.0`), while still producing a visibly broader perpendicular spectrum than the narrow-band control `A = 0.002, n = 1-2`.
  - `A = 0.0005, n = 1-3` was even cleaner energetically (`E_total = 1.31` by the same time), but it is likely moving toward an underdriven regime rather than a robust turbulence benchmark.
  - The key caveat is that none of these refined branches is steady yet: the energy histories are still monotonic, and the spectra remain steeper than the benchmark `k_\perp^{-5/3}` reference. But `A = 0.001, n = 1-3` is the first production-grid branch in this fluid-only campaign that looks like a serious candidate for a longer continuation rather than an immediate discard.
- **Long continuation of the refined branch still grows.** Extending `A = 0.001, n = 1-3` to `20000` steps (`t = 93.74`) did not rescue it into a steady state. The fluid energy kept rising, from `E_total = 5.25` at `t = 46.88` to `E_total = 9.19` at `t = 93.74`. The growth rate is gentler than the obviously overdriven branches, but this is still not an honest steady Alfvénic inertial-range run.
- **Exact benchmark path is structurally different from the Study 02 probes.** To understand whether the missing inertial range was a Study 02 setup problem or a broader solver regression, the exact upstream `alfvenic_cascade_benchmark.py` path from GANDALF `v0.4.4` was pulled into `/tmp` and run directly. This revealed several important differences from the Study 02 fluid-only probes:
  - the benchmark uses a cubic `64^3` grid, not `64^2 x 32`
  - it initializes a weak random `k^{-5/3}` spectrum with `M = 10`
  - it computes the CFL timestep once at startup and then holds `dt` fixed
  - its default forcing family is the upstream Gaussian shell drive, not the Study 02 low-`k_z` wrapper
- **Early exact-benchmark behavior looks qualitatively healthier.** By `t ≈ 10`, the exact benchmark run is already behaving differently from the Study 02 fluid-only branches: instead of clean secular growth from a tiny seed, the total energy fluctuates in the range `E_total ≈ 3–4` and shows substantial positive and negative injection episodes. That does not prove it has reached the final benchmark inertial range yet, but it is strong evidence that the earlier Study 02 fluid-only search has not been exploring the same branch as the published / upstream benchmark path.
- **Extended exact-benchmark result: more developed, but still not a clean plateau.** Pushing the exact `64^3` benchmark path forward on `main` showed that it does enter a much more turbulence-like driven state than the Study 02 probes: by `t ≈ 30` it had reached `E_total ≈ 15` and started the benchmark's averaging window. But the branch did not settle into a convincing steady plateau over the next several Alfvén times. By `t ≈ 40`, `E_total` had drifted up to `≈ 22`, and the benchmark's own printed "steady-state check" oscillated between pass/fail in a way that was clearly inconsistent with the raw energy drift. The practical conclusion is that the exact upstream benchmark path is still the right calibration anchor, but even that path needs direct scrutiny rather than blind trust in the built-in steady-state messaging.
- **Exact `32^3` benchmark snapshots finally showed the first convincing pre-blowup spectra.** Running the exact upstream benchmark at `32^3` with frequent snapshot output captured a useful averaging-window sequence even though the benchmark post-processing still crashes at the end because it calls `np.trapz` instead of `np.trapezoid`. The best early saved spectrum from this first pass is `benchmark_output/alfven32_snapshots/.../spectrum_t37.5.png`, which is the first plot in this whole phase that looks plausibly like a developing Alfvénic cascade rather than pure forcing-shell domination.
- **Longer exact `32^3` run bracketed the useful window.** Extending the same exact benchmark to `t = 44` with snapshots every `100` steps showed that the physically usable window survives a bit longer than first expected: the spectra at `t = 38.4`, `39.4`, and `40.3` remain qualitatively reasonable, while clear late-time spoilage appears by `t = 41.3` and is unmistakable by `t = 42.2` and `43.1`, where both kinetic and magnetic spectra develop artificial high-`n` bumps just before the runaway. This is encouraging for the "small spatial grid, larger `M` later" strategy: `32^3` can already produce a recognizable pre-blowup inertial-range candidate, but at the current benchmark settings it is not yet a long-lived steady branch.
- **Long exact `64^3` continuation did not converge to a cleaner cascade.** The exact `64^3` benchmark was first extended to `t = 50`, then resumed from its final checkpoint and pushed further toward `t = 150` with delayed averaging and periodic checkpoints every `10 τ_A`. This longer continuation did not reveal a hidden late-time `-5/3` range. Instead, checkpoint-derived spectra at `t = 60, 70, 80, 90, 100, 110` stayed steep through `t = 100` and then developed an obvious artificial high-`n` bump by `t = 110`, just before the resumed branch ran away at `t ≈ 114.6`. A dedicated script, `analysis/plot_benchmark_checkpoint_spectra.py`, was added to render these spectra directly from checkpoint states because the delayed averaging window meant no native snapshot PNGs were written during the resumed run.
- **Calibration matrix formalized.** To stop hand-tuning benchmark branches one command at a time, a small manifest/launcher pair was added: `configs/alfven_benchmark_matrix.yaml` and `scripts/run_alfven_benchmark_matrix.py`. The first matrix focuses on the exact `64^3` benchmark with branches from the clean `t = 60` checkpoint, varying only forcing amplitude and resistive damping (`eta = 3, f = 0.005`; `eta = 2, f = 0.004`; `eta = 3, f = 0.004`; `eta = 4, f = 0.003`), plus a `128^3` baseline smoke branch. This makes the next calibration phase reproducible and reviewable rather than relying on memory of ad hoc shell commands.
- **Automation hardening for overnight work.** Two small utilities were added to make the calibration loop less brittle. `scripts/run_exact_alfven_benchmark.py` wraps the upstream benchmark and aliases `np.trapz` to `np.trapezoid`, so completed runs no longer get mislabeled as failures by the NumPy 2 compatibility bug in the benchmark's final diagnostics. `analysis/score_benchmark_snapshots.py` provides a coarse spectral-quality score from saved snapshot CSVs, and `scripts/run_alfven_benchmark_overnight.py` uses that scorer to run remaining matrix branches sequentially and stop early if one branch finally looks acceptable. This is the first point in the project where the Alfvén calibration has been turned into an actual loop rather than a sequence of manually curated shell commands.
- **Useful diagnostics lesson.** On this promoted branch, the fluid `E_total(t)` history at `M=32` and `M=128` was essentially identical to the printed precision, while the Hermite-sector diagnostics differed. This is a concrete reminder that the current `E_total` and steady-state gates are blind to the Hermite sector and cannot by themselves certify the phase-space cascade.
- **Reusable local probe tooling.** `studies/02-collisionality-scan/scripts/test_nu_stability.py` was extended during this phase so it can now override the Alfvén forcing mode, toggle Hermite seeding, and disable Hermite forcing. That makes future A/B tests reproducible from the command line instead of depending on one-off REPL snippets.
- **Diagnostics caveat remains.** The current `steady_state` and `energy_balance` diagnostics are still not trustworthy for the Hermite problem. `E_total` excludes Hermite energy, and the historical forcing-injection bookkeeping in the Study 02 runners is not a faithful measure of injected power. These diagnostics should be treated as provisional until the Hermite-aware validation layer is cleaned up.

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
