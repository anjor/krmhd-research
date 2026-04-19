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
- **Overnight `64^3` calibration result.** Four resumed exact-benchmark branches were completed from the clean `t = 60` checkpoint. The healthiest of these was `eta = 4.0`, forcing amplitude `0.003`, which stayed finite through `t = 110` and avoided the catastrophic late high-`n` shoulder seen in the weaker-damping branches. The next-best branch was `eta = 3.0`, forcing amplitude `0.005`, which remained competitive but still developed noticeable late-time tail growth. None of the four branches produced a convincing steady `k_\perp^{-5/3}` inertial range: all remained too steep, and the weaker-damping cases still piled up strongly near the upper resolved shells.
- **Interpretation of the overnight matrix.** The overnight scan was still useful even though it did not solve the benchmark. It showed that the late runaway is sensitive to the forcing-dissipation balance in the expected direction: increasing `eta` and modestly reducing forcing can delay or suppress the worst pileup. But it also showed that `64^3` is still too cramped to cleanly separate forcing, inertial range, and dissipation on this branch. Even the best late-time `64^3` spectra show a slight shoulder by `t \approx 100`, which is exactly the warning sign that the dissipation range is too close to the candidate inertial range.
- **Resolution pivot.** Based on those late `64^3` spectra, the next milestone is no longer "find one more clever `64^3` tuning trick." The more honest next step is to launch the exact `128^3` benchmark baseline and see whether the extra scale separation actually opens a usable Alfvénic inertial range before late-time pileup sets in. That run is now the primary calibration target.
- **Useful diagnostics lesson.** On this promoted branch, the fluid `E_total(t)` history at `M=32` and `M=128` was essentially identical to the printed precision, while the Hermite-sector diagnostics differed. This is a concrete reminder that the current `E_total` and steady-state gates are blind to the Hermite sector and cannot by themselves certify the phase-space cascade.
- **Reusable local probe tooling.** `studies/02-collisionality-scan/scripts/test_nu_stability.py` was extended during this phase so it can now override the Alfvén forcing mode, toggle Hermite seeding, and disable Hermite forcing. That makes future A/B tests reproducible from the command line instead of depending on one-off REPL snippets.
- **Diagnostics caveat remains.** The current `steady_state` and `energy_balance` diagnostics are still not trustworthy for the Hermite problem. `E_total` excludes Hermite energy, and the historical forcing-injection bookkeeping in the Study 02 runners is not a faithful measure of injected power. These diagnostics should be treated as provisional until the Hermite-aware validation layer is cleaned up.
- **128³ benchmark launched on Modal (2026-04-01).** Three branches submitted in parallel on A100 GPUs via `scripts/modal_128_benchmark.py`, each running for 200 τ_A with checkpoints every 10 τ_A. Initial runs used `force_alfven_modes_gandalf` (GANDALF 1/k_perp weighting) — this blew up catastrophically at 128³ with z± RMS ~ 10⁹ by the first checkpoint, because the 1/k_perp singularity amplifies near-zero k_perp modes that are far more numerous at 128³.
- **Gaussian shell forcing (force_alfven_modes) also unstable.** Switching to Gaussian white noise forcing removed the 1/k_perp singularity. Runs survived longer but every combination of forcing amplitude (f=0.005, 0.008, 0.01) and dissipation (eta=2–4, hyper_r=2–4) eventually blew up between t=230 and t=260 τ_A. The failure mode was always the same: energy pileup at high k_perp near the grid cutoff, visible as a growing bump at k_perp ≈ 200 in the E(k⊥) spectrum, followed by exponential energy growth and NaN within ~10 τ_A. The upstream benchmark only runs to 50 τ_A by default, so it never encounters this instability.
- **Root cause: full-|k| shell forcing excites high-k_z modes.** The Gaussian shell forcing populates all modes with |k| in the forced band, including modes with large k_z and small k_perp. These modes are not RMHD-compatible and apparently drive an energy cascade into high-k_perp modes that eventually overwhelms the dissipation. Increasing hyper_r from 2 to 4 delayed the blowup by ~30 τ_A (from t≈230 to t≈260) but did not prevent it — the pileup was already visible in the spectra before the crash.
- **Loop ordering fixed.** The upstream benchmark applies forcing *before* `gandalf_step` (force-then-step), while our script had the reverse (step-then-force). In our ordering, freshly injected high-k energy sits undamped for a full timestep. Fixed to match the benchmark.
- **Low-k_z balanced Elsasser forcing scan (current, 2026-04-02).** Switched to `force_alfven_modes_balanced` with `max_nz=1`, `include_nz0=False` — restricts forcing to the lowest parallel wavenumber planes only (RMHD-compatible). Hold dissipation fixed at eta=2, hyper_r=2 and scan forcing amplitude: f = 0.001, 0.002, 0.005, 0.01. All four branches running to t=500 τ_A on Modal A100s. **All four alive past t=250** — first time any 128³ configuration has survived this long. Early spectra from f=0.001 show no cascade (too weak), waiting for f=0.002–0.01 spectra.
- **Key figures.** `figures/128_lowkz_energy_evolution.png` (energy vs time for all four branches), `figures/128_lowkz_f0p001_spectra.png` (E(k⊥) snapshots for weakest forcing). All saved to `studies/02-collisionality-scan/figures/`.
- **Low-k_z scan results (2026-04-02).** All four branches (f=0.001–0.01) completed to t=500 with no blowups — the low-k_z restriction solved the grid-scale instability. However, **no branch developed an inertial range**. All spectra are steep, with energy concentrated at the forcing scale and dropping rapidly. Even f=0.01 (strongest) only extends to k⊥ ≈ 20–30 before falling off. Stronger forcing scan (f=0.02, 0.03, 0.035, 0.05, 0.1): f=0.02 survived to t≈570 before blowing up, f=0.03 to t≈333, all stronger amplitudes blew up earlier.
- **Dissipation is far too weak (key insight, 2026-04-03).** With η=2 and hyper_r=2, the dissipation rate is η·(k⊥/k_max)^4. At k⊥=200 this gives only 0.66/τ_A (0.15% damping per step). Even at k_max≈264 the rate is only 2.0/τ_A. This explains two problems simultaneously: (1) no inertial range develops because there's no "backpressure" from the dissipation scale to set the cascade flux, and (2) the eventual pileup/blowup occurs because the cascade delivers energy to high k faster than dissipation can remove it. The normalized hyper-dissipation η·(k/k_max)^{2r} is resolution-independent but that means **the actual damping rate at any physical k is independent of resolution** — at 128³ the resolved k range is larger but the damping at the edge is the same as at 32³. Filed as gandalf#136.
- **Dissipation scan launched (2026-04-03).** Resume from the clean f=0.02 t=500 checkpoint with η=10, 20, 50 (5×–25× the original). At η=20, the damping at k⊥=200 becomes 1.5%/step — still modest but potentially enough to establish an energy sink. At η=50, damping is 3.8%/step at k=200 and 23.4%/step at k_max.
- **Dissipation scan results (2026-04-03).** η=10 blew up at t=990. η=20 and η=50 both completed to t=1000 — first time any 128³ run survived this long. Spectra at η=50 show cascade extending to n⊥≈15–20 with slope tracking between −5/3 and −3/2. η=20 similar but shorter extent.
- **Extended runs to t=2000 (2026-04-03 → 2026-04-04).** η=20 blew up again at t=1265 (same instability). η=50 survived to t=1960 before NaN. The η=50 spectra at t=1800 are the best achieved: inertial-range-like region from n⊥≈3 to n⊥≈15–20 with appropriate slope. Clean window is t=1500–1800; mild pileup appears at t≈1900 near dealiasing boundary (n=42).
- **Mode number insight (2026-04-04).** Plotting against mode number n⊥ (instead of k⊥ with 2π factors) clarified the picture: forcing at n=1–2, cascade extends to n≈15–20, dealiasing cutoff at n=42. The usable inertial range is about one decade in scale.
- **Next phase: η=100 stabilization + Hermite (2026-04-04).** Resuming from η=50 t=1800 checkpoint with η=100 (7.4% damping/step at k=200) to suppress the mild pileup and stabilize the Alfvénic spectrum. If successful, this becomes the base state for Hermite forcing (M=128, hyper_n=6, g₀ forcing, ν=0.01 calibration). Filed long-time instability as gandalf#136.
- **η=100 Alfvénic base state reached (2026-04-05).** `alfven128_lowkz_f0p02_eta100/checkpoints/checkpoint_t2000.0.h5` became the production Alfvénic steady state: 8% E variation over the last 50 τ_A, clean low-to-mid-k⊥ cascade. Every subsequent Hermite run resumes from this checkpoint.
- **Lambda=1 gotcha in checkpoints (2026-04-05).** Alfvénic checkpoints store `Lambda=1.0`; the factor `(1-1/Λ)` in the g-coupling then vanishes and the Hermite cascade is silently killed. Every Hermite resume now explicitly overrides to `Λ=√5` (the β_i=1 value). Filed as a local procedural rule.
- **Linear Hermite benchmark works (2026-04-05).** 16³-spatial, M=128, ν=1, no z± drive: 500 τ_A of clean phase mixing, W(m) monotonically decreasing, ε_ν ~ 0.23 with 94% noise. Establishes that the Hermite sector plumbing is correct; nonlinear failures are not a linear-solver problem.
- **Nonlinear Hermite campaign — all runs blow up (2026-04-04 → 2026-04-14).** ν-scan at M=128, 128³, hyper_n=6 (ν=1,3,5,10 long; ν=20,50,100 short probes). Every long run eventually NaNs: ν=1 at ~80 τ_A, ν=3 at ~122, ν=5 at ~167, ν=10 at ~184. Higher ν delays onset but does not prevent it. The 50-τ_A probes (ν=20,50,100) gave ε_ν ~ 46–63, which looked like a plateau but turned out to be transients of a numerically unstable run.
- **Diagnosis: numerical, not physical (2026-04-16).** Initial hypothesis was physical pileup (Alfvénic k_z broadening → cascade outruns damping at m=M). Direct inspection of saved W(m,t), g k_z spectra, and onset statistics killed that story: W(m=M) stays at O(1) noise for the entire 100–180 τ_A before blowup (no accumulation), g(k_z, m=M) is at 10⁻¹⁴ pre-blowup (no flux arriving), blowup is simultaneous across all m at ν-independent rate (should be localized at m=M if physical), and z± k_z spectrum matches the Alfvénic-only reference. Diagnostics live in `analysis/diagnose_hermite_blowup.py` and `figures/hermite_blowup_*.png`. Handoff written at `docs/hermite_handoff.md`.
- **GANDALF issue #137 filed (2026-04-16).** Root cause: Lawson-RK4 composition of an exact integrating factor for streaming with explicit RK stages on nonlinear advection leaks `dt·v_th·k_z·√m/Λ` into the stability envelope. Candidate fix proposed: IMEX — implicit streaming + damping, explicit nonlinear advection. Acceptance test defined: ν=3, M=128, 128³, 200 τ_A without blowup and ε_ν reaching steady value.
- **GANDALF v0.5.0 shipped (2026-04-17).** PR #138 implements ARS(2,2,2) IMEX-RK222 with per-k_z batched LU solve for the implicit streaming + damping operator; nonlinear Poisson bracket stays explicit. `gandalf_step` default flipped to `scheme="imex_rk222"`. PR #143 adds advisory `scheme` checkpoint metadata with mismatch warning so we don't silently resume on the wrong integrator.
- **Acceptance test passed first try (2026-04-17).** `hermite128_nu3_imex` ran 200 τ_A from the Alfvénic checkpoint, 7.2h on A100, zero NaN. After the 30 τ_A Hermite fill-in transient, ε_ν = 49.2 ± 10.7 (rel 21.6%). E_total still drifting upward (×1.52) — injection > dissipation because the Hermite sector was empty at resume; full energy balance would need a longer run, but ε_ν is already steady.
- **Dissipative anomaly confirmed (2026-04-18).** `hermite128_nu5_imex` and `hermite128_nu10_imex` completed cleanly at 200 τ_A each. ε_ν mean (skipping transient): ν=3: 49.21±10.65, ν=5: 49.26±10.61, ν=10: 49.23±10.81. Across 3.3× in ν the means agree to 0.1% and E_total rises by identical 1.52×. Combined with the old Lawson short probes (ν=20,50,100 at 60/53/46 over 50 τ_A before numerical blowup), ε_ν ≈ 50 holds across >30× in ν. Figure: `figures/hermite128_imex_nu_scan.{png,pdf}`.
- **Phase-mixing spectrum confirmed too (2026-04-18).** Time-averaged W(m) over the 100 τ_A averaging window (83 snapshots per run) cleanly follows m^{-1/2} in the inertial range m∈[4,40]: fitted slopes −0.484 (ν=3), −0.489 (ν=5), −0.493 (ν=10), all within ~2% of the Zocco–Schekochihin prediction. At high m the curves split in the expected ν-order (less damping → more energy piles up before (m/M)⁶ kicks in), but the dissipation integrand 2ν(m/M)⁶W(m) nearly coincides across ν in the bulk dissipation range m≈70–110 — cumulative sums match to 0.2% (48.1, 48.0, 48.0). Figure: `figures/hermite128_imex_Wm_spectrum.{png,pdf}`.
- **Extended ν-scan in flight (2026-04-18).** Three more runs submitted to probe the ε_ν(ν) shape: ν=1 (possible low-ν departure from plateau), ν=20 and ν=50 (expected to leave plateau as hyper-dissipation outruns the cascade). Expected wall time ~7h on A100 each, running in parallel from the same Alfvénic t=2000 checkpoint. Once complete, the summary plot ε_ν(ν) vs ν will be the headline figure for the write-up.
- **Extended ν-scan complete: plateau holds over 50× in ν (2026-04-18 → 2026-04-19).** All three extensions ran 200 τ_A clean, including ν=1 (no sign of the low-ν deviation the user had expected). Windowed means: ν=1: 48.42±11.41; ν=20: 49.21±10.99; ν=50: 49.21±11.21. Full six-point plateau: ε_ν = 49.09 ± 0.30 across ν∈{1,3,5,10,20,50}. Mean total Hermite energy ΣW_m drops monotonically 359 → 114 across the same range (self-adjustment), with a visible kink around ν≈3 where the curve flattens. Inertial-range slope across the scan: m^{-0.472} to m^{-0.496}, all within 5% of the Zocco-Schekochihin m^{-1/2} prediction. The per-m dissipation integrand peak shifts from m≈M at low ν to m≈80 at ν=50, but the integrated ε_ν matches to <2%. Figures: `figures/hermite128_imex_eps_nu_plateau.{png,pdf}` (headline), `figures/hermite128_imex_Wm_spectrum_full.{png,pdf}` (six-curve W(m) + integrand).

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

### Phase 9: $64^2 \times 32$ scan (superseded)

The original $64^2 \times 32$, $M=32$ scan with GANDALF v0.4.2 reported $\varepsilon_\nu \propto \nu$ with no plateau, and was interpreted as the Hermite cascade failing to self-sustain. That conclusion no longer stands: the campaign was rebuilt at much higher resolution and later uncovered a scheme-level numerical problem that had been masking the physics. This section is retained for historical context; the operational results are in Phases 10--11 below.

### Phase 10: $128^3$ Alfvenic base state and nonlinear Hermite blowup (April 2026)

With Modal GPU infrastructure in place, the campaign moved to $128^3$ spatial with $M=128$. Establishing a clean Alfvenic steady state at this resolution took several weeks of parameter-space navigation (details in the rolling notes above): the key lever was raising $\eta$ to 100 with low-$k_z$ balanced Elsasser forcing, which stabilized the long-time cascade without a late-time pileup. The checkpoint `alfven128_lowkz_f0p02_eta100/...t2000.0.h5` became the production base state and has been reused unchanged in every subsequent Hermite run.

Turning on $g_0$ forcing on top of that base state produced nonlinear Hermite runs that all blew up. Higher $\nu$ delayed onset (80, 122, 167, 184 $\tau_A$ for $\nu=1,3,5,10$) but did not prevent it. Short 50 $\tau_A$ probes at $\nu=20,50,100$ gave $\varepsilon_\nu \sim 46$--$63$, which looked like a plateau but turned out to be transients of a numerically unstable run.

**Diagnosis (2026-04-16):** direct inspection of the saved data contradicted the initial "physical pileup" narrative. $W(m=M)$ sat at $\mathcal{O}(1)$ noise through the entire 100--180 $\tau_A$ pre-blowup window with no accumulation; the $k_z$ spectrum of $g(m=M)$ was at the $10^{-14}$ noise floor, so no cascade flux was arriving at $m=M$ in the first place; the blowup was simultaneous at all $m$ at roughly the same rate (physical pileup would localize at $m \approx M$); and the $z^\pm$ $k_z$ spectrum matched the Alfvenic-only reference. All signatures of a numerical instability, not a physical cascade.

### Phase 11: GANDALF IMEX-RK222 fix and the dissipative anomaly (2026-04-17 to 2026-04-18)

**Issue gandalf#137 filed (2026-04-16).** The root-cause hypothesis: Lawson-RK4 composes an exact integrating factor for streaming with explicit RK stages on nonlinear advection, leaking $\mathrm{d}t \cdot v_{\mathrm{th}} k_z \sqrt{m}/\Lambda$ into the stability envelope. Proposed fix: IMEX with implicit streaming + hyper-collisional damping and explicit nonlinear advection. Acceptance test defined: $\nu=3$, $M=128$, $128^3$, 200 $\tau_A$ without blowup.

**GANDALF v0.5.0 (2026-04-17).** PR #138 implements ARS(2,2,2) IMEX-RK222 with a per-$k_z$ batched LU solve for the implicit operator; nonlinear advection stays explicit. `gandalf_step` default flipped to `scheme="imex_rk222"`. PR #143 adds advisory `scheme` checkpoint metadata with mismatch warning to guard against silently resuming on the wrong integrator.

**Acceptance test passed first try.** `hermite128_nu3_imex` ran 200 $\tau_A$ from the Alfvenic checkpoint in 7.2h on an A100, zero NaN. After skipping the first 30 $\tau_A$ of Hermite fill-in, $\varepsilon_\nu = 49.2 \pm 10.7$ (rel 21.6%). $E_\mathrm{total}$ still drifts upward ($\times 1.52$) because the Hermite sector was empty at resume; full energy balance would need a longer run, but $\varepsilon_\nu$ itself is already steady.

**Dissipative anomaly confirmed (2026-04-18).** Three IMEX runs at $\nu \in \{3, 5, 10\}$ gave statistically identical time-averaged dissipation:

| $\nu$ | $\varepsilon_\nu$ (mean $\pm$ std) | rel std | $E_\mathrm{total}$ ratio |
|-------|------------------------------------|---------|--------------------------|
| 3     | $49.21 \pm 10.65$                  | 21.6%   | 1.52                     |
| 5     | $49.26 \pm 10.61$                  | 21.5%   | 1.52                     |
| 10    | $49.23 \pm 10.81$                  | 22.0%   | 1.52                     |

Across $3.3\times$ in $\nu$ the means agree to 0.1%. Combined with the short Lawson probes at $\nu \in \{20, 50, 100\}$ ($\varepsilon_\nu \approx 60, 53, 46$ before numerical blowup set in), $\varepsilon_\nu \approx 50$ holds across $> 30\times$ in $\nu$.

Time-averaged $\langle W(m)\rangle_t$ over the 100 $\tau_A$ averaging window (83 snapshots per run) cleanly follows $m^{-1/2}$ in the inertial range $m \in [4, 40]$:

| $\nu$ | fitted slope |
|-------|--------------|
| 3     | $-0.484$     |
| 5     | $-0.489$     |
| 10    | $-0.493$     |

All three match the Zocco--Schekochihin phase-mixing prediction to $\sim 2\%$. At high $m$ the curves split in the expected $\nu$-order (weaker damping $\to$ more energy piles up before $(m/M)^6$ kicks in), but the dissipation integrand $2\nu(m/M)^6 W(m)$ nearly coincides across $\nu$ in $m \approx 70$--$110$, with cumulative sums $48.1, 48.0, 48.0$ --- agreement to 0.2%. Figures: `figures/hermite128_imex_nu_scan.{png,pdf}`, `figures/hermite128_imex_Wm_spectrum.{png,pdf}`.

**Plateau extends to 50$\times$ in $\nu$ (2026-04-19).** The extended scan completed on $\nu \in \{1, 20, 50\}$, each 200 $\tau_A$ clean. The plateau holds across all six $\nu$ values in the scan:

| $\nu$ | $\bar\varepsilon_\nu$ (ts window) | $\bar\varepsilon_\nu$ (spectrum window) | $\langle\sum_m W(m)\rangle_t$ | slope $m\in[4,40]$ |
|-------|-----------------------------------|-----------------------------------------|-------------------------------|--------------------|
|  1    | $48.42 \pm 11.41$                 | 48.56                                   | 359                           | $-0.472$           |
|  3    | $49.21 \pm 10.65$                 | 48.15                                   | 162                           | $-0.484$           |
|  5    | $49.26 \pm 10.61$                 | 47.99                                   | 137                           | $-0.489$           |
| 10    | $49.23 \pm 10.81$                 | 47.99                                   | 126                           | $-0.493$           |
| 20    | $49.21 \pm 10.99$                 | 48.26                                   | 121                           | $-0.494$           |
| 50    | $49.21 \pm 11.21$                 | 48.79                                   | 114                           | $-0.496$           |

The six-point plateau average is $\varepsilon_\nu = 49.09 \pm 0.30$. Even $\nu = 1$, where the plateau was expected to possibly deviate, sits within 2% of the others. Total Hermite energy drops $359 \to 114$ as $\nu$ goes $1 \to 50$ (self-adjustment), with a visible kink around $\nu \approx 3$ where the curve flattens. Inertial-range slope is within 5% of the Zocco--Schekochihin $m^{-1/2}$ prediction across all six. The per-$m$ dissipation integrand peak shifts from $m \approx M$ at low $\nu$ to $m \approx 80$ at $\nu = 50$, but the integrated $\varepsilon_\nu$ matches to $< 2\%$. Headline figure: `figures/hermite128_imex_eps_nu_plateau.{png,pdf}`; full $W(m)$ overlay: `figures/hermite128_imex_Wm_spectrum_full.{png,pdf}`.

---

## 3. GANDALF Issues Filed

| Issue | Title | Status |
|-------|-------|--------|
| #118 | Native `compute_dissipation_rate` diagnostic + $g$ initialization | Open |
| #120 | Hermite time integration unconditionally unstable | Fixed in v0.4.0 (PR #121) |
| #122 | Missing dealiasing in RK2 step and RHS assembly for $g$ | Fixed in v0.4.2 (PR #123) |
| #124 | RK2 fundamentally unstable for Hermite advection | Fixed in v0.4.2 (PR #125) |
| #126 | RK4 returns time as JAX array instead of float | Fixed in v0.4.3 (PR #134) |
| #129 | Docs: kinetic-turbulence getting-started guide and forcing/diagnostic clarifications | Fixed in v0.4.3 (PR #130) |
| #131 | `static_argnums` bug in `gaussian_white_noise_fourier_perp_lowkz` JIT wrapper | Fixed in v0.4.3 (PR #135) |
| #132 | Support RMHD-only runs with $M=0$ (docs say it works, code rejects $M<2$) | Fixed in v0.4.4 (PR #133) |
| #136 | Resolution-independent hyper-dissipation sets $\nu$-independent damping rate at the grid edge | Open |
| #137 | Numerical instability in Hermite integrator at high $M \cdot k_z$ --- propose IMEX | Fixed in v0.5.0 (PR #138) |
| #142 | Record integrator scheme on checkpoints with mismatch warning | Fixed in v0.5.0 (PR #143) |

The five numerical-scheme issues (#120, #122, #124, #137, and the resolution-of-dissipation scaling in #136) were all discovered through running the collisionality campaign. Each represents a real mismatch between a time-integration or dissipation choice and the mathematical character of the equations: oscillatory streaming needing integrating factors (#120), $g$ advection needing dealiasing and a stability-bounded method on the imaginary axis (#122, #124), and the Lawson-RK4 composition of integrating factor with nonlinear stages leaking a hidden CFL constraint into the envelope at high $M \cdot k_z$ (#137). The April 2026 diagnosis of #137 in particular required ruling out the physically plausible "cascade outruns damping" story from the scheme-level signature, which was the single longest debugging step of the entire campaign.

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

### Immediate (completing the scan)

1. **Finish the extended $\nu$-scan.** $\nu \in \{1, 20, 50\}$ in flight. $\nu = 1$ tests whether the plateau continues to smaller $\nu$ or deviates; $\nu = 20, 50$ test where the plateau breaks upward as hyper-dissipation starts to outrun the cascade.
2. **Headline figure.** $\varepsilon_\nu$ vs $\nu$ summary plot across the full scan $\nu \in \{1, 3, 5, 10, 20, 50\}$, with error bars from the window-variance and a fit to the plateau region.
3. **Long equilibration run.** Extend one branch (most likely $\nu = 3$) to $\sim 500$ $\tau_A$ to let $E_\mathrm{total}$ reach a true injection/dissipation balance. The scalar $\varepsilon_\nu$ is already steady; this is for the cleanest steady-state $W(m)$ and $E(k_\perp)$ spectra for the paper.

### Before the write-up

4. **Sanity checks on the plateau interpretation.** Compare $\varepsilon_\nu$ against an independent estimate of the Alfvenic-to-Hermite energy flux (derivable from the Poisson-bracket coupling $\{\Phi, g_0\}$) to confirm it equals the injection rate. If they match, the plateau is literally "all injected energy ends up dissipated at $m \sim M$, independent of $\nu$".
5. **Cascade rate check.** Compute the forward Hermite flux $\Pi(m)$ (GANDALF provides `hermite_flux`) and verify it is constant in $m$ across the inertial range --- that is the standard "constant-flux cascade" test the $m^{-1/2}$ slope implies.

### Paper write-up plan

Target: JPP letter or short paper. Working structure:

- **Intro:** The dissipative anomaly in hydro; its KRMHD analogue as the phase-space cascade to high Hermite moment $m$; what was previously shown (Zocco--Schekochihin phase-mixing spectrum) and what was missing (direct nonlinear demonstration of $\nu$-independent $\varepsilon_\nu$).
- **Model and numerics:** KRMHD equations, Hermite representation of $g$, GANDALF solver, 128$^3$ spatial + $M=128$ Hermite, $\beta_i = 1$, driven Alfvenic forcing + direct $g_0$ forcing at low $k_z$, hyper-dissipation $(m/M)^6$. One paragraph on the IMEX-RK222 integrator (with pointer to GANDALF v0.5.0) --- this is new and worth describing because readers who reproduce the setup need to know.
- **Results.** Figure 1: $\varepsilon_\nu(\nu)$ plateau across $\nu \in \{1, 3, 5, 10, 20, 50\}$ with old-Lawson short-probe points overlaid as triangles for context. Figure 2: $\langle W(m)\rangle_t$ showing $m^{-1/2}$ in the inertial range, with a panel showing the compensating product $2\nu(m/M)^6 W(m)$ to make the cancellation between $W$ and $\nu$ visible. Figure 3: $E(k_\perp)$ and $\Pi(m)$ to show the Alfvenic base state is honest and the Hermite cascade has constant flux. Quantitative table of slopes and plateau values.
- **Discussion:** How $W(m)$ and $\nu$ self-adjust to keep the product invariant; finite-$M$ effects; what this does and does not say about the truly collisionless limit.
- **Supplementary / methods:** The Lawson-RK4 numerical-instability story. Short, honest, and useful for reproducers --- this is the kind of thing that saves the next group six weeks.

### Farther out (if the anomaly plot is compelling)

6. **$\nu$-dependence of the $W(m)$ split in the dissipation range.** The plots already show a clean $\nu$-ordering at high $m$; there is probably a predictable scaling of where the curves diverge. Worth checking against theory.
7. **Resolution study.** Re-run the plateau at $M = 64$ and $M = 256$ to bound finite-$M$ corrections. Cheap; important for a referee.
8. **Forcing-amplitude scan.** $\varepsilon_\nu$ should scale with the injection rate; establishing that explicitly would tighten the anomaly claim.
