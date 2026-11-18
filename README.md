# krmhd-research

This repo holds my research on kinetic plasma turbulence, done largely with Claude and the [GANDALF](https://github.com/anjor/gandalf) solver. I choose the questions, make the physics calls and review the results. Claude writes and runs the code, works through the derivations and drafts the papers.

The repo is also a public test bed for long-running AI agents. Study 04 runs as a loop of unattended Claude sessions. Its rules, logs and reviews are all in this repo.

## The physics

A plasma is a gas of charged particles, such as the solar wind. Stir it and it turns turbulent. Energy passes from large eddies to small ones and ends as heat. In the solar wind, particles rarely collide. The energy then also moves into velocity space, where the spread of particle velocities develops ever finer structure. Collisions finally smooth that structure out.

GANDALF solves kinetic reduced magnetohydrodynamics (KRMHD), a standard model of this regime. It writes the velocity structure as Hermite moments, the velocity-space analogue of Fourier modes. A higher moment number m means finer structure. These studies ask how energy reaches high m and what sets the rate. More at [anjor.xyz/research](https://anjor.xyz/research/).

## How the Study 04 loop works

I wrote a brief, [loop/SETUP_PROMPT.md](studies/04-phase-space-helicity/loop/SETUP_PROMPT.md), and Claude built the loop from it. The loop decides which runs to make and passes its own gates. I read its results afterwards and can veto them.

**The protocol.** [LOOP.md](studies/04-phase-space-helicity/LOOP.md) is the rulebook every session reads first. Each session starts fresh and runs unattended, and the repo is its only memory. The rules guard against three failures: money spent unnoticed, a gate passed on a hopeful reading, and work lost between sessions. A gate is a pass-or-fail check between phases. The loop cannot edit its own rules, so it cannot loosen them: a commit that touches them stops it. It also stops and waits for me at hard stops, such as the GPU cap or a kill criterion fixed in advance.

**The runner and guards ([loop/](studies/04-phase-space-helicity/loop/)).** [run_forever.sh](studies/04-phase-space-helicity/loop/run_forever.sh) starts one session after another until a `STOP` file appears. [run_iteration.sh](studies/04-phase-space-helicity/loop/run_iteration.sh) runs one session in the loop's own clone and kills it at 120 minutes. Afterwards [guards.py](studies/04-phase-space-helicity/loop/guards.py) checks the session's commits and transcript, and writes `STOP` on any breach. The runner takes the guards from the commit the session started at, so a session cannot weaken the checks on its own work. [settings.json](studies/04-phase-space-helicity/loop/settings.json) sets the session's permissions: 100 allow rules and 239 deny rules.

**The two reviewers ([.claude/agents/](.claude/agents/)).** The [critic](.claude/agents/study04-critic.md) starts blind: it never reads the loop's logs, conclusions or earlier verdicts. It tries to falsify each derivation, gate and claim, and returns SUPPORTED, REFUTED or INCONCLUSIVE. Only SUPPORTED counts as a pass. The [GANDALF reviewer](.claude/agents/gandalf-reviewer.md) checks each solver pull request at its exact head commit, runs the tests and returns APPROVE or REQUEST CHANGES.

**The claims ledger.** [claims.md](studies/04-phase-space-helicity/claims.md) lists every claim with its evidence, the test that would falsify it and every critic verdict. A claim's text is fixed once written. A changed claim gets a new ID, and the old row is marked superseded. A claim becomes supported only after a SUPPORTED verdict.

**The GPU budget.** [loop/modal_launch.py](studies/04-phase-space-helicity/loop/modal_launch.py) is the only path to a GPU. It runs each job on one A100 GPU on Modal, a cloud service, under a cap of 200 A100-hours for the study. It reserves each run's full timeout first and refuses any launch that would pass the cap. Only the launcher writes [compute_ledger.json](studies/04-phase-space-helicity/compute_ledger.json), and an integrity hash exposes hand edits. Only I can change the cap.

**Changing GANDALF.** The loop changes the solver through issues and pull requests on [anjor/gandalf](https://github.com/anjor/gandalf) ([LOOP.md §10](studies/04-phase-space-helicity/LOOP.md#10-gandalf)). A change needs new tests, green CI and a clean APPROVE on its head commit before the loop squash-merges it. The loop then pins this repo to the merge commit. It never tags or releases GANDALF. It leaves me any pull request that changes an existing test's expected values or the CI files. It has not opened a pull request yet.

**How the loop and I talk.** [STATUS.md](studies/04-phase-space-helicity/STATUS.md) is the front page: the next block of work, the queue, compute and recent sessions. Each session appends an entry to [log/](studies/04-phase-space-helicity/log/) and commits its intent before it starts the work. [decisions.md](studies/04-phase-space-helicity/decisions.md) records each decision as a row that is never edited. [gate_reports/](studies/04-phase-space-helicity/gate_reports/) keeps every gate report and every critic report verbatim. The loop puts its questions in [QUESTIONS.md](studies/04-phase-space-helicity/QUESTIONS.md). I reply there with an `ANSWER:` or `VETO:` line and push, because the loop reads only pushed commits.

**Blind rediscovery.** [docs/rediscovery.md](studies/04-phase-space-helicity/docs/rediscovery.md) records a test run before the loop started. A fresh model context got the equations and a general quadratic form, but not the repo or the paper behind Study 04. It found exactly two conserved quantities, and one matched that paper's Γ term for term. A separate critic context confirmed both to round-off. The paper's headline invariant lay outside the search form, so the test could not find it.

**Self-checking derivations.** [paper/phase-velocity-slices/derivations/](paper/phase-velocity-slices/derivations/) backs the P1 paper and its companion (see [Papers](#papers)). Each script checks a group of equations, prints PASS or FAIL per check and exits non-zero on any failure. A `review/` folder rechecks the same results by routes that share no code with the first set.

**Where Study 04 stands.** It is in Phase 0 (theory), at Gate 1. The critic has refuted the gate's criteria eight times. I accepted their first part on 10 October 2026, and the second part is next. The loop has used 0 of its 200 A100-hours. The newest session in STATUS.md ran on 10 October 2026.

## Studies

| Study | Question | State |
|---|---|---|
| [01 Hermite convergence](studies/01-hermite-convergence/) | How many Hermite moments does a run need before its results stop changing? | Not started beyond one local test in March 2026. |
| [02 Collisionality scan](studies/02-collisionality-scan/) | Does the heating rate stay fixed as collisions become rare? | Closed on 3 November 2026 without a paper. The runs were too weakly turbulent and only reproduced a known result from linear theory ([summary](studies/02-collisionality-scan/docs/status_summary.md)). |
| 03 Echo in imbalanced turbulence: [03-echo-imbalance](studies/03-echo-imbalance/), [03-stochastic-echo](studies/03-stochastic-echo/) | The plasma echo returns energy from fine to coarse velocity structure. Does an imbalance between waves travelling up and down the magnetic field suppress it? | Started. `03-echo-imbalance` is the empty scaffold from the March 2026 plan. `03-stochastic-echo` holds the first forcing calibration and its [working note](docs/echo_cross_helicity.md). Two local runs missed the target imbalance: 0.37 against 0.6. |
| [04 Phase-space helicity](studies/04-phase-space-helicity/) | [Chandran, Mallet and Meyrand (2026)](https://arxiv.org/abs/2607.27981) found a second conserved quantity, Γ. Does conserving it change how energy reaches high m? | Runs as the loop above. Phase 0, Gate 1, no GPU hours used. |

[RESEARCH_PLAN.md](RESEARCH_PLAN.md) has the original plan for Studies 01 to 03.

## Papers

- [phase-velocity-slices](paper/phase-velocity-slices/) (P1): the Hermite spectrum of KRMHD turbulence. Phase mixing, the streaming of particles along the field, carries energy to high m. The cascade across the field drains it. Draft for the Journal of Plasma Physics, with its own runs. [PDF](paper/phase-velocity-slices/main.pdf)
- [compressive-slices](paper/compressive-slices/): the analytical companion to P1. It splits the density and field-strength fluctuations of KRMHD into passive scalars, one per phase velocity along the field. A passive scalar is carried by the flow and does not act on it. Draft. [PDF](paper/compressive-slices/main.pdf)
- [dissipative-anomaly](paper/dissipative-anomaly/): the Study 02 draft. It will not be submitted and stays as a record.
- [echo-imbalance](paper/echo-imbalance/) and [hermite-convergence](paper/hermite-convergence/): empty placeholders for Studies 03 and 01.

## Setup and running

You need [uv](https://docs.astral.sh/uv/). [pyproject.toml](pyproject.toml) pins GANDALF to tag `v0.8.1` of anjor/gandalf. [uv.lock](uv.lock) fixes that tag to commit `9e6599d` and pins every other package.

```bash
git clone https://github.com/anjor/krmhd-research.git
cd krmhd-research
uv sync --frozen   # builds .venv exactly from uv.lock, GANDALF included
```

Check the install with one derivation script. It runs 12 checks in about 30 seconds.

```bash
uv run python paper/phase-velocity-slices/derivations/truncated/t1_hierarchy.py
```

Run a whole suite of checks, or Study 04's unit tests. The suite rewrites the committed outputs in its `out/` folder.

```bash
bash paper/phase-velocity-slices/derivations/continuum/run_all.sh
uv run python -m unittest discover -s studies/04-phase-space-helicity/tests
```

Run a short simulation: the 32³ pipeline test of the P1 driver. It takes under two minutes on an M1 Pro laptop. It writes to a git-ignored `data/` folder and prints a row for [docs/run_log.md](docs/run_log.md).

```bash
uv run python paper/phase-velocity-slices/scripts/run_forced_turbulence.py paper/phase-velocity-slices/configs/forced_test32.yaml
```

Every run reads its parameters from a YAML config. Study 04's GPU runs go only through the loop's launcher.

## Layout

```
.claude/
  agents/                 the critic and the GANDALF reviewer
  skills/status/          a skill that reports a loop's state
CLAUDE.md                 standing instructions for every Claude session
RESEARCH_PLAN.md          the original plan for Studies 01 to 03
docs/                     run log, Study 02 notes and collaboration report, Study 03 note
infrastructure/           Modal app (Study 02) and Slurm scripts (P1 runs)
paper/                    one folder per paper
shared/                   forcing, diagnostics, validation and run logging
studies/
  01-hermite-convergence/
  02-collisionality-scan/
  03-echo-imbalance/
  03-stochastic-echo/
  04-phase-space-helicity/
    LOOP.md               the loop's protocol
    PLAN.md, SPEC.md      the question, outcomes fixed in advance, equations
    STATUS.md, QUESTIONS.md, decisions.md, claims.md, RUNS.md
    compute_ledger.json   GPU hours reserved and charged
    loop/                 runner, guards, launcher, permission rules
    analysis/             gate code
    derivations/          Phase 0 derivation scripts
    gate_reports/         gate and critic reports
    log/                  one file per day of sessions
    tests/
pyproject.toml, uv.lock   environment, with GANDALF pinned
```

Run outputs go to `data/` folders, which git ignores.

## Write-ups

Posts on this work are at [anjor.xyz/research](https://anjor.xyz/research/).
