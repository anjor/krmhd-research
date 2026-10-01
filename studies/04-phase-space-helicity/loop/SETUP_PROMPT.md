# Prompt: set up the Study 04 autonomous loop

Start Claude Code in `~/repos/anjor/krmhd-research`. Tell it to read this file and carry it out. It is a one-time setup task.

## Goal

Study 04 (`studies/04-phase-space-helicity/`) will run as an autonomous loop. Each iteration is a fresh agent session with no memory of earlier iterations. The repo is its only memory.

The loop carries the study from where it stands today to the Phase 4 output in `PLAN.md`. It works out what GANDALF changes the study needs and makes them. It decides which simulations to run, runs them and analyses them. It keeps a diligent log of all of it.

In this session you build and test the loop. Do no physics work. Do not start the loop. Do not launch GPU runs. When you finish, tell Anjor how to start it.

Nobody reviews the loop in real time. Three failures matter most: money spent without anyone noticing, a gate passed on a hopeful reading, and work lost between sessions. Most of what follows guards against those three.

## Read first

- `CLAUDE.md` and `pyproject.toml`
- In `studies/04-phase-space-helicity/`: `PLAN.md`, `SPEC.md`, `STATUS.md`, `QUESTIONS.md`, `decisions.md`, `claims.md`, `RUNS.md`
- `docs/run_log.md`, for the measured Modal wall times
- `../gandalf/CLAUDE.md`

## Where the study stands

Phase 0 is complete and on `main`. `STATUS.md` says the next block is Gate 1 and that it waits for Anjor. `pyproject.toml` pins GANDALF v0.6.0. The ν-scan checkpoints are local, under `studies/02-collisionality-scan/data/`. They were made on v0.5.0.

One branch is not merged: `study04/closure-shipped-base-state`. It exists locally and on `origin`, one commit ahead of `main`. It answers the Gate 1 question on the closure test. It adds question 5, on regenerating the base state. Check for an open PR, then merge the branch into `main` first. It fast-forwards.

## Anjor's decisions, 1 October 2026

Record these in `decisions.md`. For this study they replace `PLAN.md` §1, §5 and §6, and rule 1 of `CLAUDE.md`.

1. **Modal.** The agent launches production runs itself. The cap is 200 A100-hours for the whole study. A launcher script enforces it. Expected spend is about 140 to 160: about 42 for set A, 65 for set B and 25 to 40 for a new base state.
2. **GANDALF.** The agent changes GANDALF itself. The path is issue, branch, tests, PR, green CI, independent physics review, squash-merge. Then it pins this repo to the merge commit. It never bumps the version, tags, releases or publishes to PyPI.
3. **Gates.** The agent passes gates itself. A gate needs its coded check and an independent critic. The agent logs each passed gate for Anjor's veto and carries on. It stops only at a hard stop. `SPEC.md` §7 says Anjor sets the set B convergence criteria. The agent now sets them, before the launch.
4. **The invariant.** This answers Gate 1 question 2. The Γ of the plan is the paper's Γ^± (their B25 and B32). It stays the forced quantity. H0, H1 and H2 stay as written and refer to Γ^±. The loop also derives and measures the paper's phase-space helicity H_ph-sp (their B18), in post-processing. That diagnostic is exploratory, and every result from it says so. Record this as a dated clarification under `PLAN.md` §2.
5. **The base state.** This answers question 5 with its option (b). Phase 1 uses the v0.5.0 checkpoints only to test the pipeline. No science claim rests on them. The loop regenerates the Alfvénic base state on the pinned GANDALF and recalibrates the forcing. Set A starts from the new base state.

The hard stops:

- A launch would exceed the compute cap.
- One of the first two kill criteria in `PLAN.md` §7 is met. The third is not a stop: the loop writes up the null.
- A change is needed to a frozen section. These are `PLAN.md` §2 and §7 and `SPEC.md` §7.
- A change is needed to the loop's own guards: `loop/`, `.claude/` or `LOOP.md`.
- Something would leave the two repos: an email, a message to a collaborator, an arXiv submission, a post, a release.
- The study is complete.
- The loop cannot make progress, for a reason it cannot fix.

Earlier decisions still hold. There are no status emails. Work in this repo goes directly on `main`. Questions go in `QUESTIONS.md`.

## What to build

Keep it small. `CLAUDE.md` says simple scripts beat frameworks. Paths below are under `studies/04-phase-space-helicity/` unless they say otherwise. Change the layout if you see a better one.

### 1. A clone for the loop

The loop works in its own clone, `~/repos/anjor/krmhd-research-loop`. Anjor's checkout stays his. This removes races between him and the loop. It also makes a dirty tree unambiguous: it can only come from a dead iteration.

- Link `studies/02-collisionality-scan/data` in the clone to the data in Anjor's checkout. The `**/data/` ignore rule does not match a symlink, so exclude the link.
- Create a GANDALF worktree for the loop at `~/repos/anjor/gandalf-study04`, from `origin/main` of `../gandalf`. The loop never touches `../gandalf`.
- Anjor's input reaches the loop only through pushed commits. In `QUESTIONS.md` he writes lines that start `ANSWER:` or `VETO:`. Uncommitted changes are never his input.

### 2. The protocol: `LOOP.md`

Every iteration reads `LOOP.md` first. It must be complete enough for an agent that knows nothing else. Give the reason for each rule. It covers the following.

**One iteration.**

1. Orient. Read `LOOP.md`, `PLAN.md`, `SPEC.md`, `STATUS.md`, `QUESTIONS.md`, `decisions.md`, `claims.md`, `RUNS.md` and the last two log files. Look for Anjor's input: commits that are not the loop's, and `ANSWER:` and `VETO:` lines. A veto comes before everything else.
2. Recover. The runner says in the prompt if a dead iteration left a stash. Inspect it. Keep the sound work, drop the rest and log what you did.
3. Check what is in flight. Fetch finished Modal runs, validate them and reconcile their hours. Check open GANDALF PRs and CI.
4. Choose one block. Take the next open block in `STATUS.md`. If it is blocked, take the next unblocked item in the queue. If everything is waiting, set the wait time, log a short entry and end. A block is one coherent unit: a diagnostic with its tests, a derivation with its critic pass, a gate evaluation, a GANDALF change, the launch of a run set, or the analysis of a run set.
5. Write the intent in the log before the work. Say what you will do, what you expect to see, and what result would change the plan.
6. Do the work, under `CLAUDE.md` and the working rules that remain in `PLAN.md` §1.
7. Verify. Run the gate code. Get a critic pass. Record the verdict verbatim in `claims.md`.
8. Close out. Finish the log entry. Update `STATUS.md`, `claims.md`, `decisions.md`, `RUNS.md` and `QUESTIONS.md`. Commit as `Study 04 [<iteration id>]: <summary>`. Push. Confirm that the tree is clean and nothing is unpushed.

Aim to finish in 90 minutes. The runner kills the session at 120. Never end with uncommitted work. If the block is unfinished, commit it labelled WIP and say in `STATUS.md` what remains. If the last three entries tried the same block and failed, change the approach or declare a hard stop.

**Quality gates.** Gates 1, 2 and 3 check that the work is correct. Such a gate passes when three things hold. Its coded check in `analysis/gates.py` passes on raw outputs. The critic returns SUPPORTED on that evidence. No kill criterion is met. Each evaluation writes a gate report: the numbers, the thresholds, the run IDs and checksums of the data files. The report is committed, because `data/` is not. Record the pass in `decisions.md` with the report, the verdict and the commit. List it in `QUESTIONS.md` under "For review". Then continue. A failed quality gate is normal work: diagnose and fix.

**Gate 4 is an outcome.** It is evaluated once per run set. A negative result is H0 and goes to Phase 4. Nobody works on it until an effect appears. Before a launch, the quantities, the averaging windows and the criteria that will judge the runs go into a new `SPEC.md` §7a and into `gates.py`, with a critic pass. After the launch they are frozen. Changing them is a hard stop.

**Tolerances.** They come from `SPEC.md` §7 and are frozen. A tolerance that looks wrong is a hard stop.

**Validation.** Rule 2 of `CLAUDE.md` still holds. No science comes from a run that fails the validation gates. In the loop, "stop and report" means log the failure and diagnose it. It is not a hard stop.

**The critic.** Use the `study04-critic` subagent for every derivation, every gate evaluation and every interpretation. Give it `SPEC.md`, the scripts and the raw outputs. Do not give it your conclusion. Its job is to falsify. It also checks that the gate code matches `SPEC.md` §7. Log every verdict, including the unfavourable ones. Ask again only with new evidence, never to get a better answer.

**Claims.** The text of a claim is fixed once written. Its status, evidence and verdict cells may be updated. A changed claim gets a new ID.

**Decisions.** The agent decides everything that is not a hard stop. It records each decision and its reason in `decisions.md`. Points for Anjor go in `QUESTIONS.md` under "Open, not blocking", with a recommendation. The agent proceeds on its recommendation. If Anjor later answers differently, the agent adapts and logs the rework.

**Vetoes.** Anjor vetoes a passed gate or a decision with a `VETO:` line under its item. The next iteration reopens that gate or decision before any other work. It logs what has to be redone.

**Hard stops.** Write a `STOP` file in the study folder. It states the reason and the decision Anjor must make. Add the question under "Blocking" in `QUESTIONS.md`. Log, commit, push and end. The runner does not start while `STOP` exists. Anjor resumes the loop with a pushed commit that answers the question and removes `STOP`.

**Modal.** Runs go through `loop/modal_launch.py` and nothing else. Four things must be true before a launch. The gate the runs depend on has passed. The run code has passed a local smoke test at small size. The configs are committed. The hours are estimated from measured wall times and the run set has a row in `RUNS.md`. After a launch, record the app ID and set the wait time. Later iterations poll, fetch the results from the volume, run `validate_run`, reconcile the hours and update `RUNS.md` and `docs/run_log.md`. Diagnose a failed run before any relaunch. Never relaunch an unchanged failing config. Relaunches count against the cap.

**GANDALF.** Change GANDALF when the study needs a capability the solver lacks, or when you find a bug. Work only in the worktree. Never edit the installed package in `.venv`. The steps:

1. Open an issue on `anjor/gandalf`. State the need and the acceptance test.
2. Branch from `origin/main` as `study04/<slug>`.
3. Make the smallest change that works. Keep existing defaults and behaviour, unless the change is a bug fix.
4. Add tests. Run the full test suite locally. Never weaken or delete an existing test to get a pass.
5. Open a PR. Wait for green CI. This can span iterations.
6. Give the diff to the `gandalf-reviewer` subagent. Post its verdict on the PR. Fix what it finds, then have it review again.
7. Squash-merge, only when CI is green and the reviewer's APPROVE names the head commit being merged. Use `--match-head-commit`. Never use `--admin`. Never push to GANDALF `main` directly.
8. Pin `pyproject.toml` in this repo to the merge commit. Run `uv lock` and `uv sync`. Rerun this study's checks.
9. Log it. Add a row to `decisions.md`. List the PR under "For review".

One exception. The agent does not merge a PR that changes an existing test's expected values, or anything under `.github/`. It pins this repo to the PR's head commit, carries on, and lists the PR for Anjor. A bug fix that could change published results gets a prominent note for Anjor, naming the results at risk.

**Local runs.** The limits in `PLAN.md` §1 hold: up to 32³, M ≤ 64, under 20 minutes. Start long commands in the background with a log file and poll them.

**Never.** Change a frozen section. Edit or delete an earlier log entry. Use synthetic data in place of a run. Force-push. Launch anything on Modal outside the launcher. Send or publish anything outside the two repos. Edit another study's folder. Adding to `shared/` with tests is fine.

**The end.** The study is complete when the Phase 4 output exists, every claim in `claims.md` is resolved and the budgets close. Then write `STOP` with the reason "study complete".

### 3. The log

One file per day: `log/YYYY-MM-DD.md`. One entry per iteration, including iterations that only wait. Entries are append-only. A later entry corrects a wrong one.

```
## <iteration id>: <the block, in one line>
- Block: <PLAN.md reference>
- Start: commit <sha>, tree clean | recovered from stash
- Intent (written before the work):
- Expected:
- Did: <commands, scripts, files changed>
- Runs: <run IDs, where, wall time, gate output verbatim>
- Results: <numbers with units, figure paths>
- Critic: <what it was given, verdict verbatim>
- Claims: <IDs added or updated>
- Decisions: <what and why>
- GANDALF: <issue, PR, commit>
- Compute: <local minutes; the line printed by `modal_launch.py status`>
- Surprises and problems:
- For Anjor: <items raised, none of them blocking>
- Next: <next block and why>
- End: <UTC time>, outcome: done | WIP | waiting | hard stop
```

`STATUS.md` stays the short front page. It holds the next block, the queue, what is in flight, the compute used against the cap, and one line per iteration, newest first.

The runner also keeps the full session transcript under `.loop/transcripts/`. That folder is not committed.

### 4. The Modal launcher

`loop/modal_launch.py` is the only path to a GPU. It owns the one Modal function that asks for a GPU: one A100, no retries, timeout set by the launch command. The agent never writes its own Modal app. It supplies committed configs and the run code that the function calls.

- `launch` takes the configs, the run set and the timeout hours per run. It needs a clean, pushed tree and no `STOP`. It reserves one timeout per config. It refuses if hours used, plus hours reserved, plus the new reservation exceed the cap. It commits and pushes the reservation before it calls Modal. Then it starts one detached call per config and records the app ID. A detached call survives the end of the session.
- The function appends a start and an end time to the volume for each attempt. Modal can restart a preempted call, so run code must resume from its latest checkpoint. `reconcile` reads the hours from those records. Nobody types them in. If a record is missing, the full reservation is charged.
- `status` prints the cap, the hours used, the hours reserved and the hours left.
- `smoke` runs a tiny function with no GPU through the same path. It counts zero hours.

The cap lives in `loop/config.env`. The ledger lives in `compute_ledger.json` in the study folder. Only Anjor changes the cap.

Reuse the volume `krmhd-benchmark-vol`. Use `studies/02-collisionality-scan/scripts/modal_128_hermite.py` and `download_128_results.py` as references. The Modal image pins the same GANDALF commit as `pyproject.toml`. The old runner pins v0.5.0, so do not copy that line.

### 5. The reviewer subagents

Two project subagents in `.claude/agents/` at the repo root. Both start with a fresh context.

- `study04-critic` tries to falsify a claim, a gate result or an interpretation. It gets `SPEC.md`, the scripts and the raw outputs. It does not read the log, `STATUS.md` or the actor's conclusions. It may write and run its own checks. It returns SUPPORTED, REFUTED or INCONCLUSIVE, with its evidence.
- `gandalf-reviewer` reviews a GANDALF diff for physics and numerics. It checks conservation, reality conditions, dealiasing, unchanged defaults and test coverage. It runs the tests. It returns APPROVE or REQUEST CHANGES, with reasons and the commit it reviewed.

### 6. The runner

Shell scripts in `loop/`. They must run on macOS: bash 3.2, BSD tools, no GNU `timeout`.

`run_iteration.sh` runs one iteration, in the loop's clone.

Before the session:

- It takes a lock that stores its process ID. A lock whose process is gone is stale.
- It requires `main` and a clean tree. It pulls. Then it checks `STOP`, the wait time, the daily cap and the total cap.
- It checks that `gh` and `modal` are logged in.
- It compares running Modal apps with the ledger. A running app with the loop's name prefix that is not in the ledger writes `STOP`. The runner never stops a Modal app itself.

The session:

- One fresh headless session with `loop/prompt.md`, the iteration ID and a note on any stash. The iteration ID is `it-YYYYMMDD-HHMM` in UTC.
- The full transcript goes to `.loop/transcripts/`. One line per attempt goes to `.loop/runner.log`.
- The runner kills the session and its child processes after 120 minutes.

After the session:

- If the tree is dirty, it stashes the changes under the iteration ID.
- It checks that this iteration's log entry is closed, with its `End:` line. If not, it appends a stub with the exit code, the duration and the commits before and after.
- It pushes anything unpushed.
- It inspects the commits the iteration made. Each of these writes `STOP`: a change under `loop/`, `.claude/` or to `LOOP.md`; a change to a frozen section; a deleted or edited line under `log/`; a ledger change the launcher did not make. Record hashes of the frozen sections at setup to check against.
- Six iterations in a row that end as WIP or as a stub write `STOP`. Waiting does not count.
- After a session that fails within two minutes, the runner waits an hour. An auth error or a rate limit looks like this. Tune these numbers to how rate limits show up on this machine.

`run_forever.sh` runs iterations in a loop, with a pause between them. It keeps the Mac awake with `caffeinate`. It exits when `STOP` appears.

`wait.sh <minutes> <reason>` sets the wait time. The agent uses it while Modal runs or CI are in flight.

`pause.sh` and `resume.sh` are for Anjor, in his own checkout. Each makes a pushed commit that adds or removes `STOP`.

`config.env` holds the settings: the compute cap, the model, the effort level, the daily cap of 10 iterations, a total cap on iterations, an optional `--max-budget-usd` per session, the time limit and the pause.

`settings.json` holds the loop's permission rules. Suggested setup: `acceptEdits` with `--permission-prompts none` and an explicit allow and deny list. Allow `uv`, `git`, the `gh` issue, PR and run commands, read-only shell tools and read-only `modal` commands. Deny `modal run` and `modal deploy` in any form, including through `uv run`. Deny force-pushes, pushed tags, `gh pr merge --admin`, `gh release` and `gh repo delete`. Use `auto` mode only if the list proves too tight. Do not use `bypassPermissions`.

`prompt.md` holds the iteration prompt:

```
You are running one unattended iteration of the Study 04 autonomous loop in this repo.

Iteration ID: {{ITER_ID}}
Started (UTC): {{STARTED_UTC}}
GANDALF worktree: {{GANDALF_WORKTREE}}
Stash from a dead iteration: {{STASH_NOTE}}

Read studies/04-phase-space-helicity/LOOP.md and follow it. It is the full protocol for this iteration.

Nobody is watching this session and nobody can answer questions. Do one block of work. Log it. Commit and push. Then end your turn.
```

A starting point for the call. Check each flag against `claude --help` on this machine.

```
claude -p "$PROMPT" \
  --permission-mode acceptEdits --permission-prompts none \
  --settings studies/04-phase-space-helicity/loop/settings.json \
  --add-dir ~/repos/anjor/gandalf-study04 \
  --output-format stream-json --verbose
```

The runner is also the entry point for any other scheduler. A scheduled task can call `run_iteration.sh`.

### 7. Changes to existing files

- `PLAN.md`: rewrite §1, §5 and §6 to match the decisions, and point to `LOOP.md`. In §4 change only who passes each gate and who launches runs. Add the dated clarification under §2 from decision 4. Add a dated amendment note at the top. Leave the rest of §2, §3, §7 and the tasks in §4 as they are. Then record the hashes of the frozen sections.
- `SPEC.md`: update §4 and §9 where they name Anjor as the launcher or the decider. Add a run class for the new base state. Leave §7 as it is.
- `CLAUDE.md`: add one sentence to rule 1 and one to "What NOT to do". Study 04's loop changes GANDALF through PRs on the GANDALF repo, under `LOOP.md`.
- `QUESTIONS.md`: four sections. "Blocking". "Open, not blocking". "For review". "Answered". Give every question a permanent ID. Move questions 2 and 5 to "Answered", with decisions 4 and 5. Question 4, sending the collaborator paragraph, stays with Anjor and does not block.
- `claims.md`: update claim C7 for decision 4.
- `STATUS.md`: the next block is Gate 1 under the new rules. Gate 1 has no coded check yet. The loop writes one: the derivation scripts must pass their own checks. It gets critic verdicts for the supported claims that lack one. Then it passes or fails the gate. The queue then holds Phase 1, the H_ph-sp derivation and the new base state.
- `RUNS.md`: the header says the agent launches through the launcher.
- `decisions.md`: one row per decision above, and one for the loop design.
- `.gitignore`: add `.claude/settings.local.json` and `studies/04-phase-space-helicity/.loop/`.
- The first log entry records this setup session.
- Keep this file at `loop/SETUP_PROMPT.md`, as the record of the brief.

### 8. Tests

- A preflight check. `gh` is logged in and can merge on `anjor/gandalf`. Check this with `gh api`, not with a test PR. `modal` is logged in. `uv sync` works in the clone.
- `loop/selftest.sh` runs the runner against a stub `claude`, in a scratch repo with a scratch remote. Cases: a normal iteration, an entry left open, a dirty tree, a fast failure, a hang, `STOP`, the wait time, a stale lock, the daily cap, and each guard that writes `STOP`.
- Launcher tests with a stub `modal`: the cap arithmetic, refusal at the cap, refusal on a dirty tree, the reservation committed before the launch, and reconcile with and without a record.
- One real headless session with the loop's flags and settings, in a scratch clone, with a trivial prompt. It confirms that the flags are accepted, that an allowed command runs and that a direct `modal run` is denied.
- One real `smoke` launch. It proves the detached launch, the app ID capture, the volume write, the fetch and reconcile. It costs almost nothing.

## Facts about the CLI

Checked on 1 October 2026 against version 2.1.287 on another machine. The version here may differ.

From `claude --help`:

- `--permission-mode` takes `acceptEdits`, `auto`, `bypassPermissions`, `manual`, `dontAsk` and `plan`.
- `--permission-prompts none` denies anything that would prompt.
- `--settings <file>` loads additional settings. `--add-dir` grants access to another directory.
- `--output-format stream-json`, `--max-budget-usd`, `--model` and `--effort` exist.
- In `-p` mode a settings file that fails validation is silently ignored. So test that the rules take effect.

From a docs lookup, not checked on a machine:

- `--max-turns` still works but is hidden from `--help`.
- `--add-dir` does not load the other directory's `CLAUDE.md`.
- Subagents in `.claude/agents/*.md` are found automatically in `-p` runs and start with a fresh context.
- Bash rules take the form `Bash(uv run *)`. Deny beats allow. `Bash(git push --force*)` does not catch `git push -f`.
- A compound command may not match a rule. The loop should prefer single commands, `git -C` and `uv run --directory`.
- The Bash tool has a timeout. Long local runs go in the background.

`.claude/settings.local.json` in Anjor's checkout allows `Bash(modal:*)` and `Bash(git push:*)`. It is not committed, so the loop's clone will not have it. The loop's deny rules must block a direct `modal run` either way.

## Housekeeping

Three stale zero-byte files sit in `.git/` of Anjor's checkout: `REBASE_HEAD.lock`, `cowork-stale-index.lock` and `.__probe`. Cowork sessions left them. Check that no git process is running, then delete them.

## Rules for this session

- If something here conflicts with what you find on this machine, trust the machine. Tell Anjor and adjust.
- Ask Anjor before anything that costs money, apart from the `smoke` launch. Ask before anything that publishes.
- Apart from the clarification in decision 4, do not change `PLAN.md` §2 or §7, or `SPEC.md` §7.
- Commit on `main` and push.

## Done when

- The self-test and the launcher tests pass on this Mac.
- The real headless session confirms the flags and the permission rules.
- The `smoke` launch completes the round trip.
- The clone and the GANDALF worktree exist.
- Everything is committed and pushed.
- `STATUS.md` names Gate 1 under the new rules as the next block.

## Report to Anjor

- What you built and where.
- The test results.
- Anything you skipped or could not verify.
- The commands to start, watch, pause and resume the loop.
- What the first iteration will do.
- Two things only he can do. Set a spending limit in the Modal workspace, if his plan offers one. It is the only guard that does not depend on the loop. Protect `main` on `anjor/gandalf` so that it takes PRs with green CI only, and block force-pushes on `anjor/krmhd-research`.
