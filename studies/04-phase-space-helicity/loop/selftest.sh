#!/bin/bash
# selftest.sh: test the Study 04 runner against stub claude, modal and gh.
#
#     bash studies/04-phase-space-helicity/loop/selftest.sh
#
# Every case builds a scratch clone of a scratch bare remote under mktemp -d, holding copies
# of the real loop files with a test config.env (short time limits, stubs for claude, modal
# and gh) and the clone marker (git config s04.loopclone true), and runs the copied
# run_iteration.sh (or run_forever.sh, pause.sh, resume.sh, preflight.sh, guards.py) there.
# It never touches this repo's git state, real Modal, real GitHub or a real claude. Prints
# PASS/FAIL per case and a summary; exits 1 if any case failed.
# S04_SELFTEST_KEEP=1 keeps the scratch directory; S04_SELFTEST_ONLY="name ..." runs some cases.
#
# The 'sleep' processes that cases start carry a marker unique to this run (RUN_TAG), so two
# self-tests running at once never kill each other's processes.

set -u

SRC_LOOP=$(cd "$(dirname "$0")" && pwd -P) || exit 2
SRC_REPO=$(cd "$SRC_LOOP/../../.." && pwd -P) || exit 2
STUDY_REL="studies/04-phase-space-helicity"
PY="${S04_SELFTEST_PYTHON:-$SRC_REPO/.venv/bin/python}"
ONLY="${S04_SELFTEST_ONLY:-}"
KEEP="${S04_SELFTEST_KEEP:-}"

# A clean environment: no S04_ settings or git identity from the caller.
for v in $(env | sed -n 's/^\(S04_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_AUTHOR_NAME GIT_AUTHOR_EMAIL GIT_COMMITTER_NAME GIT_COMMITTER_EMAIL CLAUDECODE

[ -x "$PY" ] || { echo "selftest: no python at $PY (run uv sync)" >&2; exit 2; }
WORK=$(mktemp -d "${TMPDIR:-/tmp}/s04-selftest.XXXXXX") || exit 2
WORK=$(cd "$WORK" && pwd -P)
case "$WORK/" in "$SRC_REPO/"*) echo "selftest: scratch dir inside the repo" >&2; exit 2 ;; esac

# Git reads a scratch global config only: identity "anjor", no signing, no credential helpers.
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL="$WORK/gitconfig"
printf '[user]\n\tname = anjor\n\temail = anjor@example.invalid\n[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n[advice]\n\tdetachedHead = false\n[pull]\n\trebase = false\n' > "$GIT_CONFIG_GLOBAL"

TODAY=$(date -u +%Y%m%d)
TODAY_DASH=$(date -u +%Y-%m-%d)
ITER_N=0
PASSED=0
FAILED=0
FAIL_LINES=""
T_START=$(date +%s)
# This run's marker: 'sleep 1000.<RUN_TAG><case code>' processes belong to this run only.
RUN_TAG=$(printf '%05d%05d' $(( $$ % 100000 )) $(( RANDOM % 100000 )))

mk() { echo "1000.${RUN_TAG}$1"; }  # mk CODE: this run's marker for one case

kill_markers() {  # kill_markers TAG: kill the 'sleep 1000.<TAG>...' processes, and only those
  local p
  for p in $(pgrep -f "sleep 1000\\.$1" 2>/dev/null); do kill -KILL "$p" 2>/dev/null; done
}

cleanup() {
  kill_markers "$RUN_TAG"
  if [ -n "$KEEP" ]; then echo "selftest: scratch kept in $WORK"; else rm -rf "$WORK"; fi
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
# Scratch repos
# ---------------------------------------------------------------------------

build_template() {
  local t="$WORK/template" f
  mkdir -p "$t/$STUDY_REL/loop" "$t/$STUDY_REL/log" "$t/.claude/agents" "$WORK/gandalf-wt"
  git init -q "$WORK/gandalf-wt"
  echo "# Scratch CLAUDE.md for the loop self-test" > "$t/CLAUDE.md"
  printf '%s\n' '.venv/' '**/data/' '__pycache__/' '*.pyc' '.claude/settings.local.json' "$STUDY_REL/.loop/" > "$t/.gitignore"
  echo "# test subagent" > "$t/.claude/agents/x.md"
  for f in PLAN.md SPEC.md STATUS.md; do cp "$SRC_REPO/$STUDY_REL/$f" "$t/$STUDY_REL/$f" || return 1; done
  if [ -f "$SRC_REPO/$STUDY_REL/LOOP.md" ]; then cp "$SRC_REPO/$STUDY_REL/LOOP.md" "$t/$STUDY_REL/LOOP.md"
  else echo "# LOOP.md (placeholder for the self-test)" > "$t/$STUDY_REL/LOOP.md"; fi
  if grep -q '^## Blocking' "$SRC_REPO/$STUDY_REL/QUESTIONS.md" 2>/dev/null; then
    cp "$SRC_REPO/$STUDY_REL/QUESTIONS.md" "$t/$STUDY_REL/QUESTIONS.md"
  else
    printf '# Questions for Anjor\n\n## Blocking\n\nNone.\n\n## Open, not blocking\n\n## For review\n\n## Answered\n' > "$t/$STUDY_REL/QUESTIONS.md"
  fi
  for f in run_iteration.sh run_forever.sh wait.sh pause.sh resume.sh preflight.sh guards.py loopcommon.py prompt.md settings.json; do
    cp "$SRC_LOOP/$f" "$t/$STUDY_REL/loop/$f" || return 1
  done
  cat > "$t/$STUDY_REL/loop/config.env" <<EOF
# Test settings written by selftest.sh.
COMPUTE_CAP_A100_HOURS=20
CLAUDE_BIN="$SRC_LOOP/stubs/claude"
CLAUDE_MODEL=opus
CLAUDE_EFFORT=high
MAX_BUDGET_USD=
DAILY_ITERATION_CAP=3
TOTAL_ITERATION_CAP=400
WIP_STREAK_LIMIT=6
SESSION_TIME_LIMIT_SEC=6
PAUSE_BETWEEN_ITERATIONS_SEC=1
FAST_FAIL_SEC=3
FAST_FAIL_BACKOFF_SEC=3600
WATCHDOG_POLL_SEC=1
KILL_GRACE_SEC=2
GANDALF_WORKTREE="$WORK/gandalf-wt"
MODAL_APP_PREFIX=s04-loop
MODAL_VOLUME=krmhd-benchmark-vol
MODAL_VOLUME_ROOT=study04
STOP_ON_ANY_NEW_MODAL_APP=1
RATE_LIMIT_MARGIN_SEC=300
LOOP_GIT_NAME=krmhd-loop
LOOP_GIT_EMAIL=loop@example.invalid
MODAL_BIN="$SRC_LOOP/stubs/modal"
GH_BIN="$SRC_LOOP/stubs/gh"
PYTHON_BIN="$PY"
UV_BIN=uv
EOF
  printf '# Study 04 loop log — 2026-09-30\n\n## it-20260930-1200: seed entry for the self-test\n- Block: seed\n- Intent (written before the work): seed\n- End: 2026-09-30T12:30:00Z, outcome: done\n' \
    > "$t/$STUDY_REL/log/2026-09-30.md"
  "$PY" -c 'import sys; sys.path.insert(0, sys.argv[1]); import loopcommon as lc; lc.save_ledger(sys.argv[2], lc.empty_ledger())' \
    "$t/$STUDY_REL/loop" "$t/$STUDY_REL/compute_ledger.json" || return 1
  "$PY" "$t/$STUDY_REL/loop/guards.py" --repo "$t" frozen-record > /dev/null || return 1
  git -C "$t" init -q -b main
  git -C "$t" add -A
  git -C "$t" commit -q -m "seed" || return 1
  echo "Second commit, so that HEAD~1 exists." >> "$t/$STUDY_REL/STATUS.md"
  git -C "$t" commit -q -am "seed 2" || return 1
  git clone -q --bare "$t" "$WORK/template.git" || return 1
}

new_case() {  # new_case NAME: fresh remote + loop clone (with the clone marker). Sets C, CL, ST, CASE_NAME.
  CASE_NAME="$1"
  CASE_OK=1
  CASE_WHY=""
  OUT=""
  C="$WORK/case-$1"
  rm -rf "$C"
  mkdir -p "$C/marks" "$C/stub"
  cp -R "$WORK/template.git" "$C/remote.git"
  git clone -q "$C/remote.git" "$C/clone"
  git -C "$C/clone" config s04.loopclone true
  CL=$(cd "$C/clone" && pwd -P)
  ST="$CL/$STUDY_REL/.loop"
}

run_iter() {  # run_iter MODE [VAR=value...]: one run of the scratch runner. Sets RC, OUT, LAST_ID.
  local mode="$1"
  shift
  ITER_N=$((ITER_N + 1))
  LAST_ID="it-$TODAY-$(printf '%04d' "$ITER_N")"
  OUT="$C/out.$ITER_N"
  env S04_STUB_CLAUDE_MODE="$mode" S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" \
    S04_STUB_PYTHON="$PY" S04_TEST_ITER_ID="$LAST_ID" S04_STUB_WT="$WORK/gandalf-wt" "$@" \
    bash "$CL/$STUDY_REL/loop/run_iteration.sh" > "$OUT" 2>&1
  RC=$?
}

anjor_clone() {  # anjor_clone: a second clone acting as Anjor's checkout. Sets AN.
  AN="$C/anjor"
  [ -d "$AN" ] || git clone -q "$C/remote.git" "$AN"
  git -C "$AN" pull -q --ff-only origin main
}

gp() { "$PY" -I -S "$CL/$STUDY_REL/loop/guards.py" --repo "$CL" "$@"; }  # gp ARGS: guards.py in the case clone

loop_commit() {  # loop_commit MESSAGE: commit everything in the clone as the loop would
  git -C "$CL" add -A
  GIT_AUTHOR_NAME=krmhd-loop GIT_AUTHOR_EMAIL=loop@example.invalid \
  GIT_COMMITTER_NAME=krmhd-loop GIT_COMMITTER_EMAIL=loop@example.invalid \
    git -C "$CL" commit -q -m "$1"
}

anjor_commit() {  # anjor_commit MESSAGE: commit everything in the clone as Anjor
  git -C "$CL" add -A
  git -C "$CL" commit -q -m "$1"
}

charged_ledger() {  # charged_ledger DIR: Anjor's ledger with one charged GPU run (7.5 h), committed and pushed
  "$PY" - "$1/$STUDY_REL" <<'EOF' || return 1
import sys
sys.path.insert(0, sys.argv[1] + "/loop")
import loopcommon as lc
p = sys.argv[1] + "/compute_ledger.json"
L = lc.load_ledger(p)
L["launches"].append({"launch_id": "L001", "kind": "gpu", "run_set": "A", "app_name": "s04-loop-l001-a",
    "app_id": "ap-known", "state": "closed", "frozen": {}, "timeout_hours": 10.0,
    "runs": [{"run_id": "04_a_x", "reserved_hours": 10.0, "status": "finished", "charged_hours": 7.5, "attempts": []}]})
lc.save_ledger(p, L)
EOF
  (cd "$1" && git commit -q -am "Study 04 [launcher]: reconcile L001 (by Anjor's hand, for the test)" && git push -q origin main)
}

anjor_ledger() {  # anjor_ledger DIR reserved|launched: Anjor's ledger with an earlier launch L001, committed and pushed
  "$PY" - "$1/$STUDY_REL" "$2" <<'EOF' || return 1
import sys
sys.path.insert(0, sys.argv[1] + "/loop")
import loopcommon as lc
p = sys.argv[1] + "/compute_ledger.json"
L = lc.load_ledger(p)
if sys.argv[2] == "reserved":
    # Stuck after a launcher crash: reserved, no app ID, two runs of 5 h.
    L["launches"].append({"launch_id": "L001", "kind": "gpu", "run_set": "A", "app_name": "s04-loop-l001-a",
        "app_id": None, "state": "reserved", "frozen": {}, "timeout_hours": 5.0,
        "runs": [{"run_id": "04_a_%d" % i, "reserved_hours": 5.0, "status": "reserved", "charged_hours": None,
                  "call_id": None, "attempts": []} for i in (1, 2)]})
else:
    L["launches"].append({"launch_id": "L001", "kind": "gpu", "run_set": "A", "app_name": "s04-loop-l001-a",
        "app_id": "ap-l001", "state": "launched", "frozen": {}, "timeout_hours": 8.0,
        "runs": [{"run_id": "04_a_x", "reserved_hours": 8.0, "status": "running", "charged_hours": None,
                  "call_id": "fc-1", "attempts": []}]})
lc.save_ledger(p, L)
EOF
  (cd "$1" && git commit -q -am "Study 04 [launcher]: L001 as an earlier session left it (by Anjor's hand, for the test)" \
     && git push -q origin main)
}

# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

fail() { CASE_OK=0; CASE_WHY="${CASE_WHY:+$CASE_WHY; }$*"; }
expect_rc() { [ "$RC" = "$1" ] || fail "exit code $RC, expected $1"; }
remote_has() { git --git-dir="$C/remote.git" cat-file -e "main:$1" 2>/dev/null; }
remote_show() { git --git-dir="$C/remote.git" show "main:$1" 2>/dev/null; }
remote_subjects() { git --git-dir="$C/remote.git" log --format=%s main; }
called() { if [ -f "$C/marks/claude_called" ]; then wc -l < "$C/marks/claude_called" | tr -d ' '; else echo 0; fi; }
expect_called() { [ "$(called)" = "$1" ] || fail "claude called $(called) times, expected $1"; }
lastlog() { tail -1 "$ST/runner.log" 2>/dev/null; }
expect_lastlog() { lastlog | grep -q -- "$1" || fail "runner.log lacks '$1': $(lastlog)"; }
expect_stop() {  # expect_stop TEXT: STOP on the remote, containing TEXT
  if ! remote_has "$STUDY_REL/STOP"; then fail "no STOP on the remote"; return; fi
  remote_show "$STUDY_REL/STOP" | grep -q -- "$1" || fail "STOP lacks '$1': $(remote_show "$STUDY_REL/STOP" | sed -n 3,8p | tr '\n' ' ')"
}
expect_no_stop() { if remote_has "$STUDY_REL/STOP"; then fail "unexpected STOP: $(remote_show "$STUDY_REL/STOP" | sed -n 3,6p | tr '\n' ' ')"; fi; }
expect_synced() {
  [ -z "$(git -C "$CL" status --porcelain)" ] || fail "clone tree dirty: $(git -C "$CL" status --porcelain | head -3 | tr '\n' ' ')"
  [ "$(git -C "$CL" rev-parse HEAD)" = "$(git --git-dir="$C/remote.git" rev-parse main)" ] || fail "clone and remote differ"
  [ ! -d "$ST/lock" ] || fail "lock not released"
  [ ! -f "$ST/inflight" ] || fail "inflight file left behind"
}
day_log() { remote_show "$STUDY_REL/log/$TODAY_DASH.md"; }
replace_line() {  # replace_line FILE OLD NEW: replace a whole line (literal match)
  awk -v old="$2" -v new="$3" '{ if ($0 == old) print new; else print }' "$1" > "$1.tmp" && mv "$1.tmp" "$1"
}
no_marker() {  # no_marker M: no 'sleep M...' process alive
  local p
  p=$(pgrep -f "sleep $(echo "$1" | sed 's/\./\\./g')" 2>/dev/null | tr '\n' ' ')
  [ -z "$p" ] || fail "processes still alive for marker $1: $p"
}
stash_ref_for() {  # stash_ref_for TEXT: the stash@{n} whose message contains TEXT
  git -C "$CL" stash list --format='%gd %gs' | grep -F -- "$1" | head -1 | cut -d' ' -f1
}
stash_has() {  # stash_has TEXT CONTENT: the stash named TEXT exists and its diff contains CONTENT
  local ref
  ref=$(stash_ref_for "$1")
  if [ -z "$ref" ]; then fail "no stash named '$1': $(git -C "$CL" stash list | tr '\n' ' ')"; return; fi
  git -C "$CL" stash show -p --include-untracked "$ref" 2>/dev/null | grep -q -- "$2" \
    || git -C "$CL" stash show --include-untracked --name-only "$ref" 2>/dev/null | grep -q -- "$2" \
    || fail "stash '$1' lacks '$2'"
}

finish_case() {
  if [ "$CASE_OK" = 1 ]; then
    echo "PASS $CASE_NAME"
    PASSED=$((PASSED + 1))
  else
    echo "FAIL $CASE_NAME: $CASE_WHY"
    FAILED=$((FAILED + 1))
    FAIL_LINES="$FAIL_LINES
FAIL $CASE_NAME: $CASE_WHY"
    if [ -n "${OUT:-}" ] && [ -f "$OUT" ]; then sed 's/^/    | /' "$OUT" | tail -15; fi
  fi
}

want() {  # want NAME: run this case? (S04_SELFTEST_ONLY filter)
  [ -z "$ONLY" ] && return 0
  case " $ONLY " in *" $1 "*) return 0 ;; esac
  return 1
}

# ---------------------------------------------------------------------------
# Cases: one iteration
# ---------------------------------------------------------------------------

case_normal() {
  local args envf flag sig
  new_case normal
  run_iter normal
  expect_rc 0
  expect_called 1
  day_log | grep -q "^## $LAST_ID: stub block (normal)" || fail "agent entry not on the remote"
  day_log | grep -q "runner stub" && fail "unexpected stub entry"
  expect_no_stop
  expect_lastlog " $LAST_ID ran exit=0 "
  expect_lastlog "outcome=done"
  expect_lastlog "guards=ok"
  expect_lastlog "util_5h=0.35"
  expect_lastlog "util_7d=0.28"
  expect_synced
  args="$C/marks/claude_args.$LAST_ID"
  for flag in -p --model opus --effort high --permission-mode acceptEdits --permission-prompts none \
              --settings --setting-sources project --strict-mcp-config --disable-slash-commands \
              --no-chrome --add-dir "$WORK/gandalf-wt" --output-format stream-json --verbose; do
    grep -qx -- "$flag" "$args" || fail "claude did not get '$flag'"
  done
  grep -A1 -x -- '--setting-sources' "$args" | sed -n 2p | grep -qx project || fail "--setting-sources is not exactly 'project'"
  grep -q -- '--max-budget-usd' "$args" && fail "--max-budget-usd passed with MAX_BUDGET_USD empty"
  grep -q "^Iteration ID: $LAST_ID\$" "$args" || fail "prompt lacks the iteration ID"
  grep -q "^Stash from a dead iteration: none\$" "$args" || fail "prompt stash note is not 'none'"
  grep -q '{{' "$args" && fail "prompt has unfilled placeholders"
  envf="$C/marks/claude_env.$LAST_ID"
  for flag in "S04_ITER_ID=$LAST_ID" S04_LOOP=1 CLAUDE_CODE_DISABLE_AUTO_MEMORY=1 \
              ENABLE_CLAUDEAI_MCP_SERVERS=false CLAUDE_CODE_DISABLE_ARTIFACT=1 \
              GIT_AUTHOR_NAME=krmhd-loop GIT_COMMITTER_NAME=krmhd-loop; do
    grep -qx -- "$flag" "$envf" || fail "session env lacks $flag"
  done
  for sig in PIPE INT QUIT XFSZ; do
    grep -qx "$sig=DEFAULT" "$C/marks/claude_signals.$LAST_ID" 2>/dev/null \
      || fail "claude did not start with SIG$sig at its default: $(tr '\n' ' ' < "$C/marks/claude_signals.$LAST_ID" 2>/dev/null)"
  done
  [ "$(cat "$C/marks/claude_cwd.$LAST_ID")" = "$CL" ] || fail "session cwd is not the clone"
  cmp -s "$C/marks/claude_settings.$LAST_ID" "$CL/$STUDY_REL/loop/settings.json" \
    || fail "the --settings file claude got differs from loop/settings.json"
  grep -A1 -x -- '--settings' "$args" | grep -q -- "$CL/" \
    && fail "--settings points into the clone, not at the runner's trusted copy"
  [ "$(git --git-dir="$C/remote.git" log -1 --format=%an/%cn main)" = krmhd-loop/krmhd-loop ] || fail "agent commit not by krmhd-loop"
  [ "$(git -C "$CL" config --local --get user.name)" = krmhd-loop ] || fail "the runner did not set user.name in the clone"
  grep -qxF "/$STUDY_REL/.loop/" "$CL/.git/info/exclude" || fail ".loop/ not in .git/info/exclude"
  [ -s "$ST/transcripts/$LAST_ID.jsonl" ] || fail "no transcript"
  grep -q '"type":"result"' "$ST/transcripts/$LAST_ID.jsonl" || fail "transcript lacks the result line"
  grep -q "  $LAST_ID.jsonl\$" "$ST/transcripts.sha256" 2>/dev/null || fail "the transcript was not sealed in transcripts.sha256"
  [ ! -f "$ST/git-config.pre.$LAST_ID" ] || fail "the .git/config snapshot was left behind"
  gp session-check --transcript "$ST/transcripts/$LAST_ID.jsonl" | grep -qx 'class=none' \
    || fail "session-check classes a good session as a failure: $(gp session-check --transcript "$ST/transcripts/$LAST_ID.jsonl" | tr '\n' ' ')"
  finish_case
}

case_not_loop_clone() {
  local t0
  new_case not_loop_clone
  git -C "$CL" config --unset s04.loopclone
  run_iter normal
  expect_rc 2
  expect_called 0
  expect_no_stop
  grep -q "not the loop's clone" "$OUT" || fail "no message about the missing marker"
  [ -d "$ST" ] && fail "the runner created .loop/ in a checkout that is not the loop's clone"
  t0=$(date +%s)
  env S04_STUB_MARKS="$C/marks" bash "$CL/$STUDY_REL/loop/run_forever.sh" > "$C/forever.out" 2>&1
  [ "$?" = 2 ] || fail "run_forever.sh did not refuse (exit 2) outside the loop clone"
  [ $(( $(date +%s) - t0 )) -le 5 ] || fail "run_forever.sh took too long to refuse"
  OUT="$C/preflight.out"
  env S04_STUB_DIR="$C/stub" bash "$CL/$STUDY_REL/loop/preflight.sh" --skip-uv --allow-stubs > "$OUT" 2>&1
  grep -q '^FAIL  git config s04.loopclone' "$OUT" || fail "preflight did not flag the missing marker"
  finish_case
}

case_open_entry() {
  new_case open_entry
  run_iter open S04_MAX_BUDGET_USD=5
  expect_rc 0
  day_log | grep -q "^## $LAST_ID: runner stub (log entry not closed)" || fail "no stub entry on the remote"
  day_log | tail -3 | grep -q "outcome: stub" || fail "stub entry not closed with outcome: stub"
  remote_subjects | grep -q "runner stub (log entry not closed)" || fail "no stub commit"
  expect_lastlog "outcome=stub"
  grep -qx -- '--max-budget-usd' "$C/marks/claude_args.$LAST_ID" && grep -qx 5 "$C/marks/claude_args.$LAST_ID" \
    || fail "--max-budget-usd 5 not passed"
  expect_no_stop
  expect_synced
  finish_case
}

case_dirty_tree() {
  local first
  new_case dirty_tree
  run_iter dirty
  first=$LAST_ID
  expect_rc 0
  git -C "$CL" stash list | grep -q "loop-stash $first" || fail "no stash named loop-stash $first"
  git -C "$CL" stash show --include-untracked --name-only stash@{0} 2>/dev/null | grep -q scratch_dirty.txt \
    || fail "stash lacks the untracked file"
  remote_subjects | grep -q "runner saved uncommitted log lines" || fail "uncommitted log lines not saved"
  day_log | grep -q "^## $first: stub block (dirty)" || fail "the session's own entry was not kept"
  day_log | grep -q "^## $first: runner stub" || fail "no stub entry"
  [ -f "$ST/backoff_until" ] || fail "fast failure did not set a backoff"
  expect_synced
  # The next session is told about the stash.
  rm -f "$ST/backoff_until"
  run_iter normal
  expect_rc 0
  grep -q "loop-stash $first" "$C/marks/claude_args.$LAST_ID" || fail "next prompt does not name the stash"
  finish_case
}

case_fast_failure() {
  local until now
  new_case fast_failure
  run_iter fastfail
  expect_rc 0
  until=$(cat "$ST/backoff_until" 2>/dev/null || echo 0)
  now=$(date +%s)
  [ "$until" -gt $((now + 3000)) ] || fail "backoff_until not about an hour ahead ($until)"
  expect_lastlog "fast_fail=rate_limit"
  day_log | grep -q "^## $LAST_ID: runner stub" || fail "no stub for the failed session"
  run_iter normal
  expect_rc 10
  expect_lastlog "skip-backoff"
  expect_called 1
  finish_case
}

case_rate_limited() {
  local until now
  new_case rate_limited
  # A reset two hours ahead: the backoff runs to the reset plus the margin, past the hour.
  run_iter ratelimited S04_STUB_RESETS_IN=7200
  expect_rc 0
  now=$(date +%s)
  until=$(cat "$ST/backoff_until" 2>/dev/null || echo 0)
  if [ "$until" -lt $((now + 7200 + 300 - 30)) ] || [ "$until" -gt $((now + 7200 + 300 + 30)) ]; then
    fail "backoff_until $until is not the reset plus the margin ($((now + 7500)))"
  fi
  expect_lastlog "rate_limited=rejected/five_hour"
  expect_lastlog "util_5h=1 "
  expect_lastlog "util_7d=0.62"
  run_iter normal
  expect_rc 10
  expect_lastlog "skip-backoff"
  # A reset one minute ahead: the backoff is still at least FAST_FAIL_BACKOFF_SEC.
  rm -f "$ST/backoff_until"
  run_iter ratelimited S04_STUB_RESETS_IN=60
  now=$(date +%s)
  until=$(cat "$ST/backoff_until" 2>/dev/null || echo 0)
  if [ "$until" -lt $((now + 3600 - 30)) ] || [ "$until" -gt $((now + 3600 + 30)) ]; then
    fail "backoff_until $until is not now + FAST_FAIL_BACKOFF_SEC for a near reset"
  fi
  # A long session that ends on a usage limit backs off too.
  rm -f "$ST/backoff_until"
  run_iter ratelimited S04_STUB_RESETS_IN=7200 S04_STUB_SLEEP=4
  now=$(date +%s)
  until=$(cat "$ST/backoff_until" 2>/dev/null || echo 0)
  [ "$until" -gt $((now + 7000)) ] || fail "a usage limit after ${FAST_FAIL_SEC:-3}s set no backoff ($until)"
  expect_called 3
  finish_case
}

case_hang() {
  local t0 t1 m d
  m=$(mk 131)
  new_case hang
  t0=$(date +%s)
  run_iter hang S04_STUB_MARKER="$m"
  t1=$(date +%s)
  expect_rc 0
  [ $((t1 - t0)) -le 40 ] || fail "the runner took $((t1 - t0))s in all for a 6s session limit"
  d=$(lastlog | sed -n 's/.* duration=\([0-9]*\)s .*/\1/p')
  [ -n "$d" ] && [ "$d" -le 12 ] || fail "the session ran ${d:-?}s for a 6s limit (kill grace 2s)"
  no_marker "$m"
  expect_lastlog "timeout=1"
  lastlog | grep -q "killed=[1-9]" || fail "no processes counted as killed: $(lastlog)"
  day_log | grep -q "^- Runner: .*note timeout" || fail "stub does not say timeout"
  expect_synced
  finish_case
}

case_leftover_process() {
  local m
  m=$(mk 141)
  new_case leftover_process
  # The production poll interval: the orphans appear and lose their parent between two
  # snapshots, so only the working-directory sweep can find them.
  run_iter leftover S04_STUB_MARKER="$m" S04_WATCHDOG_POLL_SEC=15
  expect_rc 0
  no_marker "$m"
  lastlog | grep -q "killed=[2-9]" || fail "orphans in the clone and the worktree not both killed: $(lastlog)"
  expect_no_stop
  finish_case
}

case_marker_isolation() {
  local other p
  new_case marker_isolation
  other=$(printf '%05d%05d' $(( ($$ + 1) % 100000 )) 4242)
  [ "$other" = "$RUN_TAG" ] && other="9999999999"
  sleep "1000.${other}131" &
  p=$!
  kill_markers "$RUN_TAG"
  sleep 0.3
  kill -0 "$p" 2>/dev/null || fail "this run's cleanup killed another run's marker process"
  kill -KILL "$p" 2>/dev/null
  wait "$p" 2>/dev/null
  finish_case
}

case_stop_present() {
  new_case stop_present
  anjor_clone
  (cd "$AN" && bash "$STUDY_REL/loop/pause.sh" "self-test pause") > "$C/pause.out" 2>&1 || fail "pause.sh failed: $(tail -1 "$C/pause.out")"
  expect_stop "Paused by Anjor: self-test pause"
  remote_show "$STUDY_REL/QUESTIONS.md" | grep -q '^- \*\*STOP .* (anjor)\*\*' || fail "no Blocking line from pause.sh"
  [ "$(git --git-dir="$C/remote.git" log -1 --format=%an main)" = anjor ] || fail "pause commit not Anjor's"
  run_iter normal
  expect_rc 12
  expect_called 0
  expect_lastlog "skip-stop"
  (cd "$AN" && bash "$STUDY_REL/loop/pause.sh") > "$C/pause2.out" 2>&1 && fail "pause.sh did not refuse a second pause"
  (cd "$AN" && bash "$STUDY_REL/loop/resume.sh") > "$C/resume.out" 2>&1 || fail "resume.sh failed: $(tail -1 "$C/resume.out")"
  expect_no_stop
  run_iter normal
  expect_rc 0
  expect_called 1
  finish_case
}

case_wait_time() {
  local far
  new_case wait_time
  anjor_clone
  (cd "$AN" && bash "$STUDY_REL/loop/wait.sh" 3000 "too long") > /dev/null 2>&1 && fail "wait.sh accepted 3000 minutes"
  (cd "$AN" && bash "$STUDY_REL/loop/wait.sh" abc "bad") > /dev/null 2>&1 && fail "wait.sh accepted 'abc'"
  (cd "$AN" && bash "$STUDY_REL/loop/wait.sh" 30 "runs in flight" && git add -A && git commit -q -m "wait" && git push -q origin main) \
    > "$C/wait.out" 2>&1 || fail "wait.sh 30 failed"
  run_iter normal
  expect_rc 10
  expect_called 0
  expect_lastlog "skip-wait"
  expect_lastlog "runs_in_flight"
  # A WAIT further ahead than wait.sh allows did not come from wait.sh: ignored.
  far=$(( $(date +%s) + 10 * 86400 ))
  (cd "$AN" && printf 'until_epoch=%s\nreason=hand written\n' "$far" > "$STUDY_REL/WAIT" && git commit -q -am "far wait" && git push -q origin main) \
    > /dev/null 2>&1 || fail "could not push the far WAIT"
  run_iter normal
  expect_rc 0
  expect_called 1
  (cd "$AN" && git pull -q --ff-only origin main && bash "$STUDY_REL/loop/wait.sh" 0 && [ ! -e "$STUDY_REL/WAIT" ]) > /dev/null 2>&1 \
    || fail "wait.sh 0 did not remove WAIT"
  finish_case
}

case_stale_lock() {
  local dead
  new_case stale_lock
  mkdir -p "$ST/lock"
  bash -c 'exit 0' &
  dead=$!
  wait "$dead"
  echo "$dead" > "$ST/lock/pid"
  run_iter normal
  expect_rc 0
  expect_called 1
  expect_lastlog "stale_lock=$dead"
  expect_synced
  [ ! -d "$ST/lock.takeover" ] || fail "the takeover lock was left behind"
  finish_case
}

case_stale_lock_race() {
  local dead p1 p2 rc1 rc2 id1 id2
  new_case stale_lock_race
  mkdir -p "$ST/lock"
  bash -c 'exit 0' &
  dead=$!
  wait "$dead"
  echo "$dead" > "$ST/lock/pid"
  ITER_N=$((ITER_N + 1)); id1="it-$TODAY-$(printf '%04d' "$ITER_N")"
  ITER_N=$((ITER_N + 1)); id2="it-$TODAY-$(printf '%04d' "$ITER_N")"
  # Both runners read the dead pid, then wait 2 s before taking the lock over.
  env S04_STUB_CLAUDE_MODE=normal S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" S04_STUB_PYTHON="$PY" \
    S04_TEST_ITER_ID="$id1" S04_TEST_LOCK_DELAY=2 bash "$CL/$STUDY_REL/loop/run_iteration.sh" > "$C/race1.out" 2>&1 &
  p1=$!
  env S04_STUB_CLAUDE_MODE=normal S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" S04_STUB_PYTHON="$PY" \
    S04_TEST_ITER_ID="$id2" S04_TEST_LOCK_DELAY=2 bash "$CL/$STUDY_REL/loop/run_iteration.sh" > "$C/race2.out" 2>&1 &
  p2=$!
  wait "$p1"; rc1=$?
  wait "$p2"; rc2=$?
  case "$rc1/$rc2" in
    0/13|13/0) ;;
    *) fail "runner exit codes $rc1 and $rc2; exactly one should win (0) and one back off (13)" ;;
  esac
  expect_called 1
  [ ! -d "$ST/lock" ] || fail "lock not released"
  [ ! -d "$ST/lock.takeover" ] || fail "the takeover lock was left behind"
  finish_case
}

case_live_lock() {
  local holder
  new_case live_lock
  mkdir -p "$ST/lock"
  sleep 30 &
  holder=$!
  echo "$holder" > "$ST/lock/pid"
  ps -o lstart= -p "$holder" | awk '{$1 = $1; print}' > "$ST/lock/lstart"
  run_iter normal
  expect_rc 13
  expect_called 0
  expect_lastlog "skip-lock"
  [ "$(cat "$ST/lock/pid" 2>/dev/null)" = "$holder" ] || fail "the live lock was taken over"
  kill "$holder" 2>/dev/null
  wait "$holder" 2>/dev/null
  finish_case
}

case_daily_cap() {
  local day
  new_case daily_cap
  mkdir -p "$ST/transcripts"
  day=$(date -u +%Y%m%d)  # the runner's 'today', even if this run crossed midnight UTC
  touch "$ST/transcripts/it-$day-9001.jsonl" "$ST/transcripts/it-$day-9002.jsonl" "$ST/transcripts/it-$day-9003.jsonl"
  run_iter normal
  expect_rc 11
  expect_called 0
  expect_lastlog "skip-daily-cap"
  finish_case
}

case_daily_cap_from_log() {
  local day f
  new_case daily_cap_from_log
  day=$(date -u +%Y%m%d)
  f="$CL/$STUDY_REL/log/$(date -u +%Y-%m-%d).md"
  # Three entries of today in the committed log, and no transcripts (someone cleared .loop/).
  {
    printf '# Study 04 loop log — %s\n' "$(date -u +%Y-%m-%d)"
    for n in 9001 9002 9003; do printf '\n## it-%s-%s: earlier\n- Block: x\n- End: %s, outcome: done\n' "$day" "$n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"; done
  } > "$f"
  anjor_commit "Log entries of today"
  git -C "$CL" push -q origin main
  run_iter normal
  expect_rc 11
  expect_called 0
  expect_lastlog "skip-daily-cap"
  finish_case
}

case_total_cap() {
  new_case total_cap
  mkdir -p "$ST/transcripts"
  touch "$ST/transcripts/it-20260101-0001.jsonl" "$ST/transcripts/it-20260101-0002.jsonl"
  run_iter normal S04_TOTAL_ITERATION_CAP=2
  expect_rc 12
  expect_called 0
  expect_stop "Total iteration cap reached"
  finish_case
}

case_auth_failure() {
  new_case auth_failure
  run_iter normal S04_STUB_GH_RC=1
  expect_rc 14
  expect_called 0
  expect_lastlog "skip-auth"
  [ "$(cat "$ST/backoff_until" 2>/dev/null || echo 0)" -gt "$(date +%s)" ] || fail "no backoff after gh failure"
  rm -f "$ST/backoff_until"
  touch "$C/stub/modal_fail"
  run_iter normal
  expect_rc 14
  expect_called 0
  expect_lastlog "modal_app_list_failed"
  expect_no_stop
  finish_case
}

case_unknown_app() {
  new_case unknown_app
  echo '[{"App ID": "ap-x", "Description": "s04-loop-l999-a", "State": "ephemeral (detached)", "Tasks": "1", "Created at": "2026-01-01 00:00:00+00:00", "Stopped at": null}]' \
    > "$C/stub/apps.json"
  run_iter normal
  expect_rc 12
  expect_called 0
  expect_stop "ap-x"
  grep -q "app stop" "$C/stub/modal_calls" 2>/dev/null && fail "the runner tried to stop a Modal app"
  finish_case
}

case_known_app() {
  new_case known_app
  anjor_clone
  "$PY" - "$AN/$STUDY_REL" <<'EOF' || fail "could not write the ledger"
import sys
sys.path.insert(0, sys.argv[1] + "/loop")
import loopcommon as lc
p = sys.argv[1] + "/compute_ledger.json"
L = lc.load_ledger(p)
L["launches"].append({"launch_id": "L001", "kind": "gpu", "run_set": "A", "app_name": "s04-loop-l001-a",
    "app_id": "ap-known", "state": "launched", "frozen": {}, "timeout_hours": 1.0,
    "runs": [{"run_id": "04_a_x", "reserved_hours": 1.0, "status": "running", "charged_hours": None, "attempts": []}]})
lc.save_ledger(p, L)
EOF
  (cd "$AN" && git commit -q -am "Study 04 [launcher]: launched L001 set A app ap-known" && git push -q origin main) || fail "push failed"
  echo '[{"App ID": "ap-known", "Description": "s04-loop-l001-a", "State": "ephemeral (detached)", "Tasks": "1", "Created at": "2026-01-01 00:00:00+00:00", "Stopped at": null}]' \
    > "$C/stub/apps.json"
  run_iter normal
  expect_rc 0
  expect_called 1
  expect_no_stop
  finish_case
}

case_app_during_session() {
  new_case app_during_session
  run_iter modal_rogue
  expect_rc 12
  expect_stop "ap-rogue"
  expect_lastlog "guards=STOP"
  finish_case
}

case_app_prefix_only() {
  new_case app_prefix_only
  # STOP_ON_ANY_NEW_MODAL_APP=0: an app made during a session counts only with the loop's prefix.
  run_iter app_unprefixed S04_STOP_ON_ANY_NEW_MODAL_APP=0
  expect_rc 0
  expect_no_stop
  run_iter app_prefixed S04_STOP_ON_ANY_NEW_MODAL_APP=0
  expect_rc 12
  expect_stop "ap-rogue s04-loop-handmade"
  finish_case
}

guard_case() {  # guard_case MODE TEXT...: a session that breaks a guard ends in STOP containing each TEXT
  local mode="$1" text
  shift
  new_case "guard_$mode"
  run_iter "$mode"
  expect_rc 12
  expect_called 1
  for text in "$@"; do expect_stop "$text"; done
  remote_subjects | grep -q "runner STOP" || fail "no runner STOP commit on the remote"
  expect_lastlog "guards=STOP"
  expect_synced
  finish_case
}

allowed_case() {  # allowed_case MODE: a session that must not trip any guard
  new_case "allowed_$1"
  run_iter "$1"
  expect_rc 0
  expect_no_stop
  expect_lastlog "guards=ok"
  expect_lastlog "outcome=done"
  expect_synced
  finish_case
}

case_guard_ledger_forged() {
  new_case guard_ledger_forged
  anjor_clone
  charged_ledger "$AN" || fail "could not set up the charged ledger"
  run_iter ledger_forged
  expect_rc 12
  expect_stop "not a launcher-made successor"
  expect_stop "04_a_x: changed after it was charged"
  expect_synced
  finish_case
}

case_guard_ledger_deleted() {
  new_case guard_ledger_deleted
  run_iter ledger_deleted
  expect_rc 12
  expect_stop "compute_ledger.json deleted"
  expect_synced
  finish_case
}

case_transcript_t2_earlier() {
  new_case transcript_t2_earlier
  run_iter review_only
  expect_rc 0
  expect_no_stop
  run_iter merge_only
  expect_rc 0
  expect_no_stop
  expect_lastlog "guards=ok"
  finish_case
}

case_transcript_t3_denied() {
  new_case transcript_t3_denied
  run_iter uv_modal_denied
  expect_rc 0
  expect_no_stop
  stash_has "loop-stash $LAST_ID" "sneaky_run.py"
  day_log | grep -q "^## $LAST_ID: runner stub (work left uncommitted)" || fail "no stub naming the stashed work"
  finish_case
}

case_transcript_t3() {
  new_case transcript_t3
  run_iter uv_modal
  expect_rc 12
  expect_stop "T3 the session ran studies/04-phase-space-helicity/analysis/sneaky_run.py"
  stash_has "loop-stash $LAST_ID" "sneaky_run.py"
  expect_synced
  finish_case
}

case_transcript_gone() {
  new_case transcript_gone
  # The session commits and then removes its transcript: nothing it did can be checked.
  run_iter transcript_gone
  expect_rc 12
  expect_stop "transcript: the session's transcript is missing or unreadable"
  finish_case
}

case_guard_ledger_semantics() {  # launcher-subject ledger commits the launcher would never make
  local mode setup want
  for mode in ledger_release_old ledger_forge_charge ledger_no_launcher_call; do
    case "$mode" in
      ledger_release_old) setup=reserved; want="reserved before this session" ;;
      ledger_forge_charge) setup=launched; want="with no attempt records" ;;
      ledger_no_launcher_call) setup=reserved; want="T4 " ;;
    esac
    new_case "guard_$mode"
    anjor_clone
    anjor_ledger "$AN" "$setup" || fail "could not set up the ledger"
    run_iter "$mode"
    expect_rc 12
    expect_stop "$want"
    expect_lastlog "guards=STOP"
    expect_synced
    finish_case
  done
}

case_guard_report_edit() {  # a loop commit edits a gate report an earlier session committed
  new_case guard_report_edit
  mkdir -p "$CL/$STUDY_REL/gate_reports"
  printf '# Gate 2 report: it-20260930-1200\n\n| Quantity | Value | Threshold | Pass |\n|---|---|---|---|\n| slope | 0.12 | 0.05 | no |\n\nCoded check: FAIL\nResult: FAIL\n' \
    > "$CL/$STUDY_REL/gate_reports/G2_it-20260930-1200.md"
  loop_commit "Study 04 [it-20260930-1200]: Gate 2 report"
  git -C "$CL" push -q origin main
  run_iter report_edit
  expect_rc 12
  expect_stop "report-edit: $STUDY_REL/gate_reports/G2_it-20260930-1200.md (M)"
  expect_synced
  finish_case
}

case_stop_removed() {  # Anjor pauses mid-session; the session deletes the pulled STOP without a commit
  local how
  for how in staged unstaged; do
    new_case "stop_removed_$how"
    run_iter "stop_removed_$how"
    expect_rc 12
    expect_called 1
    expect_stop "stop-deleted (uncommitted)"
    expect_stop "Paused by Anjor mid-session"
    [ -f "$CL/$STUDY_REL/STOP" ] || fail "STOP is not back in the working tree"
    git -C "$CL" ls-files --error-unmatch "$STUDY_REL/STOP" > /dev/null 2>&1 || fail "STOP is not back in the index"
    expect_synced
    run_iter normal
    expect_rc 12
    expect_called 1
    expect_lastlog "skip-stop"
    finish_case
  done
}

case_stopped_runs_nothing() {  # while stopped, no binary from HEAD's config.env runs, not even for --push-only
  local evil
  new_case stopped_runs_nothing
  evil="$C/evil-python"
  printf '#!/bin/sh\ntouch "%s"\nexec "%s" "$@"\n' "$C/evil_ran" "$PY" > "$evil"
  chmod +x "$evil"
  awk -v p="$evil" '/^PYTHON_BIN=/ { print "PYTHON_BIN=\"" p "\""; next } { print }' \
    "$CL/$STUDY_REL/loop/config.env" > "$C/config.env" && cp "$C/config.env" "$CL/$STUDY_REL/loop/config.env"
  printf 'STOP\nWritten by: runner (it-x), 2026-10-02T00:00:00Z\nReason: a loop commit changed loop/config.env\n' > "$CL/$STUDY_REL/STOP"
  loop_commit "Study 04 [it-$TODAY-7775]: a session changed config.env, and the runner stopped the loop"
  git -C "$CL" push -q origin main
  run_iter normal
  expect_rc 12
  expect_called 0
  expect_lastlog "skip-stop"
  OUT="$C/pushonly.out"
  bash "$CL/$STUDY_REL/loop/run_iteration.sh" --push-only > "$OUT" 2>&1
  [ ! -e "$C/evil_ran" ] || fail "the interpreter named by HEAD's config.env ran while the loop was stopped"
  finish_case
}

case_dead_runner_git_config() {  # a session whose runner died changed .git/config
  local dead before
  new_case dead_runner_git_config
  dead="it-$TODAY-8887"
  before=$(git -C "$CL" rev-parse HEAD)
  mkdir -p "$ST"
  cp "$CL/.git/config" "$ST/git-config.pre.$dead"
  echo "$dead $before $(date +%s)" > "$ST/inflight"
  git -C "$CL" config remote.origin.pushurl /nonexistent/elsewhere.git
  run_iter normal
  expect_rc 12
  expect_called 0
  git -C "$CL" config --get remote.origin.pushurl > /dev/null && fail "the dead session's pushurl is still in .git/config"
  grep -q "pushurl" "$ST/git-config.$dead" 2>/dev/null || fail "the changed .git/config was not kept as evidence"
  expect_stop "changed the clone's .git/config"
  [ ! -f "$ST/inflight" ] || fail "inflight not cleared"
  finish_case
}

case_gitignore_negation() {  # a committed rule un-ignores data/: nothing is stashed, the run data stays
  local f="$STUDY_REL/data/base/04_base_x/diagnostics.h5"
  new_case gitignore_negation
  run_iter gitignore_negation
  expect_rc 12
  expect_stop "gitignore"
  expect_stop "not stashed"
  [ -f "$CL/$f" ] || fail "the fetched run data is gone from data/"
  grep -q "uncommitted edit" "$CL/$STUDY_REL/STATUS.md" || fail "the session's edit is gone from the working tree"
  git -C "$CL" stash list | grep -q . && fail "something was stashed: $(git -C "$CL" stash list | head -1)"
  [ -s "$ST/transcripts/$LAST_ID.jsonl" ] || fail "the transcript is gone from .loop/"
  finish_case
}

case_nested_instructions() {  # CLAUDE.md files in ignored folders stop the next session
  new_case nested_instructions
  run_iter nested_claude_md
  expect_rc 0
  run_iter normal
  expect_rc 12
  expect_called 1
  expect_stop "$STUDY_REL/data/CLAUDE.md exists and is not tracked"
  expect_stop "$STUDY_REL/data/scratch/CLAUDE.local.md exists and is not tracked"
  finish_case
}

case_run_forever_pinned() {  # a STOPped loop commit that replaced the runner never runs under run_forever
  local fp waited=0 rc
  new_case run_forever_pinned
  printf '#!/bin/bash\ntouch "%s"\nexit 0\n' "$C/evil_runner_ran" > "$CL/$STUDY_REL/loop/run_iteration.sh"
  printf 'STOP\nWritten by: runner (it-x), 2026-10-02T00:00:00Z\nReason: a loop commit replaced the runner\n' > "$CL/$STUDY_REL/STOP"
  loop_commit "Study 04 [it-$TODAY-7774]: a session replaced the runner, and the runner stopped the loop"
  git -C "$CL" push -q origin main
  OUT="$C/forever.out"
  env S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" bash "$CL/$STUDY_REL/loop/run_forever.sh" > "$OUT" 2>&1 &
  fp=$!
  while kill -0 "$fp" 2>/dev/null && [ "$waited" -lt 60 ]; do sleep 0.5; waited=$((waited + 1)); done
  if kill -0 "$fp" 2>/dev/null; then fail "run_forever.sh did not exit on STOP"; kill -TERM "$fp"; fi
  wait "$fp"
  rc=$?
  [ "$rc" = 0 ] || fail "run_forever.sh exit $rc, expected 0"
  [ ! -e "$C/evil_runner_ran" ] || fail "run_forever.sh ran the runner that the stopped loop commit wrote"
  expect_called 0
  grep -q "STOP is present" "$OUT" || fail "run_forever.sh did not report STOP"
  finish_case
}

case_wip_streak() {
  local m i=0
  new_case wip_streak
  for m in wip wip wip waiting wip wip; do
    i=$((i + 1))
    run_iter "$m" S04_DAILY_ITERATION_CAP=20
    [ "$RC" = 0 ] || fail "iteration $i ($m) exit $RC"
    if remote_has "$STUDY_REL/STOP"; then fail "STOP after iteration $i ($m), too early"; break; fi
  done
  run_iter wip S04_DAILY_ITERATION_CAP=20
  expect_rc 12
  expect_stop "6 iterations in a row ended WIP or as a stub"
  finish_case
}

case_dead_runner_recovery() {
  local dead before
  new_case dead_runner_recovery
  dead="it-$TODAY-8888"
  before=$(git -C "$CL" rev-parse HEAD)
  mkdir -p "$ST"
  echo "# sneaky" >> "$CL/$STUDY_REL/loop/config.env"
  GIT_AUTHOR_NAME=krmhd-loop GIT_AUTHOR_EMAIL=loop@example.invalid GIT_COMMITTER_NAME=krmhd-loop GIT_COMMITTER_EMAIL=loop@example.invalid \
    git -C "$CL" commit -q -am "Study 04 [$dead]: work of a session whose runner died"
  echo "half-done" > "$CL/$STUDY_REL/halfdone.txt"
  echo "$dead $before $(date +%s)" > "$ST/inflight"
  run_iter normal
  expect_rc 12
  expect_called 0
  expect_stop "guarded-path"
  day_log | grep -q "^## $dead: runner stub" || fail "no stub for the dead iteration"
  day_log | grep -q "runner died" || fail "stub does not say the runner died"
  git -C "$CL" stash list | grep -q "loop-stash $dead" || fail "dead iteration's changes not stashed"
  [ ! -f "$ST/inflight" ] || fail "inflight not cleared"
  finish_case
}

case_diverged() {
  new_case diverged
  replace_line "$CL/$STUDY_REL/STATUS.md" "Second commit, so that HEAD~1 exists." "The loop's version of this line."
  loop_commit "Study 04 [it-$TODAY-7777]: loop work that never got pushed"
  anjor_clone
  replace_line "$AN/$STUDY_REL/STATUS.md" "Second commit, so that HEAD~1 exists." "Anjor's version of this line."
  (cd "$AN" && git commit -q -am "Anjor: conflicting edit" && git push -q origin main) || fail "anjor push failed"
  run_iter normal
  expect_rc 12
  expect_called 0
  expect_stop "diverged"
  git --git-dir="$C/remote.git" for-each-ref --format='%(refname:short)' refs/heads/ | grep -q '^loop-diverged-' \
    || fail "no loop-diverged-* branch on the remote"
  git --git-dir="$C/remote.git" log --format=%s --branches='loop-diverged-*' | grep -q "loop work that never got pushed" \
    || fail "the rescue branch lacks the loop's commit"
  expect_synced
  finish_case
}

case_unpushed_work() {
  new_case unpushed_work
  echo "A line from an earlier iteration whose push failed." >> "$CL/$STUDY_REL/STATUS.md"
  loop_commit "Study 04 [it-$TODAY-7776]: earlier unpushed work"
  run_iter normal
  expect_rc 0
  expect_called 1
  remote_subjects | grep -q "earlier unpushed work" || fail "earlier unpushed work not pushed"
  expect_no_stop
  expect_synced
  finish_case
}

case_branch_left() {
  new_case branch_left
  run_iter branch_switch
  expect_rc 0
  git --git-dir="$C/remote.git" rev-parse -q --verify refs/heads/side-work > /dev/null || fail "side branch not pushed to the remote"
  [ "$(git -C "$CL" symbolic-ref -q --short HEAD)" = main ] || fail "clone not back on main"
  day_log | grep -q "left commits on branch side-work" || fail "stub entry does not name the side branch"
  expect_no_stop
  expect_synced
  finish_case
}

case_dirty_log_edit() {
  new_case dirty_log_edit
  run_iter dirty_log_edit
  expect_rc 0
  expect_no_stop
  remote_show "$STUDY_REL/log/2026-09-30.md" | grep -q "never committed" && fail "the runner committed an edit of an old log line"
  git -C "$CL" stash list | grep -q "loop-stash $LAST_ID" || fail "the edit was not stashed"
  expect_lastlog "log_left_in_stash"
  # The entry was closed, but its work is in a stash: a stub says so, and it counts as a stub.
  day_log | grep -q "^## $LAST_ID: runner stub (work left uncommitted)" || fail "no stub naming the stash"
  expect_lastlog "outcome=stub"
  expect_synced
  finish_case
}

case_wait_uncommitted() {
  new_case wait_uncommitted
  run_iter wait_uncommitted
  expect_rc 0
  remote_has "$STUDY_REL/WAIT" || fail "the runner did not commit the session's WAIT"
  expect_lastlog "outcome=waiting"
  run_iter normal
  expect_rc 10
  expect_lastlog "skip-wait"
  expect_called 1
  finish_case
}

case_app_check_postponed() {
  new_case app_check_postponed
  run_iter modal_rogue_offline
  expect_rc 0
  expect_no_stop
  expect_lastlog "apps_unchecked"
  [ -s "$ST/app_windows" ] || fail "the session window was not kept for a later check"
  rm -f "$C/stub/modal_fail"
  # A WAIT must not postpone the check further.
  anjor_clone
  (cd "$AN" && bash "$STUDY_REL/loop/wait.sh" 30 "runs in flight" > /dev/null && git add -A && git commit -q -m wait && git push -q origin main) \
    || fail "could not push WAIT"
  run_iter normal
  expect_rc 12
  expect_called 1
  expect_stop "ap-rogue"
  finish_case
}

case_data_link() {
  new_case data_link
  mkdir -p "$C/elsewhere/hermite128_nu3_imex" "$CL/studies/02-collisionality-scan"
  ln -s "$C/elsewhere" "$CL/studies/02-collisionality-scan/data"
  run_iter normal
  expect_rc 0
  expect_called 1
  [ -L "$CL/studies/02-collisionality-scan/data" ] || fail "the data link is gone"
  git -C "$CL" stash list | grep -q . && fail "the runner stashed something: $(git -C "$CL" stash list | head -1)"
  grep -qx '/studies/02-collisionality-scan/data' "$CL/.git/info/exclude" || fail "data link not added to info/exclude"
  expect_lastlog "excluded_data_link"
  expect_synced
  finish_case
}

case_predirty() {
  new_case predirty
  echo "left by a runner that died after its session" > "$CL/$STUDY_REL/leftover.txt"
  run_iter normal
  expect_rc 0
  git -C "$CL" stash list | grep -q "predirty" || fail "no predirty stash"
  grep -q "predirty" "$C/marks/claude_args.$LAST_ID" || fail "prompt does not name the predirty stash"
  expect_synced
  finish_case
}

# ---------------------------------------------------------------------------
# Cases: work that must not be lost
# ---------------------------------------------------------------------------

case_stash_pop_conflict() {
  new_case stash_pop_conflict
  run_iter stash_pop_conflict
  expect_rc 0
  expect_no_stop
  [ -z "$(git -C "$CL" ls-files -u)" ] || fail "unmerged entries left in the clone"
  stash_has "loop-stash $LAST_ID" "VALUABLE-STASH-POP-EDIT"
  [ "$(git -C "$CL" stash list | wc -l | tr -d ' ')" -ge 2 ] || fail "the session's own stash entry is gone"
  expect_lastlog "conflicts_staged="
  ls "$ST/rescue/"*.patch > /dev/null 2>&1 || fail "no copy of the conflicted diff in .loop/rescue"
  day_log | grep -q "^## $LAST_ID: runner stub" || fail "no stub entry"
  expect_synced
  # The next session hears of every stash, by position and sha.
  run_iter normal
  expect_rc 0
  grep "^Stash from a dead iteration:" "$C/marks/claude_args.$LAST_ID" | grep -q 'stash@{1} (' \
    || fail "the next prompt does not list every stash: $(grep '^Stash from' "$C/marks/claude_args.$LAST_ID")"
  finish_case
}

case_autostash_conflict() {
  new_case autostash_conflict
  run_iter autostash_conflict
  expect_rc 0
  expect_no_stop
  [ -z "$(git -C "$CL" ls-files -u)" ] || fail "unmerged entries left in the clone"
  stash_has "loop-stash $LAST_ID" "VALUABLE-AUTOSTASH-EDIT"
  git -C "$CL" stash list | grep -q autostash || fail "the autostash entry is gone"
  remote_subjects | grep -q "stub intent, never pushed" || fail "the session's unpushed commit was not pushed"
  remote_subjects | grep -q "Anjor: a second push" || fail "Anjor's second push is not on main"
  expect_synced
  finish_case
}

case_midop_rebase() {
  local br
  new_case midop_rebase
  run_iter midop_rebase
  br="loop-rescue/$LAST_ID"
  [ ! -d "$CL/.git/rebase-merge" ] && [ ! -d "$CL/.git/rebase-apply" ] || fail "the rebase is still in progress"
  [ "$(git -C "$CL" symbolic-ref -q --short HEAD)" = main ] || fail "clone not back on main"
  git --git-dir="$C/remote.git" log --format=%s "refs/heads/$br" 2>/dev/null | grep -q "resolved and more work" \
    || fail "the commit made during the stopped rebase is not on a pushed $br"
  stash_has "loop-stash $LAST_ID (edits during a stopped rebase)" "VALUABLE-MIDOP-EDIT"
  stash_has "loop-stash $LAST_ID (edits during a stopped rebase)" "midop_untracked.txt"
  grep -q "$br" "$CL/$STUDY_REL/log/$TODAY_DASH.md" || fail "the stub does not name $br"
  expect_lastlog "rescue_branch=$br"
  finish_case
}

case_midop_other() {  # a stopped merge and a stopped cherry-pick: edits saved in a stash, then aborted
  local mode op
  for mode in midop_merge midop_cherry_pick; do
    new_case "$mode"
    run_iter "$mode"
    op=merge
    [ "$mode" = midop_cherry_pick ] && op=cherry-pick
    [ ! -f "$CL/.git/MERGE_HEAD" ] && [ ! -f "$CL/.git/CHERRY_PICK_HEAD" ] || fail "the $op is still in progress"
    [ "$(git -C "$CL" symbolic-ref -q --short HEAD)" = main ] || fail "clone not back on main"
    [ -z "$(git -C "$CL" status --porcelain)" ] || fail "clone tree dirty after the rescue"
    stash_has "loop-stash $LAST_ID (edits during a stopped $op)" "VALUABLE-MIDOP-EDIT"
    stash_has "loop-stash $LAST_ID (edits during a stopped $op)" "midop_untracked.txt"
    expect_lastlog "aborted=$op"
    git -C "$CL" log --format=%s main | grep -q "stub local change" || fail "the session's commit on main was lost"
    finish_case
  done
}

case_stash_fails() {
  new_case stash_fails
  run_iter unstashable
  expect_rc 12
  expect_stop "stash failed"
  grep -q "VALUABLE-UNSTASHED-EDIT" "$CL/$STUDY_REL/QUESTIONS.md" || fail "the session's edit is gone from the working tree"
  [ -f "$CL/$STUDY_REL/unreadable.txt" ] || fail "the unreadable file is gone"
  remote_show "$STUDY_REL/QUESTIONS.md" | grep -q "VALUABLE-UNSTASHED-EDIT" && fail "the session's uncommitted edit was committed"
  # No session starts on that tree, and the existing STOP is not written again.
  run_iter normal
  expect_rc 12
  expect_called 1
  expect_lastlog "skip-stop"
  [ "$(remote_show "$STUDY_REL/STOP" | grep -c '^Also written by')" = 0 ] || fail "STOP was extended while the loop was already stopped"
  chmod 644 "$CL/$STUDY_REL/unreadable.txt"
  finish_case
}

case_stop_commit_fails() {
  new_case stop_commit_fails
  # A STOP left uncommitted (its commit failed earlier), and the index locked right now.
  printf 'STOP\nWritten by: runner (it-x), 2026-10-02T00:00:00Z\nReason: an earlier STOP whose commit failed\n' > "$CL/$STUDY_REL/STOP"
  touch "$CL/.git/index.lock"
  run_iter normal
  expect_rc 12
  expect_called 0
  expect_stop "an earlier STOP whose commit failed"
  [ -f "$CL/$STUDY_REL/STOP" ] || fail "the local STOP is gone"
  expect_lastlog "stop_published_directly="
  # Once the index is free, the runner commits it too, and local main and origin agree.
  rm -f "$CL/.git/index.lock"
  run_iter normal
  expect_rc 12
  expect_called 0
  expect_synced
  [ "$(remote_show "$STUDY_REL/STOP" | grep -c '^Reason:')" = 1 ] || fail "STOP on the remote was duplicated"
  finish_case
}

case_agent_stop_uncommitted() {
  local fp waited=0 rc
  new_case agent_stop_uncommitted
  ITER_N=$((ITER_N + 1))
  LAST_ID="it-$TODAY-$(printf '%04d' "$ITER_N")"
  OUT="$C/forever.out"
  env S04_STUB_CLAUDE_MODE=agent_stop_uncommitted S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" S04_STUB_PYTHON="$PY" \
    S04_TEST_ITER_ID="$LAST_ID" bash "$CL/$STUDY_REL/loop/run_forever.sh" > "$OUT" 2>&1 &
  fp=$!
  while kill -0 "$fp" 2>/dev/null && [ "$waited" -lt 60 ]; do sleep 0.5; waited=$((waited + 1)); done
  if kill -0 "$fp" 2>/dev/null; then fail "run_forever.sh did not stop on the uncommitted STOP"; kill -TERM "$fp"; fi
  wait "$fp"
  rc=$?
  [ "$rc" = 0 ] || fail "run_forever.sh exit $rc, expected 0"
  expect_called 1
  expect_stop "stub hard stop left uncommitted"
  remote_show "$STUDY_REL/QUESTIONS.md" | grep -q '^- \*\*STOP .* (agent)\*\*' || fail "the agent's Blocking line was not committed"
  day_log | grep -q "outcome: hard stop" || fail "the hard-stop End line is not on the remote"
  remote_subjects | grep -q "runner committed a STOP left uncommitted" || fail "no runner commit of the STOP"
  grep -q "ran exit=0" "$ST/runner.log" || fail "no ran line"
  expect_synced
  finish_case
}

case_hard_stop_no_file() {
  new_case hard_stop_no_file
  run_iter hard_stop_no_file
  expect_rc 12
  expect_stop "ends with 'outcome: hard stop', but the session wrote no STOP"
  expect_synced
  finish_case
}

case_gitignore_drop() {
  local lines_before
  new_case gitignore_drop
  run_iter normal
  lines_before=$(wc -l < "$ST/runner.log" | tr -d ' ')
  run_iter gitignore_drop
  expect_rc 12
  expect_stop "gitignore: .loop/"
  [ -f "$ST/transcripts/it-$TODAY-$(printf '%04d' $((ITER_N - 1))).jsonl" ] || fail "an earlier transcript is gone from .loop/"
  [ "$(wc -l < "$ST/runner.log" | tr -d ' ')" -gt "$lines_before" ] || fail "runner.log lost its history"
  git --git-dir="$C/remote.git" ls-tree -r --name-only main | grep -q '\.loop/' && fail ".loop/ files were committed"
  git -C "$CL" stash list | grep -q . && fail "something was stashed: $(git -C "$CL" stash list | head -1)"
  expect_synced
  finish_case
}

case_hooks() {
  new_case hooks
  run_iter hook_plant
  expect_rc 0
  remote_subjects | grep -q "runner stub (log entry not closed)" || fail "the runner's commit was blocked by the planted hook"
  expect_no_stop
  run_iter normal
  expect_rc 12
  expect_called 1
  expect_stop "git hook .git/hooks/pre-commit exists"
  finish_case
}

case_git_config_change() {
  new_case git_config_change
  run_iter git_config_change
  expect_rc 12
  expect_stop "changed the clone's .git/config"
  git -C "$CL" config --local --get core.sshCommand > /dev/null && fail "the session's .git/config change is still in place"
  grep -q sshCommand "$ST/git-config.$LAST_ID" 2>/dev/null || fail "the changed .git/config was not kept as evidence"
  [ "$(git -C "$CL" config --local --get s04.loopclone)" = true ] || fail "the restored config lost the clone marker"
  finish_case
}

case_local_files() {  # files a session would load: each one stops the loop before a session
  local what
  for what in settings untracked claude_local; do
    new_case "local_files_$what"
    case "$what" in
      settings) mkdir -p "$CL/.claude"; echo '{"env": {"GIT_AUTHOR_NAME": "anjor"}}' > "$CL/.claude/settings.local.json" ;;
      untracked) echo "# a new agent" > "$CL/.claude/agents/new-agent.md" ;;
      claude_local) echo "Notes for future iterations." > "$CL/CLAUDE.local.md" ;;
    esac
    run_iter normal
    expect_rc 12
    expect_called 0
    case "$what" in
      settings) expect_stop ".claude/settings.local.json exists" ;;
      untracked) expect_stop "untracked file under .claude/: .claude/agents/new-agent.md" ;;
      claude_local) expect_stop "CLAUDE.local.md exists" ;;
    esac
    finish_case
  done
}

case_netdown() {
  new_case netdown
  run_iter netdown_touch_loop
  expect_rc 12
  [ -f "$ST/push_pending" ] || fail "no push_pending marker after the push failed"
  [ -f "$CL/$STUDY_REL/STOP" ] || fail "no local STOP"
  git --git-dir="$C/remote.git.down" cat-file -e "main:$STUDY_REL/STOP" 2>/dev/null && fail "STOP reached a remote that was down"
  mv "$C/remote.git.down" "$C/remote.git"
  run_iter normal
  expect_rc 12
  expect_called 1
  expect_stop "guarded-path: $STUDY_REL/loop/config.env"
  [ ! -f "$ST/push_pending" ] || fail "push_pending not cleared after the push"
  expect_lastlog "push_retry="
  finish_case
}

case_run_forever_netdown() {
  local fp waited=0 rc
  new_case run_forever_netdown
  ITER_N=$((ITER_N + 1))
  LAST_ID="it-$TODAY-$(printf '%04d' "$ITER_N")"
  OUT="$C/forever.out"
  env S04_STUB_CLAUDE_MODE=netdown_touch_loop S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" S04_STUB_PYTHON="$PY" \
    S04_TEST_ITER_ID="$LAST_ID" bash "$CL/$STUDY_REL/loop/run_forever.sh" > "$OUT" 2>&1 &
  fp=$!
  while [ ! -d "$C/remote.git.down" ] && [ "$waited" -lt 60 ]; do sleep 0.25; waited=$((waited + 1)); done
  waited=0
  while ! grep -q "STOP is only local" "$OUT" 2>/dev/null && [ "$waited" -lt 80 ]; do sleep 0.25; waited=$((waited + 1)); done
  sleep 2
  kill -0 "$fp" 2>/dev/null || fail "run_forever.sh exited while its STOP was only local"
  mv "$C/remote.git.down" "$C/remote.git"
  waited=0
  while kill -0 "$fp" 2>/dev/null && [ "$waited" -lt 80 ]; do sleep 0.5; waited=$((waited + 1)); done
  if kill -0 "$fp" 2>/dev/null; then fail "run_forever.sh did not exit once STOP was pushed"; kill -TERM "$fp"; fi
  wait "$fp"
  rc=$?
  [ "$rc" = 0 ] || fail "run_forever.sh exit $rc, expected 0"
  expect_called 1
  expect_stop "guarded-path: $STUDY_REL/loop/config.env"
  grep -q "STOP is present" "$OUT" || fail "run_forever.sh did not report STOP"
  finish_case
}

case_run_forever_stop() {
  local fp waited=0 rc
  new_case run_forever_stop
  ITER_N=$((ITER_N + 1))
  LAST_ID="it-$TODAY-$(printf '%04d' "$ITER_N")"
  OUT="$C/forever.out"
  env S04_STUB_CLAUDE_MODE=agent_stop S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" S04_STUB_PYTHON="$PY" \
    S04_TEST_ITER_ID="$LAST_ID" bash "$CL/$STUDY_REL/loop/run_forever.sh" > "$OUT" 2>&1 &
  fp=$!
  while kill -0 "$fp" 2>/dev/null && [ "$waited" -lt 60 ]; do sleep 0.5; waited=$((waited + 1)); done
  if kill -0 "$fp" 2>/dev/null; then fail "run_forever.sh did not exit on STOP"; kill -TERM "$fp"; fi
  wait "$fp"
  rc=$?
  [ "$rc" = 0 ] || fail "run_forever.sh exit $rc, expected 0"
  expect_called 1
  expect_stop "Written by: agent"
  grep -q "STOP is present" "$OUT" || fail "run_forever.sh did not report STOP"
  finish_case
}

case_run_forever_signal() {
  local fp waited=0 rc m
  m=$(mk 151)
  new_case run_forever_signal
  ITER_N=$((ITER_N + 1))
  LAST_ID="it-$TODAY-$(printf '%04d' "$ITER_N")"
  OUT="$C/forever.out"
  env S04_STUB_CLAUDE_MODE=hang S04_STUB_MARKER="$m" S04_STUB_MARKS="$C/marks" S04_STUB_DIR="$C/stub" \
    S04_STUB_PYTHON="$PY" S04_TEST_ITER_ID="$LAST_ID" S04_SESSION_TIME_LIMIT_SEC=120 \
    bash "$CL/$STUDY_REL/loop/run_forever.sh" > "$OUT" 2>&1 &
  fp=$!
  while [ ! -f "$C/marks/claude_called" ] && [ "$waited" -lt 40 ]; do sleep 0.25; waited=$((waited + 1)); done
  sleep 1
  kill -TERM "$fp"
  waited=0
  while kill -0 "$fp" 2>/dev/null && [ "$waited" -lt 60 ]; do sleep 0.5; waited=$((waited + 1)); done
  if kill -0 "$fp" 2>/dev/null; then fail "run_forever.sh still running 30s after TERM"; kill -KILL "$fp"; fi
  wait "$fp"
  rc=$?
  [ "$rc" = 130 ] || fail "run_forever.sh exit $rc, expected 130"
  no_marker "$m"
  expect_lastlog "interrupted=1"
  day_log | grep -q "^- Runner: .*note interrupted" || fail "no stub saying the session was interrupted"
  expect_synced
  finish_case
}

case_preflight_stubs() {
  local rc
  new_case preflight_stubs
  mkdir -p "$CL/studies/02-collisionality-scan/data/hermite128_nu3_imex"
  OUT="$C/preflight.out"
  env S04_STUB_DIR="$C/stub" bash "$CL/$STUDY_REL/loop/preflight.sh" --skip-uv --allow-stubs > "$OUT" 2>&1
  rc=$?
  [ "$rc" = 0 ] || fail "preflight exit $rc: $(grep '^FAIL' "$OUT" | head -3 | tr '\n' ' ')"
  OUT="$C/preflight2.out"
  env S04_STUB_DIR="$C/stub" S04_STUB_GH_RC=1 bash "$CL/$STUDY_REL/loop/preflight.sh" --skip-uv --allow-stubs > "$OUT" 2>&1
  rc=$?
  [ "$rc" = 1 ] || fail "preflight with gh logged out: exit $rc, expected 1"
  grep -q '^FAIL  gh auth status' "$OUT" || fail "preflight did not report the gh failure"
  OUT="$C/preflight3.out"
  env S04_STUB_DIR="$C/stub" bash "$CL/$STUDY_REL/loop/preflight.sh" --skip-uv > "$OUT" 2>&1
  grep -q '^FAIL  config.env points' "$OUT" || fail "preflight accepted stub binaries without --allow-stubs"
  printf '#!/bin/sh\nexit 0\n' > "$CL/.git/hooks/post-checkout"
  OUT="$C/preflight4.out"
  env S04_STUB_DIR="$C/stub" bash "$CL/$STUDY_REL/loop/preflight.sh" --skip-uv --allow-stubs > "$OUT" 2>&1
  grep -q '^FAIL  the clone holds files' "$OUT" || fail "preflight did not flag a git hook"
  finish_case
}

# ---------------------------------------------------------------------------
# Cases: guards.py on its own, in a scratch clone
# ---------------------------------------------------------------------------

case_guards_log_parsing() {
  local f out
  new_case guards_log_parsing
  f="$CL/$STUDY_REL/log/2026-10-02.md"
  cat > "$f" <<'EOF'
# Study 04 loop log — 2026-10-02

## it-20261002-0001: an ordinary entry
- Block: x
- End: 2026-10-02T01:00:00Z, outcome: done

## it-20261002-0002: an entry that quotes a dead one
- Block: recover
- Quoted from the dead entry:
```
## it-20261002-0003: the dead iteration
- End: 2026-10-02T02:00:00Z, outcome: stub
```
- End: 2026-10-02T02:30:00Z, outcome: done

## it-20261002-0004: a bold End line
- **End:** 2026-10-02T03:00:00Z, outcome: done

## it-20261002-0005: a short spelling
- End: 2026-10-02T04:00:00Z, outcome: wait

## it-20261002-0006: an entry that leaves a fence open
```
- End: 2026-10-02T05:00:00Z, outcome: done
EOF
  [ "$(gp outcome --iter it-20261002-0001)" = done ] || fail "0001 not done"
  out=$(gp entry-closed --iter it-20261002-0002); [ "$out" = closed ] || fail "0002 (quoting a dead entry) reads as $out"
  [ "$(gp outcome --iter it-20261002-0002)" = done ] || fail "0002 outcome is $(gp outcome --iter it-20261002-0002)"
  [ "$(gp outcome --iter it-20261002-0003)" = missing ] || fail "a heading inside a fence counted as an entry"
  [ "$(gp outcome --iter it-20261002-0004)" = done ] || fail "'- **End:**' not read"
  [ "$(gp outcome --iter it-20261002-0005)" = waiting ] || fail "'outcome: wait' not read as waiting"
  gp entry-closed --iter it-20261002-0006 > /dev/null && fail "an End line inside an open fence closed the entry"
  gp append-stub --iter it-20261002-0006 --note test > /dev/null
  [ "$(gp outcome --iter it-20261002-0006)" = stub ] || fail "the stub after an open fence does not count"
  finish_case
}

case_guards_modal_rule() {
  local base out
  new_case guards_modal_rule
  mkdir -p "$CL/infrastructure" "$CL/$STUDY_REL/analysis"
  printf 'import modal\n\napp = modal.App("old")\n\n@app.function(gpu="A100")\ndef f():\n    pass\n' > "$CL/infrastructure/modal_app.py"
  anjor_commit "Anjor: an existing GPU app"
  base=$(git -C "$CL" rev-parse HEAD)
  printf 'import subprocess\nsubprocess.run("uv run modal run x.py", shell=True)\n' > "$CL/$STUDY_REL/analysis/a.py"
  loop_commit "loop a"
  printf 'from infrastructure import modal_app\n' > "$CL/$STUDY_REL/analysis/b.py"
  loop_commit "loop b"
  printf '#!/usr/bin/env python3\nimport modal\n' > "$CL/$STUDY_REL/analysis/runme"
  loop_commit "loop c"
  replace_line "$CL/infrastructure/modal_app.py" '@app.function(gpu="A100")' '@app.function(gpu="A100:8")'
  loop_commit "loop d"
  printf '"""Run code for the GPU function, which the launcher calls."""\nimport numpy\n' > "$CL/$STUDY_REL/analysis/ok.py"
  loop_commit "loop e"
  out=$(gp commit-guards --before "$base" --after HEAD --loop-committer krmhd-loop)
  printf '%s\n' "$out" | grep -q "modal-code: $STUDY_REL/analysis/a.py" || fail "a shell=True 'modal run' string not flagged"
  printf '%s\n' "$out" | grep -q "modal-code: $STUDY_REL/analysis/b.py" || fail "'from infrastructure import modal_app' not flagged"
  printf '%s\n' "$out" | grep -q "modal-code: $STUDY_REL/analysis/runme" || fail "an extensionless Python script not flagged"
  printf '%s\n' "$out" | grep -q "modal-code: infrastructure/modal_app.py is changed and imports Modal" || fail "an edit of an existing GPU app not flagged"
  printf '%s\n' "$out" | grep -q "analysis/ok.py" && fail "a docstring naming the GPU function was flagged"
  finish_case
}

case_guards_instruction_files() {
  local base out
  new_case guards_instruction_files
  base=$(git -C "$CL" rev-parse HEAD)
  mkdir -p "$CL/$STUDY_REL/.claude/agents" "$CL/$STUDY_REL/analysis"
  echo "Notes for later iterations." > "$CL/$STUDY_REL/CLAUDE.md"
  echo "More notes." > "$CL/$STUDY_REL/analysis/CLAUDE.local.md"
  echo "# a nested agent" > "$CL/$STUDY_REL/.claude/agents/y.md"
  loop_commit "loop leaves instructions for later sessions"
  out=$(gp commit-guards --before "$base" --after HEAD --loop-committer krmhd-loop)
  printf '%s\n' "$out" | grep -q "guarded-path: $STUDY_REL/CLAUDE.md" || fail "a study-level CLAUDE.md was not flagged"
  printf '%s\n' "$out" | grep -q "guarded-path: $STUDY_REL/analysis/CLAUDE.local.md" || fail "a nested CLAUDE.local.md was not flagged"
  printf '%s\n' "$out" | grep -q "guarded-path: $STUDY_REL/.claude/agents/y.md" || fail "a nested .claude/ file was not flagged"
  finish_case
}

case_guards_apps_format() {
  local now j rc out
  new_case guards_apps_format
  anjor_clone
  "$PY" - "$CL/$STUDY_REL" <<'EOF' || fail "could not write the ledger"
import sys
sys.path.insert(0, sys.argv[1] + "/loop")
import loopcommon as lc
p = sys.argv[1] + "/compute_ledger.json"
L = lc.load_ledger(p)
L["launches"].append({"launch_id": "L001", "kind": "gpu", "run_set": "A", "app_name": "s04-loop-l001-a",
    "app_id": "ap-launcher-1", "state": "closed", "frozen": {}, "timeout_hours": 1.0,
    "runs": [{"run_id": "04_a_x", "reserved_hours": 1.0, "status": "finished", "charged_hours": 1.0, "attempts": []}]})
lc.save_ledger(p, L)
EOF
  now=$(date +%s)
  j="$C/apps.json"
  printf '[{"App ID": "ap-handrun-9", "Description": "s04-loop-l001-a", "State": "stopped", "Created at": "%s"}, {"App ID": "ap-launcher-1", "Description": "s04-loop-l001-a", "State": "stopped", "Created at": "%s"}]\n' \
    "$(date -u -r "$now" '+%Y-%m-%d %H:%M:%S+00:00')" "$(date -u -r "$now" '+%Y-%m-%d %H:%M:%S+00:00')" > "$j"
  out=$(gp apps-check --apps-json "$j" --prefix s04-loop --window $((now - 60)) $((now + 60)) --any-name)
  rc=$?
  [ "$rc" = 1 ] || fail "apps-check exit $rc, expected 1"
  printf '%s\n' "$out" | grep -q '^ap-handrun-9 ' || fail "an app reusing a launch's name with another ID was not flagged"
  printf '%s\n' "$out" | grep -q '^ap-launcher-1 ' && fail "the launch's own app was flagged"
  printf '[{"App ID": "ap-z", "Description": "x", "State": "stopped"}]\n' > "$j"
  gp apps-check --apps-json "$j" --prefix s04-loop --window $((now - 60)) $((now + 60)) > /dev/null
  [ "$?" = 3 ] || fail "a row without 'Created at' was not a format error under --window"
  gp apps-check --apps-json "$j" --prefix s04-loop > /dev/null || fail "a row without 'Created at' failed without --window"
  # The launch is closed (every run charged), but its app still runs with a task.
  printf '[{"App ID": "ap-launcher-1", "Description": "s04-loop-l001-a", "State": "ephemeral (detached)", "Tasks": "1", "Created at": "2026-01-01 00:00:00+00:00"}]\n' > "$j"
  out=$(gp apps-check --apps-json "$j" --prefix s04-loop)
  [ "$?" = 1 ] && printf '%s\n' "$out" | grep -q "launch L001 is closed in the ledger" \
    || fail "a running app of a closed launch was not flagged: $out"
  printf '[{"App ID": "ap-launcher-1", "Description": "s04-loop-l001-a", "State": "ephemeral (detached)", "Tasks": "0", "Created at": "2026-01-01 00:00:00+00:00"}]\n' > "$j"
  gp apps-check --apps-json "$j" --prefix s04-loop > /dev/null || fail "an idle app (no tasks) of a closed launch was flagged"
  finish_case
}

case_guards_log_merge_order() {
  local base out
  new_case guards_log_merge_order
  base=$(git -C "$CL" rev-parse HEAD)
  git -C "$CL" checkout -q -b loopside
  printf '\n## it-20261002-0100: loop entry\n- Block: x\n- End: 2026-10-02T01:00:00Z, outcome: done\n' >> "$CL/$STUDY_REL/log/2026-09-30.md"
  loop_commit "Study 04 [it-20261002-0100]: entry"
  git -C "$CL" checkout -q -b anjorside "$base"
  replace_line "$CL/$STUDY_REL/log/2026-09-30.md" "- Block: seed" "- Block: seed (Anjor: see Q7)"
  anjor_commit "Anjor: annotate log"
  GIT_AUTHOR_NAME=krmhd-loop GIT_AUTHOR_EMAIL=loop@example.invalid GIT_COMMITTER_NAME=krmhd-loop GIT_COMMITTER_EMAIL=loop@example.invalid \
    git -C "$CL" merge -q --no-edit loopside > /dev/null 2>&1 || fail "the merge did not go through"
  out=$(gp commit-guards --before "$base" --after HEAD --loop-committer krmhd-loop)
  printf '%s\n' "$out" | grep -q "log-edit" && fail "a false log-edit with Anjor's side first: $out"
  finish_case
}

case_guards_frozen_duplicate() {
  local base out plan
  new_case guards_frozen_duplicate
  base=$(git -C "$CL" rev-parse HEAD)
  plan="$CL/$STUDY_REL/PLAN.md"
  awk '/^## 2\. Question/ && !done { print "## 2. Question (copy)"; print ""; print "An unchanged-looking copy."; print ""; done = 1 } { print }' "$plan" > "$plan.tmp" && mv "$plan.tmp" "$plan"
  out=$(gp frozen-check)
  printf '%s\n' "$out" | grep -q "frozen heading appears 2 times: $STUDY_REL/PLAN.md#2. Question" || fail "a duplicated frozen heading passed frozen-check: $out"
  loop_commit "loop duplicates a heading"
  out=$(gp commit-guards --before "$base" --after HEAD --loop-committer krmhd-loop)
  printf '%s\n' "$out" | grep -q "frozen: frozen heading appears 2 times" || fail "commit-guards missed the duplicate heading: $out"
  finish_case
}

case_guards_questions_clean() {
  local q
  new_case guards_questions_clean
  q="$CL/$STUDY_REL/QUESTIONS.md"
  printf 'An added line.\n' >> "$q"
  gp questions-clean > /dev/null || fail "an added line was not clean"
  git -C "$CL" checkout -q -- "$q"
  awk 'NR == 1 { print "# Changed title"; next } { print }' "$q" > "$q.tmp" && mv "$q.tmp" "$q"
  gp questions-clean > /dev/null && fail "a changed line was clean"
  finish_case
}

# ---------------------------------------------------------------------------
# Run. One function, so that bash has read the whole file before any case runs: a file
# edited while the self-test runs cannot change what it does.
# ---------------------------------------------------------------------------

run_all() {
  echo "selftest: building the scratch template in $WORK (marker tag $RUN_TAG)"
  build_template || { echo "selftest: could not build the template" >&2; exit 2; }

  want normal && case_normal
  want not_loop_clone && case_not_loop_clone
  want open_entry && case_open_entry
  want dirty_tree && case_dirty_tree
  want fast_failure && case_fast_failure
  want rate_limited && case_rate_limited
  want hang && case_hang
  want leftover_process && case_leftover_process
  want marker_isolation && case_marker_isolation
  want stop_present && case_stop_present
  want wait_time && case_wait_time
  want stale_lock && case_stale_lock
  want stale_lock_race && case_stale_lock_race
  want live_lock && case_live_lock
  want daily_cap && case_daily_cap
  want daily_cap_from_log && case_daily_cap_from_log
  want total_cap && case_total_cap
  want auth_failure && case_auth_failure
  want unknown_app && case_unknown_app
  want known_app && case_known_app
  want app_during_session && case_app_during_session
  want app_prefix_only && case_app_prefix_only
  want guard_touch_loop && guard_case touch_loop "guarded-path: $STUDY_REL/loop/config.env"
  want guard_touch_claude && guard_case touch_claude "guarded-path: .claude/agents/x.md"
  want guard_touch_loopmd && guard_case touch_loopmd "guarded-path: $STUDY_REL/LOOP.md"
  want guard_frozen && guard_case frozen "frozen item changed: $STUDY_REL/PLAN.md#2. Question"
  want guard_frozen_transient && guard_case frozen_transient "frozen: frozen item changed: $STUDY_REL/PLAN.md#2. Question"
  want guard_log_edit && guard_case log_edit "log-edit: $STUDY_REL/log/2026-09-30.md"
  want guard_ledger_hand && guard_case ledger_hand "integrity hash does not match"
  want guard_ledger_overcap && guard_case ledger_overcap "exceeds the cap"
  want guard_ledger_forged && case_guard_ledger_forged
  want guard_ledger_deleted && case_guard_ledger_deleted
  want guard_modal_code && guard_case modal_code "modal-code: $STUDY_REL/analysis/side_door.py"
  want guard_other_study && guard_case other_study "other-study: studies/02-x/file.txt"
  want guard_other_paper && guard_case other_paper "other-study: paper/echo-imbalance/notes.md"
  want guard_delete_stop && guard_case delete_stop "stop-deleted"
  want guard_force_push && guard_case force_push "history:"
  want guard_unicode_paths && guard_case unicode_paths "guarded-path: $STUDY_REL/loop/Γ.py" \
    "guarded-path: .claude/agents/Γ-critic.md" "other-study: studies/02-x/Γ.txt" "modal-code: $STUDY_REL/analysis/Γ_flux.py"
  want guard_author_spoof && guard_case author_spoof "identity: committed by krmhd-loop with author 'anjor'" \
    "guarded-path: $STUDY_REL/loop/config.env"
  want guard_cherry_pick && guard_case cherry_pick "identity: committed by krmhd-loop with author 'anjor'" \
    "guarded-path: $STUDY_REL/loop/anjor_side.txt"
  want guard_merge_ours && guard_case merge_ours "merge-discard" "$STUDY_REL/STOP"
  want guard_neuter_guards && guard_case neuter_guards "guarded-path: $STUDY_REL/loop/guards.py"
  want guard_t1_gate_no_critic && guard_case gate_pass_no_critic "T1" "gate_reports/G1_"
  want guard_t1_gate_mixed && guard_case gate_mixed "T1" "returned REFUTED" "is not about Gate 1"
  want guard_t1_gate_after && guard_case gate_after "T1" "before the commit"
  want guard_t1_gate_odd_name && guard_case gate_odd_name "T1" "is not named G<n>_"
  want guard_t2_merge_unchecked && guard_case merge_unchecked "T2 a GANDALF merge ran without --match-head-commit"
  want guard_t2_merge_exception && guard_case merge_exception "T2 merge of abcdef123456" "Existing tests changed"
  want guard_t2_merge_two && guard_case merge_two "T2 a GANDALF merge ran without --match-head-commit <sha>: gh pr merge 8"
  want guard_t2_forged_review && guard_case forged_review "T2 merge of abcdef123456 without an earlier gandalf-reviewer report"
  want guard_t3_module && guard_case uv_modal_module "T3 the session ran module analysis.sneaky_run with uv run"
  want guard_t3_stdin && guard_case uv_modal_stdin "on Python's stdin"
  want guard_t3_modal_cli && guard_case modal_cli "T3 the session ran the Modal CLI outside the launcher: modal run"
  want guard_forged_committer && guard_case forged_committer "provenance: committed by 'anjor'" "Tidy loop files"
  want guard_forged_fastimport && guard_case forged_fastimport "provenance: committed by 'anjor'" "Tidy loop files"
  want guard_ledger_semantics && case_guard_ledger_semantics
  want guard_report_edit && case_guard_report_edit
  want allowed_anjor_mid && allowed_case anjor_mid
  want allowed_ledger_launcher && allowed_case ledger_launcher
  want allowed_own_entry_edit && allowed_case own_entry_edit
  want allowed_anjor_mid_nopush && allowed_case anjor_mid_nopush
  want allowed_t1_gate_with_critic && allowed_case gate_pass_with_critic
  want allowed_t2_merge_reviewed && allowed_case merge_reviewed
  want allowed_t2_reviewer_wording && allowed_case reviewer_wording
  want allowed_t2_commit_msg && allowed_case commit_msg_merge
  want transcript_t2_earlier && case_transcript_t2_earlier
  want transcript_t3 && case_transcript_t3
  want transcript_t3_denied && case_transcript_t3_denied
  want transcript_gone && case_transcript_gone
  want stop_removed && case_stop_removed
  want stopped_runs_nothing && case_stopped_runs_nothing
  want dead_runner_git_config && case_dead_runner_git_config
  want gitignore_negation && case_gitignore_negation
  want nested_instructions && case_nested_instructions
  want run_forever_pinned && case_run_forever_pinned
  want diverged && case_diverged
  want unpushed_work && case_unpushed_work
  want branch_left && case_branch_left
  want dirty_log_edit && case_dirty_log_edit
  want wait_uncommitted && case_wait_uncommitted
  want app_check_postponed && case_app_check_postponed
  want predirty && case_predirty
  want data_link && case_data_link
  want stash_pop_conflict && case_stash_pop_conflict
  want autostash_conflict && case_autostash_conflict
  want midop_rebase && case_midop_rebase
  want midop_other && case_midop_other
  want stash_fails && case_stash_fails
  want stop_commit_fails && case_stop_commit_fails
  want agent_stop_uncommitted && case_agent_stop_uncommitted
  want hard_stop_no_file && case_hard_stop_no_file
  want gitignore_drop && case_gitignore_drop
  want hooks && case_hooks
  want git_config_change && case_git_config_change
  want local_files && case_local_files
  want netdown && case_netdown
  want wip_streak && case_wip_streak
  want dead_runner_recovery && case_dead_runner_recovery
  want run_forever_stop && case_run_forever_stop
  want run_forever_netdown && case_run_forever_netdown
  want run_forever_signal && case_run_forever_signal
  want preflight_stubs && case_preflight_stubs
  want guards_log_parsing && case_guards_log_parsing
  want guards_modal_rule && case_guards_modal_rule
  want guards_instruction_files && case_guards_instruction_files
  want guards_apps_format && case_guards_apps_format
  want guards_log_merge_order && case_guards_log_merge_order
  want guards_frozen_duplicate && case_guards_frozen_duplicate
  want guards_questions_clean && case_guards_questions_clean

  echo
  echo "selftest: $PASSED passed, $FAILED failed in $(( $(date +%s) - T_START ))s"
  if [ "$FAILED" -gt 0 ]; then
    printf '%s\n' "$FAIL_LINES" | sed '/^$/d'
    exit 1
  fi
  exit 0
}

run_all "$@"; exit $?
