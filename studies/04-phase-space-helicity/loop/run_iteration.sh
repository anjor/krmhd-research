#!/bin/bash
# run_iteration.sh: run one iteration of the Study 04 autonomous loop.
#
# Run it in the loop's clone (~/repos/anjor/krmhd-research-loop), by hand, from
# run_forever.sh, or from any scheduler:
#     studies/04-phase-space-helicity/loop/run_iteration.sh
#     studies/04-phase-space-helicity/loop/run_iteration.sh --push-only
#     <copy of this script> --repo ~/repos/anjor/krmhd-research-loop     (run_forever.sh)
# It refuses to run (exit 2, no STOP) unless 'git config s04.loopclone' is 'true' in the
# clone's own config, so it never runs in Anjor's checkout. --push-only only commits a STOP
# left in the working tree and pushes main, with git alone (run_forever.sh uses it to publish
# a STOP whose push failed); it exits 0 when local main equals origin/main, 1 when it does not.
#
# STOP means a STOP file in the working tree or a STOP committed at HEAD: a committed STOP
# that a session deleted without a commit is put back and still stops the loop. While the
# loop is stopped, nothing from HEAD's loop files runs and no binary named by its config.env
# (python, modal, gh, claude): a loop commit that changed them may be why it stopped. Only git
# runs until a sync shows STOP gone. A runner that died leaves .loop/inflight; the next one
# then takes the loop files from the commit that iteration started from (the trust anchor).
#
# Settings come from loop/config.env as committed at HEAD, read as KEY=VALUE text (never
# sourced). Any KEY there can be overridden by an environment variable S04_KEY (the self-test
# uses this; nothing else should). One line per attempt goes to .loop/runner.log, a narrative
# to .loop/runner-detail.log, and the session transcript to .loop/transcripts/<iteration
# id>.jsonl (with .stderr next to it).
#
# Exit codes: 0 a session ran; 10 waiting (WAIT file or backoff); 11 daily cap reached; 12
# STOP present (before or after the session, committed or not); 13 another runner holds the
# lock; 14 gh or modal not usable (backoff set); 2 any other precondition failed; 130
# interrupted by a signal (the post-session steps still ran).
#
# Before the session: take the lock (a stale lock is taken over under a second lock, so two
# runners cannot both win); for a runner that died, put back the .git/config its session
# changed, kill the processes it left, and finish that iteration's bookkeeping from its trust
# anchor; save the commits and edits of a half-done rebase, merge, cherry-pick or revert
# (branch loop-rescue/<id>, pushed, and a stash), then abort it; return to main; put back a
# committed STOP the tree lost; commit a STOP left in the working tree; note files in the
# clone that change what a session loads (.claude/settings.local.json, untracked .claude/
# files, CLAUDE.md or CLAUDE.local.md at any depth, git hooks, push URLs); stash a dirty tree
# (never STOP; conflicts are staged first; never while .loop/ or data/ is not ignored); sync
# main with origin (a conflict that will not rebase: keep the loop's commits on a pushed
# loop-diverged-* branch, reset a clean main to origin, STOP); then check STOP before anything
# from HEAD runs; then take the loop files from HEAD and check those files, the .gitignore
# rules, the setup, Modal apps from earlier sessions still unchecked (no session until they
# are; STOP once the oldest waiting window ended APPS_UNCHECKED_STOP_SEC ago), WAIT, the
# backoff, the daily and total caps (from transcripts and from the log), gh and modal logins,
# and running loop apps against the ledger; then record GANDALF's branches outside study04/,
# its tags, and origin's tags. At every start the Study 2 data link is excluded locally and
# the tracked files it hides (data/.gitkeep) are marked skip-worktree.
#
# The session: claude -p in its own session and process group, with SIGPIPE, SIGINT, SIGQUIT
# and SIGXFSZ reset to their defaults, and a watchdog that records its processes and kills
# the whole tree at SESSION_TIME_LIMIT_SEC.
#
# After it: kill left-over processes, including any process without a terminal whose working
# directory is in the clone or the GANDALF worktree; seal the transcript (its sha256 in
# .loop/transcripts.sha256); put back a .git/config the session changed; transcript checks T2,
# T3 and T5 (before anything is stashed); the .gitignore rules; rescue a half-done git
# operation; save commits left off main on a pushed branch; put back a committed STOP the
# session deleted (a finding); commit a STOP the session left uncommitted (with its Blocking
# line when QUESTIONS.md only gained lines); set the data link's skip-worktree bits again;
# commit the session's uncommitted log lines (only if they append) and WAIT; stash the rest;
# append a stub entry if the session's entry has no End: line or work was stashed or
# rescued; push (a failure leaves .loop/push_pending, retried before the next session); then
# the guards (guards.py commit-guards with provenance, frozen-check, ledger-verify,
# transcript T1, with the launcher's own report check, and T4, a hard-stop outcome without
# STOP, GANDALF's branches and tags against the record and origin's tags against the list
# taken before the session, streak, and apps-check over the session's time window). A
# missing transcript after a session that committed is a finding. Any finding writes STOP,
# committed and pushed (or, if the local commit fails, pushed straight onto origin/main).
# Then the backoff: a rate_limit_event whose
# status is not 'allowed' sets it to the later of now + FAST_FAIL_BACKOFF_SEC and its reset
# time + RATE_LIMIT_MARGIN_SEC; a session that failed within FAST_FAIL_SEC sets now +
# FAST_FAIL_BACKOFF_SEC.
#
# Every git command of the runner runs with hooks and fsmonitor off and signing off, so code
# a session plants in .git cannot run inside the runner.
#
# Written for macOS: bash 3.2, BSD date/stat/ps/lsof, no GNU timeout. The whole script is one
# function that bash parses before running it, so a session that edits this file on disk
# cannot change what the running runner does. For the same reason the checks run from a
# copy of the loop files (guards.py, loopcommon.py, frozen.json, config.env, settings.json,
# prompt.md, modal_launch.py) taken from the commit the session started from, under the
# interpreter's real path with -I -S, and claude gets its --settings from that copy.

set -u
umask 022

STUDY_REL="studies/04-phase-space-helicity"
LOOP_REL="$STUDY_REL/loop"
CONFIG_KEYS="COMPUTE_CAP_A100_HOURS CLAUDE_BIN CLAUDE_MODEL CLAUDE_EFFORT MAX_BUDGET_USD DAILY_ITERATION_CAP TOTAL_ITERATION_CAP SESSION_TIME_LIMIT_SEC PAUSE_BETWEEN_ITERATIONS_SEC FAST_FAIL_SEC FAST_FAIL_BACKOFF_SEC WIP_STREAK_LIMIT WATCHDOG_POLL_SEC KILL_GRACE_SEC GANDALF_WORKTREE MODAL_APP_PREFIX MODAL_VOLUME MODAL_VOLUME_ROOT LOOP_GIT_NAME LOOP_GIT_EMAIL MODAL_BIN GH_BIN PYTHON_BIN UV_BIN STOP_ON_ANY_NEW_MODAL_APP RATE_LIMIT_MARGIN_SEC APPS_UNCHECKED_STOP_SEC"
NUMERIC_KEYS="DAILY_ITERATION_CAP TOTAL_ITERATION_CAP SESSION_TIME_LIMIT_SEC PAUSE_BETWEEN_ITERATIONS_SEC FAST_FAIL_SEC FAST_FAIL_BACKOFF_SEC WIP_STREAK_LIMIT WATCHDOG_POLL_SEC KILL_GRACE_SEC RATE_LIMIT_MARGIN_SEC APPS_UNCHECKED_STOP_SEC"
NET_TIMEOUT=300   # seconds for git fetch/push; a hung network call must not hang the loop
DATA_LINK_REL="studies/02-collisionality-scan/data"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

utc_now() { date -u +%Y-%m-%dT%H:%M:%SZ; }

say() {  # say MESSAGE: to stdout and the detail log
  local line
  line="$(utc_now) [run_iteration] $*"
  echo "$line"
  if [ -d "$STATE" ]; then echo "$line" >> "$DETAIL"; fi
}

clean() {  # clean TEXT: one token for a key=value field (no spaces, at most 160 chars)
  printf '%s' "$1" | tr ' \t\n' '___' | cut -c1-160
}

note() {  # note KEY VALUE: add a key=value pair to this attempt's runner.log line
  NOTES="$NOTES $1=$(clean "$2")"
}

runlog() {  # runlog RESULT [key=value...]: the one runner.log line for this attempt
  local result="$1"
  shift
  [ "${RUNLOGGED:-0}" = 1 ] && return 0
  RUNLOGGED=1
  mkdir -p "$STATE" 2>/dev/null
  echo "$(utc_now) ${ITER_ID:--} $result $*$NOTES" >> "$RUNLOG"
  say "result: $result $*$NOTES"
}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

trim() {  # trim TEXT: without leading and trailing whitespace
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf '%s' "$s"
}

read_config_file() {  # read_config_file FILE: KEY=VALUE lines into the config variables, never executed
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    line=$(trim "$line")
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in 'export '*) line=$(trim "${line#export }") ;; esac
    key="${line%%=*}"
    [ "$key" = "$line" ] && continue
    key=$(trim "$key")
    case " $CONFIG_KEYS " in *" $key "*) ;; *) continue ;; esac
    val=$(trim "${line#*=}")
    case "$val" in
      \"*\") val="${val#\"}"; val="${val%\"}" ;;
      \'*\') val="${val#\'}"; val="${val%\'}" ;;
      *) val=$(trim "${val%%[[:space:]]#*}") ;;
    esac
    eval "$key=\$val"
  done < "$1"
}

load_config() {  # load_config FILE: read it, apply S04_ overrides, fill defaults, validate
  local f="$1" k v
  for k in $CONFIG_KEYS; do eval "$k="; done
  if [ ! -f "$f" ]; then CONFIG_ERROR="missing $f"; return 1; fi
  read_config_file "$f" || { CONFIG_ERROR="cannot read $f"; return 1; }
  for k in $CONFIG_KEYS; do
    eval "v=\${S04_$k:-}"
    if [ -n "$v" ]; then eval "$k=\$v"; fi
  done
  [ -n "$MODAL_BIN" ] || MODAL_BIN="$REPO/.venv/bin/modal"
  [ -n "$GH_BIN" ] || GH_BIN=gh
  [ -n "$PYTHON_BIN" ] || PYTHON_BIN="$REPO/.venv/bin/python"
  [ -n "$UV_BIN" ] || UV_BIN=uv
  [ -n "$RATE_LIMIT_MARGIN_SEC" ] || RATE_LIMIT_MARGIN_SEC=300
  [ -n "$APPS_UNCHECKED_STOP_SEC" ] || APPS_UNCHECKED_STOP_SEC=86400
  [ -n "$STOP_ON_ANY_NEW_MODAL_APP" ] || STOP_ON_ANY_NEW_MODAL_APP=1
  for k in CLAUDE_BIN MODAL_BIN GH_BIN PYTHON_BIN UV_BIN GANDALF_WORKTREE; do
    eval "v=\$$k"
    case "$v" in "~/"*) eval "$k=\"\$HOME/\${v#\~/}\"";; esac
  done
  for k in $NUMERIC_KEYS; do
    eval "v=\$$k"
    case "$v" in ''|*[!0-9]*) CONFIG_ERROR="$k must be a whole number (got '$v')"; return 1;; esac
  done
  case "$STOP_ON_ANY_NEW_MODAL_APP" in 0|1) ;; *) CONFIG_ERROR="STOP_ON_ANY_NEW_MODAL_APP must be 0 or 1"; return 1;; esac
  for k in CLAUDE_BIN CLAUDE_MODEL CLAUDE_EFFORT LOOP_GIT_NAME LOOP_GIT_EMAIL MODAL_APP_PREFIX GANDALF_WORKTREE; do
    eval "v=\$$k"
    [ -n "$v" ] || { CONFIG_ERROR="$k is empty"; return 1; }
  done
  [ -x "$PYTHON_BIN" ] || { CONFIG_ERROR="PYTHON_BIN $PYTHON_BIN is not executable"; return 1; }
  [ "$WATCHDOG_POLL_SEC" -ge 1 ] || WATCHDOG_POLL_SEC=1
  return 0
}

resolve_guard_python() {  # the interpreter's real path, so the clone's venv cannot hook the checks
  local p
  p=$("$PYTHON_BIN" -I -S -c 'import os, sys; print(os.path.realpath(sys.executable))' 2>/dev/null)
  if [ -n "$p" ] && [ -x "$p" ]; then GUARD_PY="$p"; else GUARD_PY="$PYTHON_BIN"; fi
}

# ---------------------------------------------------------------------------
# Git and process helpers
# ---------------------------------------------------------------------------

g() { git -C "$REPO" -c core.hooksPath=/dev/null -c core.fsmonitor=false -c commit.gpgSign=false -c core.quotePath=false "$@"; }

with_timeout() {  # with_timeout SECONDS COMMAND...: run it; status 124 if it overruns (killed)
  local secs="$1" pid start n=0
  shift
  "$@" &
  pid=$!
  start=$(date +%s)
  while kill -0 "$pid" 2>/dev/null; do
    if [ $(( $(date +%s) - start )) -ge "$secs" ]; then
      kill -TERM "$pid" 2>/dev/null
      sleep 1
      kill -KILL "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      return 124
    fi
    if [ "$n" -lt 20 ]; then sleep 0.05; else sleep 0.25; fi
    n=$((n + 1))
  done
  wait "$pid"
}

gnet() { with_timeout "$NET_TIMEOUT" git -C "$REPO" -c core.hooksPath=/dev/null -c core.fsmonitor=false -c commit.gpgSign=false "$@"; }

guard() {  # guard SUBCOMMAND ...: guards.py from the trusted copy, against this repo (status 3: not run)
  if [ -z "$GUARD_PY" ]; then
    echo "guards not run: the loop is stopped, so neither HEAD's loop files nor its config.env may run" >&2
    return 3
  fi
  "$GUARD_PY" -I -S "$TRUSTED/guards.py" --repo "$REPO" "$@"
}

make_trusted() {  # make_trusted REV: copy the loop's checking files out of commit REV
  # modal_launch.py is copied for T1, which runs the launcher's own report check from it; a
  # commit without it makes that check fail, which counts against any PASS report.
  local rev="$1" f
  mkdir -p "$TRUSTED" || return 1
  for f in guards.py loopcommon.py frozen.json config.env settings.json prompt.md modal_launch.py; do
    rm -f "$TRUSTED/$f"
    if g cat-file -e "$rev:$LOOP_REL/$f" 2>/dev/null; then
      g cat-file -p "$rev:$LOOP_REL/$f" > "$TRUSTED/$f" || return 1
    fi
  done
  [ -s "$TRUSTED/guards.py" ] && [ -s "$TRUSTED/loopcommon.py" ] && [ -s "$TRUSTED/config.env" ]
}

proc_lstart() {  # proc_lstart PID: start time of a live process, whitespace normalised
  ps -o lstart= -p "$1" 2>/dev/null | awk '{$1 = $1; print}'
}

alive() {  # alive PID: exists and is not a zombie
  kill -0 "$1" 2>/dev/null || return 1
  case "$(ps -o stat= -p "$1" 2>/dev/null)" in Z*|'') return 1;; esac
  return 0
}

session_procs() {  # session_procs ROOT: "pid lstart" for ROOT, its descendants, and its process group
  ps -axo pid=,ppid=,pgid=,lstart= 2>/dev/null | awk -v root="$1" -v self="$$" '
    { p = $1; pp[p] = $2; pg[p] = $3; $1 = ""; $2 = ""; $3 = ""; sub(/^ +/, ""); ls[p] = $0; ord[++n] = p }
    END {
      want[root] = 1
      for (i = 1; i <= n; i++) if (pg[ord[i]] == root) want[ord[i]] = 1
      changed = 1
      while (changed) {
        changed = 0
        for (i = 1; i <= n; i++) { q = ord[i]; if (!(q in want) && (pp[q] in want)) { want[q] = 1; changed = 1 } }
      }
      for (i = 1; i <= n; i++) { q = ord[i]; if ((q in want) && q != self) print q " " ls[q] }
    }'
}

track_snapshot() {  # record the session's processes (pid + start time) in its tracked file
  local line
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    case "$TRACKED_SEEN" in *"|$line|"*) continue;; esac
    TRACKED_SEEN="$TRACKED_SEEN|$line|"
    echo "$line" >> "$TRACKED_FILE"
  done <<EOF
$(session_procs "$CPID")
EOF
}

kill_pids() {  # kill_pids PID...: TERM, wait up to KILL_GRACE_SEC, then KILL whatever is left
  local p left waited=0
  [ $# -gt 0 ] || return 0
  for p in "$@"; do kill -TERM "$p" 2>/dev/null; done
  while [ "$waited" -lt "$KILL_GRACE_SEC" ]; do
    left=0
    for p in "$@"; do if alive "$p"; then left=1; fi; done
    [ "$left" = 0 ] && return 0
    sleep 1
    waited=$((waited + 1))
  done
  for p in "$@"; do if alive "$p"; then kill -KILL "$p" 2>/dev/null; fi; done
  return 0
}

sweep_tracked() {  # sweep_tracked FILE...: kill recorded processes still alive with the same start time
  local f line pid lst victims=""
  SWEPT=0
  for f in "$@"; do
    [ -f "$f" ] || continue
    while IFS= read -r line; do
      [ -n "$line" ] || continue
      pid=${line%% *}
      lst=${line#* }
      [ "$pid" = "$$" ] && continue
      alive "$pid" || continue
      [ "$(proc_lstart "$pid")" = "$lst" ] || continue
      case " $victims " in *" $pid "*) continue;; esac
      victims="$victims $pid"
    done < "$f"
  done
  [ -n "$victims" ] || return 0
  say "killing left-over session processes:$victims"
  # shellcheck disable=SC2086
  kill_pids $victims
  # shellcheck disable=SC2086
  SWEPT=$(set -- $victims; echo $#)
}

sweep_worktree() {  # the GANDALF worktree's real path, if it is safe to sweep: a git work tree, never / or $HOME or above them
  local wt home
  [ -n "${GANDALF_WORKTREE:-}" ] && [ -e "$GANDALF_WORKTREE/.git" ] || return 0
  wt=$(cd "$GANDALF_WORKTREE" 2>/dev/null && pwd -P) || return 0
  home=$(cd "${HOME:-/}" 2>/dev/null && pwd -P)
  case "$wt" in ''|/) return 0 ;; esac
  case "$REPO/" in "$wt/"*) return 0 ;; esac
  case "$home/" in "$wt/"*) return 0 ;; esac
  echo "$wt"
}

cwd_sweep() {  # cwd_sweep WHEN: kill processes without a terminal whose cwd is in the clone or the worktree
  # Claude Code runs each tool command as its own session leader, so `cmd &` in a tool call
  # outlives the session's process group. Such orphans are found by their working directory.
  # Never the runner, its ancestors (run_forever.sh) or its own helpers, never caffeinate,
  # and never a process with a terminal (Anjor's shells). A process that a session started
  # in another directory and detached from its parent before a snapshot is not found.
  local when="$1" wt victims n
  SWEPT_CWD=0
  [ -x /usr/sbin/lsof ] || return 0
  wt=$(sweep_worktree)
  ps -axo pid=,ppid=,tty=,comm= > "$TRUSTED/ps.txt" 2>/dev/null || return 0
  with_timeout 60 /usr/sbin/lsof -w -n -P -a -d cwd -u "$(id -u)" -Fpn > "$TRUSTED/cwd.txt" 2>/dev/null
  victims=$(awk -v self="$$" -v r1="$REPO" -v r2="${wt:-}" '
    function inside(path) {
      return (path == r1 || index(path, r1 "/") == 1 || (r2 != "" && (path == r2 || index(path, r2 "/") == 1)))
    }
    FNR == NR { pp[$1] = $2; tt[$1] = $3; c = $0; sub(/^ *[0-9]+ +[0-9]+ +[^ ]+ +/, "", c); comm[$1] = c; next }
    /^p/ { cur = substr($0, 2); next }
    /^n/ { if (inside(substr($0, 2))) cand[cur] = 1; next }
    END {
      a = self
      while (a != "" && a != "0" && !(a in anc)) { anc[a] = 1; a = pp[a] }
      desc[self] = 1
      changed = 1
      while (changed) { changed = 0; for (p in pp) if (!(p in desc) && (pp[p] in desc)) { desc[p] = 1; changed = 1 } }
      for (p in cand) {
        if (!(p in pp) || (p in anc) || (p in desc) || tt[p] != "??") continue
        name = comm[p]; sub(/.*\//, "", name)
        if (name == "caffeinate") continue
        print p
      }
    }' "$TRUSTED/ps.txt" "$TRUSTED/cwd.txt" | tr '\n' ' ')
  [ -n "$(trim "$victims")" ] || return 0
  say "killing processes without a terminal working in the clone or the worktree ($when):$victims"
  # shellcheck disable=SC2086
  kill_pids $victims
  # shellcheck disable=SC2086
  n=$(set -- $victims; echo $#)
  SWEPT_CWD=$n
}

kill_session_tree() {  # kill the session: its process group, every descendant, tracked processes, orphans
  local pids n
  track_snapshot
  pids=$(session_procs "$CPID" | awk '{print $1}' | tr '\n' ' ')
  kill -TERM -- "-$CPID" 2>/dev/null
  # shellcheck disable=SC2086
  kill_pids $pids
  kill -KILL -- "-$CPID" 2>/dev/null
  # shellcheck disable=SC2086
  n=$(set -- $pids; echo $#)
  sweep_tracked "$TRACKED_FILE"
  cwd_sweep "session killed"
  KILLED=$((KILLED + n + SWEPT + SWEPT_CWD))
  say "killed the session tree ($n processes, $SWEPT more from the tracked list, $SWEPT_CWD by working directory)"
}

# ---------------------------------------------------------------------------
# Signals in critical sections
# ---------------------------------------------------------------------------

crit_begin() { CRIT_DEPTH=$((CRIT_DEPTH + 1)); }

crit_end() {  # leave a critical section; before the session, a signal that arrived in it now exits
  CRIT_DEPTH=$((CRIT_DEPTH - 1))
  [ "$CRIT_DEPTH" -gt 0 ] && return 0
  CRIT_DEPTH=0
  if [ "$INTERRUPTED" = 1 ] && [ "$PHASE" = pre ]; then
    say "a signal arrived during a change to git state; the change is complete, exiting"
    runlog abort-signal
    exit 130
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Lock
# ---------------------------------------------------------------------------

lock_write() { echo "$$" > "$LOCK/pid"; proc_lstart "$$" > "$LOCK/lstart"; OWN_LOCK=1; }

take_lock() {  # mkdir is atomic; the lock records our pid and start time
  local pid lst age
  if mkdir "$LOCK" 2>/dev/null; then lock_write; return 0; fi
  pid=$(cat "$LOCK/pid" 2>/dev/null)
  lst=$(cat "$LOCK/lstart" 2>/dev/null)
  if [ -z "$pid" ]; then
    # Being created this instant, or left by a runner that died between mkdir and write.
    age=$(( $(date +%s) - $(stat -f %m "$LOCK" 2>/dev/null || echo 0) ))
    if [ "$age" -lt 60 ]; then LOCK_HOLDER=starting; return 1; fi
  elif kill -0 "$pid" 2>/dev/null; then
    if [ -z "$lst" ] || [ "$(proc_lstart "$pid")" = "$lst" ]; then LOCK_HOLDER=$pid; return 1; fi
  fi
  [ -n "${S04_TEST_LOCK_DELAY:-}" ] && sleep "$S04_TEST_LOCK_DELAY"
  # The lock is stale. Take it over under a second lock, and only if it is still the stale
  # one we read, so that two runners that both saw it stale cannot both win.
  if ! mkdir "$LOCK.takeover" 2>/dev/null; then
    age=$(( $(date +%s) - $(stat -f %m "$LOCK.takeover" 2>/dev/null || echo 0) ))
    if [ "$age" -lt 120 ]; then LOCK_HOLDER=takeover; return 1; fi
    rm -rf "$LOCK.takeover"  # left by a runner that died during a takeover
    mkdir "$LOCK.takeover" 2>/dev/null || { LOCK_HOLDER=takeover; return 1; }
  fi
  if [ "$(cat "$LOCK/pid" 2>/dev/null)" != "$pid" ] || [ "$(cat "$LOCK/lstart" 2>/dev/null)" != "$lst" ]; then
    rmdir "$LOCK.takeover" 2>/dev/null
    LOCK_HOLDER=changed
    return 1
  fi
  note stale_lock "${pid:-none}"
  say "taking over a stale lock (pid ${pid:-none} is gone)"
  rm -rf "$LOCK"
  if mkdir "$LOCK" 2>/dev/null; then
    lock_write
    rmdir "$LOCK.takeover" 2>/dev/null
    return 0
  fi
  rmdir "$LOCK.takeover" 2>/dev/null
  LOCK_HOLDER=unknown
  return 1
}

release_lock() {
  if [ "$OWN_LOCK" = 1 ] && [ "$(cat "$LOCK/pid" 2>/dev/null)" = "$$" ]; then rm -rf "$LOCK"; fi
  OWN_LOCK=0
}

# ---------------------------------------------------------------------------
# The clone
# ---------------------------------------------------------------------------

ensure_identity() {  # every commit in the clone carries the loop's name, with or without the environment
  export GIT_AUTHOR_NAME="$LOOP_GIT_NAME" GIT_AUTHOR_EMAIL="$LOOP_GIT_EMAIL"
  export GIT_COMMITTER_NAME="$LOOP_GIT_NAME" GIT_COMMITTER_EMAIL="$LOOP_GIT_EMAIL"
  if [ "$(g config --local --get user.name 2>/dev/null)" != "$LOOP_GIT_NAME" ]; then
    g config --local user.name "$LOOP_GIT_NAME" && note set_user_name "$LOOP_GIT_NAME"
  fi
  if [ "$(g config --local --get user.email 2>/dev/null)" != "$LOOP_GIT_EMAIL" ]; then
    g config --local user.email "$LOOP_GIT_EMAIL" && note set_user_email "$LOOP_GIT_EMAIL"
  fi
}

ensure_excludes() {  # the runner's state and run data never go into a stash, whatever .gitignore says
  local ex rule
  ex=$(g rev-parse --git-path info/exclude 2>/dev/null) || return 0
  case "$ex" in /*) ;; *) ex="$REPO/$ex" ;; esac
  mkdir -p "$(dirname "$ex")"
  for rule in "/$STUDY_REL/.loop/" "**/data/"; do
    if ! grep -qxF -- "$rule" "$ex" 2>/dev/null; then echo "$rule" >> "$ex"; note excluded "$rule"; fi
  done
  ensure_data_link start
}

ensure_data_link() {  # ensure_data_link start|post: keep the Study 2 data link and the files it hides out of the way of git
  # The clone links Study 2's data to Anjor's checkout. The **/data/ rule does not match a
  # symlink, and a stash would remove the link, so the link is excluded locally. The link
  # replaces a folder that holds tracked files (data/.gitkeep): git then sees them as deleted,
  # 'beyond a symbolic link', and a stash cannot put them back, so each is marked
  # skip-worktree. 'git read-tree HEAD' drops that bit, so it is set again before the
  # post-session stash too.
  local when="$1" ex path n=0
  [ -L "$REPO/$DATA_LINK_REL" ] || return 0
  # --no-index: the index holds data/.gitkeep, and without it git calls no tracked path ignored.
  if ! g check-ignore -q --no-index -- "$DATA_LINK_REL"; then
    ex=$(g rev-parse --git-path info/exclude 2>/dev/null) || return 0
    case "$ex" in /*) ;; *) ex="$REPO/$ex" ;; esac
    mkdir -p "$(dirname "$ex")"
    echo "/$DATA_LINK_REL" >> "$ex"
    note excluded_data_link "$DATA_LINK_REL"
  fi
  while IFS= read -r path; do
    [ -n "$path" ] || continue
    if g update-index --skip-worktree -- "$path" >/dev/null 2>&1; then
      n=$((n + 1))
    else
      note skip_worktree_failed "$path"
      say "could not mark $path skip-worktree; a stash will fail on it"
    fi
  done <<EOF
$(g ls-files -v -- "$DATA_LINK_REL/" 2>/dev/null | awk '$1 != "S" && $1 != "s" { sub(/^[^ ]+ /, ""); print }')
EOF
  if [ "$n" -gt 0 ]; then
    if [ "$when" = post ]; then note skip_worktree_reset "$n"; else note skip_worktree "$n"; fi
    say "marked $n tracked file(s) under the Study 2 data link skip-worktree ($when)"
  fi
  return 0
}

latest_iter() {  # the newest iteration ID with a transcript, or 'unknown'
  local id
  id=$(ls "$STATE/transcripts" 2>/dev/null | sed -n 's/\.jsonl$//p' | sort | tail -1)
  echo "${id:-unknown}"
}

keep_stop_begin() {  # remember STOP's text before a step that can remove it from the tree
  KEPT_STOP=""
  if [ -f "$STUDY/STOP" ]; then KEPT_STOP="$TRUSTED/stop.kept"; cp "$STUDY/STOP" "$KEPT_STOP"; fi
}

keep_stop_end() {  # put STOP back (untracked) if that step removed it; it is committed later
  if [ -n "$KEPT_STOP" ] && [ ! -e "$STUDY/STOP" ]; then
    cp "$KEPT_STOP" "$STUDY/STOP"
    note restored_stop 1
    say "put STOP back into the working tree after the git operation removed it"
  fi
  KEPT_STOP=""
}

dirty_status() {  # porcelain status of everything but STOP ("!!" if git status itself fails)
  g status --porcelain -- . ":(exclude)$STUDY_REL/STOP" 2>/dev/null || echo "!! git status failed"
}

stop_present() {  # STOP exists in the working tree or is committed at HEAD (a deletion that is not committed does not count)
  [ -e "$STUDY/STOP" ] || g cat-file -e "HEAD:$STUDY_REL/STOP" 2>/dev/null
}

restore_deleted_stop() {  # put back a committed STOP that the working tree or the index lost; 0 if it did
  local fixed=1
  g cat-file -e "HEAD:$STUDY_REL/STOP" 2>/dev/null || return 1
  if ! g ls-files --error-unmatch -- "$STUDY_REL/STOP" >/dev/null 2>&1; then
    g reset -q -- "$STUDY_REL/STOP" >/dev/null 2>&1
    fixed=0
  fi
  if [ ! -e "$STUDY/STOP" ]; then
    g checkout -q HEAD -- "$STUDY_REL/STOP" >/dev/null 2>&1
    fixed=0
  fi
  if [ "$fixed" = 0 ]; then say "STOP is committed but was deleted from the working tree or the index; put it back"; fi
  return "$fixed"
}

ignores_ok() {  # the study's .loop/ and data/ are ignored by git (by .gitignore or .git/info/exclude)
  g check-ignore -q --no-index -- "$STUDY_REL/.loop/runner.log" 2>/dev/null \
    && g check-ignore -q --no-index -- "$STUDY_REL/data/probe" 2>/dev/null
}

stage_all() {  # stage every change but STOP; only tracked files when .loop/ or data/ is not ignored
  if ignores_ok; then
    g add -A -- . ":(exclude)$STUDY_REL/STOP" >/dev/null 2>&1
  else
    g add -u -- . ":(exclude)$STUDY_REL/STOP" >/dev/null 2>&1
  fi
}

questions_only_gained() {  # QUESTIONS.md only gained lines since HEAD (a 'None.' placeholder or a blank line may go)
  local q="$STUDY_REL/QUESTIONS.md"
  [ -f "$REPO/$q" ] || return 1
  [ -z "$(g ls-files -u -- "$q" 2>/dev/null)" ] || return 1
  g cat-file -e "HEAD:$q" 2>/dev/null || return 1
  g diff -U0 --no-color --no-ext-diff --no-textconv HEAD -- "$q" 2>/dev/null | awk '
    /^(---|\+\+\+) / { next }
    /^-/ { s = substr($0, 2); gsub(/^[ \t]+|[ \t]+$/, "", s); gsub(/^[_*]+|[_*]+$/, "", s)
           if (s != "" && s != "None." && s != "None") bad = 1 }
    END { exit bad }'
}

record_conflicts() {  # record_conflicts ID: stage unmerged paths (with their markers) so they can be stashed
  local id="$1" f
  [ -n "$(g ls-files -u 2>/dev/null)" ] || return 0
  mkdir -p "$STATE/rescue"
  f="$STATE/rescue/$id-$(date -u +%Y%m%d-%H%M%S).patch"
  { g status --porcelain; echo; g diff --binary --no-ext-diff --no-textconv HEAD; } > "$f" 2>/dev/null
  stage_all
  note conflicts_staged "$(basename "$f")"
  RESCUE_NOTES="$RESCUE_NOTES; unmerged files (a conflicted stash pop or pull) were staged with their conflict markers and stashed, and the diff was copied to .loop/rescue/$(basename "$f")"
  say "staged unmerged files with their conflict markers (copy in .loop/rescue/$(basename "$f"))"
}

rescue_midop() {  # rescue_midop OP ID: keep a half-done operation's commits and edits, then abort it
  local op="$1" id="$2" br where s
  crit_begin
  keep_stop_begin
  if ! g symbolic-ref -q HEAD >/dev/null 2>&1 && ! g merge-base --is-ancestor HEAD main 2>/dev/null; then
    br="loop-rescue/$id"
    if g branch -f "$br" HEAD >/dev/null 2>&1; then
      if gnet push -q origin "$br" >/dev/null 2>&1; then where="pushed to origin"; else where="local only"; fi
      note rescue_branch "$br"
      RESCUE_NOTES="$RESCUE_NOTES; commits made during a stopped $op are on branch $br ($where)"
      say "commits of the stopped $op are on branch $br ($where)"
    fi
  fi
  stage_all
  s=$(g stash create "loop-stash $id (edits during a stopped $op)" 2>/dev/null)
  if [ -n "$s" ] && g stash store -m "loop-stash $id (edits during a stopped $op)" "$s" >/dev/null 2>&1; then
    s=$(g rev-parse --short=12 "$s")
    note rescue_stash "$s"
    if [ "$STASH_REF" = none ]; then STASH_REF="$s"; else STASH_REF="$STASH_REF,$s"; fi
    RESCUE_NOTES="$RESCUE_NOTES; edits made during the stopped $op are in stash $s ('loop-stash $id (edits during a stopped $op)')"
  fi
  case "$op" in
    rebase) g rebase --abort >/dev/null 2>&1 ;;
    am) g am --abort >/dev/null 2>&1 ;;
    merge) g merge --abort >/dev/null 2>&1 ;;
    cherry-pick) g cherry-pick --abort >/dev/null 2>&1 ;;
    revert) g revert --abort >/dev/null 2>&1 ;;
  esac
  if [ -d "$GITDIR/rebase-merge" ] || [ -d "$GITDIR/rebase-apply" ] || [ -f "$GITDIR/MERGE_HEAD" ] \
     || [ -f "$GITDIR/CHERRY_PICK_HEAD" ] || [ -f "$GITDIR/REVERT_HEAD" ]; then
    # The abort failed; everything is saved, so leave the operation the hard way.
    g rebase --quit >/dev/null 2>&1; g merge --quit >/dev/null 2>&1
    g cherry-pick --quit >/dev/null 2>&1; g revert --quit >/dev/null 2>&1
    rm -f "$GITDIR/MERGE_HEAD" "$GITDIR/CHERRY_PICK_HEAD" "$GITDIR/REVERT_HEAD"
    g reset -q --hard >/dev/null 2>&1
    note forced_abort "$op"
  fi
  note aborted "$op"
  keep_stop_end
  crit_end
}

clear_git_state() {  # clear_git_state pre|post ID: save, then abort, half-done git operations
  local mode="$1" id="$2" age op=""
  if [ -f "$GITDIR/index.lock" ]; then
    age=$(( $(date +%s) - $(stat -f %m "$GITDIR/index.lock" 2>/dev/null || echo 0) ))
    # After a session every session process is dead, so its lock is stale. Before one, only
    # an old lock is taken to be stale.
    if [ "$mode" = post ] || [ "$age" -gt 300 ]; then
      rm -f "$GITDIR/index.lock"; note removed_index_lock "${age}s"
    fi
  fi
  if [ -d "$GITDIR/rebase-merge" ] || [ -d "$GITDIR/rebase-apply" ]; then
    op=rebase
    [ -f "$GITDIR/rebase-apply/applying" ] && op=am
  elif [ -f "$GITDIR/MERGE_HEAD" ]; then op=merge
  elif [ -f "$GITDIR/CHERRY_PICK_HEAD" ]; then op=cherry-pick
  elif [ -f "$GITDIR/REVERT_HEAD" ]; then op=revert
  fi
  [ -n "$op" ] && rescue_midop "$op" "$id"
  if [ -f "$GITDIR/BISECT_LOG" ]; then g bisect reset >/dev/null 2>&1; note aborted bisect; fi
  return 0
}

stash_dirty() {  # stash_dirty MESSAGE: stash every uncommitted change but STOP; 1 if the tree stays dirty
  local msg="$1" sha
  STASH_LEFT=""
  [ -n "$(dirty_status)" ] || return 0
  if ! ignores_ok; then
    # A committed .gitignore rule ('!.../data/') beats .git/info/exclude: a stash would take
    # the run data and the runner's own state. Leave the tree as it is.
    STASH_LEFT="not stashed: the study's .loop/ or data/ is not ignored by git, so a stash would take the runner's state and the run data. Left in place: $(dirty_status | head -8 | tr '\n' ' ')"
    note stash_refused ignore_rules
    say "not stashing: .loop/ or data/ is not ignored by git; the tree is left as it is"
    return 1
  fi
  crit_begin
  record_conflicts "${ITER_ID:-$(latest_iter)}"
  if g stash push -u -q -m "$msg" -- . ":(exclude)$STUDY_REL/STOP" >/dev/null 2>&1; then
    sha=$(g rev-parse --short=12 -q --verify refs/stash)
    if [ "$STASH_REF" = none ]; then STASH_REF="$sha"; else STASH_REF="$STASH_REF,$sha"; fi
    note stash "$sha"
    say "stashed uncommitted changes as '$msg' ($sha)"
  else
    say "git stash failed"
  fi
  STASH_LEFT=$(dirty_status | head -8 | tr '\n' ' ')
  crit_end
  if [ -n "$STASH_LEFT" ]; then
    note stash_failed "$STASH_LEFT"
    say "the working tree is still dirty after the stash: $STASH_LEFT"
    return 1
  fi
  return 0
}

publish_stop_fallback() {  # publish_stop_fallback SHORT: push STOP onto origin/main without the local index
  local short="$1" blob base idx tree commit
  [ -f "$STUDY/STOP" ] || return 1
  gnet fetch -q origin >/dev/null 2>&1 || return 1
  base=$(g rev-parse -q --verify origin/main) || return 1
  if [ "$(g cat-file -p "$base:$STUDY_REL/STOP" 2>/dev/null)" = "$(cat "$STUDY/STOP")" ]; then return 0; fi
  blob=$(g hash-object -w -- "$STUDY/STOP") || return 1
  idx="$TRUSTED/stop-index"
  rm -f "$idx"
  GIT_INDEX_FILE="$idx" g read-tree "$base" || return 1
  GIT_INDEX_FILE="$idx" g update-index --add --cacheinfo "100644,$blob,$STUDY_REL/STOP" || return 1
  tree=$(GIT_INDEX_FILE="$idx" g write-tree) || return 1
  commit=$(g commit-tree "$tree" -p "$base" -m "Study 04 [${ITER_ID:-runner}]: runner STOP ($short), pushed without the local index") || return 1
  gnet push -q origin "$commit:refs/heads/main" >/dev/null 2>&1 || return 1
  note stop_published_directly "$(echo "$commit" | cut -c1-12)"
  say "pushed STOP straight onto origin/main as $(echo "$commit" | cut -c1-12)"
  return 0
}

commit_uncommitted_stop() {  # commit_uncommitted_stop ID: a STOP the working tree has and HEAD lacks is committed
  local id="$1" paths qs=""
  [ -f "$STUDY/STOP" ] || return 0
  [ -n "$(g status --porcelain -- "$STUDY_REL/STOP" 2>/dev/null)" ] || return 0
  crit_begin
  paths="$STUDY_REL/STOP"
  if [ -n "$(g status --porcelain -- "$STUDY_REL/QUESTIONS.md" 2>/dev/null)" ] && questions_only_gained; then
    paths="$paths $STUDY_REL/QUESTIONS.md"
    qs="+QUESTIONS.md"
  fi
  # shellcheck disable=SC2086
  if g add -- $paths >/dev/null 2>&1 \
     && g commit -q -m "Study 04 [$id]: runner committed a STOP left uncommitted" -- $paths >/dev/null 2>&1; then
    note committed_stop "STOP$qs"
    say "committed a STOP the session left uncommitted ($paths)"
  else
    say "could not commit the uncommitted STOP; publishing it directly"
    publish_stop_fallback "uncommitted STOP" || say "could not publish STOP; it stays in the working tree, and no session starts while it is there"
  fi
  crit_end
}

mark_push_pending() { echo "$(date +%s) $(utc_now) $PUSH_RESULT" > "$STATE/push_pending"; }

rebase_onto_origin() {
  if g rebase -q origin/main >/dev/null 2>&1; then return 0; fi
  g rebase --abort >/dev/null 2>&1
  return 1
}

sync_main() {  # sync_main pre|post: make local main and origin/main agree. Sets PUSH_RESULT.
  # 0 in sync; 1 network or push failure (.loop/push_pending kept); 3 diverged beyond a clean
  # rebase (pre: the local commits are saved on a branch and STOP is written); 4 local main
  # cannot be fast-forwarded (git state, not the network).
  local mode="$1" ahead behind
  PUSH_RESULT=ok
  if ! gnet fetch -q origin >/dev/null 2>&1; then PUSH_RESULT=fetch-failed; mark_push_pending; return 1; fi
  if ! g rev-parse -q --verify origin/main >/dev/null; then PUSH_RESULT=no-origin-main; mark_push_pending; return 1; fi
  ahead=$(g rev-list --count origin/main..HEAD 2>/dev/null || echo 0)
  behind=$(g rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
  if [ "$ahead" -eq 0 ]; then
    if [ "$behind" -gt 0 ]; then
      if ! FF_ERR=$(g merge -q --ff-only origin/main 2>&1); then PUSH_RESULT=ff-failed; return 4; fi
      PUSH_RESULT=pulled
    else
      PUSH_RESULT=in-sync
    fi
    rm -f "$STATE/push_pending"
    return 0
  fi
  if [ "$behind" -gt 0 ] && ! rebase_onto_origin; then
    if [ "$mode" = pre ]; then rescue_diverged; return 3; fi
    PUSH_RESULT=rebase-conflict
    mark_push_pending
    return 1
  fi
  if gnet push -q origin main >/dev/null 2>&1; then PUSH_RESULT=pushed; rm -f "$STATE/push_pending"; return 0; fi
  # Someone pushed in between: fetch, rebase and try once more.
  if gnet fetch -q origin >/dev/null 2>&1 && rebase_onto_origin && gnet push -q origin main >/dev/null 2>&1; then
    PUSH_RESULT=pushed
    rm -f "$STATE/push_pending"
    return 0
  fi
  PUSH_RESULT=push-failed
  mark_push_pending
  return 1
}

stop_write_plain() {  # stop_write_plain REASON [DECISION]: STOP without guards.py, in its format
  local reason="$1" decision="${2:-}"
  if [ -f "$STUDY/STOP" ]; then
    printf '\nAlso written by: runner (%s), %s\nReason: %s\n' "${ITER_ID:--}" "$(utc_now)" "$reason" >> "$STUDY/STOP"
  else
    printf 'STOP\nWritten by: runner (%s), %s\nReason: %s\n' "${ITER_ID:--}" "$(utc_now)" "$reason" > "$STUDY/STOP"
  fi
  if [ -n "$decision" ]; then printf 'Decision needed from Anjor: %s\n' "$decision" >> "$STUDY/STOP"; fi
}

write_stop() {  # write_stop SHORT REASON [DECISION]: STOP + Blocking line, committed and pushed
  local short="$1" reason="$2" decision="${3:-}" qflag="" paths written=0
  crit_begin
  # The Blocking line goes into QUESTIONS.md only if the session left no edits in it, and
  # only through guards.py, which runs only while its trusted copy may.
  if [ -n "$(g status --porcelain -- "$STUDY_REL/QUESTIONS.md" 2>/dev/null)" ]; then qflag=--no-questions; fi
  if [ -n "$GUARD_PY" ]; then
    if [ -n "$decision" ]; then
      env S04_ITER_ID="${ITER_ID:-}" "$GUARD_PY" -I -S "$TRUSTED/guards.py" --repo "$REPO" stop-write \
        --source runner --reason "$reason" --decision "$decision" $qflag >/dev/null 2>&1 && written=1
    else
      env S04_ITER_ID="${ITER_ID:-}" "$GUARD_PY" -I -S "$TRUSTED/guards.py" --repo "$REPO" stop-write \
        --source runner --reason "$reason" $qflag >/dev/null 2>&1 && written=1
    fi
  fi
  if [ "$written" = 0 ] || [ ! -f "$STUDY/STOP" ]; then
    stop_write_plain "$reason" "$decision"
    qflag=--no-questions
  fi
  STOP_WRITTEN=1
  say "wrote STOP: $short"
  paths="$STUDY_REL/STOP"
  if [ -z "$qflag" ] && [ -f "$STUDY/QUESTIONS.md" ]; then paths="$paths $STUDY_REL/QUESTIONS.md"; fi
  # On main, STOP is committed and pushed; anywhere else it goes straight onto origin/main.
  # shellcheck disable=SC2086
  if [ "$(g symbolic-ref -q --short HEAD 2>/dev/null)" = main ] && g add -- $paths >/dev/null 2>&1 \
     && g commit -q -m "Study 04 [${ITER_ID:-runner}]: runner STOP ($short)" -- $paths >/dev/null 2>&1; then
    if ! sync_main post; then
      say "could not push STOP ($PUSH_RESULT)"
      if [ "$PUSH_RESULT" = rebase-conflict ]; then
        publish_stop_fallback "$short" || say "could not publish STOP either; it is retried before the next session"
      fi
    fi
  else
    say "could not commit STOP; publishing it directly on origin/main"
    publish_stop_fallback "$short" || say "could not publish STOP either; it stays in the working tree, and no session starts while it is there"
  fi
  crit_end
  return 0
}

rescue_diverged() {  # keep the loop's local commits on a branch, take origin's main, write STOP
  local br pushed="not pushed"
  crit_begin
  br="loop-diverged-$(date -u +%Y%m%d-%H%M%S)"
  g branch -f "$br" HEAD >/dev/null 2>&1
  if gnet push -q origin "$br" >/dev/null 2>&1; then pushed="pushed to origin"; fi
  note rescued_branch "$br"
  if [ -n "$(g status --porcelain 2>/dev/null)" ]; then
    # Never reset a dirty tree: STOP goes on top of origin/main directly.
    write_stop "local main diverged" \
      "Local main and origin/main diverged and do not rebase cleanly. The loop's local commits are on branch $br ($pushed). The working tree was not clean, so nothing was reset: $(g status --porcelain | head -5 | tr '\n' ' ')" \
      "Merge what you want from $br into main, clean up the loop clone by hand (git status there), then resume."
  else
    g reset -q --hard origin/main
    write_stop "local main diverged" \
      "Local main and origin/main diverged and do not rebase cleanly. The loop's local commits are on branch $br ($pushed); local main now equals origin/main." \
      "Merge what you want from $br into main, delete the branch, then resume."
  fi
  crit_end
}

stash_note() {  # every stash entry with its sha, and unmerged rescue branches, for the prompt
  local s b
  s=$(g stash list --format='%gd (%h): %gs' 2>/dev/null | awk 'NR <= 12 { printf "%s%s", (NR > 1 ? "; " : ""), $0 } END { if (NR > 12) printf "; and %d more", NR - 12 }')
  b=$(g branch --no-merged main --list 'loop-*' --format='%(refname:short)' 2>/dev/null | tr '\n' ' ')
  b=$(trim "$b")
  if [ -n "$b" ]; then s="${s:+$s; }branches holding work not on main: $b"; fi
  echo "${s:-none}"
}

gandalf_refs_record() {  # gandalf_refs_record ID: before the session, GANDALF's own branches and tags, and origin's tags
  # The worktree shares its refs with Anjor's checkout of GANDALF, so its branches outside
  # study04/ and its tags are his. After the session a change to them writes STOP: a deleted
  # branch, a branch made or moved other than by Anjor's own git identity (the reflogs say),
  # and any tag made, moved or deleted, locally or on origin. A tag that origin already had
  # (Anjor's, fetched by the session) is fine. The record holds its time and that identity.
  local id="$1"
  rm -f "$STATE"/gandalf-refs.pre.* "$STATE"/gandalf-tags.origin.* "$STATE"/gandalf-tags.origin-after.*
  if guard gandalf-refs --worktree "$GANDALF_WORKTREE" > "$STATE/gandalf-refs.tmp" 2>/dev/null; then
    mv "$STATE/gandalf-refs.tmp" "$STATE/gandalf-refs.pre.$id"
  else
    rm -f "$STATE/gandalf-refs.tmp"
    note gandalf_refs_unrecorded 1
    say "could not record GANDALF's branches and tags before the session; they are not checked after it"
    return 0
  fi
  if ! with_timeout 60 git -C "$GANDALF_WORKTREE" -c core.hooksPath=/dev/null -c core.fsmonitor=false \
       ls-remote --tags origin > "$STATE/gandalf-tags.origin.$id" 2>/dev/null; then
    rm -f "$STATE/gandalf-tags.origin.$id"
    note gandalf_origin_tags unread
  fi
  return 0
}

# ---------------------------------------------------------------------------
# Modal apps against the ledger
# ---------------------------------------------------------------------------

apps_guard() {  # apps_guard WHEN: list Modal apps and check them. 0 ok, 12 STOP needed, 14 cannot list.
  local when="$1" rc out ws we
  set --
  if [ -s "$STATE/app_windows" ]; then
    while read -r ws we; do
      [ -n "${we:-}" ] && set -- "$@" --window "$ws" "$we"
    done < "$STATE/app_windows"
  fi
  [ "$STOP_ON_ANY_NEW_MODAL_APP" = 1 ] && set -- "$@" --any-name
  if ! with_timeout 120 "$MODAL_BIN" app list --json > "$STATE/apps.json" 2> "$STATE/apps.err"; then
    APPS_ERROR="modal app list failed: $(tail -1 "$STATE/apps.err" 2>/dev/null)"
    return 14
  fi
  out=$(guard apps-check --apps-json "$STATE/apps.json" --prefix "$MODAL_APP_PREFIX" "$@" 2>&1)
  rc=$?
  case "$rc" in
    0)
      rm -f "$STATE/app_windows"
      return 0
      ;;
    1)
      rm -f "$STATE/app_windows"
      APPS_PROBLEM="Modal apps that compute_ledger.json does not account for ($when):
$out
The runner never stops a Modal app. A running app may still be spending GPU time."
      return 12
      ;;
    3)
      APPS_PROBLEM="The output of 'modal app list --json' has a format the runner cannot read ($when), so it cannot check for GPU work outside the launcher:
$out"
      return 12
      ;;
    *)
      APPS_ERROR="cannot check the Modal app list: $out"
      return 14
      ;;
  esac
}

APPS_DECISION="For each app listed: find out what it is, stop it with 'modal app stop <app id>' if it should not run, and record what it cost. If you launched it yourself, say so. Then resume."
APPS_OVERDUE_DECISION="On this Mac, make 'modal app list --json' work again (login, network, CLI version). In the Modal dashboard, look for apps created during the windows above that compute_ledger.json does not know, and stop any that should not run. Then resume: the runner checks the windows before the next session. If you checked them by hand and the list still cannot be read, delete .loop/app_windows in the loop clone before you resume."

# A session's window waits in .loop/app_windows until 'modal app list --json' can be read and
# checked. Until then no session starts (the check is retried at every runner start). If the
# oldest waiting window ended APPS_UNCHECKED_STOP_SEC ago or more, the runner writes STOP, so
# that a check that cannot be made never stalls the loop without Anjor hearing of it.

apps_overdue() {  # 0 if the oldest session window still waiting for its check ended APPS_UNCHECKED_STOP_SEC ago or more
  local oldest
  [ -s "$STATE/app_windows" ] || return 1
  oldest=$(awk 'NR == 1 { print $2 }' "$STATE/app_windows")
  case "$oldest" in ''|*[!0-9]*) oldest=$(stat -f %m "$STATE/app_windows" 2>/dev/null || echo 0) ;; esac
  [ $(( $(date +%s) - oldest )) -ge "$APPS_UNCHECKED_STOP_SEC" ]
}

apps_overdue_reason() {  # the STOP text for session windows that could not be checked in time
  local ws we list="" limit
  while read -r ws we; do
    case "$ws:$we" in
      [0-9]*:[0-9]*) list="$list $(date -u -r "$ws" +%Y-%m-%dT%H:%M:%SZ) to $(date -u -r "$we" +%Y-%m-%dT%H:%M:%SZ);" ;;
      *) list="$list (unreadable line '$ws $we');" ;;
    esac
  done < "$STATE/app_windows"
  if [ "$APPS_UNCHECKED_STOP_SEC" -ge 3600 ]; then limit="$((APPS_UNCHECKED_STOP_SEC / 3600)) h"; else limit="${APPS_UNCHECKED_STOP_SEC} s"; fi
  printf "modal apps unchecked: the runner cannot read the Modal app list (%s), so for %s or more it has not been able to check for Modal apps created during these session windows (UTC):%s no session starts until they are checked." \
    "$APPS_ERROR" "$limit" "$list"
}

iter_start_epoch() {  # iter_start_epoch ID: the UTC minute an iteration ID names (it-YYYYMMDD-HHMM), as an epoch
  local d="${1#it-}"
  d="${d%%-*}${d#*-}"
  case "$d" in [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]) ;; *) return 1 ;; esac
  date -j -u -f '%Y%m%d%H%M%S' "${d}00" +%s 2>/dev/null
}

# ---------------------------------------------------------------------------
# After a session (also used to recover an iteration whose runner died)
# ---------------------------------------------------------------------------

check_git_config() {  # check_git_config ID: a .git/config the session changed is saved and put back (once)
  # The snapshot (.loop/git-config.pre.<id>) is taken just before the session starts and
  # kept until this check, so a runner that died is checked by the next one, before it runs git.
  local id="$1"
  [ -n "$GITCONFIG_SNAPSHOT" ] && [ -f "$GITCONFIG_SNAPSHOT" ] || return 0
  if ! cmp -s "$GITCONFIG_SNAPSHOT" "$GITDIR/config"; then
    cp "$GITDIR/config" "$STATE/git-config.$id" 2>/dev/null
    cp "$GITCONFIG_SNAPSHOT" "$GITDIR/config"
    GITCONFIG_CHANGED=1
    note git_config_restored "$id"
    say "the session changed .git/config; saved it as .loop/git-config.$id and put the old one back"
  fi
  rm -f "$GITCONFIG_SNAPSHOT"
  GITCONFIG_SNAPSHOT=""
}

transcript_unreadable() {  # transcript_unreadable OUTPUT BEFORE: a missing transcript stops the loop if the session committed
  [ "$TRANSCRIPT_LOST" = 1 ] && return 0
  TRANSCRIPT_LOST=1
  if [ "$(g rev-parse HEAD 2>/dev/null)" != "$2" ]; then
    add_reason "transcript missing" "transcript: the session's transcript is missing or unreadable ($(printf '%s\n' "$1" | head -1)), so its gate passes, GANDALF merges, Modal scripts and ledger commits cannot be checked"
  else
    note transcript_unchecked "$(printf '%s\n' "$1" | head -1)"
    say "transcript checks not run: $1"
  fi
}

seal_transcript() {  # seal_transcript FILE: record the finished transcript's sha256; T2 trusts only sealed transcripts
  [ -f "$1" ] || return 0
  guard transcript-seal --transcript "$1" --manifest "$STATE/transcripts.sha256" >/dev/null 2>&1 \
    || say "could not seal the transcript $1"
}

save_log_changes() {  # save_log_changes ID BEFORE: commit uncommitted log lines (appends only) and WAIT
  local id="$1" base="$2" paths=""
  if [ -n "$(g status --porcelain -- "$STUDY_REL/log" 2>/dev/null)" ]; then
    if guard log-check --base "$base" > "$STATE/logcheck.out" 2>&1; then
      paths="$STUDY_REL/log"
    else
      note log_left_in_stash "$(head -1 "$STATE/logcheck.out")"
    fi
  fi
  if [ -n "$(g status --porcelain -- "$STUDY_REL/WAIT" 2>/dev/null)" ]; then paths="$paths $STUDY_REL/WAIT"; fi
  [ -n "$paths" ] || return 0
  # shellcheck disable=SC2086
  if g add -A -- $paths && g commit -q -m "Study 04 [$id]: runner saved uncommitted log lines" -- $paths >/dev/null 2>&1; then
    # shellcheck disable=SC2086
    note saved_uncommitted "$(echo $paths | tr ' ' ',')"
    say "committed the session's uncommitted log lines / WAIT"
  fi
}

add_reason() {  # add_reason SHORT TEXT: a finding that will be written to STOP
  REASONS="$REASONS
$2"
  SHORT="${SHORT:+$SHORT, }$1"
}

finish_iteration() {  # finish_iteration ID BEFORE EXIT_CODE DURATION_SEC NOTE TRANSCRIPT SESSION_START
  local id="$1" before="$2" rc="$3" dur="$4" why="$5" transcript="$6" session_start="${7:-}" branch ec out n where title own_outcome
  STASH_REF=none; GUARDS=ok; RESCUE_NOTES=""; OUTCOME=""; REASONS=""; SHORT=""; DECISION=""; TRANSCRIPT_LOST=0
  # 0. A .git/config the session changed is put back before git runs again. The evidence file
  # .loop/git-config.<id> also carries the finding when an earlier runner start (a --push-only
  # run, say) already put the config back.
  check_git_config "$id"
  if [ "$GITCONFIG_CHANGED" = 1 ] || [ -f "$STATE/git-config.$id" ]; then
    add_reason "git config" "git: the session changed the clone's .git/config (the changed file is .loop/git-config.$id; the runner put the old one back before running git)"
    GITCONFIG_CHANGED=0
  fi
  # 1. Transcript checks that need the session's files on disk: T3 reads the scripts it ran,
  # which a stash would remove. T2 needs only the sealed transcripts, T5 only this one.
  out=$(guard transcript-check --transcript "$transcript" --checks T2,T3,T5 --transcripts-dir "$STATE/transcripts" \
          --manifest "$STATE/transcripts.sha256" --worktree "$GANDALF_WORKTREE" 2>&1)
  case $? in
    0) ;;
    1) add_reason "transcript" "$out" ;;
    *) transcript_unreadable "$out" "$before" ;;
  esac
  # 2. The .gitignore rules for .loop/ and data/. When a committed rule un-ignores them, the
  # tree is not stashed (stash_dirty refuses), because a stash would take the run data.
  out=$(guard ignore-check 2>&1) || add_reason "gitignore" "gitignore: $out"
  # 3. Half-done git operations: their commits and edits are saved, then they are aborted.
  clear_git_state post "$id"
  # 4. A session may leave the clone on another branch or a detached HEAD. Commits that are
  # not on main are kept on a branch, pushed as a backup, and named in the stub note.
  branch=$(g symbolic-ref -q --short HEAD 2>/dev/null)
  if [ "$branch" != main ]; then
    keep_stop_begin
    if [ -z "$branch" ] && ! g merge-base --is-ancestor HEAD main 2>/dev/null; then
      branch="loop-orphan-$id"
      g branch -f "$branch" HEAD >/dev/null 2>&1
    fi
    if [ -n "$branch" ] && ! g merge-base --is-ancestor "$branch" main 2>/dev/null; then
      if gnet push -q origin "$branch" >/dev/null 2>&1; then where="pushed to origin"; else where="local only"; fi
      RESCUE_NOTES="$RESCUE_NOTES; the session left commits on branch $branch ($where), not on main"
      note left_on_branch "$branch"
      say "the session left commits on branch $branch ($where)"
    fi
    stash_dirty "loop-stash $id (left on ${branch:-a detached HEAD})"
    if ! g checkout -q main >/dev/null 2>&1; then
      if ! g rev-parse -q --verify refs/heads/main >/dev/null && g checkout -q -b main origin/main >/dev/null 2>&1; then
        note recreated_main "from origin/main"
        RESCUE_NOTES="$RESCUE_NOTES; the session deleted the local main branch; the runner recreated it from origin/main"
      else
        add_reason "checkout main failed" "The session left the clone on ${branch:-a detached HEAD} and 'git checkout main' failed."
      fi
    fi
    keep_stop_end
  fi
  # 5. A committed STOP that the session deleted without a commit is put back, and the
  # deletion is a finding (only Anjor resumes the loop). A hard stop the session wrote but
  # did not commit is committed now, and honoured.
  if restore_deleted_stop; then
    note restored_stop deleted_in_tree
    add_reason "STOP deleted" "stop-deleted (uncommitted): $STUDY_REL/STOP is committed, but the session deleted it from the working tree or the index without a commit; the runner put it back. Only Anjor resumes the loop."
  fi
  commit_uncommitted_stop "$id"
  # 6. Uncommitted log lines and WAIT are committed (they are the record), the rest stashed.
  # The data link's skip-worktree bits first: a session that dropped them would otherwise
  # leave a deletion that no stash can take.
  ensure_data_link post
  save_log_changes "$id" "$before"
  if ! stash_dirty "loop-stash $id"; then
    add_reason "stash failed" "The working tree could not be stashed after the session, so no session may start on it: $STASH_LEFT"
  fi
  # 7. A stub entry when the session's entry is not closed, or its work was stashed or rescued,
  # so that the log says where the work is and the streak counts the iteration.
  own_outcome=$(guard outcome --iter "$id" 2>/dev/null | head -1)
  guard entry-closed --iter "$id" >/dev/null 2>&1
  ec=$?
  title=""
  if [ "$ec" -ne 0 ]; then
    title="runner stub (log entry not closed)"
  elif [ "$STASH_REF" != none ] || [ -n "$RESCUE_NOTES" ]; then
    title="runner stub (work left uncommitted)"
  fi
  if [ -n "$title" ]; then
    # A session Anjor ended with a signal is stubbed as 'interrupted', which the streak skips.
    stub_outcome=stub
    [ "$INTERRUPTED" = 1 ] && stub_outcome=interrupted
    if guard append-stub --iter "$id" --exit-code "$rc" --duration-sec "$dur" --before "$before" \
         --after "$(g rev-parse HEAD)" --stash "$STASH_REF" --note "$why$RESCUE_NOTES" --title "$title" \
         --outcome "$stub_outcome" >/dev/null \
       && g add -- "$STUDY_REL/log" \
       && g commit -q -m "Study 04 [$id]: $title" -- "$STUDY_REL/log" >/dev/null 2>&1; then
      note stub appended
      say "appended a stub entry for $id ($title)"
    else
      say "could not append a stub entry for $id"
      note stub failed
    fi
  fi
  OUTCOME=$(guard outcome --iter "$id" 2>/dev/null | head -1)
  # 8. Push everything. A failure is retried before the next session.
  sync_main post || say "push after the session failed ($PUSH_RESULT); kept in .loop/push_pending"
  AFTER=$(g rev-parse HEAD)
  # 9. Guards on what the session committed.
  if [ -n "$session_start" ]; then
    out=$(guard commit-guards --before "$before" --after HEAD --loop-committer "$LOOP_GIT_NAME" \
            --session-start "$session_start" 2>&1) || add_reason "commit guards" "$out"
  else
    out=$(guard commit-guards --before "$before" --after HEAD --loop-committer "$LOOP_GIT_NAME" 2>&1) \
      || add_reason "commit guards" "$out"
  fi
  out=$(guard frozen-check 2>&1) || add_reason "frozen sections" "$out"
  out=$(guard ledger-verify 2>&1) || add_reason "ledger" "$out"
  out=$(guard transcript-check --transcript "$transcript" --checks T1,T4 --before "$before" --after HEAD \
          --loop-committer "$LOOP_GIT_NAME" 2>&1)
  case $? in
    0) ;;
    1) add_reason "transcript checks" "$out" ;;
    *) transcript_unreadable "$out" "$before" ;;
  esac
  if [ "$own_outcome" = "hard stop" ] && ! stop_present; then
    add_reason "hard stop without STOP" "The entry of $id ends with 'outcome: hard stop', but the session wrote no STOP; the runner stops the loop in its place."
  fi
  if [ -n "$REASONS" ]; then
    DECISION="Each line above names a commit or a file and the rule it broke. Revert or approve each flagged change (look at the commits with git show), then resume."
  fi
  # GANDALF's own branches and tags, as recorded before the session, and origin's tags then
  # and now: a tag pushed with a refspec leaves no local tag behind.
  if [ -f "$STATE/gandalf-refs.pre.$id" ]; then
    if ! with_timeout 60 git -C "$GANDALF_WORKTREE" -c core.hooksPath=/dev/null -c core.fsmonitor=false \
         ls-remote --tags origin > "$STATE/gandalf-tags.origin-after.$id" 2>/dev/null; then
      rm -f "$STATE/gandalf-tags.origin-after.$id"
      note gandalf_origin_tags_after unread
    fi
    if [ ! -f "$STATE/gandalf-tags.origin.$id" ] || [ ! -f "$STATE/gandalf-tags.origin-after.$id" ]; then
      note gandalf_origin_tags_unchecked 1
      say "origin's tags were not compared before and after the session: a list could not be read"
    fi
    out=$(guard gandalf-refs-check --worktree "$GANDALF_WORKTREE" --before "$STATE/gandalf-refs.pre.$id" \
            --origin-tags "$STATE/gandalf-tags.origin.$id" --origin-tags-after "$STATE/gandalf-tags.origin-after.$id" 2>&1)
    case $? in
      0) ;;
      1)
        add_reason "gandalf refs" "$out"
        DECISION="${DECISION:+$DECISION }The GANDALF repo's branches outside study04/ and its tags are yours: the loop touches only study04/ branches and never tags. A branch you moved or made yourself in ~/repos/anjor/gandalf is not listed; a tag, or a branch you deleted, is, whoever changed it. Put back what the session changed (git reflog in ~/repos/anjor/gandalf shows old branch tips; check whether a new tag reached origin and started a release workflow), or approve the change if you made it yourself during the session, then resume."
        ;;
      *) note gandalf_refs_unchecked "$(printf '%s\n' "$out" | head -1)"; say "GANDALF's branches and tags not checked: $out" ;;
    esac
    rm -f "$STATE/gandalf-refs.pre.$id" "$STATE/gandalf-tags.origin.$id" "$STATE/gandalf-tags.origin-after.$id"
  fi
  out=$(guard streak --loop-committer "$LOOP_GIT_NAME" 2>&1)
  n=$(printf '%s\n' "$out" | head -1)
  case "$n" in
    ''|*[!0-9]*) add_reason "streak check" "streak: the WIP/stub count failed: $out"; n=0 ;;
  esac
  if [ "$n" -ge "$WIP_STREAK_LIMIT" ]; then
    add_reason "WIP streak" "streak: $n iterations in a row ended WIP or as a stub (limit $WIP_STREAK_LIMIT): $(printf '%s\n' "$out" | sed -n 2p)"
    DECISION="$DECISION The loop is not finishing its blocks. Read the last log entries, then change the approach or the next block, and resume."
  fi
  # 10. GPU work outside the launcher shows up as Modal apps the ledger does not know.
  APPS_PROBLEM=""
  apps_guard "after iteration $id"
  case $? in
    12) add_reason "modal apps" "$APPS_PROBLEM"; DECISION="$DECISION $APPS_DECISION" ;;
    14)
      note apps_unchecked "$APPS_ERROR"
      say "Modal app check postponed: $APPS_ERROR"
      if apps_overdue; then add_reason "modal apps unchecked" "$(apps_overdue_reason)"; DECISION="$DECISION $APPS_OVERDUE_DECISION"; fi
      ;;
  esac
  REASONS=$(printf '%s\n' "$REASONS" | sed '/^[[:space:]]*$/d')
  if [ -n "$REASONS" ]; then
    GUARDS=STOP
    write_stop "$SHORT" "$SHORT (runner checks after iteration $id):
$REASONS" "$DECISION"
  fi
  return 0
}

inflight_runner_alive() {  # the runner named in .loop/inflight is alive: hand its lock back; 0 if so
  local id before start pid lst
  read -r id before start pid lst < "$STATE/inflight"
  if [ -n "${pid:-}" ] && [ "$pid" != "$$" ] && alive "$pid" && [ "$(proc_lstart "$pid")" = "${lst:-}" ]; then
    # Its runner is alive although the lock looked stale: hand the lock back and keep out.
    say "the runner of iteration ${id:-?} (pid $pid) is still alive; not touching its iteration"
    echo "$pid" > "$LOCK/pid"
    echo "$lst" > "$LOCK/lstart"
    OWN_LOCK=0
    LOCK_HOLDER="$pid"
    return 0
  fi
  return 1
}

recover_dead_iteration() {  # a runner died mid-iteration: finish that iteration's bookkeeping now
  local dead_id dead_before dead_start dead_pid dead_lst win_start
  read -r dead_id dead_before dead_start dead_pid dead_lst < "$STATE/inflight"
  if inflight_runner_alive; then return 13; fi
  say "iteration ${dead_id:-?} has no finished bookkeeping (its runner died); recovering it"
  note recovered "${dead_id:-unknown}"
  # Its session window is checked for Modal apps like any other. Without a recorded start,
  # the minute in its iteration ID starts the window; without that, the window starts at 0,
  # so that every app the ledger does not know is flagged rather than none.
  win_start="${dead_start:-}"
  case "$win_start" in
    ''|*[!0-9]*)
      win_start=$(iter_start_epoch "${dead_id:-}") || win_start=0
      [ -n "$win_start" ] || win_start=0
      note window_start_guessed "$win_start"
      ;;
  esac
  echo "$win_start $(date +%s)" >> "$STATE/app_windows"
  if [ -z "${dead_id:-}" ] || [ -z "${dead_before:-}" ] || [ "$ANCHOR" != "$dead_before" ]; then
    # HEAD may hold that session's unchecked commits, so nothing from it runs (GUARD_PY is
    # empty): STOP is written by the runner alone.
    write_stop "dead iteration unverifiable" \
      "The runner died during iteration ${dead_id:-unknown} and its starting commit (${dead_before:-none}) is unknown, so its commits cannot be checked."
    rm -f "$STATE/inflight"
    return 0
  fi
  # The checks run from the loop files of the commit that iteration started from (the anchor).
  ITER_ID=$dead_id
  PHASE=post
  seal_transcript "$STATE/transcripts/$dead_id.jsonl"
  finish_iteration "$dead_id" "$dead_before" "unknown" "-1" \
    "the runner died before its post-session steps; recovered by the next runner" "$STATE/transcripts/$dead_id.jsonl" \
    "${dead_start:-}"
  rm -f "$STATE/inflight"
  note recovered_outcome "$OUTCOME"
  ITER_ID=""
  PHASE=pre
  return 0
}

sweep_dead_runners() {  # kill processes recorded by runners that died, then forget them
  local files f dead=0
  [ -f "$STATE/inflight" ] && dead=1
  files=$(ls "$STATE/tracked" 2>/dev/null)
  for f in $files; do
    dead=1
    sweep_tracked "$STATE/tracked/$f"
    [ "$SWEPT" -gt 0 ] && note killed_leftovers_of "$f:$SWEPT"
    rm -f "$STATE/tracked/$f"
  done
  if [ "$dead" = 1 ]; then
    cwd_sweep "left by a runner that died"
    [ "$SWEPT_CWD" -gt 0 ] && note killed_orphans "$SWEPT_CWD"
  fi
  return 0
}

push_only() {  # --push-only: commit a STOP left in the working tree and push main; 0 when in sync
  # Git only: nothing from HEAD's loop files and no binary from its config.env runs here.
  local rc on_origin=no local_stop=no
  PHASE=post
  GUARD_PY=""
  restore_deleted_stop && note restored_stop deleted_in_tree
  commit_uncommitted_stop "push-$(date -u +%Y%m%d-%H%M%S)"
  sync_main pre
  rc=$?
  if [ "$rc" -ne 0 ] && [ -e "$STUDY/STOP" ] && ! g cat-file -e "origin/main:$STUDY_REL/STOP" 2>/dev/null \
     && [ "$PUSH_RESULT" != fetch-failed ]; then
    publish_stop_fallback "local STOP"
  fi
  g cat-file -e "origin/main:$STUDY_REL/STOP" 2>/dev/null && on_origin=yes
  stop_present && local_stop=yes
  runlog push-only "push=$PUSH_RESULT stop_local=$local_stop stop_on_origin=$on_origin"
  PHASE=done
  if [ "$(g rev-parse HEAD 2>/dev/null)" = "$(g rev-parse origin/main 2>/dev/null)" ]; then return 0; fi
  return 1
}

# ---------------------------------------------------------------------------
# Traps
# ---------------------------------------------------------------------------

on_signal() {
  if [ "$PHASE" = pre ] && [ "$CRIT_DEPTH" -eq 0 ]; then
    say "signal before the session started: exiting"
    runlog abort-signal
    exit 130
  fi
  # During the session the watchdog kills the session tree; after it, and inside a change to
  # git state before it, the work finishes so that nothing is lost. Then the runner exits 130.
  INTERRUPTED=1
}

on_exit() {
  if [ "$PHASE" = session ] && [ -n "${CPID:-}" ] && kill -0 "$CPID" 2>/dev/null; then
    kill_session_tree
  fi
  # A runner that exits without its line (a shell error, say) still leaves one.
  runlog abort-unexpected "phase=$PHASE"
  release_lock
  [ -n "${TRUSTED:-}" ] && rm -rf "$TRUSTED"
}

# ---------------------------------------------------------------------------
# One iteration
# ---------------------------------------------------------------------------

main() {
  NOTES=""; ITER_ID=""; KILLED=0; SWEPT=0; SWEPT_CWD=0; PHASE=pre; INTERRUPTED=0; CRIT_DEPTH=0; TRUSTED=""
  CPID=""; RUNLOGGED=0; OWN_LOCK=0; STOP_WRITTEN=0; STASH_REF=none; STASH_LEFT=""; PUSH_RESULT=none
  GUARDS=ok; OUTCOME=""; AFTER=""; TRACKED_SEEN=""; TRACKED_FILE=""; APPS_ERROR=""; APPS_PROBLEM=""
  LOCK_HOLDER=""; GITDIR=""; GUARD_PY=""; GITCONFIG_SNAPSHOT=""; RESCUE_NOTES=""; KEPT_STOP=""; FF_ERR=""
  REASONS=""; SHORT=""; DECISION=""; CONFIG_ERROR=""; ANCHOR=HEAD; GITCONFIG_CHANGED=0; TRANSCRIPT_LOST=0
  local rc now today n n_log total total_log remaining reason started before prompt transcript errfile start
  local tick last_snap deadline timed_out session_rc duration why sc res klass backoff_until mode=run
  local clone_problems out rl_status rl_reset rl_type util5 util7 b2 repo_arg="" dead_id dead_before stopped=0

  while [ $# -gt 0 ]; do
    case "$1" in
      --push-only) mode=push ;;
      --repo) shift; repo_arg="${1:-}"; [ -n "$repo_arg" ] || { echo "usage: run_iteration.sh [--push-only] [--repo DIR]" >&2; return 2; } ;;
      --repo=*) repo_arg="${1#--repo=}" ;;
      *) echo "usage: run_iteration.sh [--push-only] [--repo DIR]" >&2; return 2 ;;
    esac
    shift
  done
  # --repo: run_forever.sh runs a copy of this script, taken from a commit it trusts, outside the clone.
  if [ -n "$repo_arg" ]; then
    REPO=$(cd "$repo_arg" && pwd -P) || return 2
  else
    LOOP_DIR=$(cd "$(dirname "$0")" && pwd -P) || return 2
    REPO=$(cd "$LOOP_DIR/../../.." && pwd -P) || return 2
  fi
  STUDY="$REPO/$STUDY_REL"
  STATE="$STUDY/.loop"
  RUNLOG="$STATE/runner.log"
  DETAIL="$STATE/runner-detail.log"
  LOCK="$STATE/lock"

  if [ "${S04_LOOP:-}" = 1 ]; then
    echo "run_iteration.sh: refusing to run inside a loop session (S04_LOOP=1)" >&2
    return 2
  fi
  export PATH="${HOME:-/tmp}/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH:/usr/sbin:/sbin"
  export GIT_TERMINAL_PROMPT=0
  # A replace ref planted in .git must not change what the runner and the guards read.
  export GIT_NO_REPLACE_OBJECTS=1
  # Only in the loop's own clone: the integrator sets this marker there and nowhere else.
  if [ "$(git -C "$REPO" config --local --get s04.loopclone 2>/dev/null)" != true ]; then
    echo "run_iteration.sh: $REPO is not the loop's clone ('git config s04.loopclone' is not 'true' there); refusing to run" >&2
    if [ -d "$STATE" ]; then echo "$(utc_now) - abort-not-loop-clone repo=$(clean "$REPO")" >> "$RUNLOG"; fi
    return 2
  fi
  mkdir -p "$STATE/transcripts" "$STATE/tracked" || { echo "cannot create $STATE" >&2; return 2; }
  trap on_exit EXIT
  trap on_signal INT TERM HUP
  TRUSTED=$(mktemp -d "${TMPDIR:-/tmp}/s04-trusted.XXXXXX") || { runlog abort-config error=mktemp; return 2; }

  # 1. One runner at a time. A lock whose process is gone is stale.
  if ! take_lock; then
    runlog skip-lock "holder=${LOCK_HOLDER:-unknown}"
    return 13
  fi
  cd "$REPO" || { runlog abort-config error=cd; return 2; }
  GITDIR=$(g rev-parse --absolute-git-dir 2>/dev/null) || { runlog abort-config error=no_git_dir; return 2; }

  # The trust anchor: the commit whose loop files and config.env this runner uses until it
  # has checked STOP. A runner that died left .loop/inflight: its session's commits are
  # unchecked, so the anchor is the commit that session started from, and the .git/config
  # it may have changed is put back before any other git command. Otherwise HEAD, which the
  # last iteration's checks passed unless STOP is there.
  if [ -f "$STATE/inflight" ]; then
    if inflight_runner_alive; then runlog skip-lock "holder=${LOCK_HOLDER:-unknown} reason=inflight_runner_alive"; return 13; fi
    read -r dead_id dead_before _ < "$STATE/inflight"
    if [ -n "${dead_id:-}" ] && [ -f "$STATE/git-config.pre.$dead_id" ]; then
      GITCONFIG_SNAPSHOT="$STATE/git-config.pre.$dead_id"
      check_git_config "$dead_id"
    fi
    if [ -n "${dead_before:-}" ] && g cat-file -e "${dead_before}^{commit}" 2>/dev/null; then
      ANCHOR=$dead_before
    else
      stopped=1  # its start is unknown: nothing from HEAD may run
    fi
  fi
  if ! make_trusted "$ANCHOR"; then runlog abort-config "error=no_loop_files_at_$(clean "$ANCHOR")"; return 2; fi
  if ! load_config "$TRUSTED/config.env"; then
    runlog abort-config "error=$(clean "$CONFIG_ERROR")"
    return 2
  fi
  ensure_identity
  ensure_excludes
  restore_deleted_stop && note restored_stop deleted_in_tree
  # While the loop is stopped, HEAD's loop files and the binaries its config.env names are
  # not run: a loop commit that changed them is what may have stopped it. Only git runs until
  # the sync shows that STOP is gone.
  if [ "$ANCHOR" = HEAD ] && stop_present; then stopped=1; fi
  if [ "$stopped" = 1 ]; then GUARD_PY=""; else resolve_guard_python; fi
  if [ "$mode" = push ]; then
    push_only
    return $?
  fi

  # 2. What a dead runner left: its processes, its iteration's bookkeeping.
  sweep_dead_runners
  if [ -f "$STATE/inflight" ]; then
    recover_dead_iteration
    rc=$?
    if [ "$rc" -eq 13 ]; then runlog skip-lock "holder=${LOCK_HOLDER:-unknown} reason=inflight_runner_alive"; return 13; fi
    if [ "$INTERRUPTED" = 1 ]; then runlog abort-signal; return 130; fi
    if stop_present; then GUARD_PY=""; fi
  fi

  # 3. The working tree: half-done operations, the branch, an uncommitted STOP, then a stash.
  clear_git_state pre "$(latest_iter)-pre"
  if [ "$(g symbolic-ref -q --short HEAD 2>/dev/null)" != main ]; then
    keep_stop_begin
    if ! g symbolic-ref -q HEAD >/dev/null 2>&1 && ! g merge-base --is-ancestor HEAD main 2>/dev/null; then
      g branch -f "loop-orphan-$(latest_iter)-pre" HEAD >/dev/null 2>&1
      gnet push -q origin "loop-orphan-$(latest_iter)-pre" >/dev/null 2>&1
      note left_on_branch "loop-orphan-$(latest_iter)-pre"
    fi
    stash_dirty "loop-stash $(latest_iter)-predirty (not on main)"
    if ! g checkout -q main >/dev/null 2>&1; then
      keep_stop_end
      say "not on main and 'git checkout main' failed"
      stop_write_plain "the loop clone is not on main and git checkout main failed."
      publish_stop_fallback "checkout main failed"
      runlog skip-stop reason=checkout_main_failed
      return 12
    fi
    keep_stop_end
  fi
  restore_deleted_stop && note restored_stop deleted_in_tree
  commit_uncommitted_stop "$(latest_iter)-pre"
  # Files that change what a session loads are noted before the stash can take them (only
  # while the guards may run; a stopped loop is checked again once STOP is gone).
  clone_problems=""
  [ -n "$GUARD_PY" ] && clone_problems=$(guard clone-check 2>&1)
  if [ -n "$(dirty_status)" ]; then
    # Only a runner that died after its session leaves a dirty tree; name the stash after
    # the newest session.
    if ! stash_dirty "loop-stash $(latest_iter)-predirty"; then
      # No session starts on a tree that cannot be stashed. Already stopped: say nothing new.
      if ! stop_present; then
        write_stop "dirty tree" "The working tree of the loop clone could not be stashed before a session: $STASH_LEFT" \
          "Look at the loop clone (git status there), save what matters, clean it, then resume."
      fi
      runlog skip-stop reason=dirty_tree
      return 12
    fi
  fi

  # 4. Sync with origin: Anjor's input arrives only as pushed commits.
  [ -f "$STATE/push_pending" ] && note push_retry "$(cut -d' ' -f2- "$STATE/push_pending")"
  sync_main pre
  rc=$?
  if [ "$rc" -eq 3 ]; then runlog skip-stop reason=diverged; return 12; fi
  if [ "$rc" -ne 0 ] && stop_present; then
    # A local STOP stops the loop whether or not it reached origin; the push is retried later.
    runlog skip-stop "reason=stop_not_synced push=$PUSH_RESULT"
    return 12
  fi
  if [ "$rc" -eq 4 ]; then
    if [ -f "$GITDIR/index.lock" ]; then runlog abort-git-busy "push=$PUSH_RESULT"; return 2; fi
    write_stop "fast-forward failed" "Local main cannot be fast-forwarded to origin/main: $FF_ERR" \
      "Look at the loop clone (git status, git log there), fix it, then resume."
    runlog skip-stop reason=ff_failed
    return 12
  fi
  if [ "$rc" -ne 0 ]; then
    runlog abort-network "push=$PUSH_RESULT"
    return 2
  fi

  # 5. STOP, before anything from HEAD runs: committed, or only in the working tree.
  restore_deleted_stop && note restored_stop deleted_in_tree
  if stop_present; then
    runlog skip-stop "reason=$(clean "$(sed -n 's/^Reason: //p' "$STUDY/STOP" 2>/dev/null | head -1)")"
    return 12
  fi
  # No STOP: HEAD is what the last iteration's checks passed, or what Anjor resumed from.
  if ! make_trusted HEAD || ! load_config "$TRUSTED/config.env"; then
    write_stop "config.env unusable" "loop/config.env (or another loop file) at HEAD does not load: ${CONFIG_ERROR:-missing loop files}"
    runlog skip-stop "reason=config"
    return 12
  fi
  resolve_guard_python
  ensure_identity

  # 6. The clone: local settings and instructions, git hooks, the .gitignore rules.
  out=$(guard clone-check 2>&1)
  clone_problems=$(printf '%s\n%s\n' "$clone_problems" "$out" | sed '/^[[:space:]]*$/d' | sort -u)
  out=$(guard ignore-check 2>&1) || clone_problems=$(printf '%s\n%s\n' "$clone_problems" "$out" | sed '/^[[:space:]]*$/d')
  if [ -n "$clone_problems" ]; then
    write_stop "loop clone unsafe" "The loop clone holds files that change what a session loads or what git runs, or has lost a .gitignore rule:
$clone_problems" "Remove those files from the loop clone (they are not committed; check how they got there), restore any .gitignore rule, then resume."
    runlog skip-stop reason=clone
    return 12
  fi

  # 7. Setup that cannot fix itself: a hard stop rather than a silent retry every pause.
  reason=""
  [ -d "$GANDALF_WORKTREE" ] || reason="the GANDALF worktree $GANDALF_WORKTREE does not exist."
  if ! guard check-settings --file "$TRUSTED/settings.json" > "$STATE/settings.check" 2>&1; then
    reason="$reason loop/settings.json is not usable (claude -p would silently ignore it): $(head -1 "$STATE/settings.check")"
  fi
  if ! grep -q '{{ITER_ID}}' "$TRUSTED/prompt.md" 2>/dev/null; then
    reason="$reason loop/prompt.md is missing or has no {{ITER_ID}} placeholder."
  fi
  if [ -n "$reason" ]; then
    write_stop "loop setup broken" "The runner cannot start a session:$reason" "Fix the setup in your checkout, push, then resume."
    runlog skip-stop reason=setup
    return 12
  fi

  # 8. Apps created during an earlier session that could not be checked then. No session
  # starts until they are checked; a check that stays impossible for too long is a STOP.
  if [ -s "$STATE/app_windows" ]; then
    apps_guard "pending check of an earlier session"
    case $? in
      12) write_stop "modal apps" "$APPS_PROBLEM" "$APPS_DECISION"; runlog skip-stop reason=modal_apps; return 12 ;;
      14)
        if apps_overdue; then
          write_stop "modal apps unchecked" "$(apps_overdue_reason)" "$APPS_OVERDUE_DECISION"
          runlog skip-stop "reason=modal_apps_unchecked error=$(clean "$APPS_ERROR")"
          return 12
        fi
        echo $(( $(date +%s) + FAST_FAIL_BACKOFF_SEC )) > "$STATE/backoff_until"
        runlog skip-auth "reason=$(clean "$APPS_ERROR") pending_windows=$(wc -l < "$STATE/app_windows" | tr -d ' ')"
        return 14
        ;;
    esac
  fi

  # 9. WAIT, set by the agent with wait.sh while runs or CI are in flight.
  remaining=$(guard wait-remaining 2>/dev/null)
  reason=$(printf '%s\n' "$remaining" | sed -n 2p)
  remaining=$(printf '%s\n' "$remaining" | head -1)
  case "$remaining" in ''|*[!0-9]*) remaining=0 ;; esac
  if [ "$remaining" -gt 0 ]; then
    runlog skip-wait "remaining=${remaining}s reason=$(clean "$reason")"
    return 10
  fi
  [ -n "$reason" ] && note wait_note "$reason"

  # 10. Backoff after a fast failure or a usage limit.
  now=$(date +%s)
  backoff_until=$(cat "$STATE/backoff_until" 2>/dev/null)
  case "$backoff_until" in ''|*[!0-9]*) backoff_until=0 ;; esac
  if [ "$backoff_until" -gt "$now" ]; then
    runlog skip-backoff "until=$(date -u -r "$backoff_until" +%Y-%m-%dT%H:%M:%SZ)"
    return 10
  fi

  # 11. Iteration caps. A session started is a session counted, whatever its outcome. The
  # transcripts are not committed, so the log's entries count too, whichever is larger.
  today=$(date -u +%Y%m%d)
  n=$(ls "$STATE/transcripts" 2>/dev/null | grep -c "^it-$today-.*\.jsonl$")
  total=$(ls "$STATE/transcripts" 2>/dev/null | grep -c '^it-.*\.jsonl$')
  out=$(guard iter-count --day "$today" 2>/dev/null)
  n_log=$(printf '%s\n' "$out" | sed -n 1p)
  total_log=$(printf '%s\n' "$out" | sed -n 2p)
  case "$n_log" in ''|*[!0-9]*) n_log=0 ;; esac
  case "$total_log" in ''|*[!0-9]*) total_log=0 ;; esac
  [ "$n_log" -gt "$n" ] && n=$n_log
  [ "$total_log" -gt "$total" ] && total=$total_log
  if [ "$n" -ge "$DAILY_ITERATION_CAP" ]; then
    runlog skip-daily-cap "today=$n cap=$DAILY_ITERATION_CAP"
    return 11
  fi
  if [ "$total" -ge "$TOTAL_ITERATION_CAP" ]; then
    write_stop "total iteration cap" \
      "Total iteration cap reached: $total sessions, TOTAL_ITERATION_CAP=$TOTAL_ITERATION_CAP." \
      "Decide whether the study may continue. To continue, raise TOTAL_ITERATION_CAP in loop/config.env, push, then resume."
    runlog skip-stop "reason=total_cap total=$total"
    return 12
  fi

  # 12. gh and modal must be logged in. Only the active github.com account counts: a broken
  # login on another host must not stall the loop.
  if ! with_timeout 60 "$GH_BIN" auth status --hostname github.com --active > "$STATE/gh.out" 2>&1; then
    echo $(( $(date +%s) + FAST_FAIL_BACKOFF_SEC )) > "$STATE/backoff_until"
    runlog skip-auth "reason=gh_auth_status_failed"
    return 14
  fi
  # 13. Running Modal apps against the ledger.
  apps_guard "before a session"
  case $? in
    12) write_stop "modal apps" "$APPS_PROBLEM" "$APPS_DECISION"; runlog skip-stop reason=modal_apps; return 12 ;;
    14) echo $(( $(date +%s) + FAST_FAIL_BACKOFF_SEC )) > "$STATE/backoff_until"
        runlog skip-auth "reason=$(clean "$APPS_ERROR")"; return 14 ;;
  esac

  # 14. The session.
  if [ -n "${S04_TEST_ITER_ID:-}" ]; then
    ITER_ID="$S04_TEST_ITER_ID"
    if [ -e "$STATE/transcripts/$ITER_ID.jsonl" ]; then runlog abort-config error=test_iter_id_reused; return 2; fi
  else
    ITER_ID=$(date -u +it-%Y%m%d-%H%M)
    if [ -e "$STATE/transcripts/$ITER_ID.jsonl" ]; then
      sleep $(( 61 - 10#$(date -u +%S) ))
      ITER_ID=$(date -u +it-%Y%m%d-%H%M)
      if [ -e "$STATE/transcripts/$ITER_ID.jsonl" ]; then runlog abort-config error=iteration_id_taken; return 2; fi
    fi
  fi
  started=$(utc_now)
  before=$(g rev-parse HEAD)
  gandalf_refs_record "$ITER_ID"
  STASH_NOTE=$(stash_note)
  prompt=$(guard render-prompt --template "$TRUSTED/prompt.md" --iter "$ITER_ID" --started "$started" \
             --worktree "$GANDALF_WORKTREE" --stash-note "$STASH_NOTE") \
    || { runlog abort-config error=render_prompt; return 2; }
  transcript="$STATE/transcripts/$ITER_ID.jsonl"
  errfile="$STATE/transcripts/$ITER_ID.stderr"
  TRACKED_FILE="$STATE/tracked/$ITER_ID"
  set -- --model "$CLAUDE_MODEL" --effort "$CLAUDE_EFFORT" \
    --permission-mode acceptEdits --permission-prompts none \
    --settings "$TRUSTED/settings.json" --setting-sources project \
    --strict-mcp-config --disable-slash-commands --no-chrome \
    --add-dir "$GANDALF_WORKTREE" --output-format stream-json --verbose
  if [ -n "$MAX_BUDGET_USD" ]; then set -- "$@" --max-budget-usd "$MAX_BUDGET_USD"; fi
  # The session gets the clone's .git/config as it is now; a change to it is undone after,
  # by this runner or, if it dies, by the next one (the snapshot lives in .loop/).
  GITCONFIG_SNAPSHOT="$STATE/git-config.pre.$ITER_ID"
  cp "$GITDIR/config" "$GITCONFIG_SNAPSHOT" || GITCONFIG_SNAPSHOT=""

  if [ "$INTERRUPTED" = 1 ]; then ITER_ID=""; runlog abort-signal; return 130; fi
  start=$(date +%s)
  echo "$ITER_ID $before $start $$ $(proc_lstart "$$")" > "$STATE/inflight"
  say "iteration $ITER_ID: starting the session at $(echo "$before" | cut -c1-12)"
  PHASE=session
  (
    unset CLAUDECODE CLAUDE_CODE_ENTRYPOINT
    S04_ITER_ID="$ITER_ID"
    S04_LOOP=1
    CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
    ENABLE_CLAUDEAI_MCP_SERVERS=false
    CLAUDE_CODE_DISABLE_ARTIFACT=1
    export S04_ITER_ID S04_LOOP CLAUDE_CODE_DISABLE_AUTO_MEMORY ENABLE_CLAUDEAI_MCP_SERVERS CLAUDE_CODE_DISABLE_ARTIFACT
    # setsid: the session leads its own process group, so the whole tree can be killed. The
    # signals a background job and Python start with ignored are put back to their defaults,
    # so that tool commands see a normal SIGPIPE, SIGINT, SIGQUIT and SIGXFSZ.
    exec "$GUARD_PY" -I -S -c 'import os, signal, sys
for s in (signal.SIGPIPE, signal.SIGXFSZ, signal.SIGINT, signal.SIGQUIT):
    signal.signal(s, signal.SIG_DFL)
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])' "$CLAUDE_BIN" -p "$prompt" "$@"
  ) < /dev/null > "$transcript" 2> "$errfile" &
  CPID=$!
  echo "$CPID $(proc_lstart "$CPID")" > "$TRACKED_FILE"
  TRACKED_SEEN="|$(cat "$TRACKED_FILE")|"

  # 15. Watchdog: snapshot the session's processes, kill the tree at the time limit.
  tick=1
  [ "$WATCHDOG_POLL_SEC" -lt 5 ] && tick=0.2
  deadline=$((start + SESSION_TIME_LIMIT_SEC))
  timed_out=0
  last_snap=0
  while kill -0 "$CPID" 2>/dev/null; do
    now=$(date +%s)
    if [ $((now - last_snap)) -ge "$WATCHDOG_POLL_SEC" ]; then track_snapshot; last_snap=$now; fi
    if [ "$INTERRUPTED" = 1 ]; then
      say "interrupted: killing the session"
      kill_session_tree
      break
    fi
    if [ "$now" -ge "$deadline" ]; then
      timed_out=1
      say "time limit of ${SESSION_TIME_LIMIT_SEC}s reached: killing the session"
      kill_session_tree
      break
    fi
    sleep "$tick"
  done
  wait "$CPID" 2>/dev/null
  session_rc=$?
  duration=$(( $(date +%s) - start ))
  PHASE=post
  # Left-over processes: anything still in the session's group, anything tracked, and any
  # process without a terminal still working in the clone or the worktree.
  reason=$(session_procs "$CPID" | awk '{print $1}' | tr '\n' ' ')
  if [ -n "$reason" ]; then
    say "killing processes left in the session's process group: $reason"
    # shellcheck disable=SC2086
    kill_pids $reason
    # shellcheck disable=SC2086
    KILLED=$((KILLED + $(set -- $reason; echo $#)))
  fi
  sweep_tracked "$TRACKED_FILE"
  KILLED=$((KILLED + SWEPT))
  cwd_sweep "after the session"
  KILLED=$((KILLED + SWEPT_CWD))
  rm -f "$TRACKED_FILE"
  echo "$start $(date +%s)" >> "$STATE/app_windows"
  seal_transcript "$transcript"
  if [ "$timed_out" = 1 ]; then
    why="timeout: killed after ${duration}s (limit ${SESSION_TIME_LIMIT_SEC}s)"
  elif [ "$INTERRUPTED" = 1 ]; then
    why="interrupted: the runner got a signal and killed the session"
  else
    why="session exited with code $session_rc"
  fi
  say "session ended: $why"

  # 16-21. Bookkeeping: transcript checks, rescue, STOP, stash, stub, push, guards, streak, apps.
  finish_iteration "$ITER_ID" "$before" "$session_rc" "$duration" "$why" "$transcript" "$start"
  rm -f "$STATE/inflight"

  # 22. Backoff. A usage limit (a rate_limit_event whose status is not 'allowed') waits until
  # its reset time plus a margin, whatever the session's length. A session that failed fast
  # (auth error, overload) waits FAST_FAIL_BACKOFF_SEC.
  sc=$(guard session-check --transcript "$transcript" --stderr "$errfile" 2>/dev/null)
  kv() { printf '%s\n' "$sc" | sed -n "s/^$1=//p" | head -1; }
  res=$(kv result); klass=$(kv class); rl_status=$(kv rl_status); rl_reset=$(kv rl_resets_at)
  rl_type=$(kv rl_type); util5=$(kv util_5h); util7=$(kv util_7d)
  backoff_until=none
  now=$(date +%s)
  case "${rl_status:-none}" in
    none|allowed*) ;;
    *)
      backoff_until=$((now + FAST_FAIL_BACKOFF_SEC))
      case "$rl_reset" in
        ''|*[!0-9]*) ;;
        *) b2=$((rl_reset + RATE_LIMIT_MARGIN_SEC)); [ "$b2" -gt "$backoff_until" ] && backoff_until=$b2 ;;
      esac
      echo "$backoff_until" > "$STATE/backoff_until"
      note rate_limited "${rl_status}${rl_type:+/$rl_type}"
      say "usage limit reported (status $rl_status${rl_type:+, $rl_type}${rl_reset:+, resets $(date -u -r "$rl_reset" +%Y-%m-%dT%H:%M:%SZ)}): backing off until $(date -u -r "$backoff_until" +%Y-%m-%dT%H:%M:%SZ)"
      ;;
  esac
  if [ "$backoff_until" = none ] && [ "$INTERRUPTED" = 0 ] && [ "$timed_out" = 0 ] && [ "$duration" -lt "$FAST_FAIL_SEC" ]; then
    if [ "$session_rc" -ne 0 ] || [ "$res" != ok ]; then
      backoff_until=$(( $(date +%s) + FAST_FAIL_BACKOFF_SEC ))
      echo "$backoff_until" > "$STATE/backoff_until"
      note fast_fail "${klass:-unknown}"
      say "fast failure (${klass:-unknown}): backing off until $(date -u -r "$backoff_until" +%Y-%m-%dT%H:%M:%SZ)"
    fi
  fi
  [ -n "$util5" ] && note util_5h "$util5"
  [ -n "$util7" ] && note util_7d "$util7"

  # 23. The runner.log line.
  runlog ran "exit=$session_rc duration=${duration}s outcome=$(clean "${OUTCOME:-unknown}") before=$(echo "$before" | cut -c1-12) after=$(echo "$AFTER" | cut -c1-12) stash=$(clean "$STASH_REF") guards=$GUARDS backoff=$backoff_until killed=$KILLED timeout=$timed_out push=$PUSH_RESULT interrupted=$INTERRUPTED"
  PHASE=done

  # 24. Exit code.
  if [ "$INTERRUPTED" = 1 ]; then return 130; fi
  if stop_present; then return 12; fi
  return 0
}

main "$@"; exit $?
