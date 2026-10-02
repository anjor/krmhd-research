#!/bin/bash
# run_forever.sh: run Study 04 loop iterations until STOP appears.
#
# Start it in the loop's clone, in a terminal you leave open (or under tmux):
#     studies/04-phase-space-helicity/loop/run_forever.sh
# It refuses to run unless 'git config s04.loopclone' is 'true' in the clone (the loop never
# runs in Anjor's own checkout). It keeps the Mac awake with caffeinate while it lives, runs
# run_iteration.sh, pauses PAUSE_BETWEEN_ITERATIONS_SEC, and repeats.
#
# It stops starting sessions as soon as STOP exists, in the working tree or committed at HEAD.
# Before it exits it makes sure the STOP is on origin, so that Anjor sees it: if the push
# failed (the network was down at the end of the session), it retries with 'run_iteration.sh
# --push-only' after every pause until the STOP is on origin. If STOP disappears meanwhile
# (Anjor resumed), it carries on.
#
# The runner it starts is a copy of run_iteration.sh taken from a commit it trusts, never the
# file in the working tree: HEAD while there is no STOP (the last iteration's checks passed),
# the start commit of an iteration whose runner died, and otherwise the copy it already has
# (at start, the newest version that the loop did not commit). So a loop commit that changes
# the runner and stops the loop never runs, not even to publish that STOP. A change Anjor
# pushes to run_iteration.sh takes effect from the iteration after the one that pulls it.
#
# Ctrl-C (or kill -TERM / -HUP on this script) ends it cleanly: the running iteration kills
# its session, saves and pushes the work, runs its checks, and only then does this exit.

set -u

STUDY_REL="studies/04-phase-space-helicity"
RUNNER_REL="$STUDY_REL/loop/run_iteration.sh"

sleep_pause() {  # sleep_pause SECONDS: sleep, but wake up for a signal
  local n=0
  while [ "$n" -lt "$1" ] && [ "$STOPPING" = 0 ]; do
    sleep 1
    n=$((n + 1))
  done
}

stop_here() {  # STOP in the working tree, or committed at HEAD
  [ -e "$STUDY/STOP" ] || git -C "$REPO" cat-file -e "HEAD:$STUDY_REL/STOP" 2>/dev/null
}

trusted_rev() {  # the commit to take the runner from now, or nothing (keep the copy we have)
  local before
  if [ -f "$STUDY/.loop/inflight" ]; then
    before=$(awk '{print $2; exit}' "$STUDY/.loop/inflight" 2>/dev/null)
    if [ -n "$before" ] && git -C "$REPO" cat-file -e "${before}^{commit}" 2>/dev/null; then echo "$before"; fi
    return 0
  fi
  stop_here || git -C "$REPO" rev-parse -q --verify HEAD 2>/dev/null
}

first_rev() {  # at start, with STOP present: the newest commit of the runner that the loop did not commit
  local name
  name=$(git -C "$REPO" show "HEAD:$STUDY_REL/loop/config.env" 2>/dev/null \
           | sed -n 's/^[[:space:]]*LOOP_GIT_NAME=[[:space:]]*\([^[:space:]#]*\).*/\1/p' | tail -1)
  git -C "$REPO" log --format='%H%x09%cn' -- "$RUNNER_REL" 2>/dev/null \
    | awk -F'\t' -v a="${name:-krmhd-loop}" '$2 != a && $2 != "krmhd-loop" { print $1; exit }'
}

pin_runner() {  # pin_runner [start]: refresh PIN from a trusted commit; 0 if PIN is usable
  local rev
  rev=$(trusted_rev)
  if [ -z "$rev" ] && [ "${1:-}" = start ]; then rev=$(first_rev); fi
  if [ -z "$rev" ] && [ "${1:-}" = start ]; then rev=$(git -C "$REPO" rev-parse -q --verify HEAD 2>/dev/null); fi
  if [ -n "$rev" ] && git -C "$REPO" show "$rev:$RUNNER_REL" > "$PIN.new" 2>/dev/null && [ -s "$PIN.new" ]; then
    mv "$PIN.new" "$PIN"
    PIN_REV=$rev
  fi
  rm -f "$PIN.new"
  [ -s "$PIN" ]
}

run_child() {  # run_child ARGS...: the pinned runner in the background, waited for; sets RC
  bash "$PIN" --repo "$REPO" "$@" &
  CHILD=$!
  wait "$CHILD"
  RC=$?
  while kill -0 "$CHILD" 2>/dev/null; do
    wait "$CHILD"
    RC=$?
  done
  CHILD=""
}

stop_on_origin() { git -C "$REPO" cat-file -e "origin/main:$STUDY_REL/STOP" 2>/dev/null; }

publish_stop() {  # STOP is present: exit only once it is on origin. 0 published, 1 lifted, 130 signal
  local tries=0
  while :; do
    if stop_on_origin; then
      echo "run_forever: STOP is present; exiting."
      if [ -f "$STUDY/.loop/push_pending" ]; then
        echo "run_forever: note: some loop commits are still only local ($(cut -d' ' -f2- "$STUDY/.loop/push_pending")); the next runner pushes them."
      fi
      if [ -f "$STUDY/STOP" ]; then sed -n '1,8p' "$STUDY/STOP"; else git -C "$REPO" show "origin/main:$STUDY_REL/STOP" | sed -n '1,8p'; fi
      return 0
    fi
    if [ "$tries" -gt 0 ] && ! stop_here; then
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run_forever: STOP is gone (Anjor resumed); carrying on."
      return 1
    fi
    tries=$((tries + 1))
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run_forever: STOP is only local so far (its push failed); retrying in ${PAUSE}s (attempt $tries)"
    sleep_pause "$PAUSE"
    [ "$STOPPING" = 1 ] && return 130
    run_child --push-only
    [ "$STOPPING" = 1 ] && return 130
  done
}

main() {
  local line
  LOOP_DIR=$(cd "$(dirname "$0")" && pwd -P) || return 2
  STUDY=$(cd "$LOOP_DIR/.." && pwd -P) || return 2
  REPO=$(cd "$LOOP_DIR/../../.." && pwd -P) || return 2
  if [ "$(git -C "$REPO" config --local --get s04.loopclone 2>/dev/null)" != true ]; then
    echo "run_forever.sh: $REPO is not the loop's clone ('git config s04.loopclone' is not 'true' there); refusing to run" >&2
    return 2
  fi
  PAUSE=""
  if [ -f "$LOOP_DIR/config.env" ]; then
    PAUSE=$(sed -n 's/^[[:space:]]*PAUSE_BETWEEN_ITERATIONS_SEC=\([0-9][0-9]*\).*/\1/p' "$LOOP_DIR/config.env" | tail -1)
  fi
  PAUSE="${S04_PAUSE_BETWEEN_ITERATIONS_SEC:-${PAUSE:-300}}"
  case "$PAUSE" in ''|*[!0-9]*) PAUSE=300 ;; esac

  STOPPING=0
  CHILD=""
  RC=0
  PIN_REV=""
  PIN_DIR=$(mktemp -d "${TMPDIR:-/tmp}/s04-forever.XXXXXX") || return 2
  PIN="$PIN_DIR/run_iteration.sh"
  trap 'rm -rf "$PIN_DIR"' EXIT
  trap 'STOPPING=1; [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null' INT TERM HUP
  if ! pin_runner start; then
    echo "run_forever.sh: cannot take run_iteration.sh from a commit (git show failed); refusing to run" >&2
    return 2
  fi

  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -i -s -w $$ >/dev/null 2>&1 &
  else
    echo "run_forever.sh: caffeinate not found; the Mac may sleep" >&2
  fi

  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run_forever: started (pause ${PAUSE}s, runner from ${PIN_REV:0:12}); Ctrl-C to stop after the current iteration"
  while :; do
    # The iteration runs in the background so that a signal reaches this script at once and
    # is forwarded; a background child ignores SIGINT, so TERM is what it gets.
    pin_runner
    run_child
    line=$(tail -1 "$STUDY/.loop/runner.log" 2>/dev/null)
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) run_forever: iteration exit $RC | $line"
    if [ "$RC" = 2 ] && printf '%s\n' "$line" | grep -q ' abort-not-loop-clone'; then
      return 2
    fi
    if [ "$RC" = 12 ] || stop_here; then
      if [ "$STOPPING" = 1 ]; then
        echo "run_forever: signal received; STOP is present; exiting."
        return 130
      fi
      publish_stop
      case $? in
        0) return 0 ;;
        130) echo "run_forever: signal received; exiting."; return 130 ;;
      esac
    fi
    if [ "$STOPPING" = 1 ]; then
      echo "run_forever: signal received; the iteration has finished its bookkeeping; exiting."
      return 130
    fi
    sleep_pause "$PAUSE"
    if [ "$STOPPING" = 1 ]; then
      echo "run_forever: signal received; exiting."
      return 130
    fi
  done
}

main "$@"; exit $?
