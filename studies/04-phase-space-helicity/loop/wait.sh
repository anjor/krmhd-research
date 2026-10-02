#!/bin/bash
# wait.sh <minutes> <reason...>: set the time before the loop's next iteration.
#
# The agent uses it while Modal runs or GANDALF CI are in flight, so that iterations do not
# start just to find nothing to do. It writes studies/04-phase-space-helicity/WAIT; the
# runner skips iterations until the time has passed. 0 minutes removes WAIT. At most 2880
# minutes (48 hours): a longer wait would let the loop go quiet for days unnoticed.
#
# It does not commit. The agent commits WAIT with its close-out, because the runner starts
# from the pushed state and a stash would otherwise swallow the file.

set -u

usage() {
  echo "usage: wait.sh <minutes 0..2880> <reason...>   (0 removes the wait)" >&2
  exit 2
}

LOOP_DIR=$(cd "$(dirname "$0")" && pwd -P) || exit 2
STUDY=$(cd "$LOOP_DIR/.." && pwd -P) || exit 2
WAIT_FILE="$STUDY/WAIT"

[ $# -ge 1 ] || usage
MINUTES="$1"
shift
case "$MINUTES" in ''|*[!0-9]*) usage ;; esac
MINUTES=$((10#$MINUTES))
if [ "$MINUTES" -gt 2880 ]; then
  echo "wait.sh: $MINUTES minutes is more than the limit of 2880 (48 hours)" >&2
  exit 2
fi

if [ "$MINUTES" -eq 0 ]; then
  rm -f "$WAIT_FILE"
  echo "WAIT removed: the next iteration may start at once."
  exit 0
fi

REASON=$(printf '%s ' "$@" | tr '\n\r\t' '   ' | sed 's/  */ /g; s/^ //; s/ $//')
[ -n "$REASON" ] || { echo "wait.sh: give a reason" >&2; usage; }

NOW=$(date -u +%s)
UNTIL=$((NOW + MINUTES * 60))
{
  echo "until_epoch=$UNTIL"
  echo "until_utc=$(date -u -r "$UNTIL" +%Y-%m-%dT%H:%M:%SZ)"
  echo "reason=$REASON"
  echo "set_by=${S04_ITER_ID:--}"
  echo "set_at_utc=$(date -u -r "$NOW" +%Y-%m-%dT%H:%M:%SZ)"
} > "$WAIT_FILE"
cat "$WAIT_FILE"
echo "Commit $(basename "$STUDY")/WAIT with your close-out."
