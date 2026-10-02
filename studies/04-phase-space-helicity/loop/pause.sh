#!/bin/bash
# pause.sh [reason...]: pause the Study 04 loop. For Anjor, in his own checkout.
#
# Commits and pushes a STOP file (and a Blocking line in QUESTIONS.md). The runner pulls
# before every iteration and does not start a session while STOP exists. A session already
# running finishes its block first; to end it at once, stop run_forever.sh on the Mac
# (Ctrl-C), which kills the session and still saves its work.
#
# Uses your own git identity. Resume with loop/resume.sh.

set -u

LOOP_DIR=$(cd "$(dirname "$0")" && pwd -P) || exit 2
REPO=$(cd "$LOOP_DIR/../../.." && pwd -P) || exit 2
STUDY_REL="studies/04-phase-space-helicity"
STUDY="$REPO/$STUDY_REL"
PYTHON_BIN=""
if [ -f "$LOOP_DIR/config.env" ]; then
  # shellcheck disable=SC1091
  . "$LOOP_DIR/config.env"
fi
PYTHON_BIN="${S04_PYTHON_BIN:-${PYTHON_BIN:-$REPO/.venv/bin/python}}"
case "$PYTHON_BIN" in "~/"*) PYTHON_BIN="$HOME/${PYTHON_BIN#\~/}" ;; esac

cd "$REPO" || exit 2
if [ "$(git symbolic-ref -q --short HEAD)" != main ]; then
  echo "pause.sh: check out main first" >&2
  exit 2
fi
if ! git pull -q --ff-only origin main; then
  echo "pause.sh: 'git pull --ff-only origin main' failed; sort that out, then pause" >&2
  exit 2
fi
if [ -e "$STUDY/STOP" ]; then
  echo "pause.sh: the loop is already stopped. STOP says:"
  cat "$STUDY/STOP"
  exit 1
fi

REASON="Paused by Anjor: ${*:-no reason given}"
PATHS="$STUDY_REL/STOP"
if [ -f "$STUDY/QUESTIONS.md" ] && git diff --quiet -- "$STUDY_REL/QUESTIONS.md" \
   && git diff --cached --quiet -- "$STUDY_REL/QUESTIONS.md"; then
  "$PYTHON_BIN" "$LOOP_DIR/guards.py" --repo "$REPO" stop-write --source anjor \
    --reason "$REASON" --decision "Resume with loop/resume.sh when ready." || exit 1
  PATHS="$PATHS $STUDY_REL/QUESTIONS.md"
else
  [ -f "$STUDY/QUESTIONS.md" ] && echo "note: QUESTIONS.md has uncommitted changes, so only STOP is written"
  "$PYTHON_BIN" "$LOOP_DIR/guards.py" --repo "$REPO" stop-write --source anjor --no-questions \
    --reason "$REASON" --decision "Resume with loop/resume.sh when ready." || exit 1
fi
# shellcheck disable=SC2086
git add -- $PATHS
# shellcheck disable=SC2086
git commit -q -m "Study 04: pause loop" -- $PATHS || { echo "pause.sh: commit failed" >&2; exit 1; }
if ! git push -q origin main; then
  echo "pause.sh: push failed. The STOP commit is only local: push it, or the loop will not see it." >&2
  exit 1
fi
echo "Paused: STOP is pushed. The runner will not start another session."
echo "A session already running finishes its block; Ctrl-C on run_forever.sh ends it now."
