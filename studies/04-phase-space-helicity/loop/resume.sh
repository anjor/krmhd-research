#!/bin/bash
# resume.sh: resume the Study 04 loop. For Anjor, in his own checkout.
#
# Shows STOP, then commits and pushes its removal. Answer the question behind the stop first:
# put an ANSWER: (or VETO:) line in QUESTIONS.md and push it, before or with this. The loop
# reads only pushed commits.

set -u

LOOP_DIR=$(cd "$(dirname "$0")" && pwd -P) || exit 2
REPO=$(cd "$LOOP_DIR/../../.." && pwd -P) || exit 2
STUDY_REL="studies/04-phase-space-helicity"
STUDY="$REPO/$STUDY_REL"

cd "$REPO" || exit 2
if [ "$(git symbolic-ref -q --short HEAD)" != main ]; then
  echo "resume.sh: check out main first" >&2
  exit 2
fi
if ! git pull -q --ff-only origin main; then
  echo "resume.sh: 'git pull --ff-only origin main' failed; sort that out, then resume" >&2
  exit 2
fi
if [ ! -e "$STUDY/STOP" ]; then
  echo "resume.sh: the loop is not stopped (no $STUDY_REL/STOP)."
  exit 1
fi

echo "STOP says:"
echo "----------------------------------------------------------------------"
cat "$STUDY/STOP"
echo "----------------------------------------------------------------------"
git rm -q -- "$STUDY_REL/STOP" || exit 1
git commit -q -m "Study 04: resume loop (remove STOP)" -- "$STUDY_REL/STOP" || { echo "resume.sh: commit failed" >&2; exit 1; }
if ! git push -q origin main; then
  echo "resume.sh: push failed. The commit is only local: push it, or the loop stays stopped." >&2
  exit 1
fi
echo "Resumed: the next iteration starts at the runner's next attempt."
echo "Answers go in QUESTIONS.md as lines that start with ANSWER: (vetoes: VETO:), in a pushed commit."
echo "If a guard stopped the loop, revert or approve the flagged commits before or with this resume."
echo "Resuming accepts HEAD as it is: a loop change to loop/ that you leave in place runs from the next iteration."
echo "If STOP names files in the loop's clone (.claude/settings.local.json, untracked .claude/ files, an untracked"
echo "CLAUDE.md or CLAUDE.local.md anywhere, data/ included, git hooks, push URLs or programs in .git/config, a tree"
echo "that could not be stashed), fix them in that clone by hand: the runner checks again before every session and"
echo "stops the loop again while they are there. If run_forever.sh has exited, start it again in the clone."
