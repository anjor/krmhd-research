#!/usr/bin/env bash
# Run every independent check c??_*.py in this folder from the repository root with `uv run python`. Each script's
# stdout and stderr go to out/<script>.out (with the repository path shortened to a relative one). The runner continues
# past failures, prints a summary (also saved in out/run_all.out) and exits 1 if any script failed.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$HERE"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/pyproject.toml" ]; do ROOT="$(dirname "$ROOT")"; done
[ -f "$ROOT/pyproject.toml" ] || { echo "repository root (pyproject.toml) not found above $HERE"; exit 1; }
command -v uv > /dev/null || { echo "uv not found"; exit 1; }
REL="${HERE#"$ROOT"/}"
OUT="$HERE/out"
mkdir -p "$OUT"
cd "$ROOT" || exit 1
SUMMARY="$OUT/run_all.out"
: > "$SUMMARY"
log() { echo "$*" | tee -a "$SUMMARY"; }

status=0
T0=$(date +%s)
log "$REL: $(date '+%Y-%m-%d %H:%M:%S'), $(uv run python --version 2>&1)"
for path in "$HERE"/c[0-9][0-9]_*.py; do
  s="$(basename "$path" .py)"
  start=$(date +%Y%m%d_%H%M%S)
  t0=$(date +%s)
  uv run python -B -u "$REL/$s.py" 2>&1 | sed "s|$ROOT/||g" > "$OUT/$s.out"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  npass=$(grep -c '\[PASS\]' "$OUT/$s.out")
  nfail=$(grep -c '\[FAIL\]' "$OUT/$s.out")
  if [ "$rc" -ne 0 ] || [ "$nfail" -ne 0 ]; then
    res=FAIL; status=1
  elif grep -q '^SKIPPED' "$OUT/$s.out"; then
    res=SKIPPED
  else
    res=PASS
  fi
  log "$s: $res (exit $rc; checks passed $npass, failed $nfail), started $start, wall $((t1 - t0)) s"
done
log "overall: $([ $status -eq 0 ] && echo PASS || echo FAIL), total wall $(($(date +%s) - T0)) s"
exit $status
