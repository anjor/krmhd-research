#!/usr/bin/env bash
# Run every independent check c??_*.py in this folder from the repository root with `uv run python` (c00_core.py is
# the shared helper module, not a check; c05_imex.py also imports ../../dynamics/dcore.py). Each run's stdout and
# stderr go to out/<run>.out (with the repository path shortened to a relative one). c01 runs twice: with its defaults,
# and as a strong-flow run (out/c01_lagrangian_pointwise_strong.out) that shows the resolution limit of the pointwise
# test; c02 runs at N = 16, 24, 32, 48. The runner continues past failures, prints a summary (also saved in
# out/run_all.out) and exits 1 if any run failed or a check script of the folder is missing from the run list below.
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

# one entry per run, in this order: "<name of the .out file>|<script>|<arguments>"
RUNS=(
  "c01_lagrangian_pointwise|c01_lagrangian_pointwise|"
  "c01_lagrangian_pointwise_strong|c01_lagrangian_pointwise|32,48,64 1.5 0.25,0.5 strong"
  "c02_forcing_same_instant|c02_forcing_same_instant|16,24,32,48"
  "c04_tilt_landau|c04_tilt_landau|"
  "c05_imex|c05_imex|"
  "c06_algebra|c06_algebra|"
)

status=0
T0=$(date +%s)
log "$REL: $(date '+%Y-%m-%d %H:%M:%S'), $(uv run python --version 2>&1)"
for path in "$HERE"/c[0-9][0-9]_*.py; do
  s="$(basename "$path" .py)"
  [ "$s" = "c00_core" ] && continue
  case " ${RUNS[*]} " in
    *"|$s|"*) ;;
    *) log "$s: FAIL (not in the run list of run_all.sh)"; status=1 ;;
  esac
done
for entry in "${RUNS[@]}"; do
  IFS='|' read -r name s args <<< "$entry"
  start=$(date +%Y%m%d_%H%M%S)
  t0=$(date +%s)
  # shellcheck disable=SC2086
  uv run python -B -u "$REL/$s.py" $args 2>&1 | sed "s|$ROOT/||g" > "$OUT/$name.out"
  rc=${PIPESTATUS[0]}
  t1=$(date +%s)
  npass=$(grep -c '\[PASS\]' "$OUT/$name.out")
  nfail=$(grep -c '\[FAIL\]' "$OUT/$name.out")
  if [ "$rc" -ne 0 ] || [ "$nfail" -ne 0 ]; then
    res=FAIL; status=1
  else
    res=PASS
  fi
  log "$name: $res (exit $rc; checks passed $npass, failed $nfail), started $start, wall $((t1 - t0)) s"
done
log "overall: $([ $status -eq 0 ] && echo PASS || echo FAIL), total wall $(($(date +%s) - T0)) s"
exit $status
