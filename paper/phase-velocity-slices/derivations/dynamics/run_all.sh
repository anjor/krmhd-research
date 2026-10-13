#!/usr/bin/env bash
# Run every check script d??*.py in this folder from the repository root with `uv run python` (dcore.py is the shared
# helper module, not a check). Each script's stdout and stderr go to out/<script>.out (with the repository path
# shortened to a relative one). The runner continues past failures, prints a summary (also saved in out/run_all.out)
# and exits 1 if any script failed or a check script of the folder is missing from the run list below.
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
  "d00_sympy_identities|d00_sympy_identities|"
  "d01_gandalf_conventions|d01_gandalf_conventions|"
  "d02_frozen_line|d02_frozen_line|"
  "d03_frozen_gandalf_step|d03_frozen_gandalf_step|"
  "d04_forcing_superposition|d04_forcing_superposition|"
  "d05_breakers|d05_breakers|"
  "d06_tilt_kz_decomposition|d06_tilt_kz_decomposition|"
  "d07_two_slice_law|d07_two_slice_law|"
  "d08_no_fjortoft|d08_no_fjortoft|"
  "d09_mirror|d09_mirror|"
  "d09b_parity_lemma_gandalf|d09b_parity_lemma_gandalf|"
  "d10_elsasser_slices|d10_elsasser_slices|"
)

status=0
T0=$(date +%s)
log "$REL: $(date '+%Y-%m-%d %H:%M:%S'), $(uv run python --version 2>&1)"
for path in "$HERE"/d[0-9][0-9]*.py; do
  s="$(basename "$path" .py)"
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
