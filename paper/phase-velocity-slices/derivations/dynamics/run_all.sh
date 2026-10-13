#!/usr/bin/env bash
# Runs every check script in this folder with uv from the repository root and writes the stdout and stderr of each
# script to out/<script>.out.  It continues past failures, prints a summary (local start time, wall time and exit
# status of each script) and exits with status 1 if any script failed or a script of the folder is not in the list.
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../.." && pwd)"     # repository root
rel="${here#"$root"/}"                       # this folder, relative to the root
mkdir -p "$here/out"
cd "$root" || exit 1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# one entry per run: "<name of the .out file>|<script>|<arguments>"
runs=(
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

failed=0
for f in "$here"/d[0-9]*.py; do
  s="$(basename "$f" .py)"
  case " ${runs[*]} " in
    *"|$s|"*) ;;
    *) echo "WARNING: $s.py is not in the run list of run_all.sh"; failed=$((failed + 1)) ;;
  esac
done

lines=()
t_all=$(date +%s)
for entry in "${runs[@]}"; do
  IFS='|' read -r name script args <<< "$entry"
  started=$(date +%Y%m%d_%H%M%S)
  t0=$(date +%s)
  # shellcheck disable=SC2086
  uv run python "$rel/$script.py" $args > "$here/out/$name.out" 2>&1
  status=$?
  wall=$(( $(date +%s) - t0 ))
  if [ "$status" -eq 0 ]; then result=PASS; else result=FAIL; failed=$((failed + 1)); fi
  line="$(printf '%s  %-28s  started %s  wall %5d s  exit %d' "$result" "$name" "$started" "$wall" "$status")"
  echo "$line"
  lines+=("$line")
done
total=$(( $(date +%s) - t_all ))

echo
echo "Summary for $rel (output in $rel/out/)"
for l in "${lines[@]}"; do echo "  $l"; done
echo "  ${#runs[@]} runs, $failed failed, total wall ${total} s"
[ "$failed" -eq 0 ] || exit 1
