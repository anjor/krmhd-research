#!/usr/bin/env bash
# Runs every check script in this folder with uv from the repository root and writes the stdout and stderr of each
# script to out/<script>.out.  c01 runs twice: with its defaults, and as a strong-flow run (out/<script>_strong.out)
# that shows the resolution limit of the pointwise test.  c02 runs at N = 16, 24, 32, 48.  The script continues past
# failures, prints a summary (local start time, wall time and exit status of each run) and exits with status 1 if any
# run failed or a script of the folder is not in the list.  c05 imports ../../dynamics/dcore.py.
set -u

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../../../../.." && pwd)"  # repository root
rel="${here#"$root"/}"                       # this folder, relative to the root
mkdir -p "$here/out"
cd "$root" || exit 1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# one entry per run: "<name of the .out file>|<script>|<arguments>"
runs=(
  "c01_lagrangian_pointwise|c01_lagrangian_pointwise|"
  "c01_lagrangian_pointwise_strong|c01_lagrangian_pointwise|32,48,64 1.5 0.25,0.5 strong"
  "c02_forcing_same_instant|c02_forcing_same_instant|16,24,32,48"
  "c04_tilt_landau|c04_tilt_landau|"
  "c05_imex|c05_imex|"
  "c06_algebra|c06_algebra|"
)

failed=0
for f in "$here"/c[0-9]*.py; do
  s="$(basename "$f" .py)"
  case "$s" in *_core) continue ;; esac
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
  line="$(printf '%s  %-32s  started %s  wall %5d s  exit %d' "$result" "$name" "$started" "$wall" "$status")"
  echo "$line"
  lines+=("$line")
done
total=$(( $(date +%s) - t_all ))

echo
echo "Summary for $rel (output in $rel/out/)"
for l in "${lines[@]}"; do echo "  $l"; done
echo "  ${#runs[@]} runs, $failed failed, total wall ${total} s"
[ "$failed" -eq 0 ] || exit 1
