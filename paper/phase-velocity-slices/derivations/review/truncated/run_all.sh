#!/bin/sh
# Run every check script of this folder (c<nn>_*.py) from the repository root with "uv run python".
# c00_cmm_numbering.py reads CMM's downloaded LaTeX source (untrusted data) and runs in isolated mode (python -I).
# Each script's stdout and stderr go to out/<script>.out; the repository root is stripped from paths printed in
# warnings, so the files do not depend on where the repository lives. The run continues past failures, prints one
# summary line per script (local start time, exit status, wall time, last line of the script) and exits with status 1
# if any script failed.
HERE=$(cd "$(dirname "$0")" && pwd -P) || exit 1
ROOT=$(cd "$HERE/../../../../.." && pwd -P) || exit 1
REL=${HERE#"$ROOT"/}
OUT="$HERE/out"
mkdir -p "$OUT" || exit 1
cd "$ROOT" || exit 1

status=0
failed=""
t_begin=$(date +%s)
for path in "$HERE"/c[0-9][0-9]_*.py; do
    s=$(basename "$path" .py)
    case "$s" in
        c00_*) iso="-I" ;;
        *) iso="" ;;
    esac
    start=$(date +%Y%m%d_%H%M%S)
    t0=$(date +%s)
    uv run python $iso "$REL/$s.py" > "$OUT/$s.out.tmp" 2>&1
    rc=$?
    t1=$(date +%s)
    awk -v root="$ROOT/" '{ while ((i = index($0, root)) > 0) $0 = substr($0, 1, i - 1) substr($0, i + length(root)); print }' \
        "$OUT/$s.out.tmp" > "$OUT/$s.out"
    rm -f "$OUT/$s.out.tmp"
    echo "$s: start $start, exit $rc, $((t1 - t0)) s :: $(grep '^== ' "$OUT/$s.out" | tail -n 1)"
    if [ "$rc" -ne 0 ]; then
        status=1
        failed="$failed $s"
    fi
done
echo "total wall time $(($(date +%s) - t_begin)) s"
if [ "$status" -ne 0 ]; then
    echo "FAILED:$failed"
else
    echo "all scripts passed"
fi
exit "$status"
