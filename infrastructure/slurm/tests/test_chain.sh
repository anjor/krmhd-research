#!/bin/bash
# test_chain.sh: local test of the cluster-side scripts (cluster_submit.sh, job.sh) and of submit.sh --dry-run, with
# stand-ins for sbatch and squeue and the real run code on the laptop's CPU.  A small scheduler loop runs the queued
# jobs one at a time, as Slurm would with one GPU.  The run is the test config of scripts/test_resume.py
# (forced_beta1.yaml at 32^3, M = 8, from rest, 0.12 tau_A; about a minute per job, mostly JIT compilation).
#
#   A  pause and resubmission: --pause-after 0 makes job 1 pause after one step (exit 3); job.sh resubmits; job 2
#      continues in place with --resume and finishes: one run folder, final .json written, resume.npz removed,
#      two 'submit' and two 'end' lines in the ledger.
#   B  MAX_RESUBMITS_PER_RUN = 0: job 1 pauses, the resubmission is refused, the run stays paused (resume.npz kept).
#   C  MAX_CONCURRENT_JOBS = 1 with another p1-* job queued: refused.
#   D  GPU_HOURS_PER_WEEK = 0.1 and a 20 min time limit: refused.
#   E  a malformed time limit and a time limit under 10 minutes: refused.
#   F  submit.sh --dry-run on a committed config with a repo-relative init.path lists the code, the config and the
#      restart file and contacts nothing; a config with an absolute init.path is refused.
#
# Usage (from the repository root): bash infrastructure/slurm/tests/test_chain.sh      Exit status 1 on failure.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
SLURM=$REPO/infrastructure/slurm
T=$(mktemp -d "${TMPDIR:-/tmp}/slurm_chain_test.XXXXXX")
trap 'rm -rf "$T"' EXIT
fails=0
pass() { echo "  PASS: $*"; }
fail() { echo "  FAIL: $*"; fails=$((fails + 1)); }
check() {  # check DESCRIPTION COMMAND...
  local d=$1
  shift
  if "$@"; then pass "$d"; else fail "$d"; fi
}

# ---- stand-ins for sbatch and squeue
mkdir -p "$T/bin"
cat > "$T/bin/sbatch" <<'EOF'
#!/bin/bash
# sbatch stand-in: records the job (name, output, exported variables, script) in $FAKE_SLURM/queue/<id>.
set -u
id=$(($(cat "$FAKE_SLURM/next_id" 2>/dev/null || echo 1000) + 1))
echo "$id" > "$FAKE_SLURM/next_id"
name="" output="" export_="" script=""
while [ $# -gt 0 ]; do
  case $1 in
    --parsable) ;;
    --job-name) name=$2; shift ;;
    --output) output=$2; shift ;;
    --export) export_=$2; shift ;;
    --time|--dependency) echo "$1 $2" >> "$FAKE_SLURM/sbatch_opts.$id"; shift ;;
    --*=*) echo "$1" >> "$FAKE_SLURM/sbatch_opts.$id" ;;
    -*) echo "$1 $2" >> "$FAKE_SLURM/sbatch_opts.$id"; shift ;;
    *) script=$1 ;;
  esac
  shift
done
mkdir -p "$FAKE_SLURM/queue"
{
  echo "name=$name"
  echo "output=$output"
  echo "script=$script"
  echo "$export_" | tr ',' '\n' | grep -v '^ALL$' | sed 's/^/env=/'
} > "$FAKE_SLURM/queue/$id"
echo "$id"
EOF
cat > "$T/bin/squeue" <<'EOF'
#!/bin/bash
# squeue stand-in: '<id> <name>' for the queued and running jobs and the jobs listed in $FAKE_SLURM/foreign.
for f in "$FAKE_SLURM"/queue/* "$FAKE_SLURM"/running/*; do
  [ -f "$f" ] && echo "$(basename "$f") $(sed -n 's/^name=//p' "$f")"
done
[ -f "$FAKE_SLURM/foreign" ] && cat "$FAKE_SLURM/foreign"
exit 0
EOF
chmod +x "$T/bin/sbatch" "$T/bin/squeue"

run_queue() {  # run the queued jobs in order of ID until none is left (at most 5)
  local i f id name output script vars
  mkdir -p "$FAKE_SLURM/running" "$FAKE_SLURM/done"
  for i in 1 2 3 4 5; do
    f=$(ls "$FAKE_SLURM"/queue/* 2>/dev/null | sort -t/ -k1 | head -n 1)
    [ -n "$f" ] || return 0
    id=$(basename "$f")
    mv "$f" "$FAKE_SLURM/running/$id"
    f=$FAKE_SLURM/running/$id
    name=$(sed -n 's/^name=//p' "$f")
    output=$(sed -n 's/^output=//p' "$f" | sed "s/%x/$name/g; s/%j/$id/g")
    script=$(sed -n 's/^script=//p' "$f")
    vars=()
    while IFS= read -r line; do vars+=("$line"); done < <(sed -n 's/^env=//p' "$f")
    (cd "$T" && env ${vars[@]+"${vars[@]}"} SLURM_JOB_ID="$id" bash "$script" > "$output" 2>&1)
    echo "$?" > "$FAKE_SLURM/done/$id.rc"
    mv "$f" "$FAKE_SLURM/done/$id"
  done
  echo "  (scheduler stopped after 5 jobs)"
}

new_site() {  # new_site NAME: a fresh "cluster" (REMOTE_ROOT with the repo and the project's environment) and its site file
  local root=$T/$1
  mkdir -p "$root/logs" "$root/fake_slurm"
  ln -s "$REPO" "$root/repo"
  ln -s "$REPO/.venv" "$root/venv"
  cat > "$root/site.env" <<EOF
SSH_HOST=nowhere.invalid
REMOTE_ROOT=$root
REMOTE_RESULTS=$root/results
LOCAL_RESULTS=$root/local_results
LOCAL_LOGS=$root/local_logs
MODULES=""
PYTHON=python3
JAX_CUDA=cuda12
SBATCH_OPTS="--gres=gpu:1 --cpus-per-task=2"
JOB_TIME=00:20:00
JAX_PLATFORMS=cpu
REQUIRE_GPU=0
EOF
  cp "$SLURM/limits.conf" "$root/limits.conf"
  echo "$root"
}

set_limit() {  # set_limit ROOT KEY VALUE
  sed "s/^$2=[^ ]*/$2=$3/" "$1/limits.conf" > "$1/limits.conf.new" && mv "$1/limits.conf.new" "$1/limits.conf"
}

submit() {  # submit ROOT ARGS...: cluster_submit.sh as submit.sh runs it on the login node
  local root=$1
  shift
  env SITE_ENV="$root/site.env" LIMITS_FILE="$root/limits.conf" FAKE_SLURM="$root/fake_slurm" PATH="$T/bin:$PATH" \
    bash "$SLURM/cluster_submit.sh" "$@"
}

run_jobs() {  # run_jobs ROOT: let the stand-in scheduler run the queue (resubmissions from job.sh go to the same queue)
  FAKE_SLURM="$1/fake_slurm" PATH="$T/bin:$PATH" LIMITS_FILE="$1/limits.conf" run_queue
}

# The test config (label chain-test)
(cd "$REPO" && uv run --quiet python - "$T/chain_test.yaml" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "paper/phase-velocity-slices/scripts")
import test_resume
p = Path(sys.argv[1])
test_resume.make_config(p)
p.write_text(p.read_text().replace("label: resume-test\n", "label: chain-test\n", 1))
EOF
) || { echo "FAIL: could not write the test config"; exit 1; }
CFG=$T/chain_test.yaml

echo "A: pause after one step, resubmission, resume to the end"
R=$(new_site A)
out=$(submit "$R" "$CFG" --pause-after 0)
echo "$out" | sed 's/^/    /'
check "first job submitted" grep -q "submitted job 1001" <<< "$out"
run_jobs "$R"
runs=("$R"/results/P1_chain-test_*)
check "one run folder" test ${#runs[@]} -eq 1
check "final .json written" test -f "${runs[0]}/$(basename "${runs[0]}").json"
check "resume.npz removed" test ! -f "${runs[0]}/resume.npz"
check "job 1 paused after one step" grep -q "paused at t = .*, step 1," "$R/logs/p1-chain-test-1001.out"
check "job 1 resubmitted job 1002 after itself" grep -q "submitted job 1002: .* resubmission 1, after job 1001" "$R/logs/p1-chain-test-1001.out"
check "job 2 waits for job 1 (--dependency afterany:1001)" grep -q "^--dependency afterany:1001$" "$R/fake_slurm/sbatch_opts.1002"
check "job 2 resumed in place" grep -q "resuming in place" "$R/logs/p1-chain-test-1002.out"
check "job 2 finished" grep -q "=== run finished" "$R/logs/p1-chain-test-1002.out"
check "two jobs ran, both exit 0 for Slurm" test "$(cat "$R"/fake_slurm/done/*.rc | tr -d '\n')" = "00"
check "ledger: 2 submit and 2 end lines" test "$(grep -c '^submit' "$R/ledger.tsv")$(grep -c '^end' "$R/ledger.tsv")" = "22"
check "ledger: end statuses 3 then 0" test "$(awk '$1 == "end" { printf "%s", $5 }' "$R/ledger.tsv")" = "30"

echo "B: resubmission limit 0"
R=$(new_site B)
set_limit "$R" MAX_RESUBMITS_PER_RUN 0
submit "$R" "$CFG" --pause-after 0 > /dev/null
run_jobs "$R"
runs=("$R"/results/P1_chain-test_*)
check "only one job ran" test "$(ls "$R"/fake_slurm/done/*.rc | wc -l | tr -d ' ')" = 1
check "resubmission refused" grep -q "REFUSED: resubmission 1 of chain-test would exceed MAX_RESUBMITS_PER_RUN = 0" "$R/logs/p1-chain-test-1001.out"
check "run left paused (resume.npz kept, no .json)" test -f "${runs[0]}/resume.npz" -a ! -f "${runs[0]}/$(basename "${runs[0]}").json"

echo "C: one job at a time"
R=$(new_site C)
echo "999 p1-other-run" > "$R/fake_slurm/foreign"
out=$(submit "$R" "$CFG")
st=$?
check "refused with status 2" test "$st" = 2
check "refusal names the queued job" grep -q "REFUSED: 1 job(s) named p1-\* already queued or running" <<< "$out"
echo "999 someone-elses-job" > "$R/fake_slurm/foreign"
out=$(submit "$R" "$CFG")
check "jobs not named p1-* do not count" grep -q "submitted job" <<< "$out"

echo "D: weekly GPU hours"
R=$(new_site D)
set_limit "$R" GPU_HOURS_PER_WEEK 0.1
out=$(submit "$R" "$CFG")
check "refused: 20 min job over a 0.1 h budget" grep -q "REFUSED: GPU hours of the past 7 days (0.000) plus this job's time limit" <<< "$out"
set_limit "$R" GPU_HOURS_PER_WEEK 0.5
out=$(submit "$R" "$CFG")
check "submitted under a 0.5 h budget" grep -q "submitted job 1001" <<< "$out"
rm -f "$R"/fake_slurm/queue/*
out=$(submit "$R" "$CFG")
check "second 20 min job refused (0.333 booked + 0.333 > 0.5)" grep -q "REFUSED: GPU hours of the past 7 days (0.333)" <<< "$out"

echo "E: time limits"
R=$(new_site E)
out=$(submit "$R" "$CFG" --time 20m 2>&1)
check "malformed time limit refused" grep -q "use HH:MM:SS or D-HH:MM:SS" <<< "$out"
out=$(submit "$R" "$CFG" --time 00:05:00)
check "time limit under 10 minutes refused" grep -q "REFUSED: time limit 00:05:00 is under 10 minutes" <<< "$out"
check "nothing queued" test -z "$(ls "$R/fake_slurm/queue" 2>/dev/null)"

echo "F: submit.sh --dry-run"
R=$(new_site F)
out=$(cd "$REPO" && SITE_ENV="$R/site.env" bash "$SLURM/submit.sh" --dry-run --time 00:30:00 \
  paper/phase-velocity-slices/configs/forced_beta0p3_n128.yaml 2>&1)
st=$?
echo "$out" | sed 's/^/    /'
if [ -z "$(cd "$REPO" && git status --porcelain -- infrastructure/slurm paper/phase-velocity-slices/scripts \
  paper/phase-velocity-slices/configs/forced_beta0p3_n128.yaml)" ]; then
  check "dry run exits 0 (everything committed)" test "$st" = 0
else
  check "dry run refuses while files are uncommitted" grep -qE "not committed|uncommitted changes" <<< "$out"
fi
check "lists the run code" grep -q "paper/phase-velocity-slices/scripts/p1lib.py" <<< "$out"
check "lists the restart file of init.path" grep -q "P1_forced-spinup64_20261015_110112/restart.npz" <<< "$out"
check "passes the options to cluster_submit.sh" grep -q "cluster_submit.sh paper/phase-velocity-slices/configs/forced_beta0p3_n128.yaml --time 00:30:00" <<< "$out"
out=$(cd "$REPO" && SITE_ENV="$R/site.env" bash "$SLURM/submit.sh" --dry-run \
  paper/phase-velocity-slices/configs/forced_beta0p3_n256.yaml 2>&1)
check "absolute init.path refused" grep -q "init.path in .* is absolute" <<< "$out"

echo "G: end to end: submit.sh, the job chain, fetch.sh (an ssh stand-in runs the commands here)"
cat > "$T/bin/ssh" <<'EOF'
#!/bin/bash
# ssh stand-in: drop the options and the host name, run the command locally (rsync's server side included)
while [ $# -gt 0 ]; do
  case $1 in
    -o|-p|-i|-l|-F|-J) shift 2 ;;
    -*) shift ;;
    *) break ;;
  esac
done
shift
exec bash -c "$*"
EOF
chmod +x "$T/bin/ssh"
R=$(new_site G)
rm "$R/repo"                          # the symlink only: here submit.sh makes the partial copy
cp "$R/site.env" "$T/G_site.env"
G_CFG=paper/phase-velocity-slices/configs/forced_test32.yaml
if [ -n "$(cd "$REPO" && git status --porcelain -- infrastructure/slurm paper/phase-velocity-slices/scripts "$G_CFG")" ]; then
  echo "  (skipped: commit infrastructure/slurm first; submit.sh sends committed files only)"
else
  out=$(cd "$REPO" && env SITE_ENV="$T/G_site.env" FAKE_SLURM="$R/fake_slurm" PATH="$T/bin:$PATH" \
    bash "$SLURM/submit.sh" --pause-after 0 "$G_CFG" 2>&1)
  echo "$out" | tail -n 1 | sed 's/^/    /'
  check "submitted through ssh" grep -q "submitted job 1001: p1-forced-test32" <<< "$out"
  check "only the code, the config and the scripts were copied (11 files)" \
    test "$(find "$R/repo" -type f | wc -l | tr -d ' ')" = 11
  check "COMMIT is HEAD" test "$(cat "$R/repo/COMMIT")" = "$(cd "$REPO" && git rev-parse HEAD)"
  FAKE_SLURM="$R/fake_slurm" PATH="$T/bin:$PATH" run_queue
  check "the chain ran two jobs" test "$(ls "$R"/fake_slurm/done/*.rc | wc -l | tr -d ' ')" = 2
  out=$(cd "$REPO" && env SITE_ENV="$T/G_site.env" FAKE_SLURM="$R/fake_slurm" PATH="$T/bin:$PATH" \
    bash "$SLURM/fetch.sh" 2>&1)
  echo "$out" | tail -n 1 | sed 's/^/    /'
  runs=("$R"/local_results/P1_forced-test32_*)
  check "fetch.sh copied the finished run" test -f "${runs[0]}/$(basename "${runs[0]}").json"
  check "fetch.sh copied the logs and the ledger" test -f "$R/local_logs/p1-forced-test32-1002.out" -a -f "$R/local_logs/ledger.tsv"
fi

echo
if [ "$fails" = 0 ]; then echo "PASS"; else echo "FAIL ($fails)"; exit 1; fi
