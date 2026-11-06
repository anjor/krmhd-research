#!/bin/bash
# job.sh: one Slurm job of a run of paper/phase-velocity-slices/scripts/run_forced_turbulence.py.  Only
# cluster_submit.sh submits it.  Slurm runs a copy of this file, so everything comes in the environment: SITE_ENV,
# P1S_HERE (this folder), P1S_CONFIG, P1S_RESUME, P1S_RESUBMIT, P1S_TIME and P1S_PAUSE_AFTER.
#
# It loads the site's modules and the environment that setup_env.sh built, prints the GPU and the versions, and stops
# at once unless the job sees exactly one GPU (CUDA_VISIBLE_DEVICES set by Slurm and one device in JAX; REQUIRE_GPU=0
# in the site file turns this off for tests on a CPU).  It runs the config with --max-wall-hours a little under the
# time limit (5% of it, at least 5 minutes and at most 1 hour, left for writing resume.npz), so that a run longer than
# one job pauses cleanly with exit status 3.  It continues the run in place (--resume) when the job is a resubmission, was asked to, or was
# requeued by Slurm.  It appends an 'end' line (time, job ID, hours used, exit status) to the ledger, and on exit status
# 3 calls cluster_submit.sh --resubmit N+1, which checks the limits again.  The results stay in REMOTE_RESULTS until
# fetch.sh on the laptop copies them back.
set -uo pipefail
t0=$(date +%s)
: "${P1S_HERE:?job.sh is submitted by cluster_submit.sh}"
. "$P1S_HERE/common.sh"
p1s_load_site "$SITE_ENV"
cd "$REMOTE_ROOT/repo" || p1s_die "no folder $REMOTE_ROOT/repo"
p1s_load_modules
[ -f "$REMOTE_ROOT/venv/bin/activate" ] || p1s_die "no environment at $REMOTE_ROOT/venv (run setup_env.sh)"
set +u
. "$REMOTE_ROOT/venv/bin/activate"
set -u
export JAX_PLATFORMS=${JAX_PLATFORMS:-cuda}

echo "=== job ${SLURM_JOB_ID:-?}, $(date -u '+%Y-%m-%d %H:%M:%S') UTC: $P1S_CONFIG, resubmission $P1S_RESUBMIT," \
  "time limit $P1S_TIME, Slurm restarts ${SLURM_RESTART_COUNT:-0}"
[ ! -f COMMIT ] || echo "code from commit $(cat COMMIT)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/GPU: /'
fi
# The job must see exactly its own GPU: JAX takes memory on every GPU it can see, which would interfere with other
# users' jobs on the node.  Slurm limits a job to its GPUs through CUDA_VISIBLE_DEVICES.
check=0
if [ "${REQUIRE_GPU:-1}" = 1 ] && [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES is not set, so the job is not limited to its GPU; not running (check the GPU request in SBATCH_OPTS)"
  check=5
fi
if [ "$check" = 0 ]; then
  XLA_PYTHON_CLIENT_PREALLOCATE=false python - <<'EOF'
import os
import sys

import jax

devs = jax.devices()
print(f"python {sys.version.split()[0]}, jax {jax.__version__}, backend {jax.default_backend()}, devices {devs}")
if os.environ.get("REQUIRE_GPU", "1") == "1":
    if jax.default_backend() != "gpu":
        sys.exit("no GPU visible to JAX: not running on the CPU")
    if len(devs) != 1:
        sys.exit(f"{len(devs)} GPUs visible to JAX, expected the job's one GPU: not running")
EOF
  check=$?
fi

rc=$check
if [ "$check" = 0 ]; then
  T_h=$(p1s_hours "$P1S_TIME") || exit 1
  first=0
  if [ "$P1S_RESUBMIT" = 0 ] && [ "${SLURM_RESTART_COUNT:-0}" = 0 ]; then first=1; fi
  if [ "$P1S_PAUSE_AFTER" != none ] && [ "$first" = 1 ]; then
    wall=$P1S_PAUSE_AFTER
  else
    wall=$(awk -v T="$T_h" 'BEGIN { m = 0.05 * T; if (m < 1 / 12) m = 1 / 12; if (m > 1) m = 1; printf "%.4f", T - m }')
  fi
  args=("$P1S_CONFIG" --base "$REMOTE_RESULTS" --max-wall-hours "$wall")
  if [ "$P1S_RESUME" = 1 ] || [ "$first" = 0 ]; then args+=(--resume); fi
  echo "running: python paper/phase-velocity-slices/scripts/run_forced_turbulence.py ${args[*]}"
  python paper/phase-velocity-slices/scripts/run_forced_turbulence.py "${args[@]}"
  rc=$?
fi

used=$(awk -v a="$t0" -v b="$(date +%s)" 'BEGIN { printf "%.4f", (b - a) / 3600 }')
printf 'end\t%s\t%s\t%s\t%s\n' "$(date +%s)" "${SLURM_JOB_ID:-?}" "$used" "$rc" >> "$REMOTE_ROOT/ledger.tsv"
case $rc in
  0) echo "=== run finished ($used h in this job)" ;;
  3)
    echo "=== run paused at the wall-time limit ($used h in this job); resubmitting"
    if ! bash "$P1S_HERE/cluster_submit.sh" "$P1S_CONFIG" --resubmit $((P1S_RESUBMIT + 1)) --time "$P1S_TIME"; then
      echo "=== not resubmitted; the run can be continued later with submit.sh --resume $P1S_CONFIG"
    fi
    exit 0 ;;
  *) echo "=== run failed (exit status $rc); not resubmitted" ;;
esac
exit "$rc"
