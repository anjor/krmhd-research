#!/bin/bash
# cluster_submit.sh: submit one Slurm job of a run of paper/phase-velocity-slices/scripts/run_forced_turbulence.py.
# It runs on the cluster: on the login node (submit.sh calls it over ssh) and inside a job (job.sh calls it to continue
# a run that paused at the wall-time limit).
#
#   cluster_submit.sh CONFIG [--resume] [--resubmit N] [--time T] [--pause-after H]
#
#   CONFIG           run config, relative to REMOTE_ROOT/repo (or absolute)
#   --resume         continue the latest unfinished run of the config's label in place (implied by --resubmit)
#   --resubmit N     the Nth automatic resubmission of a paused run (job.sh sets it)
#   --time T         Slurm time limit of the job, HH:MM:SS or D-HH:MM:SS, at least 10 minutes (default: JOB_TIME)
#   --pause-after H  the first job pauses after H hours instead of near its time limit: a test of the pause, the
#                    resubmission and the resume (resubmitted jobs use the normal limit)
#
# Before sbatch it checks the limits in limits.conf: the jobs named p1-* queued or running, not counting the job that
# calls it (MAX_CONCURRENT_JOBS); the resubmission count (MAX_RESUBMITS_PER_RUN); and, when GPU_HOURS_PER_WEEK is set,
# the GPU hours of the jobs submitted in the past 7 days plus this one's time limit.  A refusal prints
# 'REFUSED: <reason>' and exits with status 2; nothing is submitted.  Otherwise it prints the job ID and appends a
# 'submit' line (time, job ID, GPUs, hours booked, label, resubmission) to REMOTE_ROOT/ledger.tsv.  A resubmitted job
# waits for the job that submitted it to end (--dependency afterany).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/common.sh"
SITE_ENV=${SITE_ENV:-$HERE/../../../site.env}
p1s_load_site "$SITE_ENV"
p1s_load_limits "${LIMITS_FILE:-$HERE/limits.conf}"

refuse() {
  echo "REFUSED: $*"
  exit 2
}
usage="usage: cluster_submit.sh CONFIG [--resume] [--resubmit N] [--time T] [--pause-after H]"

config="" resume=0 n=0 time=$JOB_TIME pause=none
while [ $# -gt 0 ]; do
  case $1 in
    --resume) resume=1 ;;
    --resubmit|--time|--pause-after)
      [ $# -ge 2 ] || p1s_die "$1 needs a value; $usage"
      case $1 in
        --resubmit) n=$2; resume=1 ;;
        --time) time=$2 ;;
        --pause-after) pause=$2 ;;
      esac
      shift ;;
    -*) p1s_die "unknown option $1; $usage" ;;
    *) [ -z "$config" ] || p1s_die "one config per job; $usage"; config=$1 ;;
  esac
  shift
done
[ -n "$config" ] || p1s_die "$usage"
case $n in ''|*[!0-9]*) p1s_die "--resubmit needs a whole number" ;; esac
case $pause in none) ;; ''|*[!0-9.]*|.|*.*.*) p1s_die "--pause-after needs a number of hours" ;; esac
case $config in *[[:space:],\'\"]*) p1s_die "config path must not contain spaces, commas or quotes" ;; esac
case $config in /*) cfg_file=$config ;; *) cfg_file=$REMOTE_ROOT/repo/$config ;; esac
[ -f "$cfg_file" ] || p1s_die "config $cfg_file not found"
label=$(p1s_config_label "$cfg_file")
[ -n "$label" ] || p1s_die "no top-level label in $cfg_file"
hours=$(p1s_hours "$time") || exit 1
awk -v h="$hours" 'BEGIN { exit !(h >= 1 / 6) }' || refuse "time limit $time is under 10 minutes"
[ -x "$REMOTE_ROOT/venv/bin/python" ] || refuse "no environment at $REMOTE_ROOT/venv: run setup_env.sh first"

# Limits
[ "$n" -le "$MAX_RESUBMITS_PER_RUN" ] \
  || refuse "resubmission $n of $label would exceed MAX_RESUBMITS_PER_RUN = $MAX_RESUBMITS_PER_RUN"
me=${USER:-$(id -un)}
queued=$(squeue -h -u "$me" -o '%i %j') || refuse "squeue failed, so the running jobs cannot be counted"
active=$(printf '%s\n' "$queued" \
  | awk -v self="${SLURM_JOB_ID:-}" -v p="$P1S_JOB_PREFIX" 'NF >= 2 && $1 != self && index($2, p) == 1' | wc -l | tr -d ' ')
[ "$active" -lt "$MAX_CONCURRENT_JOBS" ] \
  || refuse "$active job(s) named ${P1S_JOB_PREFIX}* already queued or running (MAX_CONCURRENT_JOBS = $MAX_CONCURRENT_JOBS)"
ledger=$REMOTE_ROOT/ledger.tsv
if [ -n "$GPU_HOURS_PER_WEEK" ]; then
  week=$(p1s_week_gpu_hours "$ledger")
  awk -v u="$week" -v h="$hours" -v g="$P1S_GPUS_PER_JOB" -v b="$GPU_HOURS_PER_WEEK" 'BEGIN { exit !(u + g * h <= b) }' \
    || refuse "GPU hours of the past 7 days ($week) plus this job's time limit ($hours h) exceed GPU_HOURS_PER_WEEK = $GPU_HOURS_PER_WEEK"
fi

# Submit
mkdir -p "$REMOTE_ROOT/logs" "$REMOTE_RESULTS" || p1s_die "cannot create $REMOTE_ROOT/logs or $REMOTE_RESULTS"
read -r -a site_opts <<< "${SBATCH_OPTS:-}"
dep=()
[ -z "${SLURM_JOB_ID:-}" ] || dep=(--dependency "afterany:$SLURM_JOB_ID")
out=$(sbatch --parsable ${site_opts[@]+"${site_opts[@]}"} ${dep[@]+"${dep[@]}"} \
  --job-name "$P1S_JOB_PREFIX$label" --time "$time" --output "$REMOTE_ROOT/logs/%x-%j.out" \
  --export "ALL,SITE_ENV=$SITE_ENV,P1S_HERE=$HERE,P1S_CONFIG=$config,P1S_RESUME=$resume,P1S_RESUBMIT=$n,P1S_TIME=$time,P1S_PAUSE_AFTER=$pause" \
  "$HERE/job.sh") || p1s_die "sbatch failed: $out"
jobid=${out%%;*}
case $jobid in ''|*[!0-9_]*) p1s_die "sbatch gave no job ID: $out" ;; esac
printf 'submit\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%s)" "$jobid" "$P1S_GPUS_PER_JOB" "$hours" "$label" "$n" >> "$ledger"
echo "submitted job $jobid: $P1S_JOB_PREFIX$label, time limit $time, resubmission $n${dep[@]+, after job $SLURM_JOB_ID}"
