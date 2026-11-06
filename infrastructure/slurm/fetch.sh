#!/bin/bash
# fetch.sh: copy back to the laptop everything the cluster jobs wrote: the run folders in REMOTE_RESULTS into
# LOCAL_RESULTS (default paper/phase-velocity-slices/data), and the Slurm logs and the ledger into LOCAL_LOGS (default
# paper/phase-velocity-slices/data/slurm_logs).  Nothing is deleted on either side; files already copied and unchanged
# are skipped.  It first lists the p1-* jobs queued or running, and stops if the local disk lacks the space for what
# is still to be copied.
#
#   SITE_ENV=<site file> infrastructure/slurm/fetch.sh [--follow [MINUTES]]
#
# --follow copies every MINUTES (default 15) while p1-* jobs are queued or running and once more after the last one
# has ended, so that the output of each job is on the laptop soon after the job ends.  It runs on the laptop and only
# calls squeue, du, ls and rsync over ssh; a failed connection (VPN down, laptop asleep) is retried at the next round.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/common.sh"
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO" || exit 1
p1s_load_site "${SITE_ENV:-}"
LOCAL_RESULTS=${LOCAL_RESULTS:-paper/phase-velocity-slices/data}
LOCAL_LOGS=${LOCAL_LOGS:-paper/phase-velocity-slices/data/slurm_logs}
follow=0 minutes=15
case ${1:-} in
  "") ;;
  --follow)
    follow=1
    if [ $# -ge 2 ]; then minutes=$2; fi
    case $minutes in ''|*[!0-9]*|0) p1s_die "--follow takes a whole number of minutes" ;; esac ;;
  *) p1s_die "usage: fetch.sh [--follow [MINUTES]]" ;;
esac
mkdir -p "$LOCAL_RESULTS" "$LOCAL_LOGS" || exit 1

rsh() {  # rsh COMMAND: run COMMAND on the login node
  # shellcheck disable=SC2086
  ssh $P1S_SSH_OPTS "$SSH_HOST" "$1"
}

our_jobs() {  # the p1-* jobs of this user, queued or running: ID, name, state, elapsed, limit
  local q
  q=$(rsh 'squeue -h -u "$USER" -o "%i %j %T %M %l"') || return 1
  printf '%s\n' "$q" | awk -v p="$P1S_JOB_PREFIX" 'NF >= 2 && index($2, p) == 1'
}

fetch_once() {
  local names d remote_kb have_kb=0 free_kb
  # bash -c: the redirections must not depend on the login shell on the cluster
  names=$(rsh "bash -c 'ls -1 $REMOTE_RESULTS 2>/dev/null; true'") || return 1
  remote_kb=$(rsh "bash -c 'du -sk $REMOTE_RESULTS $REMOTE_ROOT/logs 2>/dev/null; true'" \
    | awk '{ s += $1 } END { print s + 0 }') || return 1
  for d in $names; do
    if [ -d "$LOCAL_RESULTS/$d" ]; then have_kb=$((have_kb + $(du -sk "$LOCAL_RESULTS/$d" | cut -f1))); fi
  done
  free_kb=$(df -Pk "$LOCAL_RESULTS" | awk 'NR == 2 { print $4 }')
  if [ $((remote_kb - have_kb)) -gt $((free_kb - 5 * 1024 * 1024)) ]; then
    echo "fetch.sh: about $(((remote_kb - have_kb) / 1024)) MB to copy but only $((free_kb / 1024)) MB free locally (keeping 5 GB spare); not copying" >&2
    return 2
  fi
  rsync -az --partial --exclude 'resume.tmp.npz' -e "ssh $P1S_SSH_OPTS" "$SSH_HOST:$REMOTE_RESULTS/" "$LOCAL_RESULTS/" \
    || return 1
  rsync -az -e "ssh $P1S_SSH_OPTS" "$SSH_HOST:$REMOTE_ROOT/logs/" "$LOCAL_LOGS/" || return 1
  if rsh "test -f $REMOTE_ROOT/ledger.tsv"; then
    rsync -az -e "ssh $P1S_SSH_OPTS" "$SSH_HOST:$REMOTE_ROOT/ledger.tsv" "$LOCAL_LOGS/ledger.tsv" || return 1
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S'): copied $REMOTE_RESULTS -> $LOCAL_RESULTS, logs and ledger -> $LOCAL_LOGS"
}

if [ "$follow" = 0 ]; then
  jobs_now=$(our_jobs) || p1s_die "ssh to $SSH_HOST failed (VPN? key?)"
  [ -z "$jobs_now" ] || { echo "queued or running:"; printf '%s\n' "$jobs_now" | sed 's/^/  /'; }
  fetch_once || exit $?
  exit 0
fi
while :; do
  if ! jobs_now=$(our_jobs); then
    echo "$(date '+%Y-%m-%d %H:%M:%S'): cannot reach $SSH_HOST; trying again in $minutes min" >&2
    sleep $((minutes * 60))
    continue
  fi
  st=0
  fetch_once || st=$?
  [ "$st" != 2 ] || exit 2
  if [ "$st" = 0 ] && [ -z "$jobs_now" ]; then
    echo "no p1-* jobs left; done"
    exit 0
  fi
  [ -z "$jobs_now" ] || printf '%s\n' "$jobs_now" | sed 's/^/  /'
  sleep $((minutes * 60))
done
