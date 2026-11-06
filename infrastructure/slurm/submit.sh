#!/bin/bash
# submit.sh: from the laptop, start or continue a run of paper/phase-velocity-slices/scripts/run_forced_turbulence.py
# on a Slurm cluster, or build the run environment there.
#
#   SITE_ENV=<site file> infrastructure/slurm/submit.sh [--resume] [--time T] [--pause-after H] [--dry-run] CONFIG
#   SITE_ENV=<site file> infrastructure/slurm/submit.sh --setup [--dry-run]
#
# It copies to REMOTE_ROOT/repo on the cluster, at their paths in the repository: the run code
# (run_forced_turbulence.py and the two modules it imports, h1lib.py and p1lib.py), the config, the restart file that
# the config's init.path names (repo-relative paths only) and the cluster-side scripts of this folder.  The site file
# goes to REMOTE_ROOT/site.env and the commit of HEAD to REMOTE_ROOT/repo/COMMIT.  Then it runs cluster_submit.sh
# there, which checks the limits (limits.conf) and calls sbatch; or, with --setup, setup_env.sh.  The config, the run
# code and these scripts must be committed and unchanged, so that the commit says what ran.  ssh runs in batch mode:
# it logs in with the key and never asks for a password.  --dry-run prints the files and the commands and contacts
# nothing.  --time, --pause-after and --resume are passed to cluster_submit.sh (see there).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/common.sh"
REPO=$(cd "$HERE/../.." && pwd)
cd "$REPO" || exit 1
p1s_load_site "${SITE_ENV:-}"
usage="usage: submit.sh [--resume] [--time T] [--pause-after H] [--dry-run] CONFIG | submit.sh --setup [--dry-run]"

config="" setup=0 dry=0 opts=()
while [ $# -gt 0 ]; do
  case $1 in
    --setup) setup=1 ;;
    --dry-run) dry=1 ;;
    --resume) opts+=(--resume) ;;
    --time|--pause-after)
      [ $# -ge 2 ] || p1s_die "$1 needs a value; $usage"
      opts+=("$1" "$2")
      shift ;;
    -*) p1s_die "unknown option $1; $usage" ;;
    *) [ -z "$config" ] || p1s_die "one config at a time; $usage"; config=$1 ;;
  esac
  shift
done

slurm_files="infrastructure/slurm/common.sh infrastructure/slurm/cluster_submit.sh infrastructure/slurm/job.sh
infrastructure/slurm/setup_env.sh infrastructure/slurm/requirements.txt infrastructure/slurm/limits.conf"
code_files="paper/phase-velocity-slices/scripts/run_forced_turbulence.py paper/phase-velocity-slices/scripts/h1lib.py
paper/phase-velocity-slices/scripts/p1lib.py"
files=$slurm_files
init=""
if [ "$setup" = 1 ]; then
  [ -z "$config" ] || p1s_die "--setup takes no config; $usage"
  remote="env SITE_ENV=$REMOTE_ROOT/site.env bash $REMOTE_ROOT/repo/infrastructure/slurm/setup_env.sh"
else
  [ -n "$config" ] || p1s_die "$usage"
  case $config in
    /*|../*|*/../*) p1s_die "give the config as a path inside the repository, e.g. paper/phase-velocity-slices/configs/<name>.yaml" ;;
    *[[:space:],\'\"]*) p1s_die "config path must not contain spaces, commas or quotes" ;;
  esac
  [ -f "$config" ] || p1s_die "config $config not found"
  [ -n "$(p1s_config_label "$config")" ] || p1s_die "no top-level label in $config"
  init=$(p1s_config_init_path "$config")
  case $init in
    "") ;;
    /*) p1s_die "init.path in $config is absolute ($init) and would not exist on the cluster; copy the file into the repository's data folder and give init.path relative to the repository" ;;
    *) [ -f "$init" ] || p1s_die "init.path of $config ($init) not found" ;;
  esac
  files="$slurm_files $code_files $config"
  remote="env SITE_ENV=$REMOTE_ROOT/site.env bash $REMOTE_ROOT/repo/infrastructure/slurm/cluster_submit.sh $config"
  [ ${#opts[@]} -eq 0 ] || remote="$remote ${opts[*]}"
fi

commit=$(git rev-parse HEAD)
list=$(mktemp)
trap 'rm -f "$list"' EXIT
# shellcheck disable=SC2086
printf '%s\n' $files $init > "$list"
echo "commit $commit; copying to $SSH_HOST:$REMOTE_ROOT/repo:"
sed 's/^/  /' "$list"
[ -z "$init" ] || echo "  (init file $(du -h "$init" | cut -f1))"
echo "then on $SSH_HOST: $remote"

# Everything that defines the job must be committed and unchanged (the restart file is data and is not in git).
not_committed=""
for f in $files; do
  git ls-files --error-unmatch -- "$f" >/dev/null 2>&1 || not_committed="$not_committed $f"
done
# shellcheck disable=SC2086
changed=$(git diff --name-only HEAD -- $files 2>/dev/null | tr '\n' ' ')
[ -z "$not_committed" ] || p1s_die "not committed:$not_committed"
[ -z "$changed" ] || p1s_die "uncommitted changes in: $changed"
[ "$dry" = 0 ] || exit 0

# shellcheck disable=SC2086
ssh $P1S_SSH_OPTS "$SSH_HOST" "mkdir -p $REMOTE_ROOT/repo $REMOTE_ROOT/logs $REMOTE_RESULTS" \
  || p1s_die "ssh to $SSH_HOST failed (VPN? key?)"
rsync -az --files-from="$list" -e "ssh $P1S_SSH_OPTS" ./ "$SSH_HOST:$REMOTE_ROOT/repo/" || p1s_die "copying the files failed"
rsync -az -e "ssh $P1S_SSH_OPTS" "$SITE_ENV" "$SSH_HOST:$REMOTE_ROOT/site.env" || p1s_die "copying the site file failed"
# shellcheck disable=SC2086
echo "$commit" | ssh $P1S_SSH_OPTS "$SSH_HOST" "cat > $REMOTE_ROOT/repo/COMMIT" || p1s_die "writing COMMIT failed"
# shellcheck disable=SC2086
ssh $P1S_SSH_OPTS "$SSH_HOST" "$remote"
