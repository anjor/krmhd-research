# common.sh: functions shared by the Slurm scripts in this folder (sourced, never run).
# Bash 3.2 or newer: macOS ships 3.2 for the laptop side, so no associative arrays, no ${x,,}, and empty arrays are
# expanded as ${a[@]+"${a[@]}"} under set -u.

P1S_JOB_PREFIX="p1-"                                   # Slurm job names: p1-<label of the config>
P1S_GPUS_PER_JOB=1                                     # the run code uses one device
P1S_SSH_OPTS="-o BatchMode=yes -o ConnectTimeout=20"   # key login only: ssh fails instead of asking for a password

p1s_die() {
  echo "$(basename "$0"): $*" >&2
  exit 1
}

p1s_hours() {  # p1s_hours T: a Slurm time limit HH:MM:SS or D-HH:MM:SS, in hours
  local t=$1 d=0 re='^([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$'
  if ! [[ $t =~ $re ]]; then
    echo "time limit '$1': use HH:MM:SS or D-HH:MM:SS" >&2
    return 1
  fi
  case $t in *-*) d=${t%%-*}; t=${t#*-} ;; esac
  echo "$t" | awk -F: -v d="$d" '{ printf "%.6f\n", 24 * d + $1 + $2 / 60 + $3 / 3600 }'
}

p1s_load_site() {  # p1s_load_site FILE: source the site settings (KEY=value lines, see site.env.example) and check them
  local f=${1:-} v
  [ -n "$f" ] || p1s_die "set SITE_ENV to the site settings file (see infrastructure/slurm/site.env.example)"
  [ -f "$f" ] || p1s_die "site settings file '$f' not found"
  set -a
  . "$f"
  set +a
  for v in SSH_HOST REMOTE_ROOT REMOTE_RESULTS JOB_TIME PYTHON JAX_CUDA; do
    [ -n "${!v:-}" ] || p1s_die "$v is not set in $f"
  done
  for v in REMOTE_ROOT REMOTE_RESULTS; do
    case ${!v} in /*) ;; *) p1s_die "$v must be an absolute path on the cluster" ;; esac
    case ${!v} in *[[:space:],\'\"]*) p1s_die "$v must not contain spaces, commas or quotes" ;; esac
  done
  p1s_hours "$JOB_TIME" >/dev/null || exit 1
}

p1s_load_limits() {  # p1s_load_limits FILE: MAX_CONCURRENT_JOBS, MAX_RESUBMITS_PER_RUN and GPU_HOURS_PER_WEEK (may be empty)
  [ -f "$1" ] || p1s_die "limits file '$1' not found"
  . "$1"
  case ${MAX_CONCURRENT_JOBS:-} in ''|*[!0-9]*) p1s_die "MAX_CONCURRENT_JOBS in $1 must be a whole number" ;; esac
  case ${MAX_RESUBMITS_PER_RUN:-} in ''|*[!0-9]*) p1s_die "MAX_RESUBMITS_PER_RUN in $1 must be a whole number" ;; esac
  case ${GPU_HOURS_PER_WEEK:-} in
    '') ;;
    *[!0-9.]*|.|*.*.*) p1s_die "GPU_HOURS_PER_WEEK in $1 must be a number, or empty for no weekly limit" ;;
  esac
}

p1s_load_modules() {  # load the site's modules (MODULES in the site file), if any
  [ -n "${MODULES:-}" ] || return 0
  local f
  if ! type module >/dev/null 2>&1; then
    set +u
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      if [ -f "$f" ]; then
        . "$f"
        break
      fi
    done
    set -u
  fi
  type module >/dev/null 2>&1 || p1s_die "MODULES is set but the module command is not available"
  set +u
  module load $MODULES || p1s_die "module load $MODULES failed"
  set -u
}

p1s_config_label() {  # p1s_config_label CONFIG: the top-level 'label:' of a run config
  sed -n 's/^label:[[:space:]]*\([A-Za-z0-9._-]*\).*/\1/p' "$1" | head -n 1
}

p1s_config_init_path() {  # p1s_config_init_path CONFIG: init.path of a run config (empty when it starts from rest)
  awk '/^init:/ { inside = 1; next }
       /^[^[:space:]#]/ { inside = 0 }
       inside && $1 == "path:" { print $2; exit }' "$1"
}

p1s_week_gpu_hours() {  # p1s_week_gpu_hours LEDGER: GPU hours of the jobs submitted in the past 7 days
  # A job counts its time limit until its 'end' line exists and its hours used after that.
  if [ ! -f "$1" ]; then
    echo 0.000
    return 0
  fi
  awk -v now="$(date +%s)" '
    $1 == "submit" && $2 >= now - 7 * 86400 { gpus[$3] = $4; booked[$3] = $5 }
    $1 == "end" { used[$3] = $4 }
    END { s = 0; for (j in gpus) s += gpus[j] * ((j in used) ? used[j] : booked[j]); printf "%.3f\n", s }' "$1"
}
