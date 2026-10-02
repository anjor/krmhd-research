#!/bin/bash
# preflight.sh [--skip-uv] [--allow-stubs]: check that the Study 04 loop can run here.
#
# Run it in the loop's clone before starting the loop, and again after any change to the
# machine (new CLI versions, re-login). It prints PASS, FAIL or INFO per item and exits 1 if
# anything failed. It changes nothing except what `uv sync --frozen` installs (skip that
# with --skip-uv). It calls only read commands of gh and modal: `gh auth status`, `gh api`
# GETs, `modal app list`, `modal volume list`. It also checks the clone marker
# (git config s04.loopclone), the .gitignore rules for .loop/ and data/, that the clone
# holds no local settings, untracked .claude/ files, CLAUDE.local.md or git hooks, where
# Study 2's data folder is a link, that the link is excluded and the tracked files it hides
# carry the skip-worktree bit, and that the GANDALF repo keeps reflogs and gives Anjor's own
# git identity (not the loop's), by which the runner tells his branch moves from the loop's.
#
# --allow-stubs lets config.env point at loop/stubs/ (the self-test does that); in the real
# loop that is a FAIL.

set -u

SKIP_UV=0
ALLOW_STUBS=0
for a in "$@"; do
  case "$a" in
    --skip-uv) SKIP_UV=1 ;;
    --allow-stubs) ALLOW_STUBS=1 ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "preflight.sh: unknown option $a" >&2; exit 2 ;;
  esac
done

LOOP_DIR=$(cd "$(dirname "$0")" && pwd -P) || exit 2
REPO=$(cd "$LOOP_DIR/../../.." && pwd -P) || exit 2
STUDY_REL="studies/04-phase-space-helicity"
STUDY="$REPO/$STUDY_REL"
FAILS=0

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $1"; FAILS=$((FAILS + 1)); }
info() { echo "INFO  $1"; }

export PATH="${HOME:-/tmp}/.local/bin:/opt/homebrew/bin:/usr/local/bin:$PATH"

# --- The clone marker: the runner refuses to run without it -------------------------------
if [ "$(git -C "$REPO" config --local --get s04.loopclone 2>/dev/null)" = true ]; then
  pass "this is the loop's clone (git config s04.loopclone = true)"
else
  fail "git config s04.loopclone is not 'true' in $REPO: the runner refuses to run here (set it in the loop's clone only)"
fi

# --- config.env, read as KEY=VALUE text (never sourced), with the runner's S04_ overrides ---
KEYS="COMPUTE_CAP_A100_HOURS CLAUDE_BIN CLAUDE_MODEL CLAUDE_EFFORT MAX_BUDGET_USD DAILY_ITERATION_CAP TOTAL_ITERATION_CAP SESSION_TIME_LIMIT_SEC PAUSE_BETWEEN_ITERATIONS_SEC FAST_FAIL_SEC FAST_FAIL_BACKOFF_SEC WIP_STREAK_LIMIT WATCHDOG_POLL_SEC KILL_GRACE_SEC GANDALF_WORKTREE MODAL_APP_PREFIX MODAL_VOLUME MODAL_VOLUME_ROOT LOOP_GIT_NAME LOOP_GIT_EMAIL MODAL_BIN GH_BIN PYTHON_BIN UV_BIN STOP_ON_ANY_NEW_MODAL_APP RATE_LIMIT_MARGIN_SEC APPS_UNCHECKED_STOP_SEC"
for k in $KEYS; do eval "$k="; done
trim() { local s="$1"; s="${s#"${s%%[![:space:]]*}"}"; s="${s%"${s##*[![:space:]]}"}"; printf '%s' "$s"; }
if [ -f "$LOOP_DIR/config.env" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line=$(trim "$line")
    case "$line" in ''|'#'*) continue ;; esac
    case "$line" in 'export '*) line=$(trim "${line#export }") ;; esac
    key=$(trim "${line%%=*}")
    [ "$key" = "$line" ] && continue
    case " $KEYS " in *" $key "*) ;; *) continue ;; esac
    val=$(trim "${line#*=}")
    case "$val" in
      \"*\") val="${val#\"}"; val="${val%\"}" ;;
      \'*\') val="${val#\'}"; val="${val%\'}" ;;
      *) val=$(trim "${val%%[[:space:]]#*}") ;;
    esac
    eval "$key=\$val"
  done < "$LOOP_DIR/config.env"
else
  fail "loop/config.env exists"
fi
for k in $KEYS; do
  eval "v=\${S04_$k:-}"
  if [ -n "$v" ]; then eval "$k=\$v"; fi
done
[ -n "$MODAL_BIN" ] || MODAL_BIN="$REPO/.venv/bin/modal"
[ -n "$GH_BIN" ] || GH_BIN=gh
[ -n "$PYTHON_BIN" ] || PYTHON_BIN="$REPO/.venv/bin/python"
[ -n "$UV_BIN" ] || UV_BIN=uv
[ -n "$MODAL_VOLUME" ] || MODAL_VOLUME=krmhd-benchmark-vol
for k in CLAUDE_BIN MODAL_BIN GH_BIN PYTHON_BIN UV_BIN GANDALF_WORKTREE; do
  eval "v=\$$k"
  case "$v" in "~/"*) eval "$k=\"\$HOME/\${v#\~/}\"" ;; esac
done

if [ -x "$PYTHON_BIN" ] && "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 9)' 2>/dev/null; then
  pass "python: $PYTHON_BIN ($("$PYTHON_BIN" --version 2>&1))"
else
  fail "python: $PYTHON_BIN missing or older than 3.9 (run 'uv sync' in the clone)"
fi
if "$PYTHON_BIN" - "$LOOP_DIR" <<'EOF' >/dev/null 2>&1
import sys
sys.path.insert(0, sys.argv[1])
import loopcommon as lc
cfg = lc.parse_env_file(sys.argv[1] + "/config.env")
assert float(cfg["COMPUTE_CAP_A100_HOURS"]) > 0
for key in ("DAILY_ITERATION_CAP", "TOTAL_ITERATION_CAP", "SESSION_TIME_LIMIT_SEC",
            "PAUSE_BETWEEN_ITERATIONS_SEC", "FAST_FAIL_SEC", "FAST_FAIL_BACKOFF_SEC",
            "WIP_STREAK_LIMIT", "WATCHDOG_POLL_SEC", "KILL_GRACE_SEC"):
    assert int(cfg[key]) >= 0, key
for key in ("CLAUDE_BIN", "CLAUDE_MODEL", "CLAUDE_EFFORT", "GANDALF_WORKTREE", "MODAL_APP_PREFIX",
            "LOOP_GIT_NAME", "LOOP_GIT_EMAIL"):
    assert cfg[key], key
EOF
then
  pass "config.env parses: cap ${COMPUTE_CAP_A100_HOURS} A100-h, model ${CLAUDE_MODEL}, effort ${CLAUDE_EFFORT}, session limit ${SESSION_TIME_LIMIT_SEC}s, ${DAILY_ITERATION_CAP}/day"
else
  fail "config.env parses with every required key (loopcommon.parse_env_file)"
fi
case "$CLAUDE_BIN $MODAL_BIN $GH_BIN" in
  *"/loop/stubs/"*)
    if [ "$ALLOW_STUBS" = 1 ]; then info "config.env points at loop/stubs (allowed by --allow-stubs)";
    else fail "config.env points CLAUDE_BIN, MODAL_BIN or GH_BIN at loop/stubs"; fi ;;
esac

guard() { "$PYTHON_BIN" -I -S "$LOOP_DIR/guards.py" --repo "$REPO" "$@"; }

# --- GitHub --------------------------------------------------------------------------------
if "$GH_BIN" auth status --hostname github.com --active >/dev/null 2>&1; then pass "gh auth status"; else fail "gh auth status (run 'gh auth login')"; fi
v=$("$GH_BIN" api repos/anjor/gandalf --jq .permissions.push 2>/dev/null)
[ "$v" = true ] && pass "gh can push to anjor/gandalf" || fail "gh can push to anjor/gandalf (got '$v')"
v=$("$GH_BIN" api repos/anjor/gandalf --jq .allow_squash_merge 2>/dev/null)
[ "$v" = true ] && pass "anjor/gandalf allows squash merges" || fail "anjor/gandalf allows squash merges (got '$v')"
v=$("$GH_BIN" api repos/anjor/krmhd-research --jq .permissions.push 2>/dev/null)
[ "$v" = true ] && pass "gh can push to anjor/krmhd-research" || fail "gh can push to anjor/krmhd-research (got '$v')"
if "$GH_BIN" api repos/anjor/gandalf/branches/main/protection >/dev/null 2>&1; then
  info "anjor/gandalf main has branch protection"
else
  info "anjor/gandalf main has no branch protection (Anjor: require PRs with green CI)"
fi

# --- Modal ---------------------------------------------------------------------------------
tmp=$(mktemp -d "${TMPDIR:-/tmp}/s04-preflight.XXXXXX")
trap 'rm -rf "$tmp"' EXIT
if "$MODAL_BIN" app list --json > "$tmp/apps.json" 2>"$tmp/apps.err"; then
  guard apps-check --apps-json "$tmp/apps.json" --prefix "${MODAL_APP_PREFIX:-s04-loop}" > "$tmp/apps.check" 2>&1
  case $? in
    0) pass "modal app list works; no loop app outside the ledger" ;;
    1) fail "modal app list: loop apps not in the ledger: $(head -3 "$tmp/apps.check" | tr '\n' ' ')" ;;
    *) fail "modal app list output unreadable: $(head -1 "$tmp/apps.check")" ;;
  esac
else
  fail "modal app list --json (logged in? $(tail -1 "$tmp/apps.err"))"
fi
if "$MODAL_BIN" volume list --json 2>/dev/null | grep -q "\"$MODAL_VOLUME\""; then
  pass "modal volume $MODAL_VOLUME exists"
else
  fail "modal volume $MODAL_VOLUME exists"
fi

# --- Tools ---------------------------------------------------------------------------------
if v=$("$CLAUDE_BIN" --version 2>/dev/null); then pass "claude: $v ($CLAUDE_BIN)"; else fail "claude --version ($CLAUDE_BIN)"; fi
if command -v caffeinate >/dev/null 2>&1; then pass "caffeinate"; else fail "caffeinate (run_forever.sh keeps the Mac awake with it)"; fi
if command -v "$UV_BIN" >/dev/null 2>&1; then pass "uv: $("$UV_BIN" --version 2>/dev/null)"; else fail "uv on PATH"; fi
if [ "$SKIP_UV" = 1 ]; then
  info "uv sync --frozen skipped (--skip-uv)"
elif (cd "$REPO" && "$UV_BIN" sync --frozen) > "$tmp/uv.out" 2>&1; then
  pass "uv sync --frozen in $REPO"
else
  fail "uv sync --frozen in $REPO: $(tail -1 "$tmp/uv.out")"
fi

# --- The clone -----------------------------------------------------------------------------
if [ "$(git -C "$REPO" symbolic-ref -q --short HEAD)" = main ]; then pass "repo on main"; else fail "repo on main"; fi
if [ -z "$(git -C "$REPO" status --porcelain)" ]; then pass "working tree clean"; else fail "working tree clean: $(git -C "$REPO" status --porcelain | head -3 | tr '\n' ' ')"; fi
if git -C "$REPO" rev-parse -q --verify origin/main >/dev/null && [ "$(git -C "$REPO" rev-list --count origin/main..HEAD)" = 0 ]; then
  pass "no unpushed commits (as of the last fetch)"
else
  fail "no unpushed commits (as of the last fetch)"
fi
if guard ignore-check > "$tmp/ignore.out" 2>&1; then
  pass "$STUDY_REL/.loop/ and data/ are ignored by .gitignore rules"
else
  fail "ignore rules: $(head -2 "$tmp/ignore.out" | tr '\n' ' ')"
fi
if guard clone-check > "$tmp/clone.out" 2>&1; then
  pass "no local settings, untracked .claude/ files, CLAUDE.local.md or git hooks in the clone"
else
  fail "the clone holds files a session would load or git would run: $(head -3 "$tmp/clone.out" | tr '\n' ' ')"
fi
info "clone identity: user.name '$(git -C "$REPO" config --local --get user.name 2>/dev/null)' (the runner sets it to ${LOOP_GIT_NAME:-?})"
if [ -x /usr/sbin/lsof ]; then pass "lsof (the runner finds orphaned tool processes with it)"; else fail "/usr/sbin/lsof missing"; fi
case "${STOP_ON_ANY_NEW_MODAL_APP:-1}" in
  1) info "STOP_ON_ANY_NEW_MODAL_APP=1: any Modal app created during a session and missing from the ledger stops the loop" ;;
  0) info "STOP_ON_ANY_NEW_MODAL_APP=0: only apps named ${MODAL_APP_PREFIX:-s04-loop}* are checked against session windows" ;;
  *) fail "STOP_ON_ANY_NEW_MODAL_APP must be 0 or 1" ;;
esac
data="$REPO/studies/02-collisionality-scan/data/hermite128_nu3_imex"
if [ -e "$data" ]; then pass "ν-scan data resolves: $data"; else fail "ν-scan data resolves: $data"; fi
link_rel="studies/02-collisionality-scan/data"
if [ -L "$REPO/$link_rel" ]; then
  info "$link_rel is a link to $(readlink "$REPO/$link_rel")"
  # The link hides the tracked files of the folder it replaces (data/.gitkeep). Without the
  # skip-worktree bit git sees them as deleted, and no stash can take that deletion.
  unset_bits=$(git -C "$REPO" ls-files -v -- "$link_rel/" 2>/dev/null | awk '$1 != "S" && $1 != "s" { sub(/^[^ ]+ /, ""); print }' | tr '\n' ' ')
  if [ -z "$(trim "$unset_bits")" ]; then
    pass "the tracked files under the data link are marked skip-worktree"
  else
    fail "tracked files under the data link lack the skip-worktree bit: $unset_bits(the runner sets it at its start; by hand: git update-index --skip-worktree -- <path>)"
  fi
  if git -C "$REPO" check-ignore -q --no-index -- "$link_rel"; then
    pass "the data link is excluded from git"
  else
    fail "the data link is not ignored: add /$link_rel to .git/info/exclude (the runner adds it at its start)"
  fi
fi

# --- GANDALF worktree ------------------------------------------------------------------------
if [ -d "$GANDALF_WORKTREE" ] && git -C "$GANDALF_WORKTREE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  pass "GANDALF worktree: $GANDALF_WORKTREE ($(git -C "$GANDALF_WORKTREE" rev-parse --abbrev-ref HEAD 2>/dev/null))"
  wt_top=$(cd "$GANDALF_WORKTREE" && git rev-parse --show-toplevel 2>/dev/null)
  main_gandalf=$(cd "$REPO/../gandalf" 2>/dev/null && pwd -P)
  if [ -n "$main_gandalf" ] && [ "$wt_top" = "$main_gandalf" ]; then
    fail "the GANDALF worktree is ../gandalf itself; the loop must use its own worktree"
  fi
  # The runner tells a branch Anjor moves during a session from one the loop moves by the git
  # identity in the branch's reflog, so that identity must be readable and not the loop's,
  # and the repo must keep reflogs.
  ident=$(guard gandalf-refs --worktree "$GANDALF_WORKTREE" 2>/dev/null | sed -n 's/^ *"ident": "\(.*\)",*$/\1/p')
  if [ -n "$ident" ]; then
    pass "Anjor's git identity in the GANDALF repo: $ident"
  else
    fail "Anjor's git identity in the GANDALF repo cannot be read, or is the loop's (git var GIT_COMMITTER_IDENT there): every branch he moves during a session would stop the loop"
  fi
  if [ "$(git -C "$GANDALF_WORKTREE" config --get core.logAllRefUpdates 2>/dev/null)" = false ]; then
    fail "core.logAllRefUpdates is false in the GANDALF repo: without reflogs every branch Anjor moves during a session would stop the loop"
  fi
else
  fail "GANDALF worktree $GANDALF_WORKTREE exists and is a git work tree"
fi

# --- Loop files --------------------------------------------------------------------------------
for f in run_iteration.sh run_forever.sh wait.sh pause.sh resume.sh guards.py loopcommon.py; do
  [ -f "$LOOP_DIR/$f" ] || fail "loop/$f exists"
done
if guard check-settings --file "$LOOP_DIR/settings.json" > "$tmp/settings.out" 2>&1; then
  pass "settings.json is valid and has allow and deny lists"
else
  fail "settings.json: $(head -1 "$tmp/settings.out")"
fi
missing=""
for p in ITER_ID STARTED_UTC GANDALF_WORKTREE STASH_NOTE; do
  grep -q "{{$p}}" "$LOOP_DIR/prompt.md" 2>/dev/null || missing="$missing $p"
done
[ -z "$missing" ] && pass "prompt.md has its four placeholders" || fail "prompt.md lacks:$missing"
[ -f "$STUDY/LOOP.md" ] && pass "LOOP.md exists" || fail "LOOP.md exists"
if [ -f "$LOOP_DIR/frozen.json" ] && guard frozen-check > "$tmp/frozen.out" 2>&1; then
  pass "frozen sections match loop/frozen.json"
else
  fail "frozen sections: $(head -2 "$tmp/frozen.out" 2>/dev/null | tr '\n' ' ')"
fi
if guard ledger-verify > "$tmp/ledger.out" 2>&1; then pass "ledger verifies"; else fail "ledger: $(head -2 "$tmp/ledger.out" | tr '\n' ' ')"; fi
v=$("$PYTHON_BIN" -c 'import sys; sys.path.insert(0, sys.argv[1]); import loopcommon as lc
print(lc.status_line(lc.compute_cap(sys.argv[2]), lc.load_ledger(sys.argv[2] + "/" + lc.LEDGER_REL)))' "$LOOP_DIR" "$REPO" 2>/dev/null)
[ -n "$v" ] && info "$v"
if [ -e "$STUDY/STOP" ] || git -C "$REPO" cat-file -e "HEAD:$STUDY_REL/STOP" 2>/dev/null; then
  info "STOP is present (in the working tree or committed): the runner will not start a session until it is removed"
fi
[ -e "$STUDY/WAIT" ] && info "WAIT: $(guard wait-remaining 2>/dev/null | tr '\n' ' ')"

echo
if [ "$FAILS" -eq 0 ]; then
  echo "preflight: all checks passed"
  exit 0
fi
echo "preflight: $FAILS check(s) failed"
exit 1
