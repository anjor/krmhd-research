#!/bin/bash
# vm_setup.sh install|setup|start: run the Study 04 loop on a Linux server instead of the Mac.
#
# Written for Ubuntu 24.04 (apt and systemd). Run it as the ordinary user who will own the
# loop, never as root: an unattended agent should not run as root, and a root session can read
# files that the runner's guards rely on being unreadable. The steps that need root use sudo.
#
# On a fresh server, as that user:
#     sudo apt-get update && sudo apt-get install -y git gh
#     gh auth login                          # GitHub.com, HTTPS, log in with a browser
#     gh repo clone anjor/krmhd-research ~/repos/anjor/krmhd-research-loop
#     cd ~/repos/anjor/krmhd-research-loop/studies/04-phase-space-helicity/loop
#     bash vm_setup.sh install               # lsof, curl, uv, Claude Code; git through gh's login
#     ~/.local/bin/claude                    # log in once, then /exit
#     bash vm_setup.sh setup                 # clone marker, GANDALF and its worktree, the venv
#     ~/repos/anjor/krmhd-research-loop/.venv/bin/modal token new
#     bash vm_setup.sh start                 # preflight.sh, then the s04-loop service
#
# The layout is the Mac's (LOOP.md section 2): the loop's clone ~/repos/anjor/krmhd-research-loop
# with git config s04.loopclone true; GANDALF at ~/repos/anjor/gandalf with Anjor's git
# identity, by which the runner tells his branch moves from the loop's; and the loop's GANDALF
# worktree ~/repos/anjor/gandalf-study04, made from origin/main. install and setup skip what is
# already there, so they can be run again.
#
# Never run the loop on two machines at once, because both would push to main. run_forever.sh
# on the Mac exits when the loop stops; do not start it there again once a server runs the loop.
#
# The service starts at boot and runs run_forever.sh. run_forever.sh exits once STOP is on
# origin, and the service starts it again every 30 minutes; each start pulls main before it
# looks for STOP. So the loop resumes within 30 minutes of a pushed commit that removes STOP,
# made from anywhere, the GitHub web editor included. 'sudo systemctl stop s04-loop' stops it
# and lets the running iteration save and push its work first; 'journalctl -u s04-loop -f'
# follows its output.

set -u

LOOP_DIR=$(cd "$(dirname "$0")" && pwd -P) || exit 2
REPO=$(cd "$LOOP_DIR/../../.." && pwd -P) || exit 2
BASE="$HOME/repos/anjor"
CLONE="$BASE/krmhd-research-loop"
GANDALF="$BASE/gandalf"
WORKTREE="$BASE/gandalf-study04"
SERVICE=s04-loop
# Anjor's identity in the GANDALF repo (git log shows his commits as anjor <anjor@umd.edu>).
# config.env gives the loop the same email, so the runner cannot tell his branch moves there
# from the loop's (preflight says so). On a server only the loop works in GANDALF, so nothing
# is lost; but a GANDALF branch or tag moved there by hand while a session runs stops the loop.
ANJOR_GIT_NAME="${ANJOR_GIT_NAME:-anjor}"
ANJOR_GIT_EMAIL="${ANJOR_GIT_EMAIL:-anjor@umd.edu}"
export PATH="$HOME/.local/bin:$PATH"

die() { echo "vm_setup.sh: $*" >&2; exit 1; }
step() { echo "== $*"; }

cmd_install() {  # packages, git credentials through gh, uv and Claude Code in ~/.local/bin
  step "packages: git, gh, lsof, curl"
  sudo apt-get update -q && sudo apt-get install -y -q git gh lsof curl ca-certificates || die "apt-get failed"
  gh auth status --hostname github.com > /dev/null 2>&1 || die "log in to GitHub first: gh auth login"
  step "git fetches and pushes with gh's login"
  gh auth setup-git || die "gh auth setup-git failed"
  if [ -x "$HOME/.local/bin/uv" ]; then
    step "uv: already installed"
  else
    step "uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh || die "the uv installer failed"
  fi
  if [ -x "$HOME/.local/bin/claude" ]; then
    step "Claude Code: already installed"
  else
    step "Claude Code"
    curl -fsSL https://claude.ai/install.sh | bash || die "the Claude Code installer failed"
  fi
  [ -x "$HOME/.local/bin/claude" ] || die "Claude Code is not at ~/.local/bin/claude, where loop/config.env expects it"
  echo "Next: run ~/.local/bin/claude once and log in; then 'claude -p \"say ok\"' should print ok."
  echo "Then: bash vm_setup.sh setup"
}

cmd_setup() {  # the clone marker, GANDALF with Anjor's identity, the loop's worktree, the venv
  step "the clone marker in $CLONE"
  [ "$(git -C "$CLONE" symbolic-ref -q --short HEAD)" = main ] || die "$CLONE is not on main"
  git -C "$CLONE" config s04.loopclone true || die "git config failed"
  if [ -d "$GANDALF/.git" ]; then
    step "GANDALF: $GANDALF is there"
  else
    step "GANDALF: cloning anjor/gandalf to $GANDALF"
    git clone -q https://github.com/anjor/gandalf "$GANDALF" || die "cloning anjor/gandalf failed"
  fi
  step "GANDALF: Anjor's git identity, $ANJOR_GIT_NAME <$ANJOR_GIT_EMAIL>"
  git -C "$GANDALF" config user.name "$ANJOR_GIT_NAME" && git -C "$GANDALF" config user.email "$ANJOR_GIT_EMAIL" \
    || die "git config failed in $GANDALF"
  if [ -e "$WORKTREE/.git" ]; then
    step "GANDALF worktree: $WORKTREE is there"
  else
    step "GANDALF worktree: $WORKTREE, from origin/main"
    git -C "$GANDALF" fetch -q origin && git -C "$GANDALF" worktree add -q --detach "$WORKTREE" origin/main \
      || die "making the worktree failed"
  fi
  step "the clone's Python environment"
  (cd "$CLONE" && uv sync --frozen) || die "uv sync failed"
  echo "Next: log in to Modal with $CLONE/.venv/bin/modal token new, then: bash vm_setup.sh start"
}

cmd_start() {  # preflight.sh, then the systemd service
  local out fails
  [ -e "$WORKTREE/.git" ] && [ -x "$CLONE/.venv/bin/python" ] || die "run 'bash vm_setup.sh setup' first"
  step "preflight.sh"
  out=$(bash "$LOOP_DIR/preflight.sh" 2>&1)
  printf '%s\n' "$out"
  # Two failures do not hold up the start. The Study 2 data is not on this server (QUESTIONS.md
  # Q8), and the loop does not read it before Phase 1 task 5. Anjor's GANDALF identity shares
  # the loop's email (see ANJOR_GIT_EMAIL above), which matters only if he moves GANDALF
  # branches on this server.
  fails=$(printf '%s\n' "$out" | grep '^FAIL' | grep -v -e 'data resolves' -e "Anjor's git identity in the GANDALF repo")
  [ -z "$fails" ] || die "preflight failed (see above). Fix that and run 'bash vm_setup.sh start' again; nothing was installed."
  step "the $SERVICE service"
  sudo tee "/etc/systemd/system/$SERVICE.service" > /dev/null <<EOF || die "writing the service file failed"
[Unit]
Description=Study 04 loop (run_forever.sh in $CLONE)
Wants=network-online.target
After=network-online.target

[Service]
User=$(id -un)
WorkingDirectory=$CLONE
ExecStart=$LOOP_DIR/run_forever.sh
# SIGTERM goes to run_forever.sh only. It passes it on to the running iteration, which kills
# its session, saves and pushes the work and runs its checks; anything left is killed after.
KillMode=mixed
TimeoutStopSec=900
# run_forever.sh exits once STOP is on origin. Starting it again every 30 minutes lets a pushed
# commit that removes STOP resume the loop.
Restart=always
RestartSec=1800

[Install]
WantedBy=multi-user.target
EOF
  sudo systemctl daemon-reload && sudo systemctl enable --now "$SERVICE" || die "systemctl failed"
  systemctl --no-pager status "$SERVICE" | head -5
  echo "Follow it with: journalctl -u $SERVICE -f"
}

[ "$(id -u)" != 0 ] || die "run it as the loop's ordinary user, not as root"
[ "$REPO" = "$(cd "$CLONE" 2>/dev/null && pwd -P)" ] || die "run the copy in the loop's clone, $CLONE (this one is in $REPO)"
case "${1:-}" in
  install) cmd_install ;;
  setup) cmd_setup ;;
  start) cmd_start ;;
  *) sed -n '2,33p' "$0"; exit 2 ;;
esac
