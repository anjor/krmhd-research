#!/usr/bin/env python3
"""Study 04 Modal launcher: the only path from the loop to a GPU.

Run it from the repo root:

    uv run python studies/04-phase-space-helicity/loop/modal_launch.py <command> ...

Commands
--------
launch --set S --timeout-hours H --gate-report P [--gate-report P]... --local-test P [--freeze KEY]... CONFIG...
    Reserve H A100-hours per config, commit and push the reservation, then start one
    detached Modal call per config (one A100 each, no retries, timeout H). Every gate
    report given must pass the report check (below).
smoke [--sleep S]
    The same path with a tiny CPU-only function. Charges zero hours.
reconcile [--launch L]
    Ask Modal which calls have finished and charge them from the attempt records on the
    volume. Commits and pushes the ledger if it changed.
status
    The compute line for the log (cap, used, reserved, left), then the runs in flight.
fetch RUN_ID [--checkpoints] [--dest DIR]
    Download a run's directory from the volume (default studies/04-.../data/<set>/<run>/),
    without checkpoints/ unless --checkpoints, and its attempt records into _attempts/
    there. A local copy is kept only if its size and modification time equal the volume's;
    anything else is downloaded again.
verify
    Check the ledger's structure and integrity hash.
apps --json-file FILE
    Running Modal apps with the loop's prefix that the ledger does not account for, read
    from saved ``modal app list --json`` output.

Paths on the command line are relative to the repo root.

Exit codes
----------
0   done.
1   refused: a rule failed and nothing was changed (the message names the rule); or a
    launch that sent no call to Modal and released its whole reservation.
2   usage or environment error (config, git, Modal unreachable). For reconcile: some runs
    could not be checked and were left exactly as they were; run it again later.
3   launch or smoke: the outcome of at least one run is uncertain. Its spawn raised after
    the request may have reached Modal, so its reservation is kept (status ``unknown``)
    and reconcile settles it. Never relaunch such a config before reconcile has charged it.
4   reconcile or status: hours used plus hours reserved exceed the cap (a hard stop).
130 interrupted by a signal.

Why it works this way
---------------------
Nobody watches the loop, so the cap (COMPUTE_CAP_A100_HOURS in loop/config.env, Anjor's
setting; never read from the environment) is enforced here, before money is spent. A
launch reserves one full timeout per config and refuses if used + reserved + new would
exceed the cap. The reservation is committed and pushed before Modal is called, so a
launch that dies half way is still in the ledger. Hours are charged later from the start
and end records that the Modal function writes for every attempt, in
/<root>/_attempts/<set>/<run>/ on the volume, outside the run's own folder. Nobody types
hours in. modal_app.py bounds one call's GPU time by its timeout T, counted from its
first attempt, so a preemption restart cannot spend a second timeout; reconcile therefore
holds and charges at most T per run (see charge_from_records), unless the records show
the budget was not kept. A run without records is charged its full reservation.

A spawn that raises may still have reached Modal, so its run keeps its reservation as
``unknown``; reconcile charges it from its attempt records, or the full reservation if it
has none. Only runs whose spawn was never attempted are released. A launch whose app
could not be created, or whose app was created without a single spawn, released
everything: an app runs no container without a call.

Reconcile changes a run only on a definite answer from Modal (classify_call_exception).
An error reaching Modal (a terminated stream, a protocol error, authentication,
connection, permission, a call or volume Modal cannot find, as from another workspace or
environment) leaves the run as it was and makes the command exit 2.

The launcher trusts a gate report only in the form LOOP.md gives (gate_report_problems,
gate_report_file_problems): one 'Result: PASS', one 'Coded check: PASS', one 'Critic:
VERDICT: SUPPORTED, <iteration id>, <critic report>' line whose critic report says so
too, one 'Kill criteria: none met', no failing row in a results table, a title and file
name that agree, and no unfilled template text. Each run set (base, A, B; there are no
others) launches only on the reports LOOP.md names for it, and a quality gate's report
must be its newest evaluation (check_gate_reports).

Every commit this script makes touches only compute_ledger.json and has a subject that
starts with "Study 04 [launcher]". The runner writes STOP on any other commit that
touches the ledger, and the ledger's integrity hash exposes hand edits. Loop commits are
told apart by their committer name (LOOP_GIT_NAME), which the runner sets for a session;
the author can be set per commit, the committer cannot. GitHub-made commits (web merges,
squash merges, web edits) never count as Anjor's, because the loop's gh login can make
them. Before it writes, the launcher walks the whole history, both sides of every merge
(GitHistory), and checks that HEAD's ledger extends every ledger version committed since
Anjor's last ledger commit (an old version has a valid hash too, so a rollback would
otherwise free headroom). Launch and smoke also refuse while the content of a file under
loop/ (the cap, the GPU function, this script) comes from a commit Anjor did not make.

A refusal changes nothing: no ledger write, no commit, no Modal call.

Backends: S04_MODAL_BACKEND=modal (default) uses the Modal SDK in-process;
S04_MODAL_BACKEND=stub keeps fake Modal state in S04_STUB_DIR (tests only: refused unless
origin is a local, non-GitHub repository). S04_REPO overrides the repo root (tests only).
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    tomllib = None  # type: ignore[assignment]

LOOP_DIR = Path(__file__).resolve().parent
if str(LOOP_DIR) not in sys.path:
    sys.path.insert(0, str(LOOP_DIR))

import loopcommon as lc  # noqa: E402

EXIT_OK = 0
EXIT_REFUSED = 1
EXIT_ENV = 2
EXIT_UNCERTAIN = 3
EXIT_CAP_EXCEEDED = 4
EXIT_INTERRUPTED = 130

MAX_TIMEOUT_HOURS = 24.0  # Modal's per-call limit
MIN_TIMEOUT_S = 60
STUCK_GRACE_HOURS = 1.0  # reconcile warns about a call still unfinished this long past its budget
# Records may show a little more than the budget (the watchdog's own end record, clocks of
# different containers); beyond this the records say the budget was not kept.
BUDGET_TOLERANCE_HOURS = 0.1
SMOKE_TIMEOUT_HOURS = 0.25  # modal_app.SMOKE_TIMEOUT_S / 3600
DEFAULT_SMOKE_SLEEP_S = 90
MAX_SMOKE_SLEEP_S = 600  # modal_app.SMOKE_MAX_SLEEP_S
IMAGE_PYTHON_FULL = "3.12.0"  # modal_app.PYTHON_VERSION, for picking the jax pin in uv.lock
GANDALF_PACKAGE = "gandalf-krmhd"
ATTEMPTS_DIRNAME = "_attempts"  # modal_app.ATTEMPTS_DIRNAME: records live in /<root>/_attempts/<set>/<run>/
CONFIGS_REL = f"{lc.STUDY_REL}/configs"
GATE_REPORTS_REL = f"{lc.STUDY_REL}/gate_reports"
RUNS_REL = f"{lc.STUDY_REL}/RUNS.md"
CALL_STATES = ("running", "ok", "failed", "expired", "timeout", "error")
# The plan's three GPU run sets (LOOP.md section 3: a fourth is a hard stop).
RUN_SETS = ("base", "A", "B")

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_LAUNCH_ID_RE = re.compile(r"^L(\d+)$")
_RUN_DEF_RE = re.compile(r"^def run\(", re.M)

# Gate reports (LOOP.md, "Gate reports"). Text the template has and a finished report must not.
GATE_TEMPLATE_PLACEHOLDERS = (
    "<n>", "<S>", "<iteration id>", "<name, with PLAN.md and SPEC.md references>", "<UTC time>",
    "<sha>", "<the exact command line>", "<run IDs, and where each ran>", "<run IDs and why>",
    "<path>", "<hash>", "<freeze key>", "<Lnnn>", "<its VERDICT line, verbatim>",
    "<critic report file>", "<which, with the evidence>",
)
_GATE_HEADER_KEYS = ("gate", "evaluated:", "command:", "runs:", "left out:", "frozen", "coded check:",
                     "critic:", "kill criteria:", "result:")
_PLACEHOLDER_RE = re.compile(r"<[A-Za-z][^<>\n]{0,80}>")
_LINE_DECORATION = " \t>*-+#|_`"
_ITER = r"it-\d{8}-\d{4}"
# The label of each one-line verdict, matched after Markdown decoration and in any case, so
# that 'Final result: FAIL', 'Result (rerun): FAIL' or 'Coded check (dt fit): FAIL' count too.
_VERDICT_LABELS = (
    ("Result:", re.compile(r"^(?:final\s+)?result\b")),
    ("Coded check:", re.compile(r"^coded\s+checks?\b")),
    ("Critic:", re.compile(r"^critic\b")),
    ("Kill criteria:", re.compile(r"^kill\s+criteri")),
)
# 'Critic: <its VERDICT line, verbatim>, <iteration id>, <critic report file>', with nothing
# between the verdict and the comma: 'SUPPORTED (partially)' or 'SUPPORTED?' is a hedge.
_CRITIC_LINE_RE = re.compile(r"^Critic: [`*_\"']{0,2}VERDICT: SUPPORTED[`*_\"']{0,2}, (?P<iter>" + _ITER
                             + r"), [`\"']?(?P<file>[^\s,`\"']+\.md)[`\"']?\.?$")
_OTHER_VERDICT_RE = re.compile(r"refuted|inconclusive|not supported|unsupported", re.I)
_PASS_HEADERS = ("pass", "pass?", "passed", "passes")
_PASS_CELLS = ("yes", "pass")
_NA_CELLS = ("n/a", "na", "-", "—", "–")
# Report file names (LOOP.md, "Gate reports"): G<n>_<iteration id>.md for Gates 1 to 3,
# Gbase_<iteration id>.md for the base-state gate and G4_<S>_<iteration id>.md for Gate 4.
_GATE_FILE_RE = re.compile(r"^G(?:(?P<n>[123])|4_(?P<set>[A-Za-z0-9]+)|(?P<base>base))_(?P<iter>" + _ITER
                           + r")\.md$")
# Which reports a launch of each run set rests on (LOOP.md section 9, "Before a launch"):
# (groups of which each needs one report, kinds that may come with them).
_SET_GATES: Dict[str, Tuple[Tuple[Tuple[str, ...], ...], Tuple[str, ...]]] = {
    "base": ((("G1", "Gbase"),), ("G2",)),
    "A": ((("G3",), ("Gbase",)), ("G1", "G2")),
    "B": ((("G4_A",),), ("G1", "G2", "G3", "Gbase")),
}
# Committers that are never Anjor's own commit: GitHub makes web merges, squash and rebase
# merges and web edits, and the loop's gh login, which is Anjor's account, can ask for those.
_GITHUB_COMMITTER_EMAILS = ("noreply@github.com",)
_RESET_HOW = ("Anjor resets it with a plain commit of the ledger in his own checkout (not a merge, and not "
              "through GitHub)")


# ---------------------------------------------------------------------------
# Errors and small helpers
# ---------------------------------------------------------------------------


class Refusal(Exception):
    """A launcher rule failed. Nothing was changed (exit code 1)."""

    def __init__(self, rule: str, detail: str) -> None:
        """``rule`` names the rule that failed; ``detail`` says why."""
        super().__init__(f"{rule}: {detail}")
        self.rule = rule
        self.detail = detail


class EnvError(Exception):
    """A usage or environment problem: config, git or Modal unavailable (exit code 2)."""


class BackendError(Exception):
    """Modal (or the stub) could not be queried. Nothing is charged on that basis."""


def utc_iso(t: Optional[float] = None) -> str:
    """UTC time as 2026-10-05T10:15:00Z."""
    return datetime.fromtimestamp(time.time() if t is None else t, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")


def short(text: Any, limit: int = 300) -> str:
    """One line of at most ``limit`` characters."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def short_exc(exc: BaseException, limit: int = 300) -> str:
    """``Type: message`` of an exception, on one short line."""
    return short(f"{type(exc).__name__}: {exc}", limit)


def fmt_h(hours: float) -> str:
    """Hours without trailing zeros: 8, 0.25."""
    return f"{float(hours):g}"


def timeout_seconds(hours: float) -> int:
    """A timeout in hours as whole seconds."""
    return int(round(float(hours) * 3600.0))


def note_append(old: Optional[str], text: str) -> str:
    """Append ``text`` to a launch's notes."""
    return f"{old}; {text}" if old else text


def parse_json_list(text: str) -> Any:
    """JSON printed by a CLI, tolerating notice lines printed before the JSON itself."""
    try:
        return json.loads(text)
    except ValueError:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith("["):
                return json.loads("\n".join(lines[i:]))
        raise


def repo_root() -> Path:
    """The repo the launcher works on: S04_REPO (tests) or the repo this file lives in."""
    override = os.environ.get("S04_REPO", "").strip()
    return Path(override).expanduser().resolve() if override else LOOP_DIR.parents[2]


# ---------------------------------------------------------------------------
# Settings (loop/config.env; only Anjor edits it)
# ---------------------------------------------------------------------------


@dataclass
class Settings:
    cap: float  # A100-hours for the whole study
    prefix: str  # Modal app name prefix
    volume: str  # Modal volume name
    volume_root: str  # folder on the volume for this study
    modal_bin: str  # modal CLI, used only to list apps
    loop_name: str = "krmhd-loop"  # git committer name of every loop commit (LOOP_GIT_NAME)


def load_settings(repo: Path) -> Settings:
    """Read config.env. The cap comes only from the file, never from the environment."""
    path = repo / lc.CONFIG_REL
    if not path.is_file():
        raise EnvError(f"{lc.CONFIG_REL} not found under {repo}")
    try:
        cfg = lc.parse_env_file(path)
        cap = lc.compute_cap(repo)
    except (OSError, ValueError) as exc:
        raise EnvError(f"cannot read the compute cap from {lc.CONFIG_REL}: {exc}") from None
    if not cap >= 0:
        raise EnvError(f"COMPUTE_CAP_A100_HOURS in {lc.CONFIG_REL} is {cap}; it must be >= 0")
    return Settings(
        cap=cap,
        prefix=cfg.get("MODAL_APP_PREFIX") or "s04-loop",
        volume=cfg.get("MODAL_VOLUME") or "krmhd-benchmark-vol",
        volume_root=(cfg.get("MODAL_VOLUME_ROOT") or "study04").strip("/") or "study04",
        modal_bin=cfg.get("MODAL_BIN") or str(repo / ".venv" / "bin" / "modal"),
        loop_name=cfg.get("LOOP_GIT_NAME") or "krmhd-loop",
    )


# ---------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------


def _git_env() -> Dict[str, str]:
    """The environment for git: it must never prompt for credentials."""
    env = dict(os.environ)
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    return env


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git in ``repo``. Never prompts (an unattended run must fail, not hang)."""
    try:
        proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                              env=_git_env(), timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvError(f"git {' '.join(args)}: {exc}") from None
    if check and proc.returncode != 0:
        raise EnvError(f"git {' '.join(args)} failed ({proc.returncode}): "
                       f"{short(proc.stderr or proc.stdout)}")
    return proc


def rev(repo: Path, ref: str) -> Optional[str]:
    """The commit sha ``ref`` names, or None."""
    proc = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def committed(repo: Path, rel: str) -> bool:
    """True if ``rel`` exists in the commit HEAD points to."""
    return git(repo, "cat-file", "-e", f"HEAD:{rel}", check=False).returncode == 0


def require_on_main(repo: Path) -> None:
    """Refuse unless the repo is on branch main."""
    branch = git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != "main":
        raise Refusal("branch", f"the repo is on {branch!r}; the launcher works only on main")


def require_clean(repo: Path) -> None:
    """Refuse unless the working tree is clean (untracked files count)."""
    out = git(repo, "status", "--porcelain").stdout.strip()
    if out:
        lines = out.splitlines()
        raise Refusal("clean-tree", f"the working tree is not clean ({len(lines)} entries; "
                      f"untracked files count): {'; '.join(lines[:5])}. Commit and push first, "
                      "so the runs use code that is in the repo.")


def fetch_origin(repo: Path) -> None:
    """git fetch origin; a failure is a refusal, since the launcher must see origin."""
    proc = git(repo, "fetch", "--quiet", "origin", check=False)
    if proc.returncode != 0:
        raise Refusal("fetch", f"git fetch origin failed: {short(proc.stderr or proc.stdout)}")


def ahead_behind(repo: Path) -> Tuple[int, int]:
    """(commits ahead of origin/main, commits behind it)."""
    out = git(repo, "rev-list", "--left-right", "--count", "HEAD...origin/main").stdout.split()
    return int(out[0]), int(out[1])


def require_pushed(repo: Path) -> str:
    """Fetch, then require HEAD == origin/main. Returns the HEAD sha."""
    fetch_origin(repo)
    head = rev(repo, "HEAD")
    remote = rev(repo, "origin/main")
    if head is None or remote is None:
        raise Refusal("pushed", "HEAD or origin/main does not exist")
    if head != remote:
        ahead, behind = ahead_behind(repo)
        raise Refusal("pushed", f"HEAD {head[:12]} is not origin/main {remote[:12]} (ahead {ahead}, "
                      f"behind {behind}). Push or pull first: a reservation must be committed on "
                      "top of what is on origin.")
    return head


def require_no_stop(repo: Path) -> None:
    """Refuse while STOP exists."""
    if (repo / lc.STOP_REL).exists():
        raise Refusal("stop", f"{lc.STOP_REL} exists: the loop is stopped and launches nothing "
                      "until Anjor removes it")


def commit_ledger(repo: Path, message: str) -> str:
    """Commit the ledger file and nothing else. Returns the new HEAD sha."""
    rel = lc.LEDGER_REL
    git(repo, "add", "--", rel)
    git(repo, "commit", "--quiet", "-m", message, "--", rel)
    sha = rev(repo, "HEAD")
    assert sha is not None
    return sha


def push_main(repo: Path, expect: Optional[str] = None) -> Tuple[bool, str]:
    """Push main. If git reports failure, ask the remote whether ``expect`` landed anyway."""
    proc = git(repo, "push", "--quiet", "origin", "main", check=False)
    if proc.returncode == 0:
        return True, ""
    err = short(proc.stderr or proc.stdout)
    if expect:
        probe = git(repo, "ls-remote", "origin", "refs/heads/main", check=False)
        if probe.returncode == 0 and probe.stdout.split()[:1] == [expect]:
            git(repo, "fetch", "--quiet", "origin", check=False)
            return True, ""
    return False, err


@contextlib.contextmanager
def launcher_lock(repo: Path) -> Iterator[None]:
    """One ledger-writing command at a time per clone: flock(2) on the git directory itself.

    Without it two launches started together could each save a ledger that lacks the
    other's reservation. Locking the directory leaves no file behind, git does not use
    flock, and the lock is released when the process exits, however it exits.
    """
    gitdir = Path(git(repo, "rev-parse", "--absolute-git-dir").stdout.strip())
    fd = os.open(str(gitdir), os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refusal("lock", "another modal_launch.py command is running in this clone; "
                          "wait for it to finish") from None
        yield
    finally:
        os.close(fd)


class SignalGuard:
    """SIGINT and SIGTERM handling while the ledger is written and while calls are started.

    Installed for the whole of a launch, from the reservation to its record. While
    ``raising`` is False a signal is only recorded, so that a ledger write, commit and push
    in progress finish: a launch cut off between its reservation and its record leaves
    hours that only Anjor can release. While the calls are being started ``raising`` is
    True and a signal raises KeyboardInterrupt, so that no further call starts and the
    calls already started are recorded. The runner stops a session with SIGTERM.
    """

    def __init__(self) -> None:
        """Nothing received yet; signals raise only inside ``raising_signals``."""
        self.received: List[int] = []
        self.raising = False
        self._saved: Dict[int, Any] = {}

    def _handler(self, signum: int, frame: Any) -> None:
        """Note the signal; raise KeyboardInterrupt while calls are being started."""
        self.received.append(signum)
        if self.raising:
            raise KeyboardInterrupt(f"signal {signum}")

    def __enter__(self) -> "SignalGuard":
        """Install the handler for SIGINT and SIGTERM."""
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._saved[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):  # not the main thread
                pass
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        """Put the previous handlers back."""
        for sig, old in self._saved.items():
            signal.signal(sig, old)
        return False

    @contextlib.contextmanager
    def raising_signals(self) -> Iterator[None]:
        """Let a signal interrupt the block (as KeyboardInterrupt)."""
        self.raising = True
        try:
            yield
        finally:
            self.raising = False


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def ledger_file(repo: Path) -> Path:
    """The ledger's path in ``repo``."""
    return repo / lc.LEDGER_REL


def read_ledger(repo: Path) -> Dict[str, Any]:
    """The working-tree ledger (empty if absent), without the integrity check."""
    try:
        ledger = lc.load_ledger(ledger_file(repo))
    except (OSError, ValueError) as exc:
        raise EnvError(f"cannot read {lc.LEDGER_REL}: {exc}") from None
    if not isinstance(ledger, dict):
        raise EnvError(f"{lc.LEDGER_REL} is not a JSON object")
    return ledger


def read_verified_ledger(repo: Path) -> Dict[str, Any]:
    """The ledger, refused unless verify_ledger passes.

    Every write re-stamps the integrity hash, so writing on top of a hand edit would
    launder it. Hence no command writes a ledger that fails verification.
    """
    try:
        ledger = lc.load_ledger(ledger_file(repo))
    except (OSError, ValueError) as exc:
        raise Refusal("ledger", f"cannot read {lc.LEDGER_REL}: {exc}") from None
    if not isinstance(ledger, dict):
        raise Refusal("ledger", f"{lc.LEDGER_REL} is not a JSON object")
    problems = lc.verify_ledger(ledger)
    if problems:
        raise Refusal("ledger", "verify_ledger failed: " + "; ".join(problems) + ". Only this "
                      "script writes the ledger; a hand edit is a hard stop for Anjor.")
    return ledger


def trusted_committer(name: str, email: str, loop_name: str) -> bool:
    """True for a commit Anjor made himself: its committer is neither the loop nor GitHub.

    The runner sets the loop's committer name for every session and the loop cannot set
    another (integrator decision 1). GitHub is the committer of the commits it makes for a
    web merge, a squash or rebase merge and a web edit, and the loop's gh login can ask for
    those too, so a GitHub-made commit (or a bot's) never counts as Anjor's here.
    """
    if name == loop_name or name == "GitHub" or email.strip().lower() in _GITHUB_COMMITTER_EMAILS:
        return False
    return "[bot]" not in name.lower() and "[bot]" not in email.lower()


def _cat_objects(repo: Path, names: Sequence[str]) -> Dict[str, Optional[bytes]]:
    """The content of each git object name (``<sha>:<path>`` or a blob id); None if missing.

    One ``git cat-file --batch`` for all of them, because the history can be long.
    """
    if not names:
        return {}
    query = "".join(f"{name}\n" for name in names).encode("utf-8")
    try:
        proc = subprocess.run(["git", "-C", str(repo), "--no-replace-objects", "cat-file", "--batch"],
                              input=query, capture_output=True, env=_git_env(), timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EnvError(f"git cat-file --batch: {exc}") from None
    if proc.returncode != 0:
        raise EnvError(f"git cat-file --batch failed ({proc.returncode}): "
                       f"{short(proc.stderr.decode('utf-8', 'replace'))}")
    out, pos, blobs = proc.stdout, 0, {}
    try:
        for name in names:
            end = out.index(b"\n", pos)
            header = out[pos:end].decode("utf-8", "replace")
            pos = end + 1
            if header.endswith(" missing"):
                blobs[name] = None
                continue
            size = int(header.split()[2])
            blobs[name] = out[pos:pos + size]
            pos += size + 1
    except (ValueError, IndexError):
        raise EnvError("cannot parse the output of git cat-file --batch") from None
    return blobs


@dataclass
class GitHistory:
    """Every commit reachable from HEAD, with its parents and committer, newest first.

    ``git log -- <path>`` follows a single parent through a merge and prints no file names
    for a merge, so a merge can hide a change from it: for example a loop merge whose first
    parent carries an old ledger, or one that edits loop/config.env while merging. The
    launcher's history checks walk this whole graph instead, both sides of every merge.
    """

    head: str
    order: List[str]  # newest first, parents after their children
    parents: Dict[str, Tuple[str, ...]]
    committer: Dict[str, Tuple[str, str]]  # sha -> (name, email)

    @classmethod
    def load(cls, repo: Path) -> "GitHistory":
        """Read the graph with one ``git log``; refuse a shallow clone or grafts, which hide history."""
        if git(repo, "rev-parse", "--is-shallow-repository").stdout.strip() == "true":
            raise Refusal("history", "the clone is shallow, so the launcher cannot see the whole history of "
                          "the ledger and of loop/. Fetch the full history first.")
        grafts = git(repo, "rev-parse", "--git-path", "info/grafts").stdout.strip()
        if grafts and (Path(grafts) if os.path.isabs(grafts) else repo / grafts).exists():
            raise Refusal("history", f"{grafts} exists. Grafts change the history git shows, so the "
                          "launcher cannot trust it; only Anjor can remove the file.")
        out = git(repo, "--no-replace-objects", "log", "--topo-order", "--format=%H%x1f%P%x1f%cn%x1f%ce",
                  "HEAD").stdout
        order: List[str] = []
        parents: Dict[str, Tuple[str, ...]] = {}
        committer: Dict[str, Tuple[str, str]] = {}
        for line in out.splitlines():
            fields = line.split("\x1f")
            if len(fields) != 4 or not _SHA40.match(fields[0]):
                continue
            sha = fields[0]
            order.append(sha)
            parents[sha] = tuple(fields[1].split())
            committer[sha] = (fields[2], fields[3])
        if not order:
            raise EnvError("git log printed no commits")
        return cls(head=order[0], order=order, parents=parents, committer=committer)

    def trusted(self, sha: str, loop_name: str) -> bool:
        """Did Anjor make this commit himself (see trusted_committer)?"""
        name, email = self.committer.get(sha, ("", ""))
        return trusted_committer(name, email, loop_name)

    def who(self, sha: str) -> str:
        """'<short sha> (committer <name>)' for messages."""
        return f"{sha[:12]} (committer {self.committer.get(sha, ('?', ''))[0]})"

    def ancestors(self, starts: Iterable[str]) -> Set[str]:
        """``starts`` and every commit reachable from them."""
        seen: Set[str] = set()
        stack = list(starts)
        while stack:
            sha = stack.pop()
            if sha in seen or sha not in self.parents:
                continue
            seen.add(sha)
            stack.extend(self.parents[sha])
        return seen

    def blob_ids(self, repo: Path, paths: Sequence[str]) -> Dict[str, Dict[str, Optional[str]]]:
        """For each path, the object id of its content in every commit (None where absent).

        One ``git cat-file --batch-check`` for all commits and paths.
        """
        if not paths:
            return {}
        query = "".join(f"{sha}:{path}\n" for path in paths for sha in self.order).encode("utf-8")
        try:
            proc = subprocess.run(["git", "-C", str(repo), "--no-replace-objects", "cat-file", "--batch-check"],
                                  input=query, capture_output=True, env=_git_env(), timeout=600)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise EnvError(f"git cat-file --batch-check: {exc}") from None
        lines = proc.stdout.decode("utf-8", "replace").splitlines()
        if proc.returncode != 0 or len(lines) != len(paths) * len(self.order):
            raise EnvError(f"git cat-file --batch-check failed ({proc.returncode}): "
                           f"{short(proc.stderr.decode('utf-8', 'replace'))}")
        result: Dict[str, Dict[str, Optional[str]]] = {}
        rows = iter(lines)
        for path in paths:
            result[path] = {sha: (None if line.endswith(" missing") else line.split()[0])
                            for sha, line in zip(self.order, rows)}
        return result


def _ledger_from_blob(blob: bytes, sha: str) -> Dict[str, Any]:
    """A committed ledger version as a dict; refuse anything but a JSON object."""
    try:
        data = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise Refusal("ledger", f"{lc.LEDGER_REL} in commit {sha[:12]} is not JSON") from None
    if not isinstance(data, dict):
        raise Refusal("ledger", f"{lc.LEDGER_REL} in commit {sha[:12]} is not a JSON object")
    return data


def check_ledger_succession(repo: Path, settings: Settings, history: GitHistory) -> None:
    """Refuse unless HEAD's ledger keeps everything recorded since Anjor's last ledger commit.

    An old ledger version has a valid integrity hash too, so a rollback would otherwise free
    headroom under the cap. Anjor's own ledger commits are trusted: a plain (not merge)
    commit with a trusted committer (``trusted_committer``) that changes the ledger. That is
    how he resets the ledger after an incident, and it vouches for everything before it. A
    merge does not count, so a routine pull with a merge in his checkout vouches for
    nothing. Every ledger version committed since, by any commit that is not an ancestor of
    such a commit, on either side of a merge, must be one that HEAD's ledger extends
    (lc.ledger_extends), and each one a loop or GitHub commit made must pass verify_ledger.
    So a rollback is caught however it is spread over commits, and a merge that drops the
    side carrying the charges too. The runner checks the loop's commits after the session;
    this check keeps the cap intact at the one place money is spent.
    """
    rel = lc.LEDGER_REL
    blob = history.blob_ids(repo, [rel])[rel]
    loop = settings.loop_name

    def changed(sha: str) -> bool:
        """Does the ledger in ``sha`` differ from that of any of its parents?"""
        ps = history.parents[sha]
        if not ps:
            return blob[sha] is not None
        return any(blob[sha] != blob.get(p) for p in ps)

    anchors = [c for c in history.order
               if len(history.parents[c]) <= 1 and history.trusted(c, loop) and changed(c)]
    vouched = history.ancestors(anchors)
    below = history.ancestors(p for a in anchors for p in history.parents[a])
    since = [c for c in history.order if c not in vouched and changed(c)]
    sources: Dict[str, List[str]] = {}  # version (blob id) -> the commits that wrote it, newest first
    for c in since + [a for a in anchors if a not in below]:
        if blob[c] is not None:
            sources.setdefault(blob[c], []).append(c)
    if not sources:
        return
    head_blob = blob[history.head]
    if head_blob is None:
        deleted = [history.who(c) for c in since if blob[c] is None]
        raise Refusal("ledger", f"{rel} is missing at HEAD ({history.head[:12]}), but commits since Anjor's "
                      f"last ledger commit recorded launches; it was deleted in {', '.join(deleted[:3]) or 'a merge'}. "
                      f"{_RESET_HOW}; this is a hard stop.")
    texts = _cat_objects(repo, list(sources) + ([] if head_blob in sources else [head_blob]))
    if texts.get(head_blob) is None:
        raise EnvError(f"git cannot read the ledger at HEAD ({head_blob[:12]})")
    head_ledger = _ledger_from_blob(texts[head_blob] or b"", history.head)
    head_by = history.who(sources[head_blob][0]) if head_blob in sources else "a commit Anjor's ledger commit covers"
    for version_id, commits in sources.items():
        content = texts.get(version_id)
        if content is None:
            raise EnvError(f"git cannot read the ledger version {version_id[:12]}")
        version = _ledger_from_blob(content, commits[0])
        if any(not history.trusted(c, loop) for c in commits):
            problems = lc.verify_ledger(version)
            if problems:
                raise Refusal("ledger", f"the ledger committed in {history.who(commits[0])} fails verify_ledger: "
                              f"{'; '.join(problems[:6])}. Only this script writes the ledger; a hand edit is a "
                              "hard stop for Anjor.")
        problems = lc.ledger_extends(version, head_ledger)
        if problems:
            raise Refusal("ledger", f"the ledger at HEAD ({history.head[:12]}, written by {head_by}) is not a "
                          f"launcher-made successor of the version committed in {history.who(commits[0])}: "
                          f"{'; '.join(problems[:6])}. A rolled-back or edited ledger is a hard stop: "
                          f"{_RESET_HOW}.")


def check_loop_files(repo: Path, settings: Settings, history: GitHistory) -> None:
    """Refuse while the content of any file under loop/ comes from a commit Anjor did not make.

    loop/ holds the cap (config.env), the frozen manifest, the GPU function (modal_app.py)
    and this launcher. Only Anjor changes them, and the runner writes STOP after a session
    whose commits touch them; this closes the window between such a commit and the end of
    its session. For each path (deleted ones too), the walk goes back from HEAD through
    every parent that has the same content, to the commits that produced that content. If
    one of them is a loop or GitHub commit, the path is refused, including through a merge.
    A path that is absent at HEAD counts only if Anjor's commits ever had it. The refusal
    holds until Anjor commits the file himself with other content (a revert, or any edit).
    """
    loop = settings.loop_name
    listed = git(repo, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", lc.LOOP_REL).stdout
    listed += git(repo, "--no-replace-objects", "log", "-z", "-m", "--full-history", "--no-renames", "--format=",
                  "--name-only", "HEAD", "--", lc.LOOP_REL).stdout
    names = {n for n in listed.split("\0") if n}
    flagged: List[str] = [f"{n!r} (a file name with a line break, which the launcher cannot check)"
                          for n in sorted(names) if "\n" in n]
    paths = sorted(n for n in names if "\n" not in n)
    blobs = history.blob_ids(repo, paths)
    for path in paths:
        blob = blobs[path]
        current = blob[history.head]
        if current is None and not any(blob[c] is not None and history.trusted(c, loop) for c in history.order):
            continue  # only loop commits ever had it: its absence restores Anjor's state
        origins: List[str] = []
        seen: Set[str] = set()
        stack = [history.head]
        while stack:
            sha = stack.pop()
            if sha in seen:
                continue
            seen.add(sha)
            same = [p for p in history.parents.get(sha, ()) if p in blob and blob[p] == current]
            if same:
                stack.extend(same)
            else:
                origins.append(sha)
        bad = [c for c in origins if not history.trusted(c, loop)]
        if bad:
            what = "deleted" if current is None else "content"
            flagged.append(f"{path} ({what} from {history.who(bad[0])})")
    if flagged:
        raise Refusal("guarded-files", f"{'; '.join(flagged[:5])}. Only Anjor changes loop/ (the cap, the GPU "
                      "function and this launcher live there), so nothing launches until he commits each file "
                      "himself, reverted or edited, in his own checkout (content GitHub committed, by a squash "
                      "merge or a web edit, does not count). This is a hard stop.")


def next_launch_id(ledger: Dict[str, Any]) -> str:
    """The next launch ID: L001, L002, ... (one more than the highest so far)."""
    numbers = [int(m.group(1)) for launch in ledger.get("launches", [])
               for m in [_LAUNCH_ID_RE.match(str(launch.get("launch_id", "")))] if m]
    return f"L{max(numbers, default=0) + 1:03d}"


def slug(text: str) -> str:
    """Lower case, other characters as '-', at most 40 characters."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40].strip("-") or "x"


def app_name_for(prefix: str, launch_id: str, run_set: str) -> str:
    """The Modal app name of a launch: <prefix>-<launch id>-<run set>."""
    return f"{prefix}-{launch_id.lower()}-{slug(run_set)}"


def cap_excess(cap: float, ledger: Dict[str, Any]) -> Optional[str]:
    """A message if hours used plus hours reserved exceed the cap, else None."""
    t = lc.ledger_totals(ledger)
    total = t["used"] + t["reserved"]
    if total > cap + 1e-9:
        return (f"CAP EXCEEDED: used {t['used']:.2f} + reserved {t['reserved']:.2f} = {total:.2f} "
                f"A100-h is above the cap of {cap:.1f} A100-h ({lc.CONFIG_REL}). Runs charged or held "
                "beyond their reservation did this: their attempt records show more GPU time than their "
                "budget. It is a hard stop: only Anjor decides about more compute, and the runner writes "
                "STOP after this session.")
    return None


def check_cap(cap: float, ledger: Dict[str, Any], new_hours: float) -> None:
    """Refuse if hours used + reserved + ``new_hours`` would exceed the cap."""
    t = lc.ledger_totals(ledger)
    total = t["used"] + t["reserved"] + new_hours
    if total > cap + 1e-9:
        raise Refusal("cap", f"used {t['used']:.2f} + reserved {t['reserved']:.2f} + this launch "
                      f"{new_hours:.2f} = {total:.2f} A100-h would exceed the cap of {cap:.1f} "
                      f"A100-h ({lc.CONFIG_REL}). A launch over the cap is a hard stop: only "
                      "Anjor changes the cap.")


# ---------------------------------------------------------------------------
# Launch preconditions
# ---------------------------------------------------------------------------


@dataclass
class ConfigInfo:
    rel: str
    sha256: str
    label: str
    entrypoint: str


def to_repo_rel(repo: Path, raw: str, rule: str) -> str:
    """A command-line path as a normalised repo-relative POSIX path (refuse if outside)."""
    p = Path(raw).expanduser()
    if p.is_absolute():
        try:
            rel = p.resolve().relative_to(repo).as_posix()
        except ValueError:
            raise Refusal(rule, f"{raw} is outside the repo {repo}") from None
    else:
        rel = os.path.normpath(raw).replace(os.sep, "/")
    if rel == ".." or rel.startswith("../") or rel.startswith("/") or rel == ".":
        raise Refusal(rule, f"{raw} is outside the repo")
    return rel


def _inside(repo: Path, rel: str, folder_rel: str) -> bool:
    """True if repo/rel, symlinks resolved, lies inside repo/folder_rel."""
    try:
        (repo / rel).resolve().relative_to((repo / folder_rel).resolve())
        return True
    except ValueError:
        return False


def _yaml() -> Any:
    """The yaml module (PyYAML), or an environment error."""
    try:
        import yaml
    except ImportError:
        raise EnvError("PyYAML is not installed in this environment (run uv sync)") from None
    return yaml


def check_config(repo: Path, raw: str) -> ConfigInfo:
    """A launchable config: committed YAML under configs/ whose entrypoint defines run()."""
    rel = to_repo_rel(repo, raw, "config")
    if not rel.startswith(CONFIGS_REL + "/") or not _inside(repo, rel, CONFIGS_REL):
        raise Refusal("config", f"{rel} is not under {CONFIGS_REL}/")
    if not rel.endswith((".yaml", ".yml")):
        raise Refusal("config", f"{rel} is not a .yaml/.yml file")
    path = repo / rel
    if path.is_symlink() or not path.is_file():
        raise Refusal("config", f"{rel} does not exist (or is a symlink)")
    if not committed(repo, rel):
        raise Refusal("config", f"{rel} is not committed at HEAD; configs are committed before launch")
    label = Path(rel).stem
    if not _LABEL_RE.match(label):
        raise Refusal("config", f"{rel}: the file name must match {_LABEL_RE.pattern} (it becomes the run label)")
    try:
        cfg = _yaml().safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # yaml.YAMLError, UnicodeDecodeError
        raise Refusal("config", f"{rel} does not parse as YAML: {short_exc(exc)}") from None
    if not isinstance(cfg, dict):
        raise Refusal("config", f"{rel} is not a YAML mapping")
    entry = cfg.get("entrypoint")
    if not isinstance(entry, str) or not entry.strip():
        raise Refusal("config", f"{rel} has no string key 'entrypoint' (the run code, a repo-relative .py)")
    entry_rel = to_repo_rel(repo, entry.strip(), "config")
    if entry != entry_rel:
        raise Refusal("config", f"{rel}: write the entrypoint {entry!r} as the normalised repo-relative "
                      f"path {entry_rel!r}. The GPU function joins it, as written, to the repo copy in "
                      "the container, so any other spelling would fail only after the GPU has started.")
    if (not entry_rel.startswith(lc.STUDY_REL + "/") or not entry_rel.endswith(".py")
            or not _inside(repo, entry_rel, lc.STUDY_REL)):
        raise Refusal("config", f"{rel}: entrypoint {entry_rel} is not a .py file under {lc.STUDY_REL}/")
    entry_path = repo / entry_rel
    if not entry_path.is_file() or not committed(repo, entry_rel):
        raise Refusal("config", f"{rel}: entrypoint {entry_rel} is missing or not committed")
    if not _RUN_DEF_RE.search(entry_path.read_text(encoding="utf-8", errors="replace")):
        raise Refusal("config", f"{rel}: entrypoint {entry_rel} has no line 'def run(' "
                      "(the interface is run(cfg, out_dir, ctx))")
    return ConfigInfo(rel=rel, sha256=lc.sha256_file(path), label=label, entrypoint=entry_rel)


def _label_key(line: str) -> str:
    """A report line without leading Markdown decoration, lower case, for finding labels."""
    return line.strip().lstrip(_LINE_DECORATION).lower()


def _table_cells(line: str) -> List[str]:
    """The cells of a Markdown table row."""
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def results_table_problems(lines: Sequence[str]) -> List[str]:
    """Problems with the report's results tables: a table with no rows, or a row that fails.

    A results table is one whose header has a Pass column ('| Quantity | Value | Threshold
    | Pass |' in LOOP.md). Every row's Pass cell must be 'yes' or 'PASS', so that a failed
    quantity cannot sit under 'Coded check: PASS'. A row whose Threshold and Pass cells are
    both 'n/a' or '-' only informs, and has nothing to fail. A report without such a table
    is not refused for that alone: the coded check line decides, and a frozen evaluation
    that writes no table must not block its run set for good.
    """
    problems: List[str] = []
    i = 0
    while i < len(lines):
        header = lines[i].strip()
        cells = _table_cells(header) if header.startswith("|") else []
        lowered = [c.strip("*_` ").lower() for c in cells]
        col = next((k for k, c in enumerate(lowered) if c in _PASS_HEADERS), None)
        if col is None or i + 1 >= len(lines) or not re.match(r"^\|?\s*:?-{3,}", lines[i + 1].strip()):
            i += 1
            continue
        threshold = next((k for k, c in enumerate(lowered) if c.startswith("threshold")), None)
        rows = 0
        i += 2
        while i < len(lines) and lines[i].strip().startswith("|"):
            row = _table_cells(lines[i])
            rows += 1
            value = row[col].strip("*_` ").lower() if col < len(row) else ""
            limit = row[threshold].strip("*_` ").lower() if threshold is not None and threshold < len(row) else None
            informs = value in _NA_CELLS and limit in _NA_CELLS
            if len(row) != len(cells) or (value not in _PASS_CELLS and not informs):
                problems.append(f"the results table row {short(lines[i].strip(), 100)!r} does not pass "
                                "(its Pass cell must be 'yes' or 'PASS')")
            i += 1
        if rows == 0:
            problems.append("the results table has no rows")
    return problems


def gate_report_problems(text: str) -> List[str]:
    """Why a gate report does not show a passed gate (empty if it does).

    The report must be in the form LOOP.md gives and the gate code writes:
    - exactly one 'Result:' line, and it reads 'Result: PASS';
    - exactly one 'Coded check:' line, 'Coded check: PASS';
    - exactly one 'Critic:' line, 'Critic: VERDICT: SUPPORTED, <iteration id>, <critic report
      file>', with nothing between the verdict and the comma and no other verdict named;
    - exactly one 'Kill criteria:' line, 'Kill criteria: none met';
    - no results table without rows, and no row of one that fails (results_table_problems);
    - no unfilled template text ('PASS | FAIL', or a placeholder such as '<iteration id>').
    Lines are counted by their label after Markdown decoration and in any case, so 'Final
    result: FAIL', 'Result (rerun): FAIL' or 'Coded check (dt fit): FAIL' count as a second line.
    """
    lines = text.splitlines()
    problems: List[str] = []
    found: Dict[str, Optional[str]] = {}
    for label, pattern in _VERDICT_LABELS:
        matches = [ln.rstrip() for ln in lines if pattern.match(_label_key(ln))]
        if len(matches) != 1:
            problems.append(f"{len(matches)} lines start with {label!r} (exactly one must)")
        found[label] = matches[0] if len(matches) == 1 else None
    result, coded, critic, kill = (found[label] for label, _ in _VERDICT_LABELS)
    if result is not None and result != "Result: PASS":
        problems.append(f"the Result line is {short(result, 80)!r}, not 'Result: PASS'")
    if coded is not None and coded != "Coded check: PASS":
        problems.append(f"the Coded check line is {short(coded, 80)!r}, not 'Coded check: PASS'")
    if critic is not None and (not _CRITIC_LINE_RE.match(critic) or _OTHER_VERDICT_RE.search(critic)):
        problems.append(f"the Critic line is {short(critic, 140)!r}; it must read 'Critic: VERDICT: SUPPORTED, "
                        "<iteration id>, <critic report file>', copying the critic's verdict line")
    if kill is not None and kill != "Kill criteria: none met":
        problems.append(f"the Kill criteria line is {short(kill, 100)!r}, not 'Kill criteria: none met'")
    problems += results_table_problems(lines)
    unfilled = ["PASS | FAIL"] if "PASS | FAIL" in text else []
    unfilled += [p for p in GATE_TEMPLATE_PLACEHOLDERS if p in text]
    for line in lines:
        if _label_key(line).startswith(_GATE_HEADER_KEYS):
            unfilled += [m for m in _PLACEHOLDER_RE.findall(line) if m not in unfilled]
    if unfilled:
        problems.append("unfilled template text: " + ", ".join(repr(u) for u in unfilled[:6]))
    return problems


def critic_verdict(text: str) -> Optional[str]:
    """The VERDICT line of a saved critic review (LOOP.md section 6): the first line of its
    'Report:' section, without Markdown decoration. None if there is no such section."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        key = line.strip().lstrip(_LINE_DECORATION)
        if not key.lower().startswith("report:"):
            continue
        rest = key[len("report:"):]
        for candidate in [rest] + lines[i + 1:]:
            if candidate.strip().startswith("```"):
                continue  # the report may sit in a fenced block
            value = candidate.strip().strip("`*_>").strip()
            if value:
                if value.startswith("VERDICT:"):
                    return value
                break
    return None


def gate_report_kind(rel: str) -> Optional[Tuple[str, str]]:
    """(kind, iteration id) from a report's file name: kind G1, G2, G3, Gbase or G4_<S>."""
    m = _GATE_FILE_RE.match(PurePosixPath(rel).name)
    if not m:
        return None
    kind = f"G{m.group('n')}" if m.group("n") else (f"G4_{m.group('set')}" if m.group("set") else "Gbase")
    return kind, m.group("iter")


def _expected_title(kind: str, iteration: str) -> str:
    """The title line LOOP.md gives a report: '# Gate <n> report[, set <S>]: <iteration id>'."""
    if kind.startswith("G4_"):
        return f"# Gate 4 report, set {kind[3:]}: {iteration}"
    return f"# Gate {kind[1:]} report: {iteration}"


def title_fits(kind: str, iteration: str, title: str) -> bool:
    """Does the report's first line name the gate its file name says, and nothing else?

    It is a Markdown title that names the iteration; for G<n> it names 'Gate <n>', for
    G4_<S> 'Gate 4' and 'set <S>', and for Gbase the base state ('base'), and it names no
    other gate number. So a report copied under another gate's file name is refused, while
    the wording around it, written once by the gate code and perhaps frozen, may differ.
    """
    if not title.startswith("# ") or iteration not in title:
        return False
    numbers = set(re.findall(r"\bgate\s+([0-9]+)\b", title, re.I))
    if kind == "Gbase":
        return not numbers and re.search(r"\bbase\b", title, re.I) is not None
    if kind.startswith("G4_"):
        return numbers == {"4"} and re.search(rf"\b(?i:set)\s+{re.escape(kind[3:])}\b", title) is not None
    return numbers == {kind[1:]}


def gate_report_file_problems(repo: Path, rel: str, text: str) -> List[str]:
    """What the file name, the title and the critic report it names say about a report.

    The name must be G<n>_<iteration id>.md, Gbase_<iteration id>.md or
    G4_<S>_<iteration id>.md, and the first line a title that fits it (title_fits; LOOP.md's
    form is '# Gate <n> report[, set <S>]: <iteration id>', and '# Gate base report:
    <iteration id>' for the base-state gate). The Critic line must name a committed
    gate_reports/critic_<iteration id>_<k>.md whose report starts 'VERDICT: SUPPORTED'.
    """
    kind = gate_report_kind(rel)
    if kind is None:
        return [f"the file name is not G<n>_<iteration id>.md, Gbase_<iteration id>.md or "
                f"G4_<S>_<iteration id>.md (LOOP.md, 'Gate reports')"]
    problems: List[str] = []
    title = next((ln.rstrip() for ln in text.splitlines() if ln.strip()), "")
    if not title_fits(kind[0], kind[1], title):
        problems.append(f"its title is {short(title, 80)!r}, which does not fit its file name (LOOP.md's form: "
                        f"{_expected_title(*kind)!r})")
    critic = next((ln.rstrip() for ln in text.splitlines() if _CRITIC_LINE_RE.match(ln.rstrip())), None)
    if critic is None:
        return problems  # gate_report_problems has said why
    m = _CRITIC_LINE_RE.match(critic)
    assert m is not None
    raw = m.group("file")
    critic_rel = raw if raw.startswith(lc.STUDY_REL + "/") else (
        f"{lc.STUDY_REL}/{raw}" if "/" in raw else f"{GATE_REPORTS_REL}/{raw}")
    critic_rel = os.path.normpath(critic_rel).replace(os.sep, "/")
    name_ok = re.match(rf"^critic_{re.escape(m.group('iter'))}_\d+\.md$", PurePosixPath(critic_rel).name)
    if not critic_rel.startswith(GATE_REPORTS_REL + "/") or not name_ok:
        problems.append(f"the Critic line names {raw!r}, not {GATE_REPORTS_REL}/critic_{m.group('iter')}_<k>.md")
    elif not (repo / critic_rel).is_file() or not committed(repo, critic_rel):
        problems.append(f"the critic report {critic_rel} it names is not committed")
    else:
        verdict = critic_verdict((repo / critic_rel).read_text(encoding="utf-8", errors="replace"))
        if verdict != "VERDICT: SUPPORTED":
            problems.append(f"the critic report {critic_rel} it names says {verdict or 'no VERDICT line'!r} "
                            "after 'Report:', not 'VERDICT: SUPPORTED'")
    return problems


def check_gate_report(repo: Path, raw: str) -> str:
    """A committed report under gate_reports/ that shows a passed gate. Returns its path."""
    rel = to_repo_rel(repo, raw, "gate-report")
    if not rel.startswith(GATE_REPORTS_REL + "/") or not _inside(repo, rel, GATE_REPORTS_REL):
        raise Refusal("gate-report", f"{rel} is not under {GATE_REPORTS_REL}/")
    path = repo / rel
    if not path.is_file():
        raise Refusal("gate-report", f"{rel} does not exist")
    if not committed(repo, rel):
        raise Refusal("gate-report", f"{rel} is not committed at HEAD")
    text = path.read_text(encoding="utf-8", errors="replace")
    problems = gate_report_problems(text) + gate_report_file_problems(repo, rel, text)
    if problems:
        raise Refusal("gate-report", f"{rel} does not show a passed gate: {'; '.join(problems)}. The "
                      "launcher accepts a report only in the form of LOOP.md, written by the gate code.")
    return rel


def _committed_reports(repo: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Committed gate reports by kind: {kind: [(iteration id, path), ...]}, oldest first."""
    out = git(repo, "ls-tree", "-r", "--name-only", "HEAD", "--", GATE_REPORTS_REL + "/").stdout
    reports: Dict[str, List[Tuple[str, str]]] = {}
    for rel in out.split("\n"):
        kind = gate_report_kind(rel) if rel else None
        if kind:
            reports.setdefault(kind[0], []).append((kind[1], rel))
    for items in reports.values():
        items.sort()
    return reports


def _result_lines(text: str) -> List[str]:
    """Every line of a report that the 'Result:' label matches."""
    pattern = dict(_VERDICT_LABELS)["Result:"]
    return [ln.rstrip() for ln in text.splitlines() if pattern.match(_label_key(ln))]


def check_gate_reports(repo: Path, raws: Sequence[str], run_set: str) -> List[str]:
    """The reports a launch of ``run_set`` rests on (LOOP.md section 9). Returns their paths.

    Every report given must pass (check_gate_report), and their kinds must fit the set:
    base, a Gate 1 or base-state gate report; A, a Gate 3 and a base-state gate report; B,
    set A's Gate 4 report; Gate 1 and 2 reports may come with any set. A quality gate's
    report (Gates 1 to 3), and for sets A and B the base-state gate's, must be the newest
    committed evaluation of that gate, since that is the one that stands. For Gate 4 the
    first evaluation that is not NOT DECIDED is the set's decision and is never repeated,
    so every other committed report of it must say 'Result: NOT DECIDED'.
    """
    rels: List[str] = []
    for raw in raws:
        rel = check_gate_report(repo, raw)
        if rel not in rels:
            rels.append(rel)
    if not rels:
        raise Refusal("gate-report", "no --gate-report given")
    groups, extra = _SET_GATES[run_set]
    allowed = {k for group in groups for k in group} | set(extra)
    kinds = {rel: gate_report_kind(rel)[0] for rel in rels}  # type: ignore[index]
    wrong = [f"{rel} ({kind})" for rel, kind in kinds.items() if kind not in allowed]
    missing = [" or ".join(group) for group in groups if not set(group) & set(kinds.values())]
    if wrong or missing:
        need = "; ".join(" or ".join(group) for group in groups)
        raise Refusal("gate-report", f"set {run_set} launches on these reports: {need} (LOOP.md section 9, "
                      f"'Before a launch'). " + (f"Not for this set: {', '.join(wrong)}. " if wrong else "")
                      + (f"Missing: {', '.join(missing)}." if missing else ""))
    committed_reports = _committed_reports(repo)
    for rel, kind in kinds.items():
        others = committed_reports.get(kind, [])
        if kind in ("G1", "G2", "G3") or (kind == "Gbase" and run_set != "base"):
            newest = others[-1][1] if others else rel
            if newest != rel:
                raise Refusal("gate-report", f"{rel} is not the newest {kind} report: {newest} is newer, and "
                              "the newest evaluation of a gate is the one that stands.")
        if kind.startswith("G4_"):
            for _iteration, other in others:
                if other == rel:
                    continue
                lines = _result_lines((repo / other).read_text(encoding="utf-8", errors="replace"))
                if lines != ["Result: NOT DECIDED"]:
                    said = short("; ".join(lines) or "no Result line", 80)
                    raise Refusal("gate-report", f"{other} also evaluated {kind} ({said}). The first evaluation "
                                  "that is not NOT DECIDED is the set's decision and is never repeated (LOOP.md "
                                  "section 5); this is a hard stop for Anjor.")
    return rels


def check_local_test(repo: Path, raw: str) -> str:
    """The committed record of the local smoke test. Returns its path."""
    rel = to_repo_rel(repo, raw, "local-test")
    path = repo / rel
    if not path.is_file():
        raise Refusal("local-test", f"{rel} does not exist")
    if not committed(repo, rel):
        raise Refusal("local-test", f"{rel} is not committed at HEAD (the record of the local "
                      "smoke test at small size)")
    return rel


def check_run_set(repo: Path, run_set: str, require_row: bool) -> None:
    """Refuse a set other than the plan's three and, with ``require_row``, one without a row in RUNS.md."""
    if run_set not in RUN_SETS:
        raise Refusal("set", f"run set {run_set!r} is not one of the plan's GPU run sets {', '.join(RUN_SETS)}. "
                      "A short calibration or timing run belongs to the set it serves. Another set would need "
                      "a change to this launcher, which is a change to the loop's guards and so a hard stop "
                      "(LOOP.md section 3).")
    if not require_row:
        return
    path = repo / RUNS_REL
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not re.search(rf"^\|\s*{re.escape(run_set)}\s*\|", text, re.M):
        raise Refusal("runs-row", f"{RUNS_REL} has no table row whose first cell is {run_set!r}. "
                      "The run set gets its row (with the estimate) before it launches.")


def resolve_freeze(repo: Path, keys: Sequence[str]) -> Dict[str, str]:
    """The sha256 of each ``--freeze`` key; refuse a key that does not resolve."""
    frozen: Dict[str, str] = {}
    for key in keys:
        sha = lc.frozen_key_hash(repo, key)
        if sha is None:
            raise Refusal("freeze", f"--freeze {key!r} does not resolve (file or heading missing)")
        frozen[key] = sha
    return frozen


def check_frozen_items(repo: Path, ledger: Dict[str, Any]) -> None:
    """Refuse while any frozen item (loop/frozen.json, or frozen by a launch) has changed."""
    try:
        manifest = lc.load_frozen_manifest(repo)
    except (OSError, ValueError) as exc:
        raise EnvError(f"cannot read {lc.FROZEN_REL}: {exc}") from None
    problems = lc.check_frozen(repo, manifest) + lc.check_frozen(repo, lc.ledger_frozen_entries(ledger))
    if problems:
        raise Refusal("frozen", "; ".join(problems) + ". A change to a frozen item is a hard stop "
                      "for Anjor.")


def _marker_applies(marker: str) -> bool:
    """Does a uv.lock resolution marker hold for the image (CPython 3.12 on Linux)?"""
    try:
        from packaging.markers import Marker
    except ImportError:  # crude fallback: entries split at 3.11 are the only known case
        return "< '3.11'" not in marker and '< "3.11"' not in marker
    env = {
        "python_version": IMAGE_PYTHON_FULL.rsplit(".", 1)[0],
        "python_full_version": IMAGE_PYTHON_FULL,
        "sys_platform": "linux",
        "platform_system": "Linux",
        "platform_machine": "x86_64",
        "os_name": "posix",
        "implementation_name": "cpython",
        "platform_python_implementation": "CPython",
    }
    try:
        return bool(Marker(marker).evaluate(env))
    except Exception:
        return True


def _version_key(version: str) -> Any:
    """A sort key for a version string (packaging's Version when available)."""
    try:
        from packaging.version import Version
        return (0, Version(version))
    except Exception:
        return (1, tuple(int(x) for x in re.findall(r"\d+", version)))


def lock_pins(repo: Path) -> Tuple[str, str]:
    """(gandalf commit, jax version) for the image, from uv.lock.

    The image must run the GANDALF the repo is pinned to. pyproject.toml may name a tag; if
    it names a 40-hex commit (directly or in [tool.uv.sources]), it must equal the lock.
    """
    if tomllib is None:
        raise EnvError("Python 3.11+ is needed to read uv.lock (tomllib)")
    lock_path = repo / "uv.lock"
    if not lock_path.is_file():
        raise Refusal("gandalf-pin", "uv.lock not found")
    try:
        data = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal("gandalf-pin", f"cannot parse uv.lock: {exc}") from None
    packages = [p for p in data.get("package", []) if isinstance(p, dict)]
    shas = set()
    for pkg in packages:
        if pkg.get("name") == GANDALF_PACKAGE:
            src = pkg.get("source") or {}
            m = re.search(r"#([0-9a-f]{40})\s*$", str(src.get("git", "")))
            if m:
                shas.add(m.group(1))
    if len(shas) != 1:
        raise Refusal("gandalf-pin", f"uv.lock does not give exactly one git commit for {GANDALF_PACKAGE} "
                      f"(found {sorted(shas) or 'none'})")
    sha = shas.pop()
    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        try:
            pdata = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise Refusal("gandalf-pin", f"cannot parse pyproject.toml: {exc}") from None
        pins = []
        for dep in pdata.get("project", {}).get("dependencies", []) or []:
            if re.match(rf"^\s*{re.escape(GANDALF_PACKAGE)}\b", str(dep)):
                m = re.search(r"@([0-9a-f]{40})\s*$", str(dep).strip())
                if m:
                    pins.append(m.group(1))
        source = pdata.get("tool", {}).get("uv", {}).get("sources", {}).get(GANDALF_PACKAGE)
        if isinstance(source, dict) and _SHA40.match(str(source.get("rev", ""))):
            pins.append(str(source["rev"]))
        for pin in pins:
            if pin != sha:
                raise Refusal("gandalf-pin", f"pyproject.toml pins {GANDALF_PACKAGE} to {pin[:12]} but "
                              f"uv.lock has {sha[:12]}; run uv lock and commit")
    jax = [p for p in packages if p.get("name") == "jax" and p.get("version")]
    applicable = [p for p in jax if not p.get("resolution-markers")
                  or any(_marker_applies(str(m)) for m in p["resolution-markers"])]
    jax_version = str(max(applicable, key=lambda p: _version_key(str(p["version"])))["version"]) if applicable else ""
    return sha, jax_version


def validate_timeout(hours: float) -> float:
    """The timeout in hours, refused unless 60 s <= timeout <= 24 h."""
    if not hours > 0 or hours > MAX_TIMEOUT_HOURS:
        raise Refusal("timeout", f"--timeout-hours must be > 0 and <= {MAX_TIMEOUT_HOURS:g} (got {hours})")
    if timeout_seconds(hours) < MIN_TIMEOUT_S:
        raise Refusal("timeout", f"--timeout-hours must be at least {MIN_TIMEOUT_S} s")
    return float(hours)


def preflight_common(repo: Path) -> str:
    """main, clean, pushed and up to date, no STOP. Returns the HEAD sha."""
    require_on_main(repo)
    require_clean(repo)
    head = require_pushed(repo)
    require_no_stop(repo)
    return head


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class SpawnProgress:
    """What a backend's spawn got done, updated as it goes.

    The backend raises ``attempted`` just before each spawn request and appends the call id
    when the request returns, so whatever happens, run ``i`` was certainly started if
    ``i < len(call_ids)``, may have been started if ``i < attempted`` (its request was sent
    and did not come back), and was certainly not started otherwise.
    """

    app_id: Optional[str] = None
    call_ids: List[str] = field(default_factory=list)
    attempted: int = 0
    error: Optional[str] = None
    log_path: Optional[str] = None


@dataclass
class FetchStats:
    fetched: int = 0  # files downloaded
    unchanged: int = 0  # local copies kept: same size and modification time as on the volume
    skipped: int = 0  # files under checkpoints/ left out
    nbytes: int = 0
    found: bool = True  # False: the folder is not on the volume (download with missing_ok)


def attempts_path(volume_path: str) -> str:
    """Where modal_app writes a run's attempt records: /<root>/_attempts/<set>/<run id>.

    They sit outside the run's out_dir (/<root>/<set>/<run id>), which belongs to the run
    code, so that run code that clears out_dir cannot remove the records that the budget
    of a restarted call and the charges rest on.
    """
    parts = PurePosixPath(volume_path.strip("/")).parts
    if len(parts) != 3 or ".." in parts:
        raise EnvError(f"unexpected volume path {volume_path!r} (expected /<root>/<set>/<run id>)")
    root, run_set, run_id = parts
    return f"/{root}/{ATTEMPTS_DIRNAME}/{run_set}/{run_id}"


def _unchanged(path: Path, size: int, mtime: int) -> bool:
    """True if ``path`` has this size and this modification time (whole seconds)."""
    if not mtime or not path.is_file():
        return False
    st = path.stat()
    return st.st_size == size and int(st.st_mtime) == int(mtime)


def _stamp(path: Path, mtime: int) -> None:
    """Give a downloaded file the volume's modification time, for the next fetch."""
    if mtime:
        os.utime(path, (mtime, mtime))


_APP_FAILURE_RE = re.compile(r" attempt [0-9a-f]{12} failed: ")  # modal_app._run_attempt's RuntimeError
_DESERIALIZE_RE = re.compile(r"deserializ|remote exception", re.I)


def from_the_call(exc: BaseException) -> bool:
    """True if ``exc`` was raised in the container and came back as the call's result.

    modal 1.3.5 (pinned in uv.lock) rebuilds such an exception in _utils.function_utils.
    _process_result, and _traceback.append_modal_tb sets ``__line_cache__`` on it. An
    exception raised on this side (the RPC layer, the network, the SDK) has no such mark.
    """
    return hasattr(exc, "__line_cache__")


def classify_call_exception(exc: BaseException, mexc: Any) -> Tuple[str, str]:
    """(state, detail) for an exception from ``FunctionCall.get(timeout=0)``.

    Only a definite answer about the call changes the ledger:
    - OutputExpiredError: expired (the call is over; the status comes from the records).
    - FunctionTimeoutError: timeout. InternalFailure, RemoteError: failed.
    - DeserializationError, or an ExecutionError about deserialising: the call is over but
      its result cannot be read here (expired, status from the records).
    - an exception that came back from the container (from_the_call), modal_app's own
      RuntimeError ('<run> attempt <id> failed: ...'), a SystemExit or an ImportError (a
      container that could not start): failed.
    - the builtin TimeoutError that poll_function raises while there is no output: running.
    - anything else: error. Modal could not be asked or did not answer (a terminated
      stream, a protocol error, a connection reset, authentication, not found, a local SDK
      problem), so nothing is concluded and the run stays as it was.
    Order matters: a TimeoutError that came back from the container is a failure.
    """
    if isinstance(exc, mexc.OutputExpiredError):
        return "expired", "result expired on Modal; status from the attempt records"
    if isinstance(exc, mexc.FunctionTimeoutError):
        return "timeout", short_exc(exc)
    if isinstance(exc, (mexc.InternalFailure, mexc.RemoteError)):
        return "failed", short_exc(exc, 500)
    if isinstance(exc, mexc.DeserializationError) or (
            isinstance(exc, mexc.ExecutionError) and _DESERIALIZE_RE.search(str(exc))):
        return "expired", (f"finished, but its result cannot be read here ({short_exc(exc, 200)}); "
                           "status from the attempt records")
    if from_the_call(exc) or isinstance(exc, (SystemExit, ImportError)) or (
            isinstance(exc, RuntimeError) and _APP_FAILURE_RE.search(str(exc))):
        return "failed", short_exc(exc, 500)
    if type(exc) is TimeoutError:
        return "running", ""
    return "error", f"could not query Modal: {short_exc(exc)}"


def require_test_origin(repo: Path) -> None:
    """The stub backend writes a fake launch into the ledger and pushes it; allow it only
    with a local, non-GitHub origin (a test's bare repository)."""
    for extra in ((), ("--push",)):
        proc = git(repo, "remote", "get-url", *extra, "origin", check=False)
        url = proc.stdout.strip() if proc.returncode == 0 else ""
        local = url[len("file://"):] if url.startswith("file://") else url
        path = Path(local).expanduser() if local else None
        if path is not None and not path.is_absolute():
            path = repo / path
        if not url or "github.com" in url.lower() or path is None or not path.is_dir():
            raise EnvError(f"S04_MODAL_BACKEND=stub is for the tests only, with a local bare repository "
                           f"as origin, but this repo's origin is {url or 'not set'!r}. Unset "
                           "S04_MODAL_BACKEND: the stub would push a fake launch into the real ledger.")


class StubBackend:
    """Fake Modal for tests. State lives in JSON files under S04_STUB_DIR.

    spawn_log.json     appended on every spawn, with the ledger as the REMOTE had it then
    server_calls.json  the call ids the fake Modal created
    calls.json         {"fc-...": "running"|"ok"|"failed"|"expired"|"timeout"|"error"}
    volume/<path>/     fake volume; attempt records in volume/<root>/_attempts/<set>/<run>/*.json
    apps.json          what ``modal app list --json`` would print
    Marker files:
      die_after_reserve          the launcher dies (os._exit 9) before anything reaches Modal
      fail_spawn                 spawn raises before anything reaches Modal (like a failed import)
      fail_after_app             the app is created, then spawn raises before any call request
      fail_spawn_after N         call request N+1 raises; Modal never saw it
      fail_spawn_after_start N   call request N+1 reaches Modal, then the reply is lost
      sigterm_during_spawn K     a SIGTERM arrives while call request K is in flight
      fail_read, apps_unavailable
    """

    name = "stub"

    def __init__(self, repo: Path, stub_dir: Path) -> None:
        """Fake Modal for ``repo``, with its state in ``stub_dir``."""
        self.repo = repo
        self.dir = stub_dir

    def preflight(self) -> None:
        """Create the state folder."""
        self.dir.mkdir(parents=True, exist_ok=True)

    def _load(self, name: str, default: Any) -> Any:
        """A JSON state file, or ``default`` if it does not exist."""
        path = self.dir / name
        if not path.is_file():
            return default
        return json.loads(path.read_text(encoding="utf-8"))

    def _save(self, name: str, data: Any) -> None:
        """Write a JSON state file."""
        (self.dir / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _marker_int(self, name: str) -> Optional[int]:
        """The integer in a marker file, or None if the marker is absent."""
        path = self.dir / name
        return int(path.read_text().strip() or "0") if path.exists() else None

    def _remote_ledger(self) -> Optional[Dict[str, Any]]:
        """The ledger on origin's main, as Modal would have seen it."""
        url = git(self.repo, "remote", "get-url", "origin", check=False).stdout.strip()
        if url and Path(url).is_dir():
            proc = subprocess.run(["git", "--git-dir", url, "show", f"main:{lc.LEDGER_REL}"],
                                  capture_output=True, text=True)
        else:
            proc = git(self.repo, "show", f"origin/main:{lc.LEDGER_REL}", check=False)
        return json.loads(proc.stdout) if proc.returncode == 0 else None

    def spawn(self, launch: Dict[str, Any], kind: str, sleep_s: int, progress: SpawnProgress) -> None:
        """Log the request (with the remote ledger), then fake the spawn as the marker files say."""
        log = self._load("spawn_log.json", [])
        n = len(log) + 1
        log.append({
            "n": n,
            "launch_id": launch["launch_id"],
            "kind": kind,
            "app_name": launch["app_name"],
            "sleep_s": sleep_s,
            "timeout_s": timeout_seconds(launch["timeout_hours"]),
            "run_ids": [r["run_id"] for r in launch["runs"]],
            "remote_ledger": self._remote_ledger(),
        })
        self._save("spawn_log.json", log)
        if (self.dir / "die_after_reserve").exists():
            os._exit(9)  # the launcher killed between the reservation and the Modal call
        if (self.dir / "fail_spawn").exists():
            raise RuntimeError("stub: fail_spawn")
        progress.app_id = f"ap-stub-{n}"
        if (self.dir / "fail_after_app").exists():
            raise RuntimeError("stub: the app was created, then its image failed to build")
        fail_after = self._marker_int("fail_spawn_after")
        lost_reply_after = self._marker_int("fail_spawn_after_start")
        sigterm_at = self._marker_int("sigterm_during_spawn")
        server: List[str] = self._load("server_calls.json", [])
        try:
            for i, _run in enumerate(launch["runs"], start=1):
                call_id = f"fc-stub-{n}-{i}"
                progress.attempted += 1
                try:
                    if fail_after is not None and i > fail_after:
                        raise RuntimeError(f"stub: spawn failed after {fail_after} call(s)")
                    server.append(call_id)
                    if lost_reply_after is not None and i > lost_reply_after:
                        raise ConnectionResetError("stub: the reply was lost after Modal created the call")
                    if sigterm_at is not None and i == sigterm_at:
                        os.kill(os.getpid(), signal.SIGTERM)
                        time.sleep(5)  # the launcher's handler raises KeyboardInterrupt in here
                except BaseException as exc:
                    progress.error = short_exc(exc)
                    break
                progress.call_ids.append(call_id)
        finally:
            self._save("server_calls.json", server)

    def call_status(self, call_id: str) -> Tuple[str, str]:
        """The state calls.json gives the call (running if it is absent)."""
        state = str(self._load("calls.json", {}).get(call_id, "running"))
        if state not in CALL_STATES:
            state = "failed"
        detail = "" if state == "running" else f"stub {state}"
        if state == "ok":
            detail = json.dumps({"status": "ok", "stub": True})
        if state == "error":
            detail = "could not query Modal: stub error"
        return state, detail

    def read_attempts(self, volume: str, volume_path: str) -> List[Dict[str, Any]]:
        """The attempt records of the run at ``volume_path`` on the fake volume."""
        if (self.dir / "fail_read").exists():
            raise BackendError("stub: fail_read")
        folder = self.dir / "volume" / attempts_path(volume_path).strip("/")
        if not folder.is_dir():
            return []
        return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*.json"))]

    def list_apps(self) -> Optional[List[Dict[str, Any]]]:
        """apps.json, or None while apps_unavailable exists."""
        if (self.dir / "apps_unavailable").exists():
            return None
        return self._load("apps.json", [])

    def download(self, volume: str, volume_path: str, dest: Path, include_checkpoints: bool,
                 missing_ok: bool = False) -> FetchStats:
        """Copy a folder from the fake volume, by the Modal backend's rules."""
        src = self.dir / "volume" / volume_path.strip("/")
        if not src.is_dir():
            if missing_ok:
                return FetchStats(found=False)
            raise EnvError(f"nothing on the volume at {volume}:{volume_path}")
        stats = FetchStats()
        for path in sorted(p for p in src.rglob("*") if p.is_file()):
            rel = path.relative_to(src)
            if not include_checkpoints and "checkpoints" in rel.parts:
                stats.skipped += 1
                continue
            st = path.stat()
            out = dest.joinpath(*rel.parts)
            if _unchanged(out, st.st_size, int(st.st_mtime)):
                stats.unchanged += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, out)
            _stamp(out, int(st.st_mtime))
            stats.fetched += 1
            stats.nbytes += st.st_size
        return stats


class ModalBackend:
    """The real Modal SDK (1.3.x), in-process.

    spawn: import loop/modal_app.py with the S04_* variables set, then inside
    ``app.run(detach=True)`` call ``.spawn`` once per run. A detached app keeps its calls
    running after this process exits. Modal's own output (image build, app creation) goes
    to a log file, so the command's output stays short.
    """

    name = "modal"

    def __init__(self, repo: Path, settings: Settings) -> None:
        """The Modal backend for ``repo``, which must be the repo this file is in."""
        self.repo = repo
        self.settings = settings

    def preflight(self) -> None:
        """Refuse a foreign repo, a missing modal_app.py, or a Modal SDK that does not import."""
        own = LOOP_DIR.parents[2].resolve()
        if self.repo.resolve() != own:
            raise EnvError(f"the Modal backend uploads code from {own}, but S04_REPO points at "
                           f"{self.repo}; unset S04_REPO")
        if not (LOOP_DIR / "modal_app.py").is_file():
            raise EnvError(f"{LOOP_DIR / 'modal_app.py'} not found")
        try:
            import modal  # noqa: F401
        except Exception as exc:
            raise EnvError(f"cannot import modal: {short_exc(exc)}") from None

    # Seams for tests -----------------------------------------------------------------

    def _load_app(self) -> Any:
        """Import loop/modal_app.py as module ``modal_app`` (Modal mounts it by that name).

        The import makes no request to Modal, so a failure here created nothing.
        """
        if "modal_app" in sys.modules:
            return sys.modules["modal_app"]
        spec = importlib.util.spec_from_file_location("modal_app", LOOP_DIR / "modal_app.py")
        if spec is None or spec.loader is None:
            raise EnvError("cannot load modal_app.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["modal_app"] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop("modal_app", None)
            raise
        return module

    def _function_call(self, call_id: str) -> Any:
        """A handle on a spawned call (a seam for the tests)."""
        import modal
        return modal.FunctionCall.from_id(call_id)

    def _volume(self, name: str) -> Any:
        """A lazy handle on the volume (a seam for the tests)."""
        import modal
        return modal.Volume.from_name(name)  # no create_if_missing: the volume exists

    def _log_path(self, launch: Dict[str, Any]) -> Optional[Path]:
        """Where Modal's output goes: the study's ignored data/launcher/ folder (readable by
        the agent when it diagnoses a failed launch, never committed, never uploaded), or a
        temp dir if that folder is somehow not ignored."""
        rel = f"{lc.STUDY_REL}/data/launcher/probe.log"
        ignored = git(self.repo, "check-ignore", "-q", "--no-index", rel, check=False).returncode == 0
        folder = (self.repo / lc.STUDY_REL / "data" / "launcher" if ignored
                  else Path(tempfile.gettempdir()) / "study04-launcher")
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            return None
        return folder / f"{launch['launch_id']}-{launch['app_name']}.log"

    @contextlib.contextmanager
    def _output(self, log_path: Optional[Path]) -> Iterator[None]:
        """Send Modal's own output (image build, app creation) to ``log_path``."""
        if log_path is None:
            yield
            return
        import modal
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(f"=== {utc_iso()} modal_launch.py\n")
            fh.flush()
            with contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
                with modal.enable_output():
                    yield

    def _hydrated_volume(self, name: str) -> Any:
        """The volume, looked up now. ``Volume.from_name`` is lazy, and a volume Modal cannot
        find (another workspace or environment) must be an error, never "no files"."""
        vol = self._volume(name)
        try:
            vol.hydrate()
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise BackendError(f"cannot look up the volume {name}: {short_exc(exc)}") from None
        return vol

    @staticmethod
    def _spawn_kwargs(launch: Dict[str, Any], run: Dict[str, Any], kind: str, sleep_s: int) -> Dict[str, Any]:
        """The arguments of one run's call to run_config or smoke."""
        if kind == "smoke":
            return dict(run_id=run["run_id"], run_set=launch["run_set"], launch_id=launch["launch_id"],
                        repo_commit=launch["repo_commit"], sleep_s=int(sleep_s))
        return dict(run_id=run["run_id"], run_set=launch["run_set"], config_relpath=run["config"],
                    launch_id=launch["launch_id"], repo_commit=launch["repo_commit"])

    # Backend interface ---------------------------------------------------------------

    def spawn(self, launch: Dict[str, Any], kind: str, sleep_s: int, progress: SpawnProgress) -> None:
        """Start one detached call per run inside app.run(detach=True).

        ``progress`` is updated before and after each request (see SpawnProgress), so the
        caller knows what was started even if this is cut off.
        """
        os.environ.update({
            "S04_APP_NAME": launch["app_name"],
            "S04_TIMEOUT_S": str(timeout_seconds(launch["timeout_hours"])),
            "S04_GANDALF_SHA": launch["gandalf_commit"],
            "S04_JAX_VERSION": launch.get("jax_version") or "",
            "S04_VOLUME": launch.get("volume") or self.settings.volume,
            "S04_VOLUME_ROOT": self.settings.volume_root,
        })
        log_path = self._log_path(launch)
        progress.log_path = str(log_path) if log_path else None
        module: Any = None
        with self._output(log_path):
            try:
                module = self._load_app()
                fn = module.smoke if kind == "smoke" else module.run_config
                with module.app.run(detach=True):
                    progress.app_id = module.app.app_id
                    for run in launch["runs"]:
                        kwargs = self._spawn_kwargs(launch, run, kind, sleep_s)
                        progress.attempted += 1  # from here on this run's call may exist on Modal
                        try:
                            call = fn.spawn(**kwargs)
                        except BaseException as exc:  # incl. KeyboardInterrupt from a signal
                            progress.error = short_exc(exc)
                            break  # start no more; the detached app keeps what it has
                        progress.call_ids.append(str(call.object_id))
            except BaseException as exc:
                if progress.error is None:
                    progress.error = short_exc(exc)
            if progress.app_id is None and module is not None:
                progress.app_id = getattr(getattr(module, "app", None), "app_id", None)

    def call_status(self, call_id: str) -> Tuple[str, str]:
        """``FunctionCall.get(timeout=0)``, classified by classify_call_exception.

        A KeyboardInterrupt raised here is a signal to this process and is re-raised; one
        that came back from the container is the call's outcome.
        """
        from modal import exception as mexc

        try:
            value = self._function_call(call_id).get(timeout=0)
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt) and not from_the_call(exc):
                raise
            return classify_call_exception(exc, mexc)
        try:
            return "ok", short(json.dumps(value, default=str), 500)
        except (TypeError, ValueError):
            return "ok", short(repr(value), 500)

    def read_attempts(self, volume: str, volume_path: str) -> List[Dict[str, Any]]:
        """The attempt records of the run at ``volume_path`` ([] if it has written none yet).

        BackendError if the volume or a record cannot be read: nothing may be charged on
        records that could not be seen.
        """
        from modal import exception as mexc
        from modal.volume import FileEntryType

        vol = self._hydrated_volume(volume)
        path = attempts_path(volume_path)
        try:
            entries = vol.listdir(path)
        except (mexc.NotFoundError, FileNotFoundError):
            return []  # the volume exists; the run has written no record yet
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise BackendError(f"cannot list {volume}:{path}: {short_exc(exc)}") from None
        records: List[Dict[str, Any]] = []
        for entry in entries:
            if entry.type != FileEntryType.FILE or not entry.path.endswith(".json"):
                continue
            try:
                data = b"".join(vol.read_file(entry.path))
            except KeyboardInterrupt:
                raise
            except BaseException as exc:
                raise BackendError(f"cannot read {volume}:{entry.path}: {short_exc(exc)}") from None
            try:
                record = json.loads(data.decode("utf-8"))
            except ValueError:
                continue  # unreadable record: the attempt then counts as incomplete
            if isinstance(record, dict):
                records.append(record)
        return records

    def list_apps(self) -> Optional[List[Dict[str, Any]]]:
        """``modal app list --json`` as a list, or None if it cannot be had."""
        if Path(self.settings.modal_bin).exists():
            cmd = [self.settings.modal_bin, "app", "list", "--json"]
        else:
            cmd = [sys.executable, "-m", "modal", "app", "list", "--json"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.TimeoutExpired):
            return None
        if proc.returncode != 0:
            return None
        try:
            data = parse_json_list(proc.stdout)
        except ValueError:
            return None
        return data if isinstance(data, list) else None

    def download(self, volume: str, volume_path: str, dest: Path, include_checkpoints: bool,
                 missing_ok: bool = False) -> FetchStats:
        """Copy a folder from the volume, keeping a local file only if its size and modification time match."""
        from modal import exception as mexc
        from modal.volume import FileEntryType

        vol = self._hydrated_volume(volume)
        base = volume_path.strip("/")
        try:
            entries = vol.listdir("/" + base, recursive=True)
        except (mexc.NotFoundError, FileNotFoundError):
            if missing_ok:
                return FetchStats(found=False)
            raise EnvError(f"nothing on the volume at {volume}:/{base}") from None
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            raise EnvError(f"cannot list {volume}:/{base}: {short_exc(exc)}") from None
        stats = FetchStats()
        for entry in entries:
            if entry.type != FileEntryType.FILE:
                continue
            try:
                rel = PurePosixPath(entry.path.lstrip("/")).relative_to(base)
            except ValueError:
                continue
            if not rel.parts or ".." in rel.parts:
                continue
            if not include_checkpoints and "checkpoints" in rel.parts:
                stats.skipped += 1
                continue
            out = dest.joinpath(*rel.parts)
            if _unchanged(out, entry.size, entry.mtime):
                stats.unchanged += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            part = out.with_name(out.name + ".part")
            with open(part, "wb") as fh:
                for chunk in vol.read_file(entry.path):
                    fh.write(chunk)
            os.replace(part, out)
            _stamp(out, entry.mtime)
            stats.fetched += 1
            stats.nbytes += entry.size
        return stats


def make_backend(repo: Path, settings: Settings) -> Any:
    """The Modal backend, or the stub (tests only, with a local non-GitHub origin)."""
    name = os.environ.get("S04_MODAL_BACKEND", "modal").strip() or "modal"
    if name == "stub":
        require_test_origin(repo)
        stub_dir = os.environ.get("S04_STUB_DIR", "").strip()
        if not stub_dir:
            raise EnvError("S04_MODAL_BACKEND=stub needs S04_STUB_DIR")
        return StubBackend(repo, Path(stub_dir))
    if name == "modal":
        return ModalBackend(repo, settings)
    raise EnvError(f"unknown S04_MODAL_BACKEND {name!r} (modal or stub)")


# ---------------------------------------------------------------------------
# launch and smoke
# ---------------------------------------------------------------------------


def new_launch(ledger: Dict[str, Any], settings: Settings, *, kind: str, run_set: str,
               timeout_hours: float, configs: Sequence[ConfigInfo], gate_reports: Optional[Sequence[str]],
               local_test: Optional[str], frozen: Dict[str, str], gandalf: str, jax_version: str,
               head: str) -> Dict[str, Any]:
    """A launch record in state 'reserved' (ledger schema 1). ``gate_report`` is the list of
    the reports the launch depends on (None for smoke)."""
    launch_id = next_launch_id(ledger)
    existing = {run.get("run_id") for _launch, run in lc.iter_runs(ledger)}
    labels = [c.label for c in configs] if kind == "gpu" else ["smoke"]
    t = int(time.time())
    while True:  # all runs of a launch share one timestamp; step on past any collision
        stamp = datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_ids = [f"04_{label}_{stamp}" for label in labels]
        if not set(run_ids) & existing:
            break
        t += 1
    runs = []
    for i, run_id in enumerate(run_ids):
        cfg = configs[i] if kind == "gpu" else None
        runs.append({
            "run_id": run_id,
            "config": cfg.rel if cfg else None,
            "config_sha256": cfg.sha256 if cfg else None,
            "volume_path": f"/{settings.volume_root}/{run_set}/{run_id}",
            "call_id": None,
            "reserved_hours": float(timeout_hours) if kind == "gpu" else 0.0,
            "status": "reserved",
            "charged_hours": None,
            "charge_basis": None,
            "attempts": [],
            "result": None,
        })
    return {
        "launch_id": launch_id,
        "kind": kind,
        "run_set": run_set,
        "iteration": os.environ.get("S04_ITER_ID") or None,
        "created_utc": utc_iso(),
        "repo_commit": head,
        "gandalf_commit": gandalf,
        "jax_version": jax_version,
        "volume": settings.volume,
        "app_name": app_name_for(settings.prefix, launch_id, run_set),
        "app_id": None,
        "timeout_hours": float(timeout_hours),
        "gate_report": list(gate_reports) if gate_reports else None,
        "local_test": local_test,
        "frozen": frozen,
        "state": "reserved",
        "notes": "",
        "runs": runs,
    }


def reserve_message(launch: Dict[str, Any]) -> str:
    """Subject of the commit that records a reservation."""
    n = len(launch["runs"])
    total = sum(float(r["reserved_hours"]) for r in launch["runs"])
    if launch["kind"] == "smoke":
        return f"{lc.LAUNCHER_SUBJECT_PREFIX}: reserve {launch['launch_id']} smoke: 0 A100-h"
    return (f"{lc.LAUNCHER_SUBJECT_PREFIX}: reserve {launch['launch_id']} set {launch['run_set']}: "
            f"{n} x {fmt_h(launch['timeout_hours'])} h = {fmt_h(total)} A100-h")


def undo_reservation(repo: Path, prev_head: str, existed: bool) -> None:
    """Put HEAD and the ledger back as they were. Touches nothing but the ledger path."""
    rel = lc.LEDGER_REL
    if rev(repo, "HEAD") != prev_head:
        git(repo, "reset", "--quiet", "--soft", prev_head, check=False)
    if existed:
        git(repo, "checkout", "--quiet", prev_head, "--", rel, check=False)
    else:
        git(repo, "rm", "--quiet", "--cached", "--force", "--ignore-unmatch", "--", rel, check=False)
        try:
            ledger_file(repo).unlink()
        except FileNotFoundError:
            pass


@dataclass
class SpawnOutcome:
    started: List[str]  # run ids with a call id
    uncertain: List[str]  # spawn raised: the call may exist; reservation kept, status unknown
    released: List[str]  # never sent to Modal: not_launched, charged 0


def apply_spawn_result(launch: Dict[str, Any], progress: SpawnProgress) -> SpawnOutcome:
    """Record what Modal started (see SpawnProgress for what is certain).

    Started runs get their call id and status running. A run whose spawn request went out
    and raised keeps its reservation as status unknown: Modal may have created the call,
    and reconcile settles it from the app's state and the attempt records. Only runs that
    were never sent are released (not_launched, charged 0). With no call started or
    uncertain, the launch failed and its whole reservation is released: an app, if one was
    created, cannot run a container without a call.
    """
    runs = launch["runs"]
    n_started = min(len(progress.call_ids), len(runs))
    n_attempted = min(max(progress.attempted, n_started), len(runs))
    error = short(progress.error or "interrupted", 200)
    out = SpawnOutcome([], [], [])
    for i, run in enumerate(runs):
        if i < n_started:
            run["call_id"] = progress.call_ids[i]
            run["status"] = "running"
            out.started.append(run["run_id"])
        elif i < n_attempted:
            run["status"] = "unknown"
            run["result"] = (f"spawn raised ({error}); the call may exist on Modal, so the reservation is "
                             "kept until reconcile settles it")
            out.uncertain.append(run["run_id"])
        else:
            run.update(status="not_launched", charged_hours=0.0, charge_basis="none",
                       result=f"not launched: {error}")
            out.released.append(run["run_id"])
    if out.started or out.uncertain:
        if progress.app_id:
            launch["state"] = "launched"
            launch["app_id"] = progress.app_id
        else:  # stays 'reserved': reconcile adopts the app by its name
            launch["notes"] = note_append(launch.get("notes"), "no app id came back; reconcile adopts "
                                          "the app by its name")
        if out.uncertain:
            launch["notes"] = note_append(launch.get("notes"), f"spawn of {', '.join(out.uncertain)} raised "
                                          f"({error}); reserved until reconcile settles it")
        if out.released:
            launch["notes"] = note_append(launch.get("notes"), f"{len(out.released)} of {len(runs)} runs "
                                          f"not launched: {error}")
    else:
        launch["state"] = "launch_failed"
        # The app id goes in the notes only. If the app runs after all, it is then unknown to
        # the ledger and the runner's app check writes STOP.
        extra = f" (app {progress.app_id} was created, with no call)" if progress.app_id else ""
        launch["notes"] = note_append(launch.get("notes"), f"launch failed{extra}: {error}")
    return out


def launch_message(launch: Dict[str, Any], outcome: SpawnOutcome, progress: SpawnProgress) -> str:
    """Subject of the commit that records a launch's outcome."""
    lid, n = launch["launch_id"], len(launch["runs"])
    if not outcome.started and not outcome.uncertain:
        return f"{lc.LAUNCHER_SUBJECT_PREFIX}: launch failed {lid}: {short(progress.error or 'no call started', 120)}"
    app = progress.app_id or "unknown"
    message = f"{lc.LAUNCHER_SUBJECT_PREFIX}: launched {lid} set {launch['run_set']} app {app}"
    extra = []
    if len(outcome.started) < n:
        extra.append(f"{len(outcome.started)} of {n} calls")
    if outcome.uncertain:
        extra.append(f"{len(outcome.uncertain)} uncertain")
    return message + (f" ({', '.join(extra)})" if extra else "")


def run_launch(repo: Path, settings: Settings, backend: Any, ledger: Dict[str, Any],
               launch: Dict[str, Any], kind: str, sleep_s: int) -> int:
    """Reserve (commit, push) -> start the calls -> record (commit, push) -> report."""
    lid = launch["launch_id"]
    path = ledger_file(repo)
    existed = path.exists()
    prev_head = rev(repo, "HEAD")
    assert prev_head is not None
    ledger.setdefault("launches", []).append(launch)
    progress = SpawnProgress()
    with SignalGuard() as guard:
        # a. The reservation is on origin before Modal is called.
        try:
            lc.save_ledger(path, ledger)
            sha = commit_ledger(repo, reserve_message(launch))
            ok, err = push_main(repo, expect=sha)
            if not ok:
                raise Refusal("push", f"pushing the reservation failed: {err}. Nothing was launched; "
                              "the ledger and HEAD are as before.")
        except BaseException:
            undo_reservation(repo, prev_head, existed)
            raise
        total = sum(float(r["reserved_hours"]) for r in launch["runs"])
        print(f"Reserved {lid} ({launch['kind']}, set {launch['run_set']}): {len(launch['runs'])} run(s), "
              f"{fmt_h(total)} A100-h held; commit {sha[:12]} pushed.", flush=True)

        # b. Start the calls, unless the session is being stopped.
        if guard.received:
            progress.error = "interrupted by a signal before Modal was called"
        else:
            try:
                with guard.raising_signals():
                    backend.spawn(launch, kind, sleep_s, progress)
            except BaseException as exc:
                if progress.error is None:
                    progress.error = short_exc(exc)

        # c. Record the outcome. Signals are only noted until this is pushed.
        outcome = apply_spawn_result(launch, progress)
        lc.save_ledger(path, ledger)
        committed_ok, pushed_ok, push_err = False, False, ""
        try:
            sha = commit_ledger(repo, launch_message(launch, outcome, progress))
            committed_ok = True
            pushed_ok, push_err = push_main(repo, expect=sha)
        except EnvError as exc:
            push_err = str(exc)
        interrupted = bool(guard.received)

    # d. Report.
    n = len(launch["runs"])
    if outcome.started:
        print(f"Launched {lid}, set {launch['run_set']}: app {progress.app_id}, {len(outcome.started)} of "
              f"{n} call(s) started. They are detached and keep running after this command exits.")
    if outcome.uncertain:
        print(f"UNCERTAIN {lid}: the spawn of {', '.join(outcome.uncertain)} raised ({progress.error}). "
              "Modal may have created the call, so its reservation is kept (status unknown). 'reconcile' "
              "settles it from the app's state and the run's attempt records, and charges the full "
              "reservation if there are none. Do not relaunch its config before reconcile has charged it.",
              file=sys.stderr)
    if outcome.released and (outcome.started or outcome.uncertain):
        print(f"NOT LAUNCHED: {', '.join(outcome.released)} (never sent to Modal; reservation released, "
              f"0 A100-h): {progress.error or 'interrupted'}", file=sys.stderr)
    if not outcome.started and not outcome.uncertain:
        print(f"LAUNCH FAILED {lid}: {progress.error or 'no call started'}. No call was sent to Modal; "
              "the reservation is released (0 A100-h charged).", file=sys.stderr)
    for run in launch["runs"]:
        print(f"  {run['run_id']}  status {run['status']}  call {run.get('call_id') or '-'}  "
              f"volume {launch.get('volume')}:{run['volume_path']}")
    if progress.log_path:
        print(f"Modal output: {progress.log_path}")
    if not committed_ok:
        print(f"WARNING: the launch record is saved but NOT committed ({push_err}). Do not edit, "
              "commit or restore the ledger yourself. The committed ledger still holds the "
              "reservation; after the session the runner stashes the file, and a later "
              "'reconcile' adopts the app by its name. Note this in the log for Anjor.",
              file=sys.stderr)
    elif not pushed_ok:
        print(f"WARNING: the ledger commit is local and NOT pushed ({push_err}). Push it "
              "(git push origin main) before anything else.", file=sys.stderr)
    print(lc.status_line(settings.cap, ledger))
    if interrupted:
        return EXIT_INTERRUPTED
    if outcome.uncertain:
        return EXIT_UNCERTAIN
    return EXIT_OK if outcome.started else EXIT_REFUSED


def cmd_launch(args: argparse.Namespace, repo: Path) -> int:
    """launch: check every precondition, then reserve, start and record (run_launch)."""
    settings = load_settings(repo)
    backend = make_backend(repo, settings)
    backend.preflight()
    with launcher_lock(repo):
        head = preflight_common(repo)
        history = GitHistory.load(repo)
        check_loop_files(repo, settings, history)
        ledger = read_verified_ledger(repo)
        check_ledger_succession(repo, settings, history)
        check_frozen_items(repo, ledger)
        hours = validate_timeout(args.timeout_hours)
        check_run_set(repo, args.set, require_row=False)
        configs = [check_config(repo, raw) for raw in args.configs]
        rels = [c.rel for c in configs]
        labels = [c.label for c in configs]
        if len(set(rels)) != len(rels) or len(set(labels)) != len(labels):
            raise Refusal("config", "the same config (or two configs with the same file name) "
                          "appears twice; run labels must be unique")
        gates = check_gate_reports(repo, args.gate_report or [], args.set)
        local = check_local_test(repo, args.local_test)
        check_run_set(repo, args.set, require_row=True)
        frozen = resolve_freeze(repo, args.freeze or [])
        gandalf, jax_version = lock_pins(repo)
        check_cap(settings.cap, ledger, len(configs) * hours)
        launch = new_launch(ledger, settings, kind="gpu", run_set=args.set, timeout_hours=hours,
                            configs=configs, gate_reports=gates, local_test=local, frozen=frozen,
                            gandalf=gandalf, jax_version=jax_version, head=head)
        return run_launch(repo, settings, backend, ledger, launch, "gpu", 0)


def cmd_smoke(args: argparse.Namespace, repo: Path) -> int:
    """smoke: the launch path with the CPU-only smoke function, charged 0 hours."""
    settings = load_settings(repo)
    if not 0 <= args.sleep <= MAX_SMOKE_SLEEP_S:
        raise Refusal("sleep", f"--sleep must be between 0 and {MAX_SMOKE_SLEEP_S} s")
    backend = make_backend(repo, settings)
    backend.preflight()
    with launcher_lock(repo):
        head = preflight_common(repo)
        history = GitHistory.load(repo)
        check_loop_files(repo, settings, history)
        ledger = read_verified_ledger(repo)
        check_ledger_succession(repo, settings, history)
        gandalf, jax_version = lock_pins(repo)
        check_cap(settings.cap, ledger, 0.0)
        launch = new_launch(ledger, settings, kind="smoke", run_set="smoke",
                            timeout_hours=SMOKE_TIMEOUT_HOURS, configs=[], gate_reports=None,
                            local_test=None, frozen={}, gandalf=gandalf, jax_version=jax_version,
                            head=head)
        return run_launch(repo, settings, backend, ledger, launch, "smoke", int(args.sleep))


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------


@dataclass
class AttemptSummary:
    attempts: List[Dict[str, Any]]  # ledger rows, oldest first
    complete_hours: float  # end - start (or the end record's duration) over attempts with an end record
    open: int  # attempts without a usable end record (bounded by the budget when charged)
    latest_start: Optional[float]  # epoch of the newest start record
    last_status: Optional[str]  # end status of the newest attempt
    first_start: Optional[float] = None  # epoch of the run's first attempt: the budget starts here
    open_starts: List[Optional[float]] = field(default_factory=list)  # start of each open attempt, if known


def _num(value: Any) -> Optional[float]:
    """``value`` as a float, or None (NaN too)."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x else None  # NaN -> None


def _epoch(utc: Any) -> Optional[float]:
    """Epoch seconds of a '2026-10-05T10:15:00Z' string, or None."""
    try:
        return datetime.strptime(str(utc), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def summarize_attempts(records: Sequence[Dict[str, Any]], run_id: str) -> AttemptSummary:
    """Pair the start and end records written by modal_app._Attempt.

    An attempt is complete when it has an end record and either its start record or a
    duration in the end record (modal_app writes duration_s); otherwise it is open.
    """
    by_id: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rec in records:
        if not isinstance(rec, dict) or rec.get("run_id") not in (None, run_id):
            continue
        aid, event = str(rec.get("attempt_id") or ""), rec.get("event")
        if aid and event in ("start", "end"):
            by_id.setdefault(aid, {}).setdefault(str(event), rec)
    rows: List[Tuple[float, Dict[str, Any]]] = []
    complete, n_open, latest = 0.0, 0, None
    starts: List[float] = []
    open_starts: List[Optional[float]] = []
    for aid, events in by_id.items():
        start, end = events.get("start"), events.get("end")
        ts = _num(start.get("t_epoch")) if start else None
        te = _num(end.get("t_epoch")) if end else None
        duration = _num(end.get("duration_s")) if end else None
        if ts is None and te is not None and duration is not None and duration >= 0:
            ts_derived: Optional[float] = te - duration  # the start record was lost; the end record knows it
        else:
            ts_derived = ts
        hours = None
        if ts_derived is not None and te is not None:
            hours = max(0.0, (te - ts_derived) / 3600.0)
            complete += hours
        else:
            n_open += 1
            open_starts.append(ts_derived)
        if ts_derived is not None:
            starts.append(ts_derived)
        if ts is not None and (latest is None or ts > latest):
            latest = ts
        rows.append((ts_derived if ts_derived is not None else (te or 0.0), {
            "attempt_id": aid,
            "start_utc": start.get("t_utc") if start else None,
            "end_utc": end.get("t_utc") if end else None,
            "hours": round(hours, 4) if hours is not None else None,
            "status": end.get("status") if end else None,
        }))
    rows.sort(key=lambda item: item[0])
    attempts = [row for _t, row in rows]
    return AttemptSummary(attempts, complete, n_open, latest,
                          attempts[-1]["status"] if attempts else None,
                          first_start=min(starts) if starts else None, open_starts=open_starts)


def budget_kept(summary: AttemptSummary, timeout_hours: float) -> bool:
    """Do the records agree with the budget, at most the timeout over all attempts?

    modal_app ends every attempt by the run's first start plus the timeout, so complete
    attempts add up to more only if that failed; then nothing about the budget is assumed.
    """
    return summary.complete_hours <= timeout_hours + BUDGET_TOLERANCE_HOURS


def charge_from_records(summary: AttemptSummary, timeout_hours: float) -> Tuple[float, str]:
    """(A100-hours, basis) to charge a finished run that has attempt records.

    Complete attempts count as recorded. An open attempt (no usable end record) ran at most
    until the budget ended, the run's first start plus the timeout T (modal_app's watchdog
    and Modal's timeout end it there), so it counts from its start to that moment, or a full
    T if its start is unknown; and the run as a whole is charged at most T, since its
    attempts run one after another inside that window, unless its complete attempts alone
    show more. If they do, the budget was not kept, and each open attempt counts a full T.
    Container start-up is billed but never recorded, so it is not in the charge.
    """
    complete = summary.complete_hours
    if summary.open == 0:
        return complete, "records"
    if not budget_kept(summary, timeout_hours):
        return complete + timeout_hours * summary.open, "records+full_timeout"
    budget_end = None if summary.first_start is None else summary.first_start + timeout_hours * 3600.0
    bound = 0.0
    for start in summary.open_starts:
        if start is None or budget_end is None:
            bound += timeout_hours
        else:
            bound += min(timeout_hours, max(0.0, (budget_end - start) / 3600.0))
    return max(complete, min(timeout_hours, complete + bound)), "records+budget"


def hours_to_hold(summary: AttemptSummary, timeout_hours: float) -> float:
    """The most a running call can still be charged (charge_from_records): the timeout T,
    unless its records already show the budget was not kept."""
    if budget_kept(summary, timeout_hours):
        return timeout_hours
    return summary.complete_hours + timeout_hours * max(summary.open, 1)


# End status of the newest attempt -> run status, for a call whose result Modal no longer has.
_STATUS_FROM_RECORDS = {"ok": "finished", "error": "failed", "interrupted": "failed"}


class Reconciler:
    """Charge finished runs, keep running ones reserved, adopt apps of interrupted launches."""

    def __init__(self, backend: Any, settings: Settings, ledger: Dict[str, Any],
                 only: Optional[str]) -> None:
        """Reconcile ``ledger`` in place through ``backend``; ``only`` limits it to one launch."""
        self.backend = backend
        self.settings = settings
        self.ledger = ledger
        self.only = only
        self.lines: List[str] = []
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.charged: List[str] = []
        self.updated: List[str] = []
        self.changed = False
        self._apps: Optional[List[Dict[str, Any]]] = None
        self._apps_loaded = False

    def _list_apps(self) -> Optional[List[Dict[str, Any]]]:
        """Modal's app list, fetched once per reconcile."""
        if not self._apps_loaded:
            self._apps = self.backend.list_apps()
            self._apps_loaded = True
        return self._apps

    def run(self, now: Optional[float] = None) -> None:
        """Settle every launch that is not closed or failed, and close a launch once all its runs are charged."""
        now = time.time() if now is None else now
        for launch in self.ledger.get("launches", []):
            if self.only and launch.get("launch_id") != self.only:
                continue
            if launch.get("state") in ("launch_failed", "closed"):
                continue
            pending = [r for r in launch.get("runs", [])
                       if r.get("charged_hours") is None and r.get("status") != "not_launched"]
            app_running: Optional[bool] = None
            if any(not r.get("call_id") for r in pending):
                app_running = self._app_state(launch)
            for run in pending:
                self._reconcile_run(launch, run, app_running, now)
            if all(r.get("charged_hours") is not None or r.get("status") == "not_launched"
                   for r in launch.get("runs", [])):
                launch["state"] = "closed"
                self.changed = True
                self.lines.append(f"{launch['launch_id']}: closed")

    def _app_state(self, launch: Dict[str, Any]) -> Optional[bool]:
        """Is the launch's app running? Adopts the app by name if the launcher was cut off
        after reserving. None means unknown (the runs then stay reserved)."""
        lid = launch["launch_id"]
        apps = self._list_apps()
        if apps is None:
            self.errors.append(f"{lid}: cannot list Modal apps")
            self.lines.append(f"{lid}: cannot list Modal apps; runs without a call id stay reserved")
            return None
        rows = lc.app_rows(apps)
        if launch.get("app_id"):
            match = [r for r in rows if r["id"] == launch["app_id"]]
            if not match:
                self.lines.append(f"{lid}: app {launch['app_id']} is not in Modal's app list (stopped long "
                                  "ago, or listed in another environment); runs without a call id are settled "
                                  "from their attempt records, which must be readable")
                return False
            return lc.is_running_state(match[0]["state"])
        match = [r for r in rows if r["name"] == launch.get("app_name")]
        if not match:
            self.lines.append(
                f"{lid}: reserved, but no Modal app named {launch.get('app_name')} is listed. The "
                "launcher stopped before Modal created the app, or the app is too old to list. "
                "The hours stay reserved; raise it in QUESTIONS.md for Anjor.")
            return None
        match.sort(key=lambda r: not lc.is_running_state(r["state"]))
        chosen = match[0]
        launch["app_id"] = chosen["id"]
        launch["state"] = "launched"
        launch["notes"] = note_append(launch.get("notes"),
                                      f"app {chosen['id']} adopted by name at {utc_iso()}; call ids unknown")
        for run in launch.get("runs", []):
            if run.get("status") == "reserved":
                run["status"] = "unknown"
        self.changed = True
        self.updated.append(f"adopt {lid}")
        self.lines.append(f"{lid}: adopted app {chosen['id']} by name (state {chosen['state'] or '?'})")
        return lc.is_running_state(chosen["state"])

    def _reconcile_run(self, launch: Dict[str, Any], run: Dict[str, Any],
                       app_running: Optional[bool], now: float) -> None:
        """Settle one run: unchanged on an error, reserved while running, charged once it is over."""
        rid = run["run_id"]
        timeout = float(launch.get("timeout_hours") or 0.0)
        if run.get("call_id"):
            state, detail = self.backend.call_status(run["call_id"])
        elif app_running is None:
            self.lines.append(f"{rid}: no call id and the app state is unknown; stays reserved")
            return
        elif app_running:
            state, detail = "running", ""
        else:
            state, detail = "expired", "app stopped; no call id, status from the attempt records"
        if state == "error":
            self.errors.append(f"{rid}: {detail}")
            self.lines.append(f"{rid}: {detail}; unchanged")
            return
        try:
            records = self.backend.read_attempts(launch.get("volume") or self.settings.volume,
                                                 run["volume_path"])
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            self.errors.append(f"{rid}: {short_exc(exc)}")
            self.lines.append(f"{rid}: cannot read the attempt records ({short_exc(exc)}); unchanged")
            return
        summary = summarize_attempts(records, rid)

        if state == "running":
            if run.get("status") in ("reserved", "unknown"):
                run["status"] = "running"
                self.changed = True
                self.updated.append(rid)
            if launch.get("kind") != "smoke":
                # A restart after preemption shares the call's budget (modal_app), so the run
                # needs more than its timeout only if its records show the budget was not kept.
                need = hours_to_hold(summary, timeout)
                if need > float(run.get("reserved_hours") or 0.0) + 1e-9:
                    run["reserved_hours"] = round(need, 4)
                    run["attempts"] = summary.attempts
                    self.changed = True
                    if rid not in self.updated:
                        self.updated.append(rid)
                    self.lines.append(f"{rid}: its {len(summary.attempts)} attempts already ran "
                                      f"{summary.complete_hours:.2f} h, more than its budget of {fmt_h(timeout)} h; "
                                      f"reservation raised to {need:.2f} A100-h")
            if summary.latest_start is not None:
                since = f"{max(0.0, (now - summary.latest_start) / 3600.0):.2f} h since the latest attempt started"
            else:
                since = "no start record yet"
            begun = summary.first_start if summary.first_start is not None else _epoch(launch.get("created_utc"))
            elapsed = max(0.0, (now - begun) / 3600.0) if begun is not None else 0.0
            self.lines.append(f"{rid}: running, {since} (reserved {fmt_h(run['reserved_hours'])} A100-h)")
            if elapsed > timeout + STUCK_GRACE_HOURS:
                origin = "its first attempt started" if summary.first_start is not None else "the launch"
                warning = (f"{rid}: Modal reports no result {elapsed:.1f} h after {origin}, more than its "
                           f"budget ({fmt_h(timeout)} h) plus {fmt_h(STUCK_GRACE_HOURS)} h. It may be queued for "
                           f"a GPU or stuck. Look at app {launch.get('app_id')} with 'modal app logs' and raise "
                           "it in QUESTIONS.md if it persists. Its hours stay reserved.")
                self.warnings.append(warning)
                self.lines.append(f"WARNING: {warning}")
            return

        if launch.get("kind") == "smoke":
            charged, basis = 0.0, "none"
        elif not summary.attempts:
            charged, basis = float(run.get("reserved_hours") or timeout), "full_reservation"
        else:
            charged, basis = charge_from_records(summary, timeout)
        if state == "ok":
            status = "finished"
        elif state in ("failed", "timeout"):
            status = "failed"
        else:  # expired: the outcome comes from the records
            status = _STATUS_FROM_RECORDS.get(str(summary.last_status), "unknown")
        run.update(charged_hours=round(charged, 4), charge_basis=basis, status=status,
                   attempts=summary.attempts, result=short(f"{state}: {detail}" if detail else state, 500))
        self.changed = True
        self.charged.append(rid)
        self.lines.append(f"{rid}: {status}, charged {charged:.2f} A100-h ({basis}; "
                          f"{len(summary.attempts)} attempt(s))")

    def commit_message(self) -> str:
        """Subject of the reconcile commit."""
        parts = []
        if self.charged:
            parts.append(", ".join(self.charged))
        if self.updated:
            parts.append("update " + ", ".join(self.updated))
        text = "; ".join(parts) or "record the ledger state"
        return short(f"{lc.LAUNCHER_SUBJECT_PREFIX}: reconcile {text}", 250)


def cmd_reconcile(args: argparse.Namespace, repo: Path) -> int:
    """reconcile: charge finished runs from their attempt records and commit the ledger if it changed."""
    settings = load_settings(repo)
    backend = make_backend(repo, settings)
    backend.preflight()
    with launcher_lock(repo):
        require_on_main(repo)
        fetch_origin(repo)
        _ahead, behind = ahead_behind(repo)
        if behind:
            raise Refusal("pull", f"HEAD is {behind} commit(s) behind origin/main. Pull first "
                          "(git pull --ff-only origin main); the launcher never pulls or rebases.")
        if git(repo, "status", "--porcelain", "--", lc.LEDGER_REL).stdout.strip():
            raise Refusal("ledger", f"{lc.LEDGER_REL} has uncommitted changes. The launcher commits "
                          "every change it makes, so these did not come from it (or its commit failed "
                          "and it said so). Leave the file alone and raise it for Anjor.")
        ledger = read_verified_ledger(repo)
        check_ledger_succession(repo, settings, GitHistory.load(repo))
        if args.launch and not any(l.get("launch_id") == args.launch for l in ledger.get("launches", [])):
            raise Refusal("launch", f"no launch {args.launch} in the ledger")
        rec = Reconciler(backend, settings, ledger, args.launch)
        rec.run()
        for line in rec.lines:
            print(line)
        interrupted = False
        if rec.changed:
            with SignalGuard() as guard:
                lc.save_ledger(ledger_file(repo), ledger)
                sha = commit_ledger(repo, rec.commit_message())
                ok, err = push_main(repo, expect=sha)
                interrupted = bool(guard.received)
            print(f"Ledger committed ({sha[:12]})" + (" and pushed." if ok else "."))
            if not ok:
                print(f"WARNING: the ledger commit is NOT pushed ({err}). Push it (git push origin "
                      "main) before anything else.", file=sys.stderr)
        else:
            print("No ledger change.")
        print(lc.status_line(settings.cap, ledger))
        excess = cap_excess(settings.cap, ledger)
        if excess:
            print(excess, file=sys.stderr)
        if rec.errors:
            print("Could not reconcile everything (Modal unreachable, or another workspace or environment?): "
                  + "; ".join(rec.errors) + ". Those runs are unchanged and stay reserved; run reconcile "
                  "again later.", file=sys.stderr)
        if interrupted:
            return EXIT_INTERRUPTED
        if excess:
            return EXIT_CAP_EXCEEDED
        return EXIT_ENV if rec.errors else EXIT_OK


# ---------------------------------------------------------------------------
# status, fetch, verify, apps
# ---------------------------------------------------------------------------


def cmd_status(args: argparse.Namespace, repo: Path) -> int:
    """The compute line first (it goes into the log's Compute field), then runs in flight."""
    settings = load_settings(repo)
    ledger = read_ledger(repo)
    try:
        line = lc.status_line(settings.cap, ledger)
        excess = cap_excess(settings.cap, ledger)
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise EnvError(f"the ledger is malformed: {exc}") from None
    print(line)
    for launch, run in lc.iter_runs(ledger):
        if run.get("charged_hours") is None and run.get("status") != "not_launched":
            print(f"in flight: {run.get('run_id')} ({launch.get('launch_id')}, {launch.get('kind')}, "
                  f"set {launch.get('run_set')}) status {run.get('status')}, reserved "
                  f"{fmt_h(run.get('reserved_hours') or 0)} A100-h, app {launch.get('app_id') or '-'}, "
                  f"call {run.get('call_id') or '-'}")
    problems = lc.verify_ledger(ledger)
    if problems:
        print("LEDGER PROBLEMS: " + "; ".join(problems) + " (see 'modal_launch.py verify')", file=sys.stderr)
        return EXIT_REFUSED
    if excess:
        print(excess, file=sys.stderr)
        return EXIT_CAP_EXCEEDED
    return EXIT_OK


def cmd_check_report(args: argparse.Namespace, repo: Path) -> int:
    """check-report: apply the launcher's gate-report rules without launching (no network, no git changes).

    Without --set, each file is checked as a report: the form of LOOP.md's gate-report template
    (gate_report_problems) and, for a file under gate_reports/, its name, title and critic
    report (gate_report_file_problems). The file need not be committed, so the report of a
    '--test' evaluation can be checked before anything is frozen. With --set, the reports are
    checked exactly as 'launch --set <S> --gate-report ...' checks them: committed, the newest
    evaluation of their gate, and the right kinds for the set. Exit 0 if every check passes, 1 if not.
    """
    if args.set:
        check_run_set(repo, args.set, require_row=False)
        rels = check_gate_reports(repo, args.reports, args.set)
        print(f"reports ok for a launch of set {args.set}: {', '.join(rels)}")
        return EXIT_OK
    failed = 0
    for raw in args.reports:
        path = Path(raw).expanduser()
        path = path if path.is_absolute() else repo / path
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            print(f"{raw}: cannot read: {exc}")
            failed += 1
            continue
        problems = gate_report_problems(text)
        try:
            rel = path.resolve().relative_to(repo.resolve()).as_posix()
        except ValueError:
            rel = ""
        if rel.startswith(GATE_REPORTS_REL + "/"):
            problems += gate_report_file_problems(repo, rel, text)
        else:
            print(f"{raw}: note: not under {GATE_REPORTS_REL}/, so its name, title and critic report "
                  "were not checked")
        if problems:
            failed += 1
            for problem in problems:
                print(f"{raw}: {problem}")
        else:
            print(f"{raw}: ok")
    return EXIT_REFUSED if failed else EXIT_OK


def cmd_verify(args: argparse.Namespace, repo: Path) -> int:
    """verify: the ledger's structure and integrity hash (no network)."""
    path = ledger_file(repo)
    try:
        ledger = lc.load_ledger(path)
    except (OSError, ValueError) as exc:
        print(f"ledger problem: cannot read {lc.LEDGER_REL}: {exc}")
        return EXIT_REFUSED
    if not isinstance(ledger, dict):
        print(f"ledger problem: {lc.LEDGER_REL} is not a JSON object")
        return EXIT_REFUSED
    problems = lc.verify_ledger(ledger)
    if problems:
        for problem in problems:
            print(f"ledger problem: {problem}")
        return EXIT_REFUSED
    print("ledger ok")
    if not path.exists():
        print(f"({lc.LEDGER_REL} does not exist yet: nothing has been launched)")
    return EXIT_OK


def cmd_fetch(args: argparse.Namespace, repo: Path) -> int:
    """fetch: download a run's folder from the volume to the study's data/."""
    settings = load_settings(repo)
    ledger = read_ledger(repo)
    found = [(launch, run) for launch, run in lc.iter_runs(ledger) if run.get("run_id") == args.run_id]
    if not found:
        raise Refusal("fetch", f"no run {args.run_id} in the ledger")
    launch, run = found[-1]
    dest = (Path(args.dest).expanduser().resolve() if args.dest
            else repo / lc.STUDY_REL / "data" / str(launch.get("run_set")) / str(run["run_id"]))
    backend = make_backend(repo, settings)
    backend.preflight()
    volume = launch.get("volume") or settings.volume
    records_path = attempts_path(run["volume_path"])
    records = backend.download(volume, records_path, dest / ATTEMPTS_DIRNAME, True, missing_ok=True)
    stats = backend.download(volume, run["volume_path"], dest, bool(args.checkpoints), missing_ok=records.found)
    if stats.found:
        print(f"Fetched {stats.fetched} file(s), {stats.nbytes / 1e6:.1f} MB, from {volume}:{run['volume_path']} "
              f"to {dest}")
    else:
        print(f"The run wrote no output folder ({volume}:{run['volume_path']} does not exist).")
    if stats.unchanged:
        print(f"Kept {stats.unchanged} local file(s) with the same size and modification time as on the volume.")
    if stats.skipped:
        print(f"Skipped {stats.skipped} file(s) under checkpoints/ (add --checkpoints to get them).")
    if records.found:
        print(f"Attempt records ({volume}:{records_path}): {records.fetched + records.unchanged} file(s) in "
              f"{dest / ATTEMPTS_DIRNAME}; their end records carry each attempt's error and traceback.")
    else:
        print(f"No attempt records on the volume yet ({volume}:{records_path}).")
    return EXIT_OK


def cmd_apps(args: argparse.Namespace, repo: Path) -> int:
    """apps: running apps with the loop's prefix that the ledger does not know."""
    settings = load_settings(repo)
    try:
        apps = parse_json_list(Path(args.json_file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EnvError(f"cannot read {args.json_file}: {exc}") from None
    ledger = read_ledger(repo)
    unknown = lc.unknown_running_apps(apps, ledger, settings.prefix)
    for row in unknown:
        print(f"unknown running app: {row['id']} {row['name']} ({row['state']})")
    if not unknown:
        print(f"no unknown running apps with prefix {settings.prefix}")
    return EXIT_REFUSED if unknown else EXIT_OK


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """The command-line interface."""
    parser = argparse.ArgumentParser(
        prog="modal_launch.py",
        description="Study 04 Modal launcher: the only path from the loop to a GPU.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("launch", help="reserve hours and start one detached A100 call per config")
    p.add_argument("--set", required=True, help=f"run set: {', '.join(RUN_SETS)}; RUNS.md needs a row for it")
    p.add_argument("--timeout-hours", required=True, type=float,
                   help="per-run timeout in hours (> 0, <= 24); this much is reserved per config")
    p.add_argument("--gate-report", required=True, action="append",
                   help=f"committed report under {GATE_REPORTS_REL}/ that shows a passed gate "
                        "(repeatable; every report given must pass)")
    p.add_argument("--local-test", required=True, help="committed record of the local smoke test")
    p.add_argument("--freeze", action="append", default=[],
                   help="'path#heading' or 'path' to freeze at launch (repeatable)")
    p.add_argument("configs", nargs="+", help=f"committed YAML configs under {CONFIGS_REL}/")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("smoke", help="tiny CPU-only launch through the same path (0 hours)")
    p.add_argument("--sleep", type=int, default=DEFAULT_SMOKE_SLEEP_S,
                   help=f"seconds the function sleeps (default {DEFAULT_SMOKE_SLEEP_S})")
    p.set_defaults(func=cmd_smoke)

    p = sub.add_parser("reconcile", help="charge finished runs from the attempt records")
    p.add_argument("--launch", default=None, help="only this launch id (e.g. L003)")
    p.set_defaults(func=cmd_reconcile)

    p = sub.add_parser("status", help="compute line, then runs in flight (no network)")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("fetch", help="download a run's directory from the volume")
    p.add_argument("run_id")
    p.add_argument("--checkpoints", action="store_true", help="include checkpoints/")
    p.add_argument("--dest", default=None, help="destination folder (default: the study's data/)")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("verify", help="check the ledger (no network)")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("check-report", help="check gate reports against the launcher's rules, "
                                            "without launching (no network, no git changes)")
    p.add_argument("--set", default=None,
                   help=f"check them as a launch of this set would ({', '.join(RUN_SETS)}); they must "
                        "then be committed")
    p.add_argument("reports", nargs="+", help="report files")
    p.set_defaults(func=cmd_check_report)

    p = sub.add_parser("apps", help="unknown running apps in saved 'modal app list --json' output")
    p.add_argument("--json-file", required=True)
    p.set_defaults(func=cmd_apps)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run one command and map refusals and errors to exit codes."""
    args = build_parser().parse_args(argv)
    repo = repo_root()
    try:
        return int(args.func(args, repo))
    except Refusal as exc:
        print(f"REFUSED [{exc.rule}]: {exc.detail}", file=sys.stderr)
        return EXIT_REFUSED
    except (EnvError, BackendError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_ENV
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
