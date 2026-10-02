#!/usr/bin/env python3
"""Runner checks for the Study 04 autonomous loop.

``run_iteration.sh`` calls this script before and after every headless session. The agent
calls ``stop-write`` for a hard stop and ``log-check`` before it commits log lines, and
``pause.sh`` calls ``stop-write`` for Anjor. Standard library plus ``loopcommon`` only, and
Python 3.9 compatible. The runner runs it as ``python -I -S`` from the interpreter's real
path, so nothing from the clone's virtual environment (a ``.pth`` file, say) runs with it.

Trust. The checks that compare the repo against recorded state read ``frozen.json`` and
``config.env`` from the directory this file lives in, not from the repo. The runner copies
the loop files out of the commit the session started from and runs that copy, so a session
that commits a change to the guards cannot weaken the checks applied to its own commits.
Every other input comes from ``--repo``.

Identity. A loop commit is a commit whose COMMITTER name is the loop's (LOOP_GIT_NAME). The
runner exports GIT_COMMITTER_NAME for the session and sets user.name in the clone, and
``git commit --author`` or ``git cherry-pick`` change only the author, so neither can hide a
commit from these checks. A loop commit whose author is someone else is itself flagged. A
commit with another committer is Anjor's only if it reached origin/main by a fetch before it
appeared on a local branch of the clone (the reflogs say so); one made in the clone under
another name (git fast-import, commit-tree, an emptied environment) is flagged.

Git. Every git command here runs with hooks and fsmonitor off, and paths are read with -z,
so a quoted or non-ASCII path cannot slip past a path rule.

Transcript. The checks read the stream-json transcript: each tool call, its result, and the
timestamps of both, which order them against commit times. Shell command lines are read as
the shell reads them (quotes, operators, here-documents, comments, ``bash -c``), so text in a
quoted argument such as a commit message is never taken for a command.

Subcommands (each accepts ``--repo DIR``; the default is the repo this file sits in):

  entry-closed --iter ID        log entry closed? exit 0 closed, 1 open, 2 missing
  outcome --iter ID             done | WIP | waiting | hard stop | stub | missing | unknown
  append-stub ...               close an iteration's log with a runner stub entry
  streak                        iterations in a row (newest first) that ended WIP or as a stub
  iter-count --day YYYYMMDD     iterations in the log: that day's count, then the total
  log-check --base REV          are the working-tree log changes since REV appends only?
  frozen-record                 (setup only) write loop/frozen.json
  frozen-check                  are the frozen sections unchanged and unambiguous?
  ledger-verify                 ledger integrity, structure, and hours against the cap
  commit-guards ...             inspect the commits a session made
  transcript-check ...          T1 gate passes, T2 GANDALF merges, T3 Modal outside the launcher, T4 ledger commits
  transcript-seal ...           record a finished transcript's sha256 (T2 trusts only sealed ones)
  apps-check ...                Modal apps that the ledger does not account for
  ignore-check                  are .loop/ and data/ ignored by a .gitignore rule?
  clone-check                   files in the clone that change what a session loads or runs
  questions-clean               does QUESTIONS.md only gain lines since HEAD?
  wait-remaining                seconds left on the WAIT file, and its reason
  stop-write ...                write STOP and a Blocking line in QUESTIONS.md
  render-prompt ...             fill in loop/prompt.md
  session-check ...             session result, failure class, rate limit and usage
  check-settings --file F       is the loop's settings.json usable?
"""
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import loopcommon as lc  # noqa: E402

DEFAULT_REPO = Path(__file__).resolve().parents[3]
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"
# Every git call: no hooks, no fsmonitor, no quoting of paths in human-readable output.
GIT_SAFE = ("-c", "core.hooksPath=/dev/null", "-c", "core.fsmonitor=false", "-c", "core.quotePath=false")

# An iteration's log entry starts at its heading, '## it-YYYYMMDD-HHMM: <block>' (or a space
# instead of the colon), and runs to the next iteration heading. Other '## ' headings inside
# an entry (a pasted gate report, say) do not end it, and nothing inside a ``` fence counts:
# an entry that quotes a dead entry in a fence neither splits nor closes there.
ENTRY_RE = re.compile(r"^## (it-\d{8}-\d{4})(?=[:\s]|$)")
ITER_RE = re.compile(r"^it-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})$")
# '- End:', '- **End:**', '- **End**:', '* End:'
END_RE = re.compile(r"^\s*[-*]\s+[*_]*End[*_]*\s*:")
OUTCOME_RE = re.compile(r"outcome[*_]*\s*:\s*(.*)$", re.IGNORECASE)
FENCE_RE = re.compile(r"^\s*```")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")

# wait.sh accepts at most 2880 minutes. A WAIT further ahead than that did not come from it.
MAX_WAIT_SEC = 2880 * 60

DEFAULT_DECISION = (
    "Read the reason, decide whether the loop may continue, and say so under the Blocking "
    "question in QUESTIONS.md."
)

# Modal code outside the launcher (commit-guards rule d, and T3). Imports of modal, of the
# launcher or of a Modal app module; the app-defining API; and a Modal command line in any
# form ('modal run' as words, or 'modal', 'run' as list items), in strings and docstrings as
# much as in code. Only whole-line '#' comments are exempt, and not in notebooks.
_MODAL_IMPORTS = (
    re.compile(r"\bimport\s+(?:[\w.]+(?:\s+as\s+\w+)?\s*,\s*)*modal\b"),
    re.compile(r"\bfrom\s+modal(?:\.[\w.]+)?\s+import\b"),
    re.compile(r"(?:import_module|__import__)\(\s*['\"]modal\b"),
)
_LAUNCHER_IMPORTS = (
    re.compile(r"^\s*\"?\s*(?:from\s+[\w.]+\s+)?import\s+[^#]*\b(?:modal_app|modal_runner|modal_launch)\b"),
    re.compile(r"^\s*\"?\s*from\s+[\w.]*\b(?:modal_app|modal_runner|modal_launch)\b"),
    re.compile(r"(?:import_module|__import__)\(\s*['\"][\w.]*(?:modal_app|modal_runner|modal_launch)"),
)
_MODAL_API = re.compile(r"\bmodal\.(?:App|Function|Cls|Sandbox)\b")
_MODAL_CMD = re.compile(r"\bmodal\s+(?:run|deploy|serve|shell)\b")
_MODAL_LIST_CMD = re.compile(r"modal['\"]\s*,\s*['\"](?:run|deploy|serve|shell)['\"]")

# Paths a loop commit must never touch (rule a): the loop's own guards and state, LOOP.md,
# and anything Claude Code loads as instructions or settings, at any depth (a .claude/
# folder, CLAUDE.md, CLAUDE.local.md).
GUARDED_PREFIXES = (lc.LOOP_REL + "/", lc.STUDY_REL + "/.loop/")
GUARDED_FILES = (lc.STUDY_REL + "/LOOP.md",)
INSTRUCTION_NAMES = ("CLAUDE.md", "CLAUDE.local.md")
OWN_PAPER = "paper/phase-space-helicity/"
GATE_REPORTS_REL = lc.STUDY_REL + "/gate_reports/"
LAUNCHER_REL = lc.LOOP_REL + "/modal_launch.py"

# Gate reports (LOOP.md section 5) as the launcher reads them (modal_launch.py): the file
# name gives the gate, a line labelled 'Result' says PASS, and the 'Critic:' line names the
# saved critic report. Labels are read after Markdown decoration and in any case.
LINE_DECORATION = " \t>*-+#|_`"
GATE_FILE_RE = re.compile(r"^G(?:(?P<n>[123])|4_(?P<set>[A-Za-z0-9]+)|(?P<base>base))_(?P<iter>it-\d{8}-\d{4})\.md$")
RESULT_PASS_RE = re.compile(r"^(?:final\s+)?result\b[^:\n]*:[\s`*_\"']*pass\b", re.IGNORECASE)
CRITIC_LABEL_RE = re.compile(r"^critic\b", re.IGNORECASE)
CRITIC_FILE_RE = re.compile(r"((?:[\w.-]+/)*critic_[\w.-]+\.md)")

CRITIC = "study04-critic"
REVIEWER = "gandalf-reviewer"
SUBAGENT_TOOLS = ("Task", "Agent")
TRANSCRIPT_NAME_RE = re.compile(r"^it-\d{8}-\d{4}\.jsonl$")
# Seconds of slack when a transcript timestamp (milliseconds) is compared with a commit time
# (whole seconds).
ORDER_SLACK_SEC = 1.5

# Run statuses the launcher may move an uncharged run between (modal_launch.py): a launch
# record starts or loses its calls, reconcile adopts an app or sees a call running.
UNCHARGED_NEXT = {
    "reserved": {"reserved", "running", "unknown"},
    "unknown": {"unknown", "running"},
    "running": {"running"},
}
CHARGE_TOLERANCE_HOURS = 0.01


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def utc_iso(epoch: Optional[float] = None) -> str:
    """UTC time as 2026-10-01T12:00:00Z."""
    t = time.time() if epoch is None else epoch
    return dt.datetime.fromtimestamp(t, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(repo: Path, *args: str, check: bool = False, safe: bool = True) -> subprocess.CompletedProcess:
    """Run git in ``repo`` (hooks and fsmonitor off unless ``safe`` is False) and capture bytes."""
    cmd = ["git", "-C", str(repo)] + (list(GIT_SAFE) if safe else []) + list(args)
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.decode(errors='replace').strip()}")
    return proc


def git_text(repo: Path, *args: str) -> str:
    """Stdout of a git command that must succeed, as text."""
    return git(repo, *args, check=True).stdout.decode("utf-8", errors="replace")


def split_z(data: bytes) -> List[str]:
    """NUL-separated git output as a list of paths (raw bytes decoded as the filesystem does)."""
    return [os.fsdecode(part) for part in data.split(b"\0") if part]


def blob_at(repo: Path, rev: str, path: str) -> Optional[bytes]:
    """Raw content of ``path`` at ``rev``, or None if it does not exist there."""
    proc = git(repo, "cat-file", "-p", f"{rev}:{path}")
    return proc.stdout if proc.returncode == 0 else None


def resolve_commit(repo: Path, rev: str) -> Optional[str]:
    """Full sha of ``rev`` as a commit, or None."""
    proc = git(repo, "rev-parse", "--verify", "-q", f"{rev}^{{commit}}")
    return proc.stdout.decode().strip() if proc.returncode == 0 else None


def own_config() -> Dict[str, str]:
    """config.env next to this file (the runner's trusted copy), or {} if absent."""
    path = HERE / "config.env"
    return lc.parse_env_file(path) if path.is_file() else {}


def repo_config(repo: Path) -> Dict[str, str]:
    """config.env in the repo, or {} if absent."""
    path = Path(repo) / lc.CONFIG_REL
    return lc.parse_env_file(path) if path.is_file() else {}


def config_value(repo: Path, key: str, default: str) -> str:
    """A config.env value: the trusted copy first, then the repo's file, then ``default``."""
    for cfg in (own_config(), repo_config(repo)):
        if cfg.get(key):
            return cfg[key]
    return default


def iter_date(ident: str) -> Optional[str]:
    """'it-20261001-1200' -> '2026-10-01'."""
    m = ITER_RE.match(ident)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def iter_epoch(ident: str) -> Optional[int]:
    """Start minute of an iteration, from its ID, as a UTC epoch."""
    m = ITER_RE.match(ident)
    if not m:
        return None
    y, mo, d, h, mi = (int(x) for x in m.groups())
    try:
        return int(dt.datetime(y, mo, d, h, mi, tzinfo=dt.timezone.utc).timestamp())
    except ValueError:
        return None


def appended_only(base: bytes, new: bytes) -> bool:
    """True if ``new`` is ``base`` with lines added at the end and nothing else changed.

    A base without a final newline may gain one, since adding a line after it needs one.
    """
    if not base:
        return True
    if base.endswith(b"\n"):
        return new.startswith(base)
    return new == base or new.startswith(base + b"\n")


def first_difference(base: bytes, new: bytes) -> str:
    """Describe the first earlier line that a change removed or altered."""
    old_lines = base.decode("utf-8", errors="replace").splitlines()
    new_lines = new.decode("utf-8", errors="replace").splitlines()
    for i, old in enumerate(old_lines):
        if i >= len(new_lines) or new_lines[i] != old:
            return f"line {i + 1} changed or removed: {short_text(old)!r}"
    return "content changed"


def short_text(text: str, limit: int = 80) -> str:
    """``text`` cut to ``limit`` characters with an ellipsis."""
    return text if len(text) <= limit else text[: limit - 3] + "..."


def print_lines(lines: Sequence[str], limit: int = 60) -> None:
    """Print at most ``limit`` lines, then a count of the rest."""
    for line in lines[:limit]:
        print(line)
    if len(lines) > limit:
        print(f"... and {len(lines) - limit} more")


# ---------------------------------------------------------------------------
# The log
# ---------------------------------------------------------------------------

class Entry:
    """One iteration's log entry: the file it is in, its iteration ID and its lines.

    ``live`` holds only the lines outside ``` fences, which are the ones that count.
    """

    __slots__ = ("path", "ident", "lines", "live")

    def __init__(self, path: Path, ident: str, lines: List[str]) -> None:
        self.path = path
        self.ident = ident
        self.lines = lines
        self.live: List[str] = list(lines)


def parse_entries(path: Path) -> List[Entry]:
    """The iteration entries of one daily log file, in file order (fence-aware)."""
    entries: List[Entry] = []
    current: Optional[Entry] = None
    in_fence = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            if current is not None:
                current.lines.append(line)
            continue
        m = None if in_fence else ENTRY_RE.match(line)
        if m:
            current = Entry(path, m.group(1), [line])
            entries.append(current)
            continue
        if current is not None:
            current.lines.append(line)
            if not in_fence:
                current.live.append(line)
    return entries


def all_entries(repo: Path) -> List[Entry]:
    """Every iteration entry in log/*.md, ordered by file name (the date) and position."""
    log_dir = Path(repo) / lc.LOG_REL
    if not log_dir.is_dir():
        return []
    out: List[Entry] = []
    for path in sorted(log_dir.glob("*.md")):
        out.extend(parse_entries(path))
    return out


def last_entry(repo: Path, ident: str) -> Optional[Entry]:
    """The last entry carrying ``ident``. If an iteration has several, the last one counts."""
    found = [e for e in all_entries(repo) if e.ident == ident]
    return found[-1] if found else None


def entry_closed(entry: Entry) -> bool:
    """True if the entry has an End line outside fences."""
    return any(END_RE.match(line) for line in entry.live)


def entry_outcome(entry: Entry) -> str:
    """Outcome from the entry's last End line: done, WIP, waiting, hard stop, stub or unknown."""
    ends = [line for line in entry.live if END_RE.match(line)]
    if not ends:
        return "unknown"
    m = OUTCOME_RE.search(ends[-1])
    if not m:
        return "unknown"
    text = m.group(1).strip().strip("*_`'\".").strip().lower()
    if "|" in text:  # the template's 'done | WIP | waiting | hard stop', not filled in
        return "unknown"
    for prefix, label in (("hard stop", "hard stop"), ("hard-stop", "hard stop"),
                          ("hardstop", "hard stop"), ("done", "done"), ("wip", "WIP"),
                          ("wait", "waiting"), ("stub", "stub")):
        if text.startswith(prefix):
            return label
    return "unknown"


def cmd_entry_closed(args: argparse.Namespace) -> int:
    """Exit 0 if the iteration's last entry is closed, 1 if open, 2 if there is none."""
    entry = last_entry(args.repo, args.iter)
    if entry is None:
        print("missing")
        return 2
    if entry_closed(entry):
        print("closed")
        return 0
    print("open")
    return 1


def cmd_outcome(args: argparse.Namespace) -> int:
    """Print the outcome of the iteration's last entry."""
    entry = last_entry(args.repo, args.iter)
    print("missing" if entry is None else entry_outcome(entry))
    return 0


def cmd_append_stub(args: argparse.Namespace) -> int:
    """Append the runner's stub entry for an iteration.

    The stub goes into the file that holds the iteration's last entry, so that it becomes the
    last entry for that ID; with no entry, into the file for the date in the ID.
    """
    repo = Path(args.repo)
    entry = last_entry(repo, args.iter)
    if entry is not None:
        path = entry.path
    else:
        day = iter_date(args.iter) or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
        path = repo / lc.LOG_REL / f"{day}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    if not text:
        text = f"# Study 04 loop log — {path.stem}\n"
    elif not text.endswith("\n"):
        text += "\n"
    if sum(1 for line in text.splitlines() if FENCE_RE.match(line)) % 2:
        text += "```\n"  # the session left a fence open; close it so the stub counts
    try:
        minutes = f"{float(args.duration_sec) / 60.0:.1f} min" if float(args.duration_sec) >= 0 else "unknown"
    except ValueError:
        minutes = "unknown"
    before = (args.before or "?")[:12]
    after = (args.after or "?")[:12]
    note = " ".join((args.note or "none").split())
    title = args.title or "runner stub (log entry not closed)"
    stub = (
        f"\n## {args.iter}: {title}\n"
        f"- Runner: exit code {args.exit_code}, duration {minutes}, commits {before}..{after}, "
        f"stash {args.stash or 'none'}, note {note}\n"
        f"- End: {utc_iso()}, outcome: stub\n"
    )
    path.write_text(text + stub, encoding="utf-8")
    try:
        print(path.relative_to(repo))
    except ValueError:
        print(path)
    return 0


def last_resume_epoch(repo: Path, loop_name: str) -> Optional[int]:
    """Commit time of the newest commit, not committed by the loop, that deleted STOP."""
    proc = git(repo, "log", "--format=%ct%x09%cn", "--diff-filter=D", "--", lc.STOP_REL)
    if proc.returncode != 0:
        return None
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        ct, _, committer = line.partition("\t")
        if committer != loop_name and ct.strip().isdigit():
            return int(ct)
    return None


def cmd_streak(args: argparse.Namespace) -> int:
    """Print N, the run of iterations (newest first) that ended WIP or as a stub.

    A waiting iteration is skipped; a done or hard-stop iteration ends the run. An entry with
    no parseable outcome counts like WIP. Only iterations that started after Anjor's last
    resume (a commit by someone other than the loop deleting STOP) count, so that resuming
    gives the loop a fresh allowance. Line 2 lists the IDs counted.
    """
    repo = Path(args.repo)
    latest: Dict[str, str] = {}
    for entry in all_entries(repo):
        latest[entry.ident] = entry_outcome(entry)
    name = args.loop_committer or config_value(repo, "LOOP_GIT_NAME", "krmhd-loop")
    since = last_resume_epoch(repo, name)
    count = 0
    counted: List[str] = []
    for ident in sorted(latest, reverse=True):
        if since is not None:
            start = iter_epoch(ident)
            if start is not None and start + 60 <= since:
                break
        outcome = latest[ident]
        if outcome == "waiting":
            continue
        if outcome in ("done", "hard stop"):
            break
        count += 1
        counted.append(ident)
    print(count)
    print(" ".join(counted))
    return 0


def cmd_iter_count(args: argparse.Namespace) -> int:
    """Print the number of distinct iterations in the log on ``--day``, then in all.

    Every session leaves an entry or a runner stub, and a loop commit cannot remove one
    without a log-edit STOP, so this count survives a cleared .loop/transcripts.
    """
    idents = {e.ident for e in all_entries(Path(args.repo))}
    day = args.day.replace("-", "")
    print(sum(1 for i in idents if i.startswith(f"it-{day}-")))
    print(len(idents))
    return 0


def cmd_log_check(args: argparse.Namespace) -> int:
    """Exit 0 if every working-tree change under log/ since ``--base`` only appends lines."""
    repo = Path(args.repo)
    problems: List[str] = []
    listing = git(repo, "ls-tree", "-r", "-z", "--name-only", args.base, "--", lc.LOG_REL)
    if listing.returncode != 0:
        print(f"cannot list {lc.LOG_REL} at {args.base}", file=sys.stderr)
        return 2
    for rel in split_z(listing.stdout):
        base = blob_at(repo, args.base, rel) or b""
        path = repo / rel
        if not path.is_file() or path.is_symlink():
            if base:
                problems.append(f"{rel}: deleted or replaced")
            continue
        new = path.read_bytes()
        if not appended_only(base, new):
            problems.append(f"{rel}: {first_difference(base, new)}")
    print_lines(problems)
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# Frozen sections and the ledger
# ---------------------------------------------------------------------------

FROZEN_KEYS = (
    f"{lc.STUDY_REL}/PLAN.md#2. Question",
    f"{lc.STUDY_REL}/PLAN.md#7. Kill criteria",
    f"{lc.STUDY_REL}/SPEC.md#7. Tolerances",
)


def count_matching_headings(text: str, prefix: str) -> int:
    """How many headings outside ``` fences match ``prefix`` (loopcommon.heading_matches)."""
    n = 0
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING_RE.match(line)
        if m and lc.heading_matches(m.group(2), prefix):
            n += 1
    return n


def frozen_text_problems(key: str, recorded: str, content: Optional[bytes]) -> List[str]:
    """Problems with one frozen item, given the content of its file (None = missing)."""
    rel, sep, heading = key.partition("#")
    if content is None:
        return [f"frozen item missing: {key}"]
    if not sep:
        current = hashlib.sha256(content).hexdigest()
    else:
        text = content.decode("utf-8", errors="replace")
        n = count_matching_headings(text, heading)
        if n > 1:
            return [f"frozen heading appears {n} times: {key} (the frozen section is ambiguous)"]
        section = lc.extract_section(text, heading)
        if section is None:
            return [f"frozen item missing: {key}"]
        current = lc.sha256_text(section)
    if current != recorded:
        return [f"frozen item changed: {key} (recorded {recorded[:12]}, now {current[:12]})"]
    return []


def trusted_manifest() -> Tuple[Dict[str, str], List[str]]:
    """Entries of the trusted frozen.json next to this file, and any problem reading it."""
    path = HERE / "frozen.json"
    if not path.is_file():
        return {}, [f"manifest {path} missing, so no frozen section can be checked"]
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")).get("entries", {})), []
    except (ValueError, AttributeError) as exc:
        return {}, [f"manifest {path} unreadable: {exc}"]


def cmd_frozen_record(args: argparse.Namespace) -> int:
    """Setup only: hash PLAN.md §2 and §7 and SPEC.md §7 into loop/frozen.json."""
    repo = Path(args.repo)
    entries: Dict[str, str] = {}
    for key in FROZEN_KEYS:
        sha = lc.frozen_key_hash(repo, key)
        if sha is None:
            print(f"frozen-record: cannot resolve {key}", file=sys.stderr)
            return 1
        entries[key] = sha
    data = {
        "note": ("Hashes of the frozen sections (PLAN.md §2, PLAN.md §7, SPEC.md §7), recorded at "
                 "loop setup. The runner writes STOP if the current text no longer matches. Only "
                 "Anjor changes this file."),
        "recorded_utc": utc_iso(),
        "entries": entries,
    }
    out = repo / lc.FROZEN_REL
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {lc.FROZEN_REL}: {len(entries)} entries")
    return 0


def cmd_frozen_check(args: argparse.Namespace) -> int:
    """Compare frozen sections with frozen.json (trusted copy) and the ledger's launch freezes."""
    repo = Path(args.repo)
    problems: List[str] = []
    if args.manifest:
        try:
            entries = dict(json.loads(Path(args.manifest).read_text(encoding="utf-8")).get("entries", {}))
            errors: List[str] = []
        except (OSError, ValueError, AttributeError) as exc:
            entries, errors = {}, [f"manifest {args.manifest} unreadable: {exc}"]
    else:
        entries, errors = trusted_manifest()
    problems.extend(f"frozen: {e}" for e in errors)
    try:
        entries.update(lc.ledger_frozen_entries(lc.load_ledger(repo / lc.LEDGER_REL)))
    except Exception as exc:  # any malformed ledger: report it, never skip silently
        problems.append(f"frozen: ledger unreadable, launch freezes not checked: {exc!r}")
    for key, recorded in sorted(entries.items()):
        path = repo / key.partition("#")[0]
        content = path.read_bytes() if path.is_file() else None
        problems.extend(f"frozen: {p}" for p in frozen_text_problems(key, recorded, content))
    print_lines(problems)
    return 1 if problems else 0


def cmd_ledger_verify(args: argparse.Namespace) -> int:
    """verify_ledger on the working-tree ledger, plus: used + reserved must not exceed the cap."""
    repo = Path(args.repo)
    problems: List[str] = []
    try:
        ledger = lc.load_ledger(repo / lc.LEDGER_REL)
        problems.extend(f"ledger: {p}" for p in lc.verify_ledger(ledger))
    except Exception as exc:  # malformed JSON or structure
        print(f"ledger: unreadable: {exc!r}")
        return 1
    cap_text = config_value(repo, "COMPUTE_CAP_A100_HOURS", "")
    try:
        cap = float(cap_text)
    except ValueError:
        problems.append(f"ledger: no usable COMPUTE_CAP_A100_HOURS in config.env ({cap_text!r})")
    else:
        try:
            totals = lc.ledger_totals(ledger)
        except Exception as exc:
            problems.append(f"ledger: hours unreadable: {exc!r}")
        else:
            if totals["used"] + totals["reserved"] > cap + 1e-9:
                problems.append(
                    f"ledger: used {totals['used']:.2f} + reserved {totals['reserved']:.2f} A100-h "
                    f"exceeds the cap {cap:.1f}")
    print_lines(problems)
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# Commits
# ---------------------------------------------------------------------------

class Commit:
    """One commit of a range: sha, parents, author and committer names, commit time, subject."""

    __slots__ = ("sha", "parents", "author", "committer", "time", "subject")

    def __init__(self, sha: str, parents: List[str], author: str, committer: str, time_: int,
                 subject: str) -> None:
        self.sha = sha
        self.parents = parents
        self.author = author
        self.committer = committer
        self.time = time_
        self.subject = subject


def list_commits(repo: Path, before: str, after: str) -> List[Commit]:
    """Commits in before..after, parents before children.

    A commit's time is the earlier of its author and committer dates: a rebase (the session's
    pull --rebase, or the runner's sync after it) gives a commit a new committer date but
    keeps its author date, which is when it was made.
    """
    out = git(repo, "log", "-z", "--reverse", "--topo-order",
              "--format=%H%x1f%P%x1f%an%x1f%cn%x1f%at%x1f%ct%x1f%s", f"{before}..{after}", check=True).stdout
    commits = []
    for record in out.split(b"\0"):
        record = record.strip(b"\n")
        if not record:
            continue
        fields = record.decode("utf-8", errors="replace").split("\x1f") + ["", "", "", "", "", "", ""]
        stamps = [int(x) for x in (fields[4], fields[5]) if x.isdigit()]
        commits.append(Commit(fields[0], fields[1].split(), fields[2], fields[3], min(stamps) if stamps else 0,
                              fields[6]))
    return commits


def name_status(repo: Path, base: str, commit: str) -> Dict[str, str]:
    """{path: status letter} for base..commit, renames split into delete and add."""
    proc = git(repo, "diff", "--name-status", "-z", "--no-renames", "--no-textconv", "--no-ext-diff",
               base, commit)
    if proc.returncode != 0:
        raise RuntimeError(f"git diff {base[:12]} {commit[:12]} failed: "
                           f"{proc.stderr.decode(errors='replace').strip()}")
    fields = proc.stdout.split(b"\0")
    out: Dict[str, str] = {}
    for i in range(0, len(fields) - 1, 2):
        status = fields[i].decode("ascii", errors="replace").strip()
        path = os.fsdecode(fields[i + 1])
        if status and path:
            out[path] = status[:1]
    return out


def changed_paths(repo: Path, commit: str, parents: Sequence[str]) -> List[Tuple[str, str]]:
    """(status, path) pairs a commit introduced.

    For a merge, only paths that differ from every parent count: the merge's own changes.
    What it brings in from a side branch is checked on the side branch's commits, and what it
    drops from a side is checked by the merge-discard rule.
    """
    bases = list(parents) or [EMPTY_TREE]
    per_parent = [name_status(repo, base, commit) for base in bases]
    common = set(per_parent[0])
    for changes in per_parent[1:]:
        common &= set(changes)
    return sorted((per_parent[0][p], p) for p in common)


def added_lines(repo: Path, base: str, commit: str, path: str) -> List[str]:
    """Lines that ``commit`` adds to ``path`` relative to ``base``."""
    proc = git(repo, "diff", "-U0", "--no-color", "--no-renames", "--no-textconv", "--no-ext-diff",
               base, commit, "--", ":(literal)" + path)
    lines: List[str] = []
    in_hunk = False
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith("diff --git"):
            in_hunk = False
        elif line.startswith("@@"):
            in_hunk = True
        elif in_hunk and line.startswith("+"):
            lines.append(line[1:])
    return lines


def code_kind(path: str, content: bytes) -> Optional[str]:
    """'py', 'sh' or 'nb' for a file the Modal rule reads, None for any other file."""
    if path.endswith((".py", ".pyw")):
        return "py"
    if path.endswith((".sh", ".bash", ".zsh")):
        return "sh"
    if path.endswith(".ipynb"):
        return "nb"
    name = path.rsplit("/", 1)[-1]
    if "." not in name and content.startswith(b"#!"):
        first = content.split(b"\n", 1)[0]
        if b"python" in first:
            return "py"
        if re.search(rb"\b(?:ba|z|k|da)?sh\b", first):
            return "sh"
    return None


def modal_line_hits(kind: str, lines: Iterable[str]) -> List[str]:
    """Lines that import or drive Modal, outside whole-line comments (none exempt in notebooks)."""
    hits = []
    for line in lines:
        stripped = line.strip()
        if kind != "nb" and stripped.startswith("#"):
            continue
        if (any(p.search(line) for p in _MODAL_IMPORTS) or any(p.search(line) for p in _LAUNCHER_IMPORTS)
                or _MODAL_API.search(line) or _MODAL_CMD.search(line) or _MODAL_LIST_CMD.search(line)):
            hits.append(stripped)
    return hits


def imports_modal(kind: str, text: str) -> bool:
    """True if a code file imports Modal, the launcher or a Modal app module."""
    for line in text.splitlines():
        if kind != "nb" and line.strip().startswith("#"):
            continue
        if any(p.search(line) for p in _MODAL_IMPORTS) or any(p.search(line) for p in _LAUNCHER_IMPORTS):
            return True
    return False


def under_log(path: str) -> bool:
    """True for a path in the study's log/ folder."""
    return path.startswith(lc.LOG_REL + "/")


def guarded(path: str) -> bool:
    """Rule a: loop/, .loop/, LOOP.md, and any .claude/ folder, CLAUDE.md or CLAUDE.local.md."""
    if path.startswith(GUARDED_PREFIXES) or path in GUARDED_FILES:
        return True
    parts = path.split("/")
    return ".claude" in parts[:-1] or parts[-1] in INSTRUCTION_NAMES


def other_study(path: str) -> bool:
    """Rule f: another study's folder under studies/, or another paper under paper/."""
    if path.startswith("studies/") and not path.startswith(lc.STUDY_REL + "/"):
        return True
    return path.startswith("paper/") and not path.startswith(OWN_PAPER)


def protected(path: str) -> bool:
    """Paths a loop merge must keep exactly as the side that changed them left them."""
    return guarded(path) or path in (lc.STOP_REL, lc.LEDGER_REL)


def parse_ledger_blob(blob: Optional[bytes]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """(ledger, None) for a parseable ledger blob, an empty ledger for None, else (None, error)."""
    if blob is None:
        return lc.empty_ledger(), None
    try:
        data = json.loads(blob.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        return None, f"not JSON ({exc})"
    if not isinstance(data, dict):
        return None, "not a JSON object"
    return data, None


class Ancestry:
    """Cached ``git merge-base --is-ancestor``."""

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.cache: Dict[Tuple[str, str], bool] = {}

    def is_ancestor(self, a: str, b: str) -> bool:
        """True if commit ``a`` is an ancestor of (or equal to) ``b``."""
        key = (a, b)
        if key not in self.cache:
            self.cache[key] = git(self.repo, "merge-base", "--is-ancestor", a, b).returncode == 0
        return self.cache[key]


def judge_commit(repo: Path, c: Commit, loop: str, manifest: Dict[str, str],
                 log_baseline: Callable[[str, str], Optional[bytes]], reported: Set[str],
                 start_launches: Set[str], start_reports: Set[str]) -> List[str]:
    """Every rule for one loop commit.

    ``log_baseline(path, sha)`` gives the log's baseline; ``start_launches`` holds the launch
    IDs the ledger had when the session started, ``start_reports`` the files then under
    gate_reports/.
    """
    problems: List[str] = []
    short = c.sha[:12]
    subject = short_text(c.subject, 100)
    changes = changed_paths(repo, c.sha, c.parents)
    paths = [p for _, p in changes]
    first_parent = c.parents[0] if c.parents else EMPTY_TREE
    if c.author != loop:
        problems.append(f"{short} identity: committed by {loop} with author '{c.author}' in '{subject}'; "
                        "a loop commit must carry the loop's name (git commit --author or a cherry-pick?)")
    for status, path in changes:
        if guarded(path):
            problems.append(f"{short} guarded-path: {path} ({status}) in '{subject}'")
        if other_study(path):
            problems.append(f"{short} other-study: {path} ({status}) in '{subject}'")
        if path == lc.STOP_REL and status == "D":
            problems.append(f"{short} stop-deleted: '{subject}' deletes STOP; only Anjor resumes the loop")
        if path in start_reports and status in ("M", "D", "T"):
            problems.append(f"{short} report-edit: {path} ({status}) in '{subject}'; a committed gate or critic "
                            "report is never edited or deleted, a new evaluation gets a new file")
        if under_log(path):
            base = log_baseline(path, c.sha)
            new = None if status == "D" else blob_at(repo, c.sha, path)
            if base:
                if new is None:
                    problems.append(f"{short} log-edit: {path} deleted in '{subject}'")
                elif not appended_only(base, new):
                    problems.append(f"{short} log-edit: {path} {first_difference(base, new)} in '{subject}'")
        if status != "D" and not path.startswith(lc.LOOP_REL + "/"):
            kind = code_kind(path, b"")
            content = None
            if kind is None and "." not in path.rsplit("/", 1)[-1]:
                content = blob_at(repo, c.sha, path)  # an extensionless script, by its shebang
                kind = code_kind(path, content[:256]) if content is not None else None
            if kind is not None and content is None:
                content = blob_at(repo, c.sha, path)
            if kind is not None and content is not None:
                hits = modal_line_hits(kind, added_lines(repo, first_parent, c.sha, path))
                for hit in hits[:3]:
                    problems.append(f"{short} modal-code: {path} adds {short_text(hit)!r}")
                if not hits and imports_modal(kind, content.decode("utf-8", errors="replace")):
                    problems.append(f"{short} modal-code: {path} is changed and imports Modal (a GPU app "
                                    "outside the launcher)")
    if lc.LEDGER_REL in paths:
        status_of = {path: status for status, path in changes}
        problems.extend(ledger_commit_problems(repo, c, status_of[lc.LEDGER_REL], paths, start_launches))
    problems.extend(frozen_commit_problems(repo, c, manifest, set(paths), reported))
    if len(c.parents) > 1:
        problems.extend(merge_discard_problems(repo, c))
    return problems


def _hours(value: Any) -> Optional[float]:
    """A ledger hour value as a float, or None (absent, not a number, NaN)."""
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if x == x else None


def charge_problems(launch: Dict[str, Any], run: Dict[str, Any]) -> List[str]:
    """Is a run's first charge one the launcher could have made (modal_launch.py reconcile)?

    A smoke run and a run never sent to Modal cost 0. A run without attempt records is charged
    its full reservation. Otherwise the charge lies between the hours of the complete attempts
    and that plus one timeout per open attempt (charge_from_records). The records themselves
    come from the volume and cannot be checked here.
    """
    rid = run.get("run_id")
    charged = _hours(run.get("charged_hours"))
    status = run.get("status")
    if charged is None:
        return [f"{rid}: its charge {run.get('charged_hours')!r} is not a number"]
    if status == "not_launched" or launch.get("kind") == "smoke":
        if abs(charged) > 1e-9:
            return [f"{rid}: charged {charged:.2f} A100-h, but a smoke run or a run never sent to Modal costs 0"]
        return []
    if status not in ("finished", "failed", "unknown"):
        return [f"{rid}: charged while its status is {status!r}; the launcher charges a run only once it is over"]
    timeout = _hours(launch.get("timeout_hours")) or 0.0
    attempts = run.get("attempts")
    if not isinstance(attempts, list):
        return [f"{rid}: its attempts are not a list"]
    if not attempts:
        expected = _hours(run.get("reserved_hours")) or timeout
        if abs(charged - expected) > CHARGE_TOLERANCE_HOURS:
            return [f"{rid}: charged {charged:.2f} A100-h with no attempt records; with none the launcher "
                    f"charges the full reservation, {expected:.2f}"]
        return []
    complete, n_open = 0.0, 0
    for row in attempts:
        hours = _hours(row.get("hours")) if isinstance(row, dict) else None
        if hours is None:
            n_open += 1
        else:
            complete += max(0.0, hours)
    low, high = complete, complete + n_open * timeout
    if not low - CHARGE_TOLERANCE_HOURS <= charged <= high + CHARGE_TOLERANCE_HOURS:
        return [f"{rid}: charged {charged:.2f} A100-h, but its attempt records ({complete:.2f} h complete, "
                f"{n_open} open, timeout {timeout:g} h) allow only {low:.2f} to {high:.2f}"]
    return []


def ledger_semantics(prev: Dict[str, Any], new: Dict[str, Any], start_launches: Set[str]) -> List[str]:
    """What loopcommon.ledger_extends does not check: changes only the launcher makes.

    - A launch new in this commit arrives as the launcher's reservation: state reserved, no
      app, every run reserved, uncharged and without a call.
    - A launch is marked launch_failed, and a run not_launched, only by the launch command
      that reserved it, so only for a launch reserved during this session. A reservation
      left from an earlier session is released by Anjor, never by the loop.
    - An uncharged run moves only from reserved to running or unknown, or from unknown to
      running.
    - A first charge is one reconcile could have made (charge_problems).
    """
    problems: List[str] = []
    old = {str(x.get("launch_id")): x for x in prev.get("launches", []) or [] if isinstance(x, dict)}
    for launch in new.get("launches", []) or []:
        if not isinstance(launch, dict):
            continue
        lid = str(launch.get("launch_id"))
        runs = [r for r in launch.get("runs", []) or [] if isinstance(r, dict)]
        before = old.get(lid)
        if before is None:
            if launch.get("state") != "reserved" or launch.get("app_id"):
                problems.append(f"{lid}: a new launch must arrive as a reservation (state reserved, no app "
                                f"ID), not in state {launch.get('state')!r}")
            for run in runs:
                if run.get("status") != "reserved" or run.get("charged_hours") is not None or run.get("call_id"):
                    problems.append(f"{run.get('run_id')}: a run of a new launch must be reserved, uncharged "
                                    "and without a call")
            continue
        fresh = lid not in start_launches
        if launch.get("state") == "launch_failed" and before.get("state") != "launch_failed" and not fresh:
            problems.append(f"{lid}: marked launch_failed, but it was reserved before this session; only the "
                            "launch command that reserved it releases a launch, and later only Anjor does")
        prev_runs = {str(r.get("run_id")): r for r in before.get("runs", []) or [] if isinstance(r, dict)}
        for run in runs:
            was = prev_runs.get(str(run.get("run_id")))
            if was is None or was.get("charged_hours") is not None:
                continue  # ledger_extends reports changed runs and changed charges
            rid = run.get("run_id")
            if run.get("charged_hours") is None:
                allowed = UNCHARGED_NEXT.get(str(was.get("status")), {was.get("status")})
                if run.get("status") not in allowed:
                    problems.append(f"{rid}: status went from {was.get('status')!r} to {run.get('status')!r} "
                                    "without a charge, which the launcher never does")
                continue
            if run.get("status") == "not_launched" and (was.get("status") != "reserved" or not fresh):
                problems.append(f"{rid}: released as not_launched, but it was {was.get('status')!r} "
                                + ("" if fresh else "and reserved before this session")
                                + "; only the launch command that reserved it releases a run")
            problems.extend(charge_problems(launch, run))
    return problems


def ledger_commit_problems(repo: Path, c: Commit, status: str, paths: List[str],
                           start_launches: Set[str]) -> List[str]:
    """Rule c: a loop change of the ledger is a launcher-made successor, alone, with its subject."""
    short = c.sha[:12]
    problems: List[str] = []
    if not c.subject.startswith(lc.LAUNCHER_SUBJECT_PREFIX):
        problems.append(f"{short} ledger: '{short_text(c.subject, 100)}' changes {lc.LEDGER_REL} without "
                        f"the launcher's subject '{lc.LAUNCHER_SUBJECT_PREFIX}'")
    others = [p for p in paths if p != lc.LEDGER_REL]
    if others:
        problems.append(f"{short} ledger: ledger commit also touches {', '.join(others[:5])}")
    if status == "D":
        problems.append(f"{short} ledger: {lc.LEDGER_REL} deleted; only Anjor resets the ledger")
        return problems
    new, err = parse_ledger_blob(blob_at(repo, c.sha, lc.LEDGER_REL))
    if new is None:
        problems.append(f"{short} ledger: {lc.LEDGER_REL} is {err}")
        return problems
    problems.extend(f"{short} ledger: {p}" for p in lc.verify_ledger(new))
    for parent in c.parents or [EMPTY_TREE]:
        prev_blob = blob_at(repo, parent, lc.LEDGER_REL) if parent != EMPTY_TREE else None
        prev, err = parse_ledger_blob(prev_blob)
        if prev is None:
            problems.append(f"{short} ledger: the parent's ledger ({parent[:12]}) is {err}, so this "
                            "change cannot be shown to extend it")
            continue
        try:
            succession = lc.ledger_extends(prev, new) + ledger_semantics(prev, new, start_launches)
        except Exception as exc:  # malformed structure inside: not a launcher-made successor
            succession = [f"cannot compare with the parent's ledger: {exc!r}"]
        for p in succession[:6]:
            problems.append(f"{short} ledger: not a launcher-made successor of {parent[:12]}: {p}")
    return problems


def frozen_commit_problems(repo: Path, c: Commit, manifest: Dict[str, str], touched: Set[str],
                           reported: Set[str]) -> List[str]:
    """Rule g: a frozen item changed (or became ambiguous) in this commit, even if later reverted."""
    entries = dict(manifest)
    ledger, _ = parse_ledger_blob(blob_at(repo, c.sha, lc.LEDGER_REL))
    if ledger is not None:
        try:
            entries.update(lc.ledger_frozen_entries(ledger))
        except Exception:  # the ledger rule reports a broken ledger
            pass
    problems: List[str] = []
    for key, recorded in sorted(entries.items()):
        rel = key.partition("#")[0]
        if rel not in touched:
            continue
        for p in frozen_text_problems(key, recorded, blob_at(repo, c.sha, rel)):
            if p in reported:
                continue
            reported.add(p)
            problems.append(f"{c.sha[:12]} frozen: {p} in '{short_text(c.subject, 100)}'")
    return problems


def merge_discard_problems(repo: Path, c: Commit) -> List[str]:
    """Rule h: a loop merge that does not keep a side's change to a protected path."""
    proc = git(repo, "merge-base", "--octopus", *c.parents)
    base = proc.stdout.decode().strip()
    if proc.returncode != 0 or not base:
        return [f"{c.sha[:12]} merge-discard: the merge's parents have no merge base"]
    problems: List[str] = []
    seen: Set[str] = set()
    for parent in c.parents:
        for path in name_status(repo, base, parent):
            if path in seen or not protected(path):
                continue
            if blob_at(repo, c.sha, path) != blob_at(repo, parent, path):
                seen.add(path)
                problems.append(f"{c.sha[:12]} merge-discard: the merge '{short_text(c.subject, 80)}' does "
                                f"not keep {path} as {parent[:12]} changed it")
    return problems


REFLOG_LINE_RE = re.compile(rb"^[0-9a-f]{40} ([0-9a-f]{40}) .*? (\d+) [+-]\d{4}\t(.*)$")


def reflog_entries(repo: Path, since: float) -> Tuple[List[Tuple[int, str]], List[Tuple[int, str]],
                                                         List[Tuple[int, str]]]:
    """Reflog entries since ``since``: (time, new sha) on local refs (HEAD and branches), on
    remote-tracking refs by a fetch or pull, and on remote-tracking refs by a push.

    Each list is sorted by time and holds one entry per sha, its earliest.
    """
    proc = git(repo, "rev-parse", "--git-path", "logs/HEAD", "--git-path", "logs/refs")
    paths = proc.stdout.decode("utf-8", errors="replace").splitlines()
    if proc.returncode != 0 or len(paths) != 2:
        raise RuntimeError("cannot locate the reflogs")
    head_log, refs_log = (Path(p) if os.path.isabs(p) else Path(repo) / p for p in paths)
    files = [("local", head_log)]
    for kind, sub in (("local", "heads"), ("remote", "remotes")):
        base = refs_log / sub
        if base.is_dir():
            files.extend((kind, f) for f in sorted(base.rglob("*")) if f.is_file())
    found: Dict[str, Dict[str, int]] = {"local": {}, "fetch": {}, "push": {}}
    for kind, path in files:
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        for line in raw.splitlines():
            m = REFLOG_LINE_RE.match(line)
            if not m or int(m.group(2)) < since:
                continue
            sha, when = m.group(1).decode(), int(m.group(2))
            if kind == "remote":
                kind_now = "push" if m.group(3).startswith(b"update by push") else "fetch"
            else:
                kind_now = "local"
            seen = found[kind_now]
            if sha not in seen or when < seen[sha]:
                seen[sha] = when
    return tuple(sorted((t, s) for s, t in found[k].items()) for k in ("local", "fetch", "push"))  # type: ignore[return-value]


def provenance_problems(repo: Path, commits: Sequence[Commit], since: float, ancestry: Ancestry) -> List[str]:
    """Rule provenance: a commit with another committer must have come from origin.

    Anjor's commits reach the clone only by a fetch: the remote-tracking ref holds the commit
    before any local branch does, and never first by a push from this clone. A commit that a
    local branch held first (git commit, fast-import, commit-tree with reset), or that origin
    first got by a push from here, was made in the clone under another name.
    """
    if not commits:
        return []
    try:
        local, fetched, pushed = reflog_entries(repo, since - 5)
    except RuntimeError as exc:
        return [f"{c.sha[:12]} provenance: {exc}, so the origin of '{short_text(c.subject, 80)}' (committed "
                f"by '{c.committer}') cannot be checked" for c in commits]

    def first(entries: List[Tuple[int, str]], sha: str) -> Optional[int]:
        for when, new in entries:
            if ancestry.is_ancestor(sha, new):
                return when
        return None

    problems: List[str] = []
    for c in commits:
        t_fetch, t_local, t_push = first(fetched, c.sha), first(local, c.sha), first(pushed, c.sha)
        if t_fetch is not None and (t_local is None or t_fetch <= t_local) and (t_push is None or t_fetch <= t_push):
            continue
        how = ("never came from origin by a fetch" if t_fetch is None else
               "was on a local branch before a fetch brought it" if t_local is not None and t_local < t_fetch else
               "first reached origin by a push from the clone")
        problems.append(f"{c.sha[:12]} provenance: committed by '{c.committer}', not the loop, but it {how}; "
                        f"a commit made in the clone under another name: '{short_text(c.subject, 100)}'")
    return problems


def ledger_launch_ids(repo: Path, rev: str) -> Set[str]:
    """Launch IDs in the ledger at ``rev`` (empty if it has none or cannot be read)."""
    ledger, _ = parse_ledger_blob(blob_at(repo, rev, lc.LEDGER_REL))
    if ledger is None:
        return set()
    return {str(x.get("launch_id")) for x in ledger.get("launches", []) or [] if isinstance(x, dict)}


def cmd_commit_guards(args: argparse.Namespace) -> int:
    """Inspect every loop commit (committer = the loop) in before..after. Exit 1 on any violation.

    Rules, each printed as '<short sha> <rule>: <detail>':
      identity      a loop commit whose author is not the loop (--author, cherry-pick)
      guarded-path  touches loop/, .loop/, LOOP.md, or any .claude/, CLAUDE.md, CLAUDE.local.md
      log-edit      changes or removes a line of the log that existed when the iteration
                    started, or that Anjor wrote since (lines the iteration added may be revised)
      ledger        touches compute_ledger.json other than as the launcher does: the subject,
                    the ledger alone, a successor of the parent's version (ledger_extends and
                    ledger_semantics), never a deletion
      modal-code    adds Modal code to a code file outside loop/, or edits one that imports Modal
      stop-deleted  deletes STOP (only Anjor resumes the loop)
      report-edit   changes or deletes a gate or critic report committed before the session
      other-study   touches another study's folder or another paper
      frozen        a frozen item changed or became ambiguous in a commit, even if later reverted
      merge-discard a loop merge drops a side's change to a guarded path, STOP or the ledger
      provenance    (with --session-start) a commit with another committer made in the clone
      history       the session's starting commit is not an ancestor of the end
    Commits by anyone else are not judged; their changes to the log become the baseline of
    the loop commits that descend from them.
    """
    repo = Path(args.repo)
    loop = args.loop_committer
    problems: List[str] = []
    resolved = {}
    for name, rev in (("before", args.before), ("after", args.after)):
        sha = resolve_commit(repo, rev)
        if sha is None:
            print(f"{rev[:12]} history: cannot resolve --{name} {rev}")
            return 1
        resolved[name] = sha
    before, after = resolved["before"], resolved["after"]
    if before == after:
        return 0
    ancestry = Ancestry(repo)
    if not ancestry.is_ancestor(before, after):
        problems.append(f"{after[:12]} history: {before[:12]} (HEAD when the session started) is not "
                        f"an ancestor of {after[:12]}; pushed history was rewritten")
    manifest, _ = trusted_manifest()
    nonloop_logs: List[Tuple[str, str, Optional[bytes]]] = []
    base_cache: Dict[str, Optional[bytes]] = {}

    def log_baseline(path: str, sha: str) -> Optional[bytes]:
        """The log file as the loop commit ``sha`` must extend: BEFORE, or Anjor's newer version it descends from."""
        for other, p, blob in reversed(nonloop_logs):
            if p == path and ancestry.is_ancestor(other, sha):
                return blob
        if path not in base_cache:
            base_cache[path] = blob_at(repo, before, path)
        return base_cache[path]

    reported: Set[str] = set()
    start_launches = ledger_launch_ids(repo, before)
    start_reports = set(split_z(git(repo, "ls-tree", "-r", "-z", "--name-only", before, "--",
                                    GATE_REPORTS_REL).stdout))
    others: List[Commit] = []
    for c in list_commits(repo, before, after):
        if c.committer != loop:
            others.append(c)
            for status, path in changed_paths(repo, c.sha, c.parents):
                if under_log(path):
                    nonloop_logs.append((c.sha, path, None if status == "D" else blob_at(repo, c.sha, path)))
            continue
        problems.extend(judge_commit(repo, c, loop, manifest, log_baseline, reported, start_launches,
                                     start_reports))
    if args.session_start:
        try:
            since = float(args.session_start)
        except ValueError:
            print(f"bad --session-start {args.session_start!r}")
            return 1
        problems.extend(provenance_problems(repo, others, since, ancestry))
    print_lines(problems)
    return 1 if problems else 0


# ---------------------------------------------------------------------------
# The session transcript
# ---------------------------------------------------------------------------

ISO_TIME_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.(\d{1,9}))?(?:Z|\+00:00)$")


def parse_timestamp(value: Any) -> Optional[float]:
    """An event's 'timestamp' (for example '2026-10-01T22:14:24.236Z') as a UTC epoch, or None."""
    if not isinstance(value, str):
        return None
    m = ISO_TIME_RE.match(value.strip())
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x) for x in m.groups()[:6])
    frac = float("0." + m.group(7)) if m.group(7) else 0.0
    try:
        return dt.datetime(y, mo, d, h, mi, s, tzinfo=dt.timezone.utc).timestamp() + frac
    except ValueError:
        return None


class Call:
    """One tool call in a transcript: its input, its result, where both appeared and when."""

    __slots__ = ("id", "name", "input", "result", "is_error", "denied", "use_line", "result_line",
                 "use_ts", "result_ts")

    def __init__(self, cid: str, name: str, inp: Dict[str, Any], use_line: int, use_ts: Optional[float]) -> None:
        self.id = cid
        self.name = name
        self.input = inp
        self.result: Optional[str] = None
        self.is_error: Optional[bool] = None
        self.denied = False
        self.use_line = use_line
        self.result_line = -1
        self.use_ts = use_ts
        self.result_ts: Optional[float] = None

    def subagent(self) -> str:
        """The subagent type of a Task/Agent call, or ''."""
        if self.name not in SUBAGENT_TOOLS:
            return ""
        return str(self.input.get("subagent_type", "") or "")

    def request(self) -> str:
        """A subagent call's request: its description and prompt."""
        return f"{self.input.get('description', '') or ''}\n{self.input.get('prompt', '') or ''}"

    def command(self) -> str:
        """A Bash call's command line."""
        return str(self.input.get("command", "") or "")


def finished_by(call: Call, when: float) -> bool:
    """Had the call's result arrived by ``when`` (an epoch)? An unknown time counts as yes."""
    return call.result_ts is None or call.result_ts <= when + ORDER_SLACK_SEC


def started_by(call: Call, when: float) -> bool:
    """Had the call started by ``when`` (an epoch)? An unknown time counts as yes."""
    return call.use_ts is None or call.use_ts <= when + ORDER_SLACK_SEC


_HANDBACK_START = "[Subagent hand-back]"
_HANDBACK_END = "The report follows:"


def strip_handback(text: str) -> str:
    """A subagent's own report, without the frame Claude Code puts around it.

    Claude Code 2.1.287 returns a subagent's final text to the session as a tool_result that
    starts with a '[Subagent hand-back] ... The report follows:' paragraph, can carry
    '[harness: ...]' notes, and indents every line of the report (seen on this Mac,
    2026-10-02). The agent saves the report itself, so T1 must compare the saved file with
    the report and not with the frame. Text without the frame comes back unchanged.
    """
    if not text.lstrip().startswith(_HANDBACK_START):
        return text
    cut = text.find(_HANDBACK_END)
    if cut < 0:
        return text
    lines = text[cut + len(_HANDBACK_END):].split("\n")
    while lines and (not lines[0].strip() or lines[0].strip().startswith("[harness:")):
        lines.pop(0)
    if not lines:
        return ""
    # The report is the indented block. The harness appends its own unindented lines after
    # it ('agentId: ...', '<usage>...</usage>'), which are not part of the report.
    indent = lines[0][:len(lines[0]) - len(lines[0].lstrip())]
    report = []
    for line in lines:
        if line.strip() and indent and not line.startswith(indent):
            break
        report.append(line)
    return textwrap.dedent("\n".join(report)).strip("\n")


def content_text(content: Any) -> str:
    """Text of a tool_result content: a string, or a list of {type: text, text} parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(content_text(item["content"]))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""


class Transcript:
    """The tool calls and the last rate-limit report of one stream-json transcript."""

    def __init__(self) -> None:
        self.calls: List[Call] = []
        self.rate: Optional[Dict[str, Any]] = None


def parse_transcript(path: Path) -> Transcript:
    """Parse a stream-json transcript. Raises OSError if it cannot be read.

    Calls made inside subagents are included: their events carry parent_tool_use_id.
    """
    tr = Transcript()
    calls: Dict[str, Call] = {}
    denied: Set[str] = set()
    with open(path, "rb") as fh:
        for index, raw in enumerate(fh):
            if not (b"tool_use" in raw or b"tool_result" in raw or b"rate_limit_event" in raw
                    or b"permission_denied" in raw):
                continue
            try:
                event = json.loads(raw.decode("utf-8", errors="replace"))
            except ValueError:
                continue
            if not isinstance(event, dict):
                continue
            kind = event.get("type")
            if kind == "rate_limit_event":
                if isinstance(event.get("rate_limit_info"), dict):
                    tr.rate = event["rate_limit_info"]
            elif kind == "system" and event.get("subtype") == "permission_denied":
                if event.get("tool_use_id"):
                    denied.add(str(event["tool_use_id"]))
            elif kind in ("assistant", "user"):
                message = event.get("message") or {}
                content = message.get("content") if isinstance(message, dict) else None
                if not isinstance(content, list):
                    continue
                stamp = parse_timestamp(event.get("timestamp"))
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if kind == "assistant" and item.get("type") == "tool_use":
                        cid = str(item.get("id", ""))
                        inp = item.get("input") if isinstance(item.get("input"), dict) else {}
                        call = Call(cid, str(item.get("name", "")), inp, index, stamp)
                        calls[cid] = call
                        tr.calls.append(call)
                    elif kind == "user" and item.get("type") == "tool_result":
                        call = calls.get(str(item.get("tool_use_id", "")))
                        if call is not None:
                            call.result = content_text(item.get("content"))
                            if call.name in SUBAGENT_TOOLS:
                                call.result = strip_handback(call.result)
                            call.is_error = bool(item.get("is_error"))
                            call.result_line = index
                            call.result_ts = stamp
    for call in tr.calls:
        if call.id in denied or (call.is_error and call.result and "Permission to use" in call.result
                                 and "has been denied" in call.result):
            call.denied = True
    return tr


def first_verdict(text: str) -> str:
    """The first 'VERDICT: X' in a report, as X ('' if none)."""
    m = re.search(r"VERDICT:\s*(SUPPORTED|REFUTED|INCONCLUSIVE|APPROVE|REQUEST CHANGES)", text)
    return m.group(1) if m else ""


def subagent_calls(tr: Transcript, agent: str) -> List[Call]:
    """Every call of the subagent ``agent`` that ran and returned a report, in order."""
    return [c for c in tr.calls if c.subagent() == agent and c.result and not c.is_error and not c.denied]


def file_sha256(path: Path) -> Optional[str]:
    """sha256 of a file, read in chunks, or None if it cannot be read."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def read_manifest(path: Optional[Path]) -> Dict[str, str]:
    """{transcript file name: sha256} from the runner's manifest (empty if absent)."""
    sealed: Dict[str, str] = {}
    if path is None or not path.is_file():
        return sealed
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and re.match(r"^[0-9a-f]{64}$", parts[0]):
            sealed[parts[1]] = parts[0]
    return sealed


# ---------------------------------------------------------------------------
# Shell command lines, read as the shell reads them (T2, T3, T4)
# ---------------------------------------------------------------------------

OPERATOR_CHARS = ";&|\n<>()"
SHELL_NAMES = ("bash", "sh", "zsh", "dash", "ksh")
HEREDOC_RE = re.compile(r"<<(-?)[ \t]*(['\"]?)([^\s'\"<>;&|()]+)\2")
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# Wrappers that run the command after them, with their options that take a value.
WRAPPERS = {
    "env": {"-u", "-C", "-S", "-P", "--unset", "--chdir", "--split-string"},
    "command": set(), "builtin": set(), "exec": {"-a"}, "nohup": set(), "time": set(),
    "nice": {"-n"}, "stdbuf": {"-i", "-o", "-e"}, "caffeinate": {"-t", "-w"},
    "xargs": {"-n", "-I", "-L", "-P", "-s", "-E", "-d", "-a", "-J", "-R", "-S"},
}
UV_VALUE_OPTS = {"--directory", "--project", "--with", "--with-editable", "--with-requirements",
                 "--python", "-p", "--group", "--extra", "--env-file", "--index", "--default-index",
                 "--package", "--only-group", "--no-group", "--index-url", "--extra-index-url",
                 "--config-file", "--cache-dir", "--color", "--exclude-newer", "--prerelease",
                 "--resolution", "--link-mode", "--refresh-package", "--reinstall-package",
                 "--upgrade-package", "--no-binary-package", "--no-build-package", "--find-links",
                 "-f", "--keyring-provider", "--index-strategy", "--python-platform"}
PY_VALUE_OPTS = {"-W", "-X", "--check-hash-based-pycs"}


class SimpleCommand:
    """One simple command of a command line: its words, redirections removed, and its stdin file."""

    __slots__ = ("argv", "stdin")

    def __init__(self, argv: List[str], stdin: Optional[str]) -> None:
        self.argv = argv
        self.stdin = stdin


def shell_text(cmd: str) -> str:
    """``cmd`` without what the shell does not run as commands: here-document bodies, comments
    and backslash-newline continuations. Quoted text is kept exactly."""
    out: List[str] = []
    i, n = 0, len(cmd)
    quote = ""
    pending: List[Tuple[str, bool]] = []
    word_start = True
    while i < n:
        ch = cmd[i]
        if quote == "'":
            out.append(ch)
            i += 1
            if ch == "'":
                quote = ""
            continue
        if quote == '"':
            if ch == "\\" and i + 1 < n:
                if cmd[i + 1] != "\n":
                    out.append(cmd[i:i + 2])
                i += 2
                continue
            out.append(ch)
            i += 1
            if ch == '"':
                quote = ""
            continue
        if ch == "\\" and i + 1 < n:
            if cmd[i + 1] != "\n":
                out.append(cmd[i:i + 2])
            i += 2
            word_start = False
            continue
        if ch in "'\"":
            quote = ch
            out.append(ch)
            i += 1
            word_start = False
            continue
        if ch == "#" and word_start:
            j = cmd.find("\n", i)
            i = n if j < 0 else j
            continue
        if cmd.startswith("<<", i) and not cmd.startswith("<<<", i):
            m = HEREDOC_RE.match(cmd, i)
            if m:
                pending.append((m.group(3), m.group(1) == "-"))
                out.append(" << " + m.group(3) + " ")
                i = m.end()
                word_start = True
                continue
        if ch == "\n" and pending:
            out.append("\n")
            i += 1
            for word, tabs in pending:
                while i < n:
                    j = cmd.find("\n", i)
                    line = cmd[i:] if j < 0 else cmd[i:j]
                    i = n if j < 0 else j + 1
                    if (line.lstrip("\t") if tabs else line) == word:
                        break
            pending = []
            word_start = True
            continue
        out.append(ch)
        i += 1
        word_start = ch in " \t\n;&|()<>"
    return "".join(out)


def shell_tokens(text: str) -> List[str]:
    """POSIX shell words of ``text``, with each run of operator characters as its own token."""
    lex = shlex.shlex(text, posix=True, punctuation_chars=OPERATOR_CHARS)
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    lex.commenters = ""
    return list(lex)


def unwrap(argv: List[str]) -> List[str]:
    """The command a simple command runs, without VAR=value prefixes and wrappers (env,
    command, exec, nohup, time, nice, stdbuf, caffeinate, xargs) and their options."""
    i = 0
    while i < len(argv):
        word = argv[i]
        if ASSIGNMENT_RE.match(word):
            i += 1
            continue
        name = os.path.basename(word)
        if name not in WRAPPERS:
            break
        takes = WRAPPERS[name]
        i += 1
        while i < len(argv) and (argv[i].startswith("-") or (name == "env" and ASSIGNMENT_RE.match(argv[i]))):
            if argv[i] == "--":
                i += 1
                break
            i += 2 if argv[i] in takes else 1
    return argv[i:]


def simple_commands(cmd: str, depth: int = 0) -> List[SimpleCommand]:
    """The simple commands a shell would run for ``cmd``: split at unquoted operators, with
    those inside unquoted command substitutions, ``bash -c`` (and other shells) and ``eval``.

    Text inside quotes is never split, so a commit message that mentions a command is not
    that command. Returns [] for a line that does not parse (the shell would refuse it too).
    """
    try:
        tokens = shell_tokens(shell_text(cmd))
    except ValueError:
        return []
    out: List[SimpleCommand] = []
    argv: List[str] = []
    stdin: Optional[str] = None
    redirect = ""

    def flush() -> None:
        nonlocal argv, stdin
        words = unwrap(argv)
        if words:
            prog = os.path.basename(words[0])
            if prog in SHELL_NAMES and depth < 3:
                for k in range(1, len(words) - 1):
                    flag = words[k]
                    if flag.startswith("-") and not flag.startswith("--") and "c" in flag[1:]:
                        out.extend(simple_commands(words[k + 1], depth + 1))
                        break
            elif prog == "eval" and depth < 3:
                out.extend(simple_commands(" ".join(words[1:]), depth + 1))
            out.append(SimpleCommand(words, stdin))
        argv, stdin = [], None

    for tok in tokens:
        if tok and all(ch in OPERATOR_CHARS for ch in tok):
            if "<" in tok or ">" in tok:
                redirect = tok
            else:
                flush()
                redirect = ""
            continue
        if redirect:
            if redirect == "<":
                stdin = tok
            redirect = ""
            continue
        argv.append(tok)
    flush()
    return out


def after_uv_run(words: List[str]) -> Tuple[Optional[List[str]], Optional[str], Optional[str]]:
    """For 'uv run [options] <command...>': (the command's words, the --directory, the module
    of 'uv run -m'). (None, None, None) for anything else."""
    if len(words) < 2 or os.path.basename(words[0]) != "uv" or words[1] != "run":
        return None, None, None
    i, directory = 2, None
    while i < len(words) and words[i].startswith("-"):
        opt, eq, inline = words[i].partition("=")
        if opt in ("-m", "--module"):
            return [], directory, inline if eq else (words[i + 1] if i + 1 < len(words) else "")
        if opt == "--":
            i += 1
            break
        if opt == "--directory":
            directory = inline if eq else (words[i + 1] if i + 1 < len(words) else None)
        i += 1 if (eq or opt not in UV_VALUE_OPTS) else 2
    return words[i:], directory, None


def gh_pr_merges(cmd: str) -> List[List[str]]:
    """The words of each 'gh pr merge' that ``cmd`` runs (directly, behind wrappers, through
    uv run, bash -c or eval). '--help' is not a merge, nor is a grep for the words."""
    found = []
    for sc in simple_commands(cmd):
        words = sc.argv
        inner, _directory, _module = after_uv_run(words)
        if inner is not None:
            words = unwrap(inner)
        if not words or os.path.basename(words[0]) != "gh":
            continue
        rest = words[1:]
        if len(rest) < 2 or rest[0] != "pr" or rest[1] != "merge" or "--help" in rest or "-h" in rest:
            continue
        found.append(words)
    return found


def python_runs(cmd: str) -> List[Tuple[Optional[str], str, str, List[str]]]:
    """Each Python run through 'uv run' in ``cmd``: (directory, kind, target, arguments).

    kind is 'script' (target a file), 'module' (target a module name, from -m), 'code'
    (target the -c code) or 'stdin' (target the file redirected to stdin, '' if none).
    """
    found: List[Tuple[Optional[str], str, str, List[str]]] = []
    for sc in simple_commands(cmd):
        inner, directory, module = after_uv_run(sc.argv)
        if inner is None:
            continue
        if module is not None:
            found.append((directory, "module", module, []))
            continue
        words = unwrap(inner)
        if not words:
            continue
        if re.match(r"^python[0-9.]*$", os.path.basename(words[0])):
            i = 1
            special = None
            while i < len(words) and words[i].startswith("-") and words[i] != "-":
                word = words[i]
                if word[:2] in ("-m", "-c"):
                    value = word[2:] if len(word) > 2 else (words[i + 1] if i + 1 < len(words) else "")
                    special = ("module" if word[:2] == "-m" else "code", value, words[i + 2:])
                    break
                i += 2 if word in PY_VALUE_OPTS else 1
            if special is not None:
                found.append((directory, special[0], special[1], special[2]))
            elif i >= len(words) or words[i] == "-":
                found.append((directory, "stdin", sc.stdin or "", words[i + 1:]))
            else:
                found.append((directory, "script", words[i], words[i + 1:]))
        elif words[0].endswith((".py", ".pyw")):
            found.append((directory, "script", words[0], words[1:]))
    return found


def run_candidates(repo: Path, directory: Optional[str], kind: str, target: str) -> List[Path]:
    """Files a Python run would execute; for a module, each place it could be found."""
    root = Path(repo)
    base = root
    if directory:
        d = Path(directory).expanduser()
        base = d if d.is_absolute() else root / d
    if kind in ("script", "stdin"):
        if not target:
            return []
        p = Path(target).expanduser()
        if p.is_absolute():
            return [p]
        # A redirect is opened by the shell, in the repo; a script by uv, in --directory.
        return [(base if kind == "script" else root) / p]
    if kind == "module":
        parts = [x for x in target.split(".") if x]
        if not parts:
            return []
        rel = Path(*parts)
        bases = [base] if base == root else [base, root]
        return [b / (str(rel) + ".py") for b in bases] + [b / rel / "__main__.py" for b in bases]
    return []


def launcher_runs(repo: Path, cmd: str) -> List[str]:
    """The launcher subcommands (launch, reconcile, smoke, ...) that ``cmd`` runs."""
    try:
        target = (Path(repo) / LAUNCHER_REL).resolve()
    except OSError:
        return []
    subs = []
    for directory, kind, script, args in python_runs(cmd):
        if kind != "script":
            continue
        for path in run_candidates(repo, directory, kind, script):
            try:
                if path.resolve() == target:
                    subs.append(next((a for a in args if not a.startswith("-")), ""))
            except OSError:
                continue
    return subs


# ---------------------------------------------------------------------------
# T1: a gate passes only on its own SUPPORTED critic review
# ---------------------------------------------------------------------------

def label_key(line: str) -> str:
    """A report line without leading Markdown decoration (as the launcher reads labels)."""
    return line.strip().lstrip(LINE_DECORATION)


def claims_pass(text: str) -> bool:
    """True if a report has a line labelled Result (or Final result) that reads PASS."""
    return any(RESULT_PASS_RE.match(label_key(line)) for line in text.splitlines())


def gate_matchers(name: str) -> Optional[Tuple[str, List[Any]]]:
    """(the gate's name, patterns a critic request about it matches) from a report file name."""
    m = GATE_FILE_RE.match(name)
    if not m:
        return None
    if m.group("n"):
        n = m.group("n")
        return f"Gate {n}", [re.compile(rf"\b(?:gate[\s_-]*{n}|G{n})\b", re.IGNORECASE)]
    if m.group("base"):
        return "the base-state gate", [re.compile(r"\bbase[\s_-]*state[\s_-]*gate\b|\bgate[\s_-]*base\b|\bGbase\b|\b7a\.base\b",
                                                  re.IGNORECASE)]
    s = re.escape(m.group("set"))
    return (f"Gate 4, set {m.group('set')}",
            [re.compile(r"\b(?:gate[\s_-]*4|G4)\b", re.IGNORECASE), re.compile(rf"\b(?:[Ss]et[\s_-]*{s}|7a\.{s})\b")])


def critic_file_named(repo_text: str) -> Optional[str]:
    """The repo path of the critic report named on a gate report's 'Critic:' line, or None."""
    for line in repo_text.splitlines():
        key = label_key(line)
        if not CRITIC_LABEL_RE.match(key):
            continue
        m = CRITIC_FILE_RE.search(key)
        if not m:
            return None
        raw = m.group(1)
        rel = raw if raw.startswith(lc.STUDY_REL + "/") else (
            f"{lc.STUDY_REL}/{raw}" if "/" in raw else GATE_REPORTS_REL + raw)
        return os.path.normpath(rel).replace(os.sep, "/")
    return None


def saved_verdict(text: str) -> str:
    """The verdict of a saved critic report: the first line of its 'Report:' section, as the
    launcher reads it (modal_launch.critic_verdict). '' if there is none."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        key = label_key(line)
        if not key.lower().startswith("report:"):
            continue
        for candidate in [key[len("report:"):]] + lines[i + 1:]:
            if candidate.strip().startswith("```"):
                continue
            value = candidate.strip().strip("`*_>").strip()
            if value:
                return first_verdict(value) if value.startswith("VERDICT:") else ""
    return ""


def report_saved_in(report: str, saved: str) -> bool:
    """True if a saved critic report holds ``report``: its first six non-empty lines, with
    whitespace collapsed, all appear in the saved text."""
    head = [" ".join(line.split()) for line in report.splitlines() if line.strip()][:6]
    flat = " ".join(saved.split())
    return bool(head) and all(line in flat for line in head)


def pass_reports(repo: Path, before: str, after: str, loop: str) -> List[Tuple[Commit, str, str]]:
    """(commit, path, text) of each gate report that a loop commit made say PASS.

    Every file under gate_reports/ counts, except the critic_* and local_* reports, whatever
    its name: the launcher accepts a report by its content.
    """
    found: List[Tuple[Commit, str, str]] = []
    for c in list_commits(repo, before, after):
        if c.committer != loop:
            continue
        for status, path in changed_paths(repo, c.sha, c.parents):
            name = path.rsplit("/", 1)[-1]
            if status == "D" or not path.startswith(GATE_REPORTS_REL) or name.startswith(("critic_", "local_")):
                continue
            text = (blob_at(repo, c.sha, path) or b"").decode("utf-8", errors="replace")
            if not claims_pass(text):
                continue
            parent = c.parents[0] if c.parents else None
            old = blob_at(repo, parent, path) if parent else None
            if old is not None and claims_pass(old.decode("utf-8", errors="replace")):
                continue  # it said PASS already; the commit that made it so was judged then
            found.append((c, path, text))
    return found


def check_t1(repo: Path, tr: Transcript, before: str, after: str, loop: str) -> List[str]:
    """T1: a gate report that says PASS needs its own SUPPORTED critic review, from this session.

    For each such report (named G<n>_, Gbase_ or G4_<S>_<iteration id>.md):
    - its 'Critic:' line names a critic report committed with it, whose Report says SUPPORTED;
    - that saved report holds the report of a study04-critic call of this session that
      finished before the commit, which returned SUPPORTED and whose request (or the saved
      report's title) names the gate;
    - every study04-critic review of this session that finished before the commit and whose
      request names the gate returned SUPPORTED: asked twice, the worse verdict counts.
    """
    problems: List[str] = []
    critics = subagent_calls(tr, CRITIC)
    for c, path, text in pass_reports(repo, before, after, loop):
        short = f"T1 {c.sha[:12]} {path}"
        name = path.rsplit("/", 1)[-1]
        gate = gate_matchers(name)
        if gate is None:
            problems.append(f"{short} says 'Result: PASS' but is not named G<n>_<iteration id>.md, "
                            "Gbase_<iteration id>.md or G4_<S>_<iteration id>.md, so no critic review can be "
                            "tied to its gate")
            continue
        label, patterns = gate

        def about(request: str) -> bool:
            return all(p.search(request) for p in patterns)

        prior = [k for k in critics if finished_by(k, c.time)]
        on_gate = [k for k in prior if about(k.request())]
        critic_rel = critic_file_named(text)
        if critic_rel is None:
            problems.append(f"{short} says 'Result: PASS', but its Critic: line names no critic report "
                            f"({GATE_REPORTS_REL}critic_<iteration id>_<k>.md)")
        else:
            blob = blob_at(repo, c.sha, critic_rel)
            if blob is None:
                problems.append(f"{short} names the critic report {critic_rel}, which is not committed with it")
            else:
                saved = blob.decode("utf-8", errors="replace")
                verdict = saved_verdict(saved)
                if verdict != "SUPPORTED":
                    problems.append(f"{short}: its critic report {critic_rel} says "
                                    f"{('VERDICT: ' + verdict) if verdict else 'no VERDICT'} after 'Report:'")
                linked = [k for k in prior if report_saved_in(k.result or "", saved)]
                if not linked:
                    problems.append(f"{short}: {critic_rel} does not hold the report of any {CRITIC} call that "
                                    "this session received before the commit")
                else:
                    k = linked[-1]
                    title = next((ln for ln in saved.splitlines() if ln.strip()), "")
                    if first_verdict(k.result or "") != "SUPPORTED":
                        problems.append(f"{short}: the {CRITIC} report saved in {critic_rel} returned "
                                        f"{first_verdict(k.result or '') or 'no verdict'}")
                    if not about(k.request()) and not about(title):
                        problems.append(f"{short}: the {CRITIC} review saved in {critic_rel} is not about "
                                        f"{label} (neither its request nor its title names it)")
                    elif k not in on_gate:
                        on_gate.append(k)
        if not on_gate:
            problems.append(f"{short} says 'Result: PASS', but no {CRITIC} review of {label} finished in this "
                            "session before the commit")
        for k in on_gate:
            verdict = first_verdict(k.result or "")
            if verdict != "SUPPORTED":
                problems.append(f"{short} says 'Result: PASS', but a {CRITIC} review of {label} in this session "
                                f"returned {verdict or 'no verdict'}; asked twice, the worse verdict counts")
    return problems


# ---------------------------------------------------------------------------
# T2: a GANDALF merge needs --match-head-commit and a clean APPROVE of that commit
# ---------------------------------------------------------------------------

_REVIEWED_RE = re.compile(r"Commit reviewed:\s*`?([0-9a-fA-F]{7,40})\b")
# Ways a reviewer says 'nothing changed' in the two fields that decide the exception (LOOP.md
# section 10); anything else, 'yes: <files>' or a list of tests, is a change.
NEGATIVE_ANSWERS = {"none", "no", "no changes", "no change", "0", "nothing", "none changed",
                    "not changed", "unchanged", "no files", "none found", "no tests changed"}


def same_sha(a: str, b: str) -> bool:
    """True if two hex shas (either may be abbreviated, 7+ characters) name the same commit."""
    a, b = a.lower(), b.lower()
    return len(a) >= 7 and len(b) >= 7 and (a.startswith(b) or b.startswith(a))


def latest_review_for(sha: str, reports: Sequence[str]) -> Optional[str]:
    """The last report in ``reports`` (oldest first) whose 'Commit reviewed:' names ``sha``."""
    chosen = None
    for text in reports:
        if any(same_sha(sha, s) for s in _REVIEWED_RE.findall(text)):
            chosen = text
    return chosen


def review_field(review: str, label: str) -> Optional[str]:
    """The answer after '<label>:' in a reviewer report: lower case, without Markdown
    decoration or a final full stop. None if no line has the label."""
    want = label.lower() + ":"
    for line in review.splitlines():
        key = label_key(line).replace("**", "")
        if key.lower().startswith(want):
            value = key[len(want):].strip().strip("`*_").strip().rstrip(".").strip()
            return " ".join(value.lower().split())
    return None


def approve_problems(review: str) -> List[str]:
    """Why a reviewer report is not a clean APPROVE (empty if it is)."""
    missing = []
    verdict = first_verdict(review)
    if verdict != "APPROVE":
        missing.append(f"its verdict is '{verdict or 'none'}', not APPROVE")
    for label in ("Existing tests changed", ".github or release files changed"):
        answer = review_field(review, label)
        if answer not in NEGATIVE_ANSWERS:
            missing.append(f"'{label}' is {answer!r}, not none" if answer is not None
                           else f"it has no '{label}:' line")
    return missing


def earlier_reviews(transcripts_dir: Optional[Path], current: Path, manifest: Optional[Path]) -> List[str]:
    """gandalf-reviewer reports from earlier sessions, oldest first.

    Only transcripts the runner sealed count: named it-YYYYMMDD-HHMM.jsonl, listed in the
    manifest, and unchanged since (same sha256). A file a session dropped into the folder, or
    a transcript edited after its session, is ignored.
    """
    if transcripts_dir is None or not transcripts_dir.is_dir():
        return []
    reports: List[str] = []
    for name, digest in sorted(read_manifest(manifest).items()):
        if name == current.name or not TRANSCRIPT_NAME_RE.match(name):
            continue
        path = transcripts_dir / name
        if file_sha256(path) != digest:
            continue
        try:
            reports.extend(c.result or "" for c in subagent_calls(parse_transcript(path), REVIEWER))
        except OSError:
            continue
    return reports


def match_head_commit(words: Sequence[str]) -> Optional[str]:
    """The sha given to --match-head-commit ('' if not a sha), or None without the option."""
    for k, word in enumerate(words):
        if word == "--match-head-commit":
            value = words[k + 1] if k + 1 < len(words) else ""
        elif word.startswith("--match-head-commit="):
            value = word.split("=", 1)[1]
        else:
            continue
        return value if re.match(r"^[0-9a-fA-F]{7,40}$", value) else ""
    return None


def check_t2(tr: Transcript, transcripts_dir: Optional[Path], current: Path, manifest: Optional[Path]) -> List[str]:
    """T2: each 'gh pr merge' a session ran needs --match-head-commit <sha>, and the latest
    gandalf-reviewer report naming that commit (earlier in this session, or in a sealed
    earlier transcript) must be a clean APPROVE. A failed merge attempt counts too."""
    problems: List[str] = []
    here = [(c.result_line, c.result or "") for c in subagent_calls(tr, REVIEWER)]
    older: Optional[List[str]] = None
    for c in tr.calls:
        if c.name != "Bash" or c.denied:
            continue
        for words in gh_pr_merges(c.command()):
            shown = short_text(" ".join(words), 120)
            sha = match_head_commit(words)
            if not sha:
                problems.append(f"T2 a GANDALF merge ran without --match-head-commit <sha>: {shown}")
                continue
            review = latest_review_for(sha, [text for line, text in here if 0 <= line < c.use_line])
            if review is None:
                if older is None:
                    older = earlier_reviews(transcripts_dir, current, manifest)
                review = latest_review_for(sha, older)
            if review is None:
                problems.append(f"T2 merge of {sha[:12]} without an earlier {REVIEWER} report naming that "
                                f"commit: {shown}")
                continue
            missing = approve_problems(review)
            if missing:
                problems.append(f"T2 merge of {sha[:12]}: the latest {REVIEWER} report on that commit is not a "
                                f"clean APPROVE ({'; '.join(missing)}): {shown}")
    return problems


# ---------------------------------------------------------------------------
# T3: no Python that imports Modal outside loop/; T4: ledger commits come from the launcher
# ---------------------------------------------------------------------------

MODAL_CLI_ACTIONS = ("run", "deploy", "serve", "shell", "launch")


def modal_cli_runs(cmd: str) -> List[str]:
    """Each Modal CLI command in ``cmd`` that starts an app: 'modal run|deploy|serve|shell'
    (directly or through uv run) and 'python -m modal ...'."""
    found = []
    for sc in simple_commands(cmd):
        inner, _directory, module = after_uv_run(sc.argv)
        if module is not None:
            if module == "modal" or module.startswith("modal."):
                found.append(" ".join(sc.argv))
            continue
        words = unwrap(inner) if inner is not None else sc.argv
        if not words:
            continue
        prog = os.path.basename(words[0])
        if prog == "modal" and len(words) > 1 and words[1] in MODAL_CLI_ACTIONS:
            found.append(" ".join(words))
        elif re.match(r"^python[0-9.]*$", prog) and len(words) > 2 and words[1] == "-m" \
                and (words[2] == "modal" or words[2].startswith("modal.")):
            found.append(" ".join(words))
    return found


def check_t3(repo: Path, tr: Transcript) -> List[str]:
    """T3: a call ran the Modal CLI (modal run, deploy, serve or shell, or python -m modal), or
    a 'uv run' call ran Python outside loop/ that imports Modal (a script, a module with -m, a
    file on stdin, or -c code). Denied calls do not count; failed ones do."""
    problems: List[str] = []
    loop_dir = str((Path(repo) / lc.LOOP_REL).resolve()) + os.sep
    seen: Set[str] = set()
    for c in tr.calls:
        if c.name != "Bash" or c.denied:
            continue
        for shown in modal_cli_runs(c.command()):
            problems.append(f"T3 the session ran the Modal CLI outside the launcher: {short_text(shown, 100)}")
        for directory, kind, target, _args in python_runs(c.command()):
            if kind == "code":
                if imports_modal("py", target) and target not in seen:
                    seen.add(target)
                    problems.append(f"T3 the session ran Python code that imports Modal with uv run: "
                                    f"{short_text(' '.join(target.split()), 80)!r}")
                continue
            for path in run_candidates(repo, directory, kind, target):
                try:
                    resolved = path.resolve()
                    if not resolved.is_file():
                        continue
                    text = resolved.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if not str(resolved).startswith(loop_dir) and str(resolved) not in seen and imports_modal("py", text):
                    seen.add(str(resolved))
                    what = {"module": f"module {target}", "stdin": f"{target} on Python's stdin"}.get(kind, target)
                    problems.append(f"T3 the session ran {what} with uv run, and it imports Modal outside loop/ "
                                    "(GPU work outside the launcher)")
                break  # Python runs the first candidate that exists
    return problems


LAUNCHER_COMMITS = {"launch": 2, "smoke": 2, "reconcile": 1}


def check_t4(repo: Path, tr: Transcript, before: str, after: str, loop: str) -> List[str]:
    """T4: each loop commit that changes the ledger is accounted for by a launcher run of this
    session that started before it: a launch or smoke makes at most two ledger commits (the
    reservation and the record), a reconcile one. A commit subject is not proof."""
    commits = [c for c in list_commits(repo, before, after) if c.committer == loop
               and any(p == lc.LEDGER_REL for _s, p in changed_paths(repo, c.sha, c.parents))]
    if not commits:
        return []
    slots: List[List[Any]] = []
    for call in tr.calls:
        if call.name != "Bash" or call.denied:
            continue
        for sub in launcher_runs(repo, call.command()):
            if LAUNCHER_COMMITS.get(sub, 0):
                slots.append([call, LAUNCHER_COMMITS[sub]])
    problems: List[str] = []
    for c in commits:
        slot = next((s for s in slots if s[1] > 0 and started_by(s[0], c.time)), None)
        if slot is None:
            problems.append(f"T4 {c.sha[:12]} changes the ledger, but no launcher run of this session (launch, "
                            f"smoke or reconcile) that started before it accounts for it: "
                            f"'{short_text(c.subject, 100)}'")
        else:
            slot[1] -= 1
    return problems


def cmd_transcript_check(args: argparse.Namespace) -> int:
    """Checks on what the session did, read from its transcript. 0 ok, 1 violations, 2 unreadable.

    T1 and T4 need --before, --after and --loop-committer. T2 reads earlier transcripts from
    --transcripts-dir and trusts only those sealed in --manifest (default: transcripts.sha256
    next to that folder). T3 reads the scripts from disk, so the runner runs it before it stashes.
    """
    repo = Path(args.repo)
    path = Path(args.transcript)
    checks = {c.strip().upper() for c in args.checks.split(",") if c.strip()}
    if not path.is_file():
        print(f"transcript missing: {path}")
        return 2
    try:
        tr = parse_transcript(path)
    except OSError as exc:
        print(f"transcript unreadable: {path}: {exc}")
        return 2
    problems: List[str] = []
    if checks & {"T1", "T4"}:
        if not (args.before and args.after and args.loop_committer):
            print("transcript-check: T1 and T4 need --before, --after and --loop-committer")
            return 2
        before, after = resolve_commit(repo, args.before), resolve_commit(repo, args.after)
        if before is None or after is None:
            print(f"transcript-check: cannot resolve {args.before} or {args.after}")
            return 2
        if before != after:
            if "T1" in checks:
                problems.extend(check_t1(repo, tr, before, after, args.loop_committer))
            if "T4" in checks:
                problems.extend(check_t4(repo, tr, before, after, args.loop_committer))
    if "T2" in checks:
        tdir = Path(args.transcripts_dir) if args.transcripts_dir else path.parent
        manifest = Path(args.manifest) if args.manifest else tdir.parent / "transcripts.sha256"
        problems.extend(check_t2(tr, tdir, path, manifest))
    if "T3" in checks:
        problems.extend(check_t3(repo, tr))
    print_lines(problems)
    return 1 if problems else 0


def cmd_transcript_seal(args: argparse.Namespace) -> int:
    """Record a finished transcript's sha256 in the manifest (replacing an older entry).

    The runner seals each transcript once its session is over; T2 trusts no other.
    """
    path = Path(args.transcript)
    manifest = Path(args.manifest)
    digest = file_sha256(path)
    if digest is None:
        print(f"transcript missing: {path}")
        return 2
    kept = []
    if manifest.is_file():
        kept = [line for line in manifest.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and line.split()[1:2] != [path.name]]
    kept.append(f"{digest}  {path.name}")
    tmp = manifest.with_name(manifest.name + ".tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(manifest))
    print(digest)
    return 0


# ---------------------------------------------------------------------------
# Modal apps
# ---------------------------------------------------------------------------

def parse_created(value: object) -> Optional[float]:
    """'Created at' from `modal app list --json` (ISO with offset) as an epoch."""
    if value in (None, ""):
        return None
    text = str(value).strip().replace("Z", "+00:00")
    stamp = dt.datetime.fromisoformat(text)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=dt.timezone.utc)
    return stamp.timestamp()


def load_json_list(text: str) -> object:
    """Parse CLI JSON output, tolerating notice lines printed before the JSON itself."""
    try:
        return json.loads(text)
    except ValueError:
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith("["):
                return json.loads("\n".join(lines[i:]))
        raise


def busy(item: Dict[str, Any]) -> bool:
    """True unless an app row says it has no tasks (containers) running."""
    tasks = item.get("Tasks", item.get("tasks"))
    try:
        return tasks is None or int(str(tasks).strip()) > 0
    except ValueError:
        return True


def cmd_apps_check(args: argparse.Namespace) -> int:
    """Modal apps the ledger does not account for. Exit 0 none, 1 some, 2 unreadable, 3 bad format.

    Three checks. (1) A running app whose name has the loop's prefix and that the ledger does
    not know (loopcommon.unknown_running_apps). (2) A running app with tasks whose launch the
    ledger has closed (every run charged), or failed: it spends hours the ledger no longer
    counts. (3) With --window START END: an app created inside a session window and not known
    to the ledger, in any state. With --any-name every such app counts
    (STOP_ON_ANY_NEW_MODAL_APP=1), otherwise only apps with the loop's prefix. An app is known
    if its ID is recorded on a launch, or if it carries the app name of a launch that has no
    app ID yet (the launcher died between the Modal call and its record). A `modal run` app
    stops when its run ends, so only (3) sees GPU work started outside the launcher.
    """
    repo = Path(args.repo)
    try:
        apps = load_json_list(Path(args.apps_json).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"cannot read the Modal app list {args.apps_json}: {exc}")
        return 2
    if not isinstance(apps, list):
        print("unexpected `modal app list --json` format: not a list")
        return 3
    for item in apps:
        if not isinstance(item, dict) or not ({"App ID", "app_id"} & set(item)) or not ({"State", "state"} & set(item)):
            print(f"unexpected `modal app list --json` row: {str(item)[:200]}")
            return 3
        if args.window:
            try:
                if parse_created(item.get("Created at", item.get("created_at"))) is None:
                    raise ValueError("no creation time")
            except ValueError as exc:
                print(f"unexpected 'Created at' in `modal app list --json` ({exc}): {str(item)[:200]}")
                return 3
    prefix = args.prefix or config_value(repo, "MODAL_APP_PREFIX", "s04-loop")
    lines: List[str] = []
    try:
        ledger = lc.load_ledger(repo / lc.LEDGER_REL)
        for row in lc.unknown_running_apps(apps, ledger, prefix):
            lines.append(f"{row['id']} {row['name']} [{row['state']}]: running, loop prefix, not in the ledger")
        launches = [x for x in ledger.get("launches", []) if isinstance(x, dict)]
        settled = {str(x.get("app_id")): x for x in launches
                   if x.get("app_id") and x.get("state") in ("closed", "launch_failed")}
        for item in apps:
            row = lc.app_rows([item])[0]
            launch = settled.get(row["id"])
            if launch is not None and lc.is_running_state(row["state"]) and busy(item):
                lines.append(f"{row['id']} {row['name']} [{row['state']}]: running, but its launch "
                             f"{launch.get('launch_id')} is {launch.get('state')} in the ledger, so the hours it "
                             "spends now are not counted")
    except Exception as exc:  # a ledger that cannot be read accounts for nothing
        print(f"ledger unreadable, so no app can be matched to it: {exc!r}")
        return 1
    if args.window:
        known_ids = {str(x.get("app_id")) for x in launches if x.get("app_id")}
        idless_names = {str(x.get("app_name")) for x in launches if x.get("app_name") and not x.get("app_id")}
        windows = []
        for start, end in args.window:
            try:
                windows.append((float(start), float(end)))
            except ValueError:
                print(f"bad --window {start} {end}")
                return 2
        flagged = {line.split(" ", 1)[0] for line in lines}
        for item in apps:
            row = lc.app_rows([item])[0]
            created = parse_created(item.get("Created at", item.get("created_at")))
            if row["id"] in known_ids or row["name"] in idless_names or row["id"] in flagged:
                continue
            if not args.any_name and not row["name"].startswith(prefix):
                continue
            if any(s - args.slack <= created <= e + args.slack for s, e in windows):
                lines.append(f"{row['id']} {row['name']} [{row['state']}]: created {utc_iso(created)}, "
                             f"during a loop session, not in the ledger")
    print_lines(lines)
    return 1 if lines else 0


# ---------------------------------------------------------------------------
# The clone
# ---------------------------------------------------------------------------

IGNORE_PROBES = (
    (f"{lc.STUDY_REL}/.loop/runner.log", ".loop/ (the runner's state: transcripts, runner.log, the lock)"),
    (f"{lc.STUDY_REL}/data/probe", "the study's data/ (run outputs)"),
)


def cmd_ignore_check(args: argparse.Namespace) -> int:
    """Exit 1 unless .loop/ and data/ are ignored by a .gitignore rule.

    The runner also writes both rules into .git/info/exclude, so that a stash never sweeps
    them up; this check makes sure the tracked rule is still there too.
    """
    repo = Path(args.repo)
    problems: List[str] = []
    for probe, what in IGNORE_PROBES:
        proc = git(repo, "check-ignore", "-v", "--no-index", "--", probe)
        out = proc.stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0 or not out:
            problems.append(f"{what} is not ignored by git ({probe})")
            continue
        source = out.split(":", 1)[0]
        if not source.endswith(".gitignore"):
            problems.append(f"{what} is ignored only by {source}, not by a rule in .gitignore")
    print_lines(problems)
    return 1 if problems else 0


def untracked_instruction_files(repo: Path) -> List[str]:
    """Files a session would load that git does not track: any CLAUDE.md or CLAUDE.local.md,
    and any file under a .claude/ folder, at any depth, ignored files included.

    The walk skips .git and does not follow symbolic links (the Study 2 data link).
    """
    tracked = set(split_z(git(repo, "ls-files", "-z").stdout))
    found: List[str] = []
    root = str(repo)
    for current, dirs, files in os.walk(root):
        rel_dir = os.path.relpath(current, root)
        rel_dir = "" if rel_dir == "." else rel_dir.replace(os.sep, "/")
        if not rel_dir:
            dirs[:] = [d for d in dirs if d != ".git"]
        dirs.sort()
        in_claude = ".claude" in rel_dir.split("/")
        for name in sorted(files):
            rel = f"{rel_dir}/{name}" if rel_dir else name
            if rel in tracked:
                continue
            if name in INSTRUCTION_NAMES or in_claude:
                found.append(rel)
    return found


def cmd_clone_check(args: argparse.Namespace) -> int:
    """Exit 1 if the clone holds files that change what a session loads or what git runs.

    .claude/settings.local.json; any untracked or ignored file under a .claude/ folder, and
    any untracked or ignored CLAUDE.md or CLAUDE.local.md, at any depth (data/ and
    data/scratch/ included: a session loads instructions from folders it reads); a git hook
    other than the *.sample files; and in the clone's own config core.hooksPath,
    core.fsmonitor, a push URL, a URL rewrite (insteadOf), an include, core.sshCommand, a
    filter, an external diff or textconv, or a merge driver.
    """
    repo = Path(args.repo)
    problems: List[str] = []
    if (repo / ".claude" / "settings.local.json").exists():
        problems.append(".claude/settings.local.json exists (local settings for every later session)")
    for path in untracked_instruction_files(repo):
        if path == ".claude/settings.local.json":
            continue
        if path.rsplit("/", 1)[-1] in INSTRUCTION_NAMES:
            problems.append(f"{path} exists and is not tracked (instructions for later sessions)")
        else:
            problems.append(f"untracked file under .claude/: {path}")
    proc = git(repo, "rev-parse", "--git-path", "hooks", safe=False)
    hooks_rel = proc.stdout.decode("utf-8", errors="replace").strip()
    if proc.returncode == 0 and hooks_rel:
        hooks = Path(hooks_rel) if os.path.isabs(hooks_rel) else repo / hooks_rel
        if hooks.is_dir():
            for entry in sorted(hooks.iterdir()):
                if not entry.name.endswith(".sample"):
                    problems.append(f"git hook {hooks_rel}/{entry.name} exists; git would run it")
    for key in ("core.hooksPath", "core.fsmonitor"):
        proc = git(repo, "config", "--local", "--get", key, safe=False)
        value = proc.stdout.decode("utf-8", errors="replace").strip()
        if proc.returncode == 0 and value and value.lower() not in ("false", "0", "no", "off"):
            problems.append(f"{key} is set in the clone's .git/config ({value})")
    # Where pushes go, what else git reads, and programs git runs: a clone made with git clone
    # has none of these in its own config.
    proc = git(repo, "config", "--local", "--get-regexp",
               r"^(remote\..*\.pushurl|url\..*\.(insteadof|pushinsteadof)|include\.path|includeif\..*"
               r"|core\.sshcommand|filter\..*|diff\.external|diff\..*\.(textconv|command)|merge\..*\.driver)$",
               safe=False)
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        if line.strip():
            problems.append(f"the clone's .git/config sets {short_text(line.strip(), 120)} (where git pushes, "
                            "what config it reads, or a program it runs)")
    print_lines(problems)
    return 1 if problems else 0


def cmd_questions_clean(args: argparse.Namespace) -> int:
    """Exit 0 if the working-tree QUESTIONS.md only gains lines since HEAD.

    Removing a 'None.' placeholder or blank lines is allowed, because stop-write does that.
    Exit 1 otherwise (including a conflicted or deleted file).
    """
    repo = Path(args.repo)
    path = repo / lc.QUESTIONS_REL
    if not path.is_file():
        return 1
    if git(repo, "ls-files", "-u", "--", lc.QUESTIONS_REL).stdout.strip():
        return 1
    base = (blob_at(repo, "HEAD", lc.QUESTIONS_REL) or b"").decode("utf-8", errors="replace").splitlines()
    new = path.read_bytes().decode("utf-8", errors="replace").splitlines()
    matcher = difflib.SequenceMatcher(a=base, b=new, autojunk=False)
    for tag, i1, i2, _j1, _j2 in matcher.get_opcodes():
        if tag in ("equal", "insert"):
            continue
        if all(line.strip().strip("_*") in ("None.", "None", "") for line in base[i1:i2]):
            continue
        print(f"QUESTIONS.md: line {i1 + 1} changed or removed")
        return 1
    return 0


# ---------------------------------------------------------------------------
# WAIT, STOP, prompt, session result, settings
# ---------------------------------------------------------------------------

def read_kv(path: Path) -> Dict[str, str]:
    """key=value lines of a small state file."""
    values: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, sep, value = line.partition("=")
        if sep:
            values[key.strip()] = value.strip()
    return values


def cmd_wait_remaining(args: argparse.Namespace) -> int:
    """Print the seconds left on WAIT (0 if absent, expired or garbled) and, on line 2, why."""
    path = Path(args.repo) / lc.WAIT_REL
    if not path.is_file():
        print(0)
        print("")
        return 0
    values = read_kv(path)
    reason = values.get("reason", "")
    try:
        until = int(float(values.get("until_epoch", "")))
    except ValueError:
        print(0)
        print("WAIT has no readable until_epoch; ignored")
        return 0
    remaining = until - int(time.time())
    if remaining <= 0:
        print(0)
        print(f"WAIT expired at {values.get('until_utc', utc_iso(until))}: {reason}")
    elif remaining > MAX_WAIT_SEC + 300:
        print(0)
        print(f"WAIT ends more than 48 h ahead ({values.get('until_utc', '?')}); ignored, since "
              f"wait.sh allows at most 2880 minutes: {reason}")
    else:
        print(remaining)
        print(reason)
    return 0


def _blocking_insert(text: str, item: str) -> Optional[str]:
    """Add ``item`` at the end of the '## Blocking' section; drop a lone 'None.' placeholder."""
    lines = text.splitlines()
    try:
        head = next(i for i, line in enumerate(lines) if line.strip() == "## Blocking")
    except StopIteration:
        return None
    end = next((j for j in range(head + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
    section = [line for line in lines[head + 1:end]
               if line.strip().strip("_*") not in ("None.", "None")]
    while section and not section[-1].strip():
        section.pop()
    if section and not section[-1].lstrip().startswith("- **STOP"):
        section.append("")
    if not section:
        section.append("")
    section.extend([item, ""])
    out = lines[:head + 1] + section + lines[end:]
    return "\n".join(out) + "\n"


def cmd_stop_write(args: argparse.Namespace) -> int:
    """Write (or extend) STOP and add a Blocking line to QUESTIONS.md."""
    repo = Path(args.repo)
    stop = repo / lc.STOP_REL
    now = utc_iso()
    iteration = os.environ.get("S04_ITER_ID") or "-"
    reason_lines = [line.rstrip() for line in args.reason.strip().splitlines() if line.strip()] or ["(no reason given)"]
    reason = reason_lines[0] + "".join("\n  " + line for line in reason_lines[1:])
    decision = " ".join((args.decision or DEFAULT_DECISION).split())
    if stop.is_file():
        text = stop.read_text(encoding="utf-8", errors="replace")
        if not text.endswith("\n"):
            text += "\n"
        text += (f"\nAlso written by: {args.source} ({iteration}), {now}\n"
                 f"Reason: {reason}\nDecision needed from Anjor: {decision}\n")
    else:
        text = (f"STOP\nWritten by: {args.source} ({iteration}), {now}\nReason: {reason}\n"
                f"Decision needed from Anjor: {decision}\n"
                "To resume: answer in QUESTIONS.md, delete this file, commit and push\n"
                f"(or run {lc.LOOP_REL}/resume.sh in your checkout).\n")
    stop.parent.mkdir(parents=True, exist_ok=True)
    stop.write_text(text, encoding="utf-8")
    print(f"wrote {lc.STOP_REL}")
    if args.no_questions:
        return 0
    questions = repo / lc.QUESTIONS_REL
    one_line = " ".join(" ".join(reason_lines).split())
    if len(one_line) > 400:
        one_line = one_line[:397] + "..."
    item = f"- **STOP {now[:10]} ({args.source})**: {one_line} See `STOP`."
    if questions.is_file():
        updated = _blocking_insert(questions.read_text(encoding="utf-8"), item)
        if updated is None:
            print("note: QUESTIONS.md has no '## Blocking' line; only STOP was written", file=sys.stderr)
        else:
            questions.write_text(updated, encoding="utf-8")
            print(f"added a Blocking line to {lc.QUESTIONS_REL}")
    return 0


def cmd_render_prompt(args: argparse.Namespace) -> int:
    """Fill the placeholders of loop/prompt.md."""
    text = Path(args.template).read_text(encoding="utf-8")
    for key, value in (("ITER_ID", args.iter), ("STARTED_UTC", args.started),
                       ("GANDALF_WORKTREE", args.worktree), ("STASH_NOTE", args.stash_note)):
        text = text.replace("{{" + key + "}}", value)
    sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return 0


AUTH_RE = re.compile(r"authentication_failed|invalid api key|invalid x-api-key|/login\b|not logged in|"
                     r"oauth token|\b401\b", re.IGNORECASE)
LIMIT_RE = re.compile(r"rate[_ ]limit(?:ed|_error)?\b|usage limit|\b429\b", re.IGNORECASE)


def classify_failure(result_text: str, statuses: Set[int], rl_status: str, stderr: str) -> str:
    """Why a session failed, from what the CLI reported rather than from every line it printed.

    ``result_text`` is the final result event's text, ``statuses`` the HTTP statuses of its
    api_error_status and of the API retries, ``rl_status`` the last rate_limit_event's status
    and ``stderr`` the tail of the CLI's stderr. A rate_limit_event that says 'allowed' is not
    a rate limit: every session prints one.
    """
    text = f"{result_text}\n{stderr}"
    limited = bool(rl_status) and rl_status != "none" and not rl_status.startswith("allowed")
    if limited or 429 in statuses or LIMIT_RE.search(text):
        return "rate_limit"
    if statuses & {401, 403} or AUTH_RE.search(text):
        return "auth"
    if 529 in statuses or re.search(r"overloaded", text, re.IGNORECASE):
        return "overloaded"
    if "budget" in text.lower():
        return "budget"
    return "unknown"


def _number(value: Any) -> str:
    """A JSON number as text for runner.log ('' if absent or not a number)."""
    if isinstance(value, bool) or value is None:
        return ""
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return ""


def _status(value: Any) -> Optional[int]:
    """An HTTP status from a JSON field, or None."""
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def cmd_session_check(args: argparse.Namespace) -> int:
    """key=value lines about the session, read from its transcript and stderr.

    result=ok|error|none (the last stream-json result); class=none for a session that ended
    well, else the failure class (rate_limit, auth, overloaded, budget, unknown), read from
    the result event, the API retries, the last rate_limit_event and stderr; and from the last
    rate_limit_event: rl_status (allowed when fine), rl_resets_at (epoch seconds), rl_type,
    util_5h and util_7d (the five-hour and seven-day utilizations).
    """
    result = "none"
    result_text = ""
    statuses: Set[int] = set()
    rate: Optional[Dict[str, Any]] = None
    transcript = Path(args.transcript)
    if transcript.is_file():
        with open(transcript, "rb") as fh:
            for raw in fh:
                if not (b'"result"' in raw or b"rate_limit_event" in raw or b"api_retry" in raw):
                    continue
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except ValueError:
                    continue
                if not isinstance(obj, dict):
                    continue
                if obj.get("type") == "result":
                    result = "error" if obj.get("is_error") else "ok"
                    result_text = str(obj.get("result") or "")
                    code = _status(obj.get("api_error_status"))
                    if code:
                        statuses.add(code)
                elif obj.get("type") == "rate_limit_event" and isinstance(obj.get("rate_limit_info"), dict):
                    rate = obj["rate_limit_info"]
                elif obj.get("type") == "system" and obj.get("subtype") == "api_retry":
                    code = _status(obj.get("error_status"))
                    if code:
                        statuses.add(code)
    stderr = ""
    if args.stderr and Path(args.stderr).is_file():
        stderr = "\n".join(Path(args.stderr).read_text(encoding="utf-8", errors="replace").splitlines()[-300:])
    rate = rate or {}
    rl_status = str(rate.get("status") or "none").strip() or "none"
    print(f"result={result}")
    print(f"class={'none' if result == 'ok' else classify_failure(result_text, statuses, rl_status, stderr)}")
    windows = rate.get("unifiedWindows") if isinstance(rate.get("unifiedWindows"), dict) else {}
    five = windows.get("five_hour") if isinstance(windows.get("five_hour"), dict) else {}
    seven = windows.get("seven_day") if isinstance(windows.get("seven_day"), dict) else {}
    resets = rate.get("resetsAt")
    print(f"rl_status={rl_status}")
    print(f"rl_resets_at={int(float(resets)) if _number(resets) else ''}")
    print(f"rl_type={rate.get('rateLimitType') or ''}")
    print(f"util_5h={_number(five.get('utilization'))}")
    print(f"util_7d={_number(seven.get('utilization'))}")
    return 0


def cmd_check_settings(args: argparse.Namespace) -> int:
    """The settings file must parse (claude -p ignores an invalid one silently) and have rules."""
    try:
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"settings unreadable: {exc}")
        return 1
    perms = data.get("permissions") if isinstance(data, dict) else None
    if not isinstance(perms, dict):
        print("settings: no permissions object")
        return 1
    if not isinstance(perms.get("allow"), list) or not isinstance(perms.get("deny"), list) or not perms["deny"]:
        print("settings: permissions.allow and a non-empty permissions.deny list are required")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """The argument parser with every subcommand."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo", dest="repo_sub", default=None, help="repo root (default: this file's repo)")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", dest="repo_main", default=None, help="repo root (default: this file's repo)")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("entry-closed", parents=[common])
    p.add_argument("--iter", required=True)
    p.set_defaults(func=cmd_entry_closed)

    p = sub.add_parser("outcome", parents=[common])
    p.add_argument("--iter", required=True)
    p.set_defaults(func=cmd_outcome)

    p = sub.add_parser("append-stub", parents=[common])
    p.add_argument("--iter", required=True)
    p.add_argument("--exit-code", default="?")
    p.add_argument("--duration-sec", default="-1")
    p.add_argument("--before", default="")
    p.add_argument("--after", default="")
    p.add_argument("--stash", default="none")
    p.add_argument("--note", default="")
    p.add_argument("--title", default="")
    p.set_defaults(func=cmd_append_stub)

    p = sub.add_parser("streak", parents=[common])
    p.add_argument("--loop-committer", "--loop-author", dest="loop_committer", default="")
    p.set_defaults(func=cmd_streak)

    p = sub.add_parser("iter-count", parents=[common])
    p.add_argument("--day", required=True)
    p.set_defaults(func=cmd_iter_count)

    p = sub.add_parser("log-check", parents=[common])
    p.add_argument("--base", required=True)
    p.set_defaults(func=cmd_log_check)

    p = sub.add_parser("frozen-record", parents=[common])
    p.set_defaults(func=cmd_frozen_record)

    p = sub.add_parser("frozen-check", parents=[common])
    p.add_argument("--manifest", default="")
    p.set_defaults(func=cmd_frozen_check)

    p = sub.add_parser("ledger-verify", parents=[common])
    p.set_defaults(func=cmd_ledger_verify)

    p = sub.add_parser("commit-guards", parents=[common])
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.add_argument("--loop-committer", "--loop-author", dest="loop_committer", required=True)
    p.add_argument("--session-start", default="",
                   help="epoch the session started: check where commits by other committers came from")
    p.set_defaults(func=cmd_commit_guards)

    p = sub.add_parser("transcript-check", parents=[common])
    p.add_argument("--transcript", required=True)
    p.add_argument("--checks", default="T1,T2,T3,T4")
    p.add_argument("--before", default="")
    p.add_argument("--after", default="")
    p.add_argument("--loop-committer", default="")
    p.add_argument("--transcripts-dir", default="")
    p.add_argument("--manifest", default="", help="sealed transcripts (default: transcripts.sha256 next to the folder)")
    p.set_defaults(func=cmd_transcript_check)

    p = sub.add_parser("transcript-seal", parents=[common])
    p.add_argument("--transcript", required=True)
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=cmd_transcript_seal)

    p = sub.add_parser("apps-check", parents=[common])
    p.add_argument("--apps-json", required=True)
    p.add_argument("--prefix", default="")
    p.add_argument("--window", nargs=2, action="append", metavar=("START", "END"), default=[])
    p.add_argument("--slack", type=float, default=120.0)
    p.add_argument("--any-name", action="store_true",
                   help="with --window, flag every unknown app created in a window, whatever its name")
    p.set_defaults(func=cmd_apps_check)

    p = sub.add_parser("ignore-check", parents=[common])
    p.set_defaults(func=cmd_ignore_check)

    p = sub.add_parser("clone-check", parents=[common])
    p.set_defaults(func=cmd_clone_check)

    p = sub.add_parser("questions-clean", parents=[common])
    p.set_defaults(func=cmd_questions_clean)

    p = sub.add_parser("wait-remaining", parents=[common])
    p.set_defaults(func=cmd_wait_remaining)

    p = sub.add_parser("stop-write", parents=[common])
    p.add_argument("--reason", required=True)
    p.add_argument("--decision", default="")
    p.add_argument("--source", required=True, choices=("runner", "agent", "anjor"))
    p.add_argument("--no-questions", action="store_true", help="write STOP only")
    p.set_defaults(func=cmd_stop_write)

    p = sub.add_parser("render-prompt", parents=[common])
    p.add_argument("--template", required=True)
    p.add_argument("--iter", required=True)
    p.add_argument("--started", required=True)
    p.add_argument("--worktree", required=True)
    p.add_argument("--stash-note", required=True)
    p.set_defaults(func=cmd_render_prompt)

    p = sub.add_parser("session-check", parents=[common])
    p.add_argument("--transcript", required=True)
    p.add_argument("--stderr", default="")
    p.set_defaults(func=cmd_session_check)

    p = sub.add_parser("check-settings", parents=[common])
    p.add_argument("--file", required=True)
    p.set_defaults(func=cmd_check_settings)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse the command line and run one subcommand."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    args.repo = Path(args.repo_sub or args.repo_main or DEFAULT_REPO).resolve()
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
