"""Shared helpers for the Study 04 loop: config, ledger, frozen sections, Modal app check.

Standard library only. Two programs import this module:

- ``loop/guards.py``: the runner's checks before and after each session.
- ``loop/modal_launch.py``: the only path to a GPU.

Keeping the ledger format and the section hashing in one place means the runner and the
launcher cannot disagree about what the ledger says or what "frozen" means.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STUDY_REL = "studies/04-phase-space-helicity"
LOOP_REL = f"{STUDY_REL}/loop"
LEDGER_REL = f"{STUDY_REL}/compute_ledger.json"
CONFIG_REL = f"{LOOP_REL}/config.env"
FROZEN_REL = f"{LOOP_REL}/frozen.json"
LOG_REL = f"{STUDY_REL}/log"
STOP_REL = f"{STUDY_REL}/STOP"
WAIT_REL = f"{STUDY_REL}/WAIT"
QUESTIONS_REL = f"{STUDY_REL}/QUESTIONS.md"

LEDGER_SCHEMA = 1
# Commit subjects the launcher uses for its own ledger commits. The runner treats any other
# commit that touches the ledger as a hand edit and writes STOP.
LAUNCHER_SUBJECT_PREFIX = "Study 04 [launcher]"

RUN_STATUSES = ("reserved", "running", "finished", "failed", "unknown", "not_launched")
LAUNCH_STATES = ("reserved", "launched", "launch_failed", "closed")


# ---------------------------------------------------------------------------
# config.env
# ---------------------------------------------------------------------------

def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a KEY=VALUE file (the format of loop/config.env).

    Blank lines and lines starting with '#' are skipped. A trailing ' # comment' is removed
    from unquoted values. Matching single or double quotes around a value are removed. A
    leading '~' is expanded to the home directory, as bash does in an assignment.
    """
    values: Dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = re.sub(r"\s+#.*$", "", value).strip()
        if value.startswith("~"):
            value = os.path.expanduser(value)
        values[key] = value
    return values


def compute_cap(repo: Path) -> float:
    """The A100-hour cap for the whole study, read from loop/config.env (Anjor's setting)."""
    cfg = parse_env_file(Path(repo) / CONFIG_REL)
    if "COMPUTE_CAP_A100_HOURS" not in cfg:
        raise ValueError(f"COMPUTE_CAP_A100_HOURS missing from {CONFIG_REL}")
    return float(cfg["COMPUTE_CAP_A100_HOURS"])


# ---------------------------------------------------------------------------
# Markdown sections and frozen hashes
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def heading_matches(heading_text: str, heading_prefix: str) -> bool:
    """True if a heading's text is ``heading_prefix`` or continues it with a non-alphanumeric.

    So the key '7a.B' matches '7a.B Set B' but not '7a.base Base state', and '7. Tolerances'
    matches only that heading.
    """
    if not heading_text.startswith(heading_prefix):
        return False
    rest = heading_text[len(heading_prefix):]
    return not rest or not rest[0].isalnum()


def extract_section(text: str, heading_prefix: str) -> Optional[str]:
    """Return the Markdown section whose heading text matches ``heading_prefix``.

    A heading matches if its text equals the prefix or continues it with a character that
    is not a letter or digit (see ``heading_matches``). The section runs from its heading
    line to the line before the next heading of the same or a higher level, so subsections
    are included. Trailing whitespace is stripped from each line and blank lines at either
    end are dropped, so whitespace-only edits at the edges do not change the hash. Lines
    inside fenced code blocks are never treated as headings. Returns None if no heading
    matches, and also if more than one does: an ambiguous key must not be frozen, and a
    frozen key that becomes ambiguous (a copied heading) must not be checked against a guess.
    """
    lines = text.splitlines()
    start = None
    level = 0
    matches = 0
    in_fence = False
    for i, line in enumerate(lines):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m and heading_matches(m.group(2), heading_prefix):
            matches += 1
            if start is None:
                start, level = i, len(m.group(1))
    if start is None or matches > 1:
        return None
    end = len(lines)
    in_fence = False
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) <= level:
            end = j
            break
    body = [ln.rstrip() for ln in lines[start:end]]
    while body and not body[-1]:
        body.pop()
    while body and not body[0]:
        body.pop(0)
    return "\n".join(body)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_key_hash(repo: Path, key: str) -> Optional[str]:
    """Hash of a frozen item. ``key`` is a repo-relative path, optionally with '#heading'.

    'path' hashes the whole file. 'path#7. Tolerances' hashes the Markdown section whose
    heading starts with '7. Tolerances'. Returns None if the file or the heading is missing.
    """
    rel, sep, heading = key.partition("#")
    path = Path(repo) / rel
    if not path.is_file():
        return None
    if not sep:
        return sha256_file(path)
    section = extract_section(path.read_text(encoding="utf-8"), heading)
    if section is None:
        return None
    return sha256_text(section)


def load_frozen_manifest(repo: Path) -> Dict[str, str]:
    path = Path(repo) / FROZEN_REL
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return dict(data.get("entries", {}))


def check_frozen(repo: Path, entries: Dict[str, str]) -> List[str]:
    """Compare current hashes with the recorded ones. Returns one message per mismatch."""
    problems = []
    for key, recorded in sorted(entries.items()):
        current = frozen_key_hash(repo, key)
        if current is None:
            problems.append(f"frozen item missing: {key}")
        elif current != recorded:
            problems.append(f"frozen item changed: {key} (recorded {recorded[:12]}, now {current[:12]})")
    return problems


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def empty_ledger() -> Dict[str, Any]:
    return {
        "schema": LEDGER_SCHEMA,
        "study": "04-phase-space-helicity",
        "unit": "A100-hours",
        "note": (
            "Written only by loop/modal_launch.py. The cap is in loop/config.env. "
            "A hand edit breaks the integrity hash and the runner writes STOP."
        ),
        "launches": [],
    }


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_integrity(ledger: Dict[str, Any]) -> str:
    body = {k: v for k, v in ledger.items() if k != "integrity"}
    return hashlib.sha256(_canonical(body)).hexdigest()


def load_ledger(path: Path) -> Dict[str, Any]:
    """Load the ledger. A missing file is an empty ledger (the study has spent nothing)."""
    path = Path(path)
    if not path.is_file():
        return empty_ledger()
    return json.loads(path.read_text(encoding="utf-8"))


def ledger_text(ledger: Dict[str, Any]) -> str:
    """Serialised ledger with a fresh integrity hash, in the one stable on-disk format."""
    out = {k: v for k, v in ledger.items() if k != "integrity"}
    out["integrity"] = compute_integrity(out)
    return json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def save_ledger(path: Path, ledger: Dict[str, Any]) -> None:
    Path(path).write_text(ledger_text(ledger), encoding="utf-8")


def verify_ledger(ledger: Any, require_integrity: bool = True) -> List[str]:
    """Structural and integrity checks. Returns a list of problems (empty if sound)."""
    problems: List[str] = []
    if not isinstance(ledger, dict):
        return ["ledger is not a JSON object"]
    launches = ledger.get("launches", [])
    if not isinstance(launches, list) or not all(isinstance(x, dict) for x in launches):
        return ["ledger 'launches' is not a list of objects"]
    for launch in launches:
        runs = launch.get("runs", [])
        if not isinstance(runs, list) or not all(isinstance(r, dict) for r in runs):
            return [f"{launch.get('launch_id')}: 'runs' is not a list of objects"]
    if ledger.get("schema") != LEDGER_SCHEMA:
        problems.append(f"ledger schema is {ledger.get('schema')!r}, expected {LEDGER_SCHEMA}")
    if require_integrity:
        recorded = ledger.get("integrity")
        if recorded is None:
            if ledger.get("launches"):
                problems.append("ledger has launches but no integrity hash")
        elif recorded != compute_integrity(ledger):
            problems.append("ledger integrity hash does not match its content (hand edit?)")
    seen_launch, seen_run = set(), set()
    for launch in ledger.get("launches", []):
        lid = launch.get("launch_id")
        if lid in seen_launch:
            problems.append(f"duplicate launch_id {lid}")
        seen_launch.add(lid)
        if launch.get("state") not in LAUNCH_STATES:
            problems.append(f"{lid}: bad state {launch.get('state')!r}")
        for run in launch.get("runs", []):
            rid = run.get("run_id")
            if rid in seen_run:
                problems.append(f"duplicate run_id {rid}")
            seen_run.add(rid)
            if run.get("status") not in RUN_STATUSES:
                problems.append(f"{rid}: bad status {run.get('status')!r}")
            try:
                if float(run.get("reserved_hours", 0)) < 0:
                    problems.append(f"{rid}: negative reservation")
                if run.get("charged_hours") is not None and float(run["charged_hours"]) < 0:
                    problems.append(f"{rid}: negative charge")
            except (TypeError, ValueError):
                problems.append(f"{rid}: non-numeric hours")
    return problems


_IMMUTABLE_LAUNCH_KEYS = ("kind", "run_set", "iteration", "created_utc", "repo_commit", "gandalf_commit",
                          "jax_version", "app_name", "timeout_hours", "gate_report", "local_test",
                          "frozen", "volume")
_IMMUTABLE_RUN_KEYS = ("config", "config_sha256", "volume_path")
_STATE_NEXT = {
    "reserved": {"reserved", "launched", "launch_failed", "closed"},
    "launched": {"launched", "closed"},
    "launch_failed": {"launch_failed"},
    "closed": {"closed"},
}


def ledger_extends(prev: Dict[str, Any], cur: Dict[str, Any]) -> List[str]:
    """Problems if ``cur`` is not a launcher-made successor of ``prev``.

    The launcher only appends launches, fills in fields that were empty, moves states
    forward, charges a run once and never lowers a reservation. A version that drops a
    launch, changes a charge or lowers a reservation was rolled back or edited; a valid
    integrity hash does not make it right (an old version has a valid hash too). Used by the
    launcher before it spends money and by the runner's commit guard on every loop commit
    that touches the ledger.
    """
    problems: List[str] = []
    old, new = prev.get("launches", []) or [], cur.get("launches", []) or []
    if len(new) < len(old):
        problems.append(f"{len(old) - len(new)} launch(es) removed")
    for p, c in zip(old, new):
        lid = p.get("launch_id")
        if c.get("launch_id") != lid:
            problems.append(f"launch {lid} replaced by {c.get('launch_id')}")
            continue
        for key in _IMMUTABLE_LAUNCH_KEYS:
            if key in p and c.get(key) != p.get(key):
                problems.append(f"{lid}: {key} changed")
        if p.get("app_id") and c.get("app_id") != p.get("app_id"):
            problems.append(f"{lid}: app_id changed")
        if c.get("state") not in _STATE_NEXT.get(str(p.get("state")), {p.get("state")}):
            problems.append(f"{lid}: state went from {p.get('state')} to {c.get('state')}")
        p_runs, c_runs = p.get("runs", []) or [], c.get("runs", []) or []
        if [r.get("run_id") for r in p_runs] != [r.get("run_id") for r in c_runs]:
            problems.append(f"{lid}: its runs changed")
            continue
        for a, b in zip(p_runs, c_runs):
            rid = a.get("run_id")
            for key in _IMMUTABLE_RUN_KEYS:
                if key in a and b.get(key) != a.get(key):
                    problems.append(f"{rid}: {key} changed")
            if a.get("call_id") and b.get("call_id") != a.get("call_id"):
                problems.append(f"{rid}: call_id changed")
            if a.get("charged_hours") is not None and (
                    b.get("charged_hours") != a.get("charged_hours") or b.get("status") != a.get("status")):
                problems.append(f"{rid}: changed after it was charged")
            try:
                if float(b.get("reserved_hours") or 0.0) < float(a.get("reserved_hours") or 0.0) - 1e-9:
                    problems.append(f"{rid}: reservation lowered")
            except (TypeError, ValueError):
                problems.append(f"{rid}: non-numeric reservation")
    return problems


def ledger_totals(ledger: Dict[str, Any]) -> Dict[str, float]:
    """A100-hours used (charged), reserved (in flight, not yet charged), and counts."""
    used = 0.0
    reserved = 0.0
    in_flight = 0
    for launch in ledger.get("launches", []):
        for run in launch.get("runs", []):
            if run.get("charged_hours") is not None:
                used += float(run["charged_hours"])
            elif run.get("status") != "not_launched":
                reserved += float(run.get("reserved_hours", 0.0))
                in_flight += 1
    return {
        "used": used,
        "reserved": reserved,
        "launches": float(len(ledger.get("launches", []))),
        "in_flight": float(in_flight),
    }


def status_line(cap: float, ledger: Dict[str, Any]) -> str:
    """The one line that goes into the log's Compute field."""
    t = ledger_totals(ledger)
    left = cap - t["used"] - t["reserved"]
    return (
        f"Compute cap {cap:.1f} A100-h: used {t['used']:.2f}, reserved {t['reserved']:.2f}, "
        f"left {left:.2f} (launches {int(t['launches'])}, runs in flight {int(t['in_flight'])})"
    )


def ledger_frozen_entries(ledger: Dict[str, Any]) -> Dict[str, str]:
    """Items frozen at launch time (for example SPEC.md §7a for a run set)."""
    entries: Dict[str, str] = {}
    for launch in ledger.get("launches", []):
        if launch.get("state") == "launch_failed":
            continue
        for key, sha in (launch.get("frozen") or {}).items():
            entries[key] = sha
    return entries


# ---------------------------------------------------------------------------
# Modal apps versus the ledger
# ---------------------------------------------------------------------------

_STOPPED_STATES = ("stopped", "stopping", "disabled")


def app_rows(apps: Any) -> List[Dict[str, str]]:
    """Normalise the output of ``modal app list --json`` to dicts with id, name, state."""
    rows = []
    for item in apps or []:
        if not isinstance(item, dict):
            continue
        rows.append({
            "id": str(item.get("App ID") or item.get("app_id") or ""),
            "name": str(item.get("Description") or item.get("description") or item.get("name") or ""),
            "state": str(item.get("State") or item.get("state") or ""),
        })
    return rows


def is_running_state(state: str) -> bool:
    s = state.strip().lower()
    return bool(s) and not s.startswith(_STOPPED_STATES)


def unknown_running_apps(apps: Any, ledger: Dict[str, Any], prefix: str) -> List[Dict[str, str]]:
    """Running apps with the loop's name prefix that the ledger does not account for.

    An app is accounted for if its ID is recorded on a launch, or if its name is the
    app_name of a launch that is still 'reserved' (the launcher was interrupted between the
    Modal call and the ledger update; the hours are already reserved).
    """
    known_ids = set()
    reserved_names = set()
    for launch in ledger.get("launches", []):
        if launch.get("app_id"):
            known_ids.add(launch["app_id"])
        if launch.get("state") == "reserved" and launch.get("app_name"):
            reserved_names.add(launch["app_name"])
    unknown = []
    for row in app_rows(apps):
        if not row["name"].startswith(prefix) or not is_running_state(row["state"]):
            continue
        if row["id"] in known_ids or row["name"] in reserved_names:
            continue
        unknown.append(row)
    return unknown


def iter_runs(ledger: Dict[str, Any]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
    return [(launch, run) for launch in ledger.get("launches", []) for run in launch.get("runs", [])]
