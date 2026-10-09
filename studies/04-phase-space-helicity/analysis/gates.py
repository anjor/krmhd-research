"""Study 04 gates: coded checks and the shared gate-report writer (LOOP.md §5).

A quality gate passes when its coded check here passes on raw outputs, the
`study04-critic` subagent returns `VERDICT: SUPPORTED` on that evidence, and no
kill criterion is met. Every gate writes its report in two steps, both in this
file, so that no number and no verdict is typed by hand:

1. `write_report` writes everything down to the `Coded check` line.
2. `finish_report` appends the `Critic`, `Kill criteria` and `Result` lines,
   copying the critic's VERDICT line from its saved report
   (`gate_reports/critic_<iteration id>_<k>.md`). That review must be from the same
   iteration, name the gate and the report's file name in its title, and quote in its
   request the repo commit and every data-file sha256 the report states.

The writer refuses a field that would break a line (`head_field_problems`) and a head the
launcher would misread (`head_form_problems`). `finish_report` accepts only a head that the
writer itself produced, by parsing it and rendering it again.

Gate 1 (PLAN.md §4, Phase 0): the invariant mapping and SPEC.md are accepted.
Its coded check (`gate1_check`) reruns every Phase 0 derivation script on the
pinned GANDALF, applies each script's own pass criterion (and, for D02 and
critic_as2018.py, the criteria below that turn the SPEC.md and claim text into
numbers), checks that the installed GANDALF is the commit pinned in uv.lock,
and checks that every claim the mapping and SPEC.md rest on is supported with a
SUPPORTED critic verdict. It hashes SPEC.md and docs/rediscovery.md so that the
report names what was accepted. `gates.py gate1` refuses to run on a dirty tree.

Scope. Statements of SPEC.md §1, §2, §3 and §6 are checked where a script or a
claim covers them: §1–§2 by D01, D02, the blind test and the claims, and the
parts of §1, §3 and §6 that D03 (`03_spec_solver.py`) names in its docstring
against the installed GANDALF. Other statements of those sections are accepted
as written (critic_it-20261005-1446_1.md lists them). §4 (domain and run classes), §5 (forcing) and §8
(the v0.5.0 checkpoint inventory) are plans and an inventory, accepted as the
plan of record and tested later: §5's pair forcing by Gate 3, the base drive and
the base-state parameters by the base-state gate, and §8 when Phase 1 reads the
checkpoints. §7 holds the frozen tolerances of Gates 2 to 4; §9 lists gaps.

Usage (from the repo root):

    uv run python studies/04-phase-space-helicity/analysis/gates.py gate1 --iteration <id>
    uv run python studies/04-phase-space-helicity/analysis/gates.py finish \
        --report <gate report> --critic <critic report> --kill "none met"
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
STUDY_REL = "studies/04-phase-space-helicity"
GATE_REPORTS = STUDY / "gate_reports"
SCRATCH = STUDY / "data" / "scratch"
GANDALF_PACKAGE = "gandalf-krmhd"

ITERATION_RE = re.compile(r"it-\d{8}-\d{4}")  # always used with fullmatch
UTC_STAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
FROZEN_ITEM_RE = re.compile(r"\S+ sha256 [0-9a-f]{64}, launch L\d{3,}")  # LOOP.md §5 template
CRITIC_FILE_RE = re.compile(r"critic_(it-\d{8}-\d{4})_(\d+)\.md")
REPORT_FILE_RE = re.compile(
    r"G(?:(?P<n>[123])|4_(?P<set>[A-Za-z0-9]+)|(?P<base>base))_(?P<iter>it-\d{8}-\d{4})\.md")

# How the launcher reads a gate report (loop/modal_launch.py, LOOP.md §5 "Gate reports"),
# mirrored here so that the writer cannot produce a line the launcher would count twice.
# A line's label is the line without leading Markdown decoration, in lower case.
LINE_DECORATION = " \t>*-+#|_`"
VERDICT_LABEL_RES = (re.compile(r"^(?:final\s+)?result\b"), re.compile(r"^coded\s+checks?\b"),
                     re.compile(r"^critic\b"), re.compile(r"^kill\s+criteri"))
HEADER_KEYS = ("gate", "evaluated:", "command:", "runs:", "left out:", "frozen", "coded check:",
               "critic:", "kill criteria:", "result:")
TEMPLATE_PLACEHOLDERS = (
    "PASS | FAIL", "<n>", "<S>", "<iteration id>", "<name, with PLAN.md and SPEC.md references>",
    "<UTC time>", "<sha>", "<the exact command line>", "<run IDs, and where each ran>",
    "<run IDs and why>", "<path>", "<hash>", "<freeze key>", "<Lnnn>",
    "<its VERDICT line, verbatim>", "<critic report file>", "<which, with the evidence>",
)
PLACEHOLDER_RE = re.compile(r"<[A-Za-z][^<>\n]{0,80}>")

# ---------------------------------------------------------------------------
# Gate 1 criteria. Fixed before the gate is evaluated (LOOP.md §5); change them
# only with a new SUPPORTED critic verdict and a decisions.md row.
# ---------------------------------------------------------------------------

# Each Phase 0 derivation script and the rule its output must meet:
#   "all-checks":     exit code 0, a line "ALL CHECKS PASSED", no line "FAILED..."
#   "d02":            "all-checks", plus the line "Gamma^odd (echo model) distinguishable
#                     from zero: False" (SPEC.md §2 and C18: zero in the mean by symmetry, so a
#                     consistency check), plus the rerun reference_profiles.npz reproduces the
#                     committed one (D02 is seeded)
#   "claim-verdicts": exit code 0 and exactly GATE1_CRITIC_CLAIMS lines
#                     "Claim <n> ...: SUPPORTED|REFUTED", every one SUPPORTED
#   "as2018":         exit code 0 and the numeric criteria below, from the text of C16
GATE1_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("01_invariant_mapping.py", "all-checks"),
    ("02_reference_profiles.py", "d02"),
    ("03_spec_solver.py", "all-checks"),
    ("blind_invariants.py", "all-checks"),
    ("critic_invariants.py", "claim-verdicts"),
    ("critic_as2018.py", "as2018"),
)
GATE1_CRITIC_CLAIMS = 3
GATE1_D02_ODD_LINE = "Gamma^odd (echo model) distinguishable from zero: False"
GATE1_D02_NPZ = "reference_profiles.npz"
GATE1_D02_RTOL = 1e-8  # rerun vs committed npz; seeded NumPy, so only round-off may differ
# critic_as2018.py criteria, each a reading of a phrase of claim C16 (whose text quotes no
# fitted coefficient). Each row prints the ensemble mean rate gamma and its spread `+-`
# (numpy std, ddof 0) over E realisations, E read from the section 2 header; the standard
# error is se = (+-)/sqrt(E - 1).
#   section 1, "strictly lower-triangular, so not antisymmetric": raising-only ratio
#              > AS_RAISE_ASYM_MIN; antisymmetrised ratio < AS_ANTISYM_MAX (round-off)
#   section 2, "exponentially unstable under both the Ito and the Stratonovich reading":
#              gamma > AS_NSIGMA se in every raising-only row, Ito and Heun
#   section 2, "under the Ito reading gamma <= M S": gamma/(M S) <= 1 + AS_NSIGMA se/(M S)
#              in every Ito raising-only row (the bound is analytic; the margin is noise)
#   section 2, "does not depend on the time step, under both readings": for each
#              (M, kappa, reading), gamma(dt=1e-3)/gamma(dt=3e-4) in AS_DT_RATIO_BAND, or the
#              two rates within AS_NSIGMA combined standard errors
#   section 3, "antisymmetrised, Heun: |gamma| < AS_HEUN_MAX M S in every realisation":
#              |gamma| + AS_NSIGMA (+-) < AS_HEUN_MAX M S in every Heun row; with ddof 0,
#              max_i |gamma_i - mean| <= sqrt(E - 1) (+-) < 3 (+-) for E = 6
AS_RAISE_ASYM_MIN = 0.1
AS_ANTISYM_MAX = 1e-12
AS_NSIGMA = 3.0
AS_DT_RATIO_BAND = (0.8, 1.25)
AS_HEUN_MAX = 0.05
AS_ROWS_PER_SECTION = 24  # M in (16, 32) x 2 dt x 3 kappa x (Ito, Heun)
# Claims the invariant mapping and SPEC.md §1–§2 rest on. C7 (H_ph-sp, decision 4) and
# C8 (the study's question) are open by design; C9 is superseded by C12, itself superseded
# by C14 (its analytic part, a Gate 1 input), and C13 (its GANDALF part, a Phase 1 check
# on the new base state). C1 is superseded by C17 (m = 0 correlator), C11 by C16 (no
# coefficient). C18 is the conjugation symmetry of the D02 echo model.
GATE1_CLAIMS: tuple[str, ...] = ("C2", "C3", "C4", "C5", "C6", "C10", "C14", "C16", "C17", "C18")
# Critic verdicts recorded in Phase 0, before the loop saved critic reports. A newest
# verdict that names no report passes only as an exact copy of one of these. C11's Phase 0
# verdict is hedged ("the hyper-collision statement not tested"), so it is not listed;
# C1 and C11 are superseded and not Gate 1 claims.
GATE1_PHASE0_VERDICTS: dict[str, str] = {
    "C3": "SUPPORTED (critic, worst 1.4e-17; unweighted control 1.5e-3)",
    "C5": "SUPPORTED (critic, worst 8.0e-18; generic closure breaks it, 9e-4)",
}
# Files Gate 1 accepts as written, hashed in the report so that it names what was accepted.
GATE1_ACCEPTED_FILES: tuple[str, ...] = ("SPEC.md", "docs/rediscovery.md")
GATE1_SCRIPT_TIMEOUT_S = 1800.0


# ---------------------------------------------------------------------------
# Shared report machinery (every gate)
# ---------------------------------------------------------------------------

@dataclass
class Row:
    """One row of a gate's results table. `passed` is None for a row that only informs."""

    quantity: str
    value: str
    threshold: str
    passed: bool | None


@dataclass
class GateResult:
    """Everything a gate report states above its `Coded check` line."""

    gate_number: str  # "1", "2", "3", "base" or "4"
    gate_name: str  # the `Gate:` line, with PLAN.md and SPEC.md references
    command: str
    repo_commit: str
    gandalf_commit: str
    runs: str
    left_out: str
    data_files: list[tuple[str, str]]  # (repo-relative path, sha256)
    rows: list[Row]
    run_set: str | None = None  # Gate 4 only
    frozen: list[str] = field(default_factory=list)  # base-state gate and Gate 4 only
    evaluated_utc: str = ""

    @property
    def coded_pass(self) -> bool:
        """The coded check passes when no row fails and there is at least one decisive row."""
        decisive = [r for r in self.rows if r.passed is not None]
        return bool(decisive) and all(r.passed for r in decisive)


def sha256_file(path: Path) -> str:
    """sha256 of a file's bytes, hex."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_rel(path: Path) -> str:
    """Path relative to the repo root, POSIX form."""
    return path.resolve().relative_to(REPO.resolve()).as_posix()


def _cell(text: str) -> str:
    """Text safe for one Markdown table cell (no pipes, no newlines)."""
    return " ".join(str(text).replace("|", "/").split())


def report_name(gate_number: str, iteration: str, run_set: str | None = None) -> str:
    """File name LOOP.md §5 gives a gate report."""
    if not ITERATION_RE.fullmatch(iteration):
        raise ValueError(f"iteration id {iteration!r} is not of the form it-YYYYMMDD-HHMM")
    if gate_number == "base":
        return f"Gbase_{iteration}.md"
    if gate_number == "4":
        if not run_set or not re.fullmatch(r"[A-Za-z0-9]+", run_set):
            raise ValueError("a Gate 4 report needs a run set of letters and digits")
        return f"G4_{run_set}_{iteration}.md"
    if gate_number not in ("1", "2", "3"):
        raise ValueError(f"unknown gate {gate_number!r}")
    return f"G{gate_number}_{iteration}.md"


def report_title(gate_number: str, iteration: str, run_set: str | None = None) -> str:
    """Title line LOOP.md §5 gives a gate report."""
    if gate_number == "4":
        return f"# Gate 4 report, set {run_set}: {iteration}"
    return f"# Gate {gate_number} report: {iteration}"


def parse_report_name(name: str) -> tuple[str, str | None, str]:
    """(gate number, run set or None, iteration id) from a gate report's file name, the
    inverse of `report_name`. Raises ValueError for any other name."""
    m = REPORT_FILE_RE.fullmatch(name)
    if not m:
        raise ValueError(f"{name} is not G<n>_, Gbase_ or G4_<S>_<iteration id>.md")
    gate = m.group("n") or ("4" if m.group("set") else "base")
    return gate, m.group("set"), m.group("iter")


def label_key(line: str) -> str:
    """A report line without leading Markdown decoration, in lower case (the launcher's key)."""
    return line.strip().lstrip(LINE_DECORATION).lower()


def head_form_problems(lines: list[str]) -> list[str]:
    """Lines of a report head (everything above `Coded check`) that the launcher would
    misread: a line it would count as a `Result`, `Coded check`, `Critic` or `Kill criteria`
    line, and unfilled template text."""
    problems = []
    for line in lines:
        key = label_key(line)
        if any(p.match(key) for p in VERDICT_LABEL_RES):
            problems.append(f"line reads as a verdict line: {line!r}")
        if key.startswith(HEADER_KEYS) and PLACEHOLDER_RE.search(line):
            problems.append(f"placeholder on a header line: {line!r}")
    text = "\n".join(lines)
    problems += [f"template text {p!r}" for p in TEMPLATE_PLACEHOLDERS if p in text]
    return problems


def _one_line(text: str) -> bool:
    """True when `text` holds no line break of any kind `str.splitlines` knows."""
    return len((str(text) + "x").splitlines()) == 1


def head_field_problems(result: GateResult) -> list[str]:
    """Fields of a GateResult that cannot go into a head as written: a line break in a field
    that becomes a whole line (it would start a line of its own), a repo commit that is not
    a full 40-hex sha, a data-file digest that is not 64 lower-case hex, or no data file."""
    problems = []
    fields = {"gate_name": result.gate_name, "evaluated_utc": result.evaluated_utc,
              "repo_commit": result.repo_commit, "gandalf_commit": result.gandalf_commit,
              "command": result.command, "runs": result.runs, "left_out": result.left_out}
    fields.update({f"data file {i}": f"{p} {d}" for i, (p, d) in enumerate(result.data_files)})
    fields.update({f"frozen {i}": item for i, item in enumerate(result.frozen)})
    problems += [f"{name} holds a line break" for name, value in fields.items() if not _one_line(value)]
    if not re.fullmatch(r"[0-9a-f]{40}", result.repo_commit):
        problems.append(f"repo commit {result.repo_commit!r} is not a full sha")
    if not result.data_files:
        problems.append("no data file, so the review cannot be bound to the evaluated outputs")
    problems += [f"data file {path!r} digest is not 64 lower-case hex" for path, digest in result.data_files
                 if not re.fullmatch(r"[0-9a-f]{64}", digest) or not path or " sha256 " in path]
    if not UTC_STAMP_RE.fullmatch(result.evaluated_utc):
        problems.append(f"evaluated time {result.evaluated_utc!r} is not of the form YYYY-MM-DDTHH:MM:SSZ")
    problems += [f"frozen item {item!r} is not '<key> sha256 <64 hex>, launch L<nnn>'"
                 for item in result.frozen if not FROZEN_ITEM_RE.fullmatch(item)]
    if result.gate_number in ("base", "4") and not result.frozen:
        problems.append("the base-state gate and Gate 4 need their Frozen lines (LOOP.md §5 template)")
    if result.gate_number not in ("base", "4") and result.frozen:
        problems.append(f"Gate {result.gate_number} has no Frozen line (LOOP.md §5 template)")
    if not result.rows:
        problems.append("no results row (the launcher refuses a table without rows)")
    return problems


def render_report_head(result: GateResult, iteration: str) -> str:
    """The report text from its title down to and including the `Coded check` line.

    Raises ValueError, writing nothing, if a field cannot be written as one line
    (`head_field_problems`) or if a line of the head would be misread by the launcher
    (`head_form_problems`, applied to the lines as they will be read)."""
    report_name(result.gate_number, iteration, result.run_set)  # validates gate, set, iteration
    problems = head_field_problems(result)
    if problems:
        raise ValueError("the report head cannot be written: " + "; ".join(problems))
    lines = [report_title(result.gate_number, iteration, result.run_set), ""]
    lines.append(f"Gate: {result.gate_name}")
    lines.append(f"Evaluated: {result.evaluated_utc}, repo commit {result.repo_commit}, "
                 f"GANDALF commit {result.gandalf_commit}")
    lines.append(f"Command: {result.command}")
    lines.append(f"Runs: {result.runs}")
    lines.append(f"Left out: {result.left_out}")
    lines.append("Data files:")
    lines += [f"- {path} sha256 {digest}" for path, digest in result.data_files]
    if result.frozen:
        lines += [f"Frozen: {item}" for item in result.frozen]
    lines += ["", "| Quantity | Value | Threshold | Pass |", "|---|---|---|---|"]
    for row in result.rows:
        mark = "-" if row.passed is None else ("yes" if row.passed else "no")
        threshold = "-" if row.passed is None else row.threshold
        lines.append(f"| {_cell(row.quantity)} | {_cell(row.value)} | {_cell(threshold)} | {mark} |")
    problems = head_form_problems("\n".join(lines).splitlines())
    if problems:
        raise ValueError("the report head would be misread by the launcher: " + "; ".join(problems))
    lines += ["", f"Coded check: {'PASS' if result.coded_pass else 'FAIL'}"]
    return "\n".join(lines) + "\n"


def parse_report_head(text: str, name: str) -> GateResult:
    """The GateResult that a head in the writer's form states, the inverse of
    `render_report_head` for the report file `name`. Raises ValueError at the first line
    that is not in that form. `finish_report` re-renders the result and compares, so a head
    the writer did not produce, or one edited since, is refused."""
    gate, run_set, _ = parse_report_name(name)
    lines = text.split("\n")
    pos = 2  # title and blank line; the title is compared by the re-render

    def take(pattern: str) -> re.Match:
        nonlocal pos
        m = re.fullmatch(pattern, lines[pos]) if pos < len(lines) else None
        if not m:
            got = lines[pos] if pos < len(lines) else "end of file"
            raise ValueError(f"{name}: line {pos + 1} is not in the writer's form: {got[:80]!r}")
        pos += 1
        return m

    gate_name = take(r"Gate: (.*)").group(1)
    ev = take(r"Evaluated: (.*?), repo commit (\S*), GANDALF commit (.*)")
    command = take(r"Command: (.*)").group(1)
    runs = take(r"Runs: (.*)").group(1)
    left_out = take(r"Left out: (.*)").group(1)
    take(r"Data files:")
    data_files: list[tuple[str, str]] = []
    while pos < len(lines) and lines[pos].startswith("- "):
        m = take(r"- (.+) sha256 ([0-9a-f]{64})")
        data_files.append((m.group(1), m.group(2)))
    frozen: list[str] = []
    while pos < len(lines) and lines[pos].startswith("Frozen: "):
        frozen.append(take(r"Frozen: (.*)").group(1))
    take(r"")
    take(r"\| Quantity \| Value \| Threshold \| Pass \|")
    take(r"\|---\|---\|---\|---\|")
    rows: list[Row] = []
    while pos < len(lines) and lines[pos].startswith("| "):
        cells = take(r"\| (.*) \|").group(1).split(" | ")
        if len(cells) != 4 or cells[3] not in ("yes", "no", "-"):
            raise ValueError(f"{name}: line {pos} is not a results row of the writer's form")
        rows.append(Row(cells[0], cells[1], cells[2], {"yes": True, "no": False, "-": None}[cells[3]]))
    return GateResult(gate_number=gate, gate_name=gate_name, command=command,
                      repo_commit=ev.group(2), gandalf_commit=ev.group(3), runs=runs,
                      left_out=left_out, data_files=data_files, rows=rows, run_set=run_set,
                      frozen=frozen, evaluated_utc=ev.group(1))


def write_report(result: GateResult, iteration: str, out_dir: Path = GATE_REPORTS) -> Path:
    """Write a new gate report down to its `Coded check` line. Never overwrites a report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / report_name(result.gate_number, iteration, result.run_set)
    if path.exists():
        raise FileExistsError(f"{path} exists; a new evaluation gets a new report")
    path.write_text(render_report_head(result, iteration), encoding="utf-8")
    return path


VERDICT_LINE_RE = re.compile(r"VERDICT: [A-Z]+")


def _first_report_mark(lines: list[str]) -> int | None:
    """Index of the first line labelled `report:` (after Markdown decoration, in any case),
    the line the launcher and the runner start from; None if there is none."""
    return next((i for i, line in enumerate(lines) if label_key(line).startswith("report:")), None)


def critic_verdict_line(text: str) -> str | None:
    """The VERDICT line of a saved critic review (LOOP.md §6), or None.

    The launcher (`critic_verdict`) and the runner (`saved_verdict`) take the first line
    labelled `report:`, skip blank lines and fence lines, strip Markdown decoration, and read
    the first line left. This reader accepts only the case in which no skipping or stripping
    happens: the first `report:`-labelled line reads exactly `Report:`, and the first
    non-blank line after it reads exactly `VERDICT: <WORD>`, without fence, backtick, bold or
    quote marks. Then all three readers return the same line; in every other case this one
    returns None, so the writer refuses rather than risk reading a different verdict.
    A later line labelled `report:` is allowed (a finding may start that way) unless it, or
    the first non-blank line after it, starts with `VERDICT` after decoration: then the file
    holds two verdict sections, as when a request quotes a report, and which one is the
    critic's cannot be told from the text, so this returns None.
    """
    lines = text.splitlines()
    mark = _first_report_mark(lines)
    if mark is None or lines[mark] != "Report:":
        return None
    for later in range(mark + 1, len(lines)):
        if not label_key(lines[later]).startswith("report:"):
            continue
        rest = [label_key(lines[later])[len("report:"):]] + lines[later + 1:]
        first = next((label_key(c) for c in rest if label_key(c)), "")
        if first.startswith("verdict"):
            return None
    for candidate in lines[mark + 1:]:
        if candidate.strip():
            return candidate if VERDICT_LINE_RE.fullmatch(candidate) else None
    return None


def critic_request_text(text: str) -> str | None:
    """The `Request:` section of a saved critic review (LOOP.md §6): the lines between its one
    line that reads exactly `Request:` and the first line labelled `report:`, which must read
    exactly `Report:` and come after it; None otherwise. A request that quotes a line starting
    `Report:` cuts the section short there, and `critic_verdict_line` then finds two verdict
    sections and refuses, because the launcher would read the verdict from the quoted line."""
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line == "Request:"]
    end = _first_report_mark(lines)
    if len(starts) != 1 or end is None or lines[end] != "Report:" or not starts[0] < end:
        return None
    return "\n".join(lines[starts[0] + 1:end])


KILL_MET_RE = re.compile(r"^Kill criterion ([123]) met: \S")
# PLAN.md §7: criterion 1 after Phase 1 (Gate 2), 2 after Phase 2 (Gate 3), 3 after Phase 3
# (Gate 4 of set B, the set with two resolutions; LOOP.md §5 "Set B").
KILL_GATE = {"1": ("2", None), "2": ("3", None), "3": ("4", "B")}


def decide_result(coded_pass: bool, verdict: str | None, kill: str, gate: str | None = None,
                  run_set: str | None = None) -> str:
    """`Result:` value from LOOP.md §5.

    PASS: coded check passed, verdict is `VERDICT: SUPPORTED` and no kill criterion met.
    FAIL: a kill criterion is met, or the coded check failed under a SUPPORTED verdict.
    NOT DECIDED: anything else.

    `kill` must be exactly `none met` or `Kill criterion <1|2|3> met: <evidence>` (PLAN.md
    §7). Any other text raises, so that a typo cannot decide a gate. A met criterion needs
    `gate` (and for criterion 3 `run_set`) and is accepted only for the gate that judges it:
    criterion 1 for Gate 2, criterion 2 for Gate 3, criterion 3 for Gate 4 of set B.
    """
    met = KILL_MET_RE.match(kill)
    if kill != "none met" and not met:
        raise ValueError(f"kill statement {kill!r} is neither 'none met' nor "
                         "'Kill criterion <1|2|3> met: <evidence>'")
    if met:
        want_gate, want_set = KILL_GATE[met.group(1)]
        if gate != want_gate or (want_set is not None and run_set != want_set):
            where = f"Gate {want_gate}" + (f" set {want_set}" if want_set else "")
            raise ValueError(f"kill criterion {met.group(1)} is judged by {where}, "
                             f"not Gate {gate}" + (f" set {run_set}" if run_set else ""))
    supported = verdict == "VERDICT: SUPPORTED"
    if kill != "none met":
        return "FAIL"
    if supported:
        return "PASS" if coded_pass else "FAIL"
    return "NOT DECIDED"


def report_history_problems(report: Path, iteration: str, repo: Path = REPO) -> list[str]:
    """Commits that touched a gate report and are not of its own iteration. A report that git
    tracks may be finished only if every commit that touched it has a subject starting
    `Study 04 [<its iteration id>]` (LOOP.md §12: a head committed earlier in the same
    session may get its last three lines; a report committed before the session never
    changes). A file outside the repo has no history and gives no problem."""
    try:
        rel = report.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return []
    has_head = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "-q", "HEAD"],
                              capture_output=True, text=True).returncode == 0
    if not has_head:
        return []  # a repo without commits (tests)
    out = subprocess.run(["git", "-C", str(repo), "log", "--format=%h %s", "--", rel],
                         check=True, capture_output=True, text=True).stdout
    own = f"Study 04 [{iteration}]"
    return [f"commit {line.split(' ', 1)[0]} touched {rel} and is not of {iteration}"
            for line in out.splitlines() if not line.split(" ", 1)[-1].startswith(own)]


def finish_report(report: Path, critic_report: Path, kill: str,
                  reports_dir: Path | None = None,
                  rejudge: Callable[[GateResult, str], GateResult] | None = None) -> str:
    """Append the `Critic`, `Kill criteria` and `Result` lines to a gate report.

    `critic_report` is the saved `gate_reports/critic_<iteration id>_<k>.md`; its VERDICT
    line is copied verbatim. `kill` is "none met" or a statement of which kill criterion is
    met, with the evidence. `rejudge(stated, iteration)` rebuilds the gate's result from the
    outputs the evaluation saved and hashed (Gate 1: `gate1_rejudge`); there is no default,
    so a gate without one cannot be finished. Returns the result (PASS, FAIL or NOT DECIDED).

    Refuses (ValueError, report unchanged) unless:
    - the report's file name is a gate report name, and the head is exactly what
      `render_report_head` writes for the result it states (`parse_report_head`, then
      re-render and compare byte for byte, without newline translation), so the title, the
      table, the final newline and the `Coded check` line, recomputed from the rows, are the
      writer's own;
    - `rejudge` rebuilds, from the saved outputs, the same rows and data files, so a row
      edited after the evaluation, consistently or not, is refused;
    - the report and the critic report sit in the same folder, `reports_dir` if given (the
      command line gives `gate_reports/`), and no commit of another iteration touched the
      report (`report_history_problems`);
    - the critic report is from the same iteration, its title matches its file name and
      names the gate and the report's file name;
    - its `Request:` section quotes the sha256 of the head as written, the repo commit and
      the evaluated time of the report's `Evaluated` line, every sha256 under `Data files:`
      and every sha256 on a `Frozen` line, so that the review is bound to this head and to
      the outputs this evaluation hashed (hashes only: LOOP.md §6 keeps the report and its
      numbers from the critic);
    - its verdict line is read the same way by this writer, the launcher and the runner
      (`critic_verdict_line`).

    What it cannot see: whether the critic was in fact given those files, and whether the
    iteration ID is the current session's; those rest on LOOP.md §6 and on the runner (T1).
    """
    raw = report.read_bytes()
    text = raw.decode("utf-8")
    gate, run_set, report_iteration = parse_report_name(report.name)
    lines = text.splitlines()
    result_re, coded_re, critic_re, kill_re = VERDICT_LABEL_RES
    if any(p.match(label_key(ln)) for ln in lines for p in (result_re, critic_re, kill_re)):
        raise ValueError(f"{report} is already finished")
    stated = parse_report_head(text, report.name)
    if render_report_head(stated, report_iteration) != text:
        raise ValueError(f"{report.name}: the head is not exactly what the writer produces for the "
                         "result it states (title, table, Coded check line or final newline differ)")
    folder = report.resolve().parent
    if critic_report.resolve().parent != folder or (
            reports_dir is not None and reports_dir.resolve() != folder):
        raise ValueError(f"{critic_report} and {report} must both sit in "
                         f"{reports_dir if reports_dir is not None else 'the same folder'}")
    history = report_history_problems(report, report_iteration)
    if history:
        raise ValueError(f"{report.name} may not be finished: " + "; ".join(history))
    m = CRITIC_FILE_RE.fullmatch(critic_report.name)
    if not m:
        raise ValueError(f"{critic_report.name} is not critic_<iteration id>_<k>.md")
    if report_iteration != m.group(1):
        raise ValueError(f"{critic_report.name} is not from the iteration of {report.name}; "
                         "the gate's critic review must come from the same session (LOOP.md §6)")
    critic_text = critic_report.read_text(encoding="utf-8")
    critic_title = critic_text.splitlines()[0] if critic_text else ""
    if not critic_title_ok(critic_title, critic_report.name):
        raise ValueError(f"{critic_report.name}: title does not match the file name")
    if not gate_label_re(text.splitlines()[0]).search(critic_title):
        raise ValueError(f"{critic_report.name}: title does not name the gate of {report.name}")
    if report.name not in critic_title:
        # Binds the review to this evaluation: a review of the criteria, of a claim or of an
        # earlier step of the same iteration cannot name a report that did not exist yet.
        raise ValueError(f"{critic_report.name}: title does not name the gate report {report.name}, "
                         "so it is not the review of this evaluation")
    request = critic_request_text(critic_text)
    if request is None:
        raise ValueError(f"{critic_report.name} has no single Request: section before Report:")
    frozen_hashes = [h for item in stated.frozen for h in re.findall(r"sha256 ([0-9a-f]{64})", item)]
    evidence = ([hashlib.sha256(raw).hexdigest(), stated.repo_commit, stated.evaluated_utc]
                + [digest for _, digest in stated.data_files] + frozen_hashes)
    missing = [h for h in evidence if h not in request]
    if missing:
        raise ValueError(f"{critic_report.name}: the request does not quote {len(missing)} of the "
                         f"head's sha256, the commit, the evaluated time and the hashes (first "
                         f"{missing[0][:12]}), so the review is not bound to this evaluation")
    verdict = critic_verdict_line(critic_text)
    if verdict is None:
        raise ValueError(f"{critic_report} has no VERDICT line that the writer, the launcher and "
                         "the runner would all read the same way")
    if rejudge is None:
        raise ValueError(f"no re-judge for Gate {gate}: its rows cannot be checked against its outputs")
    try:
        rebuilt = render_report_head(rejudge(stated, report_iteration), report_iteration)
    except OSError as exc:
        raise ValueError(f"{report.name}: the saved outputs cannot be read again ({exc})") from exc
    if rebuilt != text:
        raise ValueError(f"{report.name}: the rows or data files differ from those rebuilt from the "
                         "saved outputs, so the head is not the evaluation's")
    kill = " ".join(kill.split())
    result = decide_result(stated.coded_pass, verdict, kill, gate=gate, run_set=run_set)
    tail = (f"Critic: {verdict}, {m.group(1)}, {critic_report.name}\n"
            f"Kill criteria: {kill}\n"
            f"Result: {result}\n")
    report.write_bytes(raw + tail.encode("utf-8"))
    return result


# ---------------------------------------------------------------------------
# Gate 1
# ---------------------------------------------------------------------------

def uv_lock_pin(lock: Path) -> str:
    """The single GANDALF commit that uv.lock pins (40 hex)."""
    data = tomllib.loads(lock.read_text(encoding="utf-8"))
    shas = set()
    for pkg in data.get("package", []):
        if isinstance(pkg, dict) and pkg.get("name") == GANDALF_PACKAGE:
            m = re.search(r"#([0-9a-f]{40})\s*$", str((pkg.get("source") or {}).get("git", "")))
            if m:
                shas.add(m.group(1))
    if len(shas) != 1:
        raise ValueError(f"uv.lock gives {len(shas)} GANDALF commits, not one")
    pin = shas.pop()
    pyproject = lock.parent / "pyproject.toml"
    if pyproject.is_file():
        named = set(re.findall(rf"{GANDALF_PACKAGE}\s*@\s*git\+\S*?@([0-9a-f]{{40}})\b",
                               pyproject.read_text(encoding="utf-8")))
        if named - {pin}:
            raise ValueError(f"pyproject.toml names GANDALF commit(s) {sorted(named)}, uv.lock pins {pin}")
    return pin


def installed_gandalf_commit() -> str:
    """The commit of the installed GANDALF, from its PEP 610 direct_url.json, or 'unknown'."""
    try:
        raw = importlib.metadata.distribution(GANDALF_PACKAGE).read_text("direct_url.json")
        return json.loads(raw or "{}").get("vcs_info", {}).get("commit_id", "unknown")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return "unknown"


def installed_gandalf_intact() -> tuple[bool, str]:
    """Whether every installed GANDALF file with a RECORD hash still has that sha256, so that
    an edit of the installed package cannot pass as the pinned commit. Returns (ok, seen).

    At least one hashed file must lie under `krmhd/`, so that a distribution that hashes only
    its metadata does not pass. Gate 1 calls this through `isolated_install_info`, in an
    interpreter started with `-E -s` like the derivation scripts.

    Limits: RECORD itself carries no hash, so an edit of a file together with its RECORD line
    passes; files RECORD does not list (`.pth` files, `sitecustomize`, stale `.pyc`) are not
    checked. `-E -s` closes `PYTHONPATH` and the user site, not a `.pth` file in the
    environment's own site-packages."""
    import base64

    try:
        dist = importlib.metadata.distribution(GANDALF_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return False, "not installed"
    hashed = [f for f in (dist.files or []) if f.hash and f.hash.mode == "sha256"]
    changed = []
    for f in hashed:
        path = Path(f.locate())
        digest = (base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
                  .rstrip(b"=").decode() if path.is_file() else "missing")
        if digest != f.hash.value:
            changed.append(str(f))
    package = [f for f in hashed if str(f).startswith("krmhd/")]
    seen = f"{len(hashed)} files with RECORD hashes ({len(package)} under krmhd/), {len(changed)} changed"
    if changed:
        seen += ": " + ", ".join(changed[:3])
    return bool(package) and not changed, seen


def isolated_install_info(python: str = sys.executable) -> tuple[str, bool, str]:
    """(installed GANDALF commit, intact, what was seen), computed in a fresh interpreter
    started with `-E -s`, as the derivation scripts are, so that a GANDALF distribution
    named by `PYTHONPATH` or the user site cannot answer for the one the scripts import."""
    proc = subprocess.run([python, "-E", "-s", str(Path(__file__).resolve()), "install-info"],
                          capture_output=True, text=True, timeout=300)
    try:
        info = json.loads(proc.stdout)
        return str(info["commit"]), bool(info["intact"]), str(info["seen"])
    except (ValueError, KeyError, TypeError):
        return "unknown", False, f"install check failed (exit {proc.returncode})"


def judge_script(rule: str, returncode: int, stdout: str,
                 npz_new: Path | None = None, npz_ref: Path | None = None) -> tuple[bool, str]:
    """Apply a script's pass rule (GATE1_SCRIPTS) to its exit code and output.

    `npz_new` and `npz_ref` are the rerun and the committed reference_profiles.npz, used
    by the "d02" rule only. Returns (passed, what was seen).
    """
    lines = [ln.strip() for ln in stdout.splitlines()]
    if rule in ("all-checks", "d02"):
        has_pass = "ALL CHECKS PASSED" in lines
        failed = any(ln.startswith("FAILED") for ln in lines)
        ok = returncode == 0 and has_pass and not failed
        seen = f"exit {returncode}; ALL CHECKS PASSED {'present' if has_pass else 'absent'}; " \
               f"FAILED line {'present' if failed else 'absent'}"
        if rule == "d02":
            odd_ok = lines.count(GATE1_D02_ODD_LINE) == 1
            npz_ok, npz_seen = compare_npz(npz_new, npz_ref, GATE1_D02_RTOL)
            ok = ok and odd_ok and npz_ok
            seen += f"; Gamma^odd line {'False' if odd_ok else 'not False or missing'}; {npz_seen}"
        return ok, seen
    if rule == "claim-verdicts":
        verdicts = [m.group(1) for ln in lines
                    if (m := re.match(r"^Claim \d+\b.*:\s*(SUPPORTED|REFUTED)$", ln))]
        n_sup = verdicts.count("SUPPORTED")
        seen = f"exit {returncode}; {len(verdicts)} Claim verdict lines, {n_sup} SUPPORTED"
        ok = returncode == 0 and len(verdicts) == GATE1_CRITIC_CLAIMS and n_sup == GATE1_CRITIC_CLAIMS
        return ok, seen
    if rule == "as2018":
        ok, seen = judge_as2018(stdout)
        return returncode == 0 and ok, f"exit {returncode}; {seen}"
    raise ValueError(f"unknown rule {rule!r}")


def compare_npz(new: Path | None, ref: Path | None, rtol: float) -> tuple[bool, str]:
    """Whether a rerun .npz reproduces a committed one: same keys, shapes and dtypes kinds,
    numeric arrays allclose (rtol, NaN padding equal, atol = rtol x the array's max |value|),
    other arrays equal. Returns (passed, what was seen)."""
    import numpy as np  # local import: the report machinery itself needs only the stdlib

    if new is None or ref is None or not new.is_file() or not ref.is_file():
        return False, "npz missing"
    with np.load(new, allow_pickle=False) as a, np.load(ref, allow_pickle=False) as b:
        if sorted(a.files) != sorted(b.files):
            return False, f"npz keys differ ({len(a.files)} vs {len(b.files)})"
        worst, bad = 0.0, []
        for key in b.files:
            x, y = a[key], b[key]
            if x.shape != y.shape:
                bad.append(key)
                continue
            if np.issubdtype(y.dtype, np.number) and np.issubdtype(x.dtype, np.number):
                scale = float(np.nanmax(np.abs(y))) if y.size and np.isfinite(y).any() else 0.0
                if not np.allclose(x, y, rtol=rtol, atol=rtol * scale, equal_nan=True):
                    bad.append(key)
                fin = np.isfinite(y) & np.isfinite(x)
                if fin.any() and scale > 0:
                    worst = max(worst, float(np.max(np.abs(x[fin] - y[fin]))) / scale)
            elif not np.array_equal(x, y):
                bad.append(key)
    if bad:
        return False, f"npz differs in {len(bad)} arrays: {', '.join(sorted(bad)[:5])}"
    return True, f"npz reproduces committed copy ({len(b.files)} arrays, worst diff/max {worst:.1e})"


_AS_ROW_RE = re.compile(
    r"^\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(Ito|Heun)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
_AS_SEC1_RE = re.compile(r"^\s*(AS2018 raising-only|antisymmetrised d_v)\s*:.*=\s*(\S+)\s*$")
_AS_ENS_RE = re.compile(r"ensemble mean \+- std over (\d+) realisations")


def parse_as2018(stdout: str) -> dict:
    """Parse critic_as2018.py output into its section 1 ratios, its ensemble size and its
    section 2 and 3 rows.

    Rows are dicts with M, dt, kappa, interp, gamma, pm (the `+-` spread), MS (= M S) and
    ratio (= gamma/(M S)). `ens` is the E of the section 2 header, or None.
    """
    sec1: dict[str, float] = {}
    rows: dict[str, list[dict]] = {"2": [], "3": []}
    ens: list[int] = []
    section = None
    for line in stdout.splitlines():
        head = re.match(r"^=== (\w+)\.", line)
        if head:
            section = head.group(1)
            continue
        if section == "1" and (m := _AS_SEC1_RE.match(line)):
            sec1[m.group(1)] = float(m.group(2))
        elif section == "2" and (m := _AS_ENS_RE.search(line)):
            ens.append(int(m.group(1)))
        elif section in rows and (m := _AS_ROW_RE.match(line)):
            rows[section].append({"M": int(m.group(1)), "dt": float(m.group(2)),
                                  "kappa": float(m.group(4)), "interp": m.group(5),
                                  "gamma": float(m.group(6)), "pm": float(m.group(7)),
                                  "MS": float(m.group(9)), "ratio": float(m.group(10))})
    return {"sec1": sec1, "raise": rows["2"], "antisym": rows["3"],
            "ens": ens[0] if len(ens) == 1 else None}


def _se(row: dict, ens: int) -> float:
    """Standard error of a row's mean rate from its ddof-0 spread over `ens` realisations."""
    return row["pm"] / math.sqrt(ens - 1)


def judge_as2018(stdout: str) -> tuple[bool, str]:
    """The critic_as2018.py criteria (see AS_* constants). Returns (passed, what was seen)."""
    try:
        p = parse_as2018(stdout)
    except ValueError as exc:
        return False, f"unparseable output ({exc})"
    raise_r, anti_r, sec1, ens = p["raise"], p["antisym"], p["sec1"], p["ens"]
    if (len(raise_r) != AS_ROWS_PER_SECTION or len(anti_r) != AS_ROWS_PER_SECTION
            or set(sec1) != {"AS2018 raising-only", "antisymmetrised d_v"}
            or ens is None or ens < 2):
        return False, (f"output incomplete: {len(sec1)} section-1 ratios, {len(raise_r)} and "
                       f"{len(anti_r)} rows, ensemble size {ens} (expected 2, "
                       f"{AS_ROWS_PER_SECTION}, {AS_ROWS_PER_SECTION}, >= 2)")
    checks: list[tuple[str, bool]] = []
    n_bad = sum(1 for r in raise_r + anti_r
                if not all(math.isfinite(r[key]) for key in ("gamma", "pm", "MS", "ratio"))
                or r["MS"] <= 0 or r["pm"] < 0)
    checks.append((f"{n_bad} rows with a non-finite or invalid gamma, +-, M S or ratio", n_bad == 0))
    if n_bad:
        return False, "; ".join(text for text, _ in checks)
    checks.append((f"raising-only asym {sec1['AS2018 raising-only']:.2e} > {AS_RAISE_ASYM_MIN}",
                   math.isfinite(sec1["AS2018 raising-only"])
                   and sec1["AS2018 raising-only"] > AS_RAISE_ASYM_MIN))
    checks.append((f"antisym asym {sec1['antisymmetrised d_v']:.1e} < {AS_ANTISYM_MAX:.0e}",
                   math.isfinite(sec1["antisymmetrised d_v"])
                   and sec1["antisymmetrised d_v"] < AS_ANTISYM_MAX))
    signif = [r["gamma"] / _se(r, ens) if _se(r, ens) > 0 else math.inf for r in raise_r]
    checks.append((f"raising-only gamma/se {_span(signif)} > {AS_NSIGMA:g} (Ito and Heun)",
                   all(r["gamma"] > AS_NSIGMA * _se(r, ens) and r["gamma"] > 0 for r in raise_r)))
    ito = [r for r in raise_r if r["interp"] == "Ito"]
    bound_ok = all(r["gamma"] <= r["MS"] + AS_NSIGMA * _se(r, ens) for r in ito)
    checks.append((f"Ito gamma/(M S) {_span([r['ratio'] for r in ito])} <= 1 + "
                   f"{AS_NSIGMA:g} se/(M S)", len(ito) == AS_ROWS_PER_SECTION // 2 and bound_ok))
    by_key: dict[tuple, dict[float, dict]] = {}
    for r in raise_r:
        by_key.setdefault((r["M"], r["kappa"], r["interp"]), {})[r["dt"]] = r
    lo_b, hi_b = AS_DT_RATIO_BAND
    ratios, dt_ok = [], len(by_key) == AS_ROWS_PER_SECTION // 2
    for pair in by_key.values():
        if len(pair) != 2:
            dt_ok = False
            continue
        a, b = (pair[dt] for dt in sorted(pair, reverse=True))  # a: larger dt
        ratio = a["gamma"] / b["gamma"] if b["gamma"] != 0 else math.nan
        ratios.append(ratio)
        close = abs(a["gamma"] - b["gamma"]) <= AS_NSIGMA * math.hypot(_se(a, ens), _se(b, ens))
        dt_ok = dt_ok and ((math.isfinite(ratio) and lo_b <= ratio <= hi_b) or close)
    checks.append((f"dt ratios {_span(ratios)} in [{lo_b}, {hi_b}] or within {AS_NSIGMA:g} "
                   "combined se (Ito and Heun)", dt_ok))
    heun = [r for r in anti_r if r["interp"] == "Heun"]
    worst = [(abs(r["gamma"]) + AS_NSIGMA * r["pm"]) / r["MS"] for r in heun]
    checks.append((f"antisym Heun (|gamma| + {AS_NSIGMA:g} (+-))/(M S) {_span(worst)} < {AS_HEUN_MAX}",
                   len(heun) == AS_ROWS_PER_SECTION // 2 and all(w < AS_HEUN_MAX for w in worst)))
    failed = [text for text, ok in checks if not ok]
    seen = "; ".join(text for text, _ in checks)
    return not failed, seen


def _span(values: list[float]) -> str:
    """'lo–hi' of a list, for the report, with the count of non-finite values if any."""
    finite = [v for v in values if math.isfinite(v)]
    bad = len(values) - len(finite)
    if not finite:
        return f"none finite ({bad} non-finite)" if bad else "none"
    text = f"{min(finite):.3f}–{max(finite):.3f}"
    return text + (f" ({bad} non-finite)" if bad else "")


def claims_table(path: Path) -> dict[str, tuple[str, str]]:
    """{claim ID: (critic verdict cell, status cell)} from claims.md.

    Cells are taken from the right-hand end of each row, because claim text may hold `|`.
    """
    out: dict[str, tuple[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\|\s*(C\d+)\s*\|", line)
        if not m:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        out[m.group(1)] = (cells[-2], cells[-1])
    return out


LOOP_VERDICT_RE = re.compile(
    r"^VERDICT: SUPPORTED \((it-\d{8}-\d{4}), (?:`?(?:gate_reports/)?)(critic_(it-\d{8}-\d{4})_\d+\.md)`?\)$")


def judge_claim(claim: str, table: dict[str, tuple[str, str]], reports_dir: Path,
                phase0: dict[str, str] | None = None) -> tuple[bool, str]:
    """A Gate 1 claim passes when its status is exactly `supported` and its newest critic
    verdict (first `<br>`-separated entry) is one of:

    - a loop verdict, exactly `VERDICT: SUPPORTED (<iteration id>, <critic report>)` in the
      form of LOOP.md §6, where the report is `critic_<same iteration id>_<k>.md`, exists in
      `reports_dir`, has `VERDICT: SUPPORTED` as the first line of its Report, and names the
      claim ID in its title line;
    - a Phase 0 verdict, recorded before the loop saved critic reports, that is an exact
      copy of the text in `phase0` (GATE1_PHASE0_VERDICTS) for that claim.

    Anything else, hedged text included, fails.
    """
    phase0 = GATE1_PHASE0_VERDICTS if phase0 is None else phase0
    if claim not in table:
        return False, "no row in claims.md"
    verdict_cell, status = table[claim]
    newest = verdict_cell.split("<br>")[0].strip()
    seen = f"status {status!r}; newest verdict {newest[:70]!r}"
    if status != "supported":
        return False, seen
    if claim in phase0 and newest == phase0[claim]:
        return True, seen + " (Phase 0 verdict)"
    m = LOOP_VERDICT_RE.match(newest)
    if not m:
        return False, seen + "; not a recognised SUPPORTED verdict"
    if m.group(1) != m.group(3):
        return False, seen + "; report file is from another iteration"
    path = reports_dir / m.group(2)
    if not path.is_file():
        return False, seen + f"; {m.group(2)} missing"
    text = path.read_text(encoding="utf-8")
    if critic_verdict_line(text) != "VERDICT: SUPPORTED":
        return False, seen + f"; {m.group(2)} does not say VERDICT: SUPPORTED"
    title = text.splitlines()[0] if text else ""
    if not critic_title_ok(title, path.name):
        return False, seen + f"; {m.group(2)} title does not match its file name"
    if re.search(r"(?i)\bgate", title) or not re.search(rf"\bclaim {re.escape(claim)}\b", title):
        return False, seen + f"; {m.group(2)} title is not a review of claim {claim}"
    return True, seen


def critic_title_ok(title: str, file_name: str) -> bool:
    """Whether a saved critic report's title is `# Critic report <id>_<k>: ...` with the
    `<id>_<k>` of its file name `critic_<id>_<k>.md` (LOOP.md §6)."""
    m = CRITIC_FILE_RE.fullmatch(file_name)
    return bool(m) and title.startswith(f"# Critic report {m.group(1)}_{m.group(2)}: ")


def gate_label_re(report_title: str) -> re.Pattern:
    """Pattern the critic report's title must match to name the gate of a gate report title
    (`# Gate <n> report...`), in the spellings LOOP.md §6 lists. For Gate 4 the title must
    also name the set, as `set <S>` or `7a.<S>`. The patterns are the runner's own
    (`gate_matchers` in loop/guards.py, copied as text), so a title this accepts is one the
    runner's T1 check also takes as naming the gate; the report's file name alone, such as
    `Gbase_<id>.md`, does not count."""
    m = re.match(r"^# Gate (\w+) report", report_title)
    if not m:
        raise ValueError(f"not a gate report title: {report_title!r}")
    gate = m.group(1)
    if gate == "base":
        return re.compile(r"\bbase[\s_-]*state[\s_-]*gate\b|\bgate[\s_-]*base\b|\bGbase\b|\b7a\.base\b",
                          re.IGNORECASE)
    if gate == "4":
        s = re.match(r"^# Gate 4 report, set ([A-Za-z0-9]+): ", report_title)
        if not s:
            raise ValueError(f"Gate 4 report title names no set: {report_title!r}")
        run_set = re.escape(s.group(1))
        return re.compile(rf"^(?=.*(?i:\bgate[\s_-]*4\b))(?=.*\b(?:[Ss]et[\s_-]*{run_set}|7a\.{run_set})\b)")
    return re.compile(rf"\bgate[\s_-]*{gate}\b", re.IGNORECASE)


def gate1_judge(work: Path, pin: str, installed: str, intact: tuple[bool, str],
                derivations: Path = STUDY / "derivations",
                reports_dir: Path = GATE_REPORTS,
                scripts: tuple[tuple[str, str], ...] = GATE1_SCRIPTS,
                claims: tuple[str, ...] = GATE1_CLAIMS,
                study: Path = STUDY,
                accepted: tuple[str, ...] = GATE1_ACCEPTED_FILES) -> tuple[list[Row], list[tuple[str, str]]]:
    """Gate 1 rows and data files from the outputs `gate1_check` saved in `work`: each
    script's stdout and exit code, D02's rerun npz and the copy of claims.md."""
    rows = [Row("installed GANDALF commit", installed, f"uv.lock pin {pin}", installed == pin),
            Row("installed GANDALF files", intact[1], "every RECORD sha256 unchanged", intact[0])]
    data_files: list[tuple[str, str]] = []

    def saved(path: Path) -> Path:
        data_files.append((_rel_or_name(path), sha256_file(path) if path.is_file() else "missing"))
        return path

    for name, rule in scripts:
        saved(derivations / name)
        out = saved(work / f"{Path(name).stem}.stdout").read_text(encoding="utf-8")
        saved(work / f"{Path(name).stem}.stderr")
        exit_text = saved(work / f"{Path(name).stem}.exit").read_text(encoding="utf-8").strip()
        rc = int(exit_text) if re.fullmatch(r"-?\d+", exit_text) else -1
        npz_new = npz_ref = None
        if rule == "d02":
            npz_new, npz_ref = work / GATE1_D02_NPZ, derivations / GATE1_D02_NPZ
            for npz in (npz_ref, npz_new):
                if npz.is_file():
                    saved(npz)
        ok, seen = judge_script(rule, rc, out, npz_new=npz_new, npz_ref=npz_ref)
        rows.append(Row(f"derivations/{name}", seen, _RULE_TEXT[rule], ok))

    for rel in accepted:
        saved(study / rel)
    table = claims_table(saved(work / "claims.md"))
    for claim in claims:
        ok, seen = judge_claim(claim, table, reports_dir)
        rows.append(Row(f"claims.md {claim}", seen,
                        "status supported; newest critic verdict SUPPORTED", ok))
    return rows, data_files


def gate1_rejudge(stated: GateResult, iteration: str, scratch: Path = SCRATCH,
                  lock: Path = REPO / "uv.lock", **kw) -> GateResult:
    """The Gate 1 result rebuilt from the outputs saved in `scratch/gate1_<iteration>/`, with
    the install checked again, for `finish_report` to compare with the head it finishes."""
    pin = uv_lock_pin(lock)
    info = kw.pop("install_info", None) or isolated_install_info()
    rows, data_files = gate1_judge(scratch / f"gate1_{iteration}", pin, info[0], info[1:], **kw)
    return GateResult(**{**stated.__dict__, "rows": rows, "data_files": data_files})


def git_head(repo: Path = REPO) -> str:
    """Full sha of HEAD."""
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


def gate1_check(iteration: str,
                derivations: Path = STUDY / "derivations",
                claims_path: Path = STUDY / "claims.md",
                reports_dir: Path = GATE_REPORTS,
                scratch: Path = SCRATCH,
                lock: Path = REPO / "uv.lock",
                scripts: tuple[tuple[str, str], ...] = GATE1_SCRIPTS,
                claims: tuple[str, ...] = GATE1_CLAIMS,
                installed: str | None = None,
                intact: tuple[bool, str] | None = None,
                timeout_s: float = GATE1_SCRIPT_TIMEOUT_S,
                python: str = sys.executable,
                study: Path = STUDY,
                accepted: tuple[str, ...] = GATE1_ACCEPTED_FILES) -> GateResult:
    """Gate 1 coded check (PLAN.md §4 Phase 0; SPEC.md §1–§2).

    Each derivation script is copied into `scratch/gate1_<iteration>/` and run there, so
    that a script that writes beside itself (D02 writes reference_profiles.npz) leaves
    the committed copy alone. Its stdout, stderr and exit code are saved beside it, and
    claims.md is copied there, so that `gate1_judge` builds every row from saved, hashed
    files and `gate1_rejudge` can build them again when the report is finished.
    """
    if not ITERATION_RE.fullmatch(iteration):
        raise ValueError(f"iteration id {iteration!r} is not of the form it-YYYYMMDD-HHMM")
    head = _safe_head()  # read before the scripts run, so the report names what ran
    work = scratch / f"gate1_{iteration}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    pin = uv_lock_pin(lock)
    if installed is None or intact is None:
        info = isolated_install_info(python)
        installed = info[0] if installed is None else installed
        intact = info[1:] if intact is None else intact

    for name, _ in scripts:
        shutil.copy2(derivations / name, work / name)
        try:
            # -E -s: no PYTHON* variables and no user site, so `import krmhd` resolves to the
            # installed, RECORD-checked package and not to a folder named in the environment
            proc = subprocess.run([python, "-E", "-s", name], cwd=work, capture_output=True, text=True,
                                  timeout=timeout_s)
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc = -1
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = f"timed out after {timeout_s:.0f} s"
        for suffix, text in (("stdout", out), ("stderr", err), ("exit", f"{rc}\n")):
            (work / f"{Path(name).stem}.{suffix}").write_text(text, encoding="utf-8")
    shutil.copy2(claims_path, work / "claims.md")

    rows, data_files = gate1_judge(work, pin, installed, intact, derivations=derivations,
                                   reports_dir=reports_dir, scripts=scripts, claims=claims,
                                   study=study, accepted=accepted)
    return GateResult(
        gate_number="1",
        gate_name=("Gate 1, invariant mapping and SPEC.md accepted (PLAN.md §4 Phase 0; SPEC.md "
                   "§1–§3 and §6 checked where D01–D03 or a claim covers them, §4, §5 and §8 "
                   "accepted as the plan of record)"),
        command=f"uv run python {STUDY_REL}/analysis/gates.py gate1 --iteration {iteration}",
        repo_commit=head,
        gandalf_commit=installed,
        runs="none; Gate 1 reruns the Phase 0 derivation scripts listed under Data files",
        left_out="none",
        data_files=data_files,
        rows=rows,
        evaluated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


_RULE_TEXT = {
    "all-checks": "exit 0; ALL CHECKS PASSED present; no FAILED line",
    "d02": ("exit 0; ALL CHECKS PASSED present; no FAILED line; Gamma^odd line False; "
            f"npz reproduces committed copy (rtol {GATE1_D02_RTOL:.0e})"),
    "claim-verdicts": f"exit 0; {GATE1_CRITIC_CLAIMS} Claim verdict lines, all SUPPORTED",
    "as2018": (f"exit 0; raising-only asym > {AS_RAISE_ASYM_MIN}; antisym asym < {AS_ANTISYM_MAX:.0e}; "
               f"raising-only gamma > {AS_NSIGMA:g} se (Ito and Heun); Ito gamma <= M S + {AS_NSIGMA:g} se; "
               f"dt ratio in [{AS_DT_RATIO_BAND[0]}, {AS_DT_RATIO_BAND[1]}] or within {AS_NSIGMA:g} "
               f"combined se (Ito and Heun); antisym Heun |gamma| + {AS_NSIGMA:g} (+-) < {AS_HEUN_MAX} M S"),
}


# Where a hidden index bit is refused: the launcher's folders, and the lock and project files
# that the pin row reads
CLEAN_TREE_PATHS = (STUDY_REL, "shared", "uv.lock", "pyproject.toml")


def hidden_index_bits(ls_files_v: str) -> list[str]:
    """Files that `git ls-files -v` marks skip-worktree (`S`) or assume-unchanged (a lower-case
    tag): `git status` does not show a change to them."""
    return [line[2:] for line in ls_files_v.splitlines()
            if len(line) > 2 and (line[0] == "S" or line[0].islower())]


def tree_is_clean(repo: Path = REPO, paths: tuple[str, ...] = CLEAN_TREE_PATHS) -> bool:
    """True when `git status --porcelain --untracked-files=all` prints nothing (untracked files
    count, whatever `status.showUntrackedFiles` says) and no tracked file under `paths` has a
    skip-worktree or assume-unchanged bit. The bit is checked only there, because the runner
    sets skip-worktree on another study's data link (LOOP.md §2)."""
    out = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
                         check=True, capture_output=True, text=True).stdout
    bits = subprocess.run(["git", "-C", str(repo), "ls-files", "-v", "--", *paths], check=True,
                          capture_output=True, text=True).stdout
    return out.strip() == "" and not hidden_index_bits(bits)


def _rel_or_name(path: Path) -> str:
    """Repo-relative path when inside the repo, else the bare file name (tests)."""
    try:
        return repo_rel(path)
    except ValueError:
        return path.name


def _safe_head() -> str:
    try:
        return git_head()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

# The re-judge of each gate that `finish` may finish; a gate missing here cannot be finished.
REJUDGE: dict[str, Callable[[GateResult, str], GateResult]] = {"1": gate1_rejudge}

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("gate1", help="run the Gate 1 coded check and write its report")
    p1.add_argument("--iteration", required=True)
    pf = sub.add_parser("finish", help="add the Critic, Kill criteria and Result lines")
    pf.add_argument("--report", required=True, type=Path)
    pf.add_argument("--critic", required=True, type=Path)
    pf.add_argument("--kill", required=True)
    sub.add_parser("install-info", help="print the installed GANDALF commit and RECORD check as JSON")
    args = parser.parse_args(argv)

    if args.cmd == "install-info":
        ok, seen = installed_gandalf_intact()
        print(json.dumps({"commit": installed_gandalf_commit(), "intact": ok, "seen": seen}))
        return 0
    if args.cmd == "gate1":
        if not tree_is_clean():
            print("Refusing: the working tree is not clean, so the report could not name the "
                  "commit that produced it. Commit first.", file=sys.stderr)
            return 2
        result = gate1_check(args.iteration)
        # Limit: a change made and undone between the two checks is not seen.
        if not tree_is_clean() or _safe_head() != result.repo_commit:
            print("Refusing: the tree or HEAD changed while the gate ran; no report written.",
                  file=sys.stderr)
            return 2
        path = write_report(result, args.iteration)
        print(path.read_text(encoding="utf-8"), end="")
        print(f"Wrote {repo_rel(path)}")
        return 0 if result.coded_pass else 1
    gate = parse_report_name(args.report.name)[0]
    result_line = finish_report(args.report, args.critic, args.kill, reports_dir=GATE_REPORTS,
                                rejudge=REJUDGE.get(gate))
    print(f"Result: {result_line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
