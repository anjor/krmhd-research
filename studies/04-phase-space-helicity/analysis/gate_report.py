"""Study 04 gate reports: the shared writer of LOOP.md §5 and §6.

Every gate writes its report in two steps, so that no number and no verdict is typed by hand:

1. `write_report` writes the head, everything down to the `Coded check` line, from the
   GateResult that the gate builds from its saved outputs.
2. `finish_report` builds the head again from the same saved outputs, refuses unless the
   report is still exactly that head, and appends the `Critic`, `Kill criteria` and `Result`
   lines from the critic's saved review.

It imports only the standard library and calls no loader, so that a frozen evaluation
(`analysis/gate_base.py`, `analysis/gate4_<S>.py`) may import it under the launcher's rule
(LOOP.md §5, item 2). It is then frozen with that evaluation, so a change after the first
launch of a set is a hard stop. The Gate 1 to 3 checks and the git checks stay in
`analysis/gates.py`.

Threat model (Anjor's answers to QUESTIONS.md Q9 and Q10, 10 October 2026). The machinery
must catch honest mistakes (a report naming the wrong commit, a dirty tree, a check that did
not run or did not finish, a verdict or Result line that does not follow from the saved
outputs, a criterion changed after review) and a report head edited after the fact. It does
not defend against a session that sets git, Python, JAX or XLA environment variables, plants
or edits ignored files, or forges a saved critic file. The runner and the launcher are part
of the defence, and this module does not repeat their checks: the runner's T1 check refuses a
report that says PASS unless a `study04-critic` review of that gate from the same session
returned SUPPORTED, every review naming the gate did, and the saved critic file holds that
review verbatim, and it runs the launcher's report check on the report; its commit guards
refuse a change to a committed report (`report-edit`) and a second Gate 4 decision; the
launcher's `check-report`, which every launch applies, refuses a report whose form it would
misread (a second `Result` line, template text, a table row that fails).

What this module checks:
- the head at finish is byte for byte the head that the gate's `rebuild` makes from the saved
  outputs, which include the evaluation's repo commit and UTC time (Gate 1: the record that
  `gate1_check` saves), so a head edited after it was written, a report already finished, or
  outputs changed since are refused;
- the critic file is `critic_<the report's iteration id>_<k>.md` and its title names the
  report's file name, so that a review of the criteria or of a claim is not taken for the
  review of this evaluation;
- the verdict is read only from the critic's own VERDICT line (`critic_verdict_line`), never
  from text quoted in the request;
- the `Result` line follows from the coded check, that verdict and the kill statement by the
  rule of LOOP.md §5 (`decide_result`).

Stated limits, outside the model or left to the runner, the launcher and LOOP.md:
- Git runs in the inherited environment: `GIT_*` variables, the global and system config and
  the per-user exclude file (`core.excludesFile`) can change what it sees. A tracked file with
  a skip-worktree or assume-unchanged bit does not show in `git status`, so the clean-tree
  check misses a change to it (the launcher refuses such bits under `shared/` and the study
  folder with `[clean-tree]`).
- Python may run a stale or planted `.pyc` from an ignored `__pycache__/` in place of the
  committed source, and `JAX_*` and `XLA_*` variables reach the scripts.
- The saved outputs and the record of the repo commit and time live in the ignored
  `data/scratch/`. An edit of them is outside the model; a legitimate rerun into the same
  folder makes the head unfinishable, which fails safe.
- Symbolic links are followed. A saved critic file forged or edited to say SUPPORTED is caught
  only by the runner (T1).
- The writer does not check that the head reads correctly to the launcher, that the critic's
  title or request names the gate, which commits touched the report, or that the gate code at
  finish is the code of the named commit; the launcher, the runner and the critic's review of
  the evaluation cover those.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
GATE_REPORTS = STUDY / "gate_reports"

ITERATION_RE = re.compile(r"it-\d{8}-\d{4}")  # always used with fullmatch
CRITIC_FILE_RE = re.compile(r"critic_(it-\d{8}-\d{4})_(\d+)\.md")
REPORT_FILE_RE = re.compile(
    r"G(?:(?P<n>[123])|4_(?P<set>[A-Za-z0-9]+)|(?P<base>base))_(?P<iter>it-\d{8}-\d{4})\.md")
# Markdown decoration the launcher and the runner strip before they read a line's label
LINE_DECORATION = " \t>*-+#|_`"
VERDICT_LINE_RE = re.compile(r"VERDICT: [A-Z]+")


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
    """A line without leading Markdown decoration, in lower case (how the launcher and the
    runner find a line's label)."""
    return line.strip().lstrip(LINE_DECORATION).lower()


def render_report_head(result: GateResult, iteration: str) -> str:
    """The report text from its title down to and including the `Coded check` line, in the
    template of LOOP.md §5. Raises ValueError if a field that fills a line holds a line break,
    because it would start a line of its own."""
    report_name(result.gate_number, iteration, result.run_set)  # validates gate, set, iteration
    fields = [result.gate_name, result.evaluated_utc, result.repo_commit, result.gandalf_commit,
              result.command, result.runs, result.left_out, *result.frozen,
              *(f"{p} {d}" for p, d in result.data_files)]
    broken = [f for f in fields if len((str(f) + "x").splitlines()) != 1]
    if broken:
        raise ValueError(f"a report field holds a line break: {broken[0][:60]!r}")
    lines = [report_title(result.gate_number, iteration, result.run_set), ""]
    lines.append(f"Gate: {result.gate_name}")
    lines.append(f"Evaluated: {result.evaluated_utc}, repo commit {result.repo_commit}, "
                 f"GANDALF commit {result.gandalf_commit}")
    lines.append(f"Command: {result.command}")
    lines.append(f"Runs: {result.runs}")
    lines.append(f"Left out: {result.left_out}")
    lines.append("Data files:")
    lines += [f"- {path} sha256 {digest}" for path, digest in result.data_files]
    lines += [f"Frozen: {item}" for item in result.frozen]
    lines += ["", "| Quantity | Value | Threshold | Pass |", "|---|---|---|---|"]
    for row in result.rows:
        mark = "-" if row.passed is None else ("yes" if row.passed else "no")
        threshold = "-" if row.passed is None else row.threshold
        lines.append(f"| {_cell(row.quantity)} | {_cell(row.value)} | {_cell(threshold)} | {mark} |")
    lines += ["", f"Coded check: {'PASS' if result.coded_pass else 'FAIL'}"]
    return "\n".join(lines) + "\n"


def write_report(result: GateResult, iteration: str, out_dir: Path = GATE_REPORTS) -> Path:
    """Write a new gate report down to its `Coded check` line. Never overwrites a report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / report_name(result.gate_number, iteration, result.run_set)
    if path.exists():
        raise FileExistsError(f"{path} exists; a new evaluation gets a new report")
    path.write_bytes(render_report_head(result, iteration).encode("utf-8"))
    return path


def critic_verdict_line(text: str) -> str | None:
    """The critic's own VERDICT line of a saved critic review (LOOP.md §6), or None.

    A saved review is a title, a line `Request:`, the request, a line `Report:` and the
    critic's report, which starts with its VERDICT line. The verdict is read only when the
    file has exactly one line that reads `Request:` and exactly one line labelled `report:`
    (after Markdown decoration, in any case), that line reads exactly `Report:` and comes
    after `Request:`, and the first non-blank line after it reads exactly `VERDICT: <WORD>`.
    Otherwise this returns None and the writer refuses. A request that quotes a `Report:`
    line, or a report that holds one, gives two labelled lines, so a verdict quoted in the
    request is never read as the critic's, whatever follows it. With a single marker and the
    verdict on the first non-blank line after it, the launcher and the runner, which read from
    the first line labelled `report:`, read the same line.
    """
    lines = text.splitlines()
    requests = [i for i, line in enumerate(lines) if line == "Request:"]
    marks = [i for i, line in enumerate(lines) if label_key(line).startswith("report:")]
    if len(requests) != 1 or len(marks) != 1 or lines[marks[0]] != "Report:" or requests[0] > marks[0]:
        return None
    first = next((line for line in lines[marks[0] + 1:] if line.strip()), "")
    return first if VERDICT_LINE_RE.fullmatch(first) else None


def critic_title_ok(title: str, file_name: str) -> bool:
    """Whether a saved critic report's title is `# Critic report <id>_<k>: ...` with the
    `<id>_<k>` of its file name `critic_<id>_<k>.md` (LOOP.md §6)."""
    m = CRITIC_FILE_RE.fullmatch(file_name)
    return bool(m) and title.startswith(f"# Critic report {m.group(1)}_{m.group(2)}: ")


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

    As LOOP.md §5 states the rule, a met criterion gives FAIL whatever the verdict. Whether a
    criterion is met is counted under LOOP.md §5 "Kill criteria 1 and 2", where an evaluation
    the critic REFUTED does not count, so the loop states a met criterion only after that
    count; this function does not count evaluations.
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
        return "FAIL"
    if verdict == "VERDICT: SUPPORTED":
        return "PASS" if coded_pass else "FAIL"
    return "NOT DECIDED"


def finish_report(report: Path, critic_report: Path, kill: str,
                  rebuild: Callable[[str], GateResult] | None) -> str:
    """Append the `Critic`, `Kill criteria` and `Result` lines to a gate report and return
    the result (PASS, FAIL or NOT DECIDED).

    `rebuild(iteration)` is the gate's function that builds its GateResult from the outputs
    the evaluation saved (Gate 1: `gate1_result`); there is no default, so a gate without one
    cannot be finished. `critic_report` is the saved `critic_<iteration id>_<k>.md`. `kill` is
    `none met` or `Kill criterion <n> met: <evidence>` (`decide_result`).

    Refuses (ValueError, report unchanged) unless the report's bytes are exactly the head
    that `render_report_head` makes from `rebuild`, for the gate, set and iteration of the
    report's file name; the critic file is named `critic_<that iteration id>_<k>.md` and its
    title line names the report's file name; and `critic_verdict_line` reads its verdict.
    The module docstring gives the threat model and the limits.
    """
    gate, run_set, iteration = parse_report_name(report.name)
    if rebuild is None:
        raise ValueError(f"no rebuild for the gate of {report.name}: it cannot be finished")
    raw = report.read_bytes()
    try:
        rebuilt = rebuild(iteration)
    except OSError as exc:
        raise ValueError(f"{report.name}: the saved outputs cannot be read ({exc})") from exc
    if (rebuilt.gate_number, rebuilt.run_set) != (gate, run_set):
        raise ValueError(f"{report.name}: the rebuild is of Gate {rebuilt.gate_number}")
    if raw != render_report_head(rebuilt, iteration).encode("utf-8"):
        raise ValueError(f"{report.name} is not exactly the head that the saved outputs give: "
                         "edited since it was written, already finished, or the outputs changed")
    m = CRITIC_FILE_RE.fullmatch(critic_report.name)
    if not m or m.group(1) != iteration:
        raise ValueError(f"{critic_report.name} is not critic_{iteration}_<k>.md; the gate's "
                         "critic review comes from the same iteration (LOOP.md §6)")
    critic_text = critic_report.read_text(encoding="utf-8")
    if report.name not in critic_text.split("\n", 1)[0]:
        raise ValueError(f"{critic_report.name}: its title does not name {report.name}, so it "
                         "is not the review of this evaluation")
    verdict = critic_verdict_line(critic_text)
    if verdict is None:
        raise ValueError(f"{critic_report.name}: no single `Report:` line followed by the "
                         "critic's own VERDICT line")
    kill = " ".join(kill.split())
    result = decide_result(rebuilt.coded_pass, verdict, kill, gate=gate, run_set=run_set)
    tail = (f"Critic: {verdict}, {iteration}, {critic_report.name}\n"
            f"Kill criteria: {kill}\n"
            f"Result: {result}\n")
    report.write_bytes(raw + tail.encode("utf-8"))
    return result
