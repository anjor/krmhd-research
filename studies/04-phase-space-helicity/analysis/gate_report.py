"""Study 04 gate reports: the shared writer and reader of LOOP.md §5 and §6.

Every gate writes its report through this module, in two steps, so that no number and no
verdict is typed by hand: `write_report` writes everything down to the `Coded check` line,
and `finish_report` appends the `Critic`, `Kill criteria` and `Result` lines from the
critic's saved review, after checking that the head is the writer's own, that the gate's
re-judge rebuilds the same rows from the saved outputs, and that the review is bound to this
evaluation.

It imports only the standard library, and calls no loader (`importlib`, `compile`, `eval`,
`exec`), so that a frozen evaluation (`analysis/gate_base.py`, `analysis/gate4_<S>.py`) may
import it under the launcher's rule (LOOP.md §5, item 2). It is then frozen with that
evaluation, so a change after the first launch of a set is a hard stop. The Gate 1 to 3
checks, which need GANDALF and the environment, stay in `analysis/gates.py`.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
GATE_REPORTS = STUDY / "gate_reports"

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

    - neither file is a symbolic link, so the `Critic` line names the file that was read.

    What it cannot see: whether the critic was in fact given those files, whether the
    iteration ID is the current session's, and whether another review of the same session
    that names the gate returned something other than SUPPORTED; those rest on LOOP.md §6 and
    on the runner (T1).
    """
    links = [p for p in (report, critic_report) if p.is_symlink()]
    if links:
        raise ValueError(f"{links[0]} is a symbolic link; finish the files themselves")
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
