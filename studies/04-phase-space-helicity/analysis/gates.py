"""Study 04 gates: coded checks and the shared gate-report writer (LOOP.md §5).

A quality gate passes when its coded check here passes on raw outputs, the
`study04-critic` subagent returns `VERDICT: SUPPORTED` on that evidence, and no
kill criterion is met. Every gate writes its report in two steps, both in this
file, so that no number and no verdict is typed by hand:

1. `write_report` writes everything down to the `Coded check` line.
2. `finish_report` appends the `Critic`, `Kill criteria` and `Result` lines,
   copying the critic's VERDICT line from its saved report
   (`gate_reports/critic_<iteration id>_<k>.md`).

Gate 1 (PLAN.md §4, Phase 0): the invariant mapping and SPEC.md are accepted.
Its coded check (`gate1_check`) reruns every Phase 0 derivation script on the
pinned GANDALF, applies each script's own pass criterion, checks that the
installed GANDALF is the commit pinned in uv.lock, and checks that every claim
the mapping and SPEC.md rest on is supported with a SUPPORTED critic verdict.

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
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
REPO = STUDY.parents[1]
STUDY_REL = "studies/04-phase-space-helicity"
GATE_REPORTS = STUDY / "gate_reports"
SCRATCH = STUDY / "data" / "scratch"
GANDALF_PACKAGE = "gandalf-krmhd"

ITERATION_RE = re.compile(r"^it-\d{8}-\d{4}$")
CRITIC_FILE_RE = re.compile(r"critic_(it-\d{8}-\d{4})_(\d+)\.md")

# ---------------------------------------------------------------------------
# Gate 1 criteria. Fixed before the gate is evaluated (LOOP.md §5); change them
# only with a new SUPPORTED critic verdict and a decisions.md row.
# ---------------------------------------------------------------------------

# Each Phase 0 derivation script and the rule its output must meet:
#   "all-checks":     exit code 0, a line "ALL CHECKS PASSED", no line "FAILED..."
#   "claim-verdicts": exit code 0 and exactly GATE1_CRITIC_CLAIMS lines
#                     "Claim <n> ...: SUPPORTED|REFUTED", every one SUPPORTED
#   "exit-zero":      exit code 0 (the script prints measurements, not a verdict;
#                     its numbers are the critic's to judge)
GATE1_SCRIPTS: tuple[tuple[str, str], ...] = (
    ("01_invariant_mapping.py", "all-checks"),
    ("02_reference_profiles.py", "all-checks"),
    ("blind_invariants.py", "all-checks"),
    ("critic_invariants.py", "claim-verdicts"),
    ("critic_as2018.py", "exit-zero"),
)
GATE1_CRITIC_CLAIMS = 3
# Claims the invariant mapping and SPEC.md §1–§2 rest on. C7 (H_ph-sp, decision 4)
# and C8 (the study's question) are open by design; C9's GANDALF part is a Phase 1
# check. None of those three is a Gate 1 input.
GATE1_CLAIMS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6", "C10", "C11")
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
    if not ITERATION_RE.match(iteration):
        raise ValueError(f"iteration id {iteration!r} is not of the form it-YYYYMMDD-HHMM")
    if gate_number == "base":
        return f"Gbase_{iteration}.md"
    if gate_number == "4":
        if not run_set or not run_set.isalnum():
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


def render_report_head(result: GateResult, iteration: str) -> str:
    """The report text from its title down to and including the `Coded check` line."""
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
    lines += ["", f"Coded check: {'PASS' if result.coded_pass else 'FAIL'}"]
    return "\n".join(lines) + "\n"


def write_report(result: GateResult, iteration: str, out_dir: Path = GATE_REPORTS) -> Path:
    """Write a new gate report down to its `Coded check` line. Never overwrites a report."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / report_name(result.gate_number, iteration, result.run_set)
    if path.exists():
        raise FileExistsError(f"{path} exists; a new evaluation gets a new report")
    path.write_text(render_report_head(result, iteration), encoding="utf-8")
    return path


def critic_verdict_line(text: str) -> str | None:
    """The first line after `Report:` in a saved critic review (LOOP.md §6), or None."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        key = line.strip().lstrip("#>*-| ")
        if not key.lower().startswith("report:"):
            continue
        for candidate in [key[len("report:"):]] + lines[i + 1:]:
            if candidate.strip().startswith("```"):
                continue
            value = candidate.strip().strip("`*_>").strip()
            if value:
                return value if value.startswith("VERDICT:") else None
    return None


def decide_result(coded_pass: bool, verdict: str | None, kill: str) -> str:
    """`Result:` value from LOOP.md §5.

    PASS: coded check passed, verdict is `VERDICT: SUPPORTED` and no kill criterion met.
    FAIL: a kill criterion is met, or the coded check failed under a SUPPORTED verdict.
    NOT DECIDED: anything else.
    """
    supported = verdict == "VERDICT: SUPPORTED"
    kill_met = kill.strip() != "none met"
    if kill_met:
        return "FAIL"
    if supported:
        return "PASS" if coded_pass else "FAIL"
    return "NOT DECIDED"


def finish_report(report: Path, critic_report: Path, kill: str) -> str:
    """Append the `Critic`, `Kill criteria` and `Result` lines to a gate report.

    `critic_report` is the saved `gate_reports/critic_<iteration id>_<k>.md`; its VERDICT
    line is copied verbatim. `kill` is "none met" or a statement of which kill criterion is
    met, with the evidence. Returns the result (PASS, FAIL or NOT DECIDED).
    """
    text = report.read_text(encoding="utf-8")
    if re.search(r"(?mi)^\W*(critic|kill criteria|result)\s*:", text):
        raise ValueError(f"{report} is already finished")
    coded = re.findall(r"(?m)^Coded check: (PASS|FAIL)$", text)
    if len(coded) != 1:
        raise ValueError(f"{report} has no single Coded check line")
    m = CRITIC_FILE_RE.fullmatch(critic_report.name)
    if not m:
        raise ValueError(f"{critic_report.name} is not critic_<iteration id>_<k>.md")
    verdict = critic_verdict_line(critic_report.read_text(encoding="utf-8"))
    if verdict is None:
        raise ValueError(f"{critic_report} has no VERDICT line after Report:")
    kill = " ".join(kill.split())
    result = decide_result(coded[0] == "PASS", verdict, kill)
    tail = (f"Critic: {verdict}, {m.group(1)}, {critic_report.name}\n"
            f"Kill criteria: {kill}\n"
            f"Result: {result}\n")
    report.write_text(text + tail, encoding="utf-8")
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
    return shas.pop()


def installed_gandalf_commit() -> str:
    """The commit of the installed GANDALF, from its PEP 610 direct_url.json, or 'unknown'."""
    try:
        raw = importlib.metadata.distribution(GANDALF_PACKAGE).read_text("direct_url.json")
        return json.loads(raw or "{}").get("vcs_info", {}).get("commit_id", "unknown")
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return "unknown"


def judge_script(rule: str, returncode: int, stdout: str) -> tuple[bool, str]:
    """Apply a script's pass rule (GATE1_SCRIPTS) to its exit code and output.

    Returns (passed, what was seen).
    """
    lines = [ln.strip() for ln in stdout.splitlines()]
    if rule == "all-checks":
        has_pass = "ALL CHECKS PASSED" in lines
        failed = any(ln.startswith("FAILED") for ln in lines)
        seen = f"exit {returncode}; ALL CHECKS PASSED {'present' if has_pass else 'absent'}; " \
               f"FAILED line {'present' if failed else 'absent'}"
        return returncode == 0 and has_pass and not failed, seen
    if rule == "claim-verdicts":
        verdicts = [m.group(1) for ln in lines
                    if (m := re.match(r"^Claim \d+\b.*:\s*(SUPPORTED|REFUTED)$", ln))]
        n_sup = verdicts.count("SUPPORTED")
        seen = f"exit {returncode}; {len(verdicts)} Claim verdict lines, {n_sup} SUPPORTED"
        ok = returncode == 0 and len(verdicts) == GATE1_CRITIC_CLAIMS and n_sup == GATE1_CRITIC_CLAIMS
        return ok, seen
    if rule == "exit-zero":
        return returncode == 0, f"exit {returncode}"
    raise ValueError(f"unknown rule {rule!r}")


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


def judge_claim(claim: str, table: dict[str, tuple[str, str]], reports_dir: Path) -> tuple[bool, str]:
    """A Gate 1 claim passes when its status is exactly `supported` and its newest critic
    verdict (first `<br>`-separated entry) starts with SUPPORTED or `VERDICT: SUPPORTED`.
    If that verdict names a saved critic report, the report must exist and its Report
    section must start `VERDICT: SUPPORTED`. A verdict that names no report is a Phase 0
    critic verdict recorded before the loop saved reports.
    """
    if claim not in table:
        return False, "no row in claims.md"
    verdict_cell, status = table[claim]
    newest = verdict_cell.split("<br>")[0].strip()
    seen = f"status {status!r}; newest verdict {newest[:70]!r}"
    if status != "supported":
        return False, seen
    if not re.match(r"^(VERDICT: )?SUPPORTED\b", newest):
        return False, seen
    named = CRITIC_FILE_RE.search(newest)
    if named:
        path = reports_dir / named.group(0)
        if not path.is_file():
            return False, seen + f"; {named.group(0)} missing"
        if critic_verdict_line(path.read_text(encoding="utf-8")) != "VERDICT: SUPPORTED":
            return False, seen + f"; {named.group(0)} does not say VERDICT: SUPPORTED"
    return True, seen


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
                timeout_s: float = GATE1_SCRIPT_TIMEOUT_S,
                python: str = sys.executable) -> GateResult:
    """Gate 1 coded check (PLAN.md §4 Phase 0; SPEC.md §1–§2).

    Each derivation script is copied into `scratch/gate1_<iteration>/` and run there, so
    that a script that writes beside itself (D02 writes reference_profiles.npz) leaves
    the committed copy alone. Its stdout and stderr are saved beside it and hashed.
    """
    work = scratch / f"gate1_{iteration}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    rows: list[Row] = []
    data_files: list[tuple[str, str]] = []

    pin = uv_lock_pin(lock)
    have = installed if installed is not None else installed_gandalf_commit()
    rows.append(Row("installed GANDALF commit", have, f"uv.lock pin {pin}", have == pin))

    for name, rule in scripts:
        src = derivations / name
        data_files.append((_rel_or_name(src), sha256_file(src)))
        dst = work / name
        shutil.copy2(src, dst)
        try:
            proc = subprocess.run([python, name], cwd=work, capture_output=True, text=True,
                                  timeout=timeout_s)
            rc, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as exc:
            rc = -1
            out = exc.stdout.decode() if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            err = f"timed out after {timeout_s:.0f} s"
        for suffix, text in (("stdout", out), ("stderr", err)):
            log = work / f"{Path(name).stem}.{suffix}"
            log.write_text(text, encoding="utf-8")
            data_files.append((_rel_or_name(log), sha256_file(log)))
        ok, seen = judge_script(rule, rc, out)
        rows.append(Row(f"derivations/{name}", seen, _RULE_TEXT[rule], ok))

    data_files.append((_rel_or_name(claims_path), sha256_file(claims_path)))
    table = claims_table(claims_path)
    for claim in claims:
        ok, seen = judge_claim(claim, table, reports_dir)
        rows.append(Row(f"claims.md {claim}", seen,
                        "status supported; newest critic verdict SUPPORTED", ok))

    return GateResult(
        gate_number="1",
        gate_name="Gate 1, invariant mapping and SPEC.md accepted (PLAN.md §4 Phase 0; SPEC.md §1–§2)",
        command=f"uv run python {STUDY_REL}/analysis/gates.py gate1 --iteration {iteration}",
        repo_commit=_safe_head(),
        gandalf_commit=have,
        runs="none; Gate 1 reruns the Phase 0 derivation scripts listed under Data files",
        left_out="none",
        data_files=data_files,
        rows=rows,
        evaluated_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )


_RULE_TEXT = {
    "all-checks": "exit 0; ALL CHECKS PASSED present; no FAILED line",
    "claim-verdicts": f"exit 0; {GATE1_CRITIC_CLAIMS} Claim verdict lines, all SUPPORTED",
    "exit-zero": "exit 0",
}


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

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("gate1", help="run the Gate 1 coded check and write its report")
    p1.add_argument("--iteration", required=True)
    pf = sub.add_parser("finish", help="add the Critic, Kill criteria and Result lines")
    pf.add_argument("--report", required=True, type=Path)
    pf.add_argument("--critic", required=True, type=Path)
    pf.add_argument("--kill", required=True)
    args = parser.parse_args(argv)

    if args.cmd == "gate1":
        result = gate1_check(args.iteration)
        path = write_report(result, args.iteration)
        print(path.read_text(encoding="utf-8"), end="")
        print(f"Wrote {repo_rel(path)}")
        return 0 if result.coded_pass else 1
    result_line = finish_report(args.report, args.critic, args.kill)
    print(f"Result: {result_line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
