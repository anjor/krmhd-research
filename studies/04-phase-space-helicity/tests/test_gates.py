"""Tests of analysis/gates.py: the shared gate-report writer and the Gate 1 machinery.

They run on stand-in scripts, a stand-in claims table and a stand-in uv.lock in a
temporary folder, never on the real derivations, so they produce no gate evidence.

    uv run python -m unittest discover -s studies/04-phase-space-helicity/tests -v
"""

from __future__ import annotations

import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

import gates  # noqa: E402

PIN = "a" * 40
LOCK = textwrap.dedent(f"""\
    version = 1

    [[package]]
    name = "gandalf-krmhd"
    version = "0.6.0"
    source = {{ git = "https://github.com/anjor/gandalf.git?rev=v0.6.0#{PIN}" }}

    [[package]]
    name = "numpy"
    version = "2.0.0"
    """)

CLAIMS_HEADER = ("| ID | Claim | Origin | Evidence | Falsifier | Critic verdict | Status |\n"
                 "|---|---|---|---|---|---|---|\n")

SCRIPT_ALL_PASS = "print('check a ok')\nprint()\nprint('ALL CHECKS PASSED')\n"
SCRIPT_ALL_FAIL = "import sys\nprint('FAILED: [x]')\nsys.exit(1)\n"
SCRIPT_WRITES = ("from pathlib import Path\n"
                 "Path(__file__).resolve().parent.joinpath('out.npz').write_text('x')\n"
                 "print('ALL CHECKS PASSED')\n")
SCRIPT_CLAIMS = ("print('VERDICTS (tolerance 1e-10):')\n"
                 "print('  Claim 1 (W conserved): SUPPORTED')\n"
                 "print('  Claim 2 (Gamma conserved under Z and C): SUPPORTED')\n"
                 "print('  Claim 3 negative controls (E): SUPPORTED')\n"
                 "print('  Exploratory: Gamma under generic closure G: NOT conserved')\n")
SCRIPT_MEASURE = "print('=== 2. growth rates ===')\n"

CRITIC_OK = ("# Critic report it-20261002-0916_1: test\n\nRequest:\nsomething\n\n"
             "Report:\nVERDICT: SUPPORTED\nEvidence: fine\n")
CRITIC_BAD = CRITIC_OK.replace("VERDICT: SUPPORTED", "VERDICT: INCONCLUSIVE")

ITER = "it-20261002-0916"


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class JudgeScriptTests(unittest.TestCase):
    def test_all_checks(self) -> None:
        self.assertTrue(gates.judge_script("all-checks", 0, "x\n\nALL CHECKS PASSED\n")[0])
        self.assertFalse(gates.judge_script("all-checks", 1, "ALL CHECKS PASSED\n")[0])
        self.assertFalse(gates.judge_script("all-checks", 0, "x\n")[0])
        self.assertFalse(gates.judge_script("all-checks", 0, "FAILED: [a]\nALL CHECKS PASSED\n")[0])

    def test_claim_verdicts(self) -> None:
        good = "  Claim 1 (a): SUPPORTED\n  Claim 2 (b; c): SUPPORTED\n  Claim 3 x (d): SUPPORTED\n"
        self.assertTrue(gates.judge_script("claim-verdicts", 0, good)[0])
        self.assertFalse(gates.judge_script("claim-verdicts", 0, good.replace("2 (b; c): SUPPORTED",
                                                                              "2 (b; c): REFUTED"))[0])
        self.assertFalse(gates.judge_script("claim-verdicts", 0, "  Claim 1 (a): SUPPORTED\n")[0])
        self.assertFalse(gates.judge_script("claim-verdicts", 1, good)[0])

    def test_exit_zero(self) -> None:
        self.assertTrue(gates.judge_script("exit-zero", 0, "")[0])
        self.assertFalse(gates.judge_script("exit-zero", 2, "")[0])

    def test_unknown_rule(self) -> None:
        with self.assertRaises(ValueError):
            gates.judge_script("maybe", 0, "")


class ClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.reports = self.tmp / "gate_reports"
        write(self.reports / "critic_it-20261002-0916_1.md", CRITIC_OK)
        write(self.reports / "critic_it-20261002-0916_2.md", CRITIC_BAD)
        rows = [
            "| C1 | W = (1 − 1/Λ)|g_0|² + Σ|g_m|² is conserved | agent | D01 | x | SUPPORTED (critic, worst 1e-17) | supported |",
            "| C2 | a | agent | b | c | — | supported |",
            "| C3 | a | agent | b | c | VERDICT: SUPPORTED (it-20261002-0916, critic_it-20261002-0916_1.md) | supported |",
            "| C4 | a | agent | b | c | VERDICT: SUPPORTED (it-20261002-0916, critic_it-20261002-0916_2.md) | supported |",
            "| C5 | a | agent | b | c | VERDICT: SUPPORTED (it-20261002-0916, critic_it-20261002-0916_9.md) | supported |",
            "| C6 | a | agent | b | c | VERDICT: INCONCLUSIVE (it-20261002-0916, critic_it-20261002-0916_2.md)<br>SUPPORTED (old) | supported |",
            "| C7 | a | agent | b | c | SUPPORTED (old) | supported (blind SymPy only) |",
        ]
        self.claims = write(self.tmp / "claims.md", "# Claims\n\n" + CLAIMS_HEADER + "\n".join(rows) + "\n")
        self.table = gates.claims_table(self.claims)

    def test_pipes_in_claim_text(self) -> None:
        self.assertEqual(self.table["C1"], ("SUPPORTED (critic, worst 1e-17)", "supported"))

    def test_judgements(self) -> None:
        expect = {"C1": True, "C2": False, "C3": True, "C4": False, "C5": False, "C6": False,
                  "C7": False, "C99": False}
        for claim, want in expect.items():
            with self.subTest(claim=claim):
                self.assertEqual(gates.judge_claim(claim, self.table, self.reports)[0], want)


class DecideResultTests(unittest.TestCase):
    def test_matrix(self) -> None:
        sup, inc = "VERDICT: SUPPORTED", "VERDICT: INCONCLUSIVE"
        self.assertEqual(gates.decide_result(True, sup, "none met"), "PASS")
        self.assertEqual(gates.decide_result(False, sup, "none met"), "FAIL")
        self.assertEqual(gates.decide_result(True, inc, "none met"), "NOT DECIDED")
        self.assertEqual(gates.decide_result(False, "VERDICT: REFUTED", "none met"), "NOT DECIDED")
        self.assertEqual(gates.decide_result(True, sup, "criterion 1 met: residual flat in dt"), "FAIL")
        self.assertEqual(gates.decide_result(True, inc, "criterion 2 met"), "FAIL")


class Gate1CheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.deriv = self.tmp / "derivations"
        write(self.deriv / "d01.py", SCRIPT_ALL_PASS)
        write(self.deriv / "d02.py", SCRIPT_WRITES)
        write(self.deriv / "crit.py", SCRIPT_CLAIMS)
        write(self.deriv / "meas.py", SCRIPT_MEASURE)
        write(self.deriv / "bad.py", SCRIPT_ALL_FAIL)
        self.lock = write(self.tmp / "uv.lock", LOCK)
        self.reports = self.tmp / "gate_reports"
        write(self.reports / "critic_it-20261002-0916_1.md", CRITIC_OK)
        row = "| {c} | a | agent | b | c | VERDICT: SUPPORTED (it-20261002-0916, critic_it-20261002-0916_1.md) | supported |"
        self.claims = write(self.tmp / "claims.md",
                            CLAIMS_HEADER + "\n".join(row.format(c=c) for c in ("C1", "C2")) + "\n")
        self.scripts = (("d01.py", "all-checks"), ("d02.py", "all-checks"),
                        ("crit.py", "claim-verdicts"), ("meas.py", "exit-zero"))

    def run_check(self, **kw) -> gates.GateResult:
        args = dict(derivations=self.deriv, claims_path=self.claims, reports_dir=self.reports,
                    scratch=self.tmp / "scratch", lock=self.lock, scripts=self.scripts,
                    claims=("C1", "C2"), installed=PIN, timeout_s=60)
        args.update(kw)
        return gates.gate1_check(ITER, **args)

    def test_pass(self) -> None:
        result = self.run_check()
        self.assertTrue(result.coded_pass, [r for r in result.rows if not r.passed])
        self.assertEqual(len(result.rows), 1 + 4 + 2)
        # D02-like script wrote into the scratch copy, not beside the original
        self.assertFalse((self.deriv / "out.npz").exists())
        self.assertTrue((self.tmp / "scratch" / f"gate1_{ITER}" / "out.npz").exists())
        # every script, its two logs and claims.md are hashed
        self.assertEqual(len(result.data_files), 4 * 3 + 1)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for _, h in result.data_files))

    def test_failing_script(self) -> None:
        result = self.run_check(scripts=self.scripts + (("bad.py", "all-checks"),))
        self.assertFalse(result.coded_pass)
        self.assertEqual([r.quantity for r in result.rows if not r.passed], ["derivations/bad.py"])

    def test_wrong_pin(self) -> None:
        result = self.run_check(installed="b" * 40)
        self.assertFalse(result.coded_pass)
        self.assertFalse(result.rows[0].passed)

    def test_missing_claim(self) -> None:
        result = self.run_check(claims=("C1", "C2", "C3"))
        self.assertFalse(result.coded_pass)

    def test_timeout(self) -> None:
        write(self.deriv / "slow.py", "import time\ntime.sleep(5)\n")
        result = self.run_check(scripts=(("slow.py", "exit-zero"),), timeout_s=0.5)
        self.assertFalse(result.coded_pass)

    def test_lock_pin(self) -> None:
        self.assertEqual(gates.uv_lock_pin(self.lock), PIN)
        two = write(self.tmp / "two.lock", LOCK + LOCK.split("\n\n", 1)[1].replace(PIN, "c" * 40))
        with self.assertRaises(ValueError):
            gates.uv_lock_pin(two)


class ReportTests(unittest.TestCase):
    """The report form of LOOP.md §5, as the launcher reads it."""

    LABELS = ("result", "coded check", "critic", "kill criteria")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.critic_ok = write(self.tmp / "critic_it-20261002-0916_1.md", CRITIC_OK)
        self.critic_bad = write(self.tmp / "critic_it-20261002-0916_2.md", CRITIC_BAD)

    def make(self, passed: bool) -> gates.GateResult:
        return gates.GateResult(
            gate_number="1", gate_name="Gate 1 test (PLAN.md §4)", command="uv run python x.py",
            repo_commit="1" * 40, gandalf_commit=PIN, runs="none", left_out="none",
            data_files=[("studies/x/derivations/critic_invariants.py", "0" * 64)],
            rows=[gates.Row("derivations/critic_invariants.py", "exit 0 | ok", "exit 0", passed),
                  gates.Row("note", "info", "n/a", None)],
            evaluated_utc="2026-10-02T09:30:00Z")

    def label_lines(self, text: str, label: str) -> list[str]:
        return [ln for ln in text.splitlines()
                if ln.strip().lstrip("#>*-| \t").lower().startswith(label)]

    def test_head_and_finish_pass(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        self.assertEqual(path.name, f"G1_{ITER}.md")
        self.assertEqual(path.read_text().splitlines()[0], f"# Gate 1 report: {ITER}")
        self.assertEqual(gates.finish_report(path, self.critic_ok, "none met"), "PASS")
        text = path.read_text()
        for label in self.LABELS:
            self.assertEqual(len(self.label_lines(text, label)), 1, label)
        self.assertIn("Result: PASS\n", text)
        self.assertIn("Coded check: PASS\n", text)
        self.assertIn(f"Critic: VERDICT: SUPPORTED, {ITER}, critic_{ITER}_1.md\n", text)
        self.assertIn("Kill criteria: none met\n", text)
        self.assertIn("| exit 0 / ok |", text)  # no pipe inside a cell
        self.assertIn("| note | info | - | - |", text)  # informative row
        self.assertNotRegex(text, r"<[a-z][a-z ]*>")  # no template placeholders

    def test_finish_not_decided_and_fail(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        self.assertEqual(gates.finish_report(path, self.critic_bad, "none met"), "NOT DECIDED")
        other = self.tmp / "other"
        path2 = gates.write_report(self.make(False), ITER, other)
        self.assertIn("Coded check: FAIL", path2.read_text())
        self.assertIn("| no |", path2.read_text())
        self.assertEqual(gates.finish_report(path2, self.critic_ok, "none met"), "FAIL")

    def test_no_overwrite_no_refinish(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        with self.assertRaises(FileExistsError):
            gates.write_report(self.make(True), ITER, self.tmp)
        gates.finish_report(path, self.critic_ok, "none met")
        with self.assertRaises(ValueError):
            gates.finish_report(path, self.critic_ok, "none met")

    def test_no_decisive_row_fails(self) -> None:
        result = self.make(True)
        result.rows = [gates.Row("note", "info", "n/a", None)]
        self.assertFalse(result.coded_pass)

    def test_names(self) -> None:
        self.assertEqual(gates.report_name("base", ITER), f"Gbase_{ITER}.md")
        self.assertEqual(gates.report_name("4", ITER, "A"), f"G4_A_{ITER}.md")
        self.assertEqual(gates.report_title("4", ITER, "A"), f"# Gate 4 report, set A: {ITER}")
        with self.assertRaises(ValueError):
            gates.report_name("1", "it-2026-1002")
        with self.assertRaises(ValueError):
            gates.report_name("4", ITER, "A-2")

    def test_critic_verdict_line(self) -> None:
        self.assertEqual(gates.critic_verdict_line(CRITIC_OK), "VERDICT: SUPPORTED")
        fenced = CRITIC_OK.replace("Report:\n", "Report:\n```\n")
        self.assertEqual(gates.critic_verdict_line(fenced), "VERDICT: SUPPORTED")
        self.assertIsNone(gates.critic_verdict_line("Report:\nfine\n"))


if __name__ == "__main__":
    unittest.main()
