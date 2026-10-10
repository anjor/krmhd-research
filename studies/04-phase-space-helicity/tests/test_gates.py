"""Tests of analysis/gates.py and analysis/gate_report.py: the gate-report writer and Gate 1.

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
from collections.abc import Callable
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
ODD_FALSE = "Gamma^odd (echo model) distinguishable from zero: False"
SCRIPT_D02 = ("from pathlib import Path\n"
              "import numpy as np\n"
              "here = Path(__file__).resolve().parent\n"
              "np.savez(here / 'reference_profiles.npz', W=np.array([1.0, 0.5, np.nan]),\n"
              "         nl_form=np.array('antisym'), M=np.array(2))\n"
              f"print({ODD_FALSE!r})\n"
              "print('ALL CHECKS PASSED')\n")

CRITIC_OK = ("# Critic report it-20261002-0916_1: test\n\nRequest:\nsomething\n\n"
             "Report:\nVERDICT: SUPPORTED\nEvidence: fine\n")
CRITIC_BAD = CRITIC_OK.replace("VERDICT: SUPPORTED", "VERDICT: INCONCLUSIVE")

ITER = "it-20261002-0916"


def as2018_output(asym_raise: float = 1.0, asym_anti: float = 1e-16, growth: float = 0.5,
                  dt_ratio: float = 1.0, heun_growth: float = 0.6, heun_dt_ratio: float = 1.0,
                  pm_rel: float = 0.02, heun: float = 0.001, anti_pm_ms: float = 0.001,
                  ens: int | None = 6, drop_row: bool = False) -> str:
    """Stand-in stdout in the format of derivations/critic_as2018.py (not its numbers).

    Section 2 Ito rows have gamma/(M S) = `growth` at dt = 3e-4 and `growth * dt_ratio` at
    dt = 1e-3, Heun rows the same with `heun_growth` and `heun_dt_ratio`; their spread `+-`
    is `pm_rel` x |gamma|. Section 3 Heun rows have gamma/(M S) = `heun` and spread
    `anti_pm_ms` x M S. `ens=None` leaves out the ensemble-size header.
    """
    hdr = f"{'M':>3} {'dt':>7} {'T':>5} {'kappa':>8} {'interp':>6} {'gamma':>9} {'+-':>7} " \
          f"{'gamma_m':>9} {'M*S':>8} {'gamma/(M S)':>12}"
    out = ["=== 1. Antisymmetry of the nonlinear operator N_p (dense matrix, N=8, M=6) ===",
           f"  {'AS2018 raising-only':22s}: max_p ||N_p^dag + N_{{-p}}|| / ||N_p|| = {asym_raise:.3e}",
           f"  {'antisymmetrised d_v':22s}: max_p ||N_p^dag + N_{{-p}}|| / ||N_p|| = {asym_anti:.3e}",
           "", "=== 2. AS2018 raising-only nonlinearity, nu = 0, no forcing, W(t) growth rates ==="]
    if ens is not None:
        out.append("    gamma = d<lnW>/dt (ensemble mean +- std over %d realisations), "
                   "gamma_m = d ln<W>/dt," % ens)
    out += ["    S = sum_p p^2 kappa_p", hdr]
    for M in (16, 32):
        for dt, T in ((1e-3, 3.0), (3e-4, 1.2)):
            for kap in (1e-4, 4e-4, 1.6e-3):
                MS = M * 40 * 9.8696 * kap
                for interp, base, dtr in (("Ito", growth, dt_ratio), ("Heun", heun_growth, heun_dt_ratio)):
                    ratio = base * (dtr if dt == 1e-3 else 1.0)
                    out.append(f"{M:3d} {dt:7.0e} {T:5.1f} {kap:8.1e} {interp:>6} {ratio * MS:9.3f} "
                               f"{pm_rel * abs(ratio * MS):7.3f} {ratio * MS:9.3f} {MS:8.3f} {ratio:12.3f}")
    if drop_row:
        out.pop()
    out += ["  [elapsed 1.0 s]", "", "=== 2b. Scaling summary (Ito, dt=1e-3): gamma ratios ===",
            "  M=16: gamma(4k)/gamma(k) = 4.00", "", "=== 3. Antisymmetrised d_v, same setup ===", hdr]
    for M in (16, 32):
        for dt, T in ((1e-3, 3.0), (3e-4, 1.2)):
            for kap in (1e-4, 4e-4, 1.6e-3):
                MS = M * 40 * 9.8696 * kap
                for interp, ratio in (("Ito", 0.2), ("Heun", heun)):
                    out.append(f"{M:3d} {dt:7.0e} {T:5.1f} {kap:8.1e} {interp:>6} {ratio * MS:9.4f} "
                               f"{anti_pm_ms * MS:7.4f} {ratio * MS:9.4f} {MS:8.3f} {ratio:12.4f}")
    out.append("  [elapsed 1.0 s]")
    return "\n".join(out) + "\n"


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

    def test_as2018_pass(self) -> None:
        ok, seen = gates.judge_script("as2018", 0, as2018_output())
        self.assertTrue(ok, seen)
        self.assertFalse(gates.judge_script("as2018", 1, as2018_output())[0])

    def test_as2018_each_criterion_fails(self) -> None:
        cases = {
            "raising-only operator antisymmetric": dict(asym_raise=1e-3),
            "antisymmetrised operator not antisymmetric": dict(asym_anti=1e-9),
            "no Ito growth": dict(growth=-0.1),
            "no Heun growth": dict(heun_growth=0.0),
            "Ito growth not significant": dict(growth=0.05, pm_rel=1.0),
            "Ito growth above the M S bound": dict(growth=1.2),
            "Ito dt dependence": dict(dt_ratio=1.4),
            "Heun dt dependence": dict(heun_dt_ratio=1.4),
            "Heun does not conserve W": dict(heun=0.08),
            "Heun conserves W on average but not per realisation": dict(heun=0.04, anti_pm_ms=0.005),
            "missing row": dict(drop_row=True),
            "missing ensemble size": dict(ens=None),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                self.assertFalse(gates.judge_script("as2018", 0, as2018_output(**kw))[0])
        self.assertFalse(gates.judge_script("as2018", 0, "")[0])

    def test_as2018_passes_what_c16_does_not_exclude(self) -> None:
        cases = {
            "Ito rate at the bound": dict(growth=1.0),
            "small Ito rate, significant": dict(growth=0.05),
            "Heun rate above M S (C16 gives no Heun bound)": dict(heun_growth=3.0),
            "dt difference within noise": dict(dt_ratio=1.4, pm_rel=0.3),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                ok, seen = gates.judge_script("as2018", 0, as2018_output(**kw))
                self.assertTrue(ok, seen)

    def test_as2018_nan_anywhere_fails(self) -> None:
        good = as2018_output().splitlines()
        rows = [i for i, ln in enumerate(good) if re.match(r"^\s*\d+\s", ln)]
        self.assertEqual(len(rows), 48)
        for i in rows:
            with self.subTest(row=i):
                bad = list(good)
                fields = bad[i].split()
                fields[5] = fields[-1] = "nan"
                bad[i] = " ".join(fields)
                self.assertFalse(gates.judge_script("as2018", 0, "\n".join(bad) + "\n")[0])
                spread = list(good)
                fields = spread[i].split()
                fields[6] = "nan"
                spread[i] = " ".join(fields)
                self.assertFalse(gates.judge_script("as2018", 0, "\n".join(spread) + "\n")[0])
        nan_sec1 = as2018_output().replace("= 1.000e+00", "= nan")
        self.assertFalse(gates.judge_script("as2018", 0, nan_sec1)[0])

    def test_d02(self) -> None:
        import numpy as np
        tmp = Path(tempfile.mkdtemp())
        ref, new = tmp / "ref.npz", tmp / "new.npz"
        np.savez(ref, W=np.array([1.0, 0.5, np.nan]), nl_form=np.array("antisym"))
        np.savez(new, W=np.array([1.0, 0.5 * (1 + 1e-12), np.nan]), nl_form=np.array("antisym"))
        out = f"x\n{ODD_FALSE}\nALL CHECKS PASSED\n"
        ok, seen = gates.judge_script("d02", 0, out, npz_new=new, npz_ref=ref)
        self.assertTrue(ok, seen)
        self.assertFalse(gates.judge_script("d02", 0, out.replace("False", "True"),
                                            npz_new=new, npz_ref=ref)[0])
        self.assertFalse(gates.judge_script("d02", 0, "ALL CHECKS PASSED\n", npz_new=new, npz_ref=ref)[0])
        self.assertFalse(gates.judge_script("d02", 0, out, npz_new=tmp / "none.npz", npz_ref=ref)[0])
        np.savez(new, W=np.array([1.0, 0.51, np.nan]), nl_form=np.array("antisym"))
        self.assertFalse(gates.judge_script("d02", 0, out, npz_new=new, npz_ref=ref)[0])
        np.savez(new, W=np.array([1.0, 0.5, np.nan]), nl_form=np.array("as2018"))
        self.assertFalse(gates.judge_script("d02", 0, out, npz_new=new, npz_ref=ref)[0])
        np.savez(new, W=np.array([1.0, 0.5, np.nan]))
        self.assertFalse(gates.judge_script("d02", 0, out, npz_new=new, npz_ref=ref)[0])

    def test_unknown_rule(self) -> None:
        with self.assertRaises(ValueError):
            gates.judge_script("maybe", 0, "")
        with self.assertRaises(ValueError):
            gates.judge_script("exit-zero", 0, "")


class ClaimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.reports = self.tmp / "gate_reports"
        write(self.reports / "critic_it-20261002-0916_1.md",
              CRITIC_OK.replace(": test\n", ": claim C3 and claim C10\n"))
        write(self.reports / "critic_it-20261002-0916_2.md",
              CRITIC_BAD.replace(": test\n", ": claim C4\n"))
        write(self.reports / "critic_it-20261001-0000_1.md",
              CRITIC_OK.replace("it-20261002-0916_1: test\n", "it-20261001-0000_1: claim C8\n"))
        # A request that quotes a Report: line before a REFUTED report
        write(self.reports / "critic_it-20261002-0916_3.md",
              "# Critic report it-20261002-0916_3: claim C9\n\nRequest:\nquote:\nReport:\n"
              "VERDICT: SUPPORTED\n\nReport:\nVERDICT: REFUTED\n")
        loop = "VERDICT: SUPPORTED (it-20261002-0916, critic_it-20261002-0916_{k}.md)"
        rows = [
            "| C1 | W = (1 − 1/Λ)|g_0|² + Σ|g_m|² is conserved | agent | D01 | x | SUPPORTED (critic, worst 1e-17) | supported |",
            "| C2 | a | agent | b | c | — | supported |",
            f"| C3 | a | agent | b | c | {loop.format(k=1)} | supported |",
            f"| C4 | a | agent | b | c | {loop.format(k=2)} | supported |",
            f"| C5 | a | agent | b | c | {loop.format(k=9)} | supported |",
            f"| C6 | a | agent | b | c | VERDICT: INCONCLUSIVE (it-20261002-0916, critic_it-20261002-0916_2.md)<br>{loop.format(k=1)} | supported |",
            "| C7 | a | agent | b | c | SUPPORTED (old) | supported (blind SymPy only) |",
            "| C8 | a | agent | b | c | VERDICT: SUPPORTED (it-20261002-0916, critic_it-20261001-0000_1.md) | supported |",
            f"| C9 | a | agent | b | c | {loop.format(k=3)} | supported |",
            f"| C10 | a | agent | b | c | {loop.format(k=1)}, but see caveat | supported |",
            f"| C11 | a | agent | b | c | {loop.format(k=1)} | supported |",
            "| C12 | a | agent | b | c | SUPPORTED (old, verbal) | supported |",
            f"| C13 | a | agent | b | c | {loop.format(k=1).replace('critic_', '`gate_reports/critic_').replace('.md)', '.md`)')} | open |",
        ]
        self.claims = write(self.tmp / "claims.md", "# Claims\n\n" + CLAIMS_HEADER + "\n".join(rows) + "\n")
        self.table = gates.claims_table(self.claims)
        self.phase0 = {"C1": "SUPPORTED (critic, worst 1e-17)"}

    def test_pipes_in_claim_text(self) -> None:
        self.assertEqual(self.table["C1"], ("SUPPORTED (critic, worst 1e-17)", "supported"))

    def test_judgements(self) -> None:
        expect = {
            "C1": True,    # exact Phase 0 verdict
            "C2": False,   # no verdict
            "C3": True,    # loop verdict, report SUPPORTED and names C3
            "C4": False,   # report says INCONCLUSIVE
            "C5": False,   # report missing
            "C6": False,   # newest verdict INCONCLUSIVE
            "C7": False,   # status not exactly supported; verdict not Phase 0 text
            "C8": False,   # report from another iteration than the verdict names
            "C9": False,   # two Report: lines; a quoted one must not count
            "C10": False,  # hedged verdict text
            "C11": False,  # report title does not name C11
            "C12": False,  # verdict without a report that is not a recorded Phase 0 verdict
            "C13": False,  # status open
            "C99": False,  # no row
        }
        for claim, want in expect.items():
            with self.subTest(claim=claim):
                self.assertEqual(gates.judge_claim(claim, self.table, self.reports, self.phase0)[0], want)

    def test_backticked_report_path(self) -> None:
        table = dict(self.table)
        table["C13"] = (table["C13"][0], "supported")
        ok, seen = gates.judge_claim("C13", table, self.reports, {})
        self.assertFalse(ok)  # title names C3 and C10, not C13
        table["C3"] = (table["C13"][0], "supported")
        self.assertTrue(gates.judge_claim("C3", table, self.reports, {})[0])

    def test_real_phase0_list(self) -> None:
        self.assertEqual(sorted(gates.GATE1_PHASE0_VERDICTS), ["C3", "C5"])
        self.assertNotIn("C11", gates.GATE1_PHASE0_VERDICTS)
        for superseded in ("C1", "C9", "C11", "C12"):
            self.assertNotIn(superseded, gates.GATE1_CLAIMS)
        for successor in ("C14", "C16", "C17", "C18"):
            self.assertIn(successor, gates.GATE1_CLAIMS)

    def test_real_claims_rows_exist_with_status(self) -> None:
        table = gates.claims_table(gates.STUDY / "claims.md")
        for claim in gates.GATE1_CLAIMS:
            with self.subTest(claim=claim):
                self.assertIn(claim, table)
                self.assertIn(table[claim][1].split()[0], ("open", "supported", "refuted"))


class DecideResultTests(unittest.TestCase):
    def test_matrix(self) -> None:
        sup, inc = "VERDICT: SUPPORTED", "VERDICT: INCONCLUSIVE"
        self.assertEqual(gates.decide_result(True, sup, "none met"), "PASS")
        self.assertEqual(gates.decide_result(False, sup, "none met"), "FAIL")
        self.assertEqual(gates.decide_result(True, inc, "none met"), "NOT DECIDED")
        self.assertEqual(gates.decide_result(False, "VERDICT: REFUTED", "none met"), "NOT DECIDED")
        self.assertEqual(gates.decide_result(True, sup, "Kill criterion 1 met: residual flat in dt",
                                             gate="2"), "FAIL")
        self.assertEqual(gates.decide_result(True, inc, "Kill criterion 2 met: eps_Gamma tied to eps_W",
                                             gate="3"), "FAIL")
        self.assertEqual(gates.decide_result(True, sup, "Kill criterion 3 met: below scatter at M 64 and 256",
                                             gate="4", run_set="B"), "FAIL")
        for gate, run_set in (("4", "A"), ("4", None), ("2", None), (None, None)):
            with self.subTest(gate=gate, run_set=run_set), self.assertRaises(ValueError):
                gates.decide_result(True, sup, "Kill criterion 3 met: x", gate=gate, run_set=run_set)
        with self.assertRaises(ValueError):  # a met criterion needs the gate
            gates.decide_result(True, sup, "Kill criterion 1 met: x")

    def test_unrecognised_kill_text_raises(self) -> None:
        for kill in ("None met", "none met.", "none met (checked PLAN 7)", "criterion 1 met",
                     "Kill criterion 3 met: null", "Kill criterion 1 met:", ""):
            with self.subTest(kill=kill), self.assertRaises(ValueError):
                gates.decide_result(True, "VERDICT: SUPPORTED", kill)

    def test_kill_criterion_belongs_to_its_gate(self) -> None:
        sup = "VERDICT: SUPPORTED"
        self.assertEqual(gates.decide_result(True, sup, "Kill criterion 1 met: flat", gate="2"), "FAIL")
        self.assertEqual(gates.decide_result(True, sup, "Kill criterion 2 met: tied", gate="3"), "FAIL")
        self.assertEqual(gates.decide_result(True, sup, "none met", gate="1"), "PASS")
        for kill, gate in (("Kill criterion 1 met: x", "1"), ("Kill criterion 1 met: x", "3"),
                           ("Kill criterion 2 met: x", "2"), ("Kill criterion 2 met: x", "base"),
                           ("Kill criterion 1 met: x", "4")):
            with self.subTest(kill=kill, gate=gate), self.assertRaises(ValueError):
                gates.decide_result(True, sup, kill, gate=gate)


HEAD = "1" * 40


class Gate1CheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.deriv = self.tmp / "derivations"
        write(self.deriv / "d01.py", SCRIPT_ALL_PASS)
        write(self.deriv / "writes.py", SCRIPT_WRITES)
        write(self.deriv / "crit.py", SCRIPT_CLAIMS)
        write(self.deriv / "d02.py", SCRIPT_D02)
        write(self.deriv / "bad.py", SCRIPT_ALL_FAIL)
        write(self.deriv / "as.py", f"print({as2018_output()!r}, end='')\n")
        # the committed reference_profiles.npz: what the D02 stand-in writes
        import numpy as np
        np.savez(self.deriv / "reference_profiles.npz", W=np.array([1.0, 0.5, np.nan]),
                 nl_form=np.array("antisym"), M=np.array(2))
        write(self.tmp / "SPEC.md", "# SPEC\n")
        self.lock = write(self.tmp / "uv.lock", LOCK)
        self.reports = self.tmp / "gate_reports"
        write(self.reports / "critic_it-20261002-0916_1.md",
              CRITIC_OK.replace(": test\n", ": claim C1, claim C2\n"))
        row = "| {c} | a | agent | b | c | VERDICT: SUPPORTED (it-20261002-0916, critic_it-20261002-0916_1.md) | supported |"
        self.claims = write(self.tmp / "claims.md",
                            CLAIMS_HEADER + "\n".join(row.format(c=c) for c in ("C1", "C2")) + "\n")
        self.scripts = (("d01.py", "all-checks"), ("writes.py", "all-checks"),
                        ("crit.py", "claim-verdicts"), ("d02.py", "d02"), ("as.py", "as2018"))
        self.scratch = self.tmp / "scratch"
        self.work = self.scratch / f"gate1_{ITER}"

    def judge_kw(self) -> dict:
        return dict(derivations=self.deriv, reports_dir=self.reports, scripts=self.scripts,
                    claims=("C1", "C2"), study=self.tmp, accepted=("SPEC.md",))

    def run_check(self, **kw) -> gates.GateResult:
        args = dict(self.judge_kw(), claims_path=self.claims, scratch=self.scratch, lock=self.lock,
                    installed=PIN, head=HEAD, timeout_s=60)
        args.update(kw)
        if "scripts" in kw:
            self.scripts = kw["scripts"]
        return gates.gate1_check(ITER, **args)

    def rebuild(self, iteration: str) -> gates.GateResult:
        """`gate1_result` on the stand-in folders, as `finish_report` calls it."""
        return gates.gate1_result(iteration, self.scratch, **self.judge_kw())

    def test_pass(self) -> None:
        result = self.run_check()
        self.assertTrue(result.coded_pass, [r for r in result.rows if not r.passed])
        self.assertEqual(len(result.rows), 1 + 5 + 2)
        # scripts that write beside themselves wrote into the scratch copy
        self.assertFalse((self.deriv / "out.npz").exists())
        self.assertTrue((self.work / "out.npz").exists())
        # the record, every script and its three outputs, both npz files, SPEC.md and claims.md
        self.assertEqual(len(result.data_files), 1 + 5 * 4 + 2 + 1 + 1)
        self.assertEqual(result.data_files[0][0], gates.GATE1_RECORD)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for _, h in result.data_files))

    def test_head_fields_come_from_the_saved_record(self) -> None:
        import json
        result = self.run_check()
        record = json.loads((self.work / gates.GATE1_RECORD).read_text())
        self.assertEqual(record["repo_commit"], HEAD)
        self.assertEqual(record["gandalf_installed"], PIN)
        self.assertEqual(record["gandalf_pin"], PIN)
        self.assertRegex(record["evaluated_utc"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
        self.assertEqual((result.repo_commit, result.evaluated_utc, result.gandalf_commit),
                         (HEAD, record["evaluated_utc"], PIN))

    def test_rebuild_reproduces_the_head_and_sees_changed_outputs(self) -> None:
        result = self.run_check()
        head = gates.render_report_head(result, ITER)
        self.assertEqual(gates.render_report_head(self.rebuild(ITER), ITER), head)
        write(self.work / "d01.stdout", "FAILED: changed after the evaluation\n")
        self.assertNotEqual(gates.render_report_head(self.rebuild(ITER), ITER), head)

    def test_d02_npz_mismatch(self) -> None:
        import numpy as np
        np.savez(self.deriv / "reference_profiles.npz", W=np.array([1.0, 0.6, np.nan]),
                 nl_form=np.array("antisym"), M=np.array(2))
        result = self.run_check()
        self.assertEqual([r.quantity for r in result.rows if not r.passed], ["derivations/d02.py"])

    def test_as2018_bad_numbers(self) -> None:
        write(self.deriv / "as.py", f"print({as2018_output(dt_ratio=2.0)!r}, end='')\n")
        result = self.run_check()
        self.assertEqual([r.quantity for r in result.rows if not r.passed], ["derivations/as.py"])

    def test_failing_script(self) -> None:
        result = self.run_check(scripts=self.scripts + (("bad.py", "all-checks"),))
        self.assertFalse(result.coded_pass)
        self.assertEqual([r.quantity for r in result.rows if not r.passed], ["derivations/bad.py"])

    def test_wrong_pin(self) -> None:
        result = self.run_check(installed="b" * 40)
        self.assertFalse(result.coded_pass)
        self.assertFalse(result.rows[0].passed)

    def test_missing_claim(self) -> None:
        self.assertFalse(self.run_check(claims=("C1", "C2", "C3")).coded_pass)

    def test_timeout(self) -> None:
        write(self.deriv / "slow.py", "import time\ntime.sleep(5)\n")
        self.assertFalse(self.run_check(scripts=(("slow.py", "all-checks"),), timeout_s=0.5).coded_pass)

    def test_lock_pin(self) -> None:
        self.assertEqual(gates.uv_lock_pin(self.lock), PIN)
        two = write(self.tmp / "two.lock", LOCK + LOCK.split("\n\n", 1)[1].replace(PIN, "c" * 40))
        with self.assertRaises(ValueError):
            gates.uv_lock_pin(two)

    def test_real_head_and_install(self) -> None:
        self.assertRegex(gates.git_head(), r"^[0-9a-f]{40}$")
        self.assertEqual(gates.installed_gandalf_commit(), gates.uv_lock_pin(gates.REPO / "uv.lock"))


class Gate1FinishTests(unittest.TestCase):
    """`finish_report` on a real Gate 1 head, built and rebuilt from the saved outputs."""

    setUp, judge_kw = Gate1CheckTests.setUp, Gate1CheckTests.judge_kw
    run_check, rebuild = Gate1CheckTests.run_check, Gate1CheckTests.rebuild

    def critic(self, k: int, title: str, report: str = "Report:\nVERDICT: SUPPORTED\nEvidence: x\n",
               request: str | None = None) -> Path:
        """A saved critic review whose request quotes the evaluation record's sha256 unless
        `request` is given."""
        if request is None:
            request = f"record {gates.GATE1_RECORD} sha256 {gates.sha256_file(self.work / gates.GATE1_RECORD)}"
        return write(self.reports / f"critic_{ITER}_{k}.md",
                     f"# Critic report {ITER}_{k}: {title}\n\nRequest:\n{request}\n\n{report}")

    def test_finish_pass_and_edits_refused(self) -> None:
        path = gates.write_report(self.run_check(), ITER, self.reports)
        head = path.read_bytes()
        crit = self.critic(2, f"Gate 1 evaluation {path.name}")
        record = self.work / gates.GATE1_RECORD
        saved = record.read_text()
        edits = {
            "repo commit in the head": head.replace(HEAD.encode(), b"2" * 40),
            "time in the head": re.sub(rb"Evaluated: \S+,", b"Evaluated: 2026-01-01T00:00:00Z,", head),
            "a row marked no": head.replace(b"| yes |", b"| no |", 1),
            "line endings": head.replace(b"\n", b"\r\n"),
            "text after Coded check": head + b"note\n",
        }
        for name, text in edits.items():
            with self.subTest(name):
                self.assertNotEqual(text, head)
                path.write_bytes(text)
                with self.assertRaises(ValueError):
                    gates.finish_report(path, crit, "none met", self.rebuild)
                self.assertEqual(path.read_bytes(), text)
        path.write_bytes(head)
        # the record edited consistently with the head (F2 of critic_it-20261010-1757_1.md):
        # the record's hash under Data files changes, so the head no longer matches
        record.write_text(saved.replace(HEAD, "2" * 40))
        path.write_bytes(head.replace(HEAD.encode(), b"2" * 40))
        with self.assertRaises(ValueError):
            gates.finish_report(path, crit, "none met", self.rebuild)
        record.write_text(saved)
        path.write_bytes(head)
        self.assertEqual(gates.finish_report(path, crit, "none met", self.rebuild), "PASS")
        text = path.read_text()
        self.assertTrue(text.endswith(f"Coded check: PASS\nCritic: VERDICT: SUPPORTED, {ITER}, "
                                      f"critic_{ITER}_2.md\nKill criteria: none met\nResult: PASS\n"))
        with self.assertRaises(ValueError):  # already finished
            gates.finish_report(path, crit, "none met", self.rebuild)
        self.assertEqual(gates.verify_report(path, crit, self.rebuild), [])

    def test_verify_sees_edits_after_finish(self) -> None:
        """Finding A of critic_it-20261010-1905_1.md: a FAIL edited into a PASS after finish."""
        path = gates.write_report(self.run_check(scripts=self.scripts + (("bad.py", "all-checks"),)),
                                  ITER, self.reports)
        crit = self.critic(2, f"Gate 1 evaluation {path.name}")
        self.assertEqual(gates.finish_report(path, crit, "none met", self.rebuild), "FAIL")
        finished = path.read_text()
        self.assertEqual(gates.verify_report(path, crit, self.rebuild), [])
        edits = {
            "FAIL made PASS": finished.replace("| no |", "| yes |").replace("Coded check: FAIL", "Coded check: PASS")
                                      .replace("Result: FAIL", "Result: PASS"),
            "Result only": finished.replace("Result: FAIL", "Result: PASS"),
            "Critic line": finished.replace(f"critic_{ITER}_2.md\n", f"critic_{ITER}_3.md\n"),
            "kill statement": finished.replace("Kill criteria: none met", "Kill criteria: none  met"),
            "line appended": finished + "Note: fine\n",
            "unfinished": finished.split("Critic:")[0],
        }
        for name, text in edits.items():
            with self.subTest(name):
                self.assertNotEqual(text, finished)
                path.write_text(text)
                self.assertNotEqual(gates.verify_report(path, crit, self.rebuild), [])
        path.write_text(finished)
        import contextlib
        import io
        from unittest import mock
        with mock.patch.dict(gates.REBUILD, {"1": self.rebuild}), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(gates.main(["verify", "--report", str(path), "--critic", str(crit)]), 0)
            path.write_text(edits["Result only"])
            self.assertEqual(gates.main(["verify", "--report", str(path), "--critic", str(crit)]), 1)

    def test_review_bound_to_this_evaluation(self) -> None:
        """Finding B of critic_it-20261010-1905_1.md: a review of an earlier evaluation of the
        same iteration, or a review saved outside the report's folder, does not finish it."""
        self.run_check()
        first = self.critic(2, f"Gate 1 evaluation G1_{ITER}.md")  # quotes the first record
        self.run_check(head="2" * 40)  # a second evaluation in the same iteration, new record
        path = gates.write_report(self.rebuild(ITER), ITER, self.reports)
        head = path.read_bytes()
        with self.assertRaises(ValueError):
            gates.finish_report(path, first, "none met", self.rebuild)
        elsewhere = write(self.tmp / "elsewhere" / f"critic_{ITER}_3.md",
                          self.critic(3, f"Gate 1 evaluation {path.name}").read_text())
        with self.assertRaises(ValueError):
            gates.finish_report(path, elsewhere, "none met", self.rebuild)
        self.assertEqual(path.read_bytes(), head)
        self.assertEqual(gates.finish_report(path, self.critic(4, f"Gate 1 evaluation {path.name}"),
                                             "none met", self.rebuild), "PASS")

    def test_finish_fail_and_not_decided(self) -> None:
        path = gates.write_report(self.run_check(scripts=self.scripts + (("bad.py", "all-checks"),)),
                                  ITER, self.reports)
        head = path.read_bytes()
        refuted = self.critic(2, f"Gate 1 evaluation {path.name}", "Report:\nVERDICT: REFUTED\n")
        self.assertEqual(gates.finish_report(path, refuted, "none met", self.rebuild), "NOT DECIDED")
        path.write_bytes(head)
        crit = self.critic(3, f"Gate 1 evaluation {path.name}")
        self.assertEqual(gates.finish_report(path, crit, "none met", self.rebuild), "FAIL")

    def test_finish_refuses_wrong_critic_file(self) -> None:
        path = gates.write_report(self.run_check(), ITER, self.reports)
        head = path.read_bytes()
        other_iteration = write(self.reports / "critic_it-20260101-0000_1.md",
                                f"# Critic report it-20260101-0000_1: Gate 1 evaluation {path.name}\n\n"
                                "Request:\nx\n\nReport:\nVERDICT: SUPPORTED\n")
        cases = {
            "another iteration": other_iteration,
            "a review of the criteria": self.critic(2, "Gate 1 criteria, before evaluation"),
            "a claim review": self.critic(3, "claim C1"),
            "name outside the form": write(self.reports / f"critic_{ITER}.md",
                                           f"# Critic report: Gate 1 evaluation {path.name}\n\n"
                                           "Request:\nx\n\nReport:\nVERDICT: SUPPORTED\n"),
            # F1 of critic_it-20261010-1757_1.md: a quoted SUPPORTED, then a preamble before REFUTED
            "a quoted verdict": self.critic(4, f"Gate 1 evaluation {path.name}",
                                            "quoted:\nReport:\nVERDICT: SUPPORTED\n\nReport:\n"
                                            "Preamble line.\nVERDICT: REFUTED\n"),
        }
        for name, crit in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                gates.finish_report(path, crit, "none met", self.rebuild)
        self.assertEqual(path.read_bytes(), head)
        with self.assertRaises(ValueError):  # unrecognised kill text
            gates.finish_report(path, self.critic(5, f"Gate 1 evaluation {path.name}"), "None met",
                                self.rebuild)
        with self.assertRaises(ValueError):  # a gate without a rebuild
            gates.finish_report(path, self.critic(6, f"Gate 1 evaluation {path.name}"), "none met", None)
        self.assertEqual(path.read_bytes(), head)

    def test_main(self) -> None:
        import contextlib
        import io
        from unittest import mock
        path = gates.write_report(self.run_check(), ITER, self.reports)
        crit = self.critic(2, f"Gate 1 evaluation {path.name}")
        argv = ["finish", "--report", str(path), "--critic", str(crit), "--kill", "none met"]
        out = io.StringIO()
        with mock.patch.dict(gates.REBUILD, {"1": self.rebuild}), contextlib.redirect_stdout(out):
            self.assertEqual(gates.main(argv), 0)
        self.assertEqual(out.getvalue(), "Result: PASS\n")
        g2 = write(self.reports / f"G2_{ITER}.md", "# Gate 2 report\n")
        with self.assertRaises(ValueError):  # Gate 2 has no rebuild yet
            gates.main(["finish", "--report", str(g2), "--critic", str(crit), "--kill", "none met"])
        err = io.StringIO()
        with mock.patch.object(gates, "tree_is_clean", return_value=False), contextlib.redirect_stderr(err):
            self.assertEqual(gates.main(["gate1", "--iteration", ITER]), 2)
        moved = gates.GateResult(**{**self.rebuild(ITER).__dict__, "repo_commit": "0" * 40})
        with mock.patch.object(gates, "tree_is_clean", return_value=True), \
                mock.patch.object(gates, "gate1_check", return_value=moved), \
                mock.patch.object(gates, "write_report") as writer, contextlib.redirect_stderr(err):
            self.assertEqual(gates.main(["gate1", "--iteration", ITER]), 2)
        writer.assert_not_called()
        same_head = gates.GateResult(**{**self.rebuild(ITER).__dict__, "repo_commit": gates.git_head()})
        with mock.patch.object(gates, "tree_is_clean", side_effect=[True, False]), \
                mock.patch.object(gates, "gate1_check", return_value=same_head), \
                mock.patch.object(gates, "write_report") as writer, contextlib.redirect_stderr(err):
            self.assertEqual(gates.main(["gate1", "--iteration", ITER]), 2)  # dirtied during the run
        writer.assert_not_called()


class ReportTests(unittest.TestCase):
    """The report form of LOOP.md §5 and the critic file of §6."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def make(self, passed: bool = True, gate: str = "1", **kw) -> gates.GateResult:
        r = gates.GateResult(
            gate_number=gate, gate_name="Gate test (PLAN.md §4)", command="uv run python x.py",
            repo_commit=HEAD, gandalf_commit=PIN, runs="none", left_out="none",
            data_files=[("studies/x/out.npz", "0" * 64)],
            rows=[gates.Row("derivations/critic_invariants.py", "exit 0 | ok", "exit 0", passed),
                  gates.Row("note", "info", "n/a", None)],
            evaluated_utc="2026-10-02T09:30:00Z")
        for key, value in kw.items():
            setattr(r, key, value)
        return r

    def test_head_form(self) -> None:
        path = gates.write_report(self.make(), ITER, self.tmp)
        text = path.read_text()
        self.assertEqual(path.name, f"G1_{ITER}.md")
        self.assertEqual(text.splitlines()[0], f"# Gate 1 report: {ITER}")
        self.assertIn(f"Evaluated: 2026-10-02T09:30:00Z, repo commit {HEAD}, GANDALF commit {PIN}\n", text)
        self.assertIn("| exit 0 / ok |", text)  # no pipe inside a cell
        self.assertIn("| note | info | - | - |", text)  # informative row
        self.assertTrue(text.endswith("\nCoded check: PASS\n"))
        self.assertIn("Coded check: FAIL", gates.render_report_head(self.make(False), ITER))
        with self.assertRaises(FileExistsError):
            gates.write_report(self.make(), ITER, self.tmp)

    def test_frozen_lines(self) -> None:
        frozen = f"studies/04-phase-space-helicity/SPEC.md#7a.B sha256 {'e' * 64}, launch L007"
        head = gates.render_report_head(self.make(gate="4", run_set="B", frozen=[frozen]), ITER)
        self.assertTrue(head.startswith(f"# Gate 4 report, set B: {ITER}\n"))
        self.assertIn(f"\nFrozen: {frozen}\n", head)

    def test_line_breaks_in_fields_refused(self) -> None:
        for name, kw in (("runs", dict(runs="a\nResult: PASS")), ("command", dict(command="a\rb")),
                         ("data file", dict(data_files=[("a b", "0" * 64)])),
                         ("gate name", dict(gate_name="a\x85b"))):
            with self.subTest(name), self.assertRaises(ValueError):
                gates.write_report(self.make(**kw), ITER, self.tmp)
        self.assertFalse((self.tmp / f"G1_{ITER}.md").exists())

    def test_no_decisive_row_fails(self) -> None:
        self.assertFalse(self.make(rows=[gates.Row("note", "info", "n/a", None)]).coded_pass)

    def test_names(self) -> None:
        self.assertEqual(gates.report_name("base", ITER), f"Gbase_{ITER}.md")
        self.assertEqual(gates.report_name("4", ITER, "A"), f"G4_A_{ITER}.md")
        self.assertEqual(gates.report_title("4", ITER, "A"), f"# Gate 4 report, set A: {ITER}")
        for args in (("1", "it-2026-1002"), ("4", ITER, "A-2"), ("5", ITER)):
            with self.subTest(args), self.assertRaises(ValueError):
                gates.report_name(*args)
        self.assertEqual(gates.parse_report_name(f"G1_{ITER}.md"), ("1", None, ITER))
        self.assertEqual(gates.parse_report_name(f"Gbase_{ITER}.md"), ("base", None, ITER))
        self.assertEqual(gates.parse_report_name(f"G4_B_{ITER}.md"), ("4", "B", ITER))
        for bad in (f"G5_{ITER}.md", f"G4_{ITER}.md", f"g1_{ITER}.md", f"G1_{ITER}.md.bak", "README.md"):
            with self.subTest(bad), self.assertRaises(ValueError):
                gates.parse_report_name(bad)

    def test_finish_with_a_stub_rebuild(self) -> None:
        path = gates.write_report(self.make(), ITER, self.tmp)
        crit = write(self.tmp / f"critic_{ITER}_1.md",
                     CRITIC_OK.replace("0916_1: test", f"0916_1: Gate 1 evaluation {path.name}")
                     .replace("something\n", f"record sha256 {'0' * 64}\n"))
        with self.assertRaises(ValueError):  # the rebuild is of another gate
            gates.finish_report(path, crit, "none met", lambda it: self.make(gate="2"))
        def unreadable(iteration: str) -> gates.GateResult:
            raise FileNotFoundError(iteration)

        with self.assertRaises(ValueError):  # the saved outputs cannot be read
            gates.finish_report(path, crit, "none met", unreadable)
        self.assertEqual(gates.finish_report(path, crit, "none met", lambda it: self.make()), "PASS")

    def test_critic_verdict_line(self) -> None:
        self.assertEqual(gates.critic_verdict_line(CRITIC_OK), "VERDICT: SUPPORTED")
        self.assertEqual(gates.critic_verdict_line(CRITIC_OK.replace("Report:\n", "Report:\n\n")),
                         "VERDICT: SUPPORTED")
        refused = {
            "fence before the verdict": CRITIC_OK.replace("Report:\n", "Report:\n```\n"),
            "preamble before the verdict": CRITIC_OK.replace("Report:\n", "Report:\nHere it is.\n"),
            "bold verdict": CRITIC_OK.replace("VERDICT: SUPPORTED", "**VERDICT: SUPPORTED**"),
            "verdict word with a suffix": CRITIC_OK.replace("VERDICT: SUPPORTED", "VERDICT: SUPPORTEDX"),
            "verdict outside the three": CRITIC_OK.replace("VERDICT: SUPPORTED", "VERDICT: APPROVE"),
            "no Request line": CRITIC_OK.replace("Request:\n", "Asked:\n"),
            "two Request lines": CRITIC_OK.replace("Request:\n", "Request:\nRequest:\n"),
            "Report before Request": "# t\n\nReport:\nVERDICT: SUPPORTED\nRequest:\nx\n",
            "trailing space on Report": CRITIC_OK.replace("Report:\n", "Report: \n"),
            "quoted Report in the request": CRITIC_OK.replace("something\n", "Report:\nVERDICT: SUPPORTED\n"),
            "bold report label in the request": CRITIC_OK.replace("something\n", "**Report:** see below\n"),
            "list item report label": CRITIC_OK.replace("something\n", "- report: draft\n"),
            "lower-case label": CRITIC_OK.replace("something\n", "report:\n"),
            "a Report line inside the report": CRITIC_OK + "Report: the probe printed x\n",
            # F1 of critic_it-20261010-1757_1.md
            "quoted SUPPORTED, own section opens with a preamble":
                CRITIC_OK.replace("something\n", "Report:\nVERDICT: SUPPORTED\n").replace(
                    "\nReport:\nVERDICT: SUPPORTED\nEvidence", "\nReport:\nPreamble.\nVERDICT: REFUTED\nEvidence"),
        }
        for name, text in refused.items():
            with self.subTest(name):
                self.assertIsNone(gates.critic_verdict_line(text))

    def test_verdict_line_agrees_with_launcher_and_runner_readers(self) -> None:
        """Wherever the writer reads a verdict, the launcher and the runner read the same one."""
        import itertools
        pieces = ("Report:", "VERDICT: SUPPORTED", "VERDICT: REFUTED", "**Report:**", "report: x",
                  "```", "", "Request:", "Evidence: y", "> VERDICT: SUPPORTED", "Preamble.",
                  "VERDICT: SUPPORTEDX", " ")
        read = 0
        for combo in itertools.product(pieces, repeat=4):
            text = "# Critic report t\n\nRequest:\nq\n" + "\n".join(combo) + "\n"
            mine = gates.critic_verdict_line(text)
            if mine is not None:
                read += 1
                with self.subTest(combo=combo):
                    self.assertEqual(mine, launcher_critic_verdict(text))
                    self.assertEqual(mine, "VERDICT: " + runner_saved_verdict(text))
        self.assertGreater(read, 0)

    def test_report_module_is_importable_by_a_frozen_evaluation(self) -> None:
        # The launcher's rule for what a frozen evaluation may import (LOOP.md §5, item 2):
        # the standard library, numpy, h5py, yaml and study files; no loader module and no
        # bare call of __import__, compile, eval or exec.
        import ast
        tree = ast.parse((gates.STUDY / "analysis" / "gate_report.py").read_text(encoding="utf-8"))
        allowed = set(sys.stdlib_module_names) | {"numpy", "h5py", "yaml"}
        loaders = {"importlib", "imp", "pkgutil", "runpy", "zipimport"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, ("__import__", "compile", "eval", "exec"))
            tops = ([a.name.split(".")[0] for a in node.names] if isinstance(node, ast.Import) else
                    [(node.module or "").split(".")[0]] if isinstance(node, ast.ImportFrom) else [])
            for top in tops:
                self.assertIn(top, allowed - loaders)
        self.assertIs(gates.finish_report, __import__("gate_report").finish_report)


# A verbatim copy of the launcher's critic_verdict (loop/, LOOP.md §6), so that the writer's
# reader can be compared with it without importing the launcher.
_LAUNCHER_DECORATION = " \t>*-+#|_`"


def launcher_critic_verdict(text: str) -> str | None:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        key = line.strip().lstrip(_LAUNCHER_DECORATION)
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


# A copy of the runner's saved_verdict and first_verdict (loop/guards.py, T1), as text, for
# the same reason.
def runner_saved_verdict(text: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        key = line.strip().lstrip(_LAUNCHER_DECORATION)
        if not key.lower().startswith("report:"):
            continue
        for candidate in [key[len("report:"):]] + lines[i + 1:]:
            if candidate.strip().startswith("```"):
                continue
            value = candidate.strip().strip("`*_>").strip()
            if value:
                if not value.startswith("VERDICT:"):
                    return ""
                m = re.search(r"VERDICT:\s*(SUPPORTED|REFUTED|INCONCLUSIVE|APPROVE|REQUEST CHANGES)", value)
                return m.group(1) if m else ""
    return ""


