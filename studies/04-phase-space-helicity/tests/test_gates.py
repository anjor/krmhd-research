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
        self.assertEqual(gates.decide_result(True, sup, "Kill criterion 1 met: residual flat in dt"), "FAIL")
        self.assertEqual(gates.decide_result(True, inc, "Kill criterion 2 met: eps_Gamma tied to eps_W"),
                         "FAIL")

    def test_unrecognised_kill_text_raises(self) -> None:
        for kill in ("None met", "none met.", "none met (checked PLAN 7)", "criterion 1 met",
                     "Kill criterion 3 met: null", "Kill criterion 1 met:", ""):
            with self.subTest(kill=kill), self.assertRaises(ValueError):
                gates.decide_result(True, "VERDICT: SUPPORTED", kill)


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

    def run_check(self, **kw) -> gates.GateResult:
        args = dict(derivations=self.deriv, claims_path=self.claims, reports_dir=self.reports,
                    scratch=self.tmp / "scratch", lock=self.lock, scripts=self.scripts,
                    claims=("C1", "C2"), installed=PIN, timeout_s=60, study=self.tmp,
                    accepted=("SPEC.md",))
        args.update(kw)
        return gates.gate1_check(ITER, **args)

    def test_pass(self) -> None:
        result = self.run_check()
        self.assertTrue(result.coded_pass, [r for r in result.rows if not r.passed])
        self.assertEqual(len(result.rows), 1 + 5 + 2)
        # scripts that write beside themselves wrote into the scratch copy
        self.assertFalse((self.deriv / "out.npz").exists())
        self.assertTrue((self.tmp / "scratch" / f"gate1_{ITER}" / "out.npz").exists())
        # every script and its two logs, both npz files, SPEC.md and claims.md are hashed
        self.assertEqual(len(result.data_files), 5 * 3 + 2 + 1 + 1)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for _, h in result.data_files))

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
        result = self.run_check(claims=("C1", "C2", "C3"))
        self.assertFalse(result.coded_pass)

    def test_timeout(self) -> None:
        write(self.deriv / "slow.py", "import time\ntime.sleep(5)\n")
        result = self.run_check(scripts=(("slow.py", "all-checks"),), timeout_s=0.5)
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
        gate1 = ": Gate 1 evaluation\n"
        self.critic_ok = write(self.tmp / "critic_it-20261002-0916_1.md", CRITIC_OK.replace(": test\n", gate1))
        self.critic_bad = write(self.tmp / "critic_it-20261002-0916_2.md",
                                CRITIC_BAD.replace("0916_1: test\n", "0916_2" + gate1))

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
        # a request that quotes "Report:" followed by SUPPORTED, before a REFUTED report
        quoted = CRITIC_OK.replace("something\n", "Report:\nVERDICT: SUPPORTED\n").replace(
            "Report:\nVERDICT: SUPPORTED\nEvidence", "Report:\nVERDICT: REFUTED\nEvidence")
        self.assertIsNone(gates.critic_verdict_line(quoted))
        self.assertIsNone(gates.critic_verdict_line(CRITIC_OK.replace("Report:\n", "Report: \n#")))

    def test_finish_refuses_critic_not_naming_gate(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        for name, text in (("critic_it-20261002-0916_3.md", CRITIC_OK.replace("0916_1: test", "0916_3: claim C1")),
                           ("critic_it-20261002-0916_4.md", CRITIC_OK.replace("0916_1: test", "0916_4: Gate 2")),
                           ("critic_it-20261002-0916_5.md", CRITIC_OK.replace(": test", ": Gate 1"))):
            with self.subTest(name), self.assertRaises(ValueError):
                gates.finish_report(path, write(self.tmp / name, text), "none met")
        self.assertNotIn("Result:", path.read_text())

    def test_gate_label(self) -> None:
        self.assertTrue(gates.gate_label_re("# Gate 1 report: x").search("review of gate_1 criteria"))
        self.assertFalse(gates.gate_label_re("# Gate 1 report: x").search("Gate 10"))
        self.assertTrue(gates.gate_label_re("# Gate base report: x").search("the base-state gate"))
        self.assertTrue(gates.gate_label_re("# Gate 4 report, set A: x").search("Gate 4, set A"))
        self.assertTrue(gates.gate_label_re("# Gate 4 report, set A: x").search("SPEC 7a.A for gate_4"))
        self.assertFalse(gates.gate_label_re("# Gate 4 report, set A: x").search("Gate 4 criteria"))
        self.assertFalse(gates.gate_label_re("# Gate 4 report, set A: x").search("Gate 4, set B"))
        self.assertFalse(gates.gate_label_re("# Gate 4 report, set A: x").search("set A, Gate 3"))
        with self.assertRaises(ValueError):
            gates.gate_label_re("# Gate 4 report: x")

    def test_finish_refuses_critic_of_other_iteration(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        other = write(self.tmp / "critic_it-20260101-0000_1.md",
                      CRITIC_OK.replace("it-20261002-0916_1: test", "it-20260101-0000_1: Gate 1"))
        with self.assertRaises(ValueError):
            gates.finish_report(path, other, "none met")
        with self.assertRaises(ValueError):
            gates.finish_report(path, self.critic_ok, "None met")
        self.assertNotIn("Result:", path.read_text())


if __name__ == "__main__":
    unittest.main()
