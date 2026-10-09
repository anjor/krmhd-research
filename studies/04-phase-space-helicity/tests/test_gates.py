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


def same(stated: gates.GateResult, iteration: str) -> gates.GateResult:
    """A re-judge that rebuilds exactly what the head states, for tests of the machinery
    that are not about re-judging."""
    return stated


def finish(*args, **kw) -> str:
    """finish_report with the identity re-judge unless a test gives its own."""
    kw.setdefault("rejudge", same)
    return gates.finish_report(*args, **kw)


def head_sha(result: gates.GateResult) -> str:
    """sha256 of the head the writer renders for `result`."""
    import hashlib
    return hashlib.sha256(gates.render_report_head(result, ITER).encode("utf-8")).hexdigest()


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
                    claims=("C1", "C2"), installed=PIN, intact=(True, "test"), timeout_s=60, study=self.tmp,
                    accepted=("SPEC.md",))
        args.update(kw)
        return gates.gate1_check(ITER, **args)

    def test_pass(self) -> None:
        result = self.run_check()
        self.assertTrue(result.coded_pass, [r for r in result.rows if not r.passed])
        self.assertEqual(len(result.rows), 2 + 5 + 2)
        # scripts that write beside themselves wrote into the scratch copy
        self.assertFalse((self.deriv / "out.npz").exists())
        self.assertTrue((self.tmp / "scratch" / f"gate1_{ITER}" / "out.npz").exists())
        # every script and its two logs, both npz files, SPEC.md and claims.md are hashed
        self.assertEqual(len(result.data_files), 5 * 4 + 2 + 1 + 1)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", h) for _, h in result.data_files))

    def test_rejudge_rebuilds_and_catches_edits(self) -> None:
        result = self.run_check(scripts=self.scripts + (("bad.py", "all-checks"),))
        self.assertFalse(result.coded_pass)
        kw = dict(scratch=self.tmp / "scratch", lock=self.lock, install_info=(PIN, True, "test"),
                  derivations=self.deriv, reports_dir=self.reports,
                  scripts=self.scripts + (("bad.py", "all-checks"),), claims=("C1", "C2"),
                  study=self.tmp, accepted=("SPEC.md",))

        def rejudge(stated: gates.GateResult, iteration: str) -> gates.GateResult:
            return gates.gate1_rejudge(stated, iteration, **dict(kw))

        head = gates.render_report_head(result, ITER)
        self.assertEqual(gates.render_report_head(rejudge(result, ITER), ITER), head)
        # a consistent hand edit: the failing row marked yes and the coded check PASS
        edited = gates.GateResult(**{**result.__dict__, "rows": [
            gates.Row(r.quantity, r.value, r.threshold, True if r.passed is False else r.passed)
            for r in result.rows]})
        self.assertTrue(edited.coded_pass)
        self.assertNotEqual(gates.render_report_head(rejudge(edited, ITER), ITER),
                            gates.render_report_head(edited, ITER))
        # an output changed after the evaluation
        stdout = self.tmp / "scratch" / f"gate1_{ITER}" / "bad.stdout"
        stdout.write_text("ALL CHECKS PASSED\n")
        self.assertNotEqual(gates.render_report_head(rejudge(result, ITER), ITER), head)

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

    def test_installed_files_changed(self) -> None:
        result = self.run_check(intact=(False, "1 changed"))
        self.assertFalse(result.coded_pass)

    def test_real_install_intact(self) -> None:
        ok, seen = gates.installed_gandalf_intact()
        self.assertTrue(ok, seen)

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
    EVIDENCE = (f"repo commit {'1' * 40}, evaluated 2026-10-02T09:30:00Z; "
                f"studies/x/derivations/critic_invariants.py sha256 {'0' * 64}\n")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        gate1 = f": Gate 1 evaluation, G1_{ITER}.md\n"
        heads = f"head sha256 {head_sha(self.make(True))} or {head_sha(self.make(False))}\n"
        bound_ok = CRITIC_OK.replace("something\n", self.EVIDENCE + heads)
        self.critic_ok = write(self.tmp / "critic_it-20261002-0916_1.md", bound_ok.replace(": test\n", gate1))
        self.critic_bad = write(self.tmp / "critic_it-20261002-0916_2.md",
                                bound_ok.replace("VERDICT: SUPPORTED", "VERDICT: INCONCLUSIVE")
                                .replace("0916_1: test\n", "0916_2" + gate1))

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
        self.assertEqual(finish(path, self.critic_ok, "none met"), "PASS")
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
        self.assertEqual(finish(path, self.critic_bad, "none met"), "NOT DECIDED")
        other = self.tmp / "other"
        path2 = gates.write_report(self.make(False), ITER, other)
        self.assertIn("Coded check: FAIL", path2.read_text())
        self.assertIn("| no |", path2.read_text())
        critic2 = write(other / self.critic_ok.name, self.critic_ok.read_text())
        self.assertEqual(finish(path2, critic2, "none met"), "FAIL")

    def test_no_overwrite_no_refinish(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        with self.assertRaises(FileExistsError):
            gates.write_report(self.make(True), ITER, self.tmp)
        finish(path, self.critic_ok, "none met")
        with self.assertRaises(ValueError):
            finish(path, self.critic_ok, "none met")

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
        # a fence before the verdict: the launcher skips it, the writer refuses (safe direction)
        fenced = CRITIC_OK.replace("Report:\n", "Report:\n```\n")
        self.assertIsNone(gates.critic_verdict_line(fenced))
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
                           ("critic_it-20261002-0916_5.md", CRITIC_OK.replace(": test", ": Gate 1")),
                           # names the gate but not this report: a review of the criteria, say
                           ("critic_it-20261002-0916_6.md",
                            CRITIC_OK.replace("0916_1: test", "0916_6: Gate 1 criteria, before evaluation"))):
            with self.subTest(name), self.assertRaises(ValueError):
                finish(path, write(self.tmp / name, text), "none met")
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

    def test_finish_refuses_unbound_request(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        ok = self.critic_ok.read_text()
        cases = {
            "commit not quoted": ok.replace("1" * 40, "1" * 12),
            "data hash not quoted": ok.replace("0" * 64, "0" * 63 + "f"),
            "no Request line": ok.replace("Request:\n", "Asked:\n"),
            "two Request lines": ok.replace("Request:\n", "Request:\nRequest:\n"),
        }
        for k, (name, text) in enumerate(cases.items(), start=3):
            with self.subTest(name), self.assertRaises(ValueError):
                finish(path, write(self.tmp / f"critic_{ITER}_{k}.md",
                                                text.replace(f"{ITER}_1:", f"{ITER}_{k}:")), "none met")
        self.assertNotIn("Result:", path.read_text())

    def test_finish_refuses_report_without_full_commit(self) -> None:
        result = self.make(True)
        result.repo_commit = "unknown"
        with self.assertRaises(ValueError):  # the writer refuses it now, before any review
            gates.write_report(result, ITER, self.tmp)
        self.assertFalse((self.tmp / f"G1_{ITER}.md").exists())

    def test_finish_refuses_wrong_title_or_tail(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        head = path.read_text()
        for name, text in (("title of another gate", head.replace("# Gate 1 report", "# Gate 2 report")),
                           ("title with another iteration", head.replace(f"report: {ITER}", "report: it-20260101-0000")),
                           ("text after Coded check", head + "note\n"),
                           ("decorated Result line", head.replace("Left out: none", "**Result**: PASS")),
                           ("second Coded check", head.replace("Left out: none", "- coded check (dt fit): PASS"))):
            with self.subTest(name):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    finish(path, self.critic_ok, "none met")
                self.assertEqual(path.read_text(), text)
        bad_name = write(self.tmp / f"G1-{ITER}.md", head)
        with self.assertRaises(ValueError):
            finish(bad_name, self.critic_ok, "none met")

    def test_parse_report_name(self) -> None:
        self.assertEqual(gates.parse_report_name(f"G1_{ITER}.md"), ("1", None, ITER))
        self.assertEqual(gates.parse_report_name(f"Gbase_{ITER}.md"), ("base", None, ITER))
        self.assertEqual(gates.parse_report_name(f"G4_B_{ITER}.md"), ("4", "B", ITER))
        for bad in (f"G5_{ITER}.md", f"G4_{ITER}.md", f"g1_{ITER}.md", f"G1_{ITER}.md.bak", "README.md"):
            with self.subTest(bad), self.assertRaises(ValueError):
                gates.parse_report_name(bad)

    def test_head_refuses_lines_the_launcher_would_misread(self) -> None:
        cases = {
            "row named like a Result line": dict(rows=[gates.Row("Result of fit", "1", "1", True)]),
            "row named like a Critic line": dict(rows=[gates.Row("critic review", "1", "1", True)]),
            "row named like a Kill line": dict(rows=[gates.Row("Kill criterion check", "1", "1", True)]),
            "row named like a Coded check line": dict(rows=[gates.Row("coded checks", "1", "1", True)]),
            "data file named results": dict(data_files=[("result/x.npz", "0" * 64)]),
            "placeholder on Runs": dict(runs="<run IDs, and where each ran>"),
            "angle-bracket word on Left out": dict(left_out="<pending>"),
            "template PASS | FAIL": dict(command="echo 'PASS | FAIL'"),
        }
        for name, kw in cases.items():
            with self.subTest(name):
                result = self.make(True)
                for key, value in kw.items():
                    setattr(result, key, value)
                with self.assertRaises(ValueError):
                    gates.render_report_head(result, ITER)
                self.assertFalse((self.tmp / f"G1_{ITER}.md").exists())
        fine = self.make(True)
        fine.rows.append(gates.Row("derivations/critic_as2018.py", "gamma < 0.05 M S", "x < y", True))
        gates.render_report_head(fine, ITER)

    def test_verdict_line_agrees_with_launcher_reader(self) -> None:
        for name, extra in (("bold report label", "**Report:** see below\n"),
                            ("list item report label", "- report: draft\n"),
                            ("lower-case label", "report:\n")):
            with self.subTest(name):
                text = CRITIC_OK.replace("something\n", extra)
                self.assertIsNone(gates.critic_verdict_line(text))
                self.assertIsNone(gates.critic_request_text(text))
        self.assertEqual(gates.critic_request_text(CRITIC_OK), "something\n")

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
        path = gates.write_report(self.make(True), ITER, self.tmp)
        with self.assertRaises(ValueError):
            finish(path, self.critic_ok, "Kill criterion 1 met: residual flat")
        self.assertNotIn("Result:", path.read_text())

    def test_finish_refuses_critic_of_other_iteration(self) -> None:
        path = gates.write_report(self.make(True), ITER, self.tmp)
        other = write(self.tmp / "critic_it-20260101-0000_1.md",
                      CRITIC_OK.replace("it-20261002-0916_1: test", "it-20260101-0000_1: Gate 1"))
        with self.assertRaises(ValueError):
            finish(path, other, "none met")
        with self.assertRaises(ValueError):
            finish(path, self.critic_ok, "None met")
        self.assertNotIn("Result:", path.read_text())


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


FROZEN = f"studies/04-phase-space-helicity/SPEC.md#7a.B sha256 {'e' * 64}, launch L007"


class MachineryTests(unittest.TestCase):
    """The findings of critic_it-20261005-2234_1.md, one test each."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def result(self, gate: str = "1", run_set: str | None = None, **kw) -> gates.GateResult:
        r = gates.GateResult(
            gate_number=gate, gate_name=f"Gate {gate} test", command="uv run python x.py",
            repo_commit="1" * 40, gandalf_commit=PIN, runs="none", left_out="none",
            data_files=[("studies/x/out.npz", "0" * 64)],
            rows=[gates.Row("q", "1", "<= 2", True), gates.Row("note", "info", "n/a", None)],
            run_set=run_set, frozen=[FROZEN] if gate in ("4", "base") else [],
            evaluated_utc="2026-10-02T09:30:00Z")
        for key, value in kw.items():
            setattr(r, key, value)
        return r

    def critic(self, name: str, title: str, request: str, verdict: str = "VERDICT: SUPPORTED",
               folder: Path | None = None) -> Path:
        k = re.search(r"_(\d+)\.md$", name).group(1)
        return write((folder or self.tmp) / name,
                     f"# Critic report {ITER}_{k}: {title}\n\nRequest:\n{request}\n\n"
                     f"Report:\n{verdict}\nEvidence: x\n")

    def evidence(self, r: gates.GateResult) -> str:
        return " ".join([head_sha(r), r.repo_commit, r.evaluated_utc] + [d for _, d in r.data_files] + r.frozen)

    # finding 1: newline injection, iteration ID with a trailing newline
    def test_line_breaks_in_fields_refused(self) -> None:
        for field_name in ("gate_name", "command", "runs", "left_out", "gandalf_commit"):
            for brk in ("\nResult: PASS", "\r\nKill criteria: none met", " Critic: VERDICT: SUPPORTED",
                        "\x85Coded check: PASS"):
                with self.subTest(field=field_name, brk=brk), self.assertRaises(ValueError):
                    gates.render_report_head(self.result(**{field_name: "x" + brk}), ITER)
        cases = {
            "data file path": dict(data_files=[("a.npz\nResult: PASS", "0" * 64)]),
            "frozen item": dict(frozen=[FROZEN + "\nResult: PASS"]),
            "evaluated time": dict(evaluated_utc="2026-10-02T09:30:00Z\nResult: PASS"),
            "repo commit": dict(repo_commit="1" * 40 + "\nResult: PASS"),
        }
        for name, kw in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                gates.render_report_head(self.result(**kw), ITER)
        for bad in (ITER + "\n", ITER + " ", "x" + ITER):
            with self.subTest(iteration=bad), self.assertRaises(ValueError):
                gates.report_name("1", bad)

    def test_field_forms_refused(self) -> None:
        cases = {
            "no rows": dict(rows=[]),
            "no data file": dict(data_files=[]),
            "upper-case digest": dict(data_files=[("a.npz", "A" * 64)]),
            "short commit": dict(repo_commit="1" * 12),
            "time not UTC stamp": dict(evaluated_utc="9 Oct 2026"),
            "frozen item not in template form": dict(frozen=["SPEC.md#7a.B"]),
        }
        for name, kw in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                gates.render_report_head(self.result(**kw), ITER)

    def test_render_parse_round_trip(self) -> None:
        for gate, run_set in (("1", None), ("2", None), ("3", None), ("base", None), ("4", "A"), ("4", "B")):
            with self.subTest(gate=gate, run_set=run_set):
                r = self.result(gate, run_set, rows=[gates.Row("a | b", " x\ty ", "t", False),
                                                     gates.Row("c", "", "-", None)])
                name = gates.report_name(gate, ITER, run_set)
                text = gates.render_report_head(r, ITER)
                again = gates.parse_report_head(text, name)
                self.assertEqual(gates.render_report_head(again, ITER), text)
                self.assertFalse(again.coded_pass)

    # finding 2: finish trusts the head
    def test_finish_refuses_heads_the_writer_did_not_write(self) -> None:
        r = self.result()
        crit = self.critic(f"critic_{ITER}_1.md", f"Gate 1 evaluation G1_{ITER}.md", self.evidence(r))
        head = gates.render_report_head(r, ITER)
        cases = {
            "row marked no under Coded check PASS": head.replace("| <= 2 | yes |", "| <= 2 | no |"),
            "no results table": re.sub(r"\n\| Quantity.*?\n\n", "\n\n", head, flags=re.S),
            "no final newline": head.rstrip("\n"),
            "extra blank line": head.replace("Data files:", "\nData files:"),
            "Coded check FAIL over passing rows": head.replace("Coded check: PASS", "Coded check: FAIL"),
            "hand-typed row with a fourth column": head.replace("| <= 2 | yes |", "| <= 2 | yes | x |"),
        }
        self.assertNotEqual(cases["no results table"], head)
        for name, text in cases.items():
            with self.subTest(name):
                path = write(self.tmp / f"G1_{ITER}.md", text)
                with self.assertRaises(ValueError):
                    finish(path, crit, "none met")
                self.assertEqual(path.read_text(), text)
        path = write(self.tmp / f"G1_{ITER}.md", head)
        self.assertEqual(finish(path, crit, "none met"), "PASS")
        self.assertTrue(path.read_text().endswith("Kill criteria: none met\nResult: PASS\n"))

    # finding 3: critic file location; a report committed by another iteration
    def test_finish_refuses_critic_in_another_folder(self) -> None:
        r = self.result()
        path = gates.write_report(r, ITER, self.tmp / "gate_reports")
        elsewhere = self.critic(f"critic_{ITER}_1.md", f"Gate 1 evaluation G1_{ITER}.md", self.evidence(r),
                                folder=self.tmp / "scratch")
        with self.assertRaises(ValueError):
            finish(path, elsewhere, "none met")
        beside = self.critic(f"critic_{ITER}_1.md", f"Gate 1 evaluation G1_{ITER}.md", self.evidence(r),
                             folder=self.tmp / "gate_reports")
        with self.assertRaises(ValueError):  # both in a folder other than the one required
            finish(path, beside, "none met", reports_dir=self.tmp / "other")
        self.assertNotIn("Result:", path.read_text())
        self.assertEqual(finish(path, beside, "none met", reports_dir=self.tmp / "gate_reports"),
                         "PASS")

    def test_report_history(self) -> None:
        import subprocess
        repo = self.tmp / "repo"
        repo.mkdir()

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", *args], check=True,
                           capture_output=True)

        git("init", "-q")
        report = write(repo / "gate_reports" / f"G1_{ITER}.md", "head\n")
        self.assertEqual(gates.report_history_problems(report, ITER, repo), [])  # untracked
        git("add", ".")
        git("commit", "-q", "-m", f"Study 04 [{ITER}]: Gate 1 head")
        self.assertEqual(gates.report_history_problems(report, ITER, repo), [])  # same iteration
        report.write_text("head\nmore\n")
        git("commit", "-q", "-am", "Study 04 [it-20261003-0000]: touched later")
        self.assertEqual(len(gates.report_history_problems(report, ITER, repo)), 1)
        self.assertEqual(gates.report_history_problems(self.tmp / "outside.md", ITER, repo), [])

    # finding 4: the verdict readers agree, or the writer refuses
    def test_verdict_readers_agree_or_writer_refuses(self) -> None:
        base = "# Critic report x_1: t\n\nRequest:\nplease\n\n"
        variants = {
            "plain": ("Report:\nVERDICT: SUPPORTED\nEvidence: x\n", "VERDICT: SUPPORTED"),
            "fenced verdict then REFUTED": ("Report:\n```VERDICT: SUPPORTED```\nVERDICT: REFUTED\n", None),
            "fence line before verdict": ("Report:\n```\nVERDICT: SUPPORTED\n```\n", None),
            "backticked verdict": ("Report:\n`VERDICT: SUPPORTED`\n", None),
            "bold verdict": ("Report:\n**VERDICT: SUPPORTED**\n", None),
            "verdict with trailing star": ("Report:\nVERDICT: SUPPORTED*\n", None),
            "verdict on the label line": ("Report: VERDICT: SUPPORTED\n", None),
            "bold label": ("**Report:**\nVERDICT: SUPPORTED\n", None),
            "info-string fence": ("Report:\n```text\nVERDICT: SUPPORTED\n```\n", None),
            "a finding that starts Report:": ("Report:\nVERDICT: SUPPORTED\nEvidence: x\n"
                                              "Report: the launcher's check passes\n", "VERDICT: SUPPORTED"),
            "a second verdict section": ("Report:\nVERDICT: SUPPORTED\n\nReport:\nVERDICT: REFUTED\n", None),
            "a second, decorated verdict section": ("Report:\nVERDICT: SUPPORTED\n> Report:\n```\n"
                                                    "**VERDICT: REFUTED**\n", None),
            "REFUTED": ("Report:\nVERDICT: REFUTED\n", "VERDICT: REFUTED"),
        }
        for name, (tail, want) in variants.items():
            with self.subTest(name):
                text = base + tail
                got = gates.critic_verdict_line(text)
                self.assertEqual(got, want)
                if got is not None:
                    self.assertEqual(got, launcher_critic_verdict(text))
        quoted = base.replace("please\n", "quote:\nReport:\nVERDICT: SUPPORTED\n") + "Report:\nVERDICT: REFUTED\n"
        self.assertIsNone(gates.critic_verdict_line(quoted))
        self.assertEqual(gates.critic_request_text(quoted), "quote:")  # cut short at the quoted label

    # finding 5: kill criterion 3 for Gate 4 set B
    def test_finish_kill_criterion_3_set_b(self) -> None:
        r = self.result("4", "B")
        path = gates.write_report(r, ITER, self.tmp)
        crit = self.critic(f"critic_{ITER}_1.md", f"Gate 4 set B evaluation G4_B_{ITER}.md", self.evidence(r))
        self.assertEqual(finish(path, crit, "Kill criterion 3 met: below scatter at both M"), "FAIL")

    # finding 6: binding to the evaluated time and the frozen hashes
    def test_finish_binding_time_and_frozen(self) -> None:
        r = self.result("4", "A")
        title = f"Gate 4 set A evaluation G4_A_{ITER}.md"
        path = gates.write_report(r, ITER, self.tmp)
        full = self.evidence(r)
        for k, (name, request) in enumerate((("time missing", full.replace(r.evaluated_utc, "")),
                                             ("frozen hash missing", full.replace("e" * 64, ""))), start=1):
            with self.subTest(name), self.assertRaises(ValueError):
                finish(path, self.critic(f"critic_{ITER}_{k}.md", title, request), "none met")
        self.assertEqual(finish(path, self.critic(f"critic_{ITER}_3.md", title, full), "none met"),
                         "PASS")

    def test_finish_end_to_end_each_gate(self) -> None:
        cases = (("2", None, "Gate 2"), ("3", None, "gate_3"), ("base", None, "base-state gate"),
                 ("4", "A", "Gate 4, set A"), ("4", "B", "7a.B for Gate 4"))
        for k, (gate, run_set, label) in enumerate(cases, start=1):
            with self.subTest(gate=gate, run_set=run_set):
                folder = self.tmp / f"g{k}"
                r = self.result(gate, run_set)
                path = gates.write_report(r, ITER, folder)
                crit = self.critic(f"critic_{ITER}_{k}.md", f"{label} evaluation {path.name}", self.evidence(r),
                                   folder=folder)
                self.assertEqual(finish(path, crit, "none met"), "PASS")
                text = path.read_text()
                self.assertEqual(text.splitlines()[0], gates.report_title(gate, ITER, run_set))
                self.assertTrue(text.endswith(f"Critic: VERDICT: SUPPORTED, {ITER}, {crit.name}\n"
                                              "Kill criteria: none met\nResult: PASS\n"))

    # finding 9: main(), tree_is_clean, installed_gandalf_intact
    def test_main_refuses_dirty_tree_and_moved_head(self) -> None:
        from unittest import mock
        with mock.patch.object(gates, "tree_is_clean", return_value=False), \
                mock.patch.object(gates, "gate1_check") as check:
            self.assertEqual(gates.main(["gate1", "--iteration", ITER]), 2)
            check.assert_not_called()
        with mock.patch.object(gates, "tree_is_clean", return_value=True), \
                mock.patch.object(gates, "gate1_check", return_value=self.result()), \
                mock.patch.object(gates, "_safe_head", return_value="2" * 40), \
                mock.patch.object(gates, "write_report") as wr:
            self.assertEqual(gates.main(["gate1", "--iteration", ITER]), 2)
            wr.assert_not_called()

    def test_main_finish_requires_gate_reports(self) -> None:
        r = self.result()
        path = gates.write_report(r, ITER, self.tmp)
        crit = self.critic(f"critic_{ITER}_1.md", f"Gate 1 evaluation G1_{ITER}.md", self.evidence(r))
        with self.assertRaises(ValueError):
            gates.main(["finish", "--report", str(path), "--critic", str(crit), "--kill", "none met"])
        self.assertNotIn("Result:", path.read_text())

    def test_hidden_index_bits(self) -> None:
        out = "H studies/a.py\nS studies/b.py\nh studies/c.py\ns studies/d.py\nH shared/e.py\n"
        self.assertEqual(gates.hidden_index_bits(out), ["studies/b.py", "studies/c.py", "studies/d.py"])
        self.assertEqual(gates.hidden_index_bits("H x\n"), [])

    def test_tree_is_clean_sees_untracked_files(self) -> None:
        import subprocess
        repo = self.tmp / "repo"
        repo.mkdir()
        subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
        self.assertTrue(gates.tree_is_clean(repo, paths=(".",)))
        write(repo / "deep" / "dir" / "new.txt", "x")
        self.assertFalse(gates.tree_is_clean(repo, paths=(".",)))

    def test_installed_gandalf_intact_negative(self) -> None:
        import base64
        import hashlib
        from types import SimpleNamespace
        from unittest import mock
        good = write(self.tmp / "krmhd" / "a.py", "a = 1\n")
        changed = write(self.tmp / "krmhd" / "b.py", "b = 2\n")
        meta = write(self.tmp / "gandalf_krmhd.dist-info" / "METADATA", "m\n")

        class Entry:
            def __init__(self, path: Path, content: bytes) -> None:
                digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
                self.hash = SimpleNamespace(mode="sha256", value=digest)
                self.path = path

            def locate(self) -> Path:
                return self.path

            def __str__(self) -> str:
                return self.path.relative_to(self.path.parents[1]).as_posix()

        files = [Entry(good, b"a = 1\n"), Entry(changed, b"b = 1\n"),
                 Entry(self.tmp / "krmhd" / "gone.py", b"c\n")]
        with mock.patch.object(gates.importlib.metadata, "distribution",
                               return_value=SimpleNamespace(files=files)):
            ok, seen = gates.installed_gandalf_intact()
        self.assertFalse(ok)
        self.assertIn("2 changed", seen)
        with mock.patch.object(gates.importlib.metadata, "distribution",
                               return_value=SimpleNamespace(files=files[:1])):
            self.assertTrue(gates.installed_gandalf_intact()[0])
        with mock.patch.object(gates.importlib.metadata, "distribution",
                               return_value=SimpleNamespace(files=[])):
            self.assertFalse(gates.installed_gandalf_intact()[0])
        # hashed files that are not the package itself do not count
        with mock.patch.object(gates.importlib.metadata, "distribution",
                               return_value=SimpleNamespace(files=[Entry(meta, b"m\n")])):
            self.assertFalse(gates.installed_gandalf_intact()[0])

    def test_isolated_install_info(self) -> None:
        commit, ok, seen = gates.isolated_install_info()
        self.assertEqual(commit, gates.uv_lock_pin(gates.REPO / "uv.lock"))
        self.assertTrue(ok, seen)
        # an interpreter that prints nothing usable gives 'unknown' and not intact
        self.assertEqual(gates.isolated_install_info(python="/usr/bin/true"),
                         ("unknown", False, "install check failed (exit 0)"))

    def test_frozen_lines_by_gate(self) -> None:
        with self.assertRaises(ValueError):
            gates.render_report_head(self.result("4", "A", frozen=[]), ITER)
        with self.assertRaises(ValueError):
            gates.render_report_head(self.result("base", frozen=[]), ITER)
        with self.assertRaises(ValueError):
            gates.render_report_head(self.result("2", frozen=[FROZEN]), ITER)
        with self.assertRaises(ValueError):
            gates.report_name("4", ITER, "Å")

    def test_finish_reads_bytes(self) -> None:
        r = self.result()
        path = gates.write_report(r, ITER, self.tmp)
        crit = self.critic(f"critic_{ITER}_1.md", f"Gate 1 evaluation G1_{ITER}.md", self.evidence(r))
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaises(ValueError):
            finish(path, crit, "none met")

    def test_finish_needs_head_sha_and_rejudge(self) -> None:
        r = self.result()
        path = gates.write_report(r, ITER, self.tmp)
        title = f"Gate 1 evaluation G1_{ITER}.md"
        no_sha = self.critic(f"critic_{ITER}_1.md", title, self.evidence(r).replace(head_sha(r), ""))
        with self.assertRaises(ValueError):
            finish(path, no_sha, "none met")
        crit = self.critic(f"critic_{ITER}_2.md", title, self.evidence(r))
        with self.assertRaises(ValueError):  # no re-judge given
            gates.finish_report(path, crit, "none met")

        def flipped(stated: gates.GateResult, iteration: str) -> gates.GateResult:
            rows = [gates.Row(x.quantity, x.value, x.threshold, False if x.passed else x.passed)
                    for x in stated.rows]
            return gates.GateResult(**{**stated.__dict__, "rows": rows})

        with self.assertRaises(ValueError):  # the outputs say the row failed
            finish(path, crit, "none met", rejudge=flipped)
        self.assertNotIn("Result:", path.read_text())

    def test_gate_names_as_the_runner_reads_them(self) -> None:
        base = gates.gate_label_re(f"# Gate base report: {ITER}")
        self.assertFalse(base.search(f"evaluation Gbase_{ITER}.md"))
        self.assertTrue(base.search(f"Gbase evaluation Gbase_{ITER}.md"))
        set_a = gates.gate_label_re(f"# Gate 4 report, set A: {ITER}")
        self.assertFalse(set_a.search("gate 4 set a"))
        self.assertTrue(set_a.search("Gate 4 set A"))
        self.assertTrue(set_a.search("gate-4, Set A"))


if __name__ == "__main__":
    unittest.main()
