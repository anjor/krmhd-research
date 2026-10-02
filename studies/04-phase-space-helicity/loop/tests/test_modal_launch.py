"""Tests for the Study 04 Modal launcher, its Modal app and the shared loop helpers.

Run from the repo root:

    uv run python -m unittest discover -s studies/04-phase-space-helicity/loop/tests -v

No test reaches the network, Modal, GitHub or the real repo's git state:

- Launcher tests copy the launcher into a scratch git repo with a bare ``origin`` under a
  temp dir and run it as a subprocess with the stub backend (S04_MODAL_BACKEND=stub).
- The real Modal backend's logic (call status, attempt records, downloads, detached spawn)
  is tested in-process against fake Modal objects.
- The container side of ``modal_app.py`` runs locally against a temp folder that stands in
  for the volume.
- Every subprocess gets MODAL_SERVER_URL pointing at a dead local port and fake tokens, so
  an accidental Modal call would fail at once instead of reaching Modal.
"""
from __future__ import annotations

import contextlib
import enum
import fcntl
import importlib.util
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import types
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
from unittest import mock

sys.dont_write_bytecode = True

TESTS_DIR = Path(__file__).resolve().parent
LOOP_DIR = TESTS_DIR.parent
REAL_REPO = LOOP_DIR.parents[2]
if str(LOOP_DIR) not in sys.path:
    sys.path.insert(0, str(LOOP_DIR))

import loopcommon as lc  # noqa: E402

STUDY = lc.STUDY_REL
LEDGER = lc.LEDGER_REL
PREFIX = lc.LAUNCHER_SUBJECT_PREFIX
GANDALF_SHA = "6596bb85af8795a857bc8e52d275ebffa30ba6a5"
OTHER_SHA = "0123456789abcdef0123456789abcdef01234567"
LOOP_AUTHOR = "krmhd-loop"
LOOP_EMAIL = "loop@example.invalid"
ITER_ID = "it-20261005-1012"
CAP = 20.0

CFG1 = f"{STUDY}/configs/setA_c1.yaml"
CFG2 = f"{STUDY}/configs/setA_c2.yaml"
CFG3 = f"{STUDY}/configs/setA_c3.yaml"
CRITIC_ITER = "it-20261005-1012"
CRITIC_REPORT = f"{STUDY}/gate_reports/critic_{CRITIC_ITER}_1.md"
GATE1 = f"{STUDY}/gate_reports/G1_it-20261001-0900.md"
GATE = f"{STUDY}/gate_reports/G3_it-20261005-1012.md"
GATE_BASE = f"{STUDY}/gate_reports/Gbase_it-20261004-0900.md"
GATE4_A = f"{STUDY}/gate_reports/G4_A_it-20261010-0900.md"
# The reports each run set launches on by default (LOOP.md section 9, "Before a launch").
SET_GATES = {"base": (GATE1,), "A": (GATE, GATE_BASE), "B": (GATE4_A,)}
LOCAL = f"{STUDY}/local_tests/setA_smoke.log"
RUN_SET_PY = f"{STUDY}/runcode/run_set.py"
RUNS_MD = f"{STUDY}/RUNS.md"

SAFE_MODAL_ENV = {
    "MODAL_SERVER_URL": "http://127.0.0.1:9",
    "MODAL_TOKEN_ID": "ak-test-not-a-real-token",
    "MODAL_TOKEN_SECRET": "as-test-not-a-real-token",
    "MODAL_CONFIG_PATH": "/nonexistent/s04-test-modal.toml",
}

UV_LOCK = textwrap.dedent(f"""\
    version = 1
    revision = 3
    requires-python = ">=3.10"

    [[package]]
    name = "gandalf-krmhd"
    version = "0.6.0"
    source = {{ git = "https://github.com/anjor/gandalf.git?rev=v0.6.0#{GANDALF_SHA}" }}

    [[package]]
    name = "jax"
    version = "0.6.2"
    source = {{ registry = "https://pypi.org/simple" }}
    resolution-markers = [
        "python_full_version < '3.11'",
    ]

    [[package]]
    name = "jax"
    version = "0.9.1"
    source = {{ registry = "https://pypi.org/simple" }}
    resolution-markers = [
        "python_full_version >= '3.14'",
        "python_full_version == '3.13.*'",
        "python_full_version == '3.12.*'",
        "python_full_version == '3.11.*'",
    ]
    """)

PYPROJECT = textwrap.dedent("""\
    [project]
    name = "scratch"
    version = "0.1.0"
    requires-python = ">=3.10"
    dependencies = [
        "gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@{pin}",
        "jax",
    ]
    """)

SPEC = textwrap.dedent("""\
    # SPEC (scratch)

    ## 7. Tolerances

    Energy conservation within 1%.

    ## 7a. Gate 4 criteria per run set

    ### 7a.A Set A

    Criterion A: the effect exceeds three standard errors.

    ### 7a.B Set B

    Criterion B.

    ## 8. Next

    Nothing.
    """)


def gate_report(result: str = "PASS", coded: str = "PASS", critic: str = "VERDICT: SUPPORTED",
                kill: str = "none met", title: str = "# Gate 3 report: it-20261005-1012", extra: str = "",
                critic_file: str = CRITIC_REPORT, critic_iter: str = CRITIC_ITER) -> str:
    """A gate report in the form of LOOP.md ("Gate reports"), filled in as the gate code would."""
    return textwrap.dedent(f"""\
        {title}

        Gate: Gate 3, the sign flip of the Gamma injection (PLAN.md section 4, SPEC.md section 7)
        Evaluated: 2026-10-05T10:12:00Z, repo commit 0123abcd, GANDALF commit 6596bb85
        Command: uv run python studies/04-phase-space-helicity/analysis/gates.py --gate 3
        Runs: 04_g3_phi0_20261005_100000 (local), 04_g3_phipi_20261005_100500 (local)
        Left out: none
        Data files:
        - studies/04-phase-space-helicity/data/local/04_g3_phi0_20261005_100000/diag.npz sha256 0a1b2c

        | Quantity | Value | Threshold | Pass |
        |---|---|---|---|
        | sign of the flux, phi = 0 vs pi | opposite | opposite | yes |
        | <W(m)> ratio | 1.02 | < 1.05 | yes |
        {extra}
        Coded check: {coded}
        Critic: {critic}, {critic_iter}, {critic_file}
        Kill criteria: {kill}
        Result: {result}
        """)


def critic_report(verdict: str = "VERDICT: SUPPORTED", iteration: str = CRITIC_ITER) -> str:
    """A saved critic review in the form of LOOP.md section 6."""
    return textwrap.dedent(f"""\
        # Critic report {iteration}_1: Gate 3, the sign flip of the Gamma injection

        Request:
        Test whether the gate code implements SPEC.md section 7 and whether the outputs meet it.

        Report:
        {verdict}
        Evidence: recomputed the flux from diag.npz; it matches the gate code to 1e-12.
        """)


def records_dir(volume_path: str) -> str:
    """Where modal_app writes a run's attempt records: /<root>/_attempts/<set>/<run> (no leading /)."""
    root, run_set, run_id = volume_path.strip("/").split("/")
    return f"{root}/_attempts/{run_set}/{run_id}"


# The template of LOOP.md, "Gate reports", as an unfilled report would copy it.
LOOP_TEMPLATE = textwrap.dedent("""\
    # Gate <n> report[, set <S>]: <iteration id>

    Gate: <name, with PLAN.md and SPEC.md references>
    Evaluated: <UTC time>, repo commit <sha>, GANDALF commit <sha> (for the base-state gate and Gate 4, the runs' commit from the ledger)
    Command: <the exact command line>
    Runs: <run IDs, and where each ran>
    Left out: <run IDs and why> | none
    Data files:
    - <path> sha256 <hash>
    Frozen (base-state gate and Gate 4 only): <freeze key> sha256 <hash>, launch <Lnnn>

    | Quantity | Value | Threshold | Pass |
    |---|---|---|---|

    Coded check: PASS | FAIL
    Critic: <its VERDICT line, verbatim>, <iteration id>, <critic report file>
    Kill criteria: none met | <which, with the evidence>
    Result: PASS | FAIL | NOT DECIDED
    """)

TEMPLATE_FILES: Dict[str, str] = {
    ".gitignore": textwrap.dedent(f"""\
        **/data/
        __pycache__/
        *.pyc
        {STUDY}/.loop/
        {STUDY}/configs/ignored_*.yaml
        {STUDY}/gate_reports/ignored_*.md
        """),
    "CLAUDE.md": "# scratch repo for launcher tests\n",
    "uv.lock": UV_LOCK,
    "pyproject.toml": PYPROJECT.format(pin="v0.6.0"),
    f"{STUDY}/PLAN.md": "# PLAN (scratch)\n",
    f"{STUDY}/SPEC.md": SPEC,
    f"{STUDY}/RUNS.md": textwrap.dedent("""\
        # Run queue

        | Set | Config | Purpose | Gate | Est. A100-h | Status | Run ID | Data path | Actual A100-h |
        |---|---|---|---|---|---|---|---|---|
        | base | configs/base_*.yaml | test | G1 | 4 | planned | | | |
        | A | configs/setA_*.yaml | test | G3 | 6 | planned | | | |
        | B | configs/setB_*.yaml | test | G4 A | 10 | planned | | | |
        """),
    f"{STUDY}/loop/config.env": textwrap.dedent(f"""\
        # scratch config for the launcher tests
        COMPUTE_CAP_A100_HOURS={CAP:g}
        MODAL_APP_PREFIX=s04-loop
        MODAL_VOLUME=krmhd-benchmark-vol
        MODAL_VOLUME_ROOT=study04
        LOOP_GIT_NAME={LOOP_AUTHOR}
        """),
    RUN_SET_PY: "def run(cfg, out_dir, ctx):\n    return {'ok': True}\n",
    f"{STUDY}/runcode/no_run.py": "def main():\n    return 0\n",
    CFG1: f"entrypoint: {RUN_SET_PY}\nN: 32\nseed: 1\n",
    CFG2: f"entrypoint: {RUN_SET_PY}\nN: 32\nseed: 2\n",
    CFG3: f"entrypoint: {RUN_SET_PY}\nN: 32\nseed: 3\n",
    f"{STUDY}/configs/sub/setA_c1.yaml": f"entrypoint: {RUN_SET_PY}\nN: 16\n",
    f"{STUDY}/configs/bad_entry.yaml": f"entrypoint: {STUDY}/runcode/no_run.py\n",
    f"{STUDY}/configs/bad_yaml.yaml": "entrypoint: [unclosed\n",
    f"{STUDY}/configs/no_entry.yaml": "N: 32\n",
    f"{STUDY}/configs/outside_entry.yaml": "entrypoint: shared/run_elsewhere.py\n",
    f"{STUDY}/configs/dot_entry.yaml": f"entrypoint: ./{RUN_SET_PY}\n",
    f"{STUDY}/configs/padded_entry.yaml": f"entrypoint: ' {RUN_SET_PY}'\n",
    f"{STUDY}/configs/ignored_c9.yaml": f"entrypoint: {RUN_SET_PY}\n",
    f"{STUDY}/elsewhere/setA_x.yaml": f"entrypoint: {RUN_SET_PY}\n",
    "shared/run_elsewhere.py": "def run(cfg, out_dir, ctx):\n    return None\n",
    CRITIC_REPORT: critic_report(),
    GATE1: gate_report(title="# Gate 1 report: it-20261001-0900"),
    GATE: gate_report(),
    GATE_BASE: gate_report(title="# Gate base report: it-20261004-0900"),
    GATE4_A: gate_report(title="# Gate 4 report, set A: it-20261010-0900"),
    f"{STUDY}/gate_reports/G3_fail.md": gate_report(result="FAIL", coded="FAIL"),
    f"{STUDY}/gate_reports/G3_lower.md": gate_report(result="pass").replace("Result: pass", "result: pass"),
    f"{STUDY}/gate_reports/G3_template.md": LOOP_TEMPLATE,
    f"{STUDY}/gate_reports/G3_inconclusive.md": gate_report(critic="VERDICT: INCONCLUSIVE"),
    f"{STUDY}/gate_reports/ignored_G3.md": gate_report(),
    f"{STUDY}/notes/G3_elsewhere.md": gate_report(),
    LOCAL: "local smoke test of run_set.py at 16^3: ok\n",
}


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def run_cmd(cmd: List[str], cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None,
            check: bool = True, timeout: int = 180) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env, capture_output=True,
                          text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise AssertionError(f"{cmd} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    return proc


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def wait_for(path: Path, timeout: float = 60.0) -> None:
    """Poll until ``path`` exists."""
    end = time.time() + timeout
    while not path.exists():
        if time.time() > end:
            raise AssertionError(f"{path} did not appear within {timeout} s")
        time.sleep(0.05)


def make_env(base: Path) -> Dict[str, str]:
    """Environment for git and the launcher: isolated git config, no real Modal."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "S04_", "MODAL_"))}
    gitconfig = base / "gitconfig"
    gitconfig.write_text(
        "[user]\n\tname = anjor\n\temail = anjor@example.invalid\n"
        "[init]\n\tdefaultBranch = main\n[commit]\n\tgpgsign = false\n"
        "[advice]\n\tdetachedHead = false\n")
    env.update(GIT_CONFIG_GLOBAL=str(gitconfig), GIT_CONFIG_NOSYSTEM="1", GIT_TERMINAL_PROMPT="0",
               PYTHONDONTWRITEBYTECODE="1", **SAFE_MODAL_ENV)
    return env


def build_template(root: Path, env: Dict[str, str]) -> None:
    repo, remote = root / "repo", root / "remote.git"
    run_cmd(["git", "init", "-q", "--bare", "-b", "main", str(remote)], env=env)
    run_cmd(["git", "init", "-q", "-b", "main", str(repo)], env=env)
    for rel, text in TEMPLATE_FILES.items():
        write(repo / rel, text)
    for name in ("loopcommon.py", "modal_launch.py", "modal_app.py"):
        shutil.copy2(LOOP_DIR / name, repo / STUDY / "loop" / name)
    run_cmd(["git", "-C", str(repo), "add", "-A"], env=env)
    run_cmd(["git", "-C", str(repo), "commit", "-q", "-m", "Scratch study (anjor)"], env=env)
    run_cmd(["git", "-C", str(repo), "remote", "add", "origin", str(remote)], env=env)
    run_cmd(["git", "-C", str(repo), "push", "-q", "origin", "main"], env=env)


_TMP: Optional[tempfile.TemporaryDirectory] = None
_TEMPLATE: Optional[Path] = None
_ENV: Dict[str, str] = {}


def setUpModule() -> None:
    global _TMP, _TEMPLATE, _ENV
    _TMP = tempfile.TemporaryDirectory(prefix="s04-launcher-tests-")
    base = Path(_TMP.name).resolve()
    _ENV = make_env(base)
    _TEMPLATE = base / "template"
    build_template(_TEMPLATE, _ENV)


def tearDownModule() -> None:
    if _TMP is not None:
        _TMP.cleanup()


Commit = Tuple[str, str, str, str]  # sha, author name, committer name, subject


class Scratch:
    """A scratch clone with a bare origin, the launcher, and a stub-backend state dir."""

    def __init__(self, base: Path) -> None:
        assert _TEMPLATE is not None
        self.root = base / "w"
        shutil.copytree(_TEMPLATE, self.root, symlinks=True)
        self.repo = self.root / "repo"
        self.remote = self.root / "remote.git"
        self.stub = self.root / "stub"
        self.stub.mkdir()
        self.env = dict(_ENV)
        self.git("remote", "set-url", "origin", str(self.remote))
        self.git("fetch", "-q", "origin")

    # git -------------------------------------------------------------------------

    def git(self, *args: str, check: bool = True) -> str:
        return run_cmd(["git", "-C", str(self.repo), *args], env=self.env, check=check).stdout

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()

    def remote_head(self) -> str:
        return run_cmd(["git", "--git-dir", str(self.remote), "rev-parse", "main"], env=self.env).stdout.strip()

    def porcelain(self) -> str:
        return self.git("status", "--porcelain")

    def commits_since(self, base: str) -> List[Commit]:
        """(sha, author, committer, subject), oldest first."""
        out = self.git("log", "--reverse", "--format=%H%x09%an%x09%cn%x09%s", f"{base}..HEAD")
        return [tuple(line.split("\t", 3)) for line in out.splitlines() if line]  # type: ignore[misc]

    def subjects_since(self, base: str) -> List[str]:
        return [c[3] for c in self.commits_since(base)]

    def files_of(self, sha: str) -> List[str]:
        return [line for line in self.git("show", "--name-only", "--format=", sha).splitlines() if line]

    def commit_file(self, rel: str, text: str, msg: str = "Anjor edit", push: bool = True) -> None:
        write(self.repo / rel, text)
        self.git("add", "--", rel)
        self.git("commit", "-q", "-m", msg)
        if push:
            self.git("push", "-q", "origin", "main")

    def loop_env(self) -> Dict[str, str]:
        return dict(self.env, GIT_AUTHOR_NAME=LOOP_AUTHOR, GIT_AUTHOR_EMAIL=LOOP_EMAIL,
                    GIT_COMMITTER_NAME=LOOP_AUTHOR, GIT_COMMITTER_EMAIL=LOOP_EMAIL)

    def git_as_loop(self, *args: str) -> str:
        """A git command with the loop's identity (as the agent's own commits would have)."""
        return run_cmd(["git", "-C", str(self.repo), *args], env=self.loop_env()).stdout

    def git_as_github(self, *args: str) -> str:
        """A git command with GitHub as committer, as for a web merge, a squash merge or a web edit."""
        env = dict(self.env, GIT_AUTHOR_NAME="Anjor Kanekar", GIT_AUTHOR_EMAIL="anjor@example.invalid",
                   GIT_COMMITTER_NAME="GitHub", GIT_COMMITTER_EMAIL="noreply@github.com")
        return run_cmd(["git", "-C", str(self.repo), *args], env=env).stdout

    def ledger_versions(self) -> List[Dict[str, Any]]:
        """Every committed version of the ledger, oldest first."""
        shas = self.git("log", "--reverse", "--first-parent", "--format=%H", "--", LEDGER).split()
        return [json.loads(self.git("show", f"{sha}:{LEDGER}")) for sha in shas]

    def push_from_other_clone(self, rel: str, text: str) -> None:
        other = self.root / "other"
        run_cmd(["git", "clone", "-q", str(self.remote), str(other)], env=self.env)
        write(other / rel, text)
        run_cmd(["git", "-C", str(other), "add", "--", rel], env=self.env)
        run_cmd(["git", "-C", str(other), "commit", "-q", "-m", "elsewhere"], env=self.env)
        run_cmd(["git", "-C", str(other), "push", "-q", "origin", "main"], env=self.env)

    # launcher ----------------------------------------------------------------------

    def launcher_env(self, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        e = self.loop_env()
        e.update(S04_REPO=str(self.repo), S04_MODAL_BACKEND="stub", S04_STUB_DIR=str(self.stub),
                 S04_ITER_ID=ITER_ID)
        if env:
            e.update(env)
        return e

    def launcher_argv(self, *args: str) -> List[str]:
        return [sys.executable, str(self.repo / STUDY / "loop" / "modal_launch.py"), *args]

    def launcher(self, *args: str, env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        return run_cmd(self.launcher_argv(*args), cwd=self.repo, env=self.launcher_env(env), check=False)

    def launcher_popen(self, *args: str, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        return subprocess.Popen(self.launcher_argv(*args), cwd=str(self.repo), env=self.launcher_env(env),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def launch_args(self, *configs: str, run_set: str = "A", hours: float = 3.0,
                    gates: Optional[Tuple[str, ...]] = None, local: str = LOCAL,
                    extra: Tuple[str, ...] = ()) -> List[str]:
        """The launch command line; the gate reports default to the ones the set launches on."""
        args = ["launch", "--set", run_set, "--timeout-hours", str(hours)]
        for gate in (SET_GATES.get(run_set, (GATE,)) if gates is None else gates):
            args += ["--gate-report", gate]
        return args + ["--local-test", local, *extra, *configs]

    def launch(self, *configs: str, run_set: str = "A", hours: float = 3.0, gate: Optional[str] = None,
               gates: Optional[Tuple[str, ...]] = None, local: str = LOCAL, extra: Tuple[str, ...] = (),
               env: Optional[Dict[str, str]] = None) -> subprocess.CompletedProcess:
        if gate is not None:
            gates = (gate,)
        return self.launcher(*self.launch_args(*configs, run_set=run_set, hours=hours, gates=gates,
                                               local=local, extra=extra), env=env)

    def status(self) -> str:
        proc = self.launcher("status")
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.splitlines()[0]

    # state ---------------------------------------------------------------------------

    def ledger(self) -> Optional[Dict[str, Any]]:
        path = self.repo / LEDGER
        return json.loads(path.read_text()) if path.exists() else None

    def remote_ledger(self) -> Optional[Dict[str, Any]]:
        proc = run_cmd(["git", "--git-dir", str(self.remote), "show", f"main:{LEDGER}"], env=self.env, check=False)
        return json.loads(proc.stdout) if proc.returncode == 0 else None

    def spawn_log(self) -> List[Dict[str, Any]]:
        path = self.stub / "spawn_log.json"
        return json.loads(path.read_text()) if path.exists() else []

    def server_calls(self) -> List[str]:
        path = self.stub / "server_calls.json"
        return json.loads(path.read_text()) if path.exists() else []

    def snapshot(self) -> Tuple[str, str, Optional[str], int, str]:
        path = self.repo / LEDGER
        return (self.head(), self.remote_head(), path.read_text() if path.exists() else None,
                len(self.spawn_log()), self.porcelain())

    def set_calls(self, mapping: Dict[str, str]) -> None:
        (self.stub / "calls.json").write_text(json.dumps(mapping))

    def set_apps(self, apps: List[Dict[str, str]]) -> None:
        (self.stub / "apps.json").write_text(json.dumps(apps))

    def record(self, volume_path: str, attempt_id: str, event: str, t: float, run_id: str,
               status: Optional[str] = None) -> None:
        """An attempt record in the format modal_app._Attempt writes, where it writes it."""
        folder = self.stub / "volume" / records_dir(volume_path)
        folder.mkdir(parents=True, exist_ok=True)
        rec: Dict[str, Any] = {"event": event, "attempt_id": attempt_id, "run_id": run_id,
                               "launch_id": "L001", "t_epoch": t, "t_utc": lc_utc(t),
                               "call_id": None, "task_id": None, "gpu": "A100"}
        if event == "end":
            rec.update(status=status or "ok", error=None if status in (None, "ok") else "boom")
        (folder / f"{int(t)}_{attempt_id}_{event}.json").write_text(json.dumps(rec))


def lc_utc(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


class LauncherCase(unittest.TestCase):
    """Each test gets a fresh scratch repo copied from the module's template."""

    T0 = 1_790_000_000.0

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="s04-case-")
        self.s = Scratch(Path(self._tmp.name).resolve())
        self.base_head = self.s.head()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def assertOk(self, proc: subprocess.CompletedProcess) -> None:
        self.assertEqual(proc.returncode, 0, msg=f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    def assertExit(self, proc: subprocess.CompletedProcess, code: int) -> None:
        self.assertEqual(proc.returncode, code, msg=f"\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    def assertRefused(self, proc: subprocess.CompletedProcess, rule: str) -> None:
        self.assertExit(proc, 1)
        self.assertIn(f"REFUSED [{rule}]", proc.stderr)

    def assertLauncherCommits(self, base: str) -> None:
        """Every commit since ``base`` is the launcher's: loop identity, prefix, ledger only."""
        commits = self.s.commits_since(base)
        self.assertTrue(commits)
        for sha, author, committer, subject in commits:
            self.assertEqual((author, committer), (LOOP_AUTHOR, LOOP_AUTHOR), subject)
            self.assertTrue(subject.startswith(PREFIX), subject)
            self.assertEqual(self.s.files_of(sha), [LEDGER], subject)

    def assertHistoryExtends(self) -> None:
        """Each committed ledger version is a launcher-made successor of the one before."""
        versions = self.s.ledger_versions()
        self.assertTrue(versions)
        for prev, cur in zip(versions, versions[1:]):
            self.assertEqual(lc.ledger_extends(prev, cur), [])

    def launched(self, *configs: str, hours: float = 3.0) -> Dict[str, Any]:
        self.assertOk(self.s.launch(*(configs or (CFG1,)), hours=hours))
        ledger = self.s.ledger()
        assert ledger is not None
        return ledger["launches"][-1]


# ---------------------------------------------------------------------------
# launch
# ---------------------------------------------------------------------------


class LaunchTests(LauncherCase):

    def test_reservation_is_pushed_before_spawn_then_launch_recorded(self) -> None:
        proc = self.s.launch(CFG1, CFG2, hours=3)
        self.assertOk(proc)
        # The stub saw, on the REMOTE, the reservation (state reserved) when Modal was called.
        log = self.s.spawn_log()
        self.assertEqual(len(log), 1)
        remote_at_spawn = log[0]["remote_ledger"]
        self.assertIsNotNone(remote_at_spawn)
        self.assertEqual(lc.verify_ledger(remote_at_spawn), [])
        reserved = remote_at_spawn["launches"][0]
        self.assertEqual(reserved["state"], "reserved")
        self.assertIsNone(reserved["app_id"])
        self.assertEqual([r["status"] for r in reserved["runs"]], ["reserved", "reserved"])
        self.assertEqual([r["reserved_hours"] for r in reserved["runs"]], [3.0, 3.0])
        # Commit order: reserve, then launched; both the launcher's, ledger only; pushed.
        self.assertEqual(self.s.subjects_since(self.base_head),
                         [f"{PREFIX}: reserve L001 set A: 2 x 3 h = 6 A100-h",
                          f"{PREFIX}: launched L001 set A app ap-stub-1"])
        self.assertLauncherCommits(self.base_head)
        self.assertEqual(self.s.head(), self.s.remote_head())
        self.assertEqual(self.s.porcelain(), "")
        # The recorded launch.
        ledger = self.s.ledger()
        self.assertEqual(ledger, self.s.remote_ledger())
        self.assertEqual(lc.verify_ledger(ledger), [])
        launch = ledger["launches"][0]
        expected = {
            "launch_id": "L001", "kind": "gpu", "run_set": "A", "iteration": ITER_ID,
            "repo_commit": self.base_head, "gandalf_commit": GANDALF_SHA, "jax_version": "0.9.1",
            "app_name": "s04-loop-l001-a", "app_id": "ap-stub-1", "timeout_hours": 3.0,
            "gate_report": [GATE, GATE_BASE], "local_test": LOCAL, "frozen": {}, "state": "launched",
            "volume": "krmhd-benchmark-vol",
        }
        for key, value in expected.items():
            self.assertEqual(launch[key], value, key)
        self.assertRegex(launch["created_utc"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        stamps = set()
        for i, (run, cfg) in enumerate(zip(launch["runs"], (CFG1, CFG2)), start=1):
            m = re.match(r"^04_(setA_c\d)_(\d{8}_\d{6})$", run["run_id"])
            self.assertIsNotNone(m, run["run_id"])
            self.assertEqual(m.group(1), Path(cfg).stem)
            stamps.add(m.group(2))
            self.assertEqual(run["config"], cfg)
            self.assertEqual(run["config_sha256"], lc.sha256_file(self.s.repo / cfg))
            self.assertEqual(run["volume_path"], f"/study04/A/{run['run_id']}")
            self.assertEqual(run["call_id"], f"fc-stub-1-{i}")
            self.assertEqual(run["status"], "running")
            self.assertEqual(run["reserved_hours"], 3.0)
            self.assertIsNone(run["charged_hours"])
        self.assertEqual(len(stamps), 1, "all runs of a launch share one timestamp")
        # status: the compute line first, then one line per run in flight.
        status = self.s.launcher("status")
        self.assertOk(status)
        lines = status.stdout.splitlines()
        self.assertEqual(lines[0], "Compute cap 20.0 A100-h: used 0.00, reserved 6.00, left 14.00 "
                                   "(launches 1, runs in flight 2)")
        self.assertEqual(len([l for l in lines if l.startswith("in flight: ")]), 2)
        self.assertIn(lines[0], proc.stdout)  # launch prints the same line
        self.assertHistoryExtends()

    def test_absolute_config_path_inside_repo_is_accepted(self) -> None:
        self.assertOk(self.s.launch(str(self.s.repo / CFG1), hours=1))
        self.assertEqual(self.s.ledger()["launches"][0]["runs"][0]["config"], CFG1)

    def test_cap_arithmetic_after_reconcile(self) -> None:
        launch = self.launched(CFG1, CFG2, hours=3)
        r1 = launch["runs"][0]
        self.s.set_calls({r1["call_id"]: "ok"})
        self.s.record(r1["volume_path"], "a1", "start", self.T0, r1["run_id"])
        self.s.record(r1["volume_path"], "a1", "end", self.T0 + 5400, r1["run_id"], status="ok")
        self.assertOk(self.s.launcher("reconcile"))
        self.assertEqual(self.s.status(), "Compute cap 20.0 A100-h: used 1.50, reserved 3.00, "
                                          "left 15.50 (launches 1, runs in flight 1)")
        # Exactly at the cap is allowed: 1.5 + 3 + 15.5 = 20.
        self.assertOk(self.s.launch(CFG3, hours=15.5))
        self.assertEqual(self.s.status(), "Compute cap 20.0 A100-h: used 1.50, reserved 18.50, "
                                          "left 0.00 (launches 2, runs in flight 2)")
        # Anything more is refused, and the refusal changes nothing.
        snap = self.s.snapshot()
        proc = self.s.launch(CFG3, hours=0.1)
        self.assertRefused(proc, "cap")
        self.assertIn("20.10", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_refused_over_cap_changes_nothing(self) -> None:
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1, CFG2, CFG3, hours=7)  # 21 > 20
        self.assertRefused(proc, "cap")
        self.assertIn("21.00 A100-h would exceed the cap of 20.0", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        self.assertIsNone(self.s.ledger())
        self.assertEqual(self.s.spawn_log(), [])

    def test_the_cap_is_never_read_from_the_environment(self) -> None:
        env = {"S04_COMPUTE_CAP_A100_HOURS": "1000", "COMPUTE_CAP_A100_HOURS": "1000"}
        status = self.s.launcher("status", env=env)
        self.assertOk(status)
        self.assertTrue(status.stdout.startswith("Compute cap 20.0 A100-h: "), status.stdout)
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1, CFG2, CFG3, hours=7, env=env)
        self.assertRefused(proc, "cap")
        self.assertIn("cap of 20.0", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_refused_on_untracked_file(self) -> None:
        write(self.s.repo / STUDY / "scratch_note.txt", "not committed\n")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "clean-tree")
        self.assertIn("scratch_note.txt", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        self.assertEqual(self.s.spawn_log(), [])

    def test_refused_on_modified_or_staged_tracked_files(self) -> None:
        # The image uploads the working tree, so an uncommitted edit of the run code would run
        # on the GPU while the ledger records the committed version.
        (self.s.repo / RUN_SET_PY).write_text("def run(cfg, out_dir, ctx):\n    return {'edited': True}\n")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "clean-tree")
        self.assertIn("run_set.py", proc.stderr)
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "clean-tree")
        self.assertEqual(self.s.snapshot(), snap)
        self.s.git("checkout", "--", RUN_SET_PY)
        write(self.s.repo / STUDY / "staged_note.md", "staged, not committed\n")
        self.s.git("add", "--", f"{STUDY}/staged_note.md")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "clean-tree")
        self.assertIn("staged_note.md", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        self.assertEqual(self.s.spawn_log(), [])

    def test_refused_on_unpushed_commit(self) -> None:
        self.s.commit_file(f"{STUDY}/notes.md", "local only\n", push=False)
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "pushed")
        self.assertIn("ahead 1, behind 0", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        self.assertEqual(self.s.spawn_log(), [])

    def test_refused_when_behind_origin(self) -> None:
        self.s.push_from_other_clone(f"{STUDY}/notes.md", "from elsewhere\n")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "pushed")
        self.assertIn("ahead 0, behind 1", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_refused_off_main(self) -> None:
        self.s.git("checkout", "-q", "-b", "side")
        snap = self.s.snapshot()
        self.assertRefused(self.s.launch(CFG1), "branch")
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "branch")
        self.assertEqual(self.s.snapshot(), snap)

    def test_refused_when_stop_exists(self) -> None:
        self.s.commit_file(f"{STUDY}/STOP", "STOP\nReason: test\n")
        head = self.s.head()
        snap = self.s.snapshot()
        self.assertRefused(self.s.launch(CFG1), "stop")
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "stop")
        self.assertEqual(self.s.snapshot(), snap)
        self.assertEqual(self.s.head(), head)

    def test_spawn_failure_before_any_request_releases_the_reservation(self) -> None:
        (self.s.stub / "fail_spawn").touch()
        proc = self.s.launch(CFG1, CFG2, hours=3)
        self.assertExit(proc, 1)
        self.assertIn("LAUNCH FAILED L001", proc.stderr)
        self.assertIn("No call was sent to Modal", proc.stderr)
        subjects = self.s.subjects_since(self.base_head)
        self.assertEqual(len(subjects), 2)
        self.assertEqual(subjects[0], f"{PREFIX}: reserve L001 set A: 2 x 3 h = 6 A100-h")
        self.assertTrue(subjects[1].startswith(f"{PREFIX}: launch failed L001: RuntimeError: stub: fail_spawn"))
        self.assertLauncherCommits(self.base_head)
        # The reservation was on origin while Modal was called, and is released now.
        self.assertEqual(self.s.spawn_log()[0]["remote_ledger"]["launches"][0]["state"], "reserved")
        ledger = self.s.remote_ledger()
        self.assertEqual(ledger, self.s.ledger())
        launch = ledger["launches"][0]
        self.assertEqual(launch["state"], "launch_failed")
        self.assertIsNone(launch["app_id"])
        for run in launch["runs"]:
            self.assertEqual((run["status"], run["charged_hours"], run["charge_basis"]),
                             ("not_launched", 0.0, "none"))
        self.assertEqual(self.s.status(), "Compute cap 20.0 A100-h: used 0.00, reserved 0.00, "
                                          "left 20.00 (launches 1, runs in flight 0)")
        self.assertHistoryExtends()

    def test_app_created_without_any_call_is_a_failed_launch(self) -> None:
        # For example an image that fails to build: the app exists, but no call was ever
        # requested, so nothing can run on a GPU and the whole reservation is released.
        (self.s.stub / "fail_after_app").touch()
        proc = self.s.launch(CFG1, hours=3)
        self.assertExit(proc, 1)
        self.assertIn("LAUNCH FAILED L001", proc.stderr)
        launch = self.s.ledger()["launches"][0]
        self.assertEqual(launch["state"], "launch_failed")
        # The app id is kept out of app_id, so that a running app would count as unknown.
        self.assertIsNone(launch["app_id"])
        self.assertIn("ap-stub-1", launch["notes"])
        self.assertEqual(launch["runs"][0]["status"], "not_launched")
        self.assertIn("reserved 0.00", self.s.status())

    def test_a_spawn_that_raises_keeps_its_reservation(self) -> None:
        # The second request raises. Modal may have created that call, so its run keeps its
        # reservation as 'unknown'; only the third, never sent, is released.
        (self.s.stub / "fail_spawn_after").write_text("1")
        proc = self.s.launch(CFG1, CFG2, CFG3, hours=3)
        self.assertExit(proc, 3)
        self.assertIn("1 of 3 call(s) started", proc.stdout)
        self.assertIn("UNCERTAIN L001", proc.stderr)
        self.assertIn("NOT LAUNCHED", proc.stderr)
        self.assertNotIn("reservation is released (0 A100-h charged)", proc.stderr)
        launch = self.s.remote_ledger()["launches"][0]
        self.assertEqual((launch["state"], launch["app_id"]), ("launched", "ap-stub-1"))
        first, second, third = launch["runs"]
        self.assertEqual((first["status"], first["call_id"], first["charged_hours"]), ("running", "fc-stub-1-1", None))
        self.assertEqual((second["status"], second["call_id"], second["charged_hours"], second["reserved_hours"]),
                         ("unknown", None, None, 3.0))
        self.assertEqual((third["status"], third["call_id"], third["charged_hours"]), ("not_launched", None, 0.0))
        self.assertEqual(self.s.subjects_since(self.base_head)[-1],
                         f"{PREFIX}: launched L001 set A app ap-stub-1 (1 of 3 calls, 1 uncertain)")
        self.assertIn("reserved 6.00", self.s.status())
        self.assertLauncherCommits(self.base_head)
        self.assertHistoryExtends()
        # While the app runs, the uncertain run stays reserved.
        app = {"App ID": "ap-stub-1", "Description": "s04-loop-l001-a", "State": "ephemeral (detached)"}
        self.s.set_apps([app])
        self.assertOk(self.s.launcher("reconcile"))
        second = self.s.ledger()["launches"][0]["runs"][1]
        self.assertEqual((second["status"], second["charged_hours"]), ("running", None))
        # Once the app has stopped: run 1 is charged from its records, and run 2, which wrote
        # none (Modal never had it), is charged its full reservation.
        self.s.set_apps([dict(app, State="stopped")])
        self.s.set_calls({first["call_id"]: "ok"})
        self.s.record(first["volume_path"], "a1", "start", self.T0, first["run_id"])
        self.s.record(first["volume_path"], "a1", "end", self.T0 + 3600, first["run_id"], status="ok")
        self.assertOk(self.s.launcher("reconcile"))
        launch = self.s.ledger()["launches"][0]
        self.assertEqual(launch["state"], "closed")
        self.assertEqual([(r["status"], r["charged_hours"], r["charge_basis"]) for r in launch["runs"]],
                         [("finished", 1.0, "records"), ("unknown", 3.0, "full_reservation"),
                          ("not_launched", 0.0, "none")])
        self.assertIn("used 4.00, reserved 0.00", self.s.status())
        self.assertHistoryExtends()

    def test_a_call_created_before_its_spawn_raised_is_charged_from_its_records(self) -> None:
        (self.s.stub / "fail_spawn_after_start").write_text("1")
        self.assertExit(self.s.launch(CFG1, CFG2, hours=3), 3)
        self.assertEqual(self.s.server_calls(), ["fc-stub-1-1", "fc-stub-1-2"])  # Modal has both
        launch = self.s.ledger()["launches"][0]
        second = launch["runs"][1]
        self.assertEqual((second["status"], second["call_id"]), ("unknown", None))
        self.s.set_apps([{"App ID": "ap-stub-1", "Description": "s04-loop-l001-a", "State": "stopped"}])
        self.s.set_calls({launch["runs"][0]["call_id"]: "ok"})
        for run in launch["runs"]:
            self.s.record(run["volume_path"], "a1", "start", self.T0, run["run_id"])
            self.s.record(run["volume_path"], "a1", "end", self.T0 + 1800, run["run_id"], status="ok")
        self.assertOk(self.s.launcher("reconcile"))
        runs = self.s.ledger()["launches"][0]["runs"]
        self.assertEqual([(r["status"], r["charged_hours"], r["charge_basis"]) for r in runs],
                         [("finished", 0.5, "records"), ("finished", 0.5, "records")])

    def test_push_failure_leaves_ledger_and_history_untouched(self) -> None:
        hook = self.s.remote / "hooks" / "pre-receive"
        hook.write_text("#!/bin/sh\necho rejected-by-test-hook >&2\nexit 1\n")
        hook.chmod(0o755)
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "push")
        self.assertIn("rejected-by-test-hook", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)  # no ledger file, same HEAD, clean tree
        self.assertEqual(self.s.spawn_log(), [])
        # Again with a ledger that already exists.
        hook.unlink()
        self.launched(CFG1)
        hook.write_text("#!/bin/sh\nexit 1\n")
        hook.chmod(0o755)
        snap = self.s.snapshot()
        self.assertRefused(self.s.launch(CFG2), "push")
        self.assertEqual(self.s.snapshot(), snap)

    def test_hand_edited_ledger_fails_verify_and_blocks_writes(self) -> None:
        self.launched(CFG1)
        data = self.s.ledger()
        data["launches"][0]["runs"][0]["reserved_hours"] = 0.5
        self.s.commit_file(LEDGER, json.dumps(data, indent=2, sort_keys=True) + "\n", msg="hand edit")
        verify = self.s.launcher("verify")
        self.assertExit(verify, 1)
        self.assertIn("integrity", verify.stdout)
        snap = self.s.snapshot()
        self.assertRefused(self.s.launch(CFG2), "ledger")
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "ledger")
        self.assertRefused(self.s.launcher("reconcile"), "ledger")  # never re-stamps a hand edit
        self.assertEqual(self.s.snapshot(), snap)
        status = self.s.launcher("status")
        self.assertExit(status, 1)
        self.assertTrue(status.stdout.startswith("Compute cap 20.0 A100-h: "))
        self.assertIn("LEDGER PROBLEMS", status.stderr)

    def test_rolled_back_ledger_blocks_launch_and_reconcile(self) -> None:
        launch = self.launched(CFG1, CFG2, hours=3)
        launched_sha = self.s.head()
        self.s.set_calls({launch["runs"][0]["call_id"]: "failed"})
        self.assertOk(self.s.launcher("reconcile"))  # charges 3.0 h
        charged_ledger = (self.s.repo / LEDGER).read_text()
        self.assertHistoryExtends()
        # An old version has a valid integrity hash; committing it under the launcher's subject
        # would pass the integrity check and the runner's subject rule.
        self.s.git_as_loop("checkout", launched_sha, "--", LEDGER)
        self.s.git_as_loop("commit", "-q", "-m", f"{PREFIX}: reconcile tidy", "--", LEDGER)
        self.s.git("push", "-q", "origin", "main")
        self.assertOk(self.s.launcher("verify"))
        self.assertIn("used 0.00", self.s.status())  # the hours look free again
        snap = self.s.snapshot()
        proc = self.s.launch(CFG3, hours=1)
        self.assertRefused(proc, "ledger")
        self.assertIn("not a launcher-made successor", proc.stderr)
        self.assertIn("changed after it was charged", proc.stderr)
        self.assertRefused(self.s.launcher("reconcile"), "ledger")
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "ledger")
        self.assertEqual(self.s.snapshot(), snap)
        # Anjor's own commit is trusted: once he restores the ledger, launches work again.
        self.s.commit_file(LEDGER, charged_ledger, msg="Anjor restores the ledger")
        self.assertOk(self.s.launch(CFG3, hours=1))

    def test_a_rollback_split_over_two_loop_commits_is_refused(self) -> None:
        launch = self.launched(CFG1, CFG2, hours=3)
        reserved_sha = self.s.head() + "~1"  # the reservation commit
        launched_sha = self.s.head()
        self.s.set_calls({launch["runs"][0]["call_id"]: "failed"})
        self.assertOk(self.s.launcher("reconcile"))  # charges 3.0 h
        # Step 1 goes back to the reservation; step 2 moves to the launched version, a valid
        # successor of step 1. The newest pair alone looks fine.
        for sha, subject in ((reserved_sha, "reconcile tidy"), (launched_sha, "reconcile fix")):
            self.s.git_as_loop("checkout", sha, "--", LEDGER)
            self.s.git_as_loop("commit", "-q", "-m", f"{PREFIX}: {subject}", "--", LEDGER)
        self.s.git("push", "-q", "origin", "main")
        versions = self.s.ledger_versions()
        self.assertEqual(lc.ledger_extends(versions[-2], versions[-1]), [])
        self.assertIn("used 0.00", self.s.status())
        snap = self.s.snapshot()
        proc = self.s.launch(CFG3, hours=1)
        self.assertRefused(proc, "ledger")
        self.assertIn("not a launcher-made successor", proc.stderr)
        self.assertRefused(self.s.launcher("reconcile"), "ledger")
        self.assertEqual(self.s.snapshot(), snap)

    def test_loop_commits_are_known_by_their_committer_not_their_author(self) -> None:
        launch = self.launched(CFG1, hours=3)
        launched_sha = self.s.head()
        self.s.set_calls({launch["runs"][0]["call_id"]: "failed"})
        self.assertOk(self.s.launcher("reconcile"))
        charged_ledger = (self.s.repo / LEDGER).read_text()
        # A loop commit that claims Anjor as its author is still the loop's.
        self.s.git_as_loop("checkout", launched_sha, "--", LEDGER)
        self.s.git_as_loop("commit", "-q", "--author", "anjor <anjor@example.invalid>", "-m",
                           f"{PREFIX}: reconcile tidy", "--", LEDGER)
        self.s.git("push", "-q", "origin", "main")
        self.assertEqual(self.s.commits_since(launched_sha)[-1][1:3], ("anjor", LOOP_AUTHOR))
        proc = self.s.launch(CFG2, hours=1)
        self.assertRefused(proc, "ledger")
        self.assertIn(f"committer {LOOP_AUTHOR}", proc.stderr)
        # A commit Anjor made (his committer name) is trusted, whatever its author says.
        write(self.s.repo / LEDGER, charged_ledger)
        self.s.git("add", "--", LEDGER)
        self.s.git("commit", "-q", "--author", f"{LOOP_AUTHOR} <{LOOP_EMAIL}>", "-m", "Anjor restores the ledger")
        self.s.git("push", "-q", "origin", "main")
        self.assertOk(self.s.launch(CFG2, hours=1))

    def test_deleted_ledger_blocks_launches(self) -> None:
        self.launched(CFG1, hours=3)
        self.s.git_as_loop("rm", "-q", "--", LEDGER)
        self.s.git_as_loop("commit", "-q", "-m", f"{PREFIX}: reconcile cleanup")
        self.s.git("push", "-q", "origin", "main")
        self.assertIn("used 0.00, reserved 0.00", self.s.status())
        snap = self.s.snapshot()
        proc = self.s.launch(CFG2, hours=1)
        self.assertRefused(proc, "ledger")
        self.assertIn("deleted", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_loop_changes_to_the_loop_folder_block_launches(self) -> None:
        # A loop commit that raises the cap (or edits the GPU function) must not be used by
        # the launcher in the same session, before the runner can stop the loop.
        config = (self.s.repo / lc.CONFIG_REL).read_text()
        write(self.s.repo / lc.CONFIG_REL, config.replace(f"COMPUTE_CAP_A100_HOURS={CAP:g}",
                                                          "COMPUTE_CAP_A100_HOURS=1000"))
        self.s.git_as_loop("add", "--", lc.CONFIG_REL)
        self.s.git_as_loop("commit", "-q", "-m", "Study 04 [it-x]: tidy the settings")
        self.s.git("push", "-q", "origin", "main")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1, CFG2, CFG3, hours=7)
        self.assertRefused(proc, "guarded-files")
        self.assertIn("loop/config.env", proc.stderr)
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "guarded-files")
        self.assertEqual(self.s.snapshot(), snap)
        # Anjor reverts it with a commit of his own: launches work again, under the old cap.
        self.s.commit_file(lc.CONFIG_REL, config, msg="Anjor reverts the cap change")
        self.assertRefused(self.s.launch(CFG1, CFG2, CFG3, hours=7), "cap")
        self.assertOk(self.s.launch(CFG1, hours=1))

    def test_ledger_extends_rules(self) -> None:
        base = {"launches": [{"launch_id": "L001", "state": "launched", "kind": "gpu", "app_id": "ap-1",
                              "timeout_hours": 3.0, "runs": [
                                  {"run_id": "r1", "status": "running", "reserved_hours": 3.0,
                                   "charged_hours": None, "call_id": "fc-1", "config": "c.yaml"},
                                  {"run_id": "r2", "status": "finished", "reserved_hours": 3.0,
                                   "charged_hours": 1.0, "call_id": "fc-2", "config": "d.yaml"}]}]}

        def changed(fn: Any) -> List[str]:
            cur = json.loads(json.dumps(base))
            fn(cur)
            return lc.ledger_extends(base, cur)

        self.assertEqual(changed(lambda c: None), [])
        self.assertEqual(changed(lambda c: c["launches"].append({"launch_id": "L002", "runs": []})), [])
        self.assertEqual(changed(lambda c: c["launches"][0]["runs"][0].update(
            status="finished", charged_hours=2.0, charge_basis="records")), [])
        self.assertEqual(changed(lambda c: c["launches"][0].update(state="closed")), [])
        self.assertEqual(changed(lambda c: c["launches"][0]["runs"][0].update(reserved_hours=4.0)), [])
        for fn, needle in [
            (lambda c: c["launches"].pop(), "removed"),
            (lambda c: c["launches"][0].update(launch_id="L009"), "replaced"),
            (lambda c: c["launches"][0].update(timeout_hours=1.0), "timeout_hours changed"),
            (lambda c: c["launches"][0].update(app_id="ap-2"), "app_id changed"),
            (lambda c: c["launches"][0].update(state="reserved"), "state went from launched to reserved"),
            (lambda c: c["launches"][0]["runs"].pop(), "runs changed"),
            (lambda c: c["launches"][0]["runs"][0].update(config="e.yaml"), "config changed"),
            (lambda c: c["launches"][0]["runs"][0].update(call_id="fc-9"), "call_id changed"),
            (lambda c: c["launches"][0]["runs"][1].update(charged_hours=0.5), "changed after it was charged"),
            (lambda c: c["launches"][0]["runs"][1].update(status="failed"), "changed after it was charged"),
            (lambda c: c["launches"][0]["runs"][0].update(reserved_hours=1.0), "reservation lowered"),
        ]:
            problems = changed(fn)
            self.assertTrue(any(needle in p for p in problems), (needle, problems))
        # The launcher keeps no copy of these rules of its own.
        self.assertFalse(hasattr(import_launcher_module(), "ledger_extends"))

    def test_gate_report_rules(self) -> None:
        cases = [
            (f"{STUDY}/gate_reports/G3_fail.md", "the Result line is 'Result: FAIL'"),
            (f"{STUDY}/gate_reports/G3_lower.md", "the Result line is 'result: pass'"),
            (f"{STUDY}/gate_reports/G3_template.md", "unfilled template text"),
            (f"{STUDY}/gate_reports/G3_inconclusive.md", "the Critic line"),
            (f"{STUDY}/notes/G3_elsewhere.md", "is not under"),
            (f"{STUDY}/gate_reports/missing.md", "does not exist"),
            (f"{STUDY}/gate_reports/ignored_G3.md", "not committed"),
        ]
        snap = self.s.snapshot()
        for gate, needle in cases:
            with self.subTest(gate=gate):
                proc = self.s.launch(CFG1, gate=gate)
                self.assertRefused(proc, "gate-report")
                self.assertIn(needle, proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_every_gate_report_given_must_pass(self) -> None:
        # Set A depends on two reports (Gate 3 and the last base-state stage).
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1, gates=(GATE, f"{STUDY}/gate_reports/G3_fail.md"))
        self.assertRefused(proc, "gate-report")
        self.assertIn("G3_fail.md", proc.stderr)
        proc = self.s.launch(CFG1, gates=(f"{STUDY}/gate_reports/G3_template.md", GATE))
        self.assertRefused(proc, "gate-report")
        self.assertEqual(self.s.snapshot(), snap)
        self.assertOk(self.s.launch(CFG1, gates=(GATE, GATE_BASE, GATE)))
        self.assertEqual(self.s.ledger()["launches"][0]["gate_report"], [GATE, GATE_BASE])

    def test_local_test_must_be_committed(self) -> None:
        snap = self.s.snapshot()
        for local, needle in [(f"{STUDY}/local_tests/missing.log", "does not exist"),
                              (f"{STUDY}/configs/ignored_c9.yaml", "not committed")]:
            with self.subTest(local=local):
                proc = self.s.launch(CFG1, local=local)
                self.assertRefused(proc, "local-test")
                self.assertIn(needle, proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_config_rules(self) -> None:
        abs_entry = f"{STUDY}/configs/abs_entry.yaml"
        self.s.commit_file(abs_entry, f"entrypoint: {self.s.repo / RUN_SET_PY}\n", msg="absolute entrypoint")
        cases = [
            ((f"{STUDY}/elsewhere/setA_x.yaml",), "is not under"),
            ((f"{STUDY}/configs/../../../../etc/passwd.yaml",), "outside the repo"),
            ((f"{STUDY}/configs/ignored_c9.yaml",), "not committed"),
            ((f"{STUDY}/configs/missing.yaml",), "does not exist"),
            ((f"{STUDY}/configs/bad_entry.yaml",), "has no line 'def run('"),
            ((f"{STUDY}/configs/bad_yaml.yaml",), "does not parse as YAML"),
            ((f"{STUDY}/configs/no_entry.yaml",), "no string key 'entrypoint'"),
            ((f"{STUDY}/configs/outside_entry.yaml",), "is not a .py file under"),
            ((abs_entry,), "normalised repo-relative"),
            ((f"{STUDY}/configs/dot_entry.yaml",), "normalised repo-relative"),
            ((f"{STUDY}/configs/padded_entry.yaml",), "normalised repo-relative"),
            ((f"{STUDY}/RUNS.md",), "is not under"),
            ((CFG1, CFG1), "appears twice"),
            ((CFG1, f"{STUDY}/configs/sub/setA_c1.yaml"), "appears twice"),
        ]
        snap = self.s.snapshot()
        for configs, needle in cases:
            with self.subTest(configs=configs):
                proc = self.s.launch(*configs)
                self.assertRefused(proc, "config")
                self.assertIn(needle, proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_run_set_rules(self) -> None:
        snap = self.s.snapshot()
        for run_set in ("Z", "C", "a/b", "smoke", "a", "Base"):  # only the plan's three: base, A and B
            with self.subTest(run_set=run_set):
                proc = self.s.launch(CFG1, run_set=run_set, gates=(GATE,))
                self.assertRefused(proc, "set")
                self.assertIn("hard stop", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        # A planned set still needs its row in RUNS.md.
        runs_md = (self.s.repo / RUNS_MD).read_text()
        self.s.commit_file(RUNS_MD, runs_md.replace("| B | configs/setB", "| later B | configs/setB"),
                           msg="drop the set B row")
        snap = self.s.snapshot()
        self.assertRefused(self.s.launch(CFG1, run_set="B"), "runs-row")
        self.assertEqual(self.s.snapshot(), snap)
        self.s.commit_file(RUNS_MD, runs_md, msg="restore the set B row")
        self.assertOk(self.s.launch(CFG1, run_set="B"))
        self.assertEqual(self.s.ledger()["launches"][0]["app_name"], "s04-loop-l001-b")
        self.assertOk(self.s.launch(CFG2, run_set="base"))
        self.assertEqual(self.s.ledger()["launches"][1]["app_name"], "s04-loop-l002-base")

    def test_timeout_bounds(self) -> None:
        snap = self.s.snapshot()
        for hours in ("0", "-1", "24.5", "nan", "0.001"):
            with self.subTest(hours=hours):
                proc = self.s.launcher("launch", "--set", "A", f"--timeout-hours={hours}",
                                       "--gate-report", GATE, "--local-test", LOCAL, CFG1)
                self.assertRefused(proc, "timeout")
        self.assertEqual(self.s.snapshot(), snap)

    def test_freeze_records_hashes_and_a_later_change_blocks_launches(self) -> None:
        key_section = f"{STUDY}/SPEC.md#7a.A"
        key_file = RUN_SET_PY
        self.assertOk(self.s.launch(CFG1, extra=("--freeze", key_section, "--freeze", key_file)))
        frozen = self.s.ledger()["launches"][0]["frozen"]
        self.assertEqual(frozen, {key_section: lc.frozen_key_hash(self.s.repo, key_section),
                                  key_file: lc.sha256_file(self.s.repo / key_file)})
        spec = (self.s.repo / STUDY / "SPEC.md").read_text().replace("three standard errors", "two")
        self.s.commit_file(f"{STUDY}/SPEC.md", spec, msg="change a frozen criterion")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG2)
        self.assertRefused(proc, "frozen")
        self.assertIn("7a.A", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_freeze_key_must_resolve(self) -> None:
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1, extra=("--freeze", f"{STUDY}/SPEC.md#9z. Nope"))
        self.assertRefused(proc, "freeze")
        self.assertEqual(self.s.snapshot(), snap)

    def test_frozen_manifest_change_blocks_launches(self) -> None:
        key = f"{STUDY}/SPEC.md#7. Tolerances"
        manifest = {"note": "test", "entries": {key: lc.frozen_key_hash(self.s.repo, key)}}
        self.s.commit_file(lc.FROZEN_REL, json.dumps(manifest) + "\n", msg="record frozen sections")
        self.assertOk(self.s.launch(CFG1, hours=1))
        spec = (self.s.repo / STUDY / "SPEC.md").read_text().replace("within 1%", "within 2%")
        self.s.commit_file(f"{STUDY}/SPEC.md", spec, msg="loosen a tolerance")
        snap = self.s.snapshot()
        self.assertRefused(self.s.launch(CFG2, hours=1), "frozen")
        self.assertEqual(self.s.snapshot(), snap)

    def test_pyproject_pin_must_match_the_lock(self) -> None:
        self.s.commit_file("pyproject.toml", PYPROJECT.format(pin=OTHER_SHA), msg="pin elsewhere")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "gandalf-pin")
        self.assertIn("run uv lock", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        self.s.commit_file("pyproject.toml", PYPROJECT.format(pin=GANDALF_SHA), msg="pin to the lock")
        self.assertOk(self.s.launch(CFG1))

    def test_lock_without_gandalf_commit_is_refused(self) -> None:
        lock = UV_LOCK.replace(f"#{GANDALF_SHA}", "")
        self.s.commit_file("uv.lock", lock, msg="lock without a commit")
        snap = self.s.snapshot()
        self.assertRefused(self.s.launch(CFG1), "gandalf-pin")
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "gandalf-pin")
        self.assertEqual(self.s.snapshot(), snap)

    def test_second_launcher_in_the_same_clone_is_refused(self) -> None:
        fd = os.open(str(self.s.repo / ".git"), os.O_RDONLY)  # what the launcher locks
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            snap = self.s.snapshot()
            self.assertRefused(self.s.launch(CFG1), "lock")
            self.assertRefused(self.s.launcher("reconcile"), "lock")
            self.assertEqual(self.s.snapshot(), snap)
        finally:
            os.close(fd)
        self.assertOk(self.s.launch(CFG1))

    def test_launch_ids_and_run_ids_stay_unique(self) -> None:
        self.launched(CFG1, hours=1)
        self.launched(CFG1, hours=1)
        self.launched(CFG1, hours=1)
        ledger = self.s.ledger()
        self.assertEqual([l["launch_id"] for l in ledger["launches"]], ["L001", "L002", "L003"])
        run_ids = [r["run_id"] for _l, r in lc.iter_runs(ledger)]
        self.assertEqual(len(set(run_ids)), 3)
        self.assertEqual(lc.verify_ledger(ledger), [])


class GateReportLaunchTests(LauncherCase):
    """Which reports a launch of each run set rests on (LOOP.md section 9)."""

    def report(self, kind: str, iteration: str, result: str = "PASS", **kw: Any) -> str:
        """Commit a report of ``kind`` (G1, G3, Gbase or G4_<S>) for ``iteration``; returns its path."""
        name = f"{kind}_{iteration}.md"
        title = (f"# Gate 4 report, set {kind[3:]}: {iteration}" if kind.startswith("G4_")
                 else f"# Gate {kind[1:]} report: {iteration}")
        coded = "PASS" if result == "PASS" else "FAIL"
        rel = f"{STUDY}/gate_reports/{name}"
        self.s.commit_file(rel, gate_report(result=result, coded=coded, title=kw.pop("title", title), **kw),
                           msg=f"gate report {name}")
        return rel

    def test_each_set_needs_its_own_reports(self) -> None:
        snap = self.s.snapshot()
        cases = [
            ("B", (GATE,), "Missing: G4_A"),  # the verifier's case: set B on a Gate 3 report
            ("B", (GATE, GATE_BASE), "Missing: G4_A"),
            ("A", (GATE,), "Missing: Gbase"),
            ("A", (GATE_BASE,), "Missing: G3"),
            ("A", (GATE, GATE_BASE, GATE4_A), "Not for this set"),
            ("base", (GATE,), "Not for this set"),
            ("base", (GATE4_A,), "Missing: G1 or Gbase"),
        ]
        for run_set, gates, needle in cases:
            with self.subTest(run_set=run_set, gates=gates):
                proc = self.s.launch(CFG1, run_set=run_set, gates=gates)
                self.assertRefused(proc, "gate-report")
                self.assertIn(needle, proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        self.assertOk(self.s.launch(CFG1, run_set="B"))
        self.assertOk(self.s.launch(CFG2, run_set="base", gates=(GATE_BASE,)))  # a later base stage
        self.assertOk(self.s.launch(CFG3, run_set="A", gates=(GATE1, GATE, GATE_BASE)))  # earlier gates may come too
        self.assertEqual(self.s.ledger()["launches"][2]["gate_report"], [GATE1, GATE, GATE_BASE])

    def test_a_decided_gate_4_is_never_evaluated_again(self) -> None:
        # Set A's Gate 4 said FAIL (H0) first; a later PASS of the same gate does not count.
        failed = self.report("G4_A", "it-20261009-0900", result="FAIL")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1, run_set="B")
        self.assertRefused(proc, "gate-report")
        self.assertIn(failed, proc.stderr)
        self.assertIn("never repeated", proc.stderr)
        self.assertRefused(self.s.launch(CFG1, run_set="B", gates=(failed,)), "gate-report")
        self.assertEqual(self.s.snapshot(), snap)

    def test_a_not_decided_gate_4_evaluation_does_not_block(self) -> None:
        self.report("G4_A", "it-20261009-0900", result="NOT DECIDED")
        self.assertOk(self.s.launch(CFG1, run_set="B"))

    def test_the_newest_evaluation_of_a_quality_gate_stands(self) -> None:
        newer = self.report("G3", "it-20261006-0900", result="FAIL")
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1)  # the older passing Gate 3 report
        self.assertRefused(proc, "gate-report")
        self.assertIn(f"is not the newest G3 report: {newer} is newer", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)
        passed = self.report("G3", "it-20261007-0900")
        self.assertOk(self.s.launch(CFG1, gates=(passed, GATE_BASE)))

    def test_a_failed_base_stage_blocks_set_a_but_not_a_base_relaunch(self) -> None:
        self.report("Gbase", "it-20261008-0900", result="FAIL")  # the newest stage failed
        proc = self.s.launch(CFG1)
        self.assertRefused(proc, "gate-report")
        self.assertIn("is not the newest Gbase report", proc.stderr)
        # Relaunching the failed stage rests on the report on the stage before it.
        self.assertOk(self.s.launch(CFG1, run_set="base", gates=(GATE_BASE,)))

    def test_the_critic_report_must_back_the_critic_line(self) -> None:
        refuted = f"{STUDY}/gate_reports/critic_it-20261007-0900_1.md"
        self.s.commit_file(refuted, critic_report("VERDICT: REFUTED", "it-20261007-0900"), msg="critic")
        cases = [
            (dict(critic_file=refuted, critic_iter="it-20261007-0900"), "says 'VERDICT: REFUTED'"),
            (dict(critic_file=f"{STUDY}/gate_reports/critic_it-20261007-0900_2.md", critic_iter="it-20261007-0900"),
             "is not committed"),
            (dict(critic_file=f"{STUDY}/notes/critic_it-20261007-0900_1.md", critic_iter="it-20261007-0900"),
             "the Critic line names"),
            (dict(critic_file=CRITIC_REPORT, critic_iter="it-20261007-0900"), "the Critic line names"),
        ]
        for i, (kw, needle) in enumerate(cases):
            with self.subTest(case=needle):
                rel = self.report("G3", f"it-2026100{i}-2300", **kw)
                proc = self.s.launch(CFG1, gates=(rel, GATE_BASE))
                self.assertRefused(proc, "gate-report")
                self.assertIn(needle, proc.stderr)
        # The study-relative and bare spellings of the critic report both resolve.
        for i, name in enumerate((f"gate_reports/critic_{CRITIC_ITER}_1.md", f"critic_{CRITIC_ITER}_1.md")):
            rel = self.report("G3", f"it-2026101{i}-2300", critic_file=name)
            self.assertOk(self.s.launch(CFG1, gates=(rel, GATE_BASE), hours=1))

    def test_the_title_and_the_file_name_must_agree(self) -> None:
        wrong_title = self.report("G3", "it-20261006-0900", title="# Gate 1 report: it-20261006-0900")
        proc = self.s.launch(CFG1, gates=(wrong_title, GATE_BASE))
        self.assertRefused(proc, "gate-report")
        self.assertIn("its title is '# Gate 1 report: it-20261006-0900', which does not fit", proc.stderr)
        # A Gate 3 report copied in as set A's Gate 4 report.
        copied = f"{STUDY}/gate_reports/G4_A_it-20261011-0900.md"
        self.s.commit_file(copied, gate_report(title="# Gate 3 report: it-20261011-0900"), msg="copy")
        self.assertRefused(self.s.launch(CFG1, run_set="B", gates=(copied,)), "gate-report")
        # Other wording of the base-state gate's title is fine.
        base = self.report("Gbase", "it-20261009-0900", title="# Base-state gate report: it-20261009-0900")
        self.assertOk(self.s.launch(CFG1, run_set="base", gates=(base,), hours=1))
        for name in ("G3_fail.md", "Gate3_it-20261006-0900.md", "G3_it-20261006.md"):
            with self.subTest(name=name):
                rel = f"{STUDY}/gate_reports/{name}"
                self.s.commit_file(rel, gate_report(), msg=name)
                proc = self.s.launch(CFG1, gates=(rel, GATE_BASE))
                self.assertRefused(proc, "gate-report")
                self.assertIn("the file name is not", proc.stderr)


class HistoryTests(LauncherCase):
    """The ledger and loop/ checks walk the whole history: merges cannot hide a change."""

    def charge_the_cap(self) -> str:
        """Three 6 h runs, all failed and charged: 18 of the 20 A100-h used. Returns HEAD."""
        launch = self.launched(CFG1, CFG2, CFG3, hours=6)
        self.s.set_calls({r["call_id"]: "failed" for r in launch["runs"]})
        self.assertOk(self.s.launcher("reconcile"))
        self.assertIn("used 18.00, reserved 0.00, left 2.00", self.s.status())
        self.assertRefused(self.s.launch(CFG1, hours=6), "cap")
        return self.s.head()

    def merge_onto(self, first_parent: str, other: str, take_first_tree: bool = True) -> None:
        """As the loop: a merge commit with ``first_parent`` first, fast-forwarded onto main, pushed."""
        self.s.git_as_loop("checkout", "-q", "-b", "side", first_parent)
        if take_first_tree:
            self.s.git_as_loop("merge", "-q", "-s", "ours", "--no-edit", other)
        else:
            self.s.git_as_loop("merge", "-q", "--no-edit", other)
        self.s.git_as_loop("checkout", "-q", "main")
        self.s.git_as_loop("merge", "-q", "--ff-only", "side")
        self.s.git_as_loop("branch", "-q", "-D", "side")
        self.s.git("push", "-q", "origin", "main")

    def assertAllRefused(self, rule: str, needle: str) -> None:
        """launch, smoke and reconcile all refuse with ``rule``, and change nothing."""
        snap = self.s.snapshot()
        for args in (self.s.launch_args(CFG1, hours=6), ["smoke", "--sleep", "0"], ["reconcile"]):
            with self.subTest(command=args[0]):
                proc = self.s.launcher(*args)
                self.assertRefused(proc, rule)
                self.assertIn(needle, proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_a_merge_whose_first_parent_has_no_ledger_is_refused(self) -> None:
        # The verifier's case: a loop merge with the pre-ledger commit as first parent and the
        # charged main as second. git log --first-parent -- ledger sees nothing at all.
        charged = self.charge_the_cap()
        self.merge_onto(self.base_head, charged)
        self.assertFalse((self.s.repo / LEDGER).exists())
        self.assertIn("used 0.00, reserved 0.00, left 20.00", self.s.status())  # the hours look free
        self.assertEqual(self.s.git("log", "--first-parent", "--format=%H", "--", LEDGER).strip(), "")
        self.assertAllRefused("ledger", "deleted")
        self.assertEqual(len(self.s.spawn_log()), 1)  # only the first launch ever reached Modal

    def test_a_merge_whose_first_parent_has_an_older_ledger_is_refused(self) -> None:
        charged = self.charge_the_cap()
        reserved = self.s.git("rev-list", "--reverse", "HEAD").split()[1]  # the reservation commit
        self.assertEqual(self.s.commits_since(self.base_head)[0][0], reserved)
        self.merge_onto(reserved, charged)
        self.assertIn("used 0.00, reserved 18.00", self.s.status())
        self.assertAllRefused("ledger", "not a launcher-made successor")

    def test_a_merge_that_takes_the_old_ledger_from_its_second_parent_is_refused(self) -> None:
        charged = self.charge_the_cap()
        reserved = self.s.commits_since(self.base_head)[0][0]
        self.s.git_as_loop("checkout", "-q", "-b", "old", reserved)
        self.s.git_as_loop("commit", "-q", "--allow-empty", "-m", "Study 04 [it-x]: note")
        self.s.git_as_loop("checkout", "-q", "main")
        self.s.git_as_loop("merge", "-q", "--no-commit", "-s", "ours", "old")
        self.s.git_as_loop("checkout", "old", "--", LEDGER)
        self.s.git_as_loop("commit", "-q", "--no-edit")
        self.s.git("push", "-q", "origin", "main")
        self.assertEqual(len(self.s.git("rev-parse", "HEAD^@").split()), 2)
        self.assertAllRefused("ledger", "not a launcher-made successor")
        self.assertIn(charged[:12], self.s.launcher(*self.s.launch_args(CFG1, hours=6)).stderr)

    def test_a_hand_edit_laundered_by_a_later_loop_commit_is_refused(self) -> None:
        # A loop commit edits the ledger by hand (its hash goes stale); the next one re-stamps
        # the same content with a valid hash. HEAD alone looks sound.
        self.launched(CFG1, hours=2)
        data = self.s.ledger()
        data["launches"][0]["notes"] = "edited by hand"
        write(self.s.repo / LEDGER, json.dumps(data, indent=2, sort_keys=True) + "\n")
        self.s.git_as_loop("commit", "-q", "-am", f"{PREFIX}: reconcile note")
        lc.save_ledger(self.s.repo / LEDGER, data)
        self.s.git_as_loop("commit", "-q", "-am", f"{PREFIX}: reconcile stamp")
        self.s.git("push", "-q", "origin", "main")
        self.assertOk(self.s.launcher("verify"))
        self.assertAllRefused("ledger", "fails verify_ledger")

    def test_a_commit_github_made_cannot_reset_the_ledger(self) -> None:
        # The loop's gh login can squash-merge a pull request: GitHub is then the committer.
        charged = self.charge_the_cap()
        reserved = self.s.commits_since(self.base_head)[0][0]
        self.s.git_as_github("checkout", reserved, "--", LEDGER)
        self.s.git_as_github("commit", "-q", "-m", "Squash merge: tidy the ledger (#12)", "--", LEDGER)
        self.s.git("push", "-q", "origin", "main")
        self.assertAllRefused("ledger", "not a launcher-made successor")
        # Anjor's own commit does reset it.
        charged_text = self.s.git("show", f"{charged}:{LEDGER}")
        self.s.commit_file(LEDGER, charged_text, msg="Anjor restores the ledger")
        self.assertRefused(self.s.launch(CFG1, hours=6), "cap")
        self.assertOk(self.s.launch(CFG1, hours=2))

    def test_anjors_merges_are_not_refused(self) -> None:
        # Anjor pulls with a merge in his own checkout and pushes; a GitHub merge of his branch
        # lands too. Neither touches the ledger's content, and launches go on.
        self.launched(CFG1, hours=2)
        other = self.s.root / "anjor"
        run_cmd(["git", "clone", "-q", str(self.s.remote), str(other)], env=self.s.env)
        run_cmd(["git", "-C", str(other), "checkout", "-q", "-b", "local", "HEAD~2"], env=self.s.env)
        write(other / STUDY / "notes.md", "Anjor's notes\n")
        run_cmd(["git", "-C", str(other), "add", "-A"], env=self.s.env)
        run_cmd(["git", "-C", str(other), "commit", "-q", "-m", "Anjor: notes"], env=self.s.env)
        run_cmd(["git", "-C", str(other), "merge", "-q", "--no-edit", "origin/main"], env=self.s.env)
        run_cmd(["git", "-C", str(other), "push", "-q", "origin", "local:main"], env=self.s.env)
        self.s.git("pull", "-q", "--ff-only", "origin", "main")
        self.assertOk(self.s.launch(CFG2, hours=2))
        self.s.git("checkout", "-q", "-b", "pr", self.base_head)
        self.s.commit_file(f"{STUDY}/pr_notes.md", "from a pull request\n", msg="Anjor: PR work", push=False)
        self.s.git("checkout", "-q", "main")
        self.s.git_as_github("merge", "-q", "--no-ff", "--no-edit", "pr")
        self.s.git("push", "-q", "origin", "main")
        self.assertOk(self.s.launch(CFG3, hours=2))
        self.assertEqual(len(self.s.ledger()["launches"]), 3)

    def test_a_routine_pull_by_anjor_does_not_vouch_for_a_rollback(self) -> None:
        # The loop rolls the ledger back; Anjor, with a local commit, pulls with a merge and
        # pushes. That merge carries the rolled-back ledger but is not a reset he made.
        self.charge_the_cap()
        reserved = self.s.commits_since(self.base_head)[0][0]
        other = self.s.root / "anjor"
        run_cmd(["git", "clone", "-q", str(self.s.remote), str(other)], env=self.s.env)
        write(other / STUDY / "notes.md", "Anjor's notes\n")
        run_cmd(["git", "-C", str(other), "add", "-A"], env=self.s.env)
        run_cmd(["git", "-C", str(other), "commit", "-q", "-m", "Anjor: notes"], env=self.s.env)
        self.s.git_as_loop("checkout", reserved, "--", LEDGER)
        self.s.git_as_loop("commit", "-q", "-m", f"{PREFIX}: reconcile tidy", "--", LEDGER)
        self.s.git("push", "-q", "origin", "main")
        run_cmd(["git", "-C", str(other), "pull", "-q", "--no-rebase", "--no-edit", "origin", "main"], env=self.s.env)
        run_cmd(["git", "-C", str(other), "push", "-q", "origin", "main"], env=self.s.env)
        self.s.git("pull", "-q", "--ff-only", "origin", "main")
        self.assertEqual(self.s.git("log", "-1", "--format=%cn %P", "HEAD").split()[0], "anjor")
        self.assertEqual(len(self.s.git("log", "-1", "--format=%P", "HEAD").split()), 2)
        self.assertAllRefused("ledger", "not a launcher-made successor")

    def test_a_merge_cannot_hide_a_change_to_the_loop_folder(self) -> None:
        # A loop merge that raises the cap while it merges: git log --name-only prints no file
        # for a merge, so the change looked like Anjor's.
        config = (self.s.repo / lc.CONFIG_REL).read_text()
        self.s.git_as_loop("checkout", "-q", "-b", "side")
        self.s.git_as_loop("commit", "-q", "--allow-empty", "-m", "Study 04 [it-x]: side")
        self.s.git_as_loop("checkout", "-q", "main")
        self.s.git_as_loop("commit", "-q", "--allow-empty", "-m", "Study 04 [it-x]: main")
        self.s.git_as_loop("merge", "-q", "--no-commit", "--no-ff", "side")
        write(self.s.repo / lc.CONFIG_REL, config.replace(f"COMPUTE_CAP_A100_HOURS={CAP:g}",
                                                          "COMPUTE_CAP_A100_HOURS=1000"))
        self.s.git_as_loop("add", "--", lc.CONFIG_REL)
        self.s.git_as_loop("commit", "-q", "--no-edit")
        self.s.git("push", "-q", "origin", "main")
        names = self.s.git("log", "--no-renames", "--format=%H %cn", "--name-only", "--", lc.CONFIG_REL)
        self.assertEqual(names.split()[-3:], [self.base_head, "anjor", lc.CONFIG_REL])  # what the old check read
        snap = self.s.snapshot()
        proc = self.s.launch(CFG1, CFG2, CFG3, hours=7)
        self.assertRefused(proc, "guarded-files")
        self.assertIn(f"{lc.CONFIG_REL} (content from", proc.stderr)
        self.assertIn(f"(committer {LOOP_AUTHOR})", proc.stderr)
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "guarded-files")
        self.assertEqual(self.s.snapshot(), snap)

    def test_loop_folder_changes_github_made_are_refused_and_anjors_merged_ones_are_not(self) -> None:
        config = (self.s.repo / lc.CONFIG_REL).read_text()
        lower = config.replace(f"COMPUTE_CAP_A100_HOURS={CAP:g}", "COMPUTE_CAP_A100_HOURS=15")
        # A squash merge (one parent, GitHub the committer) that changes loop/.
        write(self.s.repo / lc.CONFIG_REL, lower)
        self.s.git_as_github("commit", "-q", "-am", "Lower the cap (#13)")
        self.s.git("push", "-q", "origin", "main")
        self.assertRefused(self.s.launch(CFG1, hours=1), "guarded-files")
        # A merge commit of Anjor's own branch: the content comes from his commit.
        self.s.commit_file(lc.CONFIG_REL, config, msg="Anjor reverts")
        self.s.git("checkout", "-q", "-b", "cap", self.s.head())
        self.s.commit_file(lc.CONFIG_REL, lower, msg="Anjor: lower the cap", push=False)
        self.s.git("checkout", "-q", "main")
        self.s.git_as_github("merge", "-q", "--no-ff", "--no-edit", "cap")
        self.s.git("push", "-q", "origin", "main")
        self.assertRefused(self.s.launch(CFG1, hours=16), "cap")  # the new cap of 15 holds
        self.assertOk(self.s.launch(CFG1, hours=1))

    def test_a_deleted_loop_file_counts_but_a_loop_files_own_round_trip_does_not(self) -> None:
        # The loop deletes the frozen manifest Anjor committed: refused until he commits it again.
        key = f"{STUDY}/SPEC.md#7. Tolerances"
        manifest = json.dumps({"note": "test", "entries": {key: lc.frozen_key_hash(self.s.repo, key)}}) + "\n"
        self.s.commit_file(lc.FROZEN_REL, manifest, msg="record frozen sections")
        self.s.git_as_loop("rm", "-q", "--", lc.FROZEN_REL)
        self.s.git_as_loop("commit", "-q", "-m", "Study 04 [it-x]: tidy")
        self.s.git("push", "-q", "origin", "main")
        proc = self.s.launch(CFG1, hours=1)
        self.assertRefused(proc, "guarded-files")
        self.assertIn(f"{lc.FROZEN_REL} (deleted from", proc.stderr)
        self.s.commit_file(lc.FROZEN_REL, manifest, msg="Anjor restores the manifest")
        self.assertOk(self.s.launch(CFG1, hours=1))
        # A file only the loop ever had, added and removed again, changes nothing.
        self.s.git_as_loop("commit", "-q", "--allow-empty", "-m", "noop")
        write(self.s.repo / STUDY / "loop" / "scratch.txt", "x\n")
        self.s.git_as_loop("add", "--", f"{STUDY}/loop/scratch.txt")
        self.s.git_as_loop("commit", "-q", "-m", "Study 04 [it-x]: add")
        self.s.git_as_loop("rm", "-q", "--", f"{STUDY}/loop/scratch.txt")
        self.s.git_as_loop("commit", "-q", "-m", "Study 04 [it-x]: remove")
        self.s.git("push", "-q", "origin", "main")
        self.assertOk(self.s.launch(CFG2, hours=1))

    def test_odd_file_names_under_loop_are_checked_too(self) -> None:
        # git quotes such names in its usual output; the check reads them raw.
        odd = f"{STUDY}/loop/café \"q\".py"
        write(self.s.repo / odd, "X = 1\n")
        self.s.git_as_loop("add", "--", odd)
        self.s.git_as_loop("commit", "-q", "-m", "Study 04 [it-x]: helper")
        self.s.git("push", "-q", "origin", "main")
        proc = self.s.launch(CFG1, hours=1)
        self.assertRefused(proc, "guarded-files")
        self.assertIn("café", proc.stderr)
        self.s.commit_file(odd, "X = 2\n", msg="Anjor takes the helper over")
        self.assertOk(self.s.launch(CFG1, hours=1))
        broken = f"{STUDY}/loop/two\nlines.txt"
        write(self.s.repo / broken, "x\n")
        self.s.git_as_loop("add", "--", broken)
        self.s.git_as_loop("commit", "-q", "-m", "Study 04 [it-x]: odd")
        self.s.git("push", "-q", "origin", "main")
        proc = self.s.launch(CFG2, hours=1)
        self.assertRefused(proc, "guarded-files")
        self.assertIn("line break", proc.stderr)

    def test_grafts_are_refused(self) -> None:
        self.launched(CFG1, hours=1)
        grafts = self.s.repo / ".git" / "info" / "grafts"
        grafts.parent.mkdir(parents=True, exist_ok=True)
        grafts.write_text(self.s.head() + "\n")  # would make HEAD look like a root commit
        snap = self.s.snapshot()
        proc = self.s.launch(CFG2, hours=1)
        self.assertRefused(proc, "history")
        self.assertIn("grafts", proc.stderr)
        self.assertRefused(self.s.launcher("reconcile"), "history")
        self.assertEqual(self.s.snapshot(), snap)


class SignalTests(LauncherCase):
    """The runner stops a session with SIGTERM; a launch must still leave a true record."""

    def test_signal_during_the_reservation_push_starts_nothing(self) -> None:
        marker = self.s.root / "push-started"
        hook = self.s.remote / "hooks" / "pre-receive"
        hook.write_text(f"#!/bin/sh\nif [ ! -e '{marker}' ]; then touch '{marker}'; sleep 3; fi\nexit 0\n")
        hook.chmod(0o755)
        proc = self.s.launcher_popen(*self.s.launch_args(CFG1, CFG2, hours=3))
        try:
            wait_for(marker)
            os.kill(proc.pid, signal.SIGTERM)
            out, err = proc.communicate(timeout=120)
        finally:
            if proc.poll() is None:
                proc.kill()
        self.assertEqual(proc.returncode, 130, f"\nstdout:\n{out}\nstderr:\n{err}")
        self.assertEqual(self.s.spawn_log(), [])  # Modal was never called
        self.assertEqual(self.s.subjects_since(self.base_head), [
            f"{PREFIX}: reserve L001 set A: 2 x 3 h = 6 A100-h",
            f"{PREFIX}: launch failed L001: interrupted by a signal before Modal was called"])
        launch = self.s.remote_ledger()["launches"][0]
        self.assertEqual(launch["state"], "launch_failed")
        self.assertEqual([(r["status"], r["charged_hours"]) for r in launch["runs"]],
                         [("not_launched", 0.0), ("not_launched", 0.0)])
        self.assertEqual(self.s.head(), self.s.remote_head())
        self.assertEqual(self.s.porcelain(), "")
        self.assertLauncherCommits(self.base_head)
        self.assertHistoryExtends()

    def test_signal_during_a_spawn_keeps_that_run_reserved(self) -> None:
        (self.s.stub / "sigterm_during_spawn").write_text("2")
        proc = self.s.launch(CFG1, CFG2, CFG3, hours=3)
        self.assertExit(proc, 130)
        self.assertEqual(self.s.server_calls(), ["fc-stub-1-1", "fc-stub-1-2"])  # Modal has call 2
        launch = self.s.remote_ledger()["launches"][0]
        self.assertEqual((launch["state"], launch["app_id"]), ("launched", "ap-stub-1"))
        self.assertEqual([(r["status"], r["call_id"], r["charged_hours"]) for r in launch["runs"]],
                         [("running", "fc-stub-1-1", None), ("unknown", None, None),
                          ("not_launched", None, 0.0)])
        self.assertIn("reserved 6.00", self.s.status())
        self.assertEqual(self.s.porcelain(), "")
        self.assertLauncherCommits(self.base_head)
        self.assertHistoryExtends()


class BackendChoiceTests(LauncherCase):

    def test_stub_backend_is_refused_with_a_github_origin(self) -> None:
        snap = self.s.snapshot()
        self.s.git("remote", "set-url", "origin", "https://github.com/anjor/krmhd-research.git")
        proc = self.s.launch(CFG1)
        self.assertExit(proc, 2)
        self.assertIn("S04_MODAL_BACKEND=stub is for the tests only", proc.stderr)
        self.assertExit(self.s.launcher("smoke", "--sleep", "0"), 2)
        self.assertExit(self.s.launcher("reconcile"), 2)
        # A push URL on GitHub is refused too, even with a local fetch URL.
        self.s.git("remote", "set-url", "origin", str(self.s.remote))
        self.s.git("remote", "set-url", "--push", "origin", "git@github.com:anjor/krmhd-research.git")
        self.assertExit(self.s.launch(CFG1), 2)
        # So is an origin that is not a local repository.
        self.s.git("config", "--unset", "remote.origin.pushurl")
        self.s.git("remote", "set-url", "origin", str(self.s.root / "no-such-remote.git"))
        self.assertExit(self.s.launch(CFG1), 2)
        self.assertEqual(self.s.spawn_log(), [])
        self.assertEqual((self.s.head(), self.s.remote_head(), self.s.ledger()), snap[:2] + (None,))

    def test_the_default_backend_is_modal(self) -> None:
        ml = import_launcher_module()
        settings = ml.load_settings(self.s.repo)
        env = {k: v for k, v in os.environ.items() if k != "S04_MODAL_BACKEND"}
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertIsInstance(ml.make_backend(self.s.repo, settings), ml.ModalBackend)
        with mock.patch.dict(os.environ, {"S04_MODAL_BACKEND": " "}):
            self.assertIsInstance(ml.make_backend(self.s.repo, settings), ml.ModalBackend)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


class SmokeTests(LauncherCase):

    def test_smoke_takes_the_same_path_and_charges_zero(self) -> None:
        proc = self.s.launcher("smoke", "--sleep", "0")
        self.assertOk(proc)
        log = self.s.spawn_log()
        self.assertEqual((log[0]["kind"], log[0]["sleep_s"], log[0]["timeout_s"]), ("smoke", 0, 900))
        self.assertEqual(log[0]["remote_ledger"]["launches"][0]["state"], "reserved")
        launch = self.s.ledger()["launches"][0]
        self.assertEqual((launch["kind"], launch["run_set"], launch["timeout_hours"]), ("smoke", "smoke", 0.25))
        self.assertEqual((launch["gate_report"], launch["local_test"]), (None, None))
        self.assertEqual(launch["app_name"], "s04-loop-l001-smoke")
        run = launch["runs"][0]
        self.assertRegex(run["run_id"], r"^04_smoke_\d{8}_\d{6}$")
        self.assertEqual(run["volume_path"], f"/study04/smoke/{run['run_id']}")
        self.assertEqual(run["reserved_hours"], 0.0)
        self.assertEqual(self.s.subjects_since(self.base_head),
                         [f"{PREFIX}: reserve L001 smoke: 0 A100-h",
                          f"{PREFIX}: launched L001 set smoke app ap-stub-1"])
        self.assertEqual(self.s.status(), "Compute cap 20.0 A100-h: used 0.00, reserved 0.00, "
                                          "left 20.00 (launches 1, runs in flight 1)")
        # The call finishes after two minutes: recorded, charged nothing.
        self.s.set_calls({run["call_id"]: "ok"})
        self.s.record(run["volume_path"], "s1", "start", self.T0, run["run_id"])
        self.s.record(run["volume_path"], "s1", "end", self.T0 + 120, run["run_id"], status="ok")
        self.assertOk(self.s.launcher("reconcile"))
        launch = self.s.ledger()["launches"][0]
        run = launch["runs"][0]
        self.assertEqual((run["status"], run["charged_hours"], run["charge_basis"]), ("finished", 0.0, "none"))
        self.assertEqual(len(run["attempts"]), 1)
        self.assertAlmostEqual(run["attempts"][0]["hours"], 120 / 3600, places=4)
        self.assertEqual(launch["state"], "closed")
        self.assertEqual(self.s.status(), "Compute cap 20.0 A100-h: used 0.00, reserved 0.00, "
                                          "left 20.00 (launches 1, runs in flight 0)")
        self.assertLauncherCommits(self.base_head)
        self.assertHistoryExtends()

    def test_smoke_sleep_bounds(self) -> None:
        snap = self.s.snapshot()
        self.assertRefused(self.s.launcher("smoke", "--sleep", "601"), "sleep")
        self.assertRefused(self.s.launcher("smoke", "--sleep", "-1"), "sleep")
        self.assertEqual(self.s.snapshot(), snap)


# ---------------------------------------------------------------------------
# reconcile
# ---------------------------------------------------------------------------


class ReconcileTests(LauncherCase):

    def test_complete_records_charge_their_duration(self) -> None:
        run = self.launched(CFG1, hours=3)["runs"][0]
        rid = run["run_id"]
        self.s.set_calls({run["call_id"]: "ok"})
        self.s.record(run["volume_path"], "a1", "start", self.T0, rid)
        self.s.record(run["volume_path"], "a1", "end", self.T0 + 4500, rid, status="ok")
        head = self.s.head()
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertIn(f"{rid}: finished, charged 1.25 A100-h (records; 1 attempt(s))", proc.stdout)
        launch = self.s.ledger()["launches"][0]
        run = launch["runs"][0]
        self.assertEqual((run["status"], run["charged_hours"], run["charge_basis"]), ("finished", 1.25, "records"))
        self.assertEqual(run["attempts"], [{"attempt_id": "a1", "start_utc": lc_utc(self.T0),
                                            "end_utc": lc_utc(self.T0 + 4500), "hours": 1.25, "status": "ok"}])
        self.assertEqual(launch["state"], "closed")
        self.assertEqual(self.s.subjects_since(head), [f"{PREFIX}: reconcile {rid}"])
        self.assertLauncherCommits(head)
        self.assertEqual(self.s.head(), self.s.remote_head())
        self.assertEqual(self.s.status(), "Compute cap 20.0 A100-h: used 1.25, reserved 0.00, "
                                          "left 18.75 (launches 1, runs in flight 0)")
        # Idempotent: a second reconcile changes nothing.
        head = self.s.head()
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertIn("No ledger change.", proc.stdout)
        self.assertEqual(self.s.head(), head)

    def test_a_launch_closes_only_when_all_its_runs_are_charged(self) -> None:
        launch = self.launched(CFG1, CFG2, hours=3)
        r1, r2 = launch["runs"]
        self.s.set_calls({r1["call_id"]: "ok"})
        self.s.record(r1["volume_path"], "a1", "start", self.T0, r1["run_id"])
        self.s.record(r1["volume_path"], "a1", "end", self.T0 + 3600, r1["run_id"], status="ok")
        self.s.record(r2["volume_path"], "b1", "start", time.time() - 600, r2["run_id"])
        self.assertOk(self.s.launcher("reconcile"))
        launch = self.s.ledger()["launches"][0]
        self.assertEqual(launch["state"], "launched")
        self.assertEqual([(r["status"], r["charged_hours"]) for r in launch["runs"]],
                         [("finished", 1.0), ("running", None)])
        self.assertIn("used 1.00, reserved 3.00", self.s.status())
        # The second run finishes later; the next reconcile charges it and closes the launch.
        self.s.set_calls({r1["call_id"]: "ok", r2["call_id"]: "failed"})
        self.s.record(r2["volume_path"], "b1", "end", time.time(), r2["run_id"], status="error")
        self.assertOk(self.s.launcher("reconcile"))
        launch = self.s.ledger()["launches"][0]
        self.assertEqual(launch["state"], "closed")
        self.assertEqual(launch["runs"][1]["status"], "failed")
        self.assertAlmostEqual(launch["runs"][1]["charged_hours"], 600 / 3600, places=2)
        self.assertLauncherCommits(self.base_head)
        self.assertHistoryExtends()

    def test_reconcile_commits_only_the_ledger_on_a_dirty_tree(self) -> None:
        run = self.launched(CFG1, hours=3)["runs"][0]
        self.s.set_calls({run["call_id"]: "failed"})
        runs_md = self.s.repo / STUDY / "RUNS.md"
        runs_md.write_text(runs_md.read_text() + "| note | work in progress |\n")
        write(self.s.repo / STUDY / "notes_staged.md", "staged\n")
        self.s.git("add", "--", f"{STUDY}/notes_staged.md")
        before = self.s.porcelain()
        head = self.s.head()
        self.assertOk(self.s.launcher("reconcile"))
        commits = self.s.commits_since(head)
        self.assertEqual(len(commits), 1)
        self.assertEqual(self.s.files_of(commits[0][0]), [LEDGER])
        self.assertEqual(self.s.porcelain(), before)  # the other changes are as they were
        self.assertEqual(self.s.head(), self.s.remote_head())
        self.assertEqual(self.s.ledger()["launches"][0]["runs"][0]["charged_hours"], 3.0)

    def test_no_records_charge_the_full_reservation(self) -> None:
        run = self.launched(CFG1, hours=3)["runs"][0]
        self.s.set_calls({run["call_id"]: "failed"})
        self.assertOk(self.s.launcher("reconcile"))
        run = self.s.ledger()["launches"][0]["runs"][0]
        self.assertEqual((run["status"], run["charged_hours"], run["charge_basis"]),
                         ("failed", 3.0, "full_reservation"))
        self.assertEqual(run["attempts"], [])
        self.assertIn("used 3.00, reserved 0.00", self.s.status())

    def test_an_open_attempt_is_charged_up_to_the_end_of_the_budget(self) -> None:
        # Attempt a2 has no end record. modal_app ends every attempt by the first start plus
        # the timeout (3 h), so a2 ran at most from T0+2000 s to T0+10800 s.
        run = self.launched(CFG1, hours=3)["runs"][0]
        rid = run["run_id"]
        self.s.set_calls({run["call_id"]: "timeout"})
        self.s.record(run["volume_path"], "a1", "start", self.T0, rid)
        self.s.record(run["volume_path"], "a1", "end", self.T0 + 1800, rid, status="error")
        self.s.record(run["volume_path"], "a2", "start", self.T0 + 2000, rid)
        self.assertOk(self.s.launcher("reconcile"))
        run = self.s.ledger()["launches"][0]["runs"][0]
        self.assertEqual((run["status"], run["charge_basis"]), ("failed", "records+budget"))
        self.assertAlmostEqual(run["charged_hours"], 0.5 + 8800 / 3600, places=4)
        self.assertEqual([(a["attempt_id"], a["hours"]) for a in run["attempts"]], [("a1", 0.5), ("a2", None)])

    def test_a_run_is_charged_at_most_its_budget_when_records_are_missing(self) -> None:
        # Two open attempts (each could have run to the budget's end): still at most 3 h.
        run = self.launched(CFG1, hours=3)["runs"][0]
        rid = run["run_id"]
        self.s.set_calls({run["call_id"]: "failed"})
        self.s.record(run["volume_path"], "a1", "start", self.T0, rid)
        self.s.record(run["volume_path"], "a2", "start", self.T0 + 600, rid)
        self.assertOk(self.s.launcher("reconcile"))
        run = self.s.ledger()["launches"][0]["runs"][0]
        self.assertEqual((run["status"], run["charged_hours"], run["charge_basis"]), ("failed", 3.0, "records+budget"))

    def test_records_beyond_the_budget_are_charged_in_full(self) -> None:
        # The complete attempts add up to more than the timeout, so the budget was not kept:
        # the facts are charged, and each open attempt a full timeout.
        run = self.launched(CFG1, hours=3)["runs"][0]
        rid = run["run_id"]
        self.s.set_calls({run["call_id"]: "failed"})
        self.s.record(run["volume_path"], "a1", "start", self.T0, rid)
        self.s.record(run["volume_path"], "a1", "end", self.T0 + 2.5 * 3600, rid, status="interrupted")
        self.s.record(run["volume_path"], "a2", "start", self.T0 + 2.6 * 3600, rid)
        self.s.record(run["volume_path"], "a2", "end", self.T0 + 4.6 * 3600, rid, status="interrupted")
        self.s.record(run["volume_path"], "a3", "start", self.T0 + 4.7 * 3600, rid)
        self.assertOk(self.s.launcher("reconcile"))
        run = self.s.ledger()["launches"][0]["runs"][0]
        self.assertEqual((run["charged_hours"], run["charge_basis"]), (7.5, "records+full_timeout"))

    def test_running_call_stays_reserved(self) -> None:
        run = self.launched(CFG1, hours=3)["runs"][0]
        self.s.record(run["volume_path"], "a1", "start", time.time() - 1800, run["run_id"])
        head = self.s.head()
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertIn(f"{run['run_id']}: running, 0.50 h since the latest attempt started", proc.stdout)
        self.assertIn("No ledger change.", proc.stdout)
        self.assertEqual(self.s.head(), head)
        run = self.s.ledger()["launches"][0]["runs"][0]
        self.assertEqual((run["status"], run["charged_hours"]), ("running", None))
        self.assertIn("used 0.00, reserved 3.00", self.s.status())

    def test_call_unfinished_long_past_its_timeout_is_flagged(self) -> None:
        run = self.launched(CFG1, hours=3)["runs"][0]
        self.s.record(run["volume_path"], "a1", "start", time.time() - 5 * 3600, run["run_id"])
        head = self.s.head()
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertIn("WARNING:", proc.stdout)
        self.assertIn("may be queued for a GPU or stuck", proc.stdout)
        self.assertEqual(self.s.head(), head)  # still reserved, nothing charged
        self.assertIn("reserved 3.00", self.s.status())

    def test_a_restart_shares_the_budget_and_raises_nothing(self) -> None:
        # modal_app counts the budget from the first attempt, so a restart cannot spend a
        # second timeout: the reservation stays at one timeout.
        run = self.launched(CFG1, hours=3)["runs"][0]
        rid = run["run_id"]
        self.s.record(run["volume_path"], "a1", "start", time.time() - 2 * 3600, rid)
        self.s.record(run["volume_path"], "a1", "end", time.time() - 3600, rid, status="interrupted")
        self.s.record(run["volume_path"], "a2", "start", time.time() - 3000, rid)
        head = self.s.head()
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertNotIn("raised", proc.stdout)
        self.assertIn("No ledger change.", proc.stdout)
        self.assertEqual(self.s.head(), head)
        run = self.s.ledger()["launches"][0]["runs"][0]
        self.assertEqual((run["reserved_hours"], run["charged_hours"], run["status"]), (3.0, None, "running"))
        # A restart that has not started yet holds the same.
        self.s.record(run["volume_path"], "a2", "end", time.time() - 60, rid, status="interrupted")
        self.assertOk(self.s.launcher("reconcile"))
        self.assertEqual(self.s.ledger()["launches"][0]["runs"][0]["reserved_hours"], 3.0)

    def test_preempted_runs_near_the_cap_do_not_stop_the_loop(self) -> None:
        # Three 5 h runs with one timeout of headroom under the cap of 20 (LOOP.md asks for
        # that); two are preempted at 3 h and restarted. They can spend at most 15 h.
        launch = self.launched(CFG1, CFG2, CFG3, hours=5)
        now = time.time()
        for run in launch["runs"][:2]:
            first = now - 3.5 * 3600
            self.s.record(run["volume_path"], "a1", "start", first, run["run_id"])
            self.s.record(run["volume_path"], "a1", "end", first + 3 * 3600, run["run_id"], status="interrupted")
            self.s.record(run["volume_path"], "a2", "start", first + 3.1 * 3600, run["run_id"])
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertNotIn("CAP EXCEEDED", proc.stderr)
        self.assertIn("used 0.00, reserved 15.00, left 5.00", self.s.status())

    def test_hours_over_the_cap_are_reported_as_a_hard_stop(self) -> None:
        # Records that show more GPU time than the budget (it was not kept) are held in full.
        run = self.launched(CFG1, hours=20)["runs"][0]  # exactly the cap
        rid = run["run_id"]
        self.s.record(run["volume_path"], "a1", "start", self.T0, rid)
        self.s.record(run["volume_path"], "a1", "end", self.T0 + 12 * 3600, rid, status="interrupted")
        self.s.record(run["volume_path"], "a2", "start", self.T0 + 12.1 * 3600, rid)
        self.s.record(run["volume_path"], "a2", "end", self.T0 + 21.6 * 3600, rid, status="interrupted")
        self.s.record(run["volume_path"], "a3", "start", self.T0 + 21.7 * 3600, rid)
        proc = self.s.launcher("reconcile")
        self.assertExit(proc, 4)
        self.assertIn("CAP EXCEEDED", proc.stderr)
        self.assertIn("more than its budget of 20 h; reservation raised to 41.50 A100-h", proc.stdout)
        self.assertEqual(self.s.head(), self.s.remote_head())  # the facts are committed anyway
        self.assertEqual(self.s.remote_ledger()["launches"][0]["runs"][0]["reserved_hours"], 41.5)
        status = self.s.launcher("status")
        self.assertExit(status, 4)
        self.assertEqual(status.stdout.splitlines()[0], "Compute cap 20.0 A100-h: used 0.00, reserved 41.50, "
                                                        "left -21.50 (launches 1, runs in flight 1)")
        self.assertIn("CAP EXCEEDED", status.stderr)
        self.assertRefused(self.s.launcher("smoke", "--sleep", "0"), "cap")

    def test_expired_call_takes_its_status_from_the_records(self) -> None:
        launch = self.launched(CFG1, CFG2, CFG3, hours=3)
        good, bad, cut = launch["runs"]
        self.s.set_calls({good["call_id"]: "expired", bad["call_id"]: "expired", cut["call_id"]: "expired"})
        for run, status in ((good, "ok"), (bad, "error"), (cut, "interrupted")):
            self.s.record(run["volume_path"], "x", "start", self.T0, run["run_id"])
            self.s.record(run["volume_path"], "x", "end", self.T0 + 900, run["run_id"], status=status)
        self.assertOk(self.s.launcher("reconcile"))
        good, bad, cut = self.s.ledger()["launches"][0]["runs"]
        self.assertEqual((good["status"], good["charged_hours"]), ("finished", 0.25))
        self.assertEqual((bad["status"], bad["charged_hours"]), ("failed", 0.25))
        self.assertEqual((cut["status"], cut["charged_hours"]), ("failed", 0.25))

    def test_query_errors_change_nothing(self) -> None:
        run = self.launched(CFG1, hours=3)["runs"][0]
        snap = self.s.snapshot()
        self.s.set_calls({run["call_id"]: "error"})
        proc = self.s.launcher("reconcile")
        self.assertExit(proc, 2)
        self.assertIn("unchanged", proc.stdout)
        self.assertEqual(self.s.snapshot(), snap)
        self.s.set_calls({run["call_id"]: "ok"})
        (self.s.stub / "fail_read").touch()
        proc = self.s.launcher("reconcile")
        self.assertExit(proc, 2)
        self.assertIn("cannot read the attempt records", proc.stdout)
        self.assertEqual(self.s.snapshot(), snap)

    def test_uncommitted_ledger_change_is_refused(self) -> None:
        self.launched(CFG1)
        data = self.s.ledger()
        data["launches"][0]["notes"] = "edited outside the launcher"
        lc.save_ledger(self.s.repo / LEDGER, data)  # a valid hash, but not committed
        snap = self.s.snapshot()
        proc = self.s.launcher("reconcile")
        self.assertRefused(proc, "ledger")
        self.assertIn("uncommitted changes", proc.stderr)
        self.assertEqual(self.s.snapshot(), snap)

    def test_refused_when_behind_origin(self) -> None:
        self.launched(CFG1)
        self.s.push_from_other_clone(f"{STUDY}/notes.md", "from elsewhere\n")
        snap = self.s.snapshot()
        self.assertRefused(self.s.launcher("reconcile"), "pull")
        self.assertEqual(self.s.snapshot(), snap)

    def test_reconcile_one_launch(self) -> None:
        first = self.launched(CFG1, hours=1)["runs"][0]
        second = self.launched(CFG2, hours=1)["runs"][0]
        self.s.set_calls({first["call_id"]: "failed", second["call_id"]: "failed"})
        self.assertRefused(self.s.launcher("reconcile", "--launch", "L999"), "launch")
        self.assertOk(self.s.launcher("reconcile", "--launch", "L002"))
        runs = [r for _l, r in lc.iter_runs(self.s.ledger())]
        self.assertEqual([r["charged_hours"] for r in runs], [None, 1.0])

    def test_interrupted_launch_is_adopted_by_app_name(self) -> None:
        (self.s.stub / "die_after_reserve").touch()
        proc = self.s.launch(CFG1, hours=2)
        self.assertExit(proc, 9)  # killed after the reservation push
        (self.s.stub / "die_after_reserve").unlink()
        self.assertEqual(self.s.head(), self.s.remote_head())
        self.assertEqual(self.s.porcelain(), "")
        launch = self.s.remote_ledger()["launches"][0]
        self.assertEqual((launch["state"], launch["app_id"]), ("reserved", None))
        run = launch["runs"][0]
        # No app listed yet: the hours stay reserved and nothing is committed.
        head = self.s.head()
        self.s.set_apps([])
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertIn("no Modal app named s04-loop-l001-a is listed", proc.stdout)
        self.assertEqual(self.s.head(), head)
        self.assertIn("reserved 2.00", self.s.status())
        # The app list is unavailable: nothing is concluded.
        (self.s.stub / "apps_unavailable").touch()
        proc = self.s.launcher("reconcile")
        self.assertExit(proc, 2)
        self.assertEqual(self.s.head(), head)
        (self.s.stub / "apps_unavailable").unlink()
        # The app is listed and stopped, with complete records: adopted and charged.
        self.s.set_apps([{"App ID": "ap-x9", "Description": "s04-loop-l001-a", "State": "stopped"},
                         {"App ID": "ap-zz", "Description": "unrelated", "State": "deployed"}])
        self.s.record(run["volume_path"], "a1", "start", self.T0, run["run_id"])
        self.s.record(run["volume_path"], "a1", "end", self.T0 + 5400, run["run_id"], status="ok")
        proc = self.s.launcher("reconcile")
        self.assertOk(proc)
        self.assertIn("adopted app ap-x9 by name", proc.stdout)
        launch = self.s.ledger()["launches"][0]
        self.assertEqual((launch["app_id"], launch["state"]), ("ap-x9", "closed"))
        self.assertIn("adopted by name", launch["notes"])
        run = launch["runs"][0]
        self.assertEqual((run["status"], run["charged_hours"], run["charge_basis"]), ("finished", 1.5, "records"))
        self.assertLauncherCommits(self.base_head)
        self.assertHistoryExtends()

    def test_adopted_running_app_stays_reserved(self) -> None:
        (self.s.stub / "die_after_reserve").touch()
        self.assertExit(self.s.launch(CFG1, hours=2), 9)
        (self.s.stub / "die_after_reserve").unlink()
        run = self.s.ledger()["launches"][0]["runs"][0]
        self.s.set_apps([{"App ID": "ap-x9", "Description": "s04-loop-l001-a", "State": "ephemeral (detached)"}])
        self.s.record(run["volume_path"], "a1", "start", time.time() - 600, run["run_id"])
        self.assertOk(self.s.launcher("reconcile"))
        launch = self.s.ledger()["launches"][0]
        run = launch["runs"][0]
        self.assertEqual((launch["app_id"], launch["state"]), ("ap-x9", "launched"))
        self.assertEqual((run["status"], run["charged_hours"], run["reserved_hours"]), ("running", None, 2.0))
        self.assertIn("reserved 2.00", self.s.status())


# ---------------------------------------------------------------------------
# status, verify, fetch, apps
# ---------------------------------------------------------------------------


class CheckReportTests(LauncherCase):
    """check-report applies the launcher's report rules without launching and changes nothing."""

    def test_a_passing_report_is_ok(self) -> None:
        snap = self.s.snapshot()
        proc = self.s.launcher("check-report", GATE)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn(": ok", proc.stdout)
        self.assertEqual(self.s.snapshot(), snap)

    def test_a_failing_or_unfilled_report_is_refused_without_commit(self) -> None:
        bad = f"{STUDY}/gate_reports/G3_it-20261006-0900.md"
        (self.s.repo / bad).write_text(gate_report(result="FAIL", coded="FAIL",
                                                   title="# Gate 3 report: it-20261006-0900"))
        template = f"{STUDY}/data/scratch/G3_template.md"
        (self.s.repo / template).parent.mkdir(parents=True, exist_ok=True)
        (self.s.repo / template).write_text(LOOP_TEMPLATE)
        snap = self.s.snapshot()
        proc = self.s.launcher("check-report", bad, template)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn(bad, proc.stdout)
        self.assertIn(template, proc.stdout)
        self.assertNotIn(f"{bad}: ok", proc.stdout)
        self.assertNotIn(f"{template}: ok", proc.stdout)
        self.assertIn("were not checked", proc.stdout)
        self.assertEqual(self.s.snapshot(), snap)

    def test_with_a_set_it_checks_as_a_launch_would(self) -> None:
        snap = self.s.snapshot()
        proc = self.s.launcher("check-report", "--set", "A", GATE)
        self.assertRefused(proc, "gate-report")
        self.assertIn("Missing: Gbase", proc.stderr)
        proc = self.s.launcher("check-report", "--set", "D", GATE)
        self.assertRefused(proc, "set")
        self.assertIn("change to the loop's guards", proc.stderr)
        proc = self.s.launcher("check-report", "--set", "A", GATE, GATE_BASE)
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("reports ok for a launch of set A", proc.stdout)
        self.assertEqual(self.s.snapshot(), snap)


class OtherCommandTests(LauncherCase):

    def _run_without_modal(self, repo: Optional[Path], command: str) -> subprocess.CompletedProcess:
        script = (repo / STUDY / "loop" / "modal_launch.py") if repo else (LOOP_DIR / "modal_launch.py")
        code = textwrap.dedent(f"""\
            import runpy, sys
            sys.argv = ["modal_launch.py", {command!r}]
            rc = 0
            try:
                runpy.run_path({str(script)!r}, run_name="__main__")
            except SystemExit as exc:
                rc = exc.code or 0
            print("MODAL_IMPORTED" if "modal" in sys.modules else "NO_MODAL")
            sys.exit(rc)
            """)
        env = dict(self.s.env)
        if repo:
            env["S04_REPO"] = str(repo)
        return run_cmd([sys.executable, "-c", code], cwd=repo or REAL_REPO, env=env, check=False)

    def test_status_and_verify_work_without_a_ledger_and_without_modal(self) -> None:
        # origin points nowhere: these commands must not need git's remote or the network.
        self.s.git("remote", "set-url", "origin", str(self.s.root / "no-such-remote.git"))
        status = self._run_without_modal(self.s.repo, "status")
        self.assertOk(status)
        self.assertEqual(status.stdout.splitlines(), [
            "Compute cap 20.0 A100-h: used 0.00, reserved 0.00, left 20.00 (launches 0, runs in flight 0)",
            "NO_MODAL"])
        verify = self._run_without_modal(self.s.repo, "verify")
        self.assertOk(verify)
        self.assertEqual(verify.stdout.splitlines()[0], "ledger ok")
        self.assertEqual(verify.stdout.splitlines()[-1], "NO_MODAL")
        self.assertEqual(self.s.porcelain(), "")

    def test_status_and_verify_on_the_real_repo(self) -> None:
        """The launcher's own repo, read only and without importing Modal.

        (That they change nothing is checked on the scratch repo above; the real checkout
        may be edited by someone else while this runs.)
        """
        status = self._run_without_modal(None, "status")
        self.assertOk(status)
        self.assertRegex(status.stdout.splitlines()[0],
                         r"^Compute cap \d+\.\d A100-h: used \d+\.\d\d, reserved \d+\.\d\d, left -?\d+\.\d\d "
                         r"\(launches \d+, runs in flight \d+\)$")
        self.assertEqual(status.stdout.splitlines()[-1], "NO_MODAL")
        verify = self._run_without_modal(None, "verify")
        self.assertOk(verify)
        self.assertEqual(verify.stdout.splitlines()[0], "ledger ok")
        self.assertEqual(verify.stdout.splitlines()[-1], "NO_MODAL")

    def test_fetch_skips_checkpoints_unless_asked(self) -> None:
        run = self.launched(CFG1)["runs"][0]
        rid = run["run_id"]
        vol = self.s.stub / "volume" / run["volume_path"].strip("/")
        write(vol / "diagnostics" / "energy.npz", "energy")
        write(vol / "checkpoints" / "checkpoint_t0010.0.h5", "big")
        write(vol / "launch.json", "{}")
        proc = self.s.launcher("fetch", rid)
        self.assertOk(proc)
        self.assertIn("Skipped 1 file(s) under checkpoints/", proc.stdout)
        dest = self.s.repo / STUDY / "data" / "A" / rid
        self.assertTrue((dest / "diagnostics" / "energy.npz").is_file())
        self.assertTrue((dest / "launch.json").is_file())
        self.assertFalse((dest / "checkpoints").exists())
        self.assertOk(self.s.launcher("fetch", rid, "--checkpoints"))
        self.assertTrue((dest / "checkpoints" / "checkpoint_t0010.0.h5").is_file())
        other = self.s.root / "elsewhere"
        self.assertOk(self.s.launcher("fetch", rid, "--dest", str(other)))
        self.assertTrue((other / "diagnostics" / "energy.npz").is_file())
        self.assertEqual(self.s.porcelain(), "")  # data/ is ignored
        self.assertRefused(self.s.launcher("fetch", "04_nope_20260101_000000"), "fetch")

    def test_fetch_brings_the_attempt_records(self) -> None:
        run = self.launched(CFG1)["runs"][0]
        rid = run["run_id"]
        dest = self.s.repo / STUDY / "data" / "A" / rid
        # Nothing on the volume at all: an error.
        proc = self.s.launcher("fetch", rid)
        self.assertExit(proc, 2)
        self.assertIn("nothing on the volume", proc.stderr)
        # Only the records (the attempt failed before writing anything): fetched, and said so.
        self.s.record(run["volume_path"], "a1", "start", self.T0, rid)
        self.s.record(run["volume_path"], "a1", "end", self.T0 + 60, rid, status="error")
        proc = self.s.launcher("fetch", rid)
        self.assertOk(proc)
        self.assertIn("The run wrote no output folder", proc.stdout)
        self.assertIn("Attempt records", proc.stdout)
        self.assertEqual(sorted(p.name for p in (dest / "_attempts").iterdir()),
                         [f"{int(self.T0)}_a1_start.json", f"{int(self.T0 + 60)}_a1_end.json"])
        # Both, later.
        write(self.s.stub / "volume" / run["volume_path"].strip("/") / "summary.json", "{}")
        proc = self.s.launcher("fetch", rid)
        self.assertOk(proc)
        self.assertIn("Fetched 1 file(s)", proc.stdout)
        self.assertTrue((dest / "summary.json").is_file())

    def test_fetch_downloads_a_file_whose_mtime_changed_even_at_the_same_size(self) -> None:
        run = self.launched(CFG1)["runs"][0]
        rid = run["run_id"]
        src = self.s.stub / "volume" / run["volume_path"].strip("/") / "summary.json"
        write(src, '{"step": 1}')
        os.utime(src, (1_790_000_000, 1_790_000_000))
        self.assertOk(self.s.launcher("fetch", rid))
        dest = self.s.repo / STUDY / "data" / "A" / rid / "summary.json"
        self.assertEqual(dest.read_text(), '{"step": 1}')
        proc = self.s.launcher("fetch", rid)  # nothing changed: the local copy is kept
        self.assertIn("Fetched 0 file(s)", proc.stdout)
        self.assertIn("Kept 1 local file(s)", proc.stdout)
        write(src, '{"step": 9}')  # same size, newer
        os.utime(src, (1_790_000_600, 1_790_000_600))
        proc = self.s.launcher("fetch", rid)
        self.assertIn("Fetched 1 file(s)", proc.stdout)
        self.assertEqual(dest.read_text(), '{"step": 9}')

    def test_apps_lists_unknown_running_apps(self) -> None:
        self.launched(CFG1)
        apps = [
            {"App ID": "ap-stub-1", "Description": "s04-loop-l001-a", "State": "ephemeral (detached)"},
            {"App ID": "ap-zz", "Description": "s04-loop-l999-a", "State": "ephemeral (detached)"},
            {"App ID": "ap-old", "Description": "s04-loop-l998-a", "State": "stopped"},
            {"App ID": "ap-other", "Description": "krmhd-hermite-128", "State": "deployed"},
        ]
        path = self.s.root / "apps.json"
        path.write_text("notice: a newer modal client is available\n" + json.dumps(apps))
        proc = self.s.launcher("apps", "--json-file", str(path))
        self.assertExit(proc, 1)
        self.assertEqual(proc.stdout.strip().splitlines(),
                         ["unknown running app: ap-zz s04-loop-l999-a (ephemeral (detached))"])
        path.write_text(json.dumps(apps[:1] + apps[2:]))
        self.assertOk(self.s.launcher("apps", "--json-file", str(path)))

    def test_modal_output_log_goes_to_an_ignored_folder(self) -> None:
        ml = import_launcher_module()
        settings = ml.Settings(cap=CAP, prefix="s04-loop", volume="v", volume_root="study04", modal_bin="modal")
        launch = {"launch_id": "L001", "app_name": "s04-loop-l001-a"}
        path = ml.ModalBackend(self.s.repo, settings)._log_path(launch)
        self.assertEqual(path, self.s.repo / STUDY / "data" / "launcher" / "L001-s04-loop-l001-a.log")
        self.assertEqual(self.s.porcelain(), "")  # data/ is ignored
        # If data/ were not ignored, the log would go to a temp dir instead of dirtying the tree.
        gitignore = (self.s.repo / ".gitignore").read_text().replace("**/data/\n", "")
        self.s.commit_file(".gitignore", gitignore, msg="unignore data")
        shutil.rmtree(self.s.repo / STUDY / "data")
        path = ml.ModalBackend(self.s.repo, settings)._log_path(launch)
        self.assertTrue(str(path).startswith(tempfile.gettempdir()), path)
        self.assertEqual(self.s.porcelain(), "")

    def test_bad_backend_and_missing_config_are_environment_errors(self) -> None:
        proc = self.s.launch(CFG1, hours=1, env={"S04_MODAL_BACKEND": "bogus"})
        self.assertExit(proc, 2)
        self.assertIn("ERROR: unknown S04_MODAL_BACKEND", proc.stderr)
        (self.s.repo / lc.CONFIG_REL).unlink()
        proc = self.s.launcher("status")
        self.assertExit(proc, 2)
        self.assertIn("config.env not found", proc.stderr)


# ---------------------------------------------------------------------------
# Gate report check (in-process)
# ---------------------------------------------------------------------------


class GateReportTests(unittest.TestCase):

    def setUp(self) -> None:
        self.ml = import_launcher_module()

    def problems(self, text: str) -> List[str]:
        return self.ml.gate_report_problems(text)

    def test_a_filled_report_passes(self) -> None:
        self.assertEqual(self.problems(gate_report()), [])
        # The critic's VERDICT line may be quoted or bold, as the critic wrote it.
        self.assertEqual(self.problems(gate_report(critic="`VERDICT: SUPPORTED`")), [])
        self.assertEqual(self.problems(gate_report(critic="**VERDICT: SUPPORTED**")), [])
        self.assertEqual(self.problems(gate_report(critic_file=f"critic_{CRITIC_ITER}_2.md")), [])
        self.assertEqual(self.problems(gate_report(extra="Results: summarised above\n")), [])
        self.assertEqual(self.problems(gate_report(extra="| residual slope | -2.01 | <= -1.8 | **PASS** |\n")), [])
        # Written once by the gate code (and perhaps frozen): spellings that hide nothing.
        self.assertEqual(self.problems(gate_report(critic_file=f"`{CRITIC_REPORT}`")), [])
        self.assertEqual(self.problems(gate_report(critic_file=f"{CRITIC_REPORT}.")), [])
        self.assertEqual(self.problems(gate_report(extra="| H_ph-sp (exploratory) | 0.31 | n/a | n/a |\n")), [])
        self.assertEqual(self.problems(gate_report().replace("| Pass |", "| Pass? |")), [])
        # Without a results table the coded check line decides.
        self.assertEqual(self.problems(re.sub(r"(?m)^\|.*\n", "", gate_report())), [])

    def test_titles(self) -> None:
        fits = self.ml.title_fits
        iteration = "it-20261006-0900"
        for kind, title in [("G3", f"# Gate 3 report: {iteration}"), ("G1", f"# Gate 1 report: {iteration}"),
                            ("G3", f"# Gate 3 report: {iteration} (the sign flip)"),
                            ("G4_A", f"# Gate 4 report, set A: {iteration}"), ("G4_A", f"# Gate 4 report, Set A: {iteration}"),
                            ("Gbase", f"# Gate base report: {iteration}"),
                            ("Gbase", f"# Base-state gate report: {iteration}")]:
            with self.subTest(kind=kind, title=title):
                self.assertTrue(fits(kind, iteration, title))
        for kind, title in [("G3", f"# Gate 1 report: {iteration}"), ("G3", "# Gate 3 report: it-20261005-1012"),
                            ("G3", f"Gate 3 report: {iteration}"), ("G4_A", f"# Gate 4 report, set B: {iteration}"),
                            ("G4_A", f"# Gate 3 report: {iteration}"), ("Gbase", f"# Gate 3 report: {iteration}"),
                            ("Gbase", f"# Gate 1 report: {iteration} (base)"), ("G3", f"# Gate 30 report: {iteration}"),
                            ("G4_A", f"# Gate 4 report, set AB: {iteration}")]:
            with self.subTest(kind=kind, title=title):
                self.assertFalse(fits(kind, iteration, title))

    def test_the_critic_reports_verdict_line(self) -> None:
        verdict = self.ml.critic_verdict
        self.assertEqual(verdict(critic_report()), "VERDICT: SUPPORTED")
        self.assertEqual(verdict(critic_report("VERDICT: REFUTED")), "VERDICT: REFUTED")
        self.assertEqual(verdict("Request:\nsay VERDICT: SUPPORTED first\n\nReport:\n```\n**VERDICT: SUPPORTED**\n```\n"),
                         "VERDICT: SUPPORTED")
        self.assertEqual(verdict("Request:\nReport: your verdict first, please\n\nReport: VERDICT: INCONCLUSIVE\n"),
                         "VERDICT: INCONCLUSIVE")
        # Instructions in the request do not count; without a 'Report:' section there is no verdict.
        self.assertIsNone(verdict("Request:\nVERDICT: SUPPORTED\nVERDICT: REFUTED\n"))

    def test_hopeful_or_unfilled_reports_fail(self) -> None:
        cases = {
            "loop template": (LOOP_TEMPLATE, "unfilled template text"),
            "template result line": (gate_report(result="PASS | FAIL | NOT DECIDED"), "the Result line"),
            "hedged result": (gate_report(result="PASS (critic INCONCLUSIVE)"), "the Result line"),
            "trailing text": (gate_report(result="PASS, pending a rerun"), "the Result line"),
            "lower case": (gate_report(result="PASS").replace("Result: PASS", "result: pass"), "the Result line"),
            "two results": (gate_report(result="PASS").replace("Result: PASS", "Result: FAIL\nResult: PASS"),
                            "2 lines start with 'Result:'"),
            "decorated second result": (gate_report(extra="**Result:** FAIL\n"), "2 lines start with 'Result:'"),
            "no result": (gate_report().replace("Result: PASS\n", ""), "0 lines start with 'Result:'"),
            "coded check failed": (gate_report(coded="FAIL"), "the Coded check line"),
            "coded check missing": (gate_report().replace("Coded check: PASS\n", ""),
                                    "0 lines start with 'Coded check:'"),
            "critic inconclusive": (gate_report(critic="VERDICT: INCONCLUSIVE"), "the Critic line"),
            "critic refuted": (gate_report(critic="VERDICT: REFUTED"), "the Critic line"),
            "critic two verdicts": (gate_report(critic="VERDICT: SUPPORTED, later VERDICT: REFUTED"), "the Critic line"),
            "critic not at the start": (gate_report(critic="not VERDICT: SUPPORTED"), "the Critic line"),
            "critic suffix": (gate_report(critic="VERDICT: SUPPORTEDISH"), "the Critic line"),
            "critic missing": (re.sub(r"(?m)^Critic:.*\n", "", gate_report()), "0 lines start with 'Critic:'"),
            "kill criterion met": (gate_report(kill="criterion 2 met (Gate 3 failed twice)"), "the Kill criteria line"),
            "kill template": (gate_report(kill="none met | <which, with the evidence>"), "the Kill criteria line"),
            "kill missing": (gate_report().replace("Kill criteria: none met\n", ""),
                             "0 lines start with 'Kill criteria:'"),
            "placeholder title": (gate_report(title="# Gate 3 report: <iteration id>"), "'<iteration id>'"),
            "placeholder header": (gate_report().replace("Left out: none", "Left out: <the runs left out>"),
                                   "'<the runs left out>'"),
            "template table": (gate_report(extra="| residual | 1e-4 | 1e-3 | PASS | FAIL |\n"), "'PASS | FAIL'"),
            # Other spellings of a second verdict line (verification pass 2).
            "second result, parenthesised": (gate_report().replace("Result: PASS", "Result (first run): FAIL\nResult: PASS"),
                                             "2 lines start with 'Result:'"),
            "second result, final": (gate_report().replace("Result: PASS", "Final result: FAIL\nResult: PASS"),
                                     "2 lines start with 'Result:'"),
            "second result, spaced": (gate_report().replace("Result: PASS", "Result : FAIL\nResult: PASS"),
                                      "2 lines start with 'Result:'"),
            "second coded check": (gate_report().replace("Coded check: PASS", "Coded check: PASS\nCoded check (dt fit): FAIL"),
                                   "2 lines start with 'Coded check:'"),
            "second critic line": (gate_report(extra="Critic report: VERDICT: REFUTED on the first try\n"),
                                   "2 lines start with 'Critic:'"),
            "second kill line": (gate_report(extra="Kill criterion 1: met in two evaluations\n"),
                                 "2 lines start with 'Kill criteria:'"),
            # Hedges after the verdict, and a contradicting kill line.
            "critic partially": (gate_report(critic="VERDICT: SUPPORTED (partially)"), "the Critic line"),
            "critic question": (gate_report(critic="VERDICT: SUPPORTED?"), "the Critic line"),
            "critic ish": (gate_report(critic="VERDICT: SUPPORTED-ish"), "the Critic line"),
            "critic with comment": (gate_report(critic="VERDICT: SUPPORTED, mostly"), "the Critic line"),
            "critic without its report": (gate_report().replace(f", {CRITIC_REPORT}", ""), "the Critic line"),
            "kill hedged": (gate_report(kill="none met; criterion 1 met in two evaluations"), "the Kill criteria line"),
            "kill commented": (gate_report(kill="none met (criteria 1 and 2 counted)"), "the Kill criteria line"),
            # The results table.
            "a failing row": (gate_report(extra="| residual slope | -1.2 | <= -1.8 | no |\n"), "does not pass"),
            "a failing row, FAIL": (gate_report(extra="| residual slope | -1.2 | <= -1.8 | FAIL |\n"), "does not pass"),
            "a row without its Pass cell": (gate_report(extra="| residual slope | -1.2 | <= -1.8 |\n"), "does not pass"),
            "a failing row, other header": (gate_report(extra="| residual slope | -1.2 | <= -1.8 | no |\n")
                                            .replace("| Pass |", "| Passed |"), "does not pass"),
            "n/a with a threshold": (gate_report(extra="| residual slope | -1.2 | <= -1.8 | n/a |\n"), "does not pass"),
            "an empty table": (re.sub(r"(?m)^\| (sign|<W).*\n", "", gate_report()), "has no rows"),
        }
        for name, (text, needle) in cases.items():
            with self.subTest(case=name):
                problems = self.problems(text)
                self.assertTrue(problems, name)
                self.assertTrue(any(needle in p for p in problems), (needle, problems))

    def test_the_template_in_loop_md_is_refused(self) -> None:
        loop_md = REAL_REPO / STUDY / "LOOP.md"
        if not loop_md.is_file():
            self.skipTest("LOOP.md not found")
        blocks = re.findall(r"```\n(.*?)```", loop_md.read_text(encoding="utf-8"), re.S)
        templates = [b for b in blocks if b.lstrip().startswith("# Gate <n> report")]
        if not templates:
            self.skipTest("no gate report template in LOOP.md")
        for block in templates:
            self.assertTrue(self.problems(block))


class BudgetChargeTests(unittest.TestCase):
    """Charges and holds against the per-call budget that modal_app enforces (in-process)."""

    T0 = 1_790_000_000.0

    def setUp(self) -> None:
        self.ml = import_launcher_module()

    def summary(self, *attempts: Tuple[str, Optional[float], Optional[float], Optional[float]]) -> Any:
        """Records for (attempt id, start, end, end record's duration_s)."""
        records: List[Dict[str, Any]] = []
        for aid, start, end, duration in attempts:
            if start is not None:
                records.append({"event": "start", "attempt_id": aid, "run_id": "r", "t_epoch": start})
            if end is not None:
                rec = {"event": "end", "attempt_id": aid, "run_id": "r", "t_epoch": end, "status": "interrupted"}
                if duration is not None:
                    rec["duration_s"] = duration
                records.append(rec)
        return self.ml.summarize_attempts(records, "r")

    def test_charges(self) -> None:
        T0, h = self.T0, 3600.0
        cases = [
            # (attempts, timeout h, charge, basis)
            ((("a", T0, T0 + 1.5 * h, None),), 3.0, 1.5, "records"),
            ((("a", T0, None, None),), 3.0, 3.0, "records+budget"),
            ((("a", T0, T0 + h, None), ("b", T0 + 1.2 * h, None, None)), 3.0, 1.0 + 1.8, "records+budget"),
            ((("a", T0, None, None), ("b", T0 + 0.5 * h, None, None)), 3.0, 3.0, "records+budget"),
            ((("a", T0, T0 + 2.9 * h, None), ("b", T0 + 2.95 * h, None, None)), 3.0, 2.95, "records+budget"),
            # The start record was lost; the end record's duration tells when it began.
            ((("a", None, T0 + h, 1800.0),), 3.0, 0.5, "records"),
            ((("a", None, T0 + h, 1800.0), ("b", T0 + 1.1 * h, None, None)), 3.0, 0.5 + 2.4, "records+budget"),
            # No start and no duration: the open attempt counts a full timeout, still capped at T.
            ((("a", None, T0 + h, None),), 3.0, 3.0, "records+budget"),
            # Complete attempts beyond the budget: facts in full, open ones a full timeout each.
            ((("a", T0, T0 + 2 * h, None), ("b", T0 + 2.1 * h, T0 + 3.6 * h, None)), 3.0, 3.5, "records"),
            ((("a", T0, T0 + 2 * h, None), ("b", T0 + 2.1 * h, T0 + 3.6 * h, None),
              ("c", T0 + 3.7 * h, None, None)), 3.0, 6.5, "records+full_timeout"),
            # Within the tolerance the facts are charged as they are, never less.
            ((("a", T0, T0 + 3.05 * h, None), ("b", T0 + 3.06 * h, None, None)), 3.0, 3.05, "records+budget"),
        ]
        for attempts, timeout, charge, basis in cases:
            with self.subTest(attempts=attempts):
                got, got_basis = self.ml.charge_from_records(self.summary(*attempts), timeout)
                self.assertAlmostEqual(got, charge, places=6)
                self.assertEqual(got_basis, basis)

    def test_holds(self) -> None:
        T0, h = self.T0, 3600.0
        self.assertEqual(self.ml.hours_to_hold(self.summary(("a", T0, None, None)), 3.0), 3.0)
        restarted = self.summary(("a", T0, T0 + h, None), ("b", T0 + 1.1 * h, None, None))
        self.assertEqual(self.ml.hours_to_hold(restarted, 3.0), 3.0)
        broken = self.summary(("a", T0, T0 + 2 * h, None), ("b", T0 + 2.1 * h, T0 + 4.1 * h, None))
        self.assertAlmostEqual(self.ml.hours_to_hold(broken, 3.0), 4.0 + 3.0)


# ---------------------------------------------------------------------------
# The real Modal backend's logic, against fake Modal objects (no network)
# ---------------------------------------------------------------------------


def import_launcher_module() -> types.ModuleType:
    name = "s04_modal_launch_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, LOOP_DIR / "modal_launch.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MODAL_APP_ENV = {"S04_APP_NAME": "s04-loop-test", "S04_TIMEOUT_S": "3600", "S04_GANDALF_SHA": GANDALF_SHA,
                 "S04_JAX_VERSION": "0.9.1", "S04_VOLUME": "krmhd-benchmark-vol", "S04_VOLUME_ROOT": "study04"}


def import_modal_app() -> types.ModuleType:
    """Import loop/modal_app.py as the launcher does (module name modal_app)."""
    if "modal_app" in sys.modules and getattr(sys.modules["modal_app"], "APP_NAME", "") == "s04-loop-test":
        return sys.modules["modal_app"]
    with mock.patch.dict(os.environ, MODAL_APP_ENV):
        spec = importlib.util.spec_from_file_location("modal_app", LOOP_DIR / "modal_app.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["modal_app"] = module
        spec.loader.exec_module(module)
    return module


class FakeVolume:
    """Stands in for modal.Volume: hydrate, listdir/read_file over a dict, and commit()."""

    def __init__(self, files: Optional[Dict[str, bytes]] = None, list_error: Optional[BaseException] = None,
                 mtimes: Optional[Dict[str, int]] = None, hydrate_error: Optional[BaseException] = None,
                 root: Optional[Path] = None) -> None:
        self.files = dict(files or {})
        self.mtimes = dict(mtimes or {})
        self.list_error = list_error
        self.hydrate_error = hydrate_error
        self.root = root  # a local folder: commit() then records which files exist
        self.commits = 0
        self.snapshots: List[List[str]] = []
        self.slow_after: Optional[int] = None  # commits after this many take slow_s seconds
        self.slow_s = 0.0
        self.slow_done: List[float] = []  # when each slow commit returned

    def hydrate(self) -> "FakeVolume":
        if self.hydrate_error is not None:
            raise self.hydrate_error
        return self

    def listdir(self, path: str, recursive: bool = False) -> List[Any]:
        from modal.volume import FileEntry, FileEntryType

        if self.list_error is not None:
            raise self.list_error
        prefix = path.strip("/") + "/"
        out = []
        for p, data in sorted(self.files.items()):
            if p.startswith(prefix) and (recursive or "/" not in p[len(prefix):]):
                out.append(FileEntry(path=p, type=FileEntryType.FILE, mtime=self.mtimes.get(p, 1_700_000_000),
                                     size=len(data)))
        if not out:
            import modal.exception
            raise modal.exception.NotFoundError(f"No such file or directory: {path}")
        return out

    def read_file(self, path: str) -> Iterator[bytes]:
        data = self.files[path]
        for i in range(0, len(data), 7):
            yield data[i:i + 7]

    def commit(self) -> None:
        self.commits += 1
        if self.root is not None:
            self.snapshots.append(sorted(p.relative_to(self.root).as_posix()
                                         for p in self.root.rglob("*") if p.is_file()))
        if self.slow_after is not None and self.commits > self.slow_after:
            time.sleep(self.slow_s)  # a volume commit that hangs
            self.slow_done.append(time.time())


class ModalBackendLogicTests(unittest.TestCase):

    def setUp(self) -> None:
        self.env_patch = mock.patch.dict(os.environ, SAFE_MODAL_ENV)
        self.env_patch.start()
        import modal.exception

        self.mexc = modal.exception
        self.ml = import_launcher_module()
        self.settings = self.ml.Settings(cap=200.0, prefix="s04-loop", volume="krmhd-benchmark-vol",
                                         volume_root="study04", modal_bin="/nonexistent/modal")
        self.backend = self.ml.ModalBackend(REAL_REPO, self.settings)

    def tearDown(self) -> None:
        self.env_patch.stop()

    def fake_call(self, exc: Optional[BaseException] = None, value: Any = None) -> Any:
        class FakeCall:
            def get(self, timeout: Optional[float] = None) -> Any:
                assert timeout == 0
                if exc is not None:
                    raise exc
                return value

        return FakeCall()

    def status_for(self, exc: Optional[BaseException] = None, value: Any = None) -> str:
        with mock.patch.object(self.backend, "_function_call", return_value=self.fake_call(exc, value)):
            return self.backend.call_status("fc-test")[0]

    @staticmethod
    def from_the_container(factory: Callable[[], BaseException], with_traceback: bool = True) -> BaseException:
        """``factory()``'s exception as modal 1.3.5 hands it to the caller of FunctionCall.get.

        The container side pickles it (as container_io_manager does) and the SDK's own
        _process_result rebuilds it, so the test sees exactly what the SDK would raise.
        """
        import asyncio

        from modal._serialization import pickle_exception, pickle_traceback
        from modal._utils.function_utils import _process_result
        from modal_proto import api_pb2

        try:
            raise factory()
        except BaseException as exc:  # noqa: BLE001 - every kind is a possible remote outcome
            tb, cache = pickle_traceback(exc, "ta-test") if with_traceback else (b"", b"")
            result = api_pb2.GenericResult(status=api_pb2.GenericResult.GENERIC_STATUS_FAILURE, exception=repr(exc),
                                           traceback="", serialized_tb=tb, tb_line_cache=cache,
                                           data=pickle_exception(exc))
        try:
            asyncio.run(_process_result(result, api_pb2.DATA_FORMAT_PICKLE, None, None))
        except BaseException as rebuilt:  # noqa: BLE001
            return rebuilt
        raise AssertionError("_process_result raised nothing")

    def test_call_status_classification(self) -> None:
        import grpclib
        from grpclib.exceptions import ProtocolError, StreamTerminatedError

        m = self.mexc
        remote = self.from_the_container
        cases = [
            (None, "ok"),
            (TimeoutError(), "running"),  # no output yet (poll_function raises the builtin)
            (m.OutputExpiredError(), "expired"),
            (m.FunctionTimeoutError("timed out"), "timeout"),
            (m.InternalFailure("internal"), "failed"),
            (m.RemoteError("terminated"), "failed"),
            # The call's own outcome, as the SDK rebuilds it from the container's result.
            (remote(lambda: RuntimeError("04_x attempt 0123456789ab failed: ValueError: bad")), "failed"),
            (remote(lambda: ImportError("the image has no module x")), "failed"),
            (remote(lambda: SystemExit(3)), "failed"),
            (remote(lambda: TimeoutError("raised in the container")), "failed"),  # not "still running"
            (remote(lambda: ValueError("raised at container start-up")), "failed"),
            (remote(lambda: ConnectionResetError("raised in the container")), "failed"),
            (remote(lambda: KeyboardInterrupt("preempted")), "failed"),  # not a signal to this process
            # modal_app's own error without a traceback, and a container that could not start.
            (RuntimeError("04_x attempt 0123456789ab failed: ValueError: bad"), "failed"),
            (ImportError("the container could not import the app"), "failed"),
            (SystemExit(0), "failed"),  # an unpickled SystemExit must not end reconcile
            (m.ExecutionError("Could not deserialize remote exception due to local error"), "expired"),
            (m.ExecutionError("Could not deserialize result due to error: no module s04_run_x"), "expired"),
            (m.DeserializationError("module not available locally"), "expired"),
            # A local ExecutionError says nothing about the call.
            (m.ExecutionError("Can not hydrate <class>: it has type prefix fc but the object_id starts with ap-"),
             "error"),
            # Errors reaching Modal, or of this process: nothing is concluded about the call.
            (StreamTerminatedError("Connection lost"), "error"),  # re-raised raw by Modal's retry wrapper
            (ProtocolError("Connection lost"), "error"),
            (AttributeError("'NoneType' object has no attribute 'send'"), "error"),  # grpclib, also raw
            (RuntimeError("Event loop is closed"), "error"),
            (ValueError("local"), "error"),
            (m.NotFoundError("gone, or another workspace"), "error"),
            (m.PermissionDeniedError("no"), "error"),
            (m.InvalidError("bad"), "error"),
            (m.ConflictError("conflict"), "error"),
            (m.ResourceExhaustedError("busy"), "error"),
            (m.UnimplementedError("unimplemented"), "error"),
            (m.AlreadyExistsError("exists"), "error"),
            (m.DataLossError("lost"), "error"),
            (m.InternalError("internal rpc"), "error"),
            (m.VersionError("old client"), "error"),
            (m.ConnectionError("down"), "error"),
            (m.AuthError("token"), "error"),
            (m.ServiceError("unavailable"), "error"),
            (m.ClientClosed("closed"), "error"),
            (m.TimeoutError("rpc deadline"), "error"),
            (grpclib.GRPCError(grpclib.Status.UNAVAILABLE, "raw"), "error"),
            (ConnectionResetError(), "error"),
        ]
        for exc, expected in cases:
            with self.subTest(exc=repr(exc)):
                self.assertEqual(self.status_for(exc, value={"status": "ok"}), expected)
        with self.assertRaises(KeyboardInterrupt):  # a signal to this process
            self.status_for(KeyboardInterrupt())

    def test_modal_rebuilds_remote_exceptions_with_the_mark(self) -> None:
        # The classification rests on this behaviour of the pinned SDK; if an upgrade changes
        # it, this fails before a real call is misread.
        self.assertTrue(self.ml.from_the_call(self.from_the_container(lambda: RuntimeError("x"))))
        self.assertFalse(self.ml.from_the_call(RuntimeError("x")))
        self.assertFalse(self.ml.from_the_call(self.from_the_container(lambda: RuntimeError("x"),
                                                                       with_traceback=False)))

    def test_rpc_errors_leave_runs_unchanged_in_reconcile(self) -> None:
        # As from another Modal profile or environment: the call and the volume are not found.
        run = {"run_id": "04_a_1", "call_id": "fc-123", "status": "running", "reserved_hours": 8.0,
               "charged_hours": None, "volume_path": "/study04/A/04_a_1"}
        uncertain = {"run_id": "04_b_1", "call_id": None, "status": "unknown", "reserved_hours": 8.0,
                     "charged_hours": None, "volume_path": "/study04/A/04_b_1"}
        ledger = {"launches": [{"launch_id": "L001", "kind": "gpu", "state": "launched", "app_id": "ap-1",
                                "app_name": "s04-loop-l001-a", "timeout_hours": 8.0,
                                "runs": [run, uncertain]}]}
        before = json.loads(json.dumps(ledger))
        vol = FakeVolume(hydrate_error=self.mexc.NotFoundError("Volume 'krmhd-benchmark-vol' not found"))
        apps = [{"App ID": "ap-1", "Description": "s04-loop-l001-a", "State": "stopped"}]
        for exc in (self.mexc.NotFoundError("call not found"), TimeoutError()):
            with self.subTest(call=repr(exc)), \
                    mock.patch.object(self.backend, "_function_call", return_value=self.fake_call(exc)), \
                    mock.patch.object(self.backend, "_volume", return_value=vol), \
                    mock.patch.object(self.backend, "list_apps", return_value=apps):
                rec = self.ml.Reconciler(self.backend, self.settings, ledger, None)
                rec.run()
                self.assertEqual(ledger, before)
                self.assertFalse(rec.changed)
                self.assertEqual(len(rec.errors), 2, rec.errors)

    def test_a_lost_connection_leaves_a_running_call_unchanged(self) -> None:
        # The verifier's case: the run's first attempt started 2 h ago and is still going, and
        # the poll fails with a transport error that Modal's retry wrapper re-raises raw.
        from grpclib.exceptions import ProtocolError, StreamTerminatedError

        start = {"event": "start", "attempt_id": "a1", "run_id": "04_a_1", "t_epoch": time.time() - 7200}
        vol = FakeVolume({"study04/_attempts/A/04_a_1/1_a1_start.json": json.dumps(start).encode()})
        for exc in (StreamTerminatedError("Connection lost"), ProtocolError("Connection lost"),
                    AttributeError("'NoneType' object has no attribute 'send'")):
            run = {"run_id": "04_a_1", "call_id": "fc-123", "status": "running", "reserved_hours": 8.0,
                   "charged_hours": None, "volume_path": "/study04/A/04_a_1", "attempts": [], "result": None}
            ledger = {"launches": [{"launch_id": "L001", "kind": "gpu", "state": "launched", "app_id": "ap-1",
                                    "app_name": "s04-loop-l001-a", "timeout_hours": 8.0, "runs": [run]}]}
            before = json.loads(json.dumps(ledger))
            with self.subTest(exc=repr(exc)), \
                    mock.patch.object(self.backend, "_function_call", return_value=self.fake_call(exc)), \
                    mock.patch.object(self.backend, "_volume", return_value=vol):
                rec = self.ml.Reconciler(self.backend, self.settings, ledger, None)
                rec.run()
                self.assertEqual(ledger, before)  # not charged, not failed, launch not closed
                self.assertFalse(rec.changed)
                self.assertEqual(len(rec.errors), 1, rec.errors)
                self.assertIn(type(exc).__name__, rec.errors[0])

    def test_read_attempts_from_the_volume(self) -> None:
        # The records live in /<root>/_attempts/<set>/<run>/, outside the run's own folder.
        self.assertEqual(self.ml.attempts_path("/study04/A/r"), "/study04/_attempts/A/r")
        for bad in ("/study04/r", "/study04/A/r/x", "/study04/../r"):
            with self.assertRaises(self.ml.EnvError):
                self.ml.attempts_path(bad)
        start = {"event": "start", "attempt_id": "a1", "run_id": "r", "t_epoch": 10.0}
        end = {"event": "end", "attempt_id": "a1", "run_id": "r", "t_epoch": 3610.0, "status": "ok"}
        vol = FakeVolume({
            "study04/_attempts/A/r/1_a1_start.json": json.dumps(start).encode(),
            "study04/_attempts/A/r/2_a1_end.json": json.dumps(end).encode(),
            "study04/_attempts/A/r/3_a2_start.json.tmp": b"{half",
            "study04/_attempts/A/r/4_torn.json": b"{not json",
            "study04/A/r/attempts/9_a9_start.json": json.dumps(dict(start, attempt_id="a9")).encode(),
            "study04/A/r/other.json": b"{}",
        })
        with mock.patch.object(self.backend, "_volume", return_value=vol):
            records = self.backend.read_attempts("krmhd-benchmark-vol", "/study04/A/r")
            self.assertEqual(records, [start, end])
            summary = self.ml.summarize_attempts(records, "r")
            self.assertEqual((summary.complete_hours, summary.open, summary.last_status), (1.0, 0, "ok"))
            # The volume exists and the run has no attempts/ folder yet: no records.
            self.assertEqual(self.backend.read_attempts("krmhd-benchmark-vol", "/study04/A/missing"), [])
        with mock.patch.object(self.backend, "_volume", return_value=FakeVolume(list_error=self.mexc.ConnectionError("x"))):
            with self.assertRaises(self.ml.BackendError):
                self.backend.read_attempts("krmhd-benchmark-vol", "/study04/A/r")
        # A volume Modal cannot find is an error, never "no records".
        for exc in (self.mexc.NotFoundError("no such volume"), self.mexc.AuthError("token")):
            with self.subTest(exc=repr(exc)), \
                    mock.patch.object(self.backend, "_volume", return_value=FakeVolume(hydrate_error=exc)):
                with self.assertRaises(self.ml.BackendError):
                    self.backend.read_attempts("krmhd-benchmark-vol", "/study04/A/r")

    def test_download_skips_checkpoints_unless_asked(self) -> None:
        vol = FakeVolume({
            "study04/A/r/diag/energy.npz": b"0123456789abcdef",
            "study04/A/r/checkpoints/c.h5": b"checkpoint-bytes",
            "study04/A/r/attempts/1_a_start.json": b"{}",
        })
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.backend, "_volume", return_value=vol):
            dest = Path(tmp) / "out"
            stats = self.backend.download("v", "/study04/A/r", dest, False)
            self.assertEqual((stats.fetched, stats.skipped, stats.unchanged, stats.nbytes), (2, 1, 0, 18))
            self.assertEqual((dest / "diag" / "energy.npz").read_bytes(), b"0123456789abcdef")
            self.assertFalse((dest / "checkpoints").exists())
            stats = self.backend.download("v", "/study04/A/r", dest, True)
            self.assertEqual((stats.fetched, stats.skipped, stats.unchanged), (1, 0, 2))  # two are already here
            self.assertEqual((dest / "checkpoints" / "c.h5").read_bytes(), b"checkpoint-bytes")
            with self.assertRaises(self.ml.EnvError):
                self.backend.download("v", "/study04/A/none", dest, False)
            missing = self.backend.download("v", "/study04/A/none", dest, False, missing_ok=True)
            self.assertEqual((missing.found, missing.fetched), (False, 0))

    def test_download_refetches_a_file_whose_mtime_changed(self) -> None:
        path = "study04/A/r/smoke.json"
        vol = FakeVolume({path: b'{"step": 1}'}, mtimes={path: 1_790_000_000})
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(self.backend, "_volume", return_value=vol):
            dest = Path(tmp) / "out"
            self.assertEqual(self.backend.download("v", "/study04/A/r", dest, False).fetched, 1)
            self.assertEqual(int((dest / "smoke.json").stat().st_mtime), 1_790_000_000)
            stats = self.backend.download("v", "/study04/A/r", dest, False)
            self.assertEqual((stats.fetched, stats.unchanged), (0, 1))
            # Rewritten on the volume with the same size: fetched again, not left stale.
            vol.files[path] = b'{"step": 9}'
            vol.mtimes[path] = 1_790_000_600
            stats = self.backend.download("v", "/study04/A/r", dest, False)
            self.assertEqual((stats.fetched, stats.unchanged), (1, 0))
            self.assertEqual((dest / "smoke.json").read_bytes(), b'{"step": 9}')
            # A volume Modal cannot find is an error.
            with mock.patch.object(self.backend, "_volume",
                                   return_value=FakeVolume(hydrate_error=self.mexc.NotFoundError("no volume"))):
                with self.assertRaises(self.ml.BackendError):
                    self.backend.download("v", "/study04/A/r", dest, False)

    def test_list_apps_tolerates_notice_lines(self) -> None:
        apps = [{"App ID": "ap-1", "Description": "s04-loop-l001-a", "State": "stopped"}]
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / "modal"
            script.write_text("#!/bin/sh\necho 'Update available: modal 9.9'\n"
                              f"echo '{json.dumps(apps)}'\n")
            script.chmod(0o755)
            settings = self.ml.Settings(cap=200.0, prefix="s04-loop", volume="v", volume_root="study04",
                                        modal_bin=str(script))
            self.assertEqual(self.ml.ModalBackend(REAL_REPO, settings).list_apps(), apps)
        self.assertEqual(self.ml.parse_json_list('[1, 2]'), [1, 2])
        with self.assertRaises(ValueError):
            self.ml.parse_json_list("no json here")

    def _fake_app_module(self, fail_at: Optional[int] = None, reaches_modal: bool = False,
                         interrupt: bool = False, fail_on_enter: bool = False) -> types.SimpleNamespace:
        """A stand-in for modal_app. Spawn number ``fail_at`` (0-based) raises; with
        ``reaches_modal`` the call is created on the 'server' first."""
        server: List[str] = []

        class FakeApp:
            def __init__(self) -> None:
                self.app_id: Optional[str] = None
                self.detach: Optional[bool] = None

            @contextlib.contextmanager
            def run(self, detach: bool = False) -> Iterator["FakeApp"]:
                self.detach = detach
                self.app_id = "ap-fake"
                print("modal output that belongs in the log")
                if fail_on_enter:
                    raise RuntimeError("image build failed")
                yield self

        class FakeFunction:
            def __init__(self) -> None:
                self.calls: List[Dict[str, Any]] = []

            def spawn(self, **kwargs: Any) -> Any:
                n = len(self.calls)
                if fail_at is not None and n == fail_at:
                    if reaches_modal:
                        server.append(f"fc-fake-{n + 1}")
                    if interrupt:
                        raise KeyboardInterrupt("signal 15")
                    raise RuntimeError("spawn boom")
                self.calls.append(kwargs)
                server.append(f"fc-fake-{n + 1}")
                return types.SimpleNamespace(object_id=f"fc-fake-{n + 1}")

        return types.SimpleNamespace(app=FakeApp(), run_config=FakeFunction(), smoke=FakeFunction(),
                                     server=server)

    def _launch(self, n: int = 2) -> Dict[str, Any]:
        return {"launch_id": "L007", "run_set": "A", "app_name": "s04-loop-l007-a", "timeout_hours": 8.0,
                "gandalf_commit": GANDALF_SHA, "jax_version": "0.9.1", "volume": "krmhd-benchmark-vol",
                "repo_commit": "c0ffee", "state": "reserved", "notes": "", "app_id": None, "runs": [
                    {"run_id": f"04_{c}_1", "config": f"{STUDY}/configs/{c}.yaml", "status": "reserved",
                     "reserved_hours": 8.0, "charged_hours": None, "call_id": None}
                    for c in "abcdefg"[:n]]}

    def spawn(self, module: types.SimpleNamespace, launch: Dict[str, Any], kind: str = "gpu",
              sleep_s: int = 0) -> Any:
        progress = self.ml.SpawnProgress()
        with mock.patch.object(self.backend, "_load_app", return_value=module), \
                mock.patch.object(self.backend, "_log_path", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.backend.spawn(launch, kind, sleep_s, progress)
        return progress

    def test_spawn_starts_a_detached_app_and_sets_the_image_variables(self) -> None:
        module = self._fake_app_module()
        progress = self.ml.SpawnProgress()
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(self.backend, "_load_app", return_value=module), \
                mock.patch.object(self.backend, "_log_path", return_value=Path(tmp) / "launch.log"):
            self.backend.spawn(self._launch(), "gpu", 0, progress)
            log_text = (Path(tmp) / "launch.log").read_text()
            self.assertEqual(os.environ["S04_APP_NAME"], "s04-loop-l007-a")
            self.assertEqual(os.environ["S04_TIMEOUT_S"], "28800")
            self.assertEqual(os.environ["S04_GANDALF_SHA"], GANDALF_SHA)
            self.assertEqual(os.environ["S04_JAX_VERSION"], "0.9.1")
            self.assertEqual(os.environ["S04_VOLUME_ROOT"], "study04")
        self.assertIs(module.app.detach, True)
        self.assertEqual((progress.app_id, progress.call_ids, progress.attempted, progress.error),
                         ("ap-fake", ["fc-fake-1", "fc-fake-2"], 2, None))
        self.assertEqual(module.run_config.calls[0], {
            "run_id": "04_a_1", "run_set": "A", "config_relpath": f"{STUDY}/configs/a.yaml",
            "launch_id": "L007", "repo_commit": "c0ffee"})
        self.assertEqual(module.smoke.calls, [])
        self.assertIn("modal output that belongs in the log", log_text)

    def test_a_spawn_that_raises_leaves_its_run_uncertain(self) -> None:
        for interrupt in (False, True):  # an error, or the runner's SIGTERM, while the request was out
            with self.subTest(interrupt=interrupt):
                module = self._fake_app_module(fail_at=1, reaches_modal=True, interrupt=interrupt)
                launch = self._launch(3)
                progress = self.spawn(module, launch)
                self.assertEqual((progress.app_id, progress.call_ids, progress.attempted),
                                 ("ap-fake", ["fc-fake-1"], 2))
                self.assertIn("KeyboardInterrupt" if interrupt else "spawn boom", progress.error)
                self.assertEqual(module.server, ["fc-fake-1", "fc-fake-2"])  # Modal has the second call
                outcome = self.ml.apply_spawn_result(launch, progress)
                self.assertEqual((outcome.started, outcome.uncertain, outcome.released),
                                 (["04_a_1"], ["04_b_1"], ["04_c_1"]))
                self.assertEqual((launch["state"], launch["app_id"]), ("launched", "ap-fake"))
                self.assertEqual([(r["status"], r["call_id"], r["charged_hours"], r["reserved_hours"])
                                  for r in launch["runs"]],
                                 [("running", "fc-fake-1", None, 8.0), ("unknown", None, None, 8.0),
                                  ("not_launched", None, 0.0, 8.0)])
                self.assertEqual(lc.ledger_totals({"launches": [launch]})["reserved"], 16.0)

    def test_nothing_requested_releases_everything(self) -> None:
        # The app module fails to import (no request was made), or the app's image fails to
        # build (the app may exist, but no call was requested): no GPU can run.
        cases = [("import", None), ("image build", "ap-fake")]
        for name, app_id in cases:
            with self.subTest(case=name):
                launch = self._launch(2)
                progress = self.ml.SpawnProgress()
                if name == "import":
                    with mock.patch.object(self.backend, "_load_app", side_effect=ImportError("no modal_app")), \
                            mock.patch.object(self.backend, "_log_path", return_value=None):
                        self.backend.spawn(launch, "gpu", 0, progress)
                else:
                    progress = self.spawn(self._fake_app_module(fail_on_enter=True), launch)
                self.assertEqual((progress.app_id, progress.call_ids, progress.attempted), (app_id, [], 0))
                outcome = self.ml.apply_spawn_result(launch, progress)
                self.assertEqual((outcome.started, outcome.uncertain), ([], []))
                self.assertEqual(launch["state"], "launch_failed")
                self.assertIsNone(launch["app_id"])
                self.assertEqual([r["status"] for r in launch["runs"]], ["not_launched", "not_launched"])
                self.assertEqual(lc.ledger_totals({"launches": [launch]})["reserved"], 0.0)

    def test_spawn_of_smoke(self) -> None:
        module = self._fake_app_module()
        launch = self._launch(1)
        launch["runs"] = [{"run_id": "04_smoke_1", "config": None}]
        launch["run_set"] = "smoke"
        progress = self.spawn(module, launch, "smoke", 5)
        self.assertEqual(progress.call_ids, ["fc-fake-1"])
        self.assertEqual(module.smoke.calls[0]["sleep_s"], 5)

    def test_preflight_refuses_a_foreign_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            backend = self.ml.ModalBackend(Path(tmp), self.settings)
            with self.assertRaises(self.ml.EnvError):
                backend.preflight()
        self.backend.preflight()  # the launcher's own repo, modal importable

    def test_lock_pins_of_the_real_repo(self) -> None:
        sha, jax_version = self.ml.lock_pins(REAL_REPO)
        self.assertRegex(sha, r"^[0-9a-f]{40}$")
        self.assertRegex(jax_version, r"^\d+\.\d+\.\d+$")
        self.assertNotEqual(jax_version, "0.6.2")  # the < 3.11 entry is not the image's


# ---------------------------------------------------------------------------
# modal_app.py: definitions, and the container side run locally
# ---------------------------------------------------------------------------


class ModalAppDefinitionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        with mock.patch.dict(os.environ, SAFE_MODAL_ENV):
            cls.ma = import_modal_app()

    def test_functions_image_and_settings(self) -> None:
        ma = self.ma
        self.assertEqual(ma.app.name, "s04-loop-test")
        self.assertEqual(sorted(ma.app.registered_functions), ["run_config", "smoke"])
        self.assertEqual(ma.run_config.spec.gpus, "A100")
        self.assertIsNone(ma.smoke.spec.gpus)
        self.assertEqual(ma.smoke.spec.cpu, 1.0)
        self.assertEqual(list(ma.run_config.spec.volumes), ["/data"])
        self.assertEqual(list(ma.smoke.spec.volumes), ["/data"])
        self.assertEqual(ma.TIMEOUT_S, 3600)
        self.assertEqual(ma.SMOKE_TIMEOUT_S, 900)
        self.assertEqual(ma.image_requirements(GANDALF_SHA, "0.9.1")[:2], [
            "jax[cuda12]==0.9.1", f"gandalf-krmhd @ git+https://github.com/anjor/gandalf.git@{GANDALF_SHA}"])
        self.assertEqual(ma.IMAGE_ENV["S04_GANDALF_SHA"], GANDALF_SHA)
        self.assertEqual(ma.IMAGE_ENV["S04_TIMEOUT_S"], "3600")
        self.assertTrue(all(isinstance(v, str) for v in ma.IMAGE_ENV.values()))
        # The launcher and the app agree on the smoke timeout and sleep limit.
        ml = import_launcher_module()
        self.assertEqual(ml.SMOKE_TIMEOUT_HOURS * 3600, ma.SMOKE_TIMEOUT_S)
        self.assertEqual(ml.MAX_SMOKE_SLEEP_S, ma.SMOKE_MAX_SLEEP_S)
        self.assertTrue(ml.IMAGE_PYTHON_FULL.startswith(ma.PYTHON_VERSION + "."))

    def test_the_settings_the_cap_relies_on(self) -> None:
        """One A100, the launch's timeout, no retries, one input per container, bounded start-up."""
        import modal

        seen: List[Dict[str, Any]] = []
        original = modal.App.function

        def spy(app_self: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(dict(kwargs))
            return original(app_self, *args, **kwargs)

        name = "s04_modal_app_spy"
        env = dict(SAFE_MODAL_ENV, **dict(MODAL_APP_ENV, S04_TIMEOUT_S="27000"))
        try:
            with mock.patch.dict(os.environ, env), mock.patch.object(modal.App, "function", spy):
                spec = importlib.util.spec_from_file_location(name, LOOP_DIR / "modal_app.py")
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                spec.loader.exec_module(module)
        finally:
            sys.modules.pop(name, None)
        self.assertEqual(len(seen), 2)
        gpu_fn, smoke_fn = seen
        self.assertEqual((gpu_fn["gpu"], gpu_fn["timeout"], gpu_fn["retries"], gpu_fn["single_use_containers"]),
                         ("A100", 27000, 0, True))
        self.assertEqual(gpu_fn["startup_timeout"], module.STARTUP_TIMEOUT_S)
        self.assertLessEqual(module.STARTUP_TIMEOUT_S, 1800)
        self.assertEqual(list(gpu_fn["volumes"]), ["/data"])
        self.assertIsNone(smoke_fn.get("gpu"))
        self.assertEqual((smoke_fn["cpu"], smoke_fn["timeout"], smoke_fn["retries"], smoke_fn["single_use_containers"]),
                         (1.0, 900, 0, True))
        self.assertEqual(smoke_fn["startup_timeout"], module.STARTUP_TIMEOUT_S)

    def test_upload_filter(self) -> None:
        from modal.file_pattern_matcher import FilePatternMatcher

        ignore = FilePatternMatcher(*self.ma.STUDY_IGNORE)
        for path in ("data", "data/A/run/x.h5", "configs/data/x", ".loop/runner.log", "log/2026-10-01.md",
                     "figures/f.pdf", "analysis/__pycache__/a.cpython-312.pyc", "a.pyc", "derivations/b.pyc"):
            self.assertTrue(ignore(Path(path)), path)
        for path in ("configs/setA.yaml", "analysis/gates.py", "SPEC.md", "loop/modal_app.py",
                     "derivations/reference_profiles.npz", "gate_reports/G1_x.md", "logbook.md", "database.py"):
            self.assertFalse(ignore(Path(path)), path)

    def test_import_without_the_pin_fails_locally(self) -> None:
        env = {k: v for k, v in os.environ.items() if not k.startswith("S04_")}
        env.update(SAFE_MODAL_ENV, PYTHONDONTWRITEBYTECODE="1")
        code = ("import importlib.util, sys; "
                f"spec = importlib.util.spec_from_file_location('modal_app', {str(LOOP_DIR / 'modal_app.py')!r}); "
                "m = importlib.util.module_from_spec(spec); sys.modules['modal_app'] = m; spec.loader.exec_module(m)")
        proc = run_cmd([sys.executable, "-c", code], env=env, check=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("S04_GANDALF_SHA must be the 40-hex", proc.stderr)

    def test_require_gpu(self) -> None:
        def fake_jax(backend: str, platforms: List[str]) -> types.SimpleNamespace:
            devices = [types.SimpleNamespace(platform=p) for p in platforms]
            return types.SimpleNamespace(default_backend=lambda: backend, devices=lambda: devices)

        with contextlib.redirect_stdout(io.StringIO()):
            with mock.patch.dict(sys.modules, {"jax": fake_jax("gpu", ["gpu"])}):
                self.assertEqual(self.ma._require_gpu()["jax_backend"], "gpu")
            with mock.patch.dict(sys.modules, {"jax": fake_jax("cpu", ["cpu", "cuda"])}):
                self.assertEqual(self.ma._require_gpu()["jax_backend"], "cpu")
            with mock.patch.dict(sys.modules, {"jax": fake_jax("cpu", ["cpu"])}):
                with self.assertRaises(RuntimeError) as ctx:
                    self.ma._require_gpu()
        self.assertIn("no GPU", str(ctx.exception))


RUN_CODE_OK = textwrap.dedent("""\
    import json
    import time
    from pathlib import Path


    def run(cfg, out_dir, ctx):
        ckpt = Path(out_dir) / "checkpoints" / "state.json"
        resumed = ckpt.exists()
        step = json.loads(ckpt.read_text())["step"] if resumed else 0
        step += cfg["steps"]
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps({"step": step}))
        ctx.commit()
        return {"step": step, "resumed": resumed, "attempt": ctx.attempt_id, "deadline": ctx.deadline_epoch,
                "deadline_ahead": ctx.deadline_epoch > time.time(), "volume_root": str(ctx.volume_root),
                "run_set": ctx.run_set, "launch_id": ctx.launch_id}
    """)

RUN_CODE_MARKER = textwrap.dedent("""\
    from pathlib import Path


    def run(cfg, out_dir, ctx):
        (Path(out_dir) / "run_code_was_called").write_text("yes")
        return {"deadline": ctx.deadline_epoch}
    """)

RUN_CODE_SLOW = textwrap.dedent("""\
    import time


    def run(cfg, out_dir, ctx):
        time.sleep(cfg["sleep"])  # ignores ctx.deadline_epoch
        return {"slept": cfg["sleep"]}
    """)

RUN_CODE_CLEARS = textwrap.dedent("""\
    import shutil
    from pathlib import Path


    def run(cfg, out_dir, ctx):
        # "Start clean when there is no checkpoint": everything in out_dir goes.
        for path in Path(out_dir).iterdir():
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        ctx.commit()
        return {"cleared": True}
    """)

RUN_CODE_ODD_RESULT = textwrap.dedent("""\
    import enum


    class Verdict(enum.IntEnum):
        PASS = 1


    class Label(str):
        pass


    def run(cfg, out_dir, ctx):
        return {"verdict": Verdict.PASS, "label": Label("set A"), "values": (1.5, 2)}
    """)


def watchdog_threads() -> List[Any]:
    """The budget watchdogs of modal_app still alive in this process."""
    import threading

    return [t for t in threading.enumerate() if t.name == "s04-budget-watchdog" and t.is_alive()]


class ModalAppContainerTests(unittest.TestCase):
    """The functions' bodies run locally; a temp folder stands in for the volume."""

    @classmethod
    def setUpClass(cls) -> None:
        with mock.patch.dict(os.environ, SAFE_MODAL_ENV):
            cls.ma = import_modal_app()
        cls.ml = import_launcher_module()

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="s04-container-")
        tmp = Path(self._tmp.name).resolve()
        self.repo, self.data = tmp / "repo", tmp / "data"
        write(self.repo / STUDY / "PLAN.md", "# plan\n")
        code = {
            "ok": RUN_CODE_OK,
            "marker": RUN_CODE_MARKER,
            "slow": RUN_CODE_SLOW,
            "clears": RUN_CODE_CLEARS,
            "odd_result": RUN_CODE_ODD_RESULT,
            "raises_timeout": "def run(cfg, out_dir, ctx):\n    raise TimeoutError('the run code timed out')\n",
            "exits": "import sys\n\ndef run(cfg, out_dir, ctx):\n    sys.exit(0)\n",
            "interrupted": "def run(cfg, out_dir, ctx):\n    raise KeyboardInterrupt('preempted')\n",
            "cancelled": ("from modal.exception import InputCancellation\n\n"
                          "def run(cfg, out_dir, ctx):\n    raise InputCancellation('cancelled')\n"),
            "no_run": "def main():\n    pass\n",
        }
        for name, text in code.items():
            write(self.repo / STUDY / "runcode" / f"{name}.py", text)
            write(self.repo / STUDY / "configs" / f"{name}.yaml",
                  f"entrypoint: {STUDY}/runcode/{name}.py\nsteps: 5\nsleep: 3.5\n")
        self.vol = FakeVolume(root=self.data)
        self.exits: List[int] = []
        self.patches = [mock.patch.object(self.ma, "VOLUME_MOUNT", str(self.data)),
                        mock.patch.object(self.ma, "REMOTE_REPO", str(self.repo)),
                        mock.patch.object(self.ma, "volume", self.vol),
                        mock.patch.object(self.ma, "_require_gpu",
                                          return_value={"jax_backend": "gpu", "jax_devices": ["cuda:0"]}),
                        mock.patch.object(self.ma, "_hard_exit", side_effect=self.exits.append),
                        mock.patch.dict(os.environ, SAFE_MODAL_ENV)]
        for p in self.patches:
            p.start()

    def tearDown(self) -> None:
        for p in reversed(self.patches):
            p.stop()
        self._tmp.cleanup()

    def run_dir(self, run_set: str, run_id: str) -> Path:
        return self.data / "study04" / run_set / run_id

    def records_dir(self, run_set: str, run_id: str) -> Path:
        """Where the attempt records are: outside the run's own folder."""
        return self.data / "study04" / "_attempts" / run_set / run_id

    def records(self, run_set: str, run_id: str) -> List[Dict[str, Any]]:
        folder = self.records_dir(run_set, run_id)
        return [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]

    def ends(self, run_set: str, run_id: str) -> List[Dict[str, Any]]:
        return [r for r in self.records(run_set, run_id) if r["event"] == "end"]

    def earlier_attempt(self, run_id: str, start: float, end: Optional[float] = None, run_set: str = "A",
                        start_record: bool = True) -> None:
        """Records of an earlier attempt of the run, as a preempted container left them."""
        folder = self.records_dir(run_set, run_id)
        folder.mkdir(parents=True, exist_ok=True)
        base = {"attempt_id": "early", "run_id": run_id, "launch_id": "L001"}
        if start_record:
            (folder / "1_early_start.json").write_text(json.dumps(dict(base, event="start", t_epoch=start)))
        if end is not None:
            (folder / "2_early_end.json").write_text(json.dumps(dict(base, event="end", t_epoch=end,
                                                                     status="interrupted",
                                                                     duration_s=end - start)))

    def call(self, name: str, run_id: str) -> Dict[str, Any]:
        fn = self.ma.run_config.get_raw_f()
        return fn(run_id, "A", f"{STUDY}/configs/{name}.yaml", "L001", "c0ffee")

    def test_run_config_records_attempts_and_a_restart_resumes(self) -> None:
        rid = "04_ok_20261005_101500"
        first = self.call("ok", rid)
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["result"]["step"], 5)
        self.assertFalse(first["result"]["resumed"])
        self.assertTrue(first["result"]["deadline_ahead"])
        self.assertEqual(first["result"]["volume_root"], str(self.data))
        launch_json = (self.run_dir("A", rid) / "launch.json").read_text()
        # A restart (preemption) is a new attempt with the same arguments; it resumes, and its
        # deadline counts from the first attempt.
        second = self.call("ok", rid)
        self.assertEqual((second["result"]["step"], second["result"]["resumed"]), (10, True))
        self.assertNotEqual(first["result"]["attempt"], second["result"]["attempt"])
        self.assertEqual(second["result"]["deadline"], first["result"]["deadline"])
        self.assertEqual((self.run_dir("A", rid) / "launch.json").read_text(), launch_json)  # first copy kept
        provenance = json.loads(launch_json)
        self.assertEqual((provenance["repo_commit"], provenance["gandalf_commit"]), ("c0ffee", GANDALF_SHA))
        self.assertTrue((self.run_dir("A", rid) / "config.yaml").is_file())
        records = self.records("A", rid)
        self.assertEqual(sorted(r["event"] for r in records), ["end", "end", "start", "start"])
        for rec in records:
            self.assertEqual((rec["run_id"], rec["launch_id"], rec["gpu"]), (rid, "L001", "A100"))
            self.assertIn("t_epoch", rec)
        for end in self.ends("A", rid):
            self.assertEqual((end["status"], end["jax_backend"]), ("ok", "gpu"))
            self.assertIn("deadline_utc", end)
        self.assertGreaterEqual(self.vol.commits, 6)
        self.assertEqual(self.exits, [])
        # The launcher pairs these records into two complete attempts.
        summary = self.ml.summarize_attempts(records, rid)
        self.assertEqual((len(summary.attempts), summary.open, summary.last_status), (2, 0, "ok"))
        self.assertLess(summary.complete_hours, 0.01)

    def test_the_start_record_is_committed_before_the_run_code_starts(self) -> None:
        # The first commit holds the start record and nothing else: it is committed as soon as
        # it is written, before the config is read or anything else can fail or be preempted.
        rid = "04_ok_1"
        self.call("ok", rid)
        first = self.vol.snapshots[0]
        self.assertEqual(len(first), 1, first)
        self.assertTrue(first[0].startswith(f"study04/_attempts/A/{rid}/") and first[0].endswith("_start.json"),
                        first)
        # Nothing of the accounting is inside the run's own folder, which the run code owns.
        self.assertFalse((self.run_dir("A", rid) / "attempts").exists())

    def test_the_first_attempt_gets_the_timeout_less_the_margin(self) -> None:
        rid = "04_marker_1"
        before = time.time()
        result = self.call("marker", rid)
        start = [r for r in self.records("A", rid) if r["event"] == "start"][0]["t_epoch"]
        self.assertAlmostEqual(result["result"]["deadline"], start + max(3600 - 900, 1800), places=3)
        self.assertGreaterEqual(start, before)

    def test_a_restart_gets_only_what_is_left_of_the_budget(self) -> None:
        rid = "04_marker_2"
        first_start = time.time() - 1800  # the first attempt started 30 min ago and was preempted
        self.earlier_attempt(rid, first_start, first_start + 1500)
        result = self.call("marker", rid)
        self.assertAlmostEqual(result["result"]["deadline"], first_start + 2700, places=3)
        self.assertTrue((self.run_dir("A", rid) / "run_code_was_called").exists())

    def test_an_exhausted_budget_ends_the_attempt_without_running_the_code(self) -> None:
        for name, start_record in (("start record", True), ("end record only", False)):
            with self.subTest(case=name):
                rid = f"04_marker_late_{int(start_record)}"
                first_start = time.time() - 3000  # the deadline (first start + 2700 s) has passed
                self.earlier_attempt(rid, first_start, first_start + 2900, start_record=start_record)
                with self.assertRaises(RuntimeError) as ctx, contextlib.redirect_stderr(io.StringIO()):
                    self.call("marker", rid)
                self.assertIn("budget exhausted", str(ctx.exception))
                self.assertNotIsInstance(ctx.exception, TimeoutError)
                self.assertFalse((self.run_dir("A", rid) / "run_code_was_called").exists())
                late = [e for e in self.ends("A", rid) if e["attempt_id"] != "early"]
                self.assertEqual(len(late), 1)
                self.assertEqual(late[0]["status"], "error")
                self.assertTrue(late[0]["error"].startswith("budget exhausted"), late[0]["error"])
                self.assertLess(late[0]["duration_s"], 60)
        self.assertEqual(self.exits, [])

    def test_the_watchdog_stops_an_attempt_that_runs_past_the_budget(self) -> None:
        rid = "04_slow_1"
        with mock.patch.object(self.ma, "TIMEOUT_S", 2), contextlib.redirect_stderr(io.StringIO()) as err:
            self.call("slow", rid)  # sleeps 3.5 s, past the 2 s budget; the fake exit returns
        self.assertEqual(self.exits, [self.ma.WATCHDOG_EXIT_CODE])
        ends = self.ends("A", rid)
        self.assertEqual(len(ends), 1, ends)  # the attempt's own end record was not written again
        self.assertEqual(ends[0]["status"], "error")
        self.assertTrue(ends[0]["error"].startswith("budget exhausted (watchdog)"), ends[0]["error"])
        self.assertAlmostEqual(ends[0]["duration_s"], 2.0, delta=1.0)
        self.assertIn("budget exhausted (watchdog)", err.getvalue())

    def test_run_code_that_clears_its_folder_cannot_reset_the_budget(self) -> None:
        # The run code empties out_dir; the records are elsewhere and stay.
        rid = "04_clears_1"
        self.assertEqual(self.call("clears", rid)["status"], "ok")
        self.assertEqual(list(self.run_dir("A", rid).iterdir()), [])
        self.assertEqual(sorted(r["event"] for r in self.records("A", rid)), ["end", "start"])
        # The verifier's case: an earlier attempt started 3000 s ago (the deadline, first start
        # + 2700 s, has passed), and the run's folder is gone. The restart still ends at once.
        rid = "04_marker_cleared"
        self.earlier_attempt(rid, time.time() - 3000, None)
        shutil.rmtree(self.run_dir("A", rid), ignore_errors=True)
        with self.assertRaises(RuntimeError) as ctx, contextlib.redirect_stderr(io.StringIO()):
            self.call("marker", rid)
        self.assertIn("budget exhausted", str(ctx.exception))
        self.assertFalse((self.run_dir("A", rid) / "run_code_was_called").exists())

    def test_the_watchdog_stops_the_container_even_if_its_record_hangs(self) -> None:
        # The third commit (the watchdog's end record) hangs; the watchdog waits for it at most
        # WATCHDOG_RECORD_TIMEOUT_S and then stops the container anyway.
        rid = "04_slow_hang"
        exits: List[Tuple[int, float]] = []
        self.vol.slow_after, self.vol.slow_s = 2, 2.5
        t0 = time.time()
        with mock.patch.object(self.ma, "TIMEOUT_S", 2), \
                mock.patch.object(self.ma, "WATCHDOG_RECORD_TIMEOUT_S", 0.2), \
                mock.patch.object(self.ma, "_hard_exit", side_effect=lambda code: exits.append((code, time.time()))), \
                contextlib.redirect_stderr(io.StringIO()):
            self.call("slow", rid)  # sleeps 3.5 s; the fake exit returns
            for thread in [t for t in threading.enumerate() if t.name == "s04-watchdog-record"]:
                thread.join(10)
        self.assertEqual([code for code, _t in exits], [self.ma.WATCHDOG_EXIT_CODE])
        self.assertLess(exits[0][1] - t0, 2.0 + 0.2 + 1.0)  # at the budget's end plus the wait, not later
        self.assertTrue(self.vol.slow_done and exits[0][1] < self.vol.slow_done[0], "stopped while the commit hung")
        ends = self.ends("A", rid)
        self.assertEqual(len(ends), 1, ends)  # written before the commit hung; the attempt's own was not
        self.assertTrue(ends[0]["error"].startswith("budget exhausted (watchdog)"))

    def test_the_watchdog_leaves_an_attempt_that_ends_in_time(self) -> None:
        rid = "04_marker_3"
        with mock.patch.object(self.ma, "TIMEOUT_S", 2):
            self.call("marker", rid)
            # The watchdog stands down as soon as the attempt ends, well before the budget does.
            end = time.time() + 0.5
            while time.time() < end and watchdog_threads():
                time.sleep(0.02)
            self.assertEqual(watchdog_threads(), [])
            time.sleep(2.0)  # past the budget
        self.assertEqual(self.exits, [])
        self.assertEqual([e["status"] for e in self.ends("A", rid)], ["ok"])

    def test_no_gpu_ends_the_attempt_before_the_run_code(self) -> None:
        rid = "04_marker_cpu"
        with mock.patch.object(self.ma, "_require_gpu", side_effect=RuntimeError("no GPU: backend 'cpu'")), \
                self.assertRaises(RuntimeError) as ctx, contextlib.redirect_stderr(io.StringIO()):
            self.call("marker", rid)
        self.assertIn("no GPU", str(ctx.exception))
        self.assertFalse((self.run_dir("A", rid) / "run_code_was_called").exists())
        ends = self.ends("A", rid)
        self.assertEqual([e["status"] for e in ends], ["error"])
        self.assertIn("no GPU", ends[0]["error"])

    def test_run_code_exceptions_come_back_as_runtime_error(self) -> None:
        rid = "04_raises_timeout_20261005_101500"
        container_log = io.StringIO()  # the container prints the traceback for `modal app logs`
        with self.assertRaises(RuntimeError) as ctx, contextlib.redirect_stderr(container_log):
            self.call("raises_timeout", rid)
        self.assertIn("TimeoutError: the run code timed out", container_log.getvalue())
        # Never a builtin TimeoutError: the launcher reads that as "still running".
        self.assertNotIsInstance(ctx.exception, TimeoutError)
        self.assertIn("TimeoutError: the run code timed out", str(ctx.exception))
        end = self.ends("A", rid)[0]
        self.assertEqual(end["status"], "error")
        self.assertIn("TimeoutError", end["error"])
        self.assertIn("Traceback", end["traceback"])
        with self.assertRaises(RuntimeError) as ctx, contextlib.redirect_stderr(io.StringIO()):
            self.call("no_run", "04_no_run_1")
        self.assertIn("defines no run", str(ctx.exception))

    def test_system_exit_from_run_code_is_wrapped_too(self) -> None:
        rid = "04_exits_1"
        with self.assertRaises(RuntimeError) as ctx, contextlib.redirect_stderr(io.StringIO()):
            self.call("exits", rid)
        self.assertIn("SystemExit", str(ctx.exception))
        self.assertEqual([e["status"] for e in self.ends("A", rid)], ["error"])

    def test_preemption_and_cancellation_pass_through_after_the_end_record(self) -> None:
        from modal.exception import InputCancellation

        for name, exc_type in (("interrupted", KeyboardInterrupt), ("cancelled", InputCancellation)):
            with self.subTest(case=name):
                rid = f"04_{name}_1"
                with self.assertRaises(exc_type):
                    self.call(name, rid)
                summary = self.ml.summarize_attempts(self.records("A", rid), rid)
                self.assertEqual((summary.open, summary.last_status), (0, "interrupted"))

    def test_the_result_comes_back_as_plain_json(self) -> None:
        result = self.call("odd_result", "04_odd_result_1")["result"]
        self.assertEqual(result, {"verdict": 1, "label": "set A", "values": [1.5, 2]})
        self.assertIs(type(result["verdict"]), int)
        self.assertIs(type(result["label"]), str)
        self.assertEqual(self.ma._jsonable(object()).startswith("<object object"), True)

    def test_smoke_runs_on_cpu_and_writes_its_report(self) -> None:
        with mock.patch.dict(os.environ, {}):
            result = self.ma.smoke.get_raw_f()("04_smoke_20261005_101500", "smoke", "L001", "c0ffee", sleep_s=0)
        info = result["result"]
        self.assertEqual(result["status"], "ok")
        self.assertNotIn("import_error", info)
        self.assertTrue(info["repo_mount_ok"])
        self.assertEqual(info["gandalf_sha_env"], GANDALF_SHA)
        self.assertIn("jax_version", info)
        self.assertIn("krmhd_version", info)
        self.assertNotIn("step_error", info)
        self.assertTrue(info["step"]["finite"])
        report = json.loads((self.run_dir("smoke", "04_smoke_20261005_101500") / "smoke.json").read_text())
        self.assertEqual(report["run_id"], "04_smoke_20261005_101500")
        summary = self.ml.summarize_attempts(self.records("smoke", "04_smoke_20261005_101500"),
                                             "04_smoke_20261005_101500")
        self.assertEqual((summary.open, summary.last_status), (0, "ok"))
        self.assertEqual(self.records("smoke", "04_smoke_20261005_101500")[0]["gpu"], "none")
        self.assertEqual(self.exits, [])


# ---------------------------------------------------------------------------
# loopcommon
# ---------------------------------------------------------------------------


SECTIONED = textwrap.dedent("""\
    # Title

    ## 1. Intro
    intro text

    ## 2. Question
    question text
    ### 2.1 Sub
    sub text
    ```python
    ## not a heading, inside a fence
    x = 1
    ```
    end of two

    ## 3. Next
    next text
    #### 3.0.0.1 Deep
    deep text
    """)


def _launch(lid: str, state: str, runs: List[Dict[str, Any]], **extra: Any) -> Dict[str, Any]:
    launch = {"launch_id": lid, "state": state, "kind": "gpu", "runs": runs, "app_id": None, "app_name": None}
    launch.update(extra)
    return launch


def _run(rid: str, status: str, reserved: float, charged: Optional[float] = None) -> Dict[str, Any]:
    return {"run_id": rid, "status": status, "reserved_hours": reserved, "charged_hours": charged}


class LoopcommonTests(unittest.TestCase):

    def test_parse_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.env"
            path.write_text(textwrap.dedent("""\
                # comment
                COMPUTE_CAP_A100_HOURS=200

                export FOO = bar  # trailing comment
                QUOTED="a # not a comment"
                SINGLE='x'
                TILDE=~/repos
                EMPTY=
                NOEQUALS
                """))
            values = lc.parse_env_file(path)
        self.assertEqual(values["COMPUTE_CAP_A100_HOURS"], "200")
        self.assertEqual(values["FOO"], "bar")
        self.assertEqual(values["QUOTED"], "a # not a comment")
        self.assertEqual(values["SINGLE"], "x")
        self.assertEqual(values["TILDE"], os.path.expanduser("~/repos"))
        self.assertEqual(values["EMPTY"], "")
        self.assertNotIn("NOEQUALS", values)

    def test_compute_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write(repo / lc.CONFIG_REL, "COMPUTE_CAP_A100_HOURS=150.5\n")
            self.assertEqual(lc.compute_cap(repo), 150.5)
            write(repo / lc.CONFIG_REL, "OTHER=1\n")
            with self.assertRaises(ValueError):
                lc.compute_cap(repo)
        self.assertGreater(lc.compute_cap(REAL_REPO), 0)

    def test_extract_section_boundaries(self) -> None:
        two = lc.extract_section(SECTIONED, "2. Question")
        self.assertTrue(two.startswith("## 2. Question\nquestion text\n### 2.1 Sub"))
        self.assertIn("## not a heading, inside a fence", two)  # fenced lines are not headings
        self.assertTrue(two.endswith("end of two"))
        self.assertNotIn("3. Next", two)
        sub = lc.extract_section(SECTIONED, "2.1")
        self.assertTrue(sub.startswith("### 2.1 Sub"))
        self.assertTrue(sub.endswith("end of two"))  # a level-3 section runs to the next level <= 3
        last = lc.extract_section(SECTIONED, "3. Next")
        self.assertTrue(last.endswith("deep text"))  # deeper headings belong to it; runs to EOF
        self.assertEqual(lc.extract_section(SECTIONED, "1."), "## 1. Intro\nintro text")
        self.assertIsNone(lc.extract_section(SECTIONED, "not a heading"))
        self.assertIsNone(lc.extract_section(SECTIONED, "9. Missing"))
        # Trailing whitespace and blank lines at the edges do not change the section.
        padded = SECTIONED.replace("question text\n", "question text   \n").replace("## 3. Next", "\n\n## 3. Next")
        self.assertEqual(lc.extract_section(padded, "2. Question"), two)
        # A change inside the section does.
        self.assertNotEqual(lc.extract_section(SECTIONED.replace("sub text", "sub text!"), "2. Question"), two)

    def test_frozen_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            write(repo / "doc.md", SECTIONED)
            key = "doc.md#2. Question"
            sha = lc.frozen_key_hash(repo, key)
            self.assertEqual(sha, lc.sha256_text(lc.extract_section(SECTIONED, "2. Question")))
            self.assertEqual(lc.frozen_key_hash(repo, "doc.md"), lc.sha256_file(repo / "doc.md"))
            self.assertIsNone(lc.frozen_key_hash(repo, "doc.md#9. Missing"))
            self.assertIsNone(lc.frozen_key_hash(repo, "missing.md#2. Question"))
            entries = {key: sha}
            self.assertEqual(lc.check_frozen(repo, entries), [])
            write(repo / "doc.md", SECTIONED.replace("## 1. Intro\nintro text", "## 1. Intro\nchanged"))
            self.assertEqual(lc.check_frozen(repo, entries), [])  # outside the section
            write(repo / "doc.md", SECTIONED.replace("question text", "question changed"))
            self.assertEqual(len(lc.check_frozen(repo, entries)), 1)
            self.assertIn("changed", lc.check_frozen(repo, entries)[0])
            (repo / "doc.md").unlink()
            self.assertIn("missing", lc.check_frozen(repo, entries)[0])
            write(repo / lc.FROZEN_REL, json.dumps({"note": "n", "entries": entries}))
            self.assertEqual(lc.load_frozen_manifest(repo), entries)
        self.assertEqual(lc.load_frozen_manifest(Path("/nonexistent")), {})

    def test_ledger_integrity_round_trip(self) -> None:
        ledger = lc.empty_ledger()
        self.assertEqual(lc.verify_ledger(ledger), [])  # an empty ledger needs no hash
        ledger["launches"].append(_launch("L001", "launched", [_run("r1", "running", 3.0)]))
        text = lc.ledger_text(ledger)
        data = json.loads(text)
        self.assertEqual(data["integrity"], lc.compute_integrity(data))
        self.assertEqual(lc.verify_ledger(data), [])
        self.assertEqual(lc.ledger_text(data), text)  # stable format
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ledger.json"
            lc.save_ledger(path, data)
            self.assertEqual(lc.load_ledger(path), data)
            self.assertEqual(lc.load_ledger(Path(tmp) / "absent.json"), lc.empty_ledger())
        tampered = json.loads(text)
        tampered["launches"][0]["runs"][0]["reserved_hours"] = 0.5
        self.assertTrue(any("integrity" in p for p in lc.verify_ledger(tampered)))
        unhashed = json.loads(text)
        del unhashed["integrity"]
        self.assertTrue(any("no integrity hash" in p for p in lc.verify_ledger(unhashed)))
        self.assertEqual(lc.verify_ledger(unhashed, require_integrity=False), [])

    def test_ledger_structure_checks(self) -> None:
        ledger = lc.empty_ledger()
        ledger["launches"] = [
            _launch("L001", "launched", [_run("r1", "running", 1.0), _run("r1", "weird", -1.0)]),
            _launch("L001", "bogus", [_run("r2", "finished", 1.0, charged=-2.0), _run("r3", "failed", "x")]),
        ]
        ledger["schema"] = 2
        problems = " | ".join(lc.verify_ledger(ledger, require_integrity=False))
        for needle in ("schema", "duplicate launch_id L001", "bad state 'bogus'", "duplicate run_id r1",
                       "bad status 'weird'", "negative reservation", "negative charge", "non-numeric hours"):
            self.assertIn(needle, problems)

    def test_totals_and_status_line(self) -> None:
        ledger = lc.empty_ledger()
        ledger["launches"] = [
            _launch("L001", "launched", [_run("a", "finished", 3.0, charged=1.5), _run("b", "running", 3.0),
                                         _run("c", "not_launched", 3.0, charged=0.0)]),
            _launch("L002", "launch_failed", [_run("d", "not_launched", 4.0)]),
            _launch("L003", "launched", [_run("s", "running", 0.0)], kind="smoke"),
        ]
        totals = lc.ledger_totals(ledger)
        self.assertEqual((totals["used"], totals["reserved"], totals["launches"], totals["in_flight"]),
                         (1.5, 3.0, 3.0, 2.0))
        self.assertEqual(lc.status_line(20.0, ledger), "Compute cap 20.0 A100-h: used 1.50, reserved 3.00, "
                                                       "left 15.50 (launches 3, runs in flight 2)")

    def test_ledger_frozen_entries_skip_failed_launches(self) -> None:
        ledger = lc.empty_ledger()
        ledger["launches"] = [
            _launch("L001", "launched", [], frozen={"SPEC.md#7a.A": "aaa"}),
            _launch("L002", "launch_failed", [], frozen={"SPEC.md#7a.B": "bbb"}),
            _launch("L003", "closed", [], frozen=None),
        ]
        self.assertEqual(lc.ledger_frozen_entries(ledger), {"SPEC.md#7a.A": "aaa"})

    def test_unknown_running_apps(self) -> None:
        ledger = lc.empty_ledger()
        ledger["launches"] = [
            _launch("L001", "launched", [], app_id="ap-1", app_name="s04-loop-l001-a"),
            _launch("L002", "reserved", [], app_name="s04-loop-l002-b"),
            _launch("L003", "launch_failed", [], app_name="s04-loop-l003-c"),
        ]
        apps = [
            {"App ID": "ap-1", "Description": "s04-loop-l001-a", "State": "ephemeral (detached)"},
            {"App ID": "ap-2", "Description": "s04-loop-l002-b", "State": "ephemeral (detached)"},
            {"App ID": "ap-3", "Description": "s04-loop-l003-c", "State": "ephemeral (detached)"},
            {"App ID": "ap-4", "Description": "s04-loop-l004-d", "State": "stopped"},
            {"App ID": "ap-5", "Description": "krmhd-hermite-128", "State": "deployed"},
            {"app_id": "ap-6", "description": "s04-loop-l005-e", "state": "deployed"},
            "not a dict",
        ]
        unknown = lc.unknown_running_apps(apps, ledger, "s04-loop")
        self.assertEqual([row["id"] for row in unknown], ["ap-3", "ap-6"])
        self.assertEqual(lc.unknown_running_apps(None, ledger, "s04-loop"), [])
        for state, running in (("stopped", False), ("Stopping", False), ("disabled", False), ("", False),
                               ("deployed", True), ("ephemeral (detached)", True), ("running", True)):
            self.assertEqual(lc.is_running_state(state), running, state)


if __name__ == "__main__":
    unittest.main()
