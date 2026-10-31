#!/usr/bin/env python
"""Check of run_forced_turbulence.py --resume: a run that is killed after a resume file exists and then continued in
place with --resume gives the same outputs (every float array of the run's .npz, to 1e-12) as the same run left
uninterrupted, in one run folder, with resume.npz removed at the end.  The same for a run paused twice by
--max-wall-hours 0 (exit status 3 after one step of each invocation, between samples and between the full time-step
limits) and then continued to the end.  The test config is forced_beta1.yaml at 32^3,
M = 8, from rest, for 24 steps (about a minute on a laptop CPU, mostly JIT compilation).  Exits with status 1 on failure.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/scripts/test_resume.py
"""
from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCRIPT = str(HERE / "run_forced_turbulence.py")


def make_config(path: Path) -> None:
    """forced_beta1.yaml at 32^3, M = 8, from rest, to t = 0.12 with a restart every 0.02."""
    s = (HERE.parent / "configs" / "forced_beta1.yaml").read_text()
    s = s.replace("label: forced-beta1\n", "label: resume-test\n", 1)
    s = s.replace("\n  N: 64\n", "\n  N: 32\n", 1)
    s = s.replace("  M: 32\n", "  M: 8\n", 1)
    i, j = s.index("\ninit:"), s.index("\ntime:")
    s = s[:i] + "\ninit:\n  type: rest\n" + s[j:]
    for pat, rep in [(r"  t_end: 53\.0[^\n]*\n", "  t_end: 0.12\n"), (r"  sample_every: \d+", "  sample_every: 3"),
                     (r"  restart_every: [0-9.]+", "  restart_every: 0.02"),
                     (r"  snapshot_every: [0-9.]+", "  snapshot_every: 0.05"),
                     (r"  print_every: \d+", "  print_every: 5"), (r"  recompute_every: \d+", "  recompute_every: 4")]:
        s, k = re.subn(pat, rep, s, count=1)
        assert k == 1, pat
    path.write_text(s)


def load(base: Path) -> dict[str, np.ndarray]:
    d = next(base.glob("P1_resume-test_*"))
    with np.load(d / f"{d.name}.npz") as z:
        return {k: z[k] for k in z.files}


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> list:
    """Arrays of two runs' .npz that differ in shape, type or (floats) by more than 1e-12 relative."""
    bad = []
    for k in a:
        if a[k].shape != b[k].shape or a[k].dtype != b[k].dtype:
            bad.append((k, f"{a[k].dtype}{a[k].shape} vs {b[k].dtype}{b[k].shape}"))
        elif a[k].dtype.kind in "fc" and not np.allclose(a[k], b[k], rtol=1e-12, atol=1e-300, equal_nan=True):
            bad.append((k, float(np.nanmax(np.abs(a[k] - b[k])))))
    return bad


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="resume_test_"))
    try:
        cfg = tmp / "resume_test.yaml"
        make_config(cfg)
        cmd = [sys.executable, SCRIPT, str(cfg)]
        full, split, paused = tmp / "full", tmp / "split", tmp / "paused"
        full.mkdir()
        split.mkdir()
        paused.mkdir()

        # 1. uninterrupted
        subprocess.run(cmd + ["--base", str(full)], check=True, capture_output=True)

        # 2. killed once a resume file with t >= 0.06 exists, then continued with --resume
        p = subprocess.Popen(cmd + ["--base", str(split)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)
        t_kill = None
        while p.poll() is None and t_kill is None:
            for f in split.glob("P1_resume-test_*/resume.npz"):
                try:
                    with np.load(f) as z:
                        if float(z["t"]) >= 0.06:
                            t_kill = float(z["t"])
                except Exception:   # the file may be half written
                    pass
            if t_kill is not None:
                os.killpg(p.pid, signal.SIGKILL)
            time.sleep(0.2)
        p.wait()
        if t_kill is None:
            print("FAIL: the run finished before it could be interrupted")
            return 1
        dirs = list(split.glob("P1_resume-test_*"))
        assert len(dirs) == 1 and not (dirs[0] / f"{dirs[0].name}.json").exists(), "the run was not left unfinished"
        out = subprocess.run(cmd + ["--base", str(split), "--resume"], check=True, capture_output=True, text=True)
        checks = {"resumed in place": "resuming in place" in out.stdout,
                  "one run folder": len(list(split.glob("P1_resume-test_*"))) == 1,
                  "resume.npz removed": not (dirs[0] / "resume.npz").exists()}

        # 3. paused twice by --max-wall-hours 0 (after one step of each invocation), then continued to the end
        codes, t_pause = [], []
        for extra in (["--max-wall-hours", "0"], ["--resume", "--max-wall-hours", "0"], ["--resume"]):
            r = subprocess.run(cmd + ["--base", str(paused)] + extra, capture_output=True, text=True)
            codes.append(r.returncode)
            t_pause += re.findall(r"paused at t = ([0-9.]+), step (\d+)", r.stdout)
        pdirs = list(paused.glob("P1_resume-test_*"))
        checks.update({"pause exit statuses 3, 3, 0": codes == [3, 3, 0],
                       "paused after steps 1 and 2": [int(x[1]) for x in t_pause] == [1, 2],
                       "paused run in one folder": len(pdirs) == 1,
                       "paused run: resume.npz removed": not (pdirs[0] / "resume.npz").exists()})

        # 4. compare
        a, b, c = load(full), load(split), load(paused)
        bad = compare(a, b)
        bad_p = compare(a, c)
        ok = all(checks.values()) and not bad and not bad_p
        print(f"interrupted at t = {t_kill:.4f}; paused at {t_pause}; checks {checks}; {len(a)} arrays compared; "
              f"mismatches: killed {bad if bad else 'none'}, paused {bad_p if bad_p else 'none'}")
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
