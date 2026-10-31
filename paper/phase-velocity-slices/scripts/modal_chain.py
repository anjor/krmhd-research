#!/usr/bin/env python
"""Run a forced-turbulence config on Modal past the 24 h limit of one call.

Spawns scripts/run_forced_turbulence.py on a deployed GANDALF app (its run_script_gpu function), waits for the call,
and while the run is unfinished spawns it again.  The GPU type and memory are those of the deployment, which Modal
fixes when the app is deployed: from a GANDALF clone at the pinned release,
    JAX_VERSION=0.9.1 MODAL_SCRIPT_GPU=B200 MODAL_SCRIPT_MEMORY_MB=65536 modal deploy --name gandalf-krmhd-b200 modal_app.py  Every call gets
--resume (the first starts a new run, later ones continue it in place) and --max-wall-hours (default 23), so that it
writes resume.npz and exits with status 3 before the limit.  The chain stops when the run finishes (status 0), when
a call ends any other way (an error, a non-zero status other than 3, a call that fails or is cancelled), when a paused
call made no progress, or after --max-calls calls.  The live call ID is kept in data/modal_chain_<subdir>.json: a
restarted chain waits for that call instead of spawning a second one into the same run folder.  Each event is printed
and appended to --log.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/scripts/modal_chain.py paper/phase-velocity-slices/configs/forced_beta0p3_n256.yaml \\
        --subdir p1-n256/beta0p3 --app gandalf-krmhd-b200 [--include <file> ...] [--max-calls 20]
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent.parent
DATA = HERE.parent / "data"
FUNCTION = "run_script_gpu"
SCRIPT = "paper/phase-velocity-slices/scripts/run_forced_turbulence.py"
PAUSED = 3


def pack(paths: list[str]) -> bytes:
    """Gzipped tarball of files and folders (relative to the repository root) in the layout GANDALF's run_script
    unpacks (scripts/modal_support.unpack_project): regular files only, without __pycache__."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in paths:
            p = REPO / rel
            files = [p] if p.is_file() else sorted(f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts)
            if not files:
                raise SystemExit(f"nothing to pack at {rel}")
            for f in files:
                tar.add(f, arcname=f.relative_to(REPO).as_posix(), recursive=False)
    return buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("config", help="config path relative to the repository root")
    ap.add_argument("--subdir", required=True, help="output folder on the gandalf-results volume")
    ap.add_argument("--app", default="gandalf-krmhd", help="deployed GANDALF app (fixes the GPU type and memory)")
    ap.add_argument("--include", nargs="*", default=[], help="further files to pack (repository-relative)")
    ap.add_argument("--max-wall-hours", type=float, default=23.0)
    ap.add_argument("--max-calls", type=int, default=20)
    ap.add_argument("--poll", type=float, default=300.0, help="seconds between checks of the live call")
    ap.add_argument("--log", default=None, help="event log (default data/modal_chain_<subdir>.log)")
    args = ap.parse_args()

    import modal

    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", args.subdir)
    state_path = DATA / f"modal_chain_{tag}.json"
    log_path = Path(args.log) if args.log else DATA / f"modal_chain_{tag}.log"

    def event(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(log_path, "a") as fh:
            fh.write(line + "\n")

    state = json.loads(state_path.read_text()) if state_path.exists() else {"calls": []}

    def save_state() -> None:
        state_path.write_text(json.dumps(state, indent=1))

    data = pack(["paper/phase-velocity-slices/scripts", args.config, *args.include])
    cmd = [args.config, "--base", "{output_dir}", "--resume", "--max-wall-hours", f"{args.max_wall_hours:g}"]
    env = {"JAX_PLATFORMS": "cuda", "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95"}
    fn = modal.Function.from_name(args.app, FUNCTION)
    last_t = -1.0
    while True:
        live = state.get("live")
        if live is None:
            if len(state["calls"]) >= args.max_calls:
                event(f"stopping: {args.max_calls} calls made")
                return 1
            call = fn.spawn(data, SCRIPT, cmd, args.subdir, 600.0, env)
            live = state["live"] = call.object_id
            state["calls"].append(dict(id=live, app=args.app, spawned=time.strftime("%Y-%m-%d %H:%M:%S")))
            save_state()
            event(f"spawned {live} on {args.app} ({len(data) / 1024:.0f} KiB packed): "
                  f"{SCRIPT} {' '.join(cmd)} -> gandalf-results:{args.subdir}")
        else:
            event(f"waiting for live call {live}")
        call = modal.FunctionCall.from_id(live)
        while True:
            try:
                result = call.get(timeout=args.poll)
                break
            except TimeoutError:
                continue
            except Exception as exc:          # failed, cancelled, expired or timed out
                state["live"] = None
                state["calls"][-1].update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), outcome=f"{type(exc).__name__}: {exc}")
                save_state()
                event(f"STOP: call {live} ended with {type(exc).__name__}: {exc}")
                return 1
        rc = result.get("returncode")
        tail = result.get("log_tail") or ""
        state["live"] = None
        state["calls"][-1].update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), status=result.get("status"), returncode=rc,
                                  elapsed_s=result.get("elapsed_time"))
        save_state()
        if rc == 0:
            event(f"FINISHED: call {live} returned 0 after {result.get('elapsed_time', 0):.0f} s\n{tail[-2000:]}")
            return 0
        m = re.findall(r"paused at t = ([0-9.]+), step (\d+)", tail)
        if rc == PAUSED and m:
            t = float(m[-1][0])
            event(f"paused: call {live} at t = {t}, step {m[-1][1]} after {result.get('elapsed_time', 0):.0f} s")
            if t <= last_t:
                event(f"STOP: no progress since the previous call (t = {last_t})")
                return 1
            last_t = t
            continue
        event(f"STOP: call {live} status {result.get('status')}, return code {rc}, error {result.get('error')}\n{tail[-3000:]}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
