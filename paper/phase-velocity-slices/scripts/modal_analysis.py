#!/usr/bin/env python
"""Run an analysis script on Modal next to the data, for runs whose snapshots are too large to download.

Spawns one of the paper's scripts on the CPU function (run_script_cpu) of a deployed GANDALF app, with the paper's
scripts/, analysis/ and configs/ folders packed as in modal_chain.py, waits for it, and downloads the small outputs it
wrote on the gandalf-results volume.  The script's arguments name volume paths as the container sees them
(/results/...).  The CPU count and memory are those of the deployment (MODAL_SCRIPT_CPU, MODAL_SCRIPT_MEMORY_MB).
The output of the script goes to modal_run.log in --subdir on the volume and its last lines are printed.  Network errors while waiting are retried as in modal_chain.py.  On the
B200 deployment (8 CPUs, 64 GB) h1_fieldlines.py takes about 3.5 min for a 128^3 snapshot, matching a local run to
3e-10, and 22 min for a 256^3 snapshot; snapshots can go to separate calls in parallel.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/scripts/modal_analysis.py --app gandalf-krmhd-b200-cu13 \\
        --subdir p1-n256/beta0p3/analysis \\
        --fetch 'p1-n256/beta0p3/<run ID>/fieldlines_snap_*' --to paper/phase-velocity-slices/data/<run ID> \\
        -- paper/phase-velocity-slices/analysis/h1_fieldlines.py /results/p1-n256/beta0p3/<run ID> --snaps 1,2
"""
from __future__ import annotations

import argparse
import fnmatch
import posixpath
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modal_chain import REPO, pack, wait_for  # noqa: E402

FUNCTION = "run_script_cpu"
PACKED = ["paper/phase-velocity-slices/scripts", "paper/phase-velocity-slices/analysis",
          "paper/phase-velocity-slices/configs"]


def fetch(vol, pattern: str, dest: Path) -> list[Path]:
    """Download the files on the volume matching a glob in their file name (pattern relative to the volume root)."""
    folder, name = posixpath.split(pattern.strip("/"))
    got = []
    for entry in vol.listdir(folder):
        base = posixpath.basename(entry.path)
        if not fnmatch.fnmatch(base, name):
            continue
        out = dest / base
        with open(out, "wb") as fh:
            for chunk in vol.read_file(entry.path):
                fh.write(chunk)
        got.append(out)
    return got


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        raise SystemExit("usage: modal_analysis.py [options] -- <script> [script arguments]")
    k = argv.index("--")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--app", default="gandalf-krmhd-b200-cu13", help="deployed GANDALF app (fixes the CPUs and memory)")
    ap.add_argument("--subdir", required=True, help="folder on the volume for modal_run.log")
    ap.add_argument("--fetch", action="append", default=[], help="volume glob of outputs to download (repeatable)")
    ap.add_argument("--to", default=None, help="local folder for the downloads")
    ap.add_argument("--volume", default="gandalf-results")
    ap.add_argument("--poll", type=float, default=300.0, help="seconds between checks of the call")
    args = ap.parse_args(argv[:k])
    script, script_args = argv[k + 1], argv[k + 2:]
    if args.fetch and not args.to:
        raise SystemExit("--fetch needs --to")

    import modal

    data = pack([*PACKED, script] if not any(script.startswith(p + "/") for p in PACKED) else PACKED)
    fn = modal.Function.from_name(args.app, FUNCTION)
    call = fn.spawn(data, script, script_args, args.subdir, 600.0, {"JAX_PLATFORMS": "cpu"})
    t0 = time.time()
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} spawned {call.object_id} on {args.app}/{FUNCTION}: "
          f"{script} {' '.join(script_args)}", flush=True)
    result = wait_for(call.object_id, args.poll, lambda msg: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}", flush=True),
                      tick=lambda: print(f"  ... {time.time() - t0:.0f} s", flush=True))
    print(result.get("log_tail") or result.get("traceback") or "", flush=True)
    rc = result.get("returncode")
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {call.object_id}: status {result.get('status')}, return code {rc}, "
          f"{result.get('elapsed_time', 0):.0f} s in the container", flush=True)
    if rc != 0:
        return 1
    if args.fetch:
        dest = Path(args.to) if Path(args.to).is_absolute() else REPO / args.to
        dest.mkdir(parents=True, exist_ok=True)
        vol = modal.Volume.from_name(args.volume)
        for pat in args.fetch:
            got = fetch(vol, pat, dest)
            print(f"fetched {len(got)} file(s) for {pat}: {', '.join(p.name for p in got) or 'none'}", flush=True)
            if not got:
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
