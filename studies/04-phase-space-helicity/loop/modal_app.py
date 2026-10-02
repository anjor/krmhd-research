"""Study 04 Modal app: the one function in the loop that asks for a GPU.

Only ``loop/modal_launch.py`` imports this module. The launcher sets the ``S04_*``
environment variables, imports this file under the module name ``modal_app`` and starts
the app detached, so the calls keep running after the launcher (and the agent session)
exits. Never ``modal run`` or ``modal deploy`` this file: the loop's permission rules deny
both, and a run started that way would bypass the compute ledger and the cap.

Run code interface
------------------
``run_config`` runs one committed config on one A100, with no retries and the timeout
the launch chose. The config is a YAML mapping with a string key ``entrypoint``: a
repo-relative ``.py`` file under ``studies/04-phase-space-helicity/`` that defines::

    def run(cfg: dict, out_dir: pathlib.Path, ctx: RunContext) -> object

- ``cfg`` is the parsed config.
- ``out_dir`` is ``/data/<volume root>/<run set>/<run id>/`` on the Modal volume
  ``krmhd-benchmark-vol`` (mounted at ``/data``). Everything the run saves goes there.
  ``ctx.volume_root`` (``/data``) gives read access to older data on the volume, such as
  the Study 2 checkpoints.
- Modal can restart a preempted call. The restart is a new attempt in a new container
  with the same arguments, so ``run`` must resume from its latest checkpoint in
  ``out_dir``. Call ``ctx.commit()`` after writing each checkpoint: only committed files
  survive a lost container. Close every file under ``out_dir`` before ``ctx.commit()``:
  a commit reloads the volume, and the reload fails while files on the volume are open.
  Keep a log file elsewhere, or open and close it for each write.
- Stop cleanly (write a checkpoint, return) before ``ctx.deadline_epoch`` (see Budget).
- Return a small JSON-serialisable summary. It comes back in the ledger's ``result`` as
  plain JSON (enums and other subclasses become their plain values).
- Any exception from ``run``, ``SystemExit`` included, ends the attempt with status
  ``error`` and comes back to the launcher as ``RuntimeError`` with the original type and
  message in its text. ``KeyboardInterrupt`` (Modal's preemption signal) and Modal's input
  cancellation end it with status ``interrupted`` and pass through unchanged.
- Before it calls ``run``, ``run_config`` imports JAX and checks that it has a GPU
  backend. Without one it raises at once (end record written), so a broken CUDA stack
  cannot spend a whole timeout on CPU. The check starts JAX's backends after the run code
  module is imported, so set any XLA environment variables at the top of that module,
  not inside ``run``.

The repo copy in the container is at ``/repo`` (``shared/`` and the study folder only, no
``data/``, ``log/`` or ``figures/``), and ``/repo`` is on ``sys.path``, so
``import shared.<module>`` works.

Budget
------
All attempts of one call share one budget: the timeout T of the launch, counted from the
start record of the run's FIRST attempt. When an attempt starts, it reads the run's
attempt records on the volume to find that first start, so a call that Modal restarted
after a preemption gets only what is left of T:

- ``ctx.deadline_epoch`` = first start + max(T - 15 min, T / 2).
- An attempt that starts after that deadline ends at once: it writes its start record
  and an end record with status ``error`` and an error that starts ``budget exhausted``,
  and it does not call the run code.
- A watchdog thread ends any attempt still running at first start + T: it writes the
  end record (``budget exhausted (watchdog)``), commits the volume, waiting at most
  ``WATCHDOG_RECORD_TIMEOUT_S`` for that, and stops the container. Modal's own timeout
  normally stops the first attempt at about that moment.

So one call uses at most T of GPU time across all its attempts, plus container start-up
(at most ``STARTUP_TIMEOUT_S`` per attempt). The records live outside ``out_dir``
(Accounting), so run code that clears its own folder cannot reset the budget; run code
must never touch ``/data/<root>/_attempts/``.

Accounting
----------
Each attempt writes ``<UTC stamp>_<attempt id>_start.json`` when it starts and
``..._end.json`` when it ends (in a ``finally:``) into
``/data/<root>/_attempts/<run set>/<run id>/``, and commits the volume after each. End
records carry ``status`` (``ok``, ``error`` or ``interrupted``), ``error``, ``traceback``
and ``duration_s``. ``modal_launch.py reconcile`` charges A100-hours from these records:
end minus start for each complete attempt, an open attempt up to the end of the budget,
and at most T for the run unless its complete attempts show more. ``modal_launch.py
fetch`` copies them to the run's ``_attempts/`` folder. Nobody types hours in.

The ``smoke`` function takes the same path with no GPU and counts zero hours. It checks
the image (``jax``, ``krmhd``, one tiny CPU time step), the repo mount and a volume write,
and sleeps so that a test can see the call outlive the launcher.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import math
import os
import re
import sys
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import modal
from modal.exception import InputCancellation

# ---------------------------------------------------------------------------
# Settings. The launcher sets these environment variables before it imports this file.
# The same values are baked into the image with .env(), so the import inside the
# container sees them too. The defaults keep a stray import from failing.
# ---------------------------------------------------------------------------


def _env_str(name: str, default: str) -> str:
    """An environment variable, stripped, or ``default`` if unset or blank."""
    value = os.environ.get(name, "").strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    """An integer environment variable (seconds), or ``default`` if unset or garbled."""
    try:
        return int(float(os.environ.get(name, "")))
    except (TypeError, ValueError):
        return default


APP_NAME = _env_str("S04_APP_NAME", "s04-loop-unset")
TIMEOUT_S = _env_int("S04_TIMEOUT_S", 600)  # the call's budget, and Modal's per-attempt limit, seconds
GANDALF_SHA = _env_str("S04_GANDALF_SHA", "")
JAX_VERSION = _env_str("S04_JAX_VERSION", "")
VOLUME_NAME = _env_str("S04_VOLUME", "krmhd-benchmark-vol")
VOLUME_ROOT = _env_str("S04_VOLUME_ROOT", "study04").strip("/") or "study04"

PYTHON_VERSION = "3.12"  # modal_launch.py picks the jax pin from uv.lock for this version
GPU = "A100"
SMOKE_TIMEOUT_S = 900
SMOKE_MAX_SLEEP_S = 600
STARTUP_TIMEOUT_S = 900  # container start-up per attempt: billed, but before the start record
DEADLINE_MARGIN_S = 900  # run code should be done this long before the budget ends
WATCHDOG_EXIT_CODE = 3
WATCHDOG_RECORD_TIMEOUT_S = 120  # the watchdog stops the container even if its end record hangs
ATTEMPTS_DIRNAME = "_attempts"  # records: /data/<root>/_attempts/<set>/<run id>/ (modal_launch.attempts_path)
VOLUME_MOUNT = "/data"
REMOTE_REPO = "/repo"
STUDY_REL = "studies/04-phase-space-helicity"
GANDALF_GIT = "https://github.com/anjor/gandalf.git"

# Upload filters for add_local_dir (dockerignore syntax, relative to the uploaded folder).
SHARED_IGNORE = ["**/__pycache__", "*.pyc", "**/*.pyc"]
STUDY_IGNORE = [
    "data", "**/data", ".loop", "log", "figures", "**/__pycache__", "*.pyc", "**/*.pyc",
]

_SHA40 = re.compile(r"^[0-9a-f]{40}$")

if modal.is_local() and not _SHA40.match(GANDALF_SHA):
    # A local import without the pin could only come from something other than the
    # launcher (which always sets it). Refuse rather than build an unpinned image.
    raise RuntimeError(
        "modal_app.py: S04_GANDALF_SHA must be the 40-hex gandalf-krmhd commit from uv.lock. "
        "Import this module only through loop/modal_launch.py, which sets it."
    )


def image_requirements(gandalf_sha: str, jax_version: str) -> List[str]:
    """pip requirements of the image: CUDA jax and GANDALF pinned to the repo's lock."""
    jax_req = f"jax[cuda12]=={jax_version}" if jax_version else "jax[cuda12]"
    return [
        jax_req,
        f"gandalf-krmhd @ git+{GANDALF_GIT}@{gandalf_sha}",
        "numpy",
        "h5py",
        "pyyaml",
        "scipy",
    ]


IMAGE_ENV: Dict[str, str] = {
    "S04_APP_NAME": APP_NAME,
    "S04_TIMEOUT_S": str(TIMEOUT_S),
    "S04_GANDALF_SHA": GANDALF_SHA,
    "S04_JAX_VERSION": JAX_VERSION,
    "S04_VOLUME": VOLUME_NAME,
    "S04_VOLUME_ROOT": VOLUME_ROOT,
    "PYTHONUNBUFFERED": "1",
}

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("git")
    .pip_install(*image_requirements(GANDALF_SHA, JAX_VERSION))
    .env(IMAGE_ENV)
)

if modal.is_local():
    # add_local_* must be the last image steps (Modal adds these files when a container
    # starts; any build step after them raises InvalidError). Inside the container the
    # local paths do not exist, so these lines run only on the launching machine.
    try:
        LOCAL_REPO = Path(__file__).resolve().parents[3]
    except IndexError as exc:  # pragma: no cover - only if the file is moved
        raise RuntimeError("modal_app.py must live in studies/04-phase-space-helicity/loop/") from exc
    image = image.add_local_dir(
        LOCAL_REPO / "shared", remote_path=f"{REMOTE_REPO}/shared", ignore=SHARED_IGNORE
    ).add_local_dir(
        LOCAL_REPO / STUDY_REL, remote_path=f"{REMOTE_REPO}/{STUDY_REL}", ignore=STUDY_IGNORE
    )
elif REMOTE_REPO not in sys.path:
    sys.path.insert(0, REMOTE_REPO)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME)  # exists already; never create it here


# ---------------------------------------------------------------------------
# Helpers that run inside the container
# ---------------------------------------------------------------------------


@dataclass
class RunContext:
    """What run code gets besides its config and its output directory.

    ``deadline_epoch`` is a Unix time in seconds: the start of the run's first attempt
    plus the timeout, less a margin (see Budget in the module docstring). A restarted
    call gets only what is left. ``commit()`` commits the volume so that files written so
    far survive a lost container; close every file under ``out_dir`` before calling it.
    """

    run_id: str
    run_set: str
    launch_id: str
    attempt_id: str
    out_dir: Path
    volume_root: Path
    deadline_epoch: float
    commit: Callable[[], None]
    repo_commit: str = ""
    config_relpath: str = ""


class BudgetExhausted(RuntimeError):
    """The run's budget (the timeout, from its first attempt) is used up."""


# Exceptions that end an attempt as 'interrupted' and pass through unchanged: Modal's
# preemption signal (SIGINT) and its input cancellation. Modal's container handles them
# itself (a preempted input is restarted); wrapping them would turn a restart into a failure.
_PASS_THROUGH = (KeyboardInterrupt, InputCancellation, asyncio.CancelledError)


def _utc_iso(t: float) -> str:
    """UTC time as 2026-10-05T10:15:00Z."""
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_stamp(t: float) -> str:
    """UTC time as 20261005T101500Z, for record file names."""
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _finite(value: Any) -> Optional[float]:
    """``value`` as a finite float, or None."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    """Write JSON through a temporary name and a rename, so readers never see half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _write_once(path: Path, data: bytes) -> bool:
    """Write ``data`` unless ``path`` exists (a restarted attempt keeps the first copy)."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return True


def _commit_volume() -> None:
    """Commit the volume, so that what was written so far survives a lost container."""
    volume.commit()


def _hard_exit(code: int) -> None:
    """End the container process at once, without cleanup (the tests replace this)."""
    os._exit(code)


def _jsonable(value: Any) -> Any:
    """``value`` as plain JSON types, else a short string of it.

    A round trip through JSON turns subclasses (an IntEnum member, a str subclass) into
    plain values. Modal pickles the return value, and a subclass defined in the run code
    module could not be unpickled by the launcher, which would read a finished run as
    failed.
    """
    try:
        return json.loads(json.dumps(value))
    except Exception:  # TypeError, ValueError, OverflowError, RecursionError
        return str(value)[:2000]


def attempts_dir(run_set: str, run_id: str) -> Path:
    """Where a run's attempt records live: outside its out_dir, which the run code owns."""
    return Path(VOLUME_MOUNT) / VOLUME_ROOT / ATTEMPTS_DIRNAME / run_set / run_id


class _Attempt:
    """Start and end records of one attempt (first start, or a restart after preemption)."""

    def __init__(self, run_id: str, launch_id: str, records_dir: Path, gpu: str) -> None:
        """A new attempt of ``run_id``, started now; nothing is written before start()."""
        self.attempt_id = uuid.uuid4().hex[:12]
        self.run_id = run_id
        self.t_start = time.time()
        self.dir = records_dir
        self.ids: Dict[str, Any] = {
            "attempt_id": self.attempt_id,
            "run_id": run_id,
            "launch_id": launch_id,
            "call_id": modal.current_function_call_id(),
            "task_id": os.environ.get("MODAL_TASK_ID"),
            "gpu": gpu,
        }
        self.info: Dict[str, Any] = {}  # extra fields for the end record (budget, JAX backend)
        self.finished = threading.Event()  # set when the end record has been written
        self._lock = threading.Lock()
        self._ended = False

    def _record(self, event: str, t: float, extra: Dict[str, Any]) -> None:
        """Write one record (start or end) and commit the volume."""
        record = dict(self.ids, event=event, t_epoch=t, t_utc=_utc_iso(t), **extra)
        _write_json(self.dir / f"{_utc_stamp(t)}_{self.attempt_id}_{event}.json", record)
        _commit_volume()

    def start(self) -> None:
        """Write and commit the start record."""
        self._record("start", self.t_start, {})

    def claim_end(self) -> bool:
        """Claim the right to end the attempt; True for the first caller only (main thread or watchdog)."""
        with self._lock:
            if self._ended:
                return False
            self._ended = True
            self.finished.set()
            return True

    def write_end(self, status: str, error: Optional[str], tb: Optional[str]) -> None:
        """Write and commit the end record (call after claim_end)."""
        t = time.time()
        extra = dict(self.info)
        extra.update(status=status, error=error, traceback=tb, duration_s=round(t - self.t_start, 3))
        self._record("end", t, extra)

    def end(self, status: str, error: Optional[str], tb: Optional[str]) -> bool:
        """Write the end record, once. False if the attempt was ended already (by the other thread)."""
        if not self.claim_end():
            return False
        self.write_end(status, error, tb)
        return True


def _first_start(folder: Path, run_id: str, own_start: float) -> float:
    """The start time of the run's first attempt, from the attempt records in ``folder``.

    Start records give it directly; an end record whose start record was lost gives its
    end time minus its duration. Records of other runs and torn files are ignored. A
    folder that cannot be listed raises, so that an attempt never gets a fresh budget
    because the records were unreadable.
    """
    starts: Dict[str, float] = {}
    from_ends: Dict[str, float] = {}
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        try:
            rec = json.loads((folder / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # half-written by a lost container
        if not isinstance(rec, dict) or rec.get("run_id") != run_id:
            continue
        aid, t = str(rec.get("attempt_id") or name), _finite(rec.get("t_epoch"))
        if t is None:
            continue
        if rec.get("event") == "start":
            starts[aid] = min(t, starts.get(aid, t))
        elif rec.get("event") == "end":
            duration = _finite(rec.get("duration_s"))
            if duration is not None and duration >= 0:
                from_ends[aid] = t - duration
    times = list(starts.values()) + [t for aid, t in from_ends.items() if aid not in starts]
    return min([own_start] + times)


def _work_window(budget_s: float) -> float:
    """Seconds from the first start to ``ctx.deadline_epoch``."""
    return max(budget_s - DEADLINE_MARGIN_S, 0.5 * budget_s)


def _arm_watchdog(attempt: _Attempt, budget_end: float, budget_s: float) -> threading.Thread:
    """End the attempt and the container at ``budget_end`` unless the attempt ends first.

    The run code is told to stop by ``ctx.deadline_epoch``; this is the backstop for run
    code that does not, so that a restarted call cannot use more than its budget in all.
    The end record is written in a thread of its own and waited for at most
    ``WATCHDOG_RECORD_TIMEOUT_S``: a volume commit that hangs must not keep the GPU busy.
    """

    def write_record(message: str) -> None:
        """Write the watchdog's end record; a failure is printed, never raised."""
        try:
            attempt.write_end("error", message, None)
        except BaseException as exc:  # the stop matters more than the record
            print(f"watchdog: could not write the end record: {exc!r}", file=sys.stderr, flush=True)

    def watch() -> None:
        """Wait for the attempt's end or the budget's end, whichever comes first."""
        if attempt.finished.wait(max(0.0, budget_end - time.time())):
            return  # the attempt ended in time
        if not attempt.claim_end():
            return  # the attempt ended at this very moment and writes its own record
        message = (f"budget exhausted (watchdog): the run's budget of {budget_s:g} s from its first "
                   f"attempt ended at {_utc_iso(budget_end)}; the container was stopped")
        print(message, file=sys.stderr, flush=True)
        writer = threading.Thread(target=write_record, args=(message,), name="s04-watchdog-record", daemon=True)
        writer.start()
        writer.join(WATCHDOG_RECORD_TIMEOUT_S)
        _hard_exit(WATCHDOG_EXIT_CODE)

    thread = threading.Thread(target=watch, name="s04-budget-watchdog", daemon=True)
    thread.start()
    return thread


def _run_attempt(run_id: str, run_set: str, launch_id: str, gpu: str, budget_s: float,
                 body: Callable[[_Attempt, Path, float], Any]) -> Dict[str, Any]:
    """Run ``body(attempt, out_dir, deadline_epoch)`` between a start and an end record.

    The budget (module docstring) is applied here: the deadline counts from the run's first
    attempt, an attempt past it ends at once, and a watchdog ends one that runs over.

    Any exception from the body except those in ``_PASS_THROUGH`` comes back to the caller
    as a ``RuntimeError`` with the original type and message in its text. That keeps one
    rule simple for the launcher: a builtin ``TimeoutError`` from ``FunctionCall.get(
    timeout=0)`` always means "still running", never "the run code raised TimeoutError",
    and a ``SystemExit`` from run code cannot reach the launcher's process.
    """
    out_dir = Path(VOLUME_MOUNT) / VOLUME_ROOT / run_set / run_id
    attempt = _Attempt(run_id, launch_id, attempts_dir(run_set, run_id), gpu)
    status, error, tb = "error", None, None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)  # inside: an OSError can be a TimeoutError
        attempt.start()
        first = _first_start(attempt.dir, run_id, attempt.t_start)
        deadline = first + _work_window(budget_s)
        budget_end = first + budget_s
        attempt.info.update(first_start_utc=_utc_iso(first), deadline_utc=_utc_iso(deadline),
                            budget_end_utc=_utc_iso(budget_end))
        if time.time() >= deadline:
            raise BudgetExhausted(
                f"budget exhausted: the run's first attempt started at {_utc_iso(first)}, and its "
                f"deadline {_utc_iso(deadline)} (budget {budget_s:g} s) has passed; the run code "
                "was not called")
        _arm_watchdog(attempt, budget_end, budget_s)
        result = body(attempt, out_dir, deadline)
        status = "ok"
        return {"run_id": run_id, "status": "ok", "result": _jsonable(result)}
    except _PASS_THROUGH as exc:
        status, error = "interrupted", f"{type(exc).__name__}: {exc}"[:2000]
        raise
    except BaseException as exc:  # Exception, SystemExit, GeneratorExit
        error = (str(exc) if isinstance(exc, BudgetExhausted) else f"{type(exc).__name__}: {exc}")[:2000]
        tb = traceback.format_exc()[-8000:]
        print(tb, file=sys.stderr, flush=True)
        raise RuntimeError(f"{run_id} attempt {attempt.attempt_id} failed: {error}") from None
    finally:
        try:
            attempt.end(status, error, tb)
        except Exception as exc:  # the original outcome matters more than this record
            print(f"could not write the end record: {exc!r}", file=sys.stderr, flush=True)


def _load_entrypoint(path: Path) -> Any:
    """Import the run code file and check that it defines ``run``."""
    name = "s04_run_" + re.sub(r"\W", "_", path.stem)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))  # sibling modules of the run code
    spec.loader.exec_module(module)
    if not callable(getattr(module, "run", None)):
        raise AttributeError(f"{path} defines no run(cfg, out_dir, ctx)")
    return module


def _require_gpu() -> Dict[str, Any]:
    """JAX's backend and devices; raises unless JAX has a GPU.

    JAX logs a CUDA plugin that fails to load and falls back to the CPU, where a run would
    spend its whole timeout. This turns that into an error within seconds.
    """
    import jax

    backend = str(jax.default_backend())
    devices = list(jax.devices())
    names = [str(d) for d in devices][:16]
    platforms = {str(getattr(d, "platform", "")).lower() for d in devices}
    if backend != "gpu" and not platforms & {"gpu", "cuda"}:
        raise RuntimeError(f"no GPU: JAX's default backend is {backend!r} and its devices are {names}. "
                           "The CUDA stack did not load, so the run would spend its timeout on CPU.")
    print(f"JAX backend {backend}: {names}", flush=True)
    return {"jax_backend": backend, "jax_devices": names}


def _tiny_step() -> Dict[str, Any]:
    """One IMEX-RK222 step on an 8^3 grid with M=4, on CPU. A plumbing check of the image,
    not a physics result: the numbers are arbitrary and only need to be finite."""
    from krmhd.diagnostics import compute_energy
    from krmhd.physics import initialize_random_spectrum
    from krmhd.spectral import SpectralGrid3D
    from krmhd.timestepping import compute_cfl_timestep, gandalf_step

    grid = SpectralGrid3D.create(8, 8, 8)
    state = initialize_random_spectrum(grid, M=4, amplitude=0.1, seed=1, Lambda=math.sqrt(5.0), nu=0.1)
    dt = float(compute_cfl_timestep(state, 1.0, 0.3))
    e0 = float(compute_energy(state)["total"])
    state = gandalf_step(state, dt, 0.01, 1.0, nu=0.1, scheme="imex_rk222")
    e1 = float(compute_energy(state)["total"])
    return {"grid": "8x8x8", "M": 4, "dt": dt, "energy_before": e0, "energy_after": e1,
            "finite": math.isfinite(e1)}


# ---------------------------------------------------------------------------
# The Modal functions
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu=GPU,
    timeout=TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    retries=0,
    volumes={VOLUME_MOUNT: volume},
    single_use_containers=True,
)
def run_config(run_id: str, run_set: str, config_relpath: str, launch_id: str, repo_commit: str) -> dict:
    """Run one committed config on one A100 (the only GPU function of the loop).

    Writes ``launch.json`` and a copy of the config into the run directory on the first
    attempt, imports the config's ``entrypoint``, checks that JAX has a GPU, then calls
    ``run(cfg, out_dir, ctx)``. The budget rules of the module docstring apply.
    """

    def body(attempt: _Attempt, out_dir: Path, deadline: float) -> Any:
        """Load the config and the run code, check for a GPU, then call run()."""
        import yaml

        repo = Path(REMOTE_REPO)
        cfg_path = repo / config_relpath
        raw = cfg_path.read_bytes()
        cfg = yaml.safe_load(raw.decode("utf-8"))
        if not isinstance(cfg, dict) or not isinstance(cfg.get("entrypoint"), str):
            raise ValueError(f"{config_relpath}: not a mapping with a string 'entrypoint'")
        provenance = {
            "run_id": run_id,
            "run_set": run_set,
            "launch_id": launch_id,
            "config": config_relpath,
            "repo_commit": repo_commit,
            "gandalf_commit": GANDALF_SHA,
            "jax_version": JAX_VERSION,
            "app_name": APP_NAME,
            "timeout_s": TIMEOUT_S,
            "first_attempt_utc": _utc_iso(attempt.t_start),
        }
        _write_once(out_dir / "launch.json",
                    (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        _write_once(out_dir / f"config{cfg_path.suffix or '.yaml'}", raw)
        _commit_volume()
        module = _load_entrypoint(repo / cfg["entrypoint"])
        attempt.info.update(_require_gpu())
        ctx = RunContext(
            run_id=run_id,
            run_set=run_set,
            launch_id=launch_id,
            attempt_id=attempt.attempt_id,
            out_dir=out_dir,
            volume_root=Path(VOLUME_MOUNT),
            deadline_epoch=deadline,
            commit=_commit_volume,
            repo_commit=repo_commit,
            config_relpath=config_relpath,
        )
        return module.run(cfg, out_dir, ctx)

    return _run_attempt(run_id, run_set, launch_id, GPU, TIMEOUT_S, body)


@app.function(
    image=image,
    cpu=1.0,
    timeout=SMOKE_TIMEOUT_S,
    startup_timeout=STARTUP_TIMEOUT_S,
    retries=0,
    volumes={VOLUME_MOUNT: volume},
    single_use_containers=True,
)
def smoke(run_id: str, run_set: str, launch_id: str, repo_commit: str, sleep_s: int = 90) -> dict:
    """Check the launch path with no GPU: image, repo mount, volume write, detachment.

    Fails (raises) only if ``jax`` or ``krmhd`` cannot be imported, since then the image
    is unusable for real runs. A failed CPU step is recorded but does not fail the smoke.
    """

    def body(attempt: _Attempt, out_dir: Path, deadline: float) -> Any:
        """Check the image, the repo mount and a volume write, then sleep."""
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        if REMOTE_REPO not in sys.path:
            sys.path.insert(0, REMOTE_REPO)
        study = Path(REMOTE_REPO) / STUDY_REL
        info: Dict[str, Any] = {
            "run_id": run_id,
            "launch_id": launch_id,
            "repo_commit": repo_commit,
            "attempt_id": attempt.attempt_id,
            "call_id": attempt.ids["call_id"],
            "python": sys.version.split()[0],
            "gandalf_sha_env": GANDALF_SHA,
            "jax_version_env": JAX_VERSION,
            "repo_mount_ok": (study / "PLAN.md").is_file(),
            "repo_listing": sorted(p.name for p in study.iterdir()) if study.is_dir() else [],
            "deadline_utc": _utc_iso(deadline),
        }
        try:
            import shared  # noqa: F401  (the repo's shared/ package)
            info["shared_import_ok"] = True
        except Exception as exc:
            info["shared_import_ok"] = False
            info["shared_import_error"] = repr(exc)[:300]
        try:
            import jax
            import krmhd

            info["jax_version"] = jax.__version__
            info["krmhd_version"] = str(getattr(krmhd, "__version__", "unknown"))
            info["devices"] = [str(d) for d in jax.devices()]
        except Exception as exc:
            info["import_error"] = repr(exc)[:500]
        if "import_error" not in info:
            try:
                info["step"] = _tiny_step()
            except Exception as exc:
                info["step_error"] = repr(exc)[:500]
        time.sleep(max(0, min(int(sleep_s), SMOKE_MAX_SLEEP_S)))
        info["finished_utc"] = _utc_iso(time.time())
        _write_json(out_dir / "smoke.json", info)
        _commit_volume()
        if "import_error" in info:
            raise RuntimeError(f"smoke: jax/krmhd import failed in the image: {info['import_error']}")
        return info

    return _run_attempt(run_id, run_set, launch_id, "none", SMOKE_TIMEOUT_S, body)
