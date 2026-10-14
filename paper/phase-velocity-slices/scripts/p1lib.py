"""Shared library for the GANDALF verification runs of the phase-velocity-slice paper (section 9, "Checks with
GANDALF"): run bookkeeping, the periodic box, random and localised initial fields, the slice basis of the Hermite
ladder, GANDALF state construction and time stepping, the frozen-field RK4 problem, diagnostics, time-step limits,
gates and run records.  GANDALF (krmhd 0.6.0) is used read-only.

Import this module before anything else that imports JAX or krmhd: it switches JAX to float64 and to the CPU.

Units and conventions (paper section 2; checked against GANDALF's right-hand side in derivations/dynamics/d01):
  v_A = 1, B0 = 1.  Phi = (z+ + z-)/2 (stream function), Psi = (z+ - z-)/2 (flux function), so z+- = Phi +- Psi;
  {f, h} = d_x f d_y h - d_y f d_x h;  u_E = zhat x grad Phi;  grad_par = d_z + {Psi, .};
  z+ propagates toward -z (linearly d_t z+- = +- d_z z+-).
  One compressive branch G (a combination of density and field-strength perturbations) in GANDALF's Hermite
  variables g_m (CMM's tilde G_m):  d_t g + {Phi, g} + s grad_par(A g) = -C g,  s = sqrt(beta_i) = v_th/v_A,
  u = v_par/v_th.  A = krmhd.hermite.compute_streaming_matrix(M, Lambda, closure) is the ladder (the matrix of u),
  with A[1,0] = (1 - 1/Lambda)/sqrt2; free-energy weights P = diag(p_m), p_0 = 1 - 1/Lambda, p_m = 1 (m >= 1)
  (copy closure: p_M = sqrt(M)/(sqrt(M) + sqrt(M+1))), and P A is symmetric.  C = nu diag((m/M)^n) for m >= 2 are
  GANDALF's normalised hypercollisions.
  Slices: A r_j = u_j r_j with unit-weight right eigenvectors (R^T P R = I) and left eigenvectors w_j = P r_j.  The
  slice amplitudes a_j = w_j . g obey  d_t a_j + {Phi + s u_j Psi, a_j} + s u_j d_z a_j = 0, and
  sum_m p_m g_m^2 = sum_j a_j^2 pointwise.  The invariants are Q_phi = sum_j phi(u_j) <a_j^2>; CMM's W, Gamma, H are
  Q_1/2, Q_u/2, Q_{1/u}/2 on the retained moments (zero closure; Q_{1/u} needs odd M).

Spectral conventions: GANDALF stores real fields f(x) as unnormalised rfftn coefficients f_k on arrays
[Nz, Ny, Nx//2+1(, M+1)], f(x) = (1/N^3) sum_k f_k exp(i k.x).  Volume averages are therefore
<f h> = (1/V) int f h d^3x = sum_k w_k Re(conj(f_k) h_k) / N^6, with w_k = 1 on the k_x = 0 and k_x = Nyquist planes
and 2 elsewhere.  Box side L (default 1), k_1 = 2 pi / L.  GANDALF's 2/3 mask is a cube in k: |n_x|, |n_y|, |n_z|
<= n_max with n_max = floor(N/3) (N/3 itself is retained when 3 divides N).

Run records: every run gets the ID <RUN_PREFIX>_<label>_<YYYYMMDD_HHMMSS> (local start time) and writes
<run ID>.npz and <run ID>.json to paper/phase-velocity-slices/data/<run ID>/ (git-ignored); see class Run.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import ast  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import operator  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import asdict, dataclass  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Iterable, Mapping, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import scipy.linalg  # noqa: E402
import yaml  # noqa: E402

import krmhd  # noqa: E402
from krmhd.hermite import compute_streaming_matrix  # noqa: E402
from krmhd.physics import KRMHDState, poisson_bracket_3d  # noqa: E402
from krmhd.spectral import SpectralGrid3D  # noqa: E402
from krmhd.timestepping import KRMHDFields, _krmhd_rhs_jit, compute_cfl_timestep, gandalf_step  # noqa: E402

if jnp.zeros(1).dtype != jnp.float64:  # pragma: no cover - guards against importing JAX before this module
    raise RuntimeError("p1lib needs float64 JAX: import p1lib before any other module that imports JAX or krmhd")

ArrayLike = Any  # numpy arrays or JAX arrays

# ------------------------------------------------------------------------------------------------------------------
# paths and constants
# ------------------------------------------------------------------------------------------------------------------
SCRIPTS = Path(__file__).resolve().parent          # paper/phase-velocity-slices/scripts
PAPER = SCRIPTS.parent                             # paper/phase-velocity-slices
REPO = PAPER.parent.parent                         # repository root
CONFIGS = PAPER / "configs"
DATA = PAPER / "data"                              # git-ignored
FIGURES = PAPER / "figures"

RUN_PREFIX = "P1"                                  # run-ID prefix fixed by the repository's run-log convention
SQRT2 = math.sqrt(2.0)
SQRT5 = math.sqrt(5.0)
LAWSON_RK4_LIMIT = 2.0 * SQRT2                     # RK4 is stable on the imaginary axis for |y| <= 2 sqrt2
NOT_APPLICABLE_SHORT = "not applicable: short verification run"
PRECISION = "float64 (jax_enable_x64), complex128 spectral arrays"

__all__ = [
    # bookkeeping
    "PAPER", "REPO", "CONFIGS", "DATA", "FIGURES", "RUN_PREFIX", "SQRT2", "SQRT5", "LAWSON_RK4_LIMIT",
    "NOT_APPLICABLE_SHORT", "parse_number", "load_config", "config_relpath", "make_run_id", "make_run_dir",
    "hardware_string", "environment_info", "Run", "save_run", "load_run", "latest_run_dir", "run_log_row",
    "flatten", "stack_records", "steps_for_times", "timer",
    # box and fields
    "Box", "phi_psi", "elsasser", "normalise_rms_grad",
    # velocity space
    "ladder", "pweights", "Slices", "make_slices", "coll_rates", "slice_coupling_matrix", "leakage_first_order",
    # GANDALF interface
    "make_state", "fields_of", "StepParams", "gandalf_advance", "make_rhs", "FrozenRK4", "make_full_rk4",
    "kz_only_evolve",
    # diagnostics
    "W_m", "free_energy", "slice_energies", "slice_correlation", "PHI_NAMES", "phi_values", "Q_phi",
    "Q_phi_scale", "hermite_flux_field_aligned", "hermite_flux_kz", "quartic_moments", "free_energy_density_k",
    "alfvenic_energy_density_k", "W_kperp", "W_kz", "alfvenic_energy", "nonlinearity_parameter",
    "outer_shell_fraction", "resolution_check", "dissipation_rates", "slice_leakage", "diagnostics",
    # time-step limits
    "advection_operator_norm", "slice_advection_rate", "recommended_dt", "rk_stability",
    # gates
    "relative_drift", "gate_energy_conservation", "make_gates", "gates_outcome", "require_gates", "budget_residual",
    # figures
    "JPP_SINGLE_IN", "JPP_DOUBLE_IN", "OKABE_ITO", "jpp_style", "save_figure",
]


# ------------------------------------------------------------------------------------------------------------------
# configuration, run IDs, records
# ------------------------------------------------------------------------------------------------------------------
_BINOPS: dict[type, Callable[[float, float], float]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNOPS: dict[type, Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_NAMES = {"pi": math.pi, "inf": math.inf, "e": math.e}
_FUNCS: dict[str, Callable[[float], float]] = {"sqrt": math.sqrt}


def parse_number(x: Any) -> float:
    """Convert a config entry to a float.  Numbers pass through; strings may be simple arithmetic in numbers,
    pi, inf and sqrt(.), e.g. "sqrt(5)", "-sqrt5", "inf", "2*pi" (so that Lambda = +-sqrt5 can be stated exactly
    in YAML).  Evaluated with a whitelist over the syntax tree (no eval)."""
    if isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool):
        return float(x)
    if not isinstance(x, str):
        raise TypeError(f"cannot parse {x!r} as a number")
    text = re.sub(r"sqrt\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", x.strip())

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id in _NAMES:
            return _NAMES[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            return _BINOPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNOPS:
            return _UNOPS[type(node.op)](ev(node.operand))
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
                and len(node.args) == 1 and not node.keywords):
            return _FUNCS[node.func.id](ev(node.args[0]))
        raise ValueError(f"unsupported expression in {x!r}")

    return ev(ast.parse(text, mode="eval"))


def _resolve_config(path: str | Path) -> Path:
    p = Path(path)
    for cand in (p, REPO / p, CONFIGS / p):
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(f"config {path} not found (tried as given, relative to the repository and to {CONFIGS})")


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML config with yaml.safe_load.  `path` may be absolute, relative to the working directory, to the
    repository root or to paper/phase-velocity-slices/configs.  Numbers written as strings (e.g. Lambda: "sqrt(5)")
    are left as strings; convert them with parse_number where they are used."""
    with open(_resolve_config(path)) as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} does not contain a mapping")
    return cfg


def config_relpath(path: str | Path) -> str:
    """The config path relative to the repository root, as written in the run log."""
    p = _resolve_config(path)
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def make_run_id(label: str, when: datetime | None = None) -> str:
    """Run ID <RUN_PREFIX>_<label>_<YYYYMMDD_HHMMSS> from the local start time.  Labels use letters, digits and
    '-', '.', '+' (hyphens are preferred to underscores so the ID splits unambiguously)."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+\-]*", label):
        raise ValueError(f"invalid run label {label!r}: use letters, digits, '-', '.', '+'")
    return f"{RUN_PREFIX}_{label}_{(when or datetime.now()).strftime('%Y%m%d_%H%M%S')}"


def make_run_dir(run_id: str, base: str | Path = DATA) -> Path:
    """Create and return the data folder <base>/<run ID>/ (base defaults to the git-ignored paper data folder).
    Raises FileExistsError if it exists, so that two runs never share a folder."""
    d = Path(base) / run_id
    d.mkdir(parents=True, exist_ok=False)
    return d


def hardware_string() -> str:
    """Operating system, architecture and CPU, e.g. 'Darwin arm64, Apple M1 Pro' (local query, no network)."""
    cpu = ""
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except Exception:  # pragma: no cover
            cpu = ""
    cpu = cpu or platform.processor()
    base = f"{platform.system()} {platform.machine()}"
    return f"{base}, {cpu}" if cpu else base


def environment_info() -> dict[str, str]:
    """Versions, precision and hardware recorded in every run's JSON file."""
    import scipy
    return dict(krmhd_version=getattr(krmhd, "__version__", "unknown"), jax_version=jax.__version__,
                numpy_version=np.__version__, scipy_version=scipy.__version__,
                python_version=sys.version.split()[0], precision=PRECISION, jax_backend=jax.default_backend(),
                hardware=hardware_string())


def _jsonable(x: Any) -> Any:
    """Recursively convert numpy/JAX scalars and arrays (and paths) into JSON-serialisable Python objects."""
    if isinstance(x, Mapping):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else str(v)
    if isinstance(x, (np.complexfloating, complex)):
        return [float(np.real(x)), float(np.imag(x))]
    if isinstance(x, (np.ndarray, jax.Array)):
        return _jsonable(np.asarray(x).tolist())
    return x


def flatten(d: Mapping[str, Any], sep: str = "__", prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten nested dicts of arrays/scalars into one level (keys joined by `sep`) for np.savez."""
    out: dict[str, np.ndarray] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, sep, key))
        else:
            out[key] = np.asarray(v)
    return out


def stack_records(records: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Stack a list of diagnostic dicts (e.g. from `diagnostics` at each output time) into arrays with a leading
    time axis.  All records must have the same keys and shapes."""
    if not records:
        return {}
    return {k: np.array([np.asarray(r[k]) for r in records]) for k in records[0]}


def save_run(run_dir: str | Path, run_id: str, arrays: Mapping[str, Any], meta: Mapping[str, Any]) -> tuple[Path, Path]:
    """Write <run_dir>/<run_id>.npz (compressed; nested dicts are flattened with '__') and <run_dir>/<run_id>.json."""
    run_dir = Path(run_dir)
    npz = run_dir / f"{run_id}.npz"
    js = run_dir / f"{run_id}.json"
    np.savez_compressed(npz, **flatten(arrays))
    with open(js, "w") as fh:
        json.dump(_jsonable(meta), fh, indent=2)
        fh.write("\n")
    return npz, js


def load_run(run_dir: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read a run folder written by Run.save / save_run: (arrays from the .npz, metadata from the .json)."""
    run_dir = Path(run_dir)
    npzs = sorted(run_dir.glob("*.npz"))
    jss = sorted(run_dir.glob("*.json"))
    if len(npzs) != 1 or len(jss) != 1:
        raise FileNotFoundError(f"{run_dir}: expected one .npz and one .json, found {len(npzs)} and {len(jss)}")
    with np.load(npzs[0], allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    with open(jss[0]) as fh:
        meta = json.load(fh)
    return arrays, meta


def latest_run_dir(label: str, base: str | Path = DATA) -> Path:
    """The most recent data folder <RUN_PREFIX>_<label>_<timestamp> (by the timestamp in its name)."""
    pat = re.compile(rf"^{re.escape(RUN_PREFIX)}_{re.escape(label)}_(\d{{8}}_\d{{6}})$")
    found = [(m.group(1), p) for p in Path(base).iterdir() if p.is_dir() and (m := pat.match(p.name))]
    if not found:
        raise FileNotFoundError(f"no run folder {RUN_PREFIX}_{label}_<timestamp> in {base}")
    return max(found)[1]


def run_log_row(meta: Mapping[str, Any]) -> str:
    """The docs/run_log.md row for a run: | ID | config | hardware | wall time (s) | outcome | notes |."""
    notes = str(meta.get("notes", "")).replace("|", "/").replace("\n", " ")
    return (f"| {meta['run_id']} | {meta.get('config_path', '')} | {meta.get('hardware', '')} | "
            f"{float(meta.get('wall_time_s', float('nan'))):.1f} | {meta.get('outcome', '')} | {notes} |")


class Run:
    """One verification run: its ID, data folder and timer, and the writer of its .npz and .json files.

    Usage:
        run = Run("frozen-N48", cfg, "configs/frozen_line.yaml")
        ... integrate, collect arrays, build gates with make_gates ...
        meta = run.save(arrays, gates, notes="...")
        print(run.summary(meta))            # the one-line summary the spec asks for
        print(run_log_row(meta))           # the row for docs/run_log.md
    The JSON holds the run ID, label, config path and content, start/end time, wall time, versions (krmhd, JAX,
    NumPy, SciPy, Python), precision, JAX backend, hardware, the gates and the outcome ("pass" if every applicable
    gate passed)."""

    def __init__(self, label: str, config: Mapping[str, Any], config_path: str | Path | None = None,
                 base: str | Path = DATA) -> None:
        for _ in range(3):
            self.run_id = make_run_id(label)
            try:
                self.dir = make_run_dir(self.run_id, base)
                break
            except FileExistsError:
                time.sleep(1.05)
        else:  # pragma: no cover
            raise FileExistsError(f"could not create a fresh folder for label {label}")
        self.label = label
        self.config = dict(config)
        self.config_path = config_relpath(config_path) if config_path is not None else ""
        self.t0 = time.time()
        self.started_local = datetime.now().isoformat(timespec="seconds")

    def wall(self) -> float:
        """Wall time in seconds since the run object was created."""
        return time.time() - self.t0

    def save(self, arrays: Mapping[str, Any], gates: Mapping[str, Any], notes: str = "",
             extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Write <run ID>.npz (the arrays) and <run ID>.json (metadata, gates, outcome); return the metadata."""
        meta: dict[str, Any] = dict(run_id=self.run_id, label=self.label, config_path=self.config_path,
                                    config=self.config, started_local=self.started_local,
                                    finished_local=datetime.now().isoformat(timespec="seconds"),
                                    wall_time_s=round(self.wall(), 1))
        meta.update(environment_info())
        meta["gates"] = dict(gates)
        meta["outcome"] = gates_outcome(gates)
        meta["notes"] = notes
        meta["files"] = dict(npz=f"{self.run_id}.npz", json=f"{self.run_id}.json")
        if extra:
            meta["extra"] = dict(extra)
        save_run(self.dir, self.run_id, arrays, meta)
        return _jsonable(meta)

    def summary(self, meta: Mapping[str, Any]) -> str:
        """One-line summary: ID, outcome, wall time, notes."""
        return f"{self.run_id}: {meta['outcome']}; wall {meta['wall_time_s']:.1f} s; {meta.get('notes', '')}"


def steps_for_times(times: Iterable[float], dt: float, tol: float = 1e-9) -> list[int]:
    """Step indices n with n dt = t for each output time t; raises ValueError if some t is not a multiple of dt
    (e.g. t = 0.125 at dt = 0.002), naming the nearest attainable time."""
    out = []
    for t in times:
        n = int(round(t / dt))
        if abs(n * dt - t) > tol * max(1.0, abs(t)):
            raise ValueError(f"output time {t} is not a multiple of dt = {dt}; nearest attainable times are "
                             f"{(n - 1) * dt:.6g}, {n * dt:.6g}, {(n + 1) * dt:.6g}")
        out.append(n)
    return out


def timer() -> Callable[[], float]:
    """A stopwatch: returns a function giving the seconds elapsed since the call."""
    t0 = time.time()
    return lambda: time.time() - t0


# ------------------------------------------------------------------------------------------------------------------
# the periodic box
# ------------------------------------------------------------------------------------------------------------------
class Box:
    """Triply periodic cube of side L with GANDALF's SpectralGrid3D (N^3 points, 2/3 dealiasing), plus numpy helpers
    for GANDALF's unnormalised rfftn coefficients.

    Attributes: N, L, grid (SpectralGrid3D), kx, ky, kz (1D wavenumbers), KX, KY, KZ (broadcastable to
    [Nz, Ny, Nx//2+1]), KP2 = k_perp^2, mask (2/3 band), w (rfft multiplicity weights), k1 = 2 pi/L, Ntot = N^3,
    nmax (largest retained |n_x| = |n_y| = |n_z|), kperp_max (largest retained |k_perp| = sqrt2 nmax k1, the corner
    of the square band), kperp_max2_gandalf (GANDALF's normalisation of hyper-resistivity, k_x[(N-1)//3]^2 +
    k_y[(N-1)//3]^2), nperp (integer k_perp shell index rint(|k_perp|/k1)), nz (integer |k_z| shell index)."""

    def __init__(self, N: int, L: float = 1.0) -> None:
        if N % 2:
            raise ValueError("GANDALF needs an even N")
        self.N = int(N)
        self.L = float(L)
        self.grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=self.L, Ly=self.L, Lz=self.L)
        self.kx = np.asarray(self.grid.kx, dtype=float)
        self.ky = np.asarray(self.grid.ky, dtype=float)
        self.kz = np.asarray(self.grid.kz, dtype=float)
        self.mask = np.asarray(self.grid.dealias_mask, dtype=bool)
        self.KX = self.kx[None, None, :]
        self.KY = self.ky[None, :, None]
        self.KZ = self.kz[:, None, None]
        self.KP2 = self.KX ** 2 + self.KY ** 2
        w = np.full(N // 2 + 1, 2.0)
        w[0] = 1.0
        w[-1] = 1.0
        self.w = w[None, None, :]
        self.k1 = 2.0 * np.pi / self.L
        self.Ntot = self.N ** 3
        self.jk = (jnp.asarray(self.kx), jnp.asarray(self.ky), jnp.asarray(self.kz), jnp.asarray(self.mask))
        nx = np.rint(np.abs(self.kx) / self.k1).astype(int)
        self.nmax = int(nx[self.mask.any(axis=(0, 1))].max())
        self.kperp_max = float(np.sqrt(np.max(self.KP2 * self.mask)))
        i = (N - 1) // 3
        self.kperp_max2_gandalf = float(self.kx[i] ** 2 + self.ky[i] ** 2)
        self.nperp = np.rint(np.sqrt(self.KP2) / self.k1).astype(int)          # [1, Ny, Nkx]
        self.nz = np.rint(np.abs(self.kz) / self.k1).astype(int)               # [Nz]
        self.nperp_band = int(np.max(np.where(self.mask, self.nperp, 0)))     # largest shell with retained modes
        self.nz_band = int(np.max(np.where(self.mask, self.nz[:, None, None], 0)))

    # ----- broadcasting helpers -----
    def _wb(self, ndim: int) -> np.ndarray:
        return self.w.reshape(self.w.shape + (1,) * (ndim - 3))

    def _mb(self, ndim: int) -> np.ndarray:
        return self.mask.reshape(self.mask.shape + (1,) * (ndim - 3))

    # ----- volume averages -----
    def avg(self, f: ArrayLike, h: ArrayLike) -> float:
        """<f h> = (1/V) int f h d^3x for two real fields given by rfft coefficients [Nz, Ny, Nkx]."""
        f = np.asarray(f)
        h = np.asarray(h)
        return float(np.sum(self.w * np.real(np.conj(f) * h))) / self.Ntot ** 2

    def avg2(self, f: ArrayLike) -> float:
        """<f^2> (volume average of the square of a real field)."""
        f = np.asarray(f)
        return float(np.sum(self.w * np.abs(f) ** 2)) / self.Ntot ** 2

    def avg2_last(self, f: ArrayLike) -> np.ndarray:
        """<f_i^2> for each index i of the trailing axis of f [Nz, Ny, Nkx, n]."""
        f = np.asarray(f)
        return np.einsum("zyxi->i", self._wb(f.ndim) * np.abs(f) ** 2) / self.Ntot ** 2

    def rms_grad(self, f: ArrayLike) -> float:
        """sqrt(<|grad_perp f|^2>): for f = Phi this is the rms E x B speed u_rms; for f = Psi the rms b_perp."""
        f = np.asarray(f)
        return math.sqrt(float(np.sum(self.w * self.KP2 * np.abs(f) ** 2))) / self.Ntot

    # ----- real space -----
    def coords(self, pad: int = 1) -> np.ndarray:
        """Grid coordinates 0, L/(pad N), ... in each direction (real-space arrays are indexed [z, y, x])."""
        return np.arange(self.N * pad) * self.L / (self.N * pad)

    def to_real(self, f: ArrayLike, pad: int = 1) -> np.ndarray:
        """Real-space values [z, y, x(, ...)] on a grid refined by `pad` (zero padding, exact for band-limited f;
        the Nyquist rows, empty under the 2/3 mask, are dropped)."""
        f = np.asarray(f)
        N = self.N
        if pad == 1:
            return np.fft.irfftn(f, s=(N, N, N), axes=(0, 1, 2))
        Np = pad * N
        h = N // 2
        F = np.zeros((Np, Np, Np // 2 + 1) + f.shape[3:], dtype=complex)
        i_dst = np.r_[0:h, Np - h + 1:Np]
        i_src = np.r_[0:h, N - h + 1:N]
        F[np.ix_(i_dst, i_dst, np.arange(h))] = f[np.ix_(i_src, i_src, np.arange(h))]
        return np.fft.irfftn(F, s=(Np, Np, Np), axes=(0, 1, 2)) * pad ** 3

    def plane_z(self, f: ArrayLike, z0: float = 0.0, pad: int = 1) -> np.ndarray:
        """Real-space values f(x, y, z0) on the plane z = z0, as an array [y, x(, ...)] on a grid refined by `pad`
        in x and y (for imshow use origin="lower", extent=[0, L, 0, L])."""
        f = np.asarray(f)
        N = self.N
        ph = np.exp(1j * self.kz * z0).reshape((N,) + (1,) * (f.ndim - 1))
        f2 = np.sum(f * ph, axis=0)                                   # [Ny, Nkx, ...]
        if pad == 1:
            return np.fft.irfft2(f2, s=(N, N), axes=(0, 1)) / N
        Np = pad * N
        h = N // 2
        F = np.zeros((Np, Np // 2 + 1) + f2.shape[2:], dtype=complex)
        F[np.ix_(np.r_[0:h, Np - h + 1:Np], np.arange(h))] = f2[np.ix_(np.r_[0:h, N - h + 1:N], np.arange(h))]
        return np.fft.irfft2(F, s=(Np, Np), axes=(0, 1)) * pad ** 2 / N

    def grad_perp_real(self, f: ArrayLike, pad: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(d_x f, d_y f) in real space on a grid refined by `pad`."""
        f = np.asarray(f)
        return self.to_real(1j * self.KX * f, pad), self.to_real(1j * self.KY * f, pad)

    # ----- operators (GANDALF's) -----
    def bracket(self, f: ArrayLike, h: ArrayLike) -> np.ndarray:
        """GANDALF's dealiased Poisson bracket {f, h} = d_x f d_y h - d_y f d_x h (rfft coefficients in and out)."""
        N = self.N
        return np.asarray(poisson_bracket_3d(jnp.asarray(f), jnp.asarray(h), self.jk[0], self.jk[1], N, N, N,
                                             self.jk[3]))

    def dz(self, f: ArrayLike) -> np.ndarray:
        """d_z f = i k_z f (GANDALF's derivative_z), for arrays with or without a trailing axis."""
        f = np.asarray(f)
        return 1j * self.KZ.reshape(self.KZ.shape + (1,) * (f.ndim - 3)) * f

    def gradpar(self, psi: ArrayLike, f: ArrayLike) -> np.ndarray:
        """Field-aligned derivative grad_par f = d_z f + {Psi, f} (bhat = zhat + zhat x grad Psi to first order)."""
        return self.dz(f) + self.bracket(psi, f)

    # ----- fields -----
    def random_field(self, rng: np.random.Generator, nperp: Sequence[float], nzmax: float,
                     extra: Sequence[int] = ()) -> np.ndarray:
        """Real Gaussian random field (white noise in real space, unit variance per point, then filtered) with
        k_perp/k1 in [nperp[0], nperp[1]], |k_z|/k1 <= nzmax, inside this box's 2/3 mask, k = 0 removed.  `extra`
        appends trailing axes (e.g. (M+1,) for independent Hermite moments).  The draw sequence is identical to
        derivations/dynamics/dcore.py Box.random_field, so seeded draws reproduce the fields used there.  Fields drawn
        on a small box and embedded (method embed) are the same physical fields at every resolution."""
        N = self.N
        x = rng.standard_normal((N, N, N) + tuple(extra))
        f = np.fft.rfftn(x, axes=(0, 1, 2))
        KP = np.sqrt(self.KP2)
        sel = ((KP >= self.k1 * nperp[0] - 1e-9) & (KP <= self.k1 * nperp[1] + 1e-9)
               & (np.abs(self.KZ) <= self.k1 * nzmax + 1e-9) & self.mask)
        sel = sel.reshape(sel.shape + (1,) * len(extra))
        f = f * sel
        f[0, 0, 0, ...] = 0.0
        return f

    def embed(self, f0: ArrayLike, N0: int) -> np.ndarray:
        """Embed rfft coefficients of a field on an N0^3 grid (same L) into this grid: the same physical field
        (coefficients scaled by (N/N0)^3; the N0 Nyquist rows are dropped; this box's mask applied)."""
        f0 = np.asarray(f0)
        N = self.N
        if N0 > N:
            raise ValueError("embed goes from a coarser to a finer grid")
        h0 = N0 // 2
        F = np.zeros((N, N, N // 2 + 1) + f0.shape[3:], dtype=complex)
        i_dst = np.r_[0:h0, N - h0 + 1:N]
        i_src = np.r_[0:h0, N0 - h0 + 1:N0]
        F[np.ix_(i_dst, i_dst, np.arange(h0))] = f0[np.ix_(i_src, i_src, np.arange(h0))] * (N / N0) ** 3
        return F * self._mb(F.ndim)

    def gaussian_blob(self, sigma: Sequence[float], center: Sequence[float] | None = None,
                      remove_mean: bool = True) -> np.ndarray:
        """Band-limited periodic Gaussian f(x) = exp(-sum_i (x_i - c_i)^2 / (2 sigma_i^2)) (sum over periodic images),
        from its exact Fourier coefficients (N^3/V) (2 pi)^{3/2} sigma_x sigma_y sigma_z exp(-k_i^2 sigma_i^2/2 - i k.c),
        masked to the 2/3 band.  sigma = (sigma_x, sigma_y, sigma_z) are standard deviations; the default centre is
        (L/2, L/2, 0), so the plane z = 0 cuts through the peak.  remove_mean drops the k = 0 mode (which GANDALF's
        right-hand side never evolves), leaving a peak of about 1 - mean on a slightly negative background."""
        sx, sy, sz = (float(s) for s in sigma)
        cx, cy, cz = (0.5 * self.L, 0.5 * self.L, 0.0) if center is None else (float(c) for c in center)
        amp = self.Ntot / self.L ** 3 * (2.0 * np.pi) ** 1.5 * sx * sy * sz
        f = (amp * np.exp(-0.5 * (self.KX ** 2 * sx ** 2 + self.KY ** 2 * sy ** 2 + self.KZ ** 2 * sz ** 2))
             * np.exp(-1j * (self.KX * cx + self.KY * cy + self.KZ * cz)))
        f = f * self.mask
        if remove_mean:
            f[0, 0, 0] = 0.0
        return f

    def band_leakage(self, f: ArrayLike) -> float:
        """max |f_k| outside the 2/3 band divided by max |f_k| (0 for fields made by this class)."""
        f = np.asarray(f)
        top = float(np.abs(f).max())
        return float(np.abs(f * ~self._mb(f.ndim)).max()) / top if top > 0 else 0.0


def phi_psi(zp: ArrayLike, zm: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Stream and flux functions from the Elsasser potentials: Phi = (z+ + z-)/2, Psi = (z+ - z-)/2 (GANDALF)."""
    zp = np.asarray(zp)
    zm = np.asarray(zm)
    return 0.5 * (zp + zm), 0.5 * (zp - zm)


def elsasser(phi: ArrayLike, psi: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Elsasser potentials z+- = Phi +- Psi."""
    phi = np.asarray(phi)
    psi = np.asarray(psi)
    return phi + psi, phi - psi


def normalise_rms_grad(box: Box, f: ArrayLike, target: float) -> np.ndarray:
    """f rescaled so that sqrt(<|grad_perp f|^2>) = target."""
    f = np.asarray(f)
    r = box.rms_grad(f)
    if r == 0.0:
        raise ValueError("field has no perpendicular gradient")
    return f * (target / r)


# ------------------------------------------------------------------------------------------------------------------
# velocity space: ladder, weights, slices, hypercollisions
# ------------------------------------------------------------------------------------------------------------------
def ladder(M: int, Lam: float, closure: str = "zero") -> np.ndarray:
    """GANDALF's streaming matrix A (krmhd.hermite.compute_streaming_matrix), the matrix of u = v_par/v_th on the
    Hermite coefficients: A[m, m+1] = A[m+1, m] = sqrt((m+1)/2) except A[1, 0] = (1 - 1/Lambda)/sqrt2; the copy
    ("symmetric") closure adds sqrt((M+1)/2) to A[M, M-1].  Lambda = inf allowed."""
    return np.array(compute_streaming_matrix(int(M), float(Lam), closure), dtype=float)


def pweights(M: int, Lam: float, closure: str = "zero") -> np.ndarray:
    """Free-energy weights p_m making P A symmetric: p_0 = 1 - 1/Lambda, p_m = 1, and under the copy closure
    p_M = sqrt(M)/(sqrt(M) + sqrt(M+1)).  The free energy is W = Q_1 = sum_m p_m <g_m^2>."""
    p = np.ones(M + 1)
    p[0] = 1.0 - (0.0 if np.isinf(Lam) else 1.0 / Lam)
    if closure == "symmetric":
        p[M] = math.sqrt(M) / (math.sqrt(M) + math.sqrt(M + 1.0))
    return p


@dataclass(frozen=True, eq=False)
class Slices:
    """Phase-velocity slices of the truncated hierarchy: A r_j = u_j r_j.

    Fields: M, Lam, closure, A (ladder), p (weights), u (nodes u_j, ascending), R (columns r_j, unit weight:
    R^T P R = I), Wl (columns w_j = P r_j, left eigenvectors, w_j . r_l = delta_jl).
    Slice amplitudes a_j = w_j . g, so g = sum_j a_j r_j and sum_m p_m g_m^2 = sum_j a_j^2 pointwise and per Fourier
    mode.  Sign convention: R[0, j] > 0, so a density-only g = f e_0 gives a_j = p_0 R[0, j] f with the sign of f in
    every slice."""

    M: int
    Lam: float
    closure: str
    A: np.ndarray
    p: np.ndarray
    u: np.ndarray
    R: np.ndarray
    Wl: np.ndarray

    @property
    def umax(self) -> float:
        """Largest |u_j| (the fastest slice)."""
        return float(np.abs(self.u).max())

    def amplitudes(self, g: ArrayLike) -> np.ndarray:
        """a_j = w_j . g for g [..., M+1] (any leading shape; Fourier or real space)."""
        return np.asarray(g) @ self.Wl

    def synthesize(self, a: ArrayLike) -> np.ndarray:
        """g = sum_j a_j r_j for a [..., M+1]."""
        return np.asarray(a) @ self.R.T

    def load_single(self, f: ArrayLike, j: int) -> np.ndarray:
        """g = f r_j: the field f placed in slice j alone (a_j = f, a_l = 0 for l != j)."""
        return np.asarray(f)[..., None] * self.R[:, j]

    def nearest(self, u0: float) -> int:
        """Index of the node u_j nearest to u0."""
        return int(np.argmin(np.abs(self.u - u0)))


def make_slices(M: int, Lam: float, closure: str = "zero") -> Slices:
    """Build the slice basis of GANDALF's ladder through the symmetric similarity transform P^{1/2} A P^{-1/2}
    (valid for Lambda outside [0, 1], where P is positive definite)."""
    A = ladder(M, Lam, closure)
    p = pweights(M, Lam, closure)
    if np.any(p <= 0.0):
        raise ValueError("free-energy weight not positive definite (Lambda in [0, 1])")
    sp = np.sqrt(p)
    B = (sp[:, None] * A) / sp[None, :]
    asym = float(np.abs(B - B.T).max() / np.abs(B).max())
    if asym > 1e-12:
        raise ValueError(f"P A is not symmetric (relative asymmetry {asym:.1e}); weights and ladder disagree")
    u, Y = np.linalg.eigh(0.5 * (B + B.T))
    R = Y / sp[:, None]
    R = R * np.where(R[0] < 0.0, -1.0, 1.0)[None, :]
    Wl = p[:, None] * R
    return Slices(M=int(M), Lam=float(Lam), closure=closure, A=A, p=p, u=u, R=R, Wl=Wl)


def coll_rates(M: int, nu: float, hyper_n: int) -> np.ndarray:
    """GANDALF's normalised hypercollision rates nu (m/M)^n for m >= 2 (0 for m = 0, 1), as in the Lawson-RK4 factor
    exp(-nu (m/M)^n dt) and the IMEX implicit operator."""
    m = np.arange(M + 1, dtype=float)
    r = nu * (m / M) ** hyper_n if M >= 2 else np.zeros(M + 1)
    r[:2] = 0.0
    return r


def slice_coupling_matrix(sl: Slices, rates: ArrayLike) -> np.ndarray:
    """Hypercollisions in slice variables: d a/dt|_coll = -K a with K = R^T P diag(rates) R (unit-weight slices,
    symmetric, positive semi-definite).  Its off-diagonal part is the only coupling between slices; the diagonal
    K_jj is the damping rate of slice j."""
    rates = np.asarray(rates, dtype=float)
    return sl.R.T @ (sl.p[:, None] * rates[:, None] * sl.R)


def leakage_first_order(K: np.ndarray, j0: int, t: float | np.ndarray) -> dict[str, np.ndarray]:
    """Short-time first-order prediction for a start in slice j0 alone (a = f e_j0): a_l(t) = -t K_{l j0} f + O(t^2),
    valid while t is short compared with the time over which slices l and j0 decorrelate (s |k_z| |u_l - u_j0| t << 1
    and differential advection small).  Returns, per l != j0, the amplitude ratio |a_l|/|a_j0| = t |K_{l j0}|, its
    maximum, and the energy fraction outside j0, t^2 sum_{l != j0} K_{l j0}^2 (second order in nu; the amplitude
    ratios are first order)."""
    t = np.asarray(t, dtype=float)
    col = np.delete(K[:, j0], j0)
    ratio = np.abs(col)[None, :] * t.reshape(-1, 1)
    return dict(amplitude_ratio=np.squeeze(ratio), amplitude_ratio_max=np.squeeze(ratio.max(axis=1)),
                energy_fraction_outside=np.squeeze(t.reshape(-1) ** 2 * float(np.sum(col ** 2))))


# ------------------------------------------------------------------------------------------------------------------
# GANDALF interface: state, gandalf_step, right-hand side, RK4 for the frozen-field problem, exact k_z-only evolution
# ------------------------------------------------------------------------------------------------------------------
def make_state(box: Box, zp: ArrayLike, zm: ArrayLike, g: ArrayLike, Lam: float, beta: float, nu: float = 0.0,
               t: float = 0.0, check_band: bool = True) -> KRMHDState:
    """GANDALF's KRMHDState (v_th = 1) from rfft coefficients z+, z- [Nz, Ny, Nkx] and g [Nz, Ny, Nkx, M+1].
    With check_band, raises if a field has content outside the 2/3 band (the conservation laws need it inside)."""
    zp = np.asarray(zp, dtype=complex)
    zm = np.asarray(zm, dtype=complex)
    g = np.asarray(g, dtype=complex)
    if check_band:
        for name, f in (("z+", zp), ("z-", zm), ("g", g)):
            if box.band_leakage(f) > 1e-12:
                raise ValueError(f"{name} has content outside the 2/3 band")
    return KRMHDState(z_plus=jnp.asarray(zp), z_minus=jnp.asarray(zm), g=jnp.asarray(g), M=g.shape[-1] - 1,
                      beta_i=float(beta), v_th=1.0, nu=float(nu), Lambda=float(Lam), time=float(t), grid=box.grid)


def fields_of(state: KRMHDState) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(z+, z-, g) of a GANDALF state as numpy arrays."""
    return np.asarray(state.z_plus), np.asarray(state.z_minus), np.asarray(state.g)


@dataclass(frozen=True)
class StepParams:
    """Arguments of krmhd.timestepping.gandalf_step other than the state and dt.  The scheme has no default, so every
    run pins it ("lawson_rk4" or "imex_rk222").  eta, hyper_r: GANDALF's normalised hyper-resistivity
    exp(-eta (k_perp^2/k_perp,max^2)^r dt) on z+- and every g_m; nu, hyper_n: hypercollisions exp(-nu (m/M)^n dt),
    m >= 2; eta_z, hyper_rz: parallel hyper-dissipation; closure: "zero" or "symmetric" (copy)."""

    scheme: str
    eta: float = 0.0
    hyper_r: int = 1
    nu: float = 0.0
    hyper_n: int = 1
    eta_z: float = 0.0
    hyper_rz: int = 1
    closure: str = "zero"
    v_A: float = 1.0

    def __post_init__(self) -> None:
        if self.scheme not in ("lawson_rk4", "imex_rk222"):
            raise ValueError(f"scheme must be 'lawson_rk4' or 'imex_rk222', got {self.scheme!r}")

    @classmethod
    def from_config(cls, d: Mapping[str, Any]) -> "StepParams":
        """Build from a config mapping with keys scheme, eta, hyper_r, nu, hyper_n, eta_z, hyper_rz, closure."""
        kw: dict[str, Any] = {}
        for k in ("eta", "nu", "eta_z", "v_A"):
            if k in d:
                kw[k] = parse_number(d[k])
        for k in ("hyper_r", "hyper_n", "hyper_rz"):
            if k in d:
                kw[k] = int(d[k])
        if "closure" in d:
            kw["closure"] = str(d["closure"])
        return cls(scheme=str(d["scheme"]), **kw)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gandalf_advance(state: KRMHDState, dt: float, nsteps: int, sp: StepParams) -> KRMHDState:
    """Advance a state by nsteps calls of GANDALF's gandalf_step with fixed dt and parameters sp."""
    for _ in range(int(nsteps)):
        state = gandalf_step(state, dt, eta=sp.eta, v_A=sp.v_A, nu=sp.nu, hyper_r=sp.hyper_r, hyper_n=sp.hyper_n,
                             scheme=sp.scheme, eta_z=sp.eta_z, hyper_rz=sp.hyper_rz, closure=sp.closure)
    return state


def make_rhs(box: Box, M: int, Lam: float, beta: float, closure: str = "zero", kz_zero: bool = False):
    """JIT wrappers of GANDALF's _krmhd_rhs_jit (eta = nu = 0, v_A = 1): rhs_full(z+, z-, g) -> (dz+/dt, dz-/dt,
    dg/dt) and rhs_g(z+, z-, g) -> dg/dt.  The g part is -{Phi, g} - s grad_par(A g) (collisions are not in GANDALF's
    right-hand side).  kz_zero=True sets k_z = 0, which gives the explicit part N(g) = -{Phi, g} - s{Psi, A g} that
    both GANDALF steppers integrate (streaming is removed exactly)."""
    N = box.N
    kx, ky, kz, mask = box.jk
    if kz_zero:
        kz = jnp.zeros_like(kz)
    Lam = float(Lam)
    beta = float(beta)

    @jax.jit
    def rhs_full(zp, zm, g):
        d = _krmhd_rhs_jit(KRMHDFields(zp, zm, g, 0.0), kx, ky, kz, mask, 0.0, 1.0, beta, 0.0, Lam, M, N, N, N,
                           closure=closure)
        return d.z_plus, d.z_minus, d.g

    @jax.jit
    def rhs_g(zp, zm, g):
        return _krmhd_rhs_jit(KRMHDFields(zp, zm, g, 0.0), kx, ky, kz, mask, 0.0, 1.0, beta, 0.0, Lam, M, N, N, N,
                              closure=closure).g

    return rhs_full, rhs_g


class FrozenRK4:
    """Classical RK4 on GANDALF's g right-hand side with z+- held fixed (the frozen-field problem):
        d_t g = -{Phi0, g} - s grad_par^(0)(A g) - C g,   Phi0, Psi0 from the frozen z+-,
    with optional m-diagonal collision rates C (e.g. coll_rates).  The frozen-line prediction uses
    (z+, z-) = (Psi0, -Psi0), i.e. Phi = 0 and Psi = Psi(0); z+- = 0 gives the k_z-only linear problem.

    step(zp, zm, g) -> g after one step; advance(zp, zm, g, nsteps) -> g after nsteps (jax.lax.fori_loop)."""

    def __init__(self, box: Box, M: int, Lam: float, beta: float, dt: float, rates: ArrayLike | None = None,
                 closure: str = "zero") -> None:
        _, rhs_g = make_rhs(box, M, Lam, beta, closure)
        c = jnp.asarray(np.zeros(M + 1) if rates is None else np.asarray(rates, dtype=float))
        h = float(dt)
        self.dt = h

        def f(zp, zm, g):
            return rhs_g(zp, zm, g) - c * g

        def step(zp, zm, g):
            k1 = f(zp, zm, g)
            k2 = f(zp, zm, g + 0.5 * h * k1)
            k3 = f(zp, zm, g + 0.5 * h * k2)
            k4 = f(zp, zm, g + h * k3)
            return g + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        self.step = jax.jit(step)
        self._advance = jax.jit(lambda zp, zm, g, n: jax.lax.fori_loop(0, n, lambda i, gg: step(zp, zm, gg), g))

    def advance(self, zp: ArrayLike, zm: ArrayLike, g: ArrayLike, nsteps: int) -> jax.Array:
        """g after nsteps RK4 steps with the frozen z+-."""
        return self._advance(jnp.asarray(zp), jnp.asarray(zm), jnp.asarray(g), int(nsteps))


def make_full_rk4(box: Box, M: int, Lam: float, beta: float, dt: float, rates: ArrayLike | None = None,
                  closure: str = "zero"):
    """Classical RK4 on GANDALF's full right-hand side (z+-, g; ideal Alfvenic sector, optional collision rates on g):
    returns a jitted step(zp, zm, g) -> (zp, zm, g).  An independent integrator for cross-checks of gandalf_step."""
    rhs_full, _ = make_rhs(box, M, Lam, beta, closure)
    c = jnp.asarray(np.zeros(M + 1) if rates is None else np.asarray(rates, dtype=float))
    h = float(dt)

    def f(zp, zm, g):
        a, b, d = rhs_full(zp, zm, g)
        return a, b, d - c * g

    @jax.jit
    def step(zp, zm, g):
        k1 = f(zp, zm, g)
        k2 = f(zp + 0.5 * h * k1[0], zm + 0.5 * h * k1[1], g + 0.5 * h * k1[2])
        k3 = f(zp + 0.5 * h * k2[0], zm + 0.5 * h * k2[1], g + 0.5 * h * k2[2])
        k4 = f(zp + h * k3[0], zm + h * k3[1], g + h * k3[2])
        return tuple(x + (h / 6.0) * (a + 2.0 * b + 2.0 * cc + d)
                     for x, a, b, cc, d in zip((zp, zm, g), k1, k2, k3, k4))

    return step


def kz_only_evolve(box: Box, g0: ArrayLike, sl: Slices, s: float, t: float, rates: ArrayLike | None = None) -> np.ndarray:
    """Exact solution of the k_z-only linear problem (Phi = Psi = 0): g_k(t) = exp[-(i s k_z A + C) t] g_k(0) per
    k_z, with C = diag(rates) (none by default).  Without collisions this is a_j(k, t) = exp(-i s k_z u_j t) a_j(k, 0)."""
    g0 = np.asarray(g0, dtype=complex)
    c = np.zeros(sl.M + 1) if rates is None else np.asarray(rates, dtype=float)
    out = np.empty_like(g0)
    for iz, kz in enumerate(box.kz):
        E = scipy.linalg.expm(-(1j * s * kz * sl.A + np.diag(c)) * t)
        out[iz] = g0[iz] @ E.T
    return out * box._mb(out.ndim)


# ------------------------------------------------------------------------------------------------------------------
# diagnostics (all volume averages, numpy in and out)
# ------------------------------------------------------------------------------------------------------------------
PHI_NAMES: dict[str, str] = {"1": "Q_1", "u": "Q_u", "u2": "Q_u2", "u3": "Q_u3", "1/u": "Q_inv_u"}


def W_m(box: Box, g: ArrayLike) -> np.ndarray:
    """Hermite spectrum W(m) = <g_m^2> (unweighted, volume average), m = 0..M.  Moment energies are p_m W(m)."""
    return box.avg2_last(g)


def free_energy(box: Box, g: ArrayLike, p: ArrayLike) -> float:
    """Free energy W = Q_1 = sum_m p_m <g_m^2> (twice CMM's W on the retained moments)."""
    return float(np.sum(np.asarray(p) * W_m(box, g)))


def slice_energies(box: Box, a: ArrayLike) -> np.ndarray:
    """Slice energies E_j = <a_j^2>; sum_j E_j = sum_m p_m <g_m^2>, and each E_j is conserved by the ideal dynamics."""
    return box.avg2_last(a)


def slice_correlation(box: Box, a: ArrayLike, normalise: bool = False) -> np.ndarray:
    """Slice correlation matrix C_jl = <a_j a_l> (single point, volume average); with normalise, C_jl /
    sqrt(C_jj C_ll).  Its diagonal holds the conserved slice energies; phase mixing is the decay of the off-diagonal
    part along grad_par (two-slice law)."""
    a = np.asarray(a)
    C = np.real(np.einsum("zyxi,zyxj->ij", box._wb(a.ndim) * np.conj(a), a)) / box.Ntot ** 2
    if normalise:
        d = np.sqrt(np.clip(np.diag(C), 1e-300, None))
        C = C / np.outer(d, d)
    return C


def phi_values(u: ArrayLike, name: str) -> np.ndarray:
    """phi(u_j) for name in "1", "u", "u2", "u3", "1/u" (the last needs all u_j != 0, i.e. odd M)."""
    u = np.asarray(u, dtype=float)
    if name == "1":
        return np.ones_like(u)
    if name == "u":
        return u
    if name == "u2":
        return u ** 2
    if name == "u3":
        return u ** 3
    if name == "1/u":
        if np.min(np.abs(u)) < 1e-10 * np.max(np.abs(u)):
            raise ValueError("1/u needs u_j != 0 for every node (odd M)")
        return 1.0 / u
    raise ValueError(f"unknown phi {name!r}")


def Q_phi(box: Box, a: ArrayLike, u: ArrayLike, names: Sequence[str] = ("1", "u", "u2", "u3", "1/u")) -> dict[str, float]:
    """Invariants Q_phi = sum_j phi(u_j) <a_j^2> (= <g^T P phi(A) g>), keyed "Q_1", "Q_u", "Q_u2", "Q_u3", "Q_inv_u".
    Q_inv_u is NaN when a node is at u = 0 (even M).  CMM's W, Gamma, H are Q_1/2, Q_u/2, Q_inv_u/2."""
    E = slice_energies(box, a)
    out: dict[str, float] = {}
    for name in names:
        try:
            out[PHI_NAMES[name]] = float(np.sum(phi_values(u, name) * E))
        except ValueError:
            out[PHI_NAMES[name]] = float("nan")
    return out


def Q_phi_scale(box: Box, a: ArrayLike, u: ArrayLike, name: str) -> float:
    """sum_j |phi(u_j)| <a_j^2>: a natural normalisation for the sign-indefinite Q_u, Q_u3, Q_inv_u (for Q_1 and
    Q_u2 it equals Q_phi)."""
    try:
        return float(np.sum(np.abs(phi_values(u, name)) * slice_energies(box, a)))
    except ValueError:
        return float("nan")


def hermite_flux_field_aligned(box: Box, g: ArrayLike, psi: ArrayLike, s: float, sl: Slices) -> np.ndarray:
    """Field-aligned Hermite flux Pi_{m+1/2} = 2 s p_m A_{m,m+1} <g_m grad_par g_{m+1}>, m = 0..M-1, with the full
    grad_par = d_z + {Psi, .}: the free energy per unit time from moment m to m+1, so that, collisionless,
    d/dt sum_{n<=m} p_n <g_n^2> = -Pi_{m+1/2} for any Alfvenic fields.  Equals twice GANDALF's (>= 0.7.0)
    krmhd.diagnostics.hermite_flux summed over modes with the rfft weights (gandalf#159)."""
    g = np.asarray(g)
    psi = np.asarray(psi)
    M = g.shape[-1] - 1
    out = np.zeros(M)
    for m in range(M):
        out[m] = 2.0 * s * sl.p[m] * sl.A[m, m + 1] * box.avg(g[..., m], box.gradpar(psi, g[..., m + 1]))
    return out


def hermite_flux_kz(box: Box, g: ArrayLike, s: float, sl: Slices, per_mode: bool = False) -> np.ndarray:
    """k_z-only Hermite flux Pi^z_{m+1/2} = 2 s p_m A_{m,m+1} <g_m d_z g_{m+1}>, i.e. s p_m times GANDALF's
    krmhd.diagnostics.hermite_flux summed over modes up to GANDALF 0.6.0, which omitted {Psi, .}, s and p_0 (from
    0.7.0 hermite_flux is field-aligned; see hermite_flux_field_aligned).  per_mode returns the
    k-resolved contributions [Nz, Ny, Nkx, M] (already weighted, summing to the total)."""
    g = np.asarray(g)
    M = g.shape[-1] - 1
    coef = 2.0 * s * sl.p[:M] * np.array([sl.A[m, m + 1] for m in range(M)])
    pm = -box.KZ[..., None] * coef * np.imag(np.conj(g[..., :-1]) * g[..., 1:]) * box._wb(4) / box.Ntot ** 2
    return pm if per_mode else pm.sum(axis=(0, 1, 2))


def quartic_moments(box: Box, g: ArrayLike, ms: Sequence[int] = (0, 1, 2), pad: int = 2) -> np.ndarray:
    """Single-point quartic statistic <g_m^4> for the moments ms, from real-space values on a grid refined by `pad`.
    pad >= 2 makes the mean of g^4 exact for fields inside the 2/3 band (no aliasing into k = 0)."""
    g = np.asarray(g)
    if pad < 2:
        raise ValueError("pad >= 2 is needed for an exact <g^4>")
    return np.array([float(np.mean(box.to_real(g[..., m], pad=pad) ** 4)) for m in ms])


def free_energy_density_k(box: Box, g: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Per-mode contribution [Nz, Ny, Nkx] to W = sum_m p_m <g_m^2> (rfft weights included)."""
    g = np.asarray(g)
    return box.w * np.einsum("zyxm,m->zyx", np.abs(g) ** 2, np.asarray(p, dtype=float)) / box.Ntot ** 2


def alfvenic_energy_density_k(box: Box, zp: ArrayLike, zm: ArrayLike) -> np.ndarray:
    """Per-mode contribution [Nz, Ny, Nkx] to E_A = (1/2)<|grad Phi|^2 + |grad Psi|^2>."""
    phi, psi = phi_psi(zp, zm)
    return 0.5 * box.w * box.KP2 * (np.abs(phi) ** 2 + np.abs(psi) ** 2) / box.Ntot ** 2


def W_kperp(box: Box, g: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Free energy in integer k_perp shells n = rint(|k_perp|/k1), n = 0..nperp_band (the last shells are the partly
    filled corners of the square band).  Not a single-point statistic."""
    dens = free_energy_density_k(box, g, p)
    shell = np.broadcast_to(box.nperp, dens.shape)
    return np.bincount(shell.ravel(), weights=dens.ravel(), minlength=box.nperp_band + 1)[:box.nperp_band + 1]


def W_kz(box: Box, g: ArrayLike, p: ArrayLike) -> np.ndarray:
    """Free energy in |k_z| shells n_z = rint(|k_z|/k1), n_z = 0..nz_band (Eulerian k_z spectrum; not single-point)."""
    dens = free_energy_density_k(box, g, p).sum(axis=(1, 2))
    return np.bincount(box.nz, weights=dens, minlength=box.nz_band + 1)[:box.nz_band + 1]


def alfvenic_energy(box: Box, zp: ArrayLike, zm: ArrayLike) -> dict[str, float]:
    """Alfvenic energies (volume averages, v_A = 1): E = (1/2)<|grad Phi|^2 + |grad Psi|^2> = E_plus + E_minus with
    E_pm = (1/4)<|grad z+-|^2>; E_kin, E_mag; cross-helicity <grad Phi . grad Psi> = E_plus - E_minus.  GANDALF's
    krmhd.physics.energy(state)["total"] equals N_z N^3 E in these units."""
    phi, psi = phi_psi(zp, zm)
    Ek = 0.5 * box.rms_grad(phi) ** 2
    Em = 0.5 * box.rms_grad(psi) ** 2
    Ep = 0.25 * box.rms_grad(np.asarray(zp)) ** 2
    Emi = 0.25 * box.rms_grad(np.asarray(zm)) ** 2
    return dict(E=Ek + Em, E_kin=Ek, E_mag=Em, E_plus=Ep, E_minus=Emi, cross_helicity=Ep - Emi)


def nonlinearity_parameter(box: Box, zp: ArrayLike, zm: ArrayLike, v_A: float = 1.0) -> dict[str, float]:
    """Outer-scale nonlinearity parameter chi = kbar_perp z / (kbar_par v_A), with energy-weighted mean wavenumbers
    kbar_perp = <|k_perp|>_E and kbar_par = <|k_z|>_E (weights: the Alfvenic energy per mode).  Reported for
    z = u_rms = rms|grad Phi| (chi_u) and for the rms Elsasser amplitude z_rms = sqrt((<|grad z+|^2> +
    <|grad z-|^2>)/2) (chi_z, about sqrt2 chi_u for balanced fields), with tau_nl = 1/(kbar_perp u_rms) and
    tau_A = 1/(kbar_par v_A)."""
    e = alfvenic_energy_density_k(box, zp, zm)
    tot = float(e.sum())
    kp = float(np.sum(np.sqrt(box.KP2) * e)) / tot
    kz = float(np.sum(np.abs(box.KZ) * e)) / tot
    phi, _ = phi_psi(zp, zm)
    urms = box.rms_grad(phi)
    zrms = math.sqrt(0.5 * (box.rms_grad(np.asarray(zp)) ** 2 + box.rms_grad(np.asarray(zm)) ** 2))
    return dict(kperp_mean=kp, kpar_mean=kz, u_rms=urms, z_rms=zrms, chi_u=kp * urms / (kz * v_A),
                chi_z=kp * zrms / (kz * v_A), tau_nl=1.0 / (kp * urms), tau_A=1.0 / (kz * v_A))


def outer_shell_fraction(box: Box, dens: ArrayLike) -> dict[str, float]:
    """Resolution check: fraction of a per-mode energy density [Nz, Ny, Nkx] (free_energy_density_k or
    alfvenic_energy_density_k) in the outermost complete k_perp shell n = nmax (the largest retained |n_x|) alone
    ("shell") and in n >= nmax including the partly filled corner shells ("beyond")."""
    dens = np.asarray(dens)
    shell = np.broadcast_to(box.nperp, dens.shape)
    tot = float(dens.sum())
    if tot <= 0.0:
        return dict(nmax=box.nmax, shell=0.0, beyond=0.0)
    return dict(nmax=box.nmax, shell=float(dens[shell == box.nmax].sum()) / tot,
                beyond=float(dens[shell >= box.nmax].sum()) / tot)


def resolution_check(box: Box, sl: Slices, zp: ArrayLike, zm: ArrayLike, g: ArrayLike) -> dict[str, float]:
    """Outer-shell fractions of the free energy of g and of the Alfvenic energy (see outer_shell_fraction)."""
    fg = outer_shell_fraction(box, free_energy_density_k(box, g, sl.p))
    fa = outer_shell_fraction(box, alfvenic_energy_density_k(box, zp, zm))
    return dict(nmax=box.nmax, g_shell=fg["shell"], g_beyond=fg["beyond"], alfven_shell=fa["shell"],
                alfven_beyond=fa["beyond"])


def dissipation_rates(box: Box, sl: Slices, zp: ArrayLike, zm: ArrayLike, g: ArrayLike, eta: float = 0.0,
                      hyper_r: int = 1, nu: float = 0.0, hyper_n: int = 1) -> dict[str, float]:
    """Instantaneous loss rates under GANDALF's dissipation: eps_eta_g = 2 sum_k eta (k_perp^2/k_perp,max^2)^r
    (free-energy density of g), eps_eta_A (same for the Alfvenic energy), eps_nu = 2 sum_m nu (m/M)^n p_m <g_m^2>
    (k_perp,max^2 = GANDALF's normalisation kperp_max2_gandalf), so dW/dt = -(eps_eta_g + eps_nu) and
    dE_A/dt = -eps_eta_A up to time-stepping error."""
    rk = eta * (box.KP2 / box.kperp_max2_gandalf) ** hyper_r
    eg = 2.0 * float(np.sum(rk * free_energy_density_k(box, g, sl.p)))
    ea = 2.0 * float(np.sum(rk * alfvenic_energy_density_k(box, zp, zm)))
    en = 2.0 * float(np.sum(coll_rates(sl.M, nu, hyper_n) * sl.p * W_m(box, g)))
    return dict(eps_eta_g=eg, eps_eta_A=ea, eps_nu=en)


def slice_leakage(box: Box, a: ArrayLike, j0: int) -> dict[str, float]:
    """Leakage out of slice j0: energy fraction outside j0, 1 - E_j0/sum_j E_j (second order in the coupling), and
    the amplitude ratios sqrt(sum_{l != j0} E_l / E_j0) (rms) and max_{l != j0} sqrt(E_l/E_j0) (first order)."""
    E = slice_energies(box, a)
    out = np.delete(E, j0)
    return dict(energy_fraction_outside=float(out.sum() / E.sum()),
                amplitude_ratio_rms=float(math.sqrt(out.sum() / E[j0])),
                amplitude_ratio_max=float(math.sqrt(out.max() / E[j0])))


def diagnostics(box: Box, sl: Slices, s: float, zp: ArrayLike, zm: ArrayLike, g: ArrayLike, *,
                correlation: bool = True, flux: bool = True, quartic_ms: Sequence[int] | None = (0, 1, 2),
                spectra: bool = True, pad: int = 2) -> dict[str, Any]:
    """The standard set of diagnostics at one time, as a flat dict for stack_records/np.savez:
    W_m [M+1], pW_m [M+1], E_slice [M+1], Q_1, Q_u, Q_u2, Q_u3, Q_inv_u, E_A, E_kin, E_mag, E_plus, E_minus,
    cross_helicity, res_g_shell, res_g_beyond, res_alfven_shell, res_alfven_beyond, and optionally C_slice [M+1, M+1]
    (slice correlation), Pi_fa [M] (field-aligned flux), Pi_kz [M] (k_z-only flux), g4 [len(quartic_ms)] (<g_m^4>),
    W_kperp, W_kz."""
    zp = np.asarray(zp)
    zm = np.asarray(zm)
    g = np.asarray(g)
    _, psi = phi_psi(zp, zm)
    a = sl.amplitudes(g)
    Wm = W_m(box, g)
    out: dict[str, Any] = dict(W_m=Wm, pW_m=sl.p * Wm, E_slice=slice_energies(box, a))
    out.update(Q_phi(box, a, sl.u))
    out.update(alfvenic_energy(box, zp, zm))
    out.update({f"res_{k}": v for k, v in resolution_check(box, sl, zp, zm, g).items() if k != "nmax"})
    if correlation:
        out["C_slice"] = slice_correlation(box, a)
    if flux:
        out["Pi_fa"] = hermite_flux_field_aligned(box, g, psi, s, sl)
        out["Pi_kz"] = hermite_flux_kz(box, g, s, sl)
    if quartic_ms:
        out["g4"] = quartic_moments(box, g, quartic_ms, pad=pad)
    if spectra:
        out["W_kperp"] = W_kperp(box, g, sl.p)
        out["W_kz"] = W_kz(box, g, sl.p)
    return out


# ------------------------------------------------------------------------------------------------------------------
# time-step limits of the explicit slice advection
# ------------------------------------------------------------------------------------------------------------------
def advection_operator_norm(box: Box, f: ArrayLike, tol: float = 1e-6, seed: int = 0) -> float:
    """Spectral radius of GANDALF's dealiased advection operator B a = P{f, a} on real fields inside the 2/3 band
    (P the band projection).  B is skew-symmetric there, so its spectral radius equals its norm,
    sqrt(lambda_max(-B^2)), computed by Lanczos (scipy eigsh) with two brackets per iteration.  For slice j,
    f = Phi + s u_j Psi; the Lawson-RK4 limit is dt <= 2 sqrt2 / ||B||."""
    from scipy.sparse.linalg import LinearOperator, eigsh
    N = box.N
    kx, ky, _, mask = box.jk
    fj = jnp.asarray(f)

    @jax.jit
    def apply(v):
        a = jnp.fft.rfftn(v.reshape(N, N, N), axes=(0, 1, 2)) * mask
        b = poisson_bracket_3d(fj, a, kx, ky, N, N, N, mask)
        return jnp.fft.irfftn(b, s=(N, N, N), axes=(0, 1, 2)).ravel()

    def mv(v: np.ndarray) -> np.ndarray:
        x = jnp.asarray(np.asarray(v, dtype=float).ravel())
        return -np.asarray(apply(apply(x)))

    op = LinearOperator((N ** 3, N ** 3), matvec=mv, dtype=float)
    v0 = np.random.default_rng(seed).standard_normal(N ** 3)
    lam = eigsh(op, k=1, which="LA", tol=tol, v0=v0, return_eigenvectors=False)
    return float(math.sqrt(max(float(lam[0]), 0.0)))


def slice_advection_rate(box: Box, zp: ArrayLike, zm: ArrayLike, s: float, u: ArrayLike, pad: int = 2,
                         exact: bool = False) -> dict[str, Any]:
    """Explicit slice-advection rate.  Both GANDALF steppers treat N(g) = -{Phi, g} - s{Psi, A g} explicitly; in slice
    j this is advection by zhat x grad(Phi + s u_j Psi), a skew operator whose spectral radius is bounded by
        rho = k_perp,max * max_j max_x |grad_perp(Phi + s u_j Psi)|,
    k_perp,max the largest retained |k_perp| (the band corner).  Lawson-RK4 is then stable for dt <= 2 sqrt2/rho.
    The maximum over x is taken on a grid refined by `pad` (the grid maximum of a band-limited field underestimates
    its true maximum).  Returns V [M+1] (max speed per slice), Vmax, j_fastest, kperp_max, rate (rho) and
    dt_lawson_max.  The bound is safe but not sharp (about 5 times the true radius on an 8^3 grid).  With exact,
    also the true spectral radius rho_exact = max(||B_0||, ||B_M||) of the slice operators (advection_operator_norm;
    the norm is convex in u, so the extreme nodes bound every slice) and dt_lawson_exact = 2 sqrt2/rho_exact."""
    phi, psi = phi_psi(zp, zm)
    fx, fy = box.grad_perp_real(phi, pad)
    sx, sy = box.grad_perp_real(psi, pad)
    u = np.asarray(u, dtype=float)
    V = np.array([float(np.sqrt((fx + s * uj * sx) ** 2 + (fy + s * uj * sy) ** 2).max()) for uj in u])
    rate = box.kperp_max * float(V.max())
    out: dict[str, Any] = dict(V=V, Vmax=float(V.max()), j_fastest=int(np.argmax(V)), kperp_max=box.kperp_max,
                               rate=rate, dt_lawson_max=LAWSON_RK4_LIMIT / rate if rate > 0 else math.inf)
    if exact:
        rho = max(advection_operator_norm(box, phi + s * u[0] * psi), advection_operator_norm(box, phi + s * u[-1] * psi))
        out.update(rho_exact=rho, dt_lawson_exact=LAWSON_RK4_LIMIT / rho if rho > 0 else math.inf)
    return out


def recommended_dt(box: Box, state: KRMHDState, sl: Slices, cfl_safety: float, lawson_fraction: float,
                   v_A: float = 1.0, pad: int = 2) -> dict[str, Any]:
    """dt = min(GANDALF's compute_cfl_timestep(state, v_A, cfl_safety), lawson_fraction * 2 sqrt2/rho), with rho
    from slice_advection_rate.  Returns dt, dt_cfl, which limit binds, and the slice-advection numbers."""
    dt_cfl = float(compute_cfl_timestep(state, v_A=v_A, cfl_safety=cfl_safety))
    zp, zm, _ = fields_of(state)
    lim = slice_advection_rate(box, zp, zm, math.sqrt(state.beta_i), sl.u, pad=pad)
    dt_l = lawson_fraction * lim["dt_lawson_max"]
    return dict(dt=min(dt_cfl, dt_l), dt_cfl=dt_cfl, dt_lawson=dt_l, binding="cfl" if dt_cfl <= dt_l else "lawson",
                **lim)


def rk_stability(rate: float, dt: float, nsteps: int = 1) -> dict[str, float]:
    """Stability factors on the imaginary axis at y = rate dt: classical RK4 (Lawson-RK4 at k_z = 0)
    |R4(iy)| = |1 + iy - y^2/2 - iy^3/6 + y^4/24| (<= 1 iff y <= 2 sqrt2), and the explicit part of IMEX-RK222
    |R2(iy)| = sqrt(1 + y^4/4) > 1 (pure advection grows unless damped); also their nsteps-th powers (worst-case
    amplification of the fastest slice mode over nsteps)."""
    y = rate * dt
    r4 = abs(1 + 1j * y - y ** 2 / 2 - 1j * y ** 3 / 6 + y ** 4 / 24)
    r2 = math.sqrt(1.0 + y ** 4 / 4.0)
    return dict(y=y, R4=r4, R2=r2, R4_n=r4 ** nsteps, R2_n=r2 ** nsteps)


# ------------------------------------------------------------------------------------------------------------------
# gates
# ------------------------------------------------------------------------------------------------------------------
def relative_drift(series: ArrayLike) -> float:
    """max_t |X(t)/X(0) - 1| for a time series X."""
    x = np.asarray(series, dtype=float)
    return float(np.max(np.abs(x / x[0] - 1.0)))


def gate_energy_conservation(W: ArrayLike | None = None, E_A: ArrayLike | None = None, tol: float = 0.01) -> dict[str, Any]:
    """Energy-conservation gate for ideal runs: the relative drift of the conserved free energy W = Q_1 and of the
    Alfvenic energy must be below tol (1%)."""
    out: dict[str, Any] = dict(applies=True, criterion=f"max_t |X(t)/X(0) - 1| < {tol} for X = W (free energy) and "
                                                         f"E_A (Alfvenic energy)", tol=tol)
    ok = True
    if W is not None:
        out["free_energy_rel_drift"] = relative_drift(W)
        ok = ok and out["free_energy_rel_drift"] < tol
    if E_A is not None:
        out["alfvenic_energy_rel_drift"] = relative_drift(E_A)
        ok = ok and out["alfvenic_energy_rel_drift"] < tol
    out["pass"] = bool(ok)
    return out


def make_gates(energy_conservation: Mapping[str, Any] | str, resolution: Mapping[str, Any] | None = None,
               energy_balance: Mapping[str, Any] | str = NOT_APPLICABLE_SHORT,
               steady_state: Mapping[str, Any] | str = NOT_APPLICABLE_SHORT,
               spectral_sanity: Mapping[str, Any] | str = NOT_APPLICABLE_SHORT, **others: Any) -> dict[str, Any]:
    """The gate record of a run.  energy_conservation is a dict from gate_energy_conservation (ideal runs) or a
    string "not applicable: <reason>" (e.g. for runs with hyper-resistivity or hypercollisions); energy balance,
    steady state and spectral sanity default to "not applicable: short verification run"; resolution (e.g. the
    outer-shell fractions of resolution_check at the final time) is recorded, not gated, unless it carries
    applies=True and pass.  Extra keyword gates are stored as given."""
    gates: dict[str, Any] = dict(energy_conservation=energy_conservation, energy_balance=energy_balance,
                                 steady_state=steady_state, spectral_sanity=spectral_sanity)
    if resolution is not None:
        res = dict(resolution)
        res.setdefault("applies", False)
        res.setdefault("note", "recorded: fraction of energy in the outermost complete k_perp shell (n = nmax) and "
                               "in n >= nmax including the band corners")
        gates["resolution_check"] = res
    gates.update(others)
    return _jsonable(gates)


def gates_outcome(gates: Mapping[str, Any]) -> str:
    """'pass' if every gate given as a dict with applies=True has pass=True, else 'fail'."""
    for g in gates.values():
        if isinstance(g, Mapping) and g.get("applies", False) and not g.get("pass", False):
            return "fail"
    return "pass"


def require_gates(gates: Mapping[str, Any]) -> None:
    """Raise RuntimeError naming every applicable gate that failed (CLAUDE.md: stop and report)."""
    bad = [k for k, g in gates.items() if isinstance(g, Mapping) and g.get("applies", False) and not g.get("pass")]
    if bad:
        raise RuntimeError(f"gate(s) failed: {', '.join(bad)}: " + json.dumps(_jsonable({k: gates[k] for k in bad})))


def budget_residual(t: ArrayLike, E: ArrayLike, eps: ArrayLike) -> np.ndarray:
    """Relative energy-budget residual (E(t) - E(0) + int_0^t eps dt')/E(0), with the integral by the trapezoidal
    rule over the sampled times (accurate only for output intervals short compared with the decay time)."""
    t = np.asarray(t, dtype=float)
    E = np.asarray(E, dtype=float)
    eps = np.asarray(eps, dtype=float)
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (eps[1:] + eps[:-1]) * np.diff(t))])
    return (E - E[0] + integ) / E[0]


# ------------------------------------------------------------------------------------------------------------------
# figures (optional helpers for the analysis scripts; matplotlib is imported only here)
# ------------------------------------------------------------------------------------------------------------------
JPP_SINGLE_IN = 3.4
JPP_DOUBLE_IN = 7.0
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]


def jpp_style(usetex: bool = True) -> bool:
    """Set matplotlib rcParams for the Journal of Plasma Physics: 10 pt serif text, viridis for images, the Okabe-Ito
    palette for lines, small legends.  With usetex, text.usetex is switched on if LaTeX renders a test string;
    returns whether LaTeX is in use (False means mathtext fallback, which the caller should report)."""
    import matplotlib
    if "matplotlib.pyplot" not in sys.modules:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 8, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "font.family": "serif", "mathtext.fontset": "cm", "image.cmap": "viridis",
        "axes.prop_cycle": cycler(color=OKABE_ITO), "legend.frameon": False, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "figure.dpi": 150, "lines.linewidth": 1.2,
    })
    ok = False
    if usetex and shutil.which("latex"):
        plt.rcParams["text.usetex"] = True
        plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
        try:
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r"$k_\perp \rho_i$, $W(m)$")
            fig.canvas.draw()
            plt.close(fig)
            ok = True
        except Exception:
            plt.rcParams["text.usetex"] = False
    else:
        plt.rcParams["text.usetex"] = False
    return ok


def save_figure(fig: Any, name: str, dpi: int = 300) -> tuple[Path, Path]:
    """Save a figure as figures/<name>.pdf and figures/<name>.png; returns both paths."""
    FIGURES.mkdir(parents=True, exist_ok=True)
    pdf = FIGURES / f"{name}.pdf"
    png = FIGURES / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    return pdf, png
