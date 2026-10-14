#!/usr/bin/env python
"""Figure fig:tilt: the k_z-based split of the Hermite flux misreports phase mixing as unmixing when the field lines
are tilted.

Physics. Free streaming (Lambda -> infinity, collisionless, no flow) along a uniformly tilted static field,
grad_par = d_z + b0 d_x, from a density perturbation g(k, 0) = c_k e_0: each Fourier mode evolves with
k_par = k_z + b0 k_x, |g_m|^2 is Poisson in m with mean lambda = (s k_par t)^2/2, and the field-aligned Hermite flux
Pi_m(k) = |c_k|^2 (2 lambda/t) e^-lambda lambda^m/m! is non-negative at every m and t (pure phase mixing). The flux
that a k_z-based diagnostic reports (GANDALF's krmhd.diagnostics.hermite_flux) is Pi^z_m(k) = (k_z/k_par) Pi_m(k),
which is negative on the modes with k_z k_par < 0. For a Gaussian spectrum S(k) ~ exp(-k_z^2/2 sigma_z^2 -
k_x^2/2 sigma_x^2) with effective tilt angle alpha, tan(alpha) = b0 sigma_x/sigma_z, the reported backward and
forward fluxes are (der-dynamics eq. 2.5; paper app:tilt), with y = k_par/sigma_par, tau = s sigma_par t,
    B^z_m = int dy phi(y) [cos a sin a phi(|y| cot a)/|y| - cos^2 a Phi(-|y| cot a)] Pi_m(sigma_par y, t),
    F^z_m = int dy phi(y) [cos a sin a phi(|y| cot a)/|y| + cos^2 a Phi(+|y| cot a)] Pi_m(sigma_par y, t),
phi and Phi the standard normal density and distribution function, while the true backward flux is zero. The figure
shows the reported backward share B^z_m/(F^z_m + B^z_m) against tau at m = 1, 4, 16 for three tilt angles (quadrature,
x = tau y as integration variable), and, as markers, GANDALF's own hermite_flux (backward share) and
phase_unmixing_energy (unmixing share of the energy at m) evaluated on the exact tilted-field solution in a periodic
box at alpha = 45 deg: the construction of derivations/dynamics/d06_tilt_kz_decomposition.py (T2), i.e. a 32^3 grid,
a random Gaussian spectrum cut at |k_par| < sqrt(2M/3)/(s t(tau_cut)), and the exact solution
g(k, t) = exp(-i s k_par t A) e_0 c_k through the eigen-decomposition of GANDALF's ladder A at M = 256 (no mode
reaches the truncation). The checks reproduce d06's printed numbers (T2, T3, T4).

The numbers are deterministic (no time integration). They are computed from configs/figures.yaml (section "tilt"),
cached in data/computed/fig_tilt.npz (+ .json), and plotted from the cache to figures/tilt.pdf and .png.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/fig_tilt.py [--recompute] [--tag TAG]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import yaml  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.special import gammaln, ndtr  # noqa: E402

from krmhd.diagnostics import hermite_flux, phase_mixing_energy, phase_unmixing_energy  # noqa: E402
from krmhd.hermite import compute_streaming_matrix  # noqa: E402
from krmhd.physics import KRMHDState  # noqa: E402
from krmhd.spectral import SpectralGrid3D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
CONFIG = os.path.join(PAPER, "configs", "figures.yaml")
CACHE_DIR = os.path.join(PAPER, "data", "computed")
FIG_DIR = os.path.join(PAPER, "figures")
NAME = "tilt"
GATES = {g: "not applicable: deterministic computation, no time integration"
         for g in ("energy_conservation", "energy_balance", "steady_state", "spectral_sanity")}
SQRT2PI = math.sqrt(2.0 * math.pi)


# ------------------------------------------------------------------------------------------------------------------
# configuration, metadata, cache
# ------------------------------------------------------------------------------------------------------------------
def load_config() -> tuple[dict, dict]:
    """This figure's section of configs/figures.yaml and the shared style section."""
    with open(CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    return cfg[NAME], cfg["style"]


def config_hash(section: dict) -> str:
    """Short hash of a config section; the cache is reused only if it matches."""
    return hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode()).hexdigest()[:16]


def environment() -> dict:
    """Versions, precision and hardware recorded with the cached numbers."""
    import krmhd
    import mpmath
    import scipy
    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                             timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        cpu = platform.processor()
    return dict(krmhd=getattr(krmhd, "__version__", "unknown"), jax=jax.__version__, numpy=np.__version__,
                scipy=scipy.__version__, mpmath=mpmath.__version__, python=platform.python_version(),
                precision="float64 (jax_enable_x64 = True)", hardware=f"{platform.system()} {platform.machine()}, {cpu}")


def cache_paths() -> tuple[str, str]:
    return os.path.join(CACHE_DIR, f"fig_{NAME}.npz"), os.path.join(CACHE_DIR, f"fig_{NAME}.json")


def read_meta() -> dict | None:
    npz, js = cache_paths()
    if not (os.path.exists(npz) and os.path.exists(js)):
        return None
    with open(js) as fh:
        return json.load(fh)


def write_meta(meta: dict) -> None:
    with open(cache_paths()[1], "w") as fh:
        json.dump(meta, fh, indent=1, default=float)


class Checks:
    """Named pass/fail comparisons, printed and stored with the cache."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def __call__(self, name: str, ok: bool, detail: str = "") -> bool:
        ok = bool(ok)
        self.rows.append(dict(name=name, passed=ok, detail=detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" :: {detail}" if detail else ""), flush=True)
        return ok

    @property
    def all_passed(self) -> bool:
        return all(r["passed"] for r in self.rows)


# ------------------------------------------------------------------------------------------------------------------
# Gaussian ensemble (der-dynamics eqs. 2.5 and 2.6)
# ------------------------------------------------------------------------------------------------------------------
def _phi(y: float) -> float:
    """Standard normal density."""
    return math.exp(-0.5 * y * y) / SQRT2PI


def ensemble_fluxes(alpha: float, tau: float, m: int) -> tuple[float, float, float]:
    """Reported backward flux B^z_m, reported forward flux F^z_m and true net flux of the Gaussian ensemble, all
    multiplied by the same positive factor tau^2/2 (units sigma_par = s = 1, t = tau; integration variable
    x = tau y = s k_par t, so the free-streaming kernel x^2 e^{-x^2/2} (x^2/2)^m/m! is fixed while the tilt factor
    varies on the scale x ~ tau)."""
    ca, sa = math.cos(alpha), math.sin(alpha)
    cot = ca / sa

    def kernel(x: float) -> float:
        lam = 0.5 * x * x
        if lam <= 0.0:
            return 0.0
        return math.exp(math.log(x * x) - lam + m * math.log(lam) - gammaln(m + 1))

    def w_back(x: float) -> float:
        y = x / tau
        return _phi(y) * (ca * sa * _phi(y * cot) / y - ca * ca * float(ndtr(-y * cot))) if y > 0 else 0.0

    def w_fwd(x: float) -> float:
        y = x / tau
        return _phi(y) * (ca * sa * _phi(y * cot) / y + ca * ca * float(ndtr(y * cot))) if y > 0 else 0.0

    xc = math.sqrt(2.0 * m + 2.0)
    kw = dict(limit=500, epsabs=0.0, epsrel=1e-12, points=[max(xc - 8.0, 0.0), xc, xc + 12.0])
    top = xc + 40.0
    B = quad(lambda x: w_back(x) * kernel(x), 0.0, top, **kw)[0]
    F = quad(lambda x: w_fwd(x) * kernel(x), 0.0, top, **kw)[0]
    T = quad(lambda x: _phi(x / tau) * kernel(x), 0.0, top, **kw)[0]
    return B, F, T


def backward_share(alpha: float, tau: float, m: int) -> float:
    B, F, _ = ensemble_fluxes(alpha, tau, m)
    return B / (F + B) if F + B > 0 else 0.0


def true_flux_closed(tau: float, m: int) -> float:
    """Exact true net flux t Pi_m of the ensemble (der-dynamics eq. 2.6), tau = s sigma_par t."""
    log_df = gammaln(2 * m + 2) - m * math.log(2.0) - gammaln(m + 1)        # (2m+1)!! = (2m+1)!/(2^m m!)
    return 2.0 * math.exp((m + 1) * math.log(tau * tau / 2.0) + log_df - gammaln(m + 1)
                          - (m + 1.5) * math.log1p(tau * tau))


# ------------------------------------------------------------------------------------------------------------------
# exact tilted-field solution in a periodic box with GANDALF's diagnostics (d06, T2)
# ------------------------------------------------------------------------------------------------------------------
def poisson_flux(kpar: np.ndarray, t: float, m: int, s: float) -> np.ndarray:
    """Field-aligned free-streaming flux through m per unit |c_k|^2: (2 lambda/t) e^-lambda lambda^m/m!."""
    lam = 0.5 * (s * kpar * t) ** 2
    out = np.zeros_like(lam)
    pos = lam > 0
    out[pos] = np.exp(np.log(2.0 * lam[pos] / t) - lam[pos] + m * np.log(lam[pos]) - gammaln(m + 1))
    return out


def box_diagnostics(bcfg: dict, s: float, chk: Checks) -> dict:
    """GANDALF's hermite_flux, phase_mixing_energy and phase_unmixing_energy on the exact solution, the expectation
    of the same shares over realisations of this box spectrum (|c_k|^2 -> S_k), and the d06 identities."""
    N, M, b0 = int(bcfg["N"]), int(bcfg["M"]), float(bcfg["b0"])
    lam_big = float(bcfg["Lambda"])
    grid = SpectralGrid3D.create(Nx=N, Ny=N, Nz=N, Lx=1.0, Ly=1.0, Lz=1.0)
    kx, ky, kz = np.asarray(grid.kx), np.asarray(grid.ky), np.asarray(grid.kz)
    mask = np.asarray(grid.dealias_mask)
    shape = (N, N, N // 2 + 1)
    KX = np.broadcast_to(kx[None, None, :], shape)
    KY = np.broadcast_to(ky[None, :, None], shape)
    KZ = np.broadcast_to(kz[:, None, None], shape)
    wk = np.full(N // 2 + 1, 2.0)
    wk[0] = wk[-1] = 1.0
    wk = np.broadcast_to(wk[None, None, :], shape)                 # rfft weights: full-k sums
    KPAR = KZ + b0 * KX
    sz, sx, sy = (2.0 * np.pi * float(bcfg[k]) for k in ("sigma_z", "sigma_x", "sigma_y"))
    spar = math.sqrt(sz ** 2 + b0 ** 2 * sx ** 2)
    t_cut = float(bcfg["tau_cut"]) / (s * spar)
    kcut = math.sqrt(2.0 * M / 3.0) / (s * t_cut)
    S = np.exp(-KZ ** 2 / (2 * sz ** 2) - KX ** 2 / (2 * sx ** 2) - KY ** 2 / (2 * sy ** 2)) * mask
    S = S * (np.abs(KPAR) < kcut)
    rng = np.random.default_rng(int(bcfg["seed"]))
    c = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) * np.sqrt(S)
    c[0, 0, 0] = 0.0
    A = np.asarray(compute_streaming_matrix(M, lam_big, "zero"), dtype=float)
    chk("box: GANDALF's ladder is symmetric for Lambda -> infinity", np.abs(A - A.T).max() == 0.0)
    evals, R = np.linalg.eigh(A)
    coef = R.T[:, 0]                                               # e_0 in the eigenbasis
    opp = (KZ * KPAR < 0) & mask & (S > 0)
    opp_w = float((wk * S)[opp].sum() / (wk * S)[mask].sum())
    alpha = math.degrees(math.atan(b0 * sx / sz))
    print(f"    box: N = {N}, M = {M}, b0 = {b0}, alpha = {alpha:.2f} deg, sigma_par = {spar:.4f}, cut |k_par| < "
          f"{kcut:.2f}, weight on modes with k_z k_par < 0 = {opp_w:.4f} (continuum alpha/pi = {alpha / 180:.4f})")
    m_diag = [int(x) for x in bcfg["m_diag"]]
    taus = [float(x) for x in bcfg["tau"]]
    coup = np.sqrt(2.0 * (np.arange(M) + 1.0))                     # 2 A_{m,m+1}
    zeros = jnp.zeros(shape, dtype=jnp.complex128)
    out = {k: np.zeros((len(taus), len(m_diag))) for k in ("bflux", "unmix", "bflux_exp", "unmix_exp")}
    worst_id, worst_neg = 0.0, 0.0
    for i, tau in enumerate(taus):
        t = tau / (s * spar)
        ph = np.exp(-1j * s * KPAR[..., None] * t * evals[None, None, None, :])
        g = ((ph * coef) @ R.T) * c[..., None]                     # exact solution, every mode
        st = KRMHDState(z_plus=zeros, z_minus=zeros, g=jnp.asarray(g), M=M, beta_i=s * s, v_th=1.0, nu=0.0,
                        Lambda=lam_big, time=t, grid=grid)
        fz = np.asarray(hermite_flux(st))                          # -k_z sqrt(2(m+1)) Im(g_{m+1} g_m^*)
        true = -s * KPAR[..., None] * coup * np.imag(np.conj(g[..., :-1]) * g[..., 1:])
        with np.errstate(divide="ignore", invalid="ignore"):
            pred = np.where(np.abs(KPAR[..., None]) > 0, (KZ / KPAR)[..., None] * true, 0.0)
        worst_id = max(worst_id, float(np.abs(fz - pred).max() / np.abs(fz).max()))
        worst_neg = min(worst_neg, float(true.min() / np.abs(true).max()))
        for j, m in enumerate(m_diag):
            un = phase_unmixing_energy(st, m)
            mi = phase_mixing_energy(st, m)
            f = fz[..., m]
            back = -float((wk * np.where(f < 0, f, 0.0)).sum())
            fwd = float((wk * np.where(f > 0, f, 0.0)).sum())
            out["unmix"][i, j] = un / (mi + un)
            out["bflux"][i, j] = back / (back + fwd)
            # expectation over realisations of this spectrum: |c_k|^2 -> S_k (the factor 2 cancels in the shares)
            with np.errstate(divide="ignore", invalid="ignore"):
                rep = np.where(np.abs(KPAR) > 0, (KZ / KPAR) * poisson_flux(KPAR, t, m, s) * S, 0.0)
            lam = 0.5 * (s * KPAR * t) ** 2
            Wm = np.where(lam > 0, S * np.exp(-lam + m * np.log(np.where(lam > 0, lam, 1.0)) - gammaln(m + 1)), 0.0)
            be = -float((wk * np.where(rep < 0, rep, 0.0)).sum())
            fe = float((wk * np.where(rep > 0, rep, 0.0)).sum())
            ue = float((wk * Wm * (rep < 0)).sum())
            me = float((wk * Wm * (rep > 0)).sum())
            out["bflux_exp"][i, j] = be / (be + fe)
            out["unmix_exp"][i, j] = ue / (ue + me)
        print(f"    tau = {tau:4.2f}: " + "; ".join(
            f"m={m}: unmixing {out['unmix'][i, j]:.3f} (expectation {out['unmix_exp'][i, j]:.3f}), backward flux "
            f"{out['bflux'][i, j]:.3f} ({out['bflux_exp'][i, j]:.3f})" for j, m in enumerate(m_diag)), flush=True)
    rb = bcfg_ref = None  # noqa: F841 (placeholders for readability below)
    del rb, bcfg_ref
    chk("box: hermite_flux = (k_z/k_par) x field-aligned flux on every mode and m (d06 T2)", worst_id < 1e-12,
        f"max relative difference {worst_id:.1e}")
    chk("box: field-aligned flux >= 0 on every mode (pure phase mixing)", worst_neg > -1e-13, f"min/max {worst_neg:.1e}")
    return dict(tau=np.array(taus), m=np.array(m_diag), opposite_weight=opp_w, k_cut=kcut, alpha_deg=alpha,
                sigma_par=spar, **out)


# ------------------------------------------------------------------------------------------------------------------
# computation
# ------------------------------------------------------------------------------------------------------------------
def compute(cfg: dict, tag: str | None, chash: str) -> dict:
    """Compute and check every number of the figure; write the cache; return the metadata."""
    t0 = time.time()
    start = time.strftime("%Y%m%d_%H%M%S")
    chk = Checks()
    ref = cfg["checks"]
    s = float(cfg["s"])
    key: dict[str, Any] = {}

    print("Gaussian ensemble: checks against d06 (T3, T4)")
    worst, worst_cf = 0.0, 0.0
    for a_deg, tau, m, val in ref["share_table"]:
        B, F, T = ensemble_fluxes(math.radians(a_deg), float(tau), int(m))
        worst = max(worst, abs(B / (F + B) - val))
        true = 2.0 * T / tau ** 2                                  # undo the common factor tau^2/2
        worst_cf = max(worst_cf, abs(true - true_flux_closed(float(tau), int(m)) / tau) / true)
    chk(f"d06 T3: backward share B/(F+B), {len(ref['share_table'])} entries (alpha = 14.04..63.43 deg, tau = 0.5, 2, 8, "
        f"m = 1, 4, 16)", worst < ref["share_tol"], f"max |diff| {worst:.1e}")
    chk("d06 T4: true net flux = closed form (der-dynamics eq. 2.6)", worst_cf < ref["closed_form_tol"],
        f"max relative difference {worst_cf:.1e}")
    late = ref["late"]
    vals = [backward_share(math.radians(late["alpha_deg"]), float(tau), int(late["m"])) for tau in late["tau"]]
    chk(f"d06 T3c: late-time share at alpha = {late['alpha_deg']} deg, m = {late['m']}, tau = {late['tau']}",
        max(abs(x - y) for x, y in zip(vals, late["share"])) < ref["share_tol"], ", ".join(f"{x:.4f}" for x in vals))
    key["late_share_alpha45_m4"] = dict(zip([str(x) for x in late["tau"]], vals))

    print("Gaussian ensemble: curves")
    alphas = [float(a) for a in cfg["alpha_deg"]]
    ms = [int(m) for m in cfg["m"]]
    taus = np.logspace(math.log10(cfg["tau_min"]), math.log10(cfg["tau_max"]), int(cfg["n_tau"]))
    share = np.zeros((len(alphas), len(ms), taus.size))
    for i, a in enumerate(alphas):
        for j, m in enumerate(ms):
            share[i, j] = [backward_share(math.radians(a), float(tau), m) for tau in taus]
            for tq in (2.0, 8.0, 100.0):
                key[f"share_alpha{a:g}_m{m}_tau{tq:g}"] = backward_share(math.radians(a), tq, m)
    for i, a in enumerate(alphas):
        print(f"    alpha = {a:g} deg: share at tau = {taus[-1]:g}: " +
              ", ".join(f"m={m}: {share[i, j, -1]:.4f}" for j, m in enumerate(ms)))

    print("GANDALF diagnostics on the exact tilted-field solution (d06 T2)")
    box = box_diagnostics(cfg["box"], s, chk)
    rb = ref["box"]
    chk("d06 T2: weight on modes with k_z k_par < 0 and the spectral cut",
        abs(box["opposite_weight"] - rb["opposite_weight"]) < rb["tol"] and abs(box["k_cut"] - rb["k_cut"]) < 0.05,
        f"{box['opposite_weight']:.4f}, {box['k_cut']:.3f}")
    for tq, un_ref, bf_ref in zip(rb["tau"], rb["unmixing_share"], rb["backward_flux_share"]):
        i = int(np.argmin(np.abs(box["tau"] - tq)))
        idx = [int(np.where(box["m"] == m)[0][0]) for m in rb["m"]]
        un = box["unmix"][i, idx]
        bf = box["bflux"][i, idx]
        chk(f"d06 T2 at tau = {tq}: unmixing and backward-flux shares at m = {rb['m']}",
            abs(box["tau"][i] - tq) < 1e-12 and np.abs(un - un_ref).max() < rb["tol"]
            and np.abs(bf - bf_ref).max() < rb["tol"],
            "unmixing " + ", ".join(f"{x:.3f}" for x in un) + "; backward flux " + ", ".join(f"{x:.3f}" for x in bf))
    a_box = box["alpha_deg"]
    cont = np.array([[backward_share(math.radians(a_box), float(tau), int(m)) for m in box["m"]]
                     for tau in box["tau"]])
    sel = cont > 0.02
    key["box_alpha_deg"] = a_box
    key["box_unmixing_share_max"] = float(box["unmix"].max())
    key["box_backward_flux_share_max"] = float(box["bflux"].max())
    key["box_expectation_over_continuum_range"] = [float((box["bflux_exp"][sel] / cont[sel]).min()),
                                                   float((box["bflux_exp"][sel] / cont[sel]).max())]
    key["box_realisation_over_continuum_range"] = [float((box["bflux"][sel] / cont[sel]).min()),
                                                   float((box["bflux"][sel] / cont[sel]).max())]
    print(f"    backward-flux share where the continuum value exceeds 0.02: expectation/continuum in "
          f"{key['box_expectation_over_continuum_range'][0]:.3f}..{key['box_expectation_over_continuum_range'][1]:.3f}"
          f", realisation/continuum in {key['box_realisation_over_continuum_range'][0]:.3f}.."
          f"{key['box_realisation_over_continuum_range'][1]:.3f}")
    arrays = dict(tau=taus, alpha_deg=np.array(alphas), m=np.array(ms), share=share, box_tau=box["tau"],
                  box_m=box["m"], box_bflux=box["bflux"], box_unmix=box["unmix"], box_bflux_exp=box["bflux_exp"],
                  box_unmix_exp=box["unmix_exp"], box_continuum=cont, box_alpha_deg=np.array(a_box),
                  box_opposite_weight=np.array(box["opposite_weight"]), box_k_cut=np.array(box["k_cut"]))
    wall = time.time() - t0
    label = f"fig-{NAME}"
    meta = dict(figure=NAME, label=label, run_id=(f"{tag}_{label}_{start}" if tag else f"{label}_{start}"),
                start_local=start, config_file=os.path.relpath(CONFIG, os.path.dirname(os.path.dirname(PAPER))),
                config=cfg, config_hash=chash, environment=environment(), wall_s=round(wall, 2), gates=GATES,
                checks=chk.rows, all_checks_pass=chk.all_passed, key_numbers=key)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_paths()[0], **arrays)
    write_meta(meta)
    print(f"computed in {wall:.1f} s; {sum(r['passed'] for r in chk.rows)}/{len(chk.rows)} checks passed")
    return meta


# ------------------------------------------------------------------------------------------------------------------
# plotting (from the cache only)
# ------------------------------------------------------------------------------------------------------------------
def setup_style(style: dict, usetex: bool):
    """JPP figure style: 10 pt serif text, LaTeX rendering, inward ticks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fs = style["font_size_pt"]
    plt.rcParams.update({
        "text.usetex": usetex, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm", "font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs,
        "ytick.labelsize": fs, "legend.fontsize": style["legend_font_size_pt"], "axes.linewidth": 0.6,
        "lines.linewidth": 1.2, "lines.markersize": 4.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "legend.frameon": False,
        "legend.handlelength": 1.8, "savefig.pad_inches": 0.02,
        "text.latex.preamble": r"\usepackage{amsmath}"})
    return plt


def code(name: str, usetex: bool) -> str:
    """A GANDALF function name in typewriter type."""
    return r"\texttt{" + name.replace("_", r"\_") + "}" if usetex else name


def plot(style: dict) -> str:
    """Draw the figure from the cached numbers; returns the rendering mode used."""
    d = np.load(cache_paths()[0])
    pal = style["palette"]
    black = style["reference_color"]
    rendering = "usetex"
    for usetex in ((True, False) if style["usetex"] else (False,)):
        plt = setup_style(style, usetex)
        fig, axes = plt.subplots(1, 3, figsize=(style["double_width_in"], 2.45), sharey=True,
                                 gridspec_kw=dict(wspace=0.08))
        ia45 = int(np.argmin(np.abs(d["alpha_deg"] - float(d["box_alpha_deg"]))))
        curves, marks = [], []
        for k, ax in enumerate(axes):
            m = int(d["m"][k])
            ax.axhline(0.5, color=black, lw=0.6, ls=(0, (1, 2)), zorder=1)
            for i, a in enumerate(d["alpha_deg"]):
                h = ax.plot(d["tau"], d["share"][i, k], color=pal[i], lw=1.2, zorder=2,
                            label=rf"$\alpha = {a:.4g}^\circ$")
                if k == 0:
                    curves += h
            jb = int(np.where(d["box_m"] == m)[0][0])
            h1 = ax.plot(d["box_tau"], d["box_bflux"][:, jb], ls="none", marker="o", ms=4.0, color=pal[ia45],
                         mec=pal[ia45], zorder=4, label=r"backward-flux share, " + code("hermite_flux", usetex))
            h2 = ax.plot(d["box_tau"], d["box_unmix"][:, jb], ls="none", marker="s", ms=4.0, mfc="white",
                         mec=pal[ia45], mew=0.9, zorder=3,
                         label=r"energy share, " + code("phase_unmixing_energy", usetex))
            if k == 0:
                marks = h1 + h2
            ax.set_xscale("log")
            ax.set_xlim(d["tau"][0], d["tau"][-1])
            ax.set_ylim(0.0, 0.55)
            ax.set_xlabel(r"$\tau = s\sigma_\parallel t$")
            ax.text(0.04, 0.95, rf"$m = {m}$", transform=ax.transAxes, ha="left", va="top")
            ax.text(0.0, 1.025, f"({'abc'[k]})", transform=ax.transAxes, ha="left", va="bottom")
        axes[0].set_ylabel(r"$B^z_m/(F^z_m + B^z_m)$")
        axes[0].legend(handles=curves, loc="lower right", handlelength=1.6)
        axes[1].legend(handles=marks, loc="lower right", handlelength=1.0,
                       title=rf"GANDALF, $\alpha = {float(d['box_alpha_deg']):.4g}^\circ$", title_fontsize=8)
        os.makedirs(FIG_DIR, exist_ok=True)
        try:
            fig.savefig(os.path.join(FIG_DIR, f"{NAME}.pdf"), bbox_inches="tight")
            fig.savefig(os.path.join(FIG_DIR, f"{NAME}.png"), dpi=style["png_dpi"], bbox_inches="tight")
            plt.close(fig)
            rendering = "usetex" if usetex else "mathtext"
            break
        except (RuntimeError, OSError) as exc:      # LaTeX unavailable: fall back to mathtext
            print(f"LaTeX rendering failed ({exc}); falling back to mathtext")
            plt.close(fig)
    return rendering


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recompute", action="store_true", help="recompute even if the cache matches the config")
    ap.add_argument("--tag", default=None, help="prefix of the run ID stored with the cache")
    args = ap.parse_args()
    cfg, style = load_config()
    chash = config_hash(cfg)
    meta = read_meta()
    if args.recompute or meta is None or meta.get("config_hash") != chash:
        meta = compute(cfg, args.tag, chash)
    else:
        print(f"using cached numbers from {meta['run_id']} ({cache_paths()[0]})")
    if not meta["all_checks_pass"]:
        print("FAILED CHECKS:", *[r["name"] for r in meta["checks"] if not r["passed"]], sep="\n  ")
        return 1
    meta["rendering"] = plot(style)
    write_meta(meta)
    print(f"wrote {os.path.join(FIG_DIR, NAME)}.pdf/.png ({meta['rendering']}); run {meta['run_id']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
