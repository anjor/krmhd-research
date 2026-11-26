#!/usr/bin/env python
"""Block spreads of the model-agreement numbers of section 7, and the sink time against the Alfvenic energy budget.

Each run's accepted window (configs/h1_acceptance.yaml, as in h1_pilot.py) is cut into the four equal blocks of the
steady-state block check (h1lib.block_averages); the whole window is averaged as h1_pilot.py does (h1lib.time_average).  Every quoted agreement between data and model (the table CLAIMS) is
evaluated on the whole window, which reproduces the number in the text, and on each block average:
  * "fixed": the block's data against the model of the whole window (the scatter of the data alone);
  * "own T": the block's data against the model with the block's own 1/T, the median of its D_perp(m)/W(m) over
    2 <= m <= 16, where the claim involves the two-channel model (T is measured from the same data, so part of the
    scatter cancels in the ratio).
The spread is the range of the block values.  Ratios of two runs (M = 32 against 64) carry instead the standard error
of the ratio from the block standard errors of the two runs.

For every run of tables 2 and 3 it also gives T, tau_nl, c_T = T/tau_nl, the Alfvenic energy-containing time E_A/eps_A
(window means of E_A and of the measured Alfvenic injection rate), (c_T^2/2) beta_i/chi^2 and the measured m_*, each
with the range over the blocks.

Reads <run>/<run>.npz (ts__t, ts__acc_sum_dt, ts__pWm, ts__Pi_fa, ts__E, ts__kperp_out, ts__u_out, ts__chi_out,
ts__acc_inj_g, ts__acc_inj_A, ts__acc_loss_eta, ts__acc_loss_nu, ts__kF_z2, ts__tiltF2), <run>.json (config),
<run>/fieldlines_snap_*.npz (kappa_par, F__E_par, g0__E_par, kappa1_m_par), <run>/h1_analysis.json (h1_pilot.py) and
data/h1_age_model.json (h1_age_model.py).  Writes data/h1_block_spreads.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_block_spreads.py
"""
from __future__ import annotations

import json
import math
import sys
from functools import cached_property
from pathlib import Path
from typing import Callable

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import p1lib as P  # noqa: E402
import h1lib as H  # noqa: E402
from h1_pilot import T_RANGE, formula, two_channel  # noqa: E402

N_BLOCKS = 4
RUNS = dict(  # label: run ID (tables 2 and 3)
    b1=      "P1_forced-beta1_20261015_122613",
    b03=     "P1_forced-beta0p3_20261016_145938",
    b01=     "P1_forced-beta0p1dt_20261016_234138",
    b1_M64=  "P1_forced-beta1-m64_20261018_144645",
    b03_M64= "P1_forced-beta0p3-m64_20261018_144656",
    b01_M64= "P1_forced-beta0p1-m64_20261018_144649",
    b03_N96= "P1_forced-beta0p3-n96_20261019_200712",
    b03_N128="P1_forced-beta0p3-n128_20261018_184515",
    b03_N256="P1_forced-beta0p3-n256_20261031_115113",
    b1_N128= "P1_forced-beta1-n128_20261018_184509",
    b1_mf1=  "P1_forced-beta1-mf1_20261019_201321",
    b005=    "P1_forced-beta0p05-m128_20261110_110114",
    b03_M192="P1_forced-beta0p3-m192_20261110_110132",
    chi075=  "P1_forced-beta0p05-chi0p7-m128_20261110_115435",
    chi056=  "P1_forced-beta0p05-chi0p5-m256_20261110_115438",
    chi040=  "P1_forced-beta0p05-chi0p35-m512-eta33_20261111_202716",
)
SCAN = ("b005", "b01_M64", "b03_M192", "chi075", "chi056", "chi040")


class View:
    """Averages of one run over one interval (the whole window or a block): W(m), Pi_{m+1/2}, D_perp(m), D_nu(m)
    and the injection rate eps, sampled Alfvenic quantities, and the derived sink rate."""

    def __init__(self, run: "Run", d: dict, b: int | None) -> None:
        i = 0 if b is None else b
        self.run, self.b = run, b
        self.eps = float(d["eps"][i])
        self.W, self.Pi = d["W"][i], d["Pi"][i]
        self.Dp, self.Dn = d["Dp"][i], d["Dn"][i]
        self.q = self.Dp / self.W
        m = np.arange(len(self.W))
        self.inv_T = float(np.median(self.q[(m >= T_RANGE[0]) & (m <= min(T_RANGE[1], run.M // 2))]))
        self.T = 1.0 / self.inv_T
        self.tau_nl = 1.0 / (float(d["kperp_out"][i]) * float(d["u_out"][i]))
        self.chi = float(d["chi_out"][i])               # mean of the sampled chi_out, as h1_pilot.py averages it
        self.E_A, self.eps_A = float(d["E"][i]), float(d["epsA"][i])
        self.nu_share = float(self.Dn.sum() / (self.Dn.sum() + self.Dp.sum()))

    def model(self, own_T: bool, name: str = "traced") -> dict:
        """Two-channel model per unit injection rate, with the T of the whole window or of this view."""
        return self.run.model(self.T if own_T else self.run.full.T, name)

    def ratio_W(self, own_T: bool, name: str = "traced") -> np.ndarray:
        return (self.W / self.eps) / self.model(own_T, name)["W"]

    def ratio_Pi(self, own_T: bool, name: str = "traced") -> np.ndarray:
        return (self.Pi / self.eps) / self.model(own_T, name)["Pi"]


class Run:
    def __init__(self, label: str, run_id: str, acc: dict) -> None:
        rd = P.DATA / run_id
        self.label, self.run_id = label, run_id
        meta = json.load(open(rd / f"{run_id}.json"))
        cfg = meta["config"]
        v, st = cfg["velocity"], cfg["stepper"]
        self.M, self.Lam, self.beta = int(v["M"]), P.parse_number(v["Lambda"]), P.parse_number(v["beta_i"])
        self.s = math.sqrt(self.beta)
        self.sl = P.make_slices(self.M, self.Lam, v["closure"])
        self.nu, self.hn = P.parse_number(st["nu"]), int(st["hyper_n"])
        self.mf = int(cfg["forcing"]["density"].get("moment", 0))
        self.k1 = 2 * math.pi / float(cfg["box"]["L"])
        self.N = int(cfg["box"]["N"])
        self.an = json.load(open(rd / "h1_analysis.json"))
        keys = ("t", "acc_sum_dt", "pWm", "Pi_fa", "E", "kperp_out", "u_out", "chi_out", "kF_z2", "tiltF2", "acc_inj_g",
                "acc_inj_A", "acc_loss_eta", "acc_loss_nu")
        with np.load(rd / f"{run_id}.npz") as z:
            ts = {k: z[f"ts__{k}"] for k in keys}             # not ts__C_slice: (M + 1)^2 per sample
        run_acc = (acc.get("runs") or {}).get(run_id, {})
        width = float(run_acc.get("window", cfg["gates"]["steady_window"]))
        t = ts["t"]
        win = H.window_slice(t, width) if t[-1] - t[0] >= width else np.ones_like(t, dtype=bool)
        first = int(np.where(win)[0][0])
        sampled = dict(W=ts["pWm"], Pi=ts["Pi_fa"], E=ts["E"], kperp_out=ts["kperp_out"], u_out=ts["u_out"],
                       chi_out=ts["chi_out"], kF_z2=ts["kF_z2"], tiltF2=ts["tiltF2"])
        accum = dict(eps=ts["acc_inj_g"], epsA=ts["acc_inj_A"], Dp=ts["acc_loss_eta"], Dn=ts["acc_loss_nu"])
        fd = H.block_averages(t, ts["acc_sum_dt"], first, 1, sampled, accum)
        for k, x in sampled.items():                     # the whole window as h1_pilot.py averages it (trapezoidal)
            fd[k] = H.time_average(x, ts["acc_sum_dt"], first)[None]
        self.full = View(self, fd, None)
        bl = H.block_averages(t, ts["acc_sum_dt"], first, N_BLOCKS, sampled, accum)
        self.blocks = [View(self, bl, b) for b in range(N_BLOCKS)]
        self.window = [float(t[first]), float(t[-1])]
        self.kF_z, self.sig_tilt = math.sqrt(float(fd["kF_z2"][0])), math.sqrt(float(fd["tiltF2"][0]))
        zs = [np.load(f) for f in sorted(rd.glob("fieldlines_snap_*.npz"))]
        self.kap = zs[0]["kappa_par"]
        self.PF = np.mean([z["F__E_par"] for z in zs], axis=0)
        self.Pg0 = np.mean([z["g0__E_par"] / z["g0__E_par"].sum() for z in zs], axis=0)
        self.k1m = np.mean([z["kappa1_m_par"] for z in zs], axis=0)
        self.kappa1 = float(np.sum(self.kap * self.PF) / self.PF.sum())
        self.c_m = np.array(self.an["no_echo"]["coefficient_c_m"], dtype=float)
        self._models: dict = {}

    def model(self, T: float, name: str) -> dict:
        key = (round(T, 12), name)
        if key not in self._models:
            if name == "traced":
                k, w = self.kap, self.PF
            elif name == "g0":
                k, w = self.kap, self.Pg0
            elif name == "single":
                k, w = np.array([self.kF_z]), np.array([1.0])
            elif name == "gauss":
                xi = np.linspace(-4, 4, 81)
                k, w = np.abs(self.kF_z + self.sig_tilt * xi), np.exp(-xi ** 2 / 2)
            else:
                raise KeyError(name)
            self._models[key] = two_channel(self.sl, self.s, k, w, T, self.nu, self.hn, 1.0, self.mf)
        return self._models[key]

    def views(self) -> list[View]:
        return [self.full] + self.blocks

    @cached_property
    def fit_m_max(self) -> int:
        return int(H.fit_decay_window(self.full.W / self.full.eps, self.M)["m_max"])


# ------------------------------------------------------------------------------------------------------------------
# claims: (label, where, text, evaluate(runs, view index or None, own_T) -> value); views: None = whole window
# ------------------------------------------------------------------------------------------------------------------
def rng(x: np.ndarray) -> list[float]:
    return [float(np.nanmin(x)), float(np.nanmax(x))]


def maxdev(x: np.ndarray) -> float:
    return float(np.nanmax(np.abs(x - 1.0)))


def V(r: Run, b: int | None) -> View:
    return r.full if b is None else r.blocks[b]


def collapse(R: dict, b: int | None, against: str) -> float:
    """rms relative residual of D_perp/W about a joint fit c0 + c1 ln(x), x = age or m (h1_beta_scan.py)."""
    rows = []
    for lab in ("b1_M64", "b03_M64", "b01_M64", "b1", "b03", "b01"):
        r = R[lab]
        q = V(r, b).q
        for m in range(1, r.M // 2 + 1):
            rows.append((math.sqrt(2 * m) / (r.s * r.kappa1), q[m], m))
    rows = np.array(rows)
    x = np.log(rows[:, 0] if against == "age" else rows[:, 2])
    A = np.vstack([np.ones(len(rows)), x]).T
    c, *_ = np.linalg.lstsq(A, rows[:, 1], rcond=None)
    return float(np.sqrt(np.mean(((rows[:, 1] - A @ c) / rows[:, 1]) ** 2)))


def ell_over_skT(r: Run, b: int | None, m_hi: int | None = None) -> float:
    """Fitted decay length over 2 <= m <= m_hi (default M/2) over s kappa_1 T of the same view."""
    v = V(r, b)
    ell = H.fit_decay(v.W / v.eps, 2, m_hi if m_hi is not None else r.M // 2)
    return ell / (r.s * r.kappa1 * v.T)


def mstar_model_over_fit(r: Run, b: int | None) -> float:
    v = V(r, b)
    ell = H.fit_decay(v.W / v.eps, 2, r.fit_m_max)
    return (r.s * r.kappa1 * v.T) ** 2 / ell ** 2


def skT_fit_range_over_ell(r: Run, b: int | None) -> float:
    """s kappa_1 / <D_perp/W> (rate averaged uniformly in sqrt(2m) over the windowed fit range) over the fitted ell."""
    v = V(r, b)
    m = np.arange(r.M + 1)
    x = np.sqrt(2.0 * m[2:r.fit_m_max + 1])
    rate = float(np.trapezoid(v.q[2:r.fit_m_max + 1], x) / (x[-1] - x[0]))
    return (r.s * r.kappa1 / rate) / H.fit_decay(v.W / v.eps, 2, r.fit_m_max)


def noecho(r: Run, b: int | None) -> np.ndarray:
    v = V(r, b)
    with np.errstate(divide="ignore", invalid="ignore"):
        return v.Pi / (r.s * r.c_m * r.k1m[:-1] * v.W[:-1])


def age_model(R: dict, AGE: dict, lab: str, b: int | None, key: str) -> np.ndarray:
    """Data of the view against the age-dependent model of h1_age_model.py (fixed: fitted on the whole window)."""
    r, a = R[lab], AGE["runs"][R[lab].run_id]
    m = np.array(a["m"])
    full_W = r.full.W[m] / r.full.eps
    v = V(r, b)
    if key == "W":
        return (v.W[m] / v.eps) / (full_W / np.array(a["W_over_age"]))
    return v.q[m] / np.array(a["Dperp_over_W_model"])


def sink_agreement(R: dict, b: int | None) -> float:
    """Largest (max - min)/mean over the chi >= 0.56 runs of D_perp/W x tau_nl at ages 12..30 tau_nl."""
    ages = np.arange(12.0, 30.01, 2.0)
    rows = []
    for lab in ("b005", "chi075", "chi056"):
        r, v = R[lab], V(R[lab], b)
        m = np.arange(1, r.M // 2 + 1)
        age = np.sqrt(2.0 * m) / (r.s * r.kappa1) / v.tau_nl
        rows.append(np.interp(ages, age, v.q[m] * v.tau_nl, left=np.nan, right=np.nan))
    X = np.array(rows)
    return float(np.nanmax((np.nanmax(X, 0) - np.nanmin(X, 0)) / np.nanmean(X, 0)))


def build_claims(AGE: dict) -> list[tuple[str, str, str, bool, Callable]]:
    """(label, where, quoted text, uses the two-channel model, f(R, b, own_T))."""
    C = []

    def add(label: str, where: str, text: str, uses_model: bool, f: Callable) -> None:
        C.append((label, where, text, uses_model, f))

    add("W_pilot", "07:80, 01:13, tab 1", "W within 8% for 1 <= m <= 30 (beta 1, M 32)", True,
        lambda R, b, o: maxdev(V(R["b1"], b).ratio_W(o)[1:31]))
    add("Pi_pilot_24", "07:80, 01:13, tab 1", "field-aligned flux within 3-7% for m <= 24", True,
        lambda R, b, o: rng(V(R["b1"], b).ratio_Pi(o)[0:25]))
    add("Pi_pilot_16", "07:80", "for 1 <= m <= 16 the flux lies 4-7% above the model", True,
        lambda R, b, o: rng(V(R["b1"], b).ratio_Pi(o)[1:17]))
    add("q_pilot_range", "07:71", "D_perp/W between 0.37 and 0.47 for 0 <= m <= 16", False,
        lambda R, b, o: rng(V(R["b1"], b).q[0:17]))
    add("q_pilot_rise", "07:71, 01:13, tab 1", "D_perp/W proportional to W within 20% (max/min over 1 <= m <= 16)", False,
        lambda R, b, o: float(V(R["b1"], b).q[1:17].max() / V(R["b1"], b).q[1:17].min()))
    add("kappa_eff_pilot", "07:80", "flux-implied kappa Pi/(s c_m W) over 3 <= m <= 23, units of k_1", False,
        lambda R, b, o: rng(V(R["b1"], b).Pi[3:24] / (R["b1"].s * R["b1"].c_m[3:24] * V(R["b1"], b).W[3:24])
                            / R["b1"].k1))
    add("noecho_pilot", "07:80", "Pi over the no-echo flux, 1 <= m <= 16", False,
        lambda R, b, o: rng(noecho(R["b1"], b)[1:17]))
    add("Pi_pilot_top", "07:80", "flux below the model by up to 12% for m >= 27 (min of Pi/model over 27..31)", True,
        lambda R, b, o: float(V(R["b1"], b).ratio_Pi(o)[27:32].min()))
    add("W_pilot_32", "07:80", "W 14% below the model at m = 32", True,
        lambda R, b, o: float(V(R["b1"], b).ratio_W(o)[32]))
    add("single_kappa", "07:80", "single kappa = k_1: model W 15-16% too high for 2 <= m <= 16 (model/W)", True,
        lambda R, b, o: rng(1.0 / V(R["b1"], b).ratio_W(o, "single")[2:17]))
    add("gauss_tilt", "07:80", "Gaussian tilt: W 0.72 to 1.22 times the model (1 <= m <= 32)", True,
        lambda R, b, o: rng(V(R["b1"], b).ratio_W(o, "gauss")[1:33]))
    add("g0_kappa", "07:80", "P(kappa) of g_0: W too low by up to 1.9 at m = 24 (W/model at 24)", True,
        lambda R, b, o: float(V(R["b1"], b).ratio_W(o, "g0")[24]))
    add("coll_pilot", "07:107", "hypercollisions remove 57-61% of the injected free energy (sum D_nu / eps)", False,
        lambda R, b, o: float(V(R["b1"], b).Dn.sum() / V(R["b1"], b).eps))
    for lab, txt in (("b1", "M 32"), ("b1_M64", "M 64")):
        add(f"W_beta1_{lab}", "07:155", f"beta 1: W within 7% of the model for 1 <= m <= M/2 ({txt})", True,
            lambda R, b, o, lab=lab: maxdev(V(R[lab], b).ratio_W(o)[1:R[lab].M // 2 + 1]))
    for lab, txt in (("b03", "21%, beta 0.3, M 32"), ("b01", "39%, beta 0.1, M 32"), ("b03_M64", "20%, beta 0.3, M 64"),
                     ("b01_M64", "42%, beta 0.1, M 64")):
        add(f"W_above_{lab}", "07:155", f"W above the model by up to {txt} (max of W/model - 1, 1 <= m <= M/2)", True,
            lambda R, b, o, lab=lab: float(V(R[lab], b).ratio_W(o)[1:R[lab].M // 2 + 1].max() - 1.0))
    add("collapse_age", "07:153", "rms scatter 4.5% of D_perp/W about the joint fit in ln(age)", False,
        lambda R, b, o: collapse(R, b, "age"))
    add("collapse_m", "07:153", "against 12.6% when plotted against m", False,
        lambda R, b, o: collapse(R, b, "m"))
    for lab in ("b1", "b03", "b01", "b1_M64", "b03_M64", "b01_M64"):
        add(f"ell_{lab}", "07:118", f"fitted ell_* over s kappa_1 T (2 <= m <= M/2), within 7% in every run ({lab})",
            False, lambda R, b, o, lab=lab: ell_over_skT(R[lab], b))
    add("formula_b01", "07:116", "beta 0.1, M 32: the formula (kappa = k_1, measured T) within about 10% up to m = 16",
        False, lambda R, b, o: rng((V(R["b01"], b).W[1:17] / V(R["b01"], b).eps)
                                   / formula(np.arange(1, 17), 1.0, R["b01"].s, R["b01"].kF_z, V(R["b01"], b).T)))
    add("W_mf1", "07:86", "g_1 forced: W within 3.5% of the model for 0 <= m <= 32", True,
        lambda R, b, o: maxdev(V(R["b1_mf1"], b).ratio_W(o)[0:33]))
    add("Pi_mf1", "07:86", "g_1 forced: flux 2-4% above the model for 1 <= m <= M/2", True,
        lambda R, b, o: rng(V(R["b1_mf1"], b).ratio_Pi(o)[1:R["b1_mf1"].M // 2 + 1]))
    for lab, txt in (("b1_mf1", "75%"), ("b1_M64", "66%")):
        add(f"Pi16_{lab}", "07:86", f"flux at m = 16 carries {txt} of the injection ({lab})", False,
            lambda R, b, o, lab=lab: float(V(R[lab], b).Pi[16] / V(R[lab], b).eps))
        add(f"coll_{lab}", "07:86", f"hypercollisions remove {'61%' if lab == 'b1_mf1' else '50%'} ({lab}; model "
            f"{'60%' if lab == 'b1_mf1' else '48%'})", True,
            lambda R, b, o, lab=lab: [float(V(R[lab], b).Dn.sum() / V(R[lab], b).eps),
                                      float(np.sum(2.0 * P.coll_rates(R[lab].M, R[lab].nu, R[lab].hn)
                                                   * V(R[lab], b).model(o)["W"]))])
    for lab in ("b1", "b03", "b01", "b1_M64", "b03_M64", "b01_M64"):
        add(f"share_{lab}", "07:174", "collisional share, measured minus model (within 0.06 at M 32, 0.02 at M 64)",
            True, lambda R, b, o, lab=lab: float(V(R[lab], b).nu_share - np.sum(
                2.0 * P.coll_rates(R[lab].M, R[lab].nu, R[lab].hn) * V(R[lab], b).model(o)["W"])))
    for lab in ("b03_N128", "b03_N256", "b1_N128"):
        add(f"W_{lab}", "07:185", "W within -9% and +13% of the model at 128^3 and 256^3 (1 <= m <= M/2)", True,
            lambda R, b, o, lab=lab: [float(x - 1.0) for x in rng(V(R[lab], b).ratio_W(o)[1:R[lab].M // 2 + 1])])
        add(f"Pi_{lab}", "07:185", "flux 3-15% above the model (0 <= m <= M/2)", True,
            lambda R, b, o, lab=lab: rng(V(R[lab], b).ratio_Pi(o)[0:R[lab].M // 2 + 1]))
        add(f"share_{lab}", "07:185", "collisional shares within 0.02 of the model", True,
            lambda R, b, o, lab=lab: float(V(R[lab], b).nu_share - np.sum(
                2.0 * P.coll_rates(R[lab].M, R[lab].nu, R[lab].hn) * V(R[lab], b).model(o)["W"])))
    for lab in ("b03_M64", "b03_N96", "b03_N128", "b03_N256", "b1_M64", "b1_N128"):
        add(f"ell_{lab}_res", "07:185", "decay length over s kappa_1 T: within 5%, except 14% and 17% faster at "
            "beta 0.3, 128^3 and 256^3 (ratio < 1 is faster)", False, lambda R, b, o, lab=lab: ell_over_skT(R[lab], b))
    for lab in SCAN:
        add(f"mstar_{lab}", "07:147, 07:149", "(s kappa_1 T)^2/2 over the fitted m_* (within 12% at beta <= 0.1 and "
            "chi >= 0.56; 1.3 at beta 0.3; 2.8 at chi 0.40)", False, lambda R, b, o, lab=lab: mstar_model_over_fit(R[lab], b))
        add(f"noecho_{lab}", "07:84", "flux over the no-echo flux, 8 <= m <= M/2 (old text 0.85-1.00)", False,
            lambda R, b, o, lab=lab: rng(noecho(R[lab], b)[8:R[lab].M // 2 + 1]))
        add(f"skTfit_{lab}", "07:170", "s kappa_1 T with the rate averaged over the fit range, over the fitted ell_* "
            "(within 9% at chi 0.40 and 5% in the others)", False, lambda R, b, o, lab=lab: skT_fit_range_over_ell(R[lab], b))
    add("Pi_chi040_256", "07:84", "chi 0.40: flux 37% below the uniform-sink model at m = 256", True,
        lambda R, b, o: float(V(R["chi040"], b).ratio_Pi(False)[256]))
    add("sink_agree", "07:170", "sinks of the chi >= 0.56 runs agree within about 10% for ages >= 12 tau_nl", False,
        lambda R, b, o: sink_agreement(R, b))
    for lab, lo, txt in (("b03_M64", 2, "beta 0.3, m >= 2"), ("b1_M64", 4, "beta 1, m >= 4")):
        add(f"age_q_{lab}", "07:168", f"age model: D_perp/W within 5% ({txt}, to M/2)", False,
            lambda R, b, o, lab=lab, lo=lo: maxdev(age_model(R, AGE, lab, b, "q")[lo - 1:]))
    for lab in ("b01_M64", "b03_M64", "b1_M64"):
        add(f"age_W8_{lab}", "07:168", "age model: W within 7% for m >= 8 in all three runs", False,
            lambda R, b, o, lab=lab: maxdev(age_model(R, AGE, lab, b, "W")[7:]))
        add(f"age_Wlow_{lab}", "07:168", "age model above the measured W: 7% at m = 1 (fit run), 8-14% for m <= 3 "
            "(others); model/W - 1 over m = 1..3", False,
            lambda R, b, o, lab=lab: [float(x) for x in (1.0 / age_model(R, AGE, lab, b, "W")[0:3] - 1.0)])
    return C


def m_ratio(R: dict, lo: str, hi: str) -> dict:
    """W(m)/eps of the M = 64 run over that of the M = 32 run for 1 <= m <= 10, with the standard error from the
    block standard errors of both, and 1/T of both with theirs."""
    def mean_se(r: Run) -> tuple[np.ndarray, np.ndarray]:
        X = np.array([v.W[:11] / v.eps for v in r.blocks])
        return r.full.W[:11] / r.full.eps, X.std(0, ddof=1) / math.sqrt(N_BLOCKS) / X.mean(0)
    (a, sa), (c, sc) = mean_se(R[lo]), mean_se(R[hi])
    ratio = c / a
    se = ratio * np.sqrt(sa ** 2 + sc ** 2)
    it = [np.array([v.inv_T for v in R[k].blocks]) for k in (lo, hi)]
    se_T = [float(x.std(ddof=1) / math.sqrt(N_BLOCKS) / x.mean()) for x in it]
    dT = R[hi].full.inv_T / R[lo].full.inv_T - 1.0
    return dict(W_ratio_m1_10=ratio[1:].tolist(), W_ratio_se_m1_10=se[1:].tolist(),
                max_dev=float(np.abs(ratio[1:] - 1).max()), se_at_max=float(se[1:][np.abs(ratio[1:] - 1).argmax()]),
                inv_T_change=float(dT), inv_T_change_se=float((1 + dT) * math.hypot(*se_T)))


def run_table(r: Run) -> dict:
    """Task 2: T, tau_nl, c_T, E_A/eps_A, (c_T^2/2) beta/chi^2 and m_* (whole window and range over blocks)."""
    def vals(v: View) -> dict:
        cT = v.T / v.tau_nl
        return dict(T=v.T, tau_nl=v.tau_nl, c_T=cT, chi=v.chi, E_A=v.E_A, eps_A=v.eps_A, E_A_over_eps_A=v.E_A / v.eps_A,
                    T_over_EA_eps_A=v.T / (v.E_A / v.eps_A), cT2_beta_over_2chi2=cT ** 2 * r.beta / (2 * v.chi ** 2))
    full = vals(r.full)
    blocks = [vals(v) for v in r.blocks]
    out = dict(run_id=r.run_id, beta=r.beta, N=r.N, M=r.M, window=r.window, **full,
               blocks={k: rng(np.array([bv[k] for bv in blocks])) for k in full})
    win = H.fit_decay_window(r.full.W / r.full.eps, r.M)
    ell_b = [H.fit_decay(v.W / v.eps, 2, win["m_max"]) for v in r.blocks]
    out.update(m_star=win["ell"] ** 2 / 2, m_star_status=win["status"], m_star_fit_m_max=win["m_max"],
               m_star_blocks=rng(np.array(ell_b) ** 2 / 2),
               m_star_se=float(win["ell"] * np.std(ell_b, ddof=1) / math.sqrt(N_BLOCKS)),
               m_star_full_range_fit=H.fit_decay(r.full.W / r.full.eps, 2, r.M // 2) ** 2 / 2,
               m_star_model=(r.s * r.kappa1 * r.full.T) ** 2 / 2,
               cT_se=float(np.std([v.T / v.tau_nl for v in r.blocks], ddof=1) / math.sqrt(N_BLOCKS)))
    return out


def main() -> int:
    acc = P.load_config("h1_acceptance.yaml")
    R = {lab: Run(lab, rid, acc) for lab, rid in RUNS.items()}
    AGE = json.load(open(P.DATA / "h1_age_model.json"))
    out: dict = dict(n_blocks=N_BLOCKS, runs={lab: r.run_id for lab, r in R.items()})

    # reproduction checks: the whole-window views equal h1_pilot's averages
    worst = 0.0
    for r in R.values():
        an = r.an
        worst = max(worst, abs(r.full.inv_T / an["P_a"]["inv_T"] - 1), abs(r.full.tau_nl / an["outer"]["tau_nl"] - 1),
                    abs(r.full.chi / an["outer"]["chi_out"] - 1),
                    float(np.nanmax(np.abs(r.full.ratio_W(False)[:len(an["P_b"]["traced P(kappa) of the injected pattern"]["W_over_model_m"])]
                                    / np.array(an["P_b"]["traced P(kappa) of the injected pattern"]["W_over_model_m"]) - 1))))
    out["reproduction_max_rel_diff"] = worst
    print(f"whole-window views against h1_analysis.json (1/T, tau_nl, chi, W/model): max relative difference {worst:.1e}")

    claims = []
    for label, where, text, uses_model, f in build_claims(AGE):
        row = dict(label=label, where=where, text=text, whole=f(R, None, False),
                   blocks_fixed=[f(R, b, False) for b in range(N_BLOCKS)])
        if uses_model:
            row["blocks_own_T"] = [f(R, b, True) for b in range(N_BLOCKS)]
        claims.append(row)

        def fmt(x) -> str:
            return "[" + ", ".join(f"{y:.3f}" for y in x) + "]" if isinstance(x, list) else f"{x:.3f}"
        print(f"{label:22s} {where:18s} whole {fmt(row['whole'])}; blocks (fixed model) "
              + " ".join(fmt(x) for x in row["blocks_fixed"])
              + (("; own T " + " ".join(fmt(x) for x in row["blocks_own_T"])) if uses_model else "") + f"   | {text}")
    out["claims"] = claims
    out["M_doubling"] = {lo: m_ratio(R, lo, hi) for lo, hi in (("b1", "b1_M64"), ("b03", "b03_M64"), ("b01", "b01_M64"))}
    for k, v in out["M_doubling"].items():
        print(f"M 32 -> 64 ({k}): max |W64/W32 - 1| over 1 <= m <= 10 {v['max_dev']:.3f} +- {v['se_at_max']:.3f}; "
              f"1/T change {v['inv_T_change']:+.3f} +- {v['inv_T_change_se']:.3f}")
    out["run_table"] = {lab: run_table(r) for lab, r in R.items()}
    print("label      T      tau_nl  c_T (blocks)        E_A/eps_A  T/(E_A/eps_A)  cT^2 b/2chi^2   m_* (status, blocks)")
    for lab, d in out["run_table"].items():
        print(f"{lab:9s} {d['T']:.3f}  {d['tau_nl']:.4f}  {d['c_T']:5.2f} ({d['blocks']['c_T'][0]:.1f}-{d['blocks']['c_T'][1]:.1f}) "
              f"  {d['E_A_over_eps_A']:.3f} ({d['blocks']['E_A_over_eps_A'][0]:.2f}-{d['blocks']['E_A_over_eps_A'][1]:.2f})"
              f"  {d['T_over_EA_eps_A']:.2f}  {d['cT2_beta_over_2chi2']:8.1f}  {d['m_star']:.1f} ({d['m_star_status']}, "
              f"{d['m_star_blocks'][0]:.1f}-{d['m_star_blocks'][1]:.1f}); full-range fit {d['m_star_full_range_fit']:.1f}")
    for b, lo, hi in ((1.0, "b1_M64", "b1_N128"), (0.3, "b03_M64", "b03_N128")):
        a, c = out["run_table"][lo], out["run_table"][hi]
        r = c["c_T"] / a["c_T"]
        se = r * math.hypot(a["cT_se"] / a["c_T"], c["cT_se"] / c["c_T"])
        out[f"cT_128_over_64_beta{b:g}"] = dict(ratio=r, se=se, c_T_64=a["c_T"], c_T_128=c["c_T"])
        print(f"beta {b:g}: T/tau_nl {a['c_T']:.2f} +- {a['cT_se']:.2f} (64^3) -> {c['c_T']:.2f} +- {c['cT_se']:.2f} "
              f"(128^3), ratio {r:.3f} +- {se:.3f}")
    (P.DATA / "h1_block_spreads.json").write_text(json.dumps(P._jsonable(out), indent=1))
    print(f"wrote {P.DATA / 'h1_block_spreads.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
