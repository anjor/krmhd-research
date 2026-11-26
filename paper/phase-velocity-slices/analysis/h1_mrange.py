#!/usr/bin/env python
"""The m-ranges of the section-5 statistics under one rule (second referee read, items 1.10 and 2.5).

Rule.  A statistic that compares the data with the collision-free theory is taken over 2 <= m <= m_c, where m_c is
the largest m at which the hypercollisions remove free energy at no more than CRIT = 13% of the rate of the
perpendicular sink, 2 nu_m <= CRIT/T (the criterion the m_* scan already meets at the top of its fit ranges).  m = 0 is
forced and m = 1 has no hypercollisions and holds the freshest free energy, so the range starts at m = 2 as the fits
of T and ell_* already do.  With nu_m = nu (m/M)^n the crossing is m = M [r/(2 nu)]^(1/n): m_c depends on T only as
T^(1/6), so the rule changes m_c by 4.5% if CRIT changes by 30%.

For every run of tables 2 and 3 the script gives
  * the m at which 2 nu_m reaches CRIT/T and 1/T, with T the median of D_perp/W over 2 <= m <= 16 (tables 2 and 3);
  * m_c and T over 2 <= m <= m_c ("T_c"), solved together as a fixed point (m_c from T, T from m_c);
  * c_T and the decay length ell_* fitted over 2 <= m <= m_c, against the old ones (T over 2..16, ell_* over 2..M/2);
  * the key numbers of section 5, old range against new: D_perp/W (range and max/min), W/model and Pi/model (the
    two-channel model with the traced P(kappa); with T for the old range and with T_c for the new), the flux-implied
    kappa Pi/(s c_m W), and Pi/(s c_m kappa_1(m) W) with the same ratio of the uniform-sink model.
Every number is evaluated on the whole window and on the four blocks (data of the block against the model of the
whole window, as h1_block_spreads.py does).

Reads, through h1_block_spreads.Run: <run>/<run>.npz (ts__t, ts__acc_sum_dt, ts__pWm, ts__Pi_fa, ts__E, ts__kperp_out,
ts__u_out, ts__chi_out, ts__acc_inj_g, ts__acc_inj_A, ts__acc_loss_eta, ts__acc_loss_nu, ts__kF_z2, ts__tiltF2),
<run>.json (config), <run>/fieldlines_snap_*.npz (kappa_par, F__E_par, g0__E_par, kappa1_m_par) and
<run>/h1_analysis.json (no_echo.coefficient_c_m).  Writes data/h1_mrange.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_mrange.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))
import p1lib as P  # noqa: E402
import h1lib as H  # noqa: E402
from h1_block_spreads import N_BLOCKS, RUNS, Run, View  # noqa: E402
from h1_pilot import T_RANGE  # noqa: E402

CRIT = 0.13
M_LO = 2
TABLE2 = ("b1", "b03", "b01", "b1_M64", "b03_M64", "b01_M64", "b03_N96", "b03_N128", "b03_N256", "b1_N128", "b1_mf1")
TABLE3 = ("b005", "b01_M64", "b03_M192", "chi075", "chi056", "chi040")


def m_cross(M: int, nu: float, hn: int, rate: float) -> float:
    """The real m at which the free-energy removal rate of the hypercollisions, 2 nu (m/M)^n, equals `rate`."""
    return M * (rate / (2.0 * nu)) ** (1.0 / hn)


def median_rate(q: np.ndarray, lo: int, hi: int) -> float:
    """1/T: the median of D_perp(m)/W(m) = q over lo <= m <= hi."""
    m = np.arange(len(q))
    return float(np.median(q[(m >= lo) & (m <= hi)]))


def m_c_rule(q: np.ndarray, M: int, nu: float, hn: int) -> dict:
    """The rule's range 2 <= m <= m_c for one run: m_c = floor(m at which 2 nu_m = CRIT/T), at most M/2, with 1/T the
    median of q = D_perp/W over 2..m_c.  Solved by iteration from T over 2..16; m_c depends on T only as T^(1/6), so
    it converges in a few steps; a two-value cycle keeps the larger m_c (as h1lib.fit_decay_window does)."""
    inv_T = median_rate(q, *T_RANGE)
    path: list[int] = []
    for it in range(1, 30):
        m_c = min(int(math.floor(m_cross(M, nu, hn, CRIT * inv_T))), M // 2)
        if path and m_c == path[-1]:
            break
        if m_c in path:
            m_c = max(m_c, path[-1])
            inv_T = median_rate(q, M_LO, m_c)
            break
        path.append(m_c)
        inv_T = median_rate(q, M_LO, m_c)
    return dict(m_c=m_c, inv_T=inv_T, iterations=it, path=path)


def rng(x: np.ndarray) -> list[float]:
    return [float(np.nanmin(x)), float(np.nanmax(x))]


def maxdev(x: np.ndarray) -> float:
    return float(np.nanmax(np.abs(np.asarray(x) - 1.0)))


def stats(r: Run, v: View, lo: int, hi: int, T: float, q_lo: int | None = None, q_hi: int | None = None) -> dict:
    """Key numbers of section 5 for one view over lo <= m <= hi (D_perp/W over q_lo..q_hi if given); the model uses
    the whole-window data's T given here."""
    q_lo = lo if q_lo is None else q_lo
    q_hi = hi if q_hi is None else q_hi
    md = r.model(T, "traced")
    W, Pi = v.W / v.eps, v.Pi / v.eps
    sl = slice(lo, hi + 1)
    slp = slice(lo, min(hi, r.M - 1) + 1)                 # fluxes Pi_{m+1/2} exist for m < M
    q = v.q[q_lo:q_hi + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        noecho = v.Pi / (r.s * r.c_m * r.k1m[:-1] * v.W[:-1])
        model_noecho = md["Pi"] / (r.s * r.c_m * md["kappa_1"][:-1] * md["W"][:-1])
        kap = v.Pi / (r.s * r.c_m * v.W[:-1]) / r.k1
    return dict(q_range=rng(q), q_max_over_min=float(q.max() / q.min()),
                W_over_model_maxdev=maxdev(W[sl] / md["W"][sl]), W_over_model=rng(W[sl] / md["W"][sl]),
                Pi_over_model=rng(Pi[slp] / md["Pi"][slp]), kappa_eff=rng(kap[slp]),
                noecho=rng(noecho[slp]), model_noecho=rng(model_noecho[slp]),
                kappa1_m=rng(r.k1m[sl] / r.k1))


def blocks_of(f, r: Run) -> list:
    return [f(b) for b in r.blocks]


def run_rows(r: Run) -> dict:
    full = r.full
    inv_T_old = full.inv_T
    fp = m_c_rule(full.q, r.M, r.nu, r.hn)
    m_c, inv_T_c = fp["m_c"], fp["inv_T"]
    T_old, T_c = 1.0 / inv_T_old, 1.0 / inv_T_c
    ell_old = H.fit_decay(full.W / full.eps, M_LO, r.M // 2)
    ell_c = H.fit_decay(full.W / full.eps, M_LO, m_c)
    out = dict(run_id=r.run_id, beta=r.beta, N=r.N, M=r.M, nu=r.nu,
               m_at_013_over_T=m_cross(r.M, r.nu, r.hn, CRIT * inv_T_old), m_at_1_over_T=m_cross(r.M, r.nu, r.hn, inv_T_old),
               m_c=m_c, m_c_iterations=fp["iterations"], m_c_path=fp["path"],
               T_old=T_old, T_c=T_c, T_c_over_T_old=T_c / T_old,
               c_T_old=T_old / full.tau_nl, c_T_c=T_c / full.tau_nl,
               ell_old=ell_old, ell_c=ell_c, ell_c_over_ell_old=ell_c / ell_old,
               ell_old_over_skT_old=ell_old / (r.s * r.kappa1 * T_old),
               ell_c_over_skT_c=ell_c / (r.s * r.kappa1 * T_c),
               ell_c_over_skT_old=ell_c / (r.s * r.kappa1 * T_old),
               m_star_old=ell_old ** 2 / 2, m_star_c=ell_c ** 2 / 2,
               twonu_T_at_16=float(2.0 * P.coll_rates(r.M, r.nu, r.hn)[min(16, r.M)] * T_old),
               twonu_T_at_M2=float(2.0 * P.coll_rates(r.M, r.nu, r.hn)[r.M // 2] * T_old))
    # the model fitted the same way: the form W sqrt(2m) ~ exp(-sqrt(2m)/ell) is the large-m limit, and over a short
    # range from m = 2 the factor sqrt(2m)/c_m (0.94 at m = 2, 0.99 at m = 9) biases the fitted ell upwards; the
    # data's ell over the model's ell on the same range removes that bias
    ell_mod_old = H.fit_decay(r.model(T_old, "traced")["W"], M_LO, r.M // 2)
    ell_mod_c = H.fit_decay(r.model(T_c, "traced")["W"], M_LO, m_c)
    out.update(ell_model_old=ell_mod_old, ell_model_c=ell_mod_c, ell_old_over_model=ell_old / ell_mod_old,
               ell_c_over_model=ell_c / ell_mod_c)
    # block spread of the new T and ell_* (blocks' own data, each with the whole window's m_c)
    out["T_c_blocks"] = rng(np.array(blocks_of(lambda v: 1.0 / median_rate(v.q, M_LO, m_c), r)))
    out["T_old_blocks"] = rng(np.array(blocks_of(lambda v: v.T, r)))
    out["ell_c_blocks"] = rng(np.array(blocks_of(lambda v: H.fit_decay(v.W / v.eps, M_LO, m_c), r)))
    out["ell_old_blocks"] = rng(np.array(blocks_of(lambda v: H.fit_decay(v.W / v.eps, M_LO, r.M // 2), r)))
    # key numbers, old ranges of section 5 (pilot's ranges; 8..M/2 for the no-echo ratio of the other runs) and new
    old = dict(q=stats(r, full, 0, 16, T_old, 0, 16), q1=stats(r, full, 1, 16, T_old),
               W=stats(r, full, 1, min(30, r.M), T_old), W_M2=stats(r, full, 1, r.M // 2, T_old),
               Pi=stats(r, full, 0, min(24, r.M - 1), T_old), Pi_M2=stats(r, full, 0, r.M // 2, T_old),
               kap=stats(r, full, 3, min(23, r.M - 1), T_old), ne16=stats(r, full, 1, 16, T_old),
               ne8=stats(r, full, 8, r.M // 2, T_old))
    out["old"] = {k: v for k, v in old.items()}
    out["new"] = stats(r, full, M_LO, m_c, T_c)
    out["new_T_old"] = stats(r, full, M_LO, m_c, T_old)       # the new range with the old T, to separate the effects
    keys = ("q_range", "q_max_over_min", "W_over_model_maxdev", "Pi_over_model", "kappa_eff", "noecho", "model_noecho")
    nb = [stats(r, v, M_LO, m_c, T_c) for v in r.blocks]
    out["new_blocks"] = {k: (rng(np.array([b[k] for b in nb])) if not isinstance(nb[0][k], list)
                             else [float(min(b[k][0] for b in nb)), float(max(b[k][1] for b in nb))]) for k in keys}
    return out


def main() -> int:
    acc = P.load_config("h1_acceptance.yaml")
    R = {lab: Run(lab, rid, acc) for lab, rid in RUNS.items()}
    rows = {lab: run_rows(r) for lab, r in R.items()}
    print(f"rule: 2 <= m <= m_c, 2 nu_m <= {CRIT}/T at m_c, T the median of D_perp/W over the same range")
    print("label     M   m(0.13/T) m(1/T)  2nuT@16  m_c (it)  T_old  T_c   ratio   c_T old->new   "
          "ell_old ell_c  ell/skT old->new   m*_old m*_c")
    for lab, d in rows.items():
        print(f"{lab:9s} {d['M']:3d}  {d['m_at_013_over_T']:6.1f}  {d['m_at_1_over_T']:6.1f}  {d['twonu_T_at_16']:6.3f}  "
              f"{d['m_c']:4d} ({d['m_c_iterations']})  {d['T_old']:.3f} {d['T_c']:.3f} {d['T_c_over_T_old']:.3f}  "
              f"{d['c_T_old']:5.2f}->{d['c_T_c']:5.2f}   {d['ell_old']:6.2f} {d['ell_c']:6.2f}  "
              f"{d['ell_old_over_skT_old']:.3f}->{d['ell_c_over_skT_c']:.3f}   {d['m_star_old']:6.1f} {d['m_star_c']:6.1f}")
    print("\ndecay length of the data over that of the model fitted the same way: old range 2..M/2 -> new 2..m_c")
    for lab, d in rows.items():
        print(f"{lab:9s} ell data/model {d['ell_old_over_model']:.3f} -> {d['ell_c_over_model']:.3f}   (model ell "
              f"{d['ell_model_old']:.2f} -> {d['ell_model_c']:.2f}; data ell blocks old {d['ell_old_blocks'][0]:.2f}-"
              f"{d['ell_old_blocks'][1]:.2f}, new {d['ell_c_blocks'][0]:.2f}-{d['ell_c_blocks'][1]:.2f}; T blocks old "
              f"{d['T_old_blocks'][0]:.3f}-{d['T_old_blocks'][1]:.3f}, new {d['T_c_blocks'][0]:.3f}-{d['T_c_blocks'][1]:.3f})")
    # M = 32 against M = 64 at the same beta_i, on the same moments 2..m_c of the M = 32 run
    pairs = {}
    for lo, hi in (("b1", "b1_M64"), ("b03", "b03_M64"), ("b01", "b01_M64")):
        a, b, mc = R[lo], R[hi], rows[lo]["m_c"]
        pairs[lo] = dict(m_c=mc, T_M32=rows[lo]["T_c"], T_M64_same_range=1.0 / median_rate(b.full.q, M_LO, mc),
                         T_M32_2_16=rows[lo]["T_old"], T_M64_2_16=rows[hi]["T_old"], T_M64_own_m_c=rows[hi]["T_c"],
                         ell_M32=rows[lo]["ell_c"], ell_M64_same_range=H.fit_decay(b.full.W / b.full.eps, M_LO, mc),
                         ell_M32_2_16=rows[lo]["ell_old"], ell_M64_2_32=rows[hi]["ell_old"],
                         ell_M64_own_m_c=rows[hi]["ell_c"])
        d = pairs[lo]
        print(f"{lo:4s} vs {hi}: over 2..{mc}: T {d['T_M32']:.3f} (M 32) vs {d['T_M64_same_range']:.3f} (M 64), ratio "
              f"{d['T_M32'] / d['T_M64_same_range']:.3f}; over 2..16: {d['T_M32_2_16']:.3f} vs {d['T_M64_2_16']:.3f}, ratio "
              f"{d['T_M32_2_16'] / d['T_M64_2_16']:.3f}.  ell over 2..{mc}: {d['ell_M32']:.2f} vs {d['ell_M64_same_range']:.2f}; "
              f"old fits (2..M/2): {d['ell_M32_2_16']:.2f} vs {d['ell_M64_2_32']:.2f}")
    print("\nkey numbers, new range 2..m_c with T_c (blocks):")
    for lab, d in rows.items():
        n, b = d["new"], d["new_blocks"]
        print(f"{lab:9s} m_c {d['m_c']:3d}: D/W {n['q_range'][0]:.3f}-{n['q_range'][1]:.3f} max/min {n['q_max_over_min']:.3f} "
              f"({b['q_max_over_min'][0]:.2f}-{b['q_max_over_min'][1]:.2f}); |W/model-1| {n['W_over_model_maxdev']:.3f} "
              f"({b['W_over_model_maxdev'][0]:.3f}-{b['W_over_model_maxdev'][1]:.3f}); Pi/model {n['Pi_over_model'][0]:.3f}-"
              f"{n['Pi_over_model'][1]:.3f} ({b['Pi_over_model'][0]:.3f}-{b['Pi_over_model'][1]:.3f}); kappa_eff "
              f"{n['kappa_eff'][0]:.3f}-{n['kappa_eff'][1]:.3f} ({b['kappa_eff'][0]:.3f}-{b['kappa_eff'][1]:.3f}); "
              f"Pi/noecho {n['noecho'][0]:.3f}-{n['noecho'][1]:.3f} ({b['noecho'][0]:.3f}-{b['noecho'][1]:.3f}); "
              f"model {n['model_noecho'][0]:.3f}-{n['model_noecho'][1]:.3f}")
    out = dict(rule=dict(crit=CRIT, m_lo=M_LO, text="2 <= m <= m_c, m_c the largest m with 2 nu_m <= 0.13/T, "
                         "1/T the median of D_perp/W over the same range"),
               table2=list(TABLE2), table3=list(TABLE3), runs=rows, M32_vs_M64=pairs)
    (P.DATA / "h1_mrange.json").write_text(json.dumps(P._jsonable(out), indent=1))
    print(f"wrote {P.DATA / 'h1_mrange.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
