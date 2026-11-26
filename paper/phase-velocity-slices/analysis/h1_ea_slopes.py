#!/usr/bin/env python
"""Slopes of the Alfvenic spectrum E_A(k_perp) at each resolution (second referee read, item 1.4).

Section 5.1 says that at 64^3 E_A follows k_perp^{-5/3} over 2 <= k_perp/k_1 <= 8, and section 5.6 that the slope
"fitted over the middle 60% of the resolved shells" is -4.1 at 64^3 and -2.9 at 256^3.  The two are different fits:
the second is the gate's slope (h1lib.spectral_slope over the shells 1..nmax), whose middle 60% lies mostly in the
dissipation range at 64^3.  For each run this script gives, on the window-averaged E_A(k_perp):
  * the gate's slope (middle 60% of shells 1..nmax), which reproduces the numbers of section 5.6;
  * the least-squares slope over the fixed range 2 <= k_perp/k_1 <= 8, the same at every N;
  * the same with the shell sums corrected for the number of lattice modes in each shell (E_c), whose fluctuation
    about 2 pi n makes the slope between neighbouring shells alternate;
  * local slopes of E_c between neighbouring shells, and the first shell where the local slope is steeper than -4;
  * the first shell where the hyperdissipation rate exceeds an estimate of the nonlinear rate of that shell.

Reads <run>/<run>.npz (ts__t, ts__acc_sum_dt, ts__EAk, nperp, nmax, ts__kperp_out, ts__u_out) and <run>.json
(config: box, stepper eta and hyper_r, gate window).  Writes data/h1_ea_slopes.json.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/h1_ea_slopes.py
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

RUNS = dict(b1="P1_forced-beta1_20261015_122613", b1_M64="P1_forced-beta1-m64_20261018_144645",
            b03_M64="P1_forced-beta0p3-m64_20261018_144656", b03_N96="P1_forced-beta0p3-n96_20261019_200712",
            b03_N128="P1_forced-beta0p3-n128_20261018_184515", b1_N128="P1_forced-beta1-n128_20261018_184509",
            b03_N256="P1_forced-beta0p3-n256_20261031_115113")
FIT = (2, 8)


def slopes(run_id: str, acc: dict) -> dict:
    rd = P.DATA / run_id
    cfg = json.load(open(rd / f"{run_id}.json"))["config"]
    with np.load(rd / f"{run_id}.npz") as z:
        t, dts, EAk = z["ts__t"], z["ts__acc_sum_dt"], z["ts__EAk"]
        nperp, nmax = z["nperp"].astype(float), int(z["nmax"])
        kperp_out, u_out = z["ts__kperp_out"], z["ts__u_out"]
    width = float(((acc.get("runs") or {}).get(run_id, {})).get("window", cfg["gates"]["steady_window"]))
    first = int(np.where(H.window_slice(t, width))[0][0])
    E = H.time_average(EAk, dts, first)
    tau_nl = 1.0 / float(H.time_average(kperp_out, dts, first) * H.time_average(u_out, dts, first))
    sel = (nperp >= 1) & (nperp <= nmax)
    gate = H.spectral_slope(nperp[sel], E[sel])
    f = (nperp >= FIT[0]) & (nperp <= FIT[1])
    fit = float(np.polyfit(np.log(nperp[f]), np.log(E[f]), 1)[0])
    # shell n sums the lattice modes with rint(|k_perp|/k_1) = n, whose number fluctuates about 2 pi n; E_c is the
    # shell sum per mode times 2 pi n, the spectrum with that fluctuation removed
    N = int(cfg["box"]["N"])
    ix = np.fft.fftfreq(N, d=1.0 / N)
    ix = ix[np.abs(ix) <= nmax]
    count = np.bincount(np.rint(np.hypot(*np.meshgrid(ix, ix))).astype(int).ravel(), minlength=len(nperp))
    Ec = np.where(count[:len(nperp)] > 0, E * 2 * np.pi * nperp / np.maximum(count[:len(nperp)], 1), np.nan)
    fit_c = float(np.polyfit(np.log(nperp[f]), np.log(Ec[f]), 1)[0])
    n = nperp[sel]
    lk, lE = np.log(n), np.log(Ec[sel])
    local = np.diff(lE) / np.diff(lk)                        # slope between shells n_i and n_{i+1}
    steep = [float(n[i]) for i in range(len(local)) if local[i] < -4.0 and n[i] >= 2]
    st = cfg["stepper"]
    eta, r = P.parse_number(st["eta"]), int(st.get("hyper_r", 2))
    # GANDALF's hyperdissipation rate eta (k_perp^2/k_perp,max^2)^r, k_perp,max^2 = 2 (nmax k_1)^2 (p1lib.Box), against
    # the nonlinear rate k_perp u_k of shell n, u_k^2 ~ n E_A(n) (E_A(n) the energy in the unit-width shell n; an
    # estimate, with E_A = (u^2 + b^2)/2 ~ u^2)
    k1 = 2.0 * math.pi / float(cfg["box"]["L"])
    rate_eta = eta * (n ** 2 / (2.0 * nmax ** 2)) ** r
    rate_nl = n * k1 * np.sqrt(n * Ec[sel])
    cross = [float(n[i]) for i in range(len(n)) if rate_eta[i] > rate_nl[i]]
    k_eq = cross[0] if cross else None
    lo, hi = n.size // 5, 4 * n.size // 5
    return dict(run_id=run_id, N=N, nmax=nmax, window_start=float(t[first]), tau_nl=tau_nl,
                slope_gate_middle60=gate, gate_shells=[float(n[lo]), float(n[hi - 1])],
                slope_2_8=fit, slope_2_8_mode_count_corrected=fit_c,
                slope_gate_mode_count_corrected=H.spectral_slope(nperp[sel], Ec[sel]),
                compensated_n53_Ec={str(int(k)): float(e * k ** (5 / 3)) for k, e in zip(n, Ec[sel])},
                local_slopes={f"{int(n[i])}-{int(n[i + 1])}": float(local[i]) for i in range(len(local))},
                first_shell_steeper_than_4=steep[0] if steep else None,
                first_shell_eta_rate_above_nl_rate=k_eq)


def main() -> int:
    acc = P.load_config("h1_acceptance.yaml")
    out = {lab: slopes(rid, acc) for lab, rid in RUNS.items()}
    print("label      N   nmax  gate slope (shells)  [mode-corrected]  slope 2..8 [mode-corrected]  first n with local < -4"
          "   first n with eta rate > nl rate")
    for lab, d in out.items():
        print(f"{lab:9s} {d['N']:4d} {d['nmax']:4d}   {d['slope_gate_middle60']:6.2f} ({d['gate_shells'][0]:.0f}-"
              f"{d['gate_shells'][1]:.0f}) [{d['slope_gate_mode_count_corrected']:6.2f}]   {d['slope_2_8']:6.2f} "
              f"[{d['slope_2_8_mode_count_corrected']:6.2f}]        {d['first_shell_steeper_than_4']}"
              f"                 {d['first_shell_eta_rate_above_nl_rate']}")
        print("          local (E_c): " + " ".join(f"{k}:{v:.2f}" for k, v in list(d["local_slopes"].items())[:16]))
        print("          n^(5/3) E_c: " + " ".join(f"{k}:{v:.3f}" for k, v in list(d["compensated_n53_Ec"].items())[:16]))
    (P.DATA / "h1_ea_slopes.json").write_text(json.dumps(P._jsonable(out), indent=1))
    print(f"wrote {P.DATA / 'h1_ea_slopes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
