#!/usr/bin/env python
"""Figure fig:dictionary: the Hermite spectrum of decorrelated and of correlated slices (paper section 5,
eq:dictionary, eq:prefactor and the Kanekar et al. 2015 closed form).

Physics. The Hermite spectrum is a quadratic form in the two-slice correlation, W(m) = iint C(u,u') phi_m(u)
phi_m(u') du du', with phi_m = sqrt(p_m rho) r_m the orthonormal images of the Hermite basis.  Delta-correlated
slices, C = e(u) delta(u - u'), give W(m) = int e phi_m^2 du; for e = F_M at Lambda = infinity this is
Gamma(m+1/2)/(sqrt2 pi m!) ~ 1/(pi sqrt(2m)).  Panel (a) shows W(m) in units of its large-m asymptote:
  - the exact delta-correlated spectrum with e = F_M, pi sqrt(2m) Gamma(m+1/2)/(sqrt2 pi m!);
  - the continuum delta-correlated spectrum with e proportional to rho at Lambda = +-sqrt5,
    pi p_0 sqrt(2m) int rho phi_m^2 du, by the K-point Gauss rule of rho du built from GANDALF's ladder
    (int rho phi_m^2 du = sum_j Y_mj^2 rho(u_j), Y the orthonormal eigenvectors of P^(1/2) A P^(-1/2));
  - the forced linear spectrum of Kanekar et al. (2015): white noise of unit intensity on g_0 of one Fourier mode,
    d g = -i s k_z A g dt + dB, with an outgoing (radiation) closure g_(M+1) = -i sgn(k_z) g_M at M = 600, steady
    covariance S from the Lyapunov equation L S + S L^H = -Q; W(m) = p_m S_mm in units of eps_W/(s |k_z| sqrt(2m)),
    eps_W = p_0 the injection rate of the free energy.
Panel (b) shows the local slope d ln W/d ln m for slice correlations C = [e(u) e(u')]^(1/2) c(u - u') with e = F_M
and cosine transform chat(kappa) = (1 + kappa^2 ell^2)^(-p/2), for which (paper section 5)
    W(m) = (1/pi) int_0^inf chat(kappa) exp(-kappa^2/2) (kappa^2/2)^m/m! dkappa  ->  W ~ m^(-(p+1)/2),
with the derivative d W/d m from the same integral weighted by ln(kappa^2/2) - psi(m+1) (no finite differences).

The numbers are deterministic.  They are computed from configs/figures.yaml (section "dictionary"), checked against
derivations/truncated/out/t4_dictionary.out (A2, C2, D1), cached in data/computed/fig_dictionary.npz (+ .json), and
plotted from the cache to figures/dictionary.pdf and .png.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/fig_dictionary.py [--recompute]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from scipy.integrate import quad  # noqa: E402
from scipy.linalg import eigh_tridiagonal, solve_continuous_lyapunov  # noqa: E402
from scipy.special import digamma, gammaln, wofz  # noqa: E402

from krmhd.hermite import compute_streaming_matrix  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import (FIGURES_YAML, REPO, Checks, cache_paths, config_hash, environment, lam_tex,  # noqa: E402
                      lam_value, load_figures_config, panel_label, read_cache_meta, render, write_cache)

NAME = "dictionary"
GATES = {g: "not applicable: deterministic computation, no time integration"
         for g in ("energy_conservation", "energy_balance", "steady_state", "spectral_sanity")}


# ------------------------------------------------------------------------------------------------------------------
# physics
# ------------------------------------------------------------------------------------------------------------------
def gh_ratio(m: np.ndarray) -> np.ndarray:
    """Gamma(m+1/2)/(sqrt2 pi m!): W(m) of delta-correlated slices with density F_M at Lambda = infinity."""
    m = np.asarray(m, dtype=float)
    return np.exp(gammaln(m + 0.5) - gammaln(m + 1.0)) / (np.sqrt(2.0) * np.pi)


def weights(M: int, lam: float) -> np.ndarray:
    """Free-energy weights p_0 = 1 - 1/Lambda, p_m = 1 (zero closure)."""
    p = np.ones(M + 1)
    p[0] = 1.0 - (0.0 if np.isinf(lam) else 1.0 / lam)
    return p


def ladder(M: int, lam: float) -> np.ndarray:
    """GANDALF's zero-closure streaming matrix A (krmhd.hermite.compute_streaming_matrix)."""
    return np.asarray(compute_streaming_matrix(int(M), float(lam), "zero"), dtype=float)


def spectral_density(u: np.ndarray, lam: float) -> np.ndarray:
    """rho(u) = F_M(u)/|eps(u + i0)|^2, eps = 1 - [1 + u Z(u)]/Lambda, Z(u + i0) = i sqrt(pi) w(u)."""
    u = np.asarray(u, dtype=float)
    F = np.exp(-u ** 2) / np.sqrt(np.pi)
    if np.isinf(lam):
        return F
    eps = 1.0 - (1.0 + u * 1j * np.sqrt(np.pi) * wofz(u)) / lam
    return F / np.abs(eps) ** 2


def forced_spectrum(M: int, lam: float, kz: float, s: float) -> np.ndarray:
    """Steady W(m) = p_m <|g_m|^2> of d g = -i s k_z A g dt + dB (unit white noise on g_0) with the outgoing closure
    at m = M (last diagonal entry -|k_z| s sqrt((M+1)/2)): Lyapunov solve L S + S L^H = -Q."""
    L = -1j * s * kz * ladder(M, lam).astype(complex)
    L[M, M] += -abs(kz) * s * np.sqrt((M + 1) / 2.0)
    Q = np.zeros((M + 1, M + 1), complex)
    Q[0, 0] = 1.0
    S = solve_continuous_lyapunov(L, -Q)
    return weights(M, lam) * np.real(np.diag(S))


def continuum_delta(lam: float, K: int, m_max: int) -> np.ndarray:
    """int rho phi_m^2 du for m = 0..m_max by the (K+1)-point Gauss rule of rho du (Golub-Welsch on the symmetrised
    ladder P^(1/2) A P^(-1/2)): sum_j Y_mj^2 rho(u_j)."""
    p = weights(K, lam)
    sp = np.sqrt(p)
    J = (sp[:, None] * ladder(K, lam)) / sp[None, :]
    off = 0.5 * (np.diag(J, 1) + np.diag(J, -1))
    u, Y = eigh_tridiagonal(np.zeros(K + 1), off)
    return (Y[:m_max + 1, :] ** 2) @ spectral_density(u, lam)


def powerlaw_W(m: float, p: float, ell: float, epsrel: float, slope: bool = False) -> tuple[float, float]:
    """W(m) = (1/pi) int_0^inf chat(kappa) exp(-kappa^2/2)(kappa^2/2)^m/m! dkappa for chat = (1 + kappa^2 ell^2)^(-p/2),
    written with x = kappa^2/2 as a Gamma(m + 1/2) average; with slope, also d ln W/d ln m."""
    def logw(x: float) -> float:          # log of the Gamma(m+1/2, 1) density times its normalisation
        return (m - 0.5) * np.log(x) - x - gammaln(m + 1.0)

    def chat(x: float) -> float:
        return (1.0 + 2.0 * x * ell * ell) ** (-0.5 * p)

    half = 40.0 * np.sqrt(m + 1.0) + 40.0
    a, b = max(0.0, m - half), m + half
    pts = [min(max(m, a + 1e-9), b - 1e-9)]
    f0 = lambda x: chat(x) * np.exp(logw(x)) if x > 0 else 0.0
    W = quad(f0, a, b, points=pts, epsabs=0.0, epsrel=epsrel, limit=400)[0] / (np.sqrt(2.0) * np.pi)
    if not slope:
        return W, np.nan
    psi = digamma(m + 1.0)
    f1 = lambda x: chat(x) * np.exp(logw(x)) * (np.log(x) - psi) if x > 0 else 0.0
    dW = quad(f1, a, b, points=pts, epsabs=0.0, epsrel=epsrel, limit=400)[0] / (np.sqrt(2.0) * np.pi)
    return W, m * dW / W


# ------------------------------------------------------------------------------------------------------------------
# computation
# ------------------------------------------------------------------------------------------------------------------
def compute(cfg: dict, chash: str) -> dict:
    """Compute and check every number of the figure; write the cache; return the metadata."""
    t0 = time.time()
    start = time.strftime("%Y%m%d_%H%M%S")
    chk = Checks()
    ref = cfg["checks"]
    key: dict = {}

    print("(a) delta-correlated slices and the forced linear spectrum")
    m_d = np.arange(1, int(cfg["delta"]["m_max"]) + 1)
    delta_norm = np.pi * np.sqrt(2.0 * m_d) * gh_ratio(m_d)
    dl = ref["delta_large_m"]
    got = [float(np.pi * np.sqrt(2.0 * m) * gh_ratio(m)) for m in dl["m"]]
    chk("t4 A2: pi sqrt(2m) Gamma(m+1/2)/(sqrt2 pi m!) at m = 50, 200",
        max(abs(x - y) for x, y in zip(got, dl["value"])) < dl["tol"], ", ".join(f"{x:.10f}" for x in got))

    fc = cfg["forced"]
    M, kz, s = int(fc["M"]), float(fc["kz"]), float(fc["s"])
    lams = [lam_value(x) for x in fc["Lambda"]]
    m_f = np.arange(1, int(fc["m_plot_max"]) + 1)
    forced = {}
    for L in lams:
        W = forced_spectrum(M, L, kz, s)
        forced[L] = W[m_f] * abs(kz) * s * np.sqrt(2.0 * m_f) / weights(M, L)[0]
    fr = ref["forced"]
    worst = 0.0
    for Lr, row in zip(fr["Lambda"], fr["value"]):
        L = lam_value(Lr)
        vals = forced[L][np.array(fr["m"]) - 1]
        worst = max(worst, float(np.abs(vals - np.array(row)).max()))
        print(f"    Lambda = {L:+.4f}: W(m) s|k_z| sqrt(2m)/eps_W at m = {fr['m']}: {np.round(vals, 4).tolist()}")
    chk("t4 C2: forced spectrum on the ladder, outgoing closure, M = 600 (table, Lambda = inf, +-sqrt5)",
        worst < fr["tol"], f"max |diff| {worst:.1e}")
    ms = np.array(fr["m"])
    dev_k = float(np.abs(forced[np.inf][ms - 1] / (np.pi * np.sqrt(2.0 * ms) * gh_ratio(ms)) - 1.0).max())
    chk("t4 C2: Lambda = inf forced spectrum against the Kanekar closed form, max relative deviation",
        abs(dev_k - fr["kanekar_max_rel_dev"]) < fr["kanekar_tol"], f"{dev_k:.2e}")
    key["forced_inf_vs_closed_form_max_rel_dev"] = dev_k

    cc = cfg["continuum"]
    K = int(cc["K_gauss"])
    cont = {}
    cref = ref["continuum"]
    for Lr in cc["Lambda"]:
        L = lam_value(Lr)
        I = continuum_delta(L, K, int(fc["m_plot_max"]))
        cont[L] = np.pi * weights(K, L)[0] * np.sqrt(2.0 * m_f) * I[m_f]
        if np.isinf(L):
            e = float(np.abs(I[m_f] / gh_ratio(m_f) - 1.0).max())
            chk(f"Gauss rule of F_M du (K = {K}): int F_M psi_m^2 du = Gamma(m+1/2)/(sqrt2 pi m!), m <= {m_f[-1]}",
                e < cref["inf_tol"], f"max relative error {e:.1e}")
        sel = m_f <= 200
        d = float(np.abs(forced[L][sel] - cont[L][sel]).max())
        chk(f"forced ladder = continuum delta-correlated slices with e ~ rho, Lambda = {L:+.4f} (m <= 200)",
            d < cref["vs_forced_tol"], f"max |diff| in units of the asymptote {d:.1e}")
        key[f"forced_minus_continuum_max_{lam_tex(L)}"] = d
        key[f"continuum_m1_{lam_tex(L)}"] = float(cont[L][0])
    for L in lams:
        if not np.isinf(L):
            r = float(cont[L][0] / (np.pi * np.sqrt(2.0) * gh_ratio(1)))
            key[f"m1_ratio_to_(1-1/Lambda)_closed_form_{lam_tex(L)}"] = r
            print(f"    Lambda = {L:+.4f}: at m = 1 the finite-Lambda spectrum is {r:.4f} times (1 - 1/Lambda) x the "
                  f"closed form ({100 * abs(1 - r):.1f}% {'below' if r < 1 else 'above'})")

    print("(b) power-law slice correlations: W(m) and its local slope")
    pc = cfg["powerlaw"]
    ps = [float(p) for p in pc["p"]]
    ells = [float(e) for e in pc["ell"]]
    m_pl = np.logspace(np.log10(pc["m_min"]), np.log10(pc["m_max"]), int(pc["n_m"]))
    eps = float(pc["quad_epsrel"])
    slopes = np.zeros((len(ells), len(ps), m_pl.size))
    for i, ell in enumerate(ells):
        for j, p in enumerate(ps):
            slopes[i, j] = [powerlaw_W(float(m), p, ell, eps, slope=True)[1] for m in m_pl]
    pr = ref["powerlaw"]
    mr = np.array(pr["m"], dtype=float)
    worst_r, worst_s = 0.0, 0.0
    for row in pr["rows"]:
        ell, p = float(row[0]), float(row[1])
        Wm = np.array([powerlaw_W(float(m), p, ell, eps)[0] for m in mr])
        pred = (1.0 + 2.0 * mr * ell * ell) ** (-0.5 * p) / (np.pi * np.sqrt(2.0 * mr))
        ratio = Wm / pred
        worst_r = max(worst_r, float(np.abs(ratio - np.array(row[2:2 + mr.size])).max()))
        sel = np.isin(mr, pr["fit_m"])
        fit = float(np.polyfit(np.log(mr[sel]), np.log(Wm[sel]), 1)[0])
        worst_s = max(worst_s, abs(fit - row[-1]))
    chk("t4 D1: W(m)/[chat(sqrt2m)/(pi sqrt2m)] at m = 4..400 and the fitted slope over m = 128..400 (ell = 0.15, 1; "
        "p = 0.5..2.5)", worst_r < pr["tol"] and worst_s < pr["tol"],
        f"max |diff| ratio {worst_r:.1e}, slope {worst_s:.1e}")
    for i, ell in enumerate(ells):
        for j, p in enumerate(ps):
            key[f"slope_m1e4_ell{ell:g}_p{p:g}"] = float(slopes[i, j, -1])
            key[f"slope_m100_ell{ell:g}_p{p:g}"] = float(np.interp(np.log(100.0), np.log(m_pl), slopes[i, j]))
    dev_pl = float(np.abs(slopes[ells.index(1.0), :, -1] + (np.array(ps) + 1) / 2).max()) if 1.0 in ells else np.nan
    chk("local slope at m = 10^4 within 0.01 of -(p+1)/2 (ell = 1)", dev_pl < 0.01, f"max |diff| {dev_pl:.1e}")
    for L in lams:
        key[f"forced_m1_{lam_tex(L)}"] = float(forced[L][0])
        key[f"forced_m200_{lam_tex(L)}"] = float(forced[L][199])

    arrays = dict(m_delta=m_d, delta_norm=delta_norm, m_forced=m_f, Lambda=np.array(lams),
                  forced_norm=np.array([forced[L] for L in lams]),
                  continuum_norm=np.array([cont[lam_value(x)] for x in cc["Lambda"]]),
                  Lambda_continuum=np.array([lam_value(x) for x in cc["Lambda"]]),
                  m_powerlaw=m_pl, slopes=slopes, p=np.array(ps), ell=np.array(ells))
    wall = time.time() - t0
    meta = dict(figure=NAME, computed_local=start, config_file=str(FIGURES_YAML.relative_to(REPO)), config=cfg,
                config_hash=chash, environment=environment(), wall_s=round(wall, 2), gates=GATES, checks=chk.rows,
                all_checks_pass=chk.all_passed, key_numbers=key)
    write_cache(NAME, arrays, meta)
    print(f"computed in {wall:.1f} s; {sum(r['passed'] for r in chk.rows)}/{len(chk.rows)} checks passed")
    return meta


# ------------------------------------------------------------------------------------------------------------------
# plotting (from the cache only)
# ------------------------------------------------------------------------------------------------------------------
def plot(style: dict, cfg: dict) -> str:
    """Draw the figure from the cached numbers; returns the rendering mode used."""
    d = np.load(cache_paths(NAME)[0])
    pal = style["palette"]
    black = style["reference_color"]
    nmk = int(cfg["forced"]["n_markers"])

    def draw(plt):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(style["double_width_in"], 2.75),
                                       gridspec_kw=dict(wspace=0.3))
        # (a) W(m) in units of the large-m asymptote
        axa.axhline(1.0, color="0.6", lw=0.6, zorder=1)
        axa.plot(d["m_delta"], d["delta_norm"], color=black, lw=1.2, zorder=2,
                 label=r"$\delta$-correlated, $e\propto F_M$")
        colour = {}
        for i, L in enumerate(d["Lambda"]):
            colour[float(L)] = pal[i]
        for L, row in zip(d["Lambda_continuum"], d["continuum_norm"]):
            if np.isinf(L):
                continue
            axa.plot(d["m_forced"], row, color=colour[float(L)], lw=1.0, zorder=2,
                     label=rf"$\delta$-correlated, $e\propto\varrho$, $\Lambda={lam_tex(float(L))}$")
        idx = np.unique(np.round(np.logspace(0, np.log10(d["m_forced"][-1]), nmk)).astype(int)) - 1
        marks = ["o", "s", "^"]
        for i, L in enumerate(d["Lambda"]):
            axa.plot(d["m_forced"][idx], d["forced_norm"][i][idx], ls="none", marker=marks[i], ms=4.0, mfc="white",
                     mec=colour[float(L)], mew=1.0, zorder=3, label=rf"forced, $\Lambda={lam_tex(float(L))}$")
        axa.set_xscale("log")
        axa.set_xlim(0.8, 500)
        axa.set_ylim(0.82, 1.02)
        axa.set_xlabel(r"$m$")
        axa.set_ylabel(r"$\pi(2m)^{1/2}\,W(m)\big/\!\int e\,\mathrm{d}u$")
        axa.legend(loc="lower right", fontsize=style["legend_font_size_pt"] - 1, handlelength=1.6,
                   labelspacing=0.3)
        # (b) local slopes
        m = d["m_powerlaw"]
        lss = ["-", (0, (4, 2))]
        for j, p in enumerate(d["p"]):
            axb.axhline(-(p + 1) / 2, color=pal[j], lw=0.6, ls=":", zorder=1)
            for i, ell in enumerate(d["ell"]):
                axb.plot(m, d["slopes"][i, j], color=pal[j], lw=1.1, ls=lss[i % 2], zorder=2)
            axb.text(0.7 * m[-1], -(p + 1) / 2 - 0.035, rf"$p={p:g}$", color=pal[j], ha="right", va="top",
                     fontsize=style["legend_font_size_pt"])
        hs = [axb.plot([], [], color=black, lw=1.1, ls=lss[i % 2], label=rf"$\ell={ell:g}$")[0]
              for i, ell in enumerate(d["ell"])]
        axb.set_xscale("log")
        axb.set_xlim(m[0], m[-1])
        axb.set_ylim(-1.9, -0.38)
        axb.set_xlabel(r"$m$")
        axb.set_ylabel(r"$\mathrm{d}\ln W/\mathrm{d}\ln m$")
        axb.legend(handles=hs, loc="upper right", handlelength=2.2)
        panel_label(axa, "(a)")
        panel_label(axb, "(b)")
        return fig

    return render(draw, NAME, style)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recompute", action="store_true", help="recompute even if the cache matches the config")
    args = ap.parse_args()
    cfg, style = load_figures_config(NAME)
    chash = config_hash(cfg)
    meta = read_cache_meta(NAME)
    if args.recompute or meta is None or meta.get("config_hash") != chash:
        meta = compute(cfg, chash)
    else:
        print(f"using cached numbers computed {meta.get('computed_local')} ({cache_paths(NAME)[0]})")
    if not meta["all_checks_pass"]:
        print("FAILED CHECKS:", *[r["name"] for r in meta["checks"] if not r["passed"]], sep="\n  ")
        return 1
    meta["rendering"] = plot(style, cfg)
    write_cache(NAME, None, meta)
    print(f"wrote figures/{NAME}.pdf and .png ({meta['rendering']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
