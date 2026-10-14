#!/usr/bin/env python
"""Figure fig:spectrum: the spectral density of the branch operator, the Gauss rule of GANDALF's truncated ladder,
and the largest phase velocity u_max of the truncated hierarchy.

Physics. One compressive branch obeys d_t G + {Phi, G} + s grad_par(L G) = 0 with L G = u G - (u F_M/Lambda) M_0[G],
u = v_par/v_th, F_M(u) = exp(-u^2)/sqrt(pi). For Lambda outside [0, 1] the operator L has purely continuous spectrum
on the real line with spectral density (paper eq:varrho)
    rho(u) = F_M(u)/|eps(u + i0)|^2,   eps(zeta) = 1 - [1 + zeta Z(zeta)]/Lambda,
Z the plasma dispersion function, Z(u + i0) = i sqrt(pi) w(u) (Faddeeva function). Its total mass is
Lambda/(Lambda - 1). GANDALF's zero-closure Hermite ladder A (krmhd.hermite.compute_streaming_matrix) is the Jacobi
matrix of rho du: its eigenvalues u_j are the nodes and lambda_j = R_0j^2 (right eigenvectors with R^T P R = I,
P = diag(1 - 1/Lambda, 1, ..., 1)) the Christoffel weights of the (M+1)-point Gauss rule (paper eq:gauss), so
lambda_j/Delta u_j approximates rho(u_j). Panel (a) shows rho for Lambda = sqrt5, -sqrt5, infinity and these
weights at M = 16. Panel (b) shows u_max = max_j |u_j| for the zero closure (g_{M+1} = 0) and the copy closure
(g_{M+1} = g_{M-1}) against M, with the Airy-Langer asymptotic forms (paper eq:umax)
    zero: u_max = nu^(1/2) + 2^(-1/3) a_1 nu^(-1/6) - 2^(-2/3) a_1^2 nu^(-5/6)/10 + O(nu^(-3/2)),  nu = 2M + 3,
    copy: u_max = nu^(1/2) + 2^(-1/3) a'_1 nu^(-1/6) - 2^(-5/3)(4/(5 a'_1) + a'_1^2/5) nu^(-5/6) + ...,  nu = 2M + 1,
a_1 and a'_1 the first zeros of Ai and Ai'.

The numbers are deterministic (no time integration). They are computed from configs/figures.yaml (section
"spectrum"), checked against the outputs of derivations/continuum/s04_cvk_transform.py, s05_ladder_ops.py and
derivations/truncated/t2_eigenvalues.py, cached in data/computed/fig_spectrum.npz (+ .json), and plotted from the
cache to figures/spectrum.pdf and .png.

Usage (from the repository root):
    uv run python paper/phase-velocity-slices/analysis/fig_spectrum.py [--recompute]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np  # noqa: E402
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
from scipy.integrate import quad  # noqa: E402
from scipy.linalg import eigh_tridiagonal  # noqa: E402
from scipy.special import ai_zeros, wofz  # noqa: E402

from krmhd.hermite import compute_streaming_matrix  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from figstyle import (Checks, config_hash, environment, lam_tex, lam_value, load_figures_config,  # noqa: E402
                      panel_label, read_cache_meta, render, write_cache, cache_paths, FIGURES_YAML, REPO)

NAME = "spectrum"
GATES = {g: "not applicable: deterministic computation, no time integration"
         for g in ("energy_conservation", "energy_balance", "steady_state", "spectral_sanity")}


# ------------------------------------------------------------------------------------------------------------------
# physics
# ------------------------------------------------------------------------------------------------------------------
def maxwellian(u: np.ndarray) -> np.ndarray:
    """F_M(u) = exp(-u^2)/sqrt(pi), normalised in u = v_par/v_th."""
    return np.exp(-np.asarray(u, dtype=float) ** 2) / np.sqrt(np.pi)


def dielectric(u: np.ndarray, lam: float) -> np.ndarray:
    """eps(u + i0) = 1 - [1 + u Z(u + i0)]/Lambda on the real axis, Z(u + i0) = i sqrt(pi) w(u)."""
    u = np.asarray(u, dtype=float)
    if np.isinf(lam):
        return np.ones_like(u) + 0j
    Z = 1j * np.sqrt(np.pi) * wofz(u)
    return 1.0 - (1.0 + u * Z) / lam


def spectral_density(u: np.ndarray, lam: float) -> np.ndarray:
    """rho(u) = F_M(u)/|eps(u + i0)|^2 (paper eq:varrho); rho = F_M for Lambda = infinity."""
    return maxwellian(u) / np.abs(dielectric(u, lam)) ** 2


def weights(M: int, lam: float, closure: str = "zero") -> np.ndarray:
    """Diagonal of the weight Pi that symmetrises GANDALF's ladder: p_0 = 1 - 1/Lambda, p_m = 1, and for the copy
    closure p_M = sqrt(M)/(sqrt(M) + sqrt(M+1))."""
    p = np.ones(M + 1)
    p[0] = 1.0 - (0.0 if np.isinf(lam) else 1.0 / lam)
    if closure == "symmetric":
        p[M] = np.sqrt(M) / (np.sqrt(M) + np.sqrt(M + 1.0))
    return p


def ladder(M: int, lam: float, closure: str = "zero") -> np.ndarray:
    """GANDALF's streaming matrix A (the matrix of u on the Hermite coefficients g_0..g_M)."""
    return np.asarray(compute_streaming_matrix(int(M), float(lam), closure), dtype=float)


def symmetrised(M: int, lam: float, closure: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Pi^(1/2) A Pi^(-1/2) for GANDALF's ladder (symmetric tridiagonal), the weight Pi and its asymmetry."""
    A = ladder(M, lam, closure)
    p = weights(M, lam, closure)
    sp = np.sqrt(p)
    S = (sp[:, None] * A) / sp[None, :]
    return S, p, float(np.abs(S - S.T).max())


def gauss_rule(M: int, lam: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Nodes u_j (eigenvalues of A) and Christoffel weights lambda_j = R_0j^2 of the zero-closure ladder; also the
    asymmetry of the symmetrised matrix and max |eig(A) - u_j| (A diagonalised directly)."""
    S, p, asym = symmetrised(M, lam, "zero")
    u, Q = np.linalg.eigh(0.5 * (S + S.T))
    R = Q / np.sqrt(p)[:, None]
    lam_j = R[0, :] ** 2
    direct = np.sort(np.linalg.eigvals(ladder(M, lam)).real)
    return u, lam_j, asym, float(np.abs(direct - u).max())


def umax(M: int, lam: float, closure: str) -> float:
    """Largest phase velocity u_max = max_j u_j of GANDALF's ladder (spectrum symmetric under u -> -u)."""
    S, _, _ = symmetrised(M, lam, closure)
    off = 0.5 * (np.diag(S, 1) + np.diag(S, -1))
    return float(eigh_tridiagonal(np.zeros(M + 1), off, eigvals_only=True, select="i",
                                  select_range=(M, M))[0])


def umax_asymptotic(M: np.ndarray, closure: str, terms: int = 3) -> np.ndarray:
    """Asymptotic u_max(M) of der-truncated eqs. (17) (zero closure) and (18) (copy closure)."""
    a1, a1p = ai_zeros(1)[0][0], ai_zeros(1)[1][0]
    M = np.asarray(M, dtype=float)
    if closure == "zero":
        nu = 2.0 * M + 3.0
        out = np.sqrt(nu) + 2.0 ** (-1 / 3) * a1 * nu ** (-1 / 6)
        if terms >= 3:
            out = out - 2.0 ** (-2 / 3) * a1 ** 2 / 10.0 * nu ** (-5 / 6)
    else:
        nu = 2.0 * M + 1.0
        out = np.sqrt(nu) + 2.0 ** (-1 / 3) * a1p * nu ** (-1 / 6)
        if terms >= 3:
            out = out - 2.0 ** (-5 / 3) * (4.0 / (5.0 * a1p) + a1p ** 2 / 5.0) * nu ** (-5 / 6)
    return out


def nu_of(M: np.ndarray, closure: str) -> np.ndarray:
    return 2.0 * np.asarray(M, dtype=float) + (3.0 if closure == "zero" else 1.0)


# ------------------------------------------------------------------------------------------------------------------
# computation
# ------------------------------------------------------------------------------------------------------------------
def compute(cfg: dict, chash: str) -> dict:
    """Compute and check every number of the figure; write the cache; return the metadata."""
    t0 = time.time()
    start = time.strftime("%Y%m%d_%H%M%S")
    chk = Checks()
    ref = cfg["checks"]
    key: dict[str, Any] = {}

    print("(a) spectral density and the Gauss rule of GANDALF's ladder")
    dens = cfg["density"]
    lams = [lam_value(x) for x in dens["Lambda"]]
    u = np.linspace(dens["u_range"][0], dens["u_range"][1], int(dens["n_u"]))
    rho = np.array([spectral_density(u, L) for L in lams])
    a, b = dens["mass_quad_range"]
    for L in lams:
        mass = quad(lambda x: float(spectral_density(np.array([x]), L)[0]), a, b, limit=1000, epsabs=1e-14,
                    epsrel=1e-13)[0]
        mu0 = 1.0 if np.isinf(L) else L / (L - 1.0)
        chk(f"int rho du = Lambda/(Lambda - 1), Lambda = {L:+.4f}", abs(mass - mu0) < ref["mass_tol"],
            f"{mass:.15f} vs {mu0:.15f}")
        r0 = float(spectral_density(np.array([0.0]), L)[0])
        pred = maxwellian(0.0) / (1.0 - (0.0 if np.isinf(L) else 1.0 / L)) ** 2
        chk(f"rho(0) = F_M(0)/(1 - 1/Lambda)^2, Lambda = {L:+.4f}", abs(r0 / pred - 1.0) < 1e-14, f"{r0:.10f}")
        key[f"rho0_{lam_tex(L)}"] = r0
        key[f"mass_{lam_tex(L)}"] = mass

    gcfg = cfg["gauss"]
    Mg = int(gcfg["M"])
    nodes, wts, wdu, devs = [], [], [], []
    for L in lams:
        uj, lj, asym, ediff = gauss_rule(Mg, L)
        du = np.gradient(uj)
        sel = np.abs(uj) < float(gcfg["window"])
        dev = float(np.abs(lj[sel] / (spectral_density(uj[sel], L) * du[sel]) - 1.0).max())
        mu0 = 1.0 if np.isinf(L) else L / (L - 1.0)
        chk(f"Gauss rule M = {Mg}, Lambda = {L:+.4f}: Pi^(1/2) A Pi^(-1/2) symmetric, eig(A) = nodes, "
            f"sum lambda_j = Lambda/(Lambda - 1)",
            asym < 1e-14 and ediff < 1e-12 and abs(lj.sum() - mu0) < 1e-13,
            f"asym {asym:.1e}, |eig(A) - u_j| {ediff:.1e}, sum lambda_j - mu0 = {lj.sum() - mu0:.1e}")
        worst = 0.0
        for k in range(2 * Mg + 2):
            mk = quad(lambda x: x ** k * float(spectral_density(np.array([x]), L)[0]), a, b, limit=2000,
                      epsabs=0.0, epsrel=1e-12)[0]
            sk = quad(lambda x: abs(x) ** k * float(spectral_density(np.array([x]), L)[0]), a, b, limit=2000,
                      epsabs=0.0, epsrel=1e-12)[0]
            worst = max(worst, abs(float(np.sum(lj * uj ** k)) - mk) / sk)
        chk(f"Gauss exactness sum_j lambda_j u_j^k = int rho u^k du, k <= 2M+1 = {2 * Mg + 1}, Lambda = {L:+.4f}",
            worst < 1e-9, f"max relative error {worst:.1e}")
        nodes.append(uj)
        wts.append(lj)
        wdu.append(lj / du)
        devs.append(dev)
        key[f"gauss_dev_M{Mg}_{lam_tex(L)}"] = dev
        print(f"    Lambda = {L:+.4f}: max_(|u_j| < {gcfg['window']}) |lambda_j/(rho(u_j) Delta u_j) - 1| = {dev:.3e}; "
              f"lambda_j/Delta u_j at u = 0: {float((lj / du)[np.argmin(np.abs(uj))]):.4f}, rho(0) = "
              f"{float(spectral_density(np.array([0.0]), L)[0]):.4f}")
    gd = ref["gauss_dev"]
    for Lr, vr in zip(gd["Lambda"], gd["value"]):
        i = [j for j, L in enumerate(lams) if L == lam_value(Lr)][0]
        chk(f"s05 L5 (M = 16, Lambda = {lam_value(Lr):+.4f}): max |lambda_j/(rho Delta u) - 1| = {vr}",
            abs(devs[i] - vr) < gd["tol"], f"{devs[i]:.4e}")

    print("(b) u_max of GANDALF's ladder")
    ucfg = cfg["umax"]
    Lu = lam_value(ucfg["Lambda"])
    M_list = np.unique(np.round(np.logspace(np.log10(ucfg["M_min"]), np.log10(ucfg["M_max"]),
                                            int(ucfg["n_M"]))).astype(int))
    umax_vals = {c: np.array([umax(int(M), Lu, c) for M in M_list]) for c in ucfg["closures"]}
    tab = ref["umax_table"]
    worst = 0.0
    for c in ("zero", "symmetric"):
        for M, val in zip(tab["M"], tab[c]):
            worst = max(worst, abs(umax(int(M), Lu, c) - val))
            # the direct (non-symmetric) eigenvalues of A agree with the tridiagonal solver
            direct = float(np.abs(np.linalg.eigvals(ladder(int(M), Lu, c)).real).max())
            worst = max(worst, abs(direct - val))
    chk("t2 E: u_max table (M = 16..256, both closures, Lambda = sqrt5)", worst < tab["tol"], f"max |diff| {worst:.1e}")
    lam_dep = 0.0
    for M in M_list[M_list >= 32]:
        for c in ("zero", "symmetric"):
            base = umax(int(M), Lu, c)
            for L in (-np.sqrt(5.0), np.inf):
                lam_dep = max(lam_dep, abs(umax(int(M), L, c) - base))
    chk("u_max independent of Lambda (sqrt5, -sqrt5, inf) for M >= 32", lam_dep < 1e-12, f"max diff {lam_dep:.1e}")
    key["umax_Lambda_dependence_M_ge_32"] = lam_dep
    asy = ref["umax_asymptotics"]
    f3 = [float(umax_asymptotic(np.array([M]), "zero")[0]) for M in asy["M_formula"]]
    chk("t2 F: three-term zero-closure formula at M = 64, 128", max(abs(x - y) for x, y in zip(f3, asy["formula"]))
        < asy["tol"], ", ".join(f"{x:.6f}" for x in f3))
    rem = {}
    for c in ("zero", "symmetric"):
        rem[c] = [(umax(int(M), Lu, c) - float(umax_asymptotic(np.array([M]), c)[0])) * nu_of(M, c) ** 1.5
                  for M in asy["M_remainder"]]
        chk(f"t2 F: remainder (u_max - three-term) nu^(3/2), {c} closure, M = {asy['M_remainder']}",
            max(abs(x - y) for x, y in zip(rem[c], asy[f"remainder_{c}"])) < asy["tol"],
            ", ".join(f"{x:+.6f}" for x in rem[c]))
    for M in (16, 64, 128, 1024):
        for c in ("zero", "symmetric"):
            key[f"umax_{c}_M{M}"] = umax(M, Lu, c)
            key[f"umax_{c}_M{M}_minus_3term"] = umax(M, Lu, c) - float(umax_asymptotic(np.array([M]), c)[0])
    Mc = np.logspace(np.log10(ucfg["M_min"]), np.log10(ucfg["M_max"]), int(ucfg["n_curve"]))
    arrays = dict(
        u=u, rho=rho, Lambda=np.array(lams), M_gauss=np.array(Mg), nodes=np.array(nodes), weights=np.array(wts),
        weights_over_du=np.array(wdu), gauss_dev=np.array(devs), Lambda_umax=np.array(Lu), M_list=M_list,
        umax_zero=umax_vals["zero"], umax_copy=umax_vals["symmetric"], M_curve=Mc,
        asym3_zero=umax_asymptotic(Mc, "zero", 3), asym2_zero=umax_asymptotic(Mc, "zero", 2),
        asym3_copy=umax_asymptotic(Mc, "symmetric", 3), asym2_copy=umax_asymptotic(Mc, "symmetric", 2))
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
def plot(style: dict) -> str:
    """Draw the figure from the cached numbers; returns the rendering mode used."""
    d = np.load(cache_paths(NAME)[0])
    pal = style["palette"]
    black = style["reference_color"]

    def draw(plt):
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(style["double_width_in"], 2.6),
                                       gridspec_kw=dict(wspace=0.3))
        # (a) spectral density and Gauss weights
        marks = ["o", "s", "^"]
        for i, L in enumerate(d["Lambda"]):
            axa.plot(d["u"], d["rho"][i], color=pal[i], lw=1.2, label=rf"$\Lambda = {lam_tex(float(L))}$", zorder=2)
            axa.plot(d["nodes"][i], d["weights_over_du"][i], ls="none", marker=marks[i], ms=4.2, mfc="white",
                     mec=pal[i], mew=1.0, zorder=3)
        axa.plot([], [], ls="none", marker="o", ms=4.2, mfc="white", mec=black, mew=1.0,
                 label=rf"$\lambda_j/\Delta u_j$, $M = {int(d['M_gauss'])}$")
        axa.set_xlim(d["u"][0], d["u"][-1])
        axa.set_xticks([-4, -2, 0, 2, 4])
        axa.set_ylim(0.0, 2.0)
        axa.set_xlabel(r"$u$")
        axa.set_ylabel(r"$\varrho(u)$")
        axa.legend(loc="upper right", handlelength=1.6)
        # (b) u_max - (2M + 3)^(1/2)
        Mc = d["M_curve"]
        ref = np.sqrt(2.0 * Mc + 3.0)
        Ml = d["M_list"]
        refl = np.sqrt(2.0 * Ml + 3.0)
        hm = []
        for col, key, lab, mk in ((pal[0], "zero", r"zero closure", "o"), (pal[1], "copy", r"copy closure", "s")):
            axb.plot(Mc, d[f"asym3_{key}"] - ref, color=col, lw=1.0, zorder=2)
            axb.plot(Mc, d[f"asym2_{key}"] - ref, color=col, lw=0.9, ls=(0, (4, 2.5)), zorder=2)
            hm += axb.plot(Ml, d[f"umax_{key}"] - refl, ls="none", marker=mk, ms=3.6, mfc="white", mec=col,
                           mew=0.9, label=lab, zorder=3)
        hl = axb.plot([], [], color=black, lw=1.0, label=r"asymptotics, 3 terms")
        hl += axb.plot([], [], color=black, lw=0.9, ls=(0, (4, 2.5)), label=r"asymptotics, 2 terms")
        axb.set_xscale("log")
        axb.set_xlim(3.5, 1200)
        axb.set_xlabel(r"$M$")
        axb.set_ylabel(r"$u_{\max} - (2M + 3)^{1/2}$")
        leg = axb.legend(handles=hm, loc="upper left")
        axb.add_artist(leg)
        axb.legend(handles=hl, loc="lower right")
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
    meta["rendering"] = plot(style)
    write_cache(NAME, None, meta)
    print(f"wrote figures/{NAME}.pdf and .png ({meta['rendering']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
