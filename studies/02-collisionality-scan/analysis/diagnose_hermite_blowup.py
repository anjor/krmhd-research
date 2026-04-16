#!/usr/bin/env python3
"""Diagnose the nonlinear-Hermite blowup mechanism at M=128.

Reads locally-downloaded spectra + a single pre-blowup checkpoint for each of
the four blown-up runs (ν=1, 3, 5, 10). Produces figures that let us decide
whether the failure is physical pileup at m=M or a numerical streaming CFL
issue, and whether the k_z broadening narrative in docs/hermite_handoff.md
is supported by the data.

Assumes data layout produced by
`scripts/download_128_results.py --hermite-blowup` plus per-branch
`--checkpoint tXXXX.X` downloads.

Usage:
    uv run python studies/02-collisionality-scan/analysis/diagnose_hermite_blowup.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from krmhd.io import load_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STUDY_DIR = PROJECT_ROOT / "studies/02-collisionality-scan"
DATA_DIR = STUDY_DIR / "data/benchmark_128"
FIG_DIR = STUDY_DIR / "figures"


@dataclass(frozen=True)
class RunMeta:
    label: str
    nu: float
    blowup_t: float       # approximate absolute time of blowup
    ckpt_tag: str         # pre-blowup checkpoint we downloaded


RUNS = [
    RunMeta("hermite128_nu1p0_v3",  1.0,  2185.0, "t2170.0"),
    RunMeta("hermite128_nu3_long",  3.0,  2122.0, "t2120.0"),
    RunMeta("hermite128_nu5_long",  5.0,  2167.0, "t2160.0"),
    RunMeta("hermite128_nu10_long", 10.0, 2184.0, "t2180.0"),
]

ALFVEN_REF = "alfven128_lowkz_f0p02_eta100"
ALFVEN_REF_CKPT = "t2000.0"

HYPER_N = 6           # matches script config
HERMITE_START_T = 2000.0   # all runs expand to M=128 at this absolute time


def load_spectra(branch: str) -> list[dict]:
    """Return list of dicts sorted by time."""
    d = DATA_DIR / branch / "spectra"
    files = sorted(d.glob("spectrum_t*.npz"))
    out = [dict(np.load(f)) for f in files]
    out.sort(key=lambda s: float(s["time"]))
    return out


def stack_Wm_t(spectra: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (t[nt], W[nt, M+1]) from a list of spectrum dicts."""
    t = np.array([float(s["time"]) for s in spectra])
    W = np.stack([np.asarray(s["E_m"]) for s in spectra])
    return t, W


def compute_eps_nu(W_m: np.ndarray, nu: float, M: int, hyper_n: int = HYPER_N) -> float:
    """ε_ν = 2ν Σ_{m≥2} (m/M)^n W_m (matches modal_128_hermite.py:217)."""
    m_idx = np.arange(len(W_m))
    rates = nu * (m_idx / M) ** hyper_n
    rates[:2] = 0.0
    return float(2.0 * np.sum(rates * W_m))


def kz_spectrum(field: np.ndarray) -> np.ndarray:
    """Sum |field|^2 over the non-k_z axes, return array of length Nz.

    field has k_z as axis 0 (leading), other Fourier axes after. For a real
    FFT in x, this double-counts ikx=0/Nyquist but that's consistent across
    k_z and fine for comparison.
    """
    tail_axes = tuple(range(1, field.ndim))
    return np.sum(np.abs(field) ** 2, axis=tail_axes)


def kz_axis(grid) -> np.ndarray:
    kz = np.asarray(grid.kz)
    # Return absolute n_z = kz * L / (2π). Grid uses L=1 for the perpendicular
    # box, but the parallel box is typically 2π — check via first nonzero kz.
    # Simpler: just return kz directly, plot against it.
    return kz


# ---------------------------------------------------------------------------
# Figure 1: W(m, t) pcolormesh per run + Σ W and ε_ν side panel
# ---------------------------------------------------------------------------
def plot_Wmt_per_run() -> None:
    fig, axes = plt.subplots(len(RUNS), 2, figsize=(14, 3.5 * len(RUNS)),
                              gridspec_kw={"width_ratios": [2.4, 1.0]})
    if len(RUNS) == 1:
        axes = np.array([axes])

    for row, run in enumerate(RUNS):
        spectra = load_spectra(run.label)
        if not spectra:
            print(f"  No spectra for {run.label}")
            continue
        t, W = stack_Wm_t(spectra)
        M = W.shape[1] - 1
        m = np.arange(W.shape[1])

        # Clip to hermite phase
        hermite = t >= HERMITE_START_T
        t = t[hermite]
        W = W[hermite]

        ax_main, ax_side = axes[row]

        # pcolormesh: use log10 clipped to avoid -inf
        Wpos = np.where(W > 0, W, np.nan)
        vmin = np.nanpercentile(Wpos, 5)
        vmax = np.nanpercentile(Wpos, 99.5)
        pcm = ax_main.pcolormesh(
            t - HERMITE_START_T, m, np.log10(Wpos.T),
            cmap="viridis",
            vmin=np.log10(vmin), vmax=np.log10(vmax), shading="auto",
        )
        ax_main.axvline(run.blowup_t - HERMITE_START_T, color="red",
                        ls="--", lw=1.5, label="blowup")
        ax_main.set_ylabel("Hermite moment $m$")
        ax_main.set_xlabel(r"$t - t_0\;(\tau_A)$")
        ax_main.set_title(fr"{run.label}  —  $\nu={run.nu}$")
        ax_main.legend(loc="upper left", fontsize=8)
        plt.colorbar(pcm, ax=ax_main, label=r"$\log_{10} W(m,t)$")

        # Side panel: Σ W(m) and W(m=M) and ε_ν(t)
        W_total = W.sum(axis=1)
        W_top = W[:, -1]
        eps = np.array([compute_eps_nu(w, run.nu, M) for w in W])

        tt = t - HERMITE_START_T
        ax_side.semilogy(tt, W_total, label=r"$\Sigma_m W_m$", color="C0")
        ax_side.semilogy(tt, W_top, label=r"$W(m{=}M)$", color="C1")
        ax_side.semilogy(tt, eps, label=r"$\varepsilon_\nu$", color="C3")
        ax_side.axvline(run.blowup_t - HERMITE_START_T, color="red", ls="--", lw=1)
        ax_side.set_xlabel(r"$t - t_0\;(\tau_A)$")
        ax_side.legend(fontsize=8, loc="best")
        ax_side.grid(alpha=0.3, which="both")

    plt.tight_layout()
    out = FIG_DIR / "hermite_blowup_Wmt.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 2: W(m=M-k, t) across runs — does W(M) at blowup scale with ν?
# ---------------------------------------------------------------------------
def plot_Wm_threshold_comparison() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    offsets = [0, 5, 20]  # m = M, M-5, M-20

    for ax, off in zip(axes, offsets):
        for run in RUNS:
            spectra = load_spectra(run.label)
            if not spectra:
                continue
            t, W = stack_Wm_t(spectra)
            M = W.shape[1] - 1
            hermite = t >= HERMITE_START_T
            t = t[hermite]
            W = W[hermite]
            ax.semilogy(t - HERMITE_START_T, W[:, M - off],
                         label=fr"$\nu={run.nu}$")
            ax.axvline(run.blowup_t - HERMITE_START_T,
                       color="gray", ls=":", alpha=0.3)
        ax.set_xlabel(r"$t - t_0\;(\tau_A)$")
        ax.set_title(fr"$W(m{{=}}M{{-}}{off},\,t)$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$W_m$")
    plt.tight_layout()
    out = FIG_DIR / "hermite_blowup_Wm_threshold_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 3: k_z spectra of z± and of g (at m=M and Σ_m) from pre-blowup
# checkpoints, plus the Alfvénic-only steady-state reference.
# ---------------------------------------------------------------------------
def plot_kz_spectra() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_zpm, ax_g = axes

    # Alfvénic reference
    ref_path = (
        DATA_DIR / ALFVEN_REF / "checkpoints"
        / f"checkpoint_{ALFVEN_REF_CKPT}.h5"
    )
    state_ref, grid_ref, _ = load_checkpoint(str(ref_path))
    kz = kz_axis(grid_ref)
    order = np.argsort(kz)
    kz_sorted = kz[order]
    pos = kz_sorted > 0  # plot absolute kz > 0

    E_zp_ref = kz_spectrum(np.asarray(state_ref.z_plus))[order]
    E_zm_ref = kz_spectrum(np.asarray(state_ref.z_minus))[order]
    ax_zpm.loglog(kz_sorted[pos], E_zp_ref[pos] + E_zm_ref[pos],
                   "k-", lw=2, label=f"Alfvénic ref ({ALFVEN_REF_CKPT})")

    # Hermite runs
    for run in RUNS:
        ckpt_path = (
            DATA_DIR / run.label / "checkpoints"
            / f"checkpoint_{run.ckpt_tag}.h5"
        )
        if not ckpt_path.exists():
            print(f"  Missing: {ckpt_path}")
            continue
        state, grid, _ = load_checkpoint(str(ckpt_path))
        kz = kz_axis(grid)
        order = np.argsort(kz)
        kz_sorted = kz[order]
        pos = kz_sorted > 0

        z_plus = np.asarray(state.z_plus)
        z_minus = np.asarray(state.z_minus)
        g = np.asarray(state.g)  # (Nz, Ny, Nkx, M+1)
        M = state.M

        E_zpm = (kz_spectrum(z_plus) + kz_spectrum(z_minus))[order]
        E_g_total = kz_spectrum(g)[order]      # sum over Ny, Nkx, and m
        E_g_top = kz_spectrum(g[..., M])[order]

        ax_zpm.loglog(kz_sorted[pos], E_zpm[pos],
                       label=fr"$\nu={run.nu}$  {run.ckpt_tag}")
        ax_g.loglog(kz_sorted[pos], E_g_total[pos], ls="-",
                     label=fr"$\nu={run.nu}$  $\Sigma_m$")
        ax_g.loglog(kz_sorted[pos], E_g_top[pos], ls="--", alpha=0.7,
                     label=fr"$\nu={run.nu}$  $m{{=}}M$")

    for ax in (ax_zpm, ax_g):
        ax.set_xlabel(r"$k_z$")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    ax_zpm.set_ylabel(r"$E(k_z)$  (from $z^\pm$)")
    ax_zpm.set_title(r"$z^\pm$ parallel spectrum — pre-blowup vs. Alfvénic reference")
    ax_g.set_ylabel(r"$E(k_z)$  (from $g$)")
    ax_g.set_title(r"$g$ parallel spectrum — pre-blowup checkpoints")

    plt.tight_layout()
    out_zpm = FIG_DIR / "hermite_blowup_kz_spectrum_zpm.png"
    out_g = FIG_DIR / "hermite_blowup_kz_spectrum_g.png"
    # Save the combined figure under both names for the plan's expected paths.
    plt.savefig(out_zpm, dpi=150, bbox_inches="tight")
    plt.savefig(out_g, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_zpm}")
    print(f"Saved: {out_g}")
    plt.close()


# ---------------------------------------------------------------------------
# Figure 4: ε_ν(t) drift across runs on one panel
# ---------------------------------------------------------------------------
def plot_eps_nu_vs_t() -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for run in RUNS:
        spectra = load_spectra(run.label)
        if not spectra:
            continue
        t, W = stack_Wm_t(spectra)
        M = W.shape[1] - 1
        hermite = t >= HERMITE_START_T
        t = t[hermite]
        W = W[hermite]
        eps = np.array([compute_eps_nu(w, run.nu, M) for w in W])
        ax.semilogy(t - HERMITE_START_T, eps, label=fr"$\nu={run.nu}$")
        ax.axvline(run.blowup_t - HERMITE_START_T,
                    color="gray", ls=":", alpha=0.3)
    ax.set_xlabel(r"$t - t_0\;(\tau_A)$")
    ax.set_ylabel(r"$\varepsilon_\nu$")
    ax.set_title(r"Collisional dissipation rate vs. time (all blown-up runs)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    plt.tight_layout()
    out = FIG_DIR / "hermite_blowup_eps_nu_vs_t.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Summary text: blowup-time amplitudes
# ---------------------------------------------------------------------------
def print_blowup_summary() -> None:
    print("\n=== Blowup-time amplitudes ===")
    print(f"{'run':<24} {'nu':>5} {'t-t0(τA)':>10} {'W(M)':>12} {'ΣW':>12} {'ε_ν':>10}")
    for run in RUNS:
        spectra = load_spectra(run.label)
        if not spectra:
            continue
        t, W = stack_Wm_t(spectra)
        M = W.shape[1] - 1
        idx = int(np.argmin(np.abs(t - run.blowup_t)))
        # Prefer the spectrum just BEFORE blowup (W not yet NaN).
        while idx > 0 and (not np.all(np.isfinite(W[idx]))):
            idx -= 1
        eps = compute_eps_nu(W[idx], run.nu, M)
        print(f"{run.label:<24} {run.nu:>5} {t[idx]-HERMITE_START_T:>10.2f} "
              f"{W[idx, -1]:>12.3e} {W[idx].sum():>12.3e} {eps:>10.3e}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_Wmt_per_run()
    plot_Wm_threshold_comparison()
    plot_kz_spectra()
    plot_eps_nu_vs_t()
    print_blowup_summary()


if __name__ == "__main__":
    main()
