"""Shared helpers for the figure scripts of section 8 ("Checks with GANDALF"): the figure style (Journal of Plasma
Physics widths, 10 pt text, LaTeX rendering with a mathtext fallback, viridis for 2D data, the Okabe-Ito palette of
configs/figures.yaml for lines), saving PDF and PNG, Landau-parameter parsing and labels, named pass/fail checks, and
the cache of the computed (non-simulation) figures in data/computed/ (git-ignored).

The simulation figures read only the saved .npz files of their runs; the computed figures cache their numbers with a
hash of their config section and plot from the cache, so every script is re-runnable without recomputation.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent              # paper/phase-velocity-slices/analysis
PAPER = HERE.parent                                  # paper/phase-velocity-slices
REPO = PAPER.parent.parent
CONFIGS = PAPER / "configs"
FIGURES_YAML = CONFIGS / "figures.yaml"
FIG_DIR = PAPER / "figures"
DATA = PAPER / "data"
CACHE_DIR = DATA / "computed"


# ------------------------------------------------------------------------------------------------------------------
# configuration
# ------------------------------------------------------------------------------------------------------------------
def load_figures_config(section: str | None = None) -> tuple[dict, dict]:
    """(section of configs/figures.yaml, its shared style section); section None returns the whole file."""
    with open(FIGURES_YAML) as fh:
        cfg = yaml.safe_load(fh)
    return (cfg if section is None else cfg[section]), cfg["style"]


def style_config() -> dict:
    """The style section of configs/figures.yaml (widths, font sizes, palette, usetex, PNG resolution)."""
    return load_figures_config()[1]


def lam_value(x: Any) -> float:
    """Landau parameter Lambda from a config entry: a number or one of 'sqrt5', '-sqrt5', 'inf', '-inf'."""
    if isinstance(x, str):
        t = x.strip().lower()
        sign = -1.0 if t.startswith("-") else 1.0
        t = t.lstrip("+-")
        if t == "inf":
            return sign * np.inf
        if t.startswith("sqrt"):
            return sign * float(np.sqrt(float(t[4:].strip("()"))))
        return sign * float(t)
    return float(x)


def lam_tex(lam: float) -> str:
    """LaTeX form of a Landau parameter (\\sqrt{5}, -\\sqrt{5}, \\infty or the number)."""
    if np.isinf(lam):
        return r"\infty" if lam > 0 else r"-\infty"
    if abs(abs(lam) - np.sqrt(5.0)) < 1e-12:
        return r"\sqrt{5}" if lam > 0 else r"-\sqrt{5}"
    return f"{lam:g}"


def config_hash(section: dict) -> str:
    """Short hash of a config section; a cache is reused only if its hash matches."""
    return hashlib.sha256(json.dumps(section, sort_keys=True, default=str).encode()).hexdigest()[:16]


def environment() -> dict:
    """Versions, precision and hardware stored with cached numbers (local queries only)."""
    import scipy
    try:
        import krmhd
        kv = getattr(krmhd, "__version__", "unknown")
    except ImportError:  # pragma: no cover
        kv = "not installed"
    try:
        import jax
        jv = jax.__version__
    except ImportError:  # pragma: no cover
        jv = "not installed"
    cpu = ""
    if platform.system() == "Darwin":
        try:
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                                 timeout=5).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            cpu = ""
    cpu = cpu or platform.processor()
    return dict(krmhd=kv, jax=jv, numpy=np.__version__, scipy=scipy.__version__, python=platform.python_version(),
                precision="float64", hardware=f"{platform.system()} {platform.machine()}, {cpu}")


# ------------------------------------------------------------------------------------------------------------------
# checks and cache
# ------------------------------------------------------------------------------------------------------------------
class Checks:
    """Named pass/fail comparisons, printed as they are made and stored with the cached numbers."""

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


def cache_paths(name: str) -> tuple[Path, Path]:
    """data/computed/fig_<name>.npz and .json."""
    return CACHE_DIR / f"fig_{name}.npz", CACHE_DIR / f"fig_{name}.json"


def read_cache_meta(name: str) -> dict | None:
    """Metadata of the cached numbers of a computed figure, or None if there is no cache."""
    npz, js = cache_paths(name)
    if not (npz.exists() and js.exists()):
        return None
    with open(js) as fh:
        return json.load(fh)


def write_cache(name: str, arrays: dict | None, meta: dict) -> None:
    """Write the arrays (if given) and the metadata of a computed figure."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    npz, js = cache_paths(name)
    if arrays is not None:
        np.savez(npz, **arrays)
    with open(js, "w") as fh:
        json.dump(meta, fh, indent=1, default=float)
        fh.write("\n")


# ------------------------------------------------------------------------------------------------------------------
# style and output
# ------------------------------------------------------------------------------------------------------------------
def setup_style(style: dict, usetex: bool = True):
    """JPP style: 10 pt serif text (Computer Modern under LaTeX), inward ticks on all sides, small legends, viridis
    for images and the Okabe-Ito colour cycle of configs/figures.yaml.  Returns matplotlib.pyplot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycler import cycler
    fs = style["font_size_pt"]
    use = bool(usetex and style.get("usetex", True) and shutil.which("latex"))
    plt.rcParams.update({
        "text.usetex": use, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm", "font.size": fs, "axes.labelsize": fs, "xtick.labelsize": fs,
        "ytick.labelsize": fs, "legend.fontsize": style["legend_font_size_pt"], "axes.linewidth": 0.6,
        "lines.linewidth": 1.2, "lines.markersize": 4.0, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5, "legend.frameon": False,
        "legend.handlelength": 1.8, "savefig.pad_inches": 0.02, "image.cmap": "viridis",
        "axes.prop_cycle": cycler(color=style["palette"]),
        "text.latex.preamble": r"\usepackage{amsmath}"})
    return plt


def save(fig, name: str, style: dict) -> tuple[Path, Path]:
    """Save figures/<name>.pdf and figures/<name>.png (tight bounding box)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = FIG_DIR / f"{name}.pdf", FIG_DIR / f"{name}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=style["png_dpi"], bbox_inches="tight")
    return pdf, png


def render(draw, name: str, style: dict) -> str:
    """Call draw(plt) -> fig with LaTeX text and save it; if LaTeX fails, redraw with mathtext.  Returns the
    rendering used ("usetex" or "mathtext"), which the caller reports."""
    modes = (True, False) if style.get("usetex", True) else (False,)
    last: Exception | None = None
    for usetex in modes:
        plt = setup_style(style, usetex)
        fig = draw(plt)
        try:
            save(fig, name, style)
            plt.close(fig)
            return "usetex" if plt.rcParams["text.usetex"] else "mathtext"
        except (RuntimeError, OSError) as exc:      # LaTeX unavailable or failing: fall back to mathtext
            print(f"LaTeX rendering failed ({exc}); falling back to mathtext")
            plt.close(fig)
            last = exc
    raise RuntimeError(f"could not render {name}") from last


def panel_label(ax, text: str) -> None:
    """Panel label (a), (b), ... above the top-left corner of an axes."""
    ax.text(0.0, 1.025, text, transform=ax.transAxes, ha="left", va="bottom")
