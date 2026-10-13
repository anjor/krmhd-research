"""c00_cmm_numbering: numbers the displayed equations of Appendix B of CMM (Chandran, Mallet & Meyrand 2026,
arXiv:2607.27981v1) from its LaTeX source, maps their LaTeX labels to the numbers (B1), (B2), ..., and checks that the
CMM equation numbers quoted in the paper and in the truncated-hierarchy scripts, (B4)-(B7), (B13), (B15)-(B18),
(B23)-(B25), (B31) and (B32), point at the intended equations. Supports the CMM equation numbers cited in Sections 2, 4
and 8 of the paper.

The LaTeX source is untrusted downloaded data. It is fetched by the shared helper ../../fetch_cmm.py, run as a separate
process, into paper/phase-velocity-slices/data/cmm/ (git-ignored), and this script only reads it as text, so it must be
run with python -I. If the source cannot be obtained, the script prints SKIPPED and exits 0. The path of CMM's .tex
file may be given as the first argument instead.

Numbering rule: \\begin{equation} blocks and the rows of align and eqnarray environments (rows end with \\\\ unless they
carry \\nonumber or \\notag) are counted in order; starred environments are skipped.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent                  # .../derivations/review/truncated
FETCH = HERE.parents[1] / "fetch_cmm.py"                # .../derivations/fetch_cmm.py
CMM_DIR = HERE.parents[2] / "data" / "cmm"              # .../phase-velocity-slices/data/cmm (git-ignored)
HEADING = r"\section{Additional quadratic invariants in KRMHD}"
# reference values: CMM's LaTeX label for each Appendix B equation number quoted in the paper
EXPECTED = {4: "eq:dGpmdt", 5: "eq:GpGm", 6: "eq:g09sigma", 7: "eq:Lambdapm", 13: "eq:Wpmcompr",
            15: "eq:Wcomprtotal", 16: "eq:defkappa", 17: "eq:Hentrtotal", 18: "eq:Hpm", 23: "eq:dM0dt",
            24: "eq:dM1dt", 25: "eq:Gammapm", 31: "eq:WcomprHerm", 32: "eq:GammapmHerm"}
N_EQUATIONS = 32                                        # numbered equations in Appendix B of v1 (reference value)

FAILED: list[str] = []
N_CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    """Print one PASS/FAIL line and remember failures."""
    global N_CHECKS
    N_CHECKS += 1
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" :: {detail}" if detail else ""), flush=True)
    if not ok:
        FAILED.append(label)


def show(path: Path) -> str:
    """`path` relative to the working directory when possible (keeps local absolute paths out of the output)."""
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return path.name


def appendix_tex() -> Path | None:
    """The .tex file under CMM_DIR that contains CMM's Appendix B, if any."""
    if not CMM_DIR.is_dir():
        return None
    for path in sorted(CMM_DIR.rglob("*.tex")):
        if path.is_file() and HEADING in path.read_text(encoding="utf-8", errors="replace"):
            return path
    return None


def find_source() -> Path | None:
    """CMM's .tex file: the first argument, else the downloaded source (fetched by ../../fetch_cmm.py if missing).

    The fetch helper runs as a separate process in isolated mode; its output is printed indented."""
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        return path if path.is_file() else None
    path = appendix_tex()
    if path is None and FETCH.is_file():
        print(f"   CMM source not found in {show(CMM_DIR)}; running {FETCH.name}", flush=True)
        try:
            res = subprocess.run([sys.executable, "-I", str(FETCH), str(CMM_DIR)], capture_output=True, text=True,
                                 timeout=600, check=False)
            for line in (res.stdout + res.stderr).splitlines():
                print("     " + line)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"     fetch failed: {exc!r}")
        path = appendix_tex()
    return path


def number_equations(src: str) -> list[tuple[int, list[str]]]:
    """(n, labels) of the numbered displayed equations of Appendix B, in order (rule in the module docstring)."""
    start = src.index(HEADING)
    end = src.find(r"\section", start + 10)
    end = len(src) if end < 0 else end
    app = src[start:end]
    app = re.sub(r"(?<!\\)%.*", "", app)               # strip comments
    n = 0
    out = []
    for m in re.finditer(r"\\begin\{(equation|align|eqnarray)(\*?)\}(.*?)\\end\{\1\2\}", app, re.S):
        env, star, body = m.group(1), m.group(2), m.group(3)
        if star:
            continue
        if env == "equation":
            n += 1
            out.append((n, re.findall(r"\\label\{([^}]*)\}", body)))
        else:
            for r in re.split(r"\\\\", body):
                if r.strip() == "":
                    continue
                if r"\nonumber" in r or r"\notag" in r:
                    continue
                n += 1
                out.append((n, re.findall(r"\\label\{([^}]*)\}", r)))
    return out


def main() -> int:
    path = find_source()
    if path is None:
        print("SKIPPED: the LaTeX source of CMM (arXiv:2607.27981v1) is not available (run "
              "paper/phase-velocity-slices/derivations/fetch_cmm.py with network access, or pass the path of CMM's "
              ".tex file)", flush=True)
        return 0
    src = path.read_text(encoding="utf-8", errors="replace")
    print(f"   source: {show(path)}", flush=True)
    check("Appendix B (Additional quadratic invariants in KRMHD) found in the source", HEADING in src)
    if HEADING in src:
        eqs = number_equations(src)
        for k, lab in eqs:
            print(f"(B{k}) {lab}")
        labels = dict(eqs)
        check(f"Appendix B has {N_EQUATIONS} numbered equations", len(eqs) == N_EQUATIONS, f"{len(eqs)}")
        for k, lab in EXPECTED.items():
            check(f"(B{k}) is {lab}", labels.get(k) == [lab], f"{labels.get(k)}")
    print(f"== c00_cmm_numbering: {N_CHECKS - len(FAILED)}/{N_CHECKS} checks passed", flush=True)
    if FAILED:
        print("FAILED CHECKS:", *FAILED, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
