"""Check a paper folder against the mechanical style rules in CLAUDE.md.

Usage: uv run python paper/check_style.py paper/<name>

It reads every .tex file in the folder except sections/parked/. It flags
theorem-style environments anywhere, and terms the style rules keep out of
the main text. Appendix files (named like A-branches.tex) may use the terms.
It exits with 1 if it found anything, so it can gate a build.
"""

import re
import sys
from pathlib import Path

ENVIRONMENTS = re.compile(
    r"\\begin\{(theorem|proposition|corollary|lemma|proof|definition)\*?\}|\\newtheorem"
)
TERMS: dict[str, re.Pattern[str]] = {
    "Borel": re.compile(r"\bBorel\b"),
    "diffeomorphism": re.compile(r"diffeomorph", re.IGNORECASE),
    "self-adjoint": re.compile(r"self-adjoint", re.IGNORECASE),
    "L^2": re.compile(r"(?<![A-Za-z_\\])L\^\{?2\}?|L\u00b2"),
    "spectral projection": re.compile(r"spectral projection", re.IGNORECASE),
    "Ito": re.compile(r"\bIt\\\^\{?o\}?|\bIt\u00f4\b|\bIto\b"),
    "classical solution": re.compile(r"classical solution", re.IGNORECASE),
}
COMMENT = re.compile(r"(?<!\\)%.*$")


def is_appendix(path: Path) -> bool:
    """Return True for appendix sections, which the repo names like A-branches.tex."""
    return re.match(r"^[A-Z]-", path.name) is not None


def check(folder: Path) -> list[str]:
    """Return one line per style problem found in the folder's .tex files."""
    flags: list[str] = []
    for path in sorted(folder.rglob("*.tex")):
        if "parked" in path.parts:
            continue
        for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = COMMENT.sub("", raw)
            where = f"{path.relative_to(folder)}:{n}"
            if ENVIRONMENTS.search(line):
                flags.append(f"{where}: theorem-style environment")
            if is_appendix(path):
                continue
            for term, pattern in TERMS.items():
                if pattern.search(line):
                    flags.append(f"{where}: '{term}' in the main text")
    return flags


def main(argv: list[str]) -> int:
    """Print the flags for the paper folder in argv[1] and return the exit code."""
    if len(argv) != 2:
        print(__doc__)
        return 2
    folder = Path(argv[1])
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        return 2
    flags = check(folder)
    print("\n".join(flags) if flags else "No style flags.")
    return 1 if flags else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
