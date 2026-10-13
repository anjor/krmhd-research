"""Fetch the LaTeX source of Chandran, Mallet & Meyrand (2026), arXiv:2607.27981v1 (CMM).

The check scripts that read CMM's equation labels need the LaTeX source of the paper. This helper downloads the arXiv
e-print (a gzipped tar archive) into a new, empty folder, by default paper/phase-velocity-slices/data/cmm/ (git-ignored),
and extracts only its .tex, .cls and .bbl members, using tarfile's 'data' extraction filter (no absolute paths, no
members outside the folder, no links or device files). Nothing from the archive is executed. The archive is untrusted
data, so run the helper with python -I, from the repository root:

    uv run python -I paper/phase-velocity-slices/derivations/fetch_cmm.py [DEST]

Scripts that need the source call it as a separate process (never by import). The exit status is 0 if DEST holds the
source afterwards (downloaded now or earlier) and 1 otherwise.
"""
import hashlib
import io
import sys
import tarfile
import urllib.request
from pathlib import Path

ARXIV_ID = "2607.27981v1"
URL = f"https://arxiv.org/e-print/{ARXIV_ID}"
KEEP = (".tex", ".cls", ".bbl")
MAX_BYTES = 50 * 2**20          # the e-print is about 0.23 MB; refuse anything much larger
DEFAULT_DEST = Path(__file__).resolve().parent.parent / "data" / "cmm"


def has_source(folder: Path) -> bool:
    """True if `folder` holds at least one .tex file and one .cls file, i.e. an extracted e-print."""
    return folder.is_dir() and any(folder.rglob("*.tex")) and any(folder.rglob("*.cls"))


def show(path: Path) -> str:
    """`path` relative to the working directory when possible (keeps local absolute paths out of logs)."""
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def sha256(data: bytes) -> str:
    """Hex SHA-256 digest of `data` (printed for provenance)."""
    return hashlib.sha256(data).hexdigest()


def fetch(dest: Path) -> int:
    """Download the e-print into the empty folder `dest` and extract its .tex, .cls and .bbl members.

    Returns 0 on success and 1 on failure; on failure every file this call wrote is removed again.
    """
    if not hasattr(tarfile, "data_filter"):
        print("this Python's tarfile has no extraction filters (Python 3.12+ or a security backport is needed)")
        return 1
    created = not dest.exists()
    dest.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    try:
        request = urllib.request.Request(URL, headers={"User-Agent": "krmhd-research fetch_cmm.py"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise ValueError(f"download larger than {MAX_BYTES} bytes")
        archive = dest / f"arXiv-{ARXIV_ID}.e-print"
        archive.write_bytes(data)
        written.append(archive)
        print(f"downloaded {URL}: {len(data)} bytes, sha256 {sha256(data)}")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile() and Path(m.name).suffix.lower() in KEEP]
            written.extend(dest / m.name for m in members)
            tf.extractall(dest, members=members, filter="data")
        for m in members:
            path = dest / m.name
            print(f"  extracted {m.name}: {path.stat().st_size} bytes, sha256 {sha256(path.read_bytes())}")
        if not has_source(dest):
            raise ValueError("the archive holds no .tex and .cls files")
    except Exception as exc:  # network or HTTP error, not a tar archive, unsafe member, ...
        print(f"could not obtain the CMM source: {exc!r}")
        for path in written:
            if path.is_file():
                path.unlink()
        if created:
            for folder in sorted((p for p in dest.rglob("*") if p.is_dir()), reverse=True):
                if not any(folder.iterdir()):
                    folder.rmdir()
            if not any(dest.iterdir()):
                dest.rmdir()
        return 1
    return 0


def main() -> int:
    """Fetch into DEST (first argument) or the default folder unless the source is already there."""
    dest = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_DEST
    if has_source(dest):
        print(f"CMM source already present in {show(dest)}")
        return 0
    if dest.exists() and any(dest.iterdir()):
        print(f"{show(dest)} is not empty and holds no CMM source; remove it or choose another folder")
        return 1
    return fetch(dest)


if __name__ == "__main__":
    sys.exit(main())
