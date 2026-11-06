#!/bin/bash
# setup_env.sh: build REMOTE_ROOT/venv, the Python environment of the run code, on the cluster's login node.  Run it
# once, and again after requirements.txt changes (it is idempotent); submit.sh --setup copies the files and runs it.
# JAX comes from PyPI with its CUDA wheels (JAX_CUDA in the site file: cuda12 needs an NVIDIA driver >= 525, cuda13
# needs >= 580), GANDALF from its tag on GitHub, and NumPy and SciPy at the versions of the earlier GPU runs
# (requirements.txt).  The login node may have no GPU; job.sh checks the GPU at the start of every job.
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
. "$HERE/common.sh"
SITE_ENV=${SITE_ENV:-$HERE/../../../site.env}
p1s_load_site "$SITE_ENV"
p1s_load_modules
case $PYTHON in
  uv:*)
    # PYTHON=uv:3.11: a CPython that uv downloads, for systems whose python3 is older than 3.11 (Ubuntu 22.04 has
    # 3.10).  uv itself is installed with the system's pip into ~/.local/bin if it is not on the PATH.
    uv=$(command -v uv || echo "$HOME/.local/bin/uv")
    if [ ! -x "$uv" ]; then
      python3 -m pip install --user --quiet uv || p1s_die "could not install uv (python3 -m pip install --user uv)"
    fi
    [ -x "$uv" ] || p1s_die "uv not found at $uv after installing it"
    [ -d "$REMOTE_ROOT/venv" ] || "$uv" venv --seed --python "${PYTHON#uv:}" "$REMOTE_ROOT/venv" ;;
  *)
    "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
      || p1s_die "$PYTHON is $("$PYTHON" -V 2>&1); the run code needs Python 3.11 or newer (set PYTHON, e.g. uv:3.11)"
    [ -d "$REMOTE_ROOT/venv" ] || "$PYTHON" -m venv "$REMOTE_ROOT/venv" ;;
esac
set +u
. "$REMOTE_ROOT/venv/bin/activate"
set -u
jaxv=$(sed -n 's/^jax==\([0-9.]*\).*/\1/p' "$HERE/requirements.txt")
[ -n "$jaxv" ] || p1s_die "no jax==<version> line in $HERE/requirements.txt"
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r "$HERE/requirements.txt" "jax[$JAX_CUDA]==$jaxv"
python - <<'EOF'
import sys

import jax
import krmhd
import numpy
import scipy

print(f"python {sys.version.split()[0]}, jax {jax.__version__}, numpy {numpy.__version__}, scipy {scipy.__version__}, "
      f"GANDALF (krmhd) {getattr(krmhd, '__version__', 'unknown')}")
EOF
echo "environment ready: $REMOTE_ROOT/venv"
