# CUDA "devel" image (not "runtime") is required: pykeops JIT-compiles CUDA
# kernels at first use, which needs nvcc present at runtime, not just the
# CUDA runtime libraries.
#
# Match the tag (12.4.1 here) to a CUDA version your torch build supports --
# check locally with `python -c "import torch; print(torch.version.cuda)"`
# and adjust both this tag and the torch install below to match.
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# System dependencies pykeops needs to compile its formulas: a C++ compiler
# (build-essential) and cmake. python3.11 + venv for a clean interpreter;
# curl/git are just convenience for future debugging inside the container.
# libgl1/libglib2.0-0/libxcb1/libxrender1/libsm6/libxext6 are needed by
# opencv-python (cv2) at import time -- pip/uv installing the Python
# package does not pull in these OS-level shared libraries, only apt does.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    python3.11 \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libxrender1 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (matches your local workflow)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /workspace

# Copy dependency manifests first so Docker can cache this layer and skip
# re-resolving/downloading packages on every rebuild when only your .py
# files change.
COPY pyproject.toml uv.lock* ./

# Pin uv to Python 3.12: torch==2.5.1 only ships wheels for cp312/cp313, and
# uv's default (downloading the newest available interpreter) grabbed 3.14
# here, which has no matching torch wheel. uv will download and manage this
# Python build itself -- the apt-installed python3.11 above is not what's
# used for the project venv.
# --frozen fails if the lockfile is out of date rather than silently
# re-resolving; drop --frozen for the first run if you haven't committed a
# lockfile yet, then re-add it once you have.
RUN uv sync --frozen --python 3.12 || uv sync --python 3.12

# Now copy the rest of the project (this layer rebuilds on every code change,
# but the dependency layer above stays cached).
COPY . .

# Default to an interactive shell; override with a specific command at
# `docker run` time (e.g. `uv run python evaluation/evaluate_paper_protocol.py --help`).
CMD ["bash"]