#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
INSTALL_SYSTEM_DEPS=0
RUN_TESTS=0
PYTORCH_FLAVOR="${PYTORCH_FLAVOR:-cpu}"

usage() {
  cat <<'EOF'
Usage: scripts/setup_fedora_python.sh [options]

Creates a Python virtual environment and installs the Python packages needed by
the AMD backend, training script, tests, and Streamlit heatmap dashboard.

Options:
  --system-deps    Install Fedora packages with sudo dnf first.
  --venv PATH      Virtual environment directory. Default: .venv
  --python PATH    Python executable to use. Default: python3.12, python3.13,
                   python3.11, then python3.
  --run-tests      Run backend tests after installation.
  --cuda           Install the default PyTorch wheel, which may include CUDA
                   runtime packages on Linux. Default: CPU-only PyTorch.
  -h, --help       Show this help.

Examples:
  scripts/setup_fedora_python.sh --system-deps
  scripts/setup_fedora_python.sh --python python3.12 --run-tests
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system-deps)
      INSTALL_SYSTEM_DEPS=1
      shift
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --run-tests)
      RUN_TESTS=1
      shift
      ;;
    --cuda)
      PYTORCH_FLAVOR="cuda"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$INSTALL_SYSTEM_DEPS" -eq 1 ]]; then
  sudo dnf install -y \
    python3 python3-pip python3-devel python3-virtualenv \
    gcc gcc-c++ make cmake pkgconf-pkg-config \
    qt5-qtbase-devel \
    libglvnd-glx libjpeg-turbo-devel zlib-devel
fi

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in python3.12 python3.13 python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "No Python 3 executable found." >&2
  exit 1
fi

PY_VERSION="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"

case "$PY_VERSION" in
  3.10|3.11|3.12|3.13)
    ;;
  *)
    cat >&2 <<EOF
Selected Python is $PY_VERSION.

This project uses PyTorch, torchvision, OpenCV, timm, and grad-cam. Prebuilt
ML wheels are often unavailable for the newest Python releases. On Fedora,
install Python 3.12 or 3.13 and rerun:

  PYTHON_BIN=python3.12 scripts/setup_fedora_python.sh

You can still force a different interpreter with --python, but installation may fail.
EOF
    ;;
esac

echo "Project root: $ROOT_DIR"
echo "Python: $PYTHON_BIN ($PY_VERSION)"
echo "Virtualenv: $VENV_DIR"

"$PYTHON_BIN" -m venv "$VENV_DIR"

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ "$PYTORCH_FLAVOR" == "cpu" ]]; then
  echo "Installing CPU-only PyTorch wheels."
  python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
else
  echo "Installing default PyTorch wheels."
  python -m pip install torch torchvision
fi

python -m pip install -r "$ROOT_DIR/requirements.txt"

if [[ -f "$ROOT_DIR/scripts/heatmap_dashboard/requirements_dashboard.txt" ]]; then
  python -m pip install -r "$ROOT_DIR/scripts/heatmap_dashboard/requirements_dashboard.txt"
fi

python -m pip check

python - <<'PY'
import importlib

modules = [
    "cv2",
    "flask",
    "flask_cors",
    "matplotlib",
    "numpy",
    "pandas",
    "pydicom",
    "PIL",
    "pytorch_grad_cam",
    "captum",
    "requests",
    "sklearn",
    "streamlit",
    "timm",
    "torch",
    "torchvision",
    "tqdm",
]

missing = []
for module in modules:
    try:
        importlib.import_module(module)
    except Exception as exc:
        missing.append((module, exc))

if missing:
    print("Missing or broken Python modules:")
    for module, exc in missing:
        print(f"  - {module}: {exc}")
    raise SystemExit(1)

print("Python dependency import check passed.")
PY

if [[ "$RUN_TESTS" -eq 1 ]]; then
  (cd "$ROOT_DIR" && python -m pytest backend/tests)
fi

cat <<EOF

Setup complete.

Activate the environment:
  source "$VENV_DIR/bin/activate"

Run the backend:
  python -m backend

Run the heatmap dashboard:
  streamlit run "$ROOT_DIR/scripts/heatmap_dashboard/gradcam_dashboard.py"
EOF
