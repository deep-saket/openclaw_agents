#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_PATH="${BASH_SOURCE[0]}"
else
  SCRIPT_PATH="$0"
fi

ROOT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
cd "${ROOT_DIR}"

echo "[setup] repo root: ${ROOT_DIR}"

REQUIRED_PYTHON_MAJOR=3
REQUIRED_PYTHON_MINOR=11
REQUIRED_PYTHON_LINE="${REQUIRED_PYTHON_MAJOR}.${REQUIRED_PYTHON_MINOR}"

run_with_sudo() {
  if [[ "${EUID:-$(id -u)}" -eq 0 ]]; then
    "$@"
    return
  fi
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  echo "[error] Need elevated privileges for: $*" >&2
  echo "[error] Re-run this script with an account that has sudo/root access." >&2
  exit 1
}

install_python_if_missing() {
  if command -v python3.11 >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || command -v python >/dev/null 2>&1; then
    return
  fi

  echo "[setup] Python not found. Attempting to install Python ${REQUIRED_PYTHON_LINE}.x ..."
  OS_NAME="$(uname -s | tr '[:upper:]' '[:lower:]')"

  if [[ "${OS_NAME}" == "darwin" ]]; then
    if command -v brew >/dev/null 2>&1; then
      brew install python@3.11
      return
    fi
    echo "[error] Homebrew not found. Install Homebrew first: https://brew.sh/" >&2
    exit 1
  fi

  if [[ "${OS_NAME}" == "linux" ]]; then
    if command -v apt-get >/dev/null 2>&1; then
      run_with_sudo apt-get update
      run_with_sudo apt-get install -y python3.11 python3.11-venv python3-pip
      return
    fi
    if command -v dnf >/dev/null 2>&1; then
      run_with_sudo dnf install -y python3.11 python3.11-pip
      return
    fi
    if command -v yum >/dev/null 2>&1; then
      run_with_sudo yum install -y python3.11 python3.11-pip
      return
    fi
    if command -v pacman >/dev/null 2>&1; then
      run_with_sudo pacman -Sy --noconfirm python python-pip
      return
    fi
  fi

  echo "[error] Could not auto-install Python ${REQUIRED_PYTHON_LINE}.x on this platform." >&2
  echo "[error] Please install Python ${REQUIRED_PYTHON_LINE}.x manually, then re-run this script." >&2
  exit 1
}

validate_python_line() {
  local cmd="$1"
  local ver
  ver="$("${cmd}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || true)"
  [[ "${ver}" == "${REQUIRED_PYTHON_LINE}" ]]
}

install_python_if_missing

PYTHON_CMD=""
if command -v python3.11 >/dev/null 2>&1; then
  PYTHON_CMD="python3.11"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD="python"
else
  echo "[error] Python not found. Install Python ${REQUIRED_PYTHON_LINE}.x and retry." >&2
  exit 1
fi

if ! validate_python_line "${PYTHON_CMD}"; then
  echo "[setup] Found ${PYTHON_CMD}, but it is not Python ${REQUIRED_PYTHON_LINE}.x. Trying to install required line..."
  install_python_if_missing
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_CMD="python3.11"
  fi
fi

if ! validate_python_line "${PYTHON_CMD}"; then
  echo "[error] Required Python line is ${REQUIRED_PYTHON_LINE}.x, but current is incompatible." >&2
  echo "[error] Install Python ${REQUIRED_PYTHON_LINE}.x and rerun this script." >&2
  exit 1
fi

echo "[setup] using python: ${PYTHON_CMD}"

if [[ -d ".venv" ]]; then
  echo "[setup] deleting existing virtual environment at .venv"
  rm -rf .venv
fi

echo "[setup] creating fresh virtual environment at .venv"
"${PYTHON_CMD}" -m venv .venv

if [[ -x ".venv/bin/python" ]]; then
  VENV_PY=".venv/bin/python"
elif [[ -x ".venv/Scripts/python.exe" ]]; then
  VENV_PY=".venv/Scripts/python.exe"
else
  echo "[error] Could not find venv python executable." >&2
  exit 1
fi

echo "[setup] upgrading pip"
"${VENV_PY}" -m pip install --upgrade pip

if [[ -f "requirements.txt" ]]; then
  echo "[setup] installing requirements.txt"
  "${VENV_PY}" -m pip install -r requirements.txt
fi

echo "[setup] installing project in editable mode"
"${VENV_PY}" -m pip install -e .

if [[ ! -f ".env" ]]; then
  echo "[setup] creating .env from .env.example"
  if [[ -f ".env.example" ]]; then
    cp ".env.example" ".env"
  else
    : > ".env"
  fi
fi

if ! grep -q "^NVIDIA_API_KEY=" .env; then
  echo "NVIDIA_API_KEY=" >> .env
fi
if ! grep -q "^NVIDIA_BASE_URL=" .env; then
  echo "NVIDIA_BASE_URL=https://integrate.api.nvidia.com" >> .env
fi
if ! grep -q "^OPENAI_API_KEY=" .env; then
  echo "OPENAI_API_KEY=" >> .env
fi

echo
echo "[done] Collection Agent setup complete."
echo
echo "Next steps:"
echo "1) Add your API key(s) to ${ROOT_DIR}/.env"
echo "2) Activate venv:"
if [[ -x ".venv/bin/activate" ]]; then
  echo "   source .venv/bin/activate"
else
  echo "   source .venv/Scripts/activate"
fi
echo "3) Run UI:"
echo "   python -m agents.collection_agent.ui.server"
echo "4) Open:"
echo "   http://127.0.0.1:8060/"
