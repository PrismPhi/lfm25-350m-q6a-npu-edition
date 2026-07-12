#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${LFM2_5_PYTHON:-}" ]]; then
  PYTHON="${LFM2_5_PYTHON}"
elif [[ -x "${HOME}/lfm2.5-qnn-venv/bin/python" ]]; then
  PYTHON="${HOME}/lfm2.5-qnn-venv/bin/python"
else
  PYTHON="python3"
fi

if [[ "${1:-}" == "--python" ]]; then
  [[ -n "${2:-}" ]] || { echo "[preflight] --python requires a path" >&2; exit 2; }
  PYTHON="$2"
  shift 2
fi

command -v "$PYTHON" >/dev/null 2>&1 || {
  echo "[preflight] Python not found: $PYTHON" >&2
  echo "[preflight] Set LFM2_5_PYTHON or pass --python /path/to/qnn-venv/bin/python." >&2
  exit 2
}

exec "$PYTHON" "$SCRIPT_DIR/scripts/install.py" "$@"
