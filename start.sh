#!/usr/bin/env bash
# Packout insert builder - one-step launcher for macOS and Linux.
#   First run: creates a private Python environment in .venv and installs the dependencies (a few minutes).
#   Every run: starts the app and opens it in your browser at http://localhost:8501
# Needs Python 3.9 - 3.12 (https://www.python.org/downloads/).  Set PORT=1234 to use another port.
set -e
cd "$(dirname "$0")"
PORT="${PORT:-8501}"
URL="http://localhost:$PORT"

pick_python() {
  for c in python3.11 python3.12 python3.10 python3.9 python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
      echo "$c"; return 0
    fi
  done
  return 1
}

if [ ! -x .venv/bin/python ]; then
  PY="$(pick_python)" || {
    echo "Could not find Python 3.9 - 3.12. Install Python 3.11 from https://www.python.org/downloads/ and run this again."
    exit 1
  }
  echo "Setting up (first run only) with $($PY -V) ..."
  "$PY" -m venv .venv
fi

# (Re)install when requirements.txt changed since the last install
if ! cmp -s requirements.txt .venv/requirements.installed.txt 2>/dev/null; then
  echo "Installing dependencies - this takes a few minutes the first time ..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null
  .venv/bin/python -m pip install -r requirements.txt
  cp requirements.txt .venv/requirements.installed.txt
fi

mkdir -p static outputs                           # folders the app serves/writes to

if [ -z "$NO_BROWSER" ]; then                       # open the browser once the server answers
  ( for _ in $(seq 1 90); do
      if curl -fs "$URL/_stcore/health" >/dev/null 2>&1; then .venv/bin/python -m webbrowser "$URL" >/dev/null 2>&1; break; fi
      sleep 1
    done ) &
fi

echo "Packout builder is starting at $URL  (press Ctrl+C or close this window to stop)"
exec .venv/bin/python -m streamlit run app.py --server.port "$PORT"
