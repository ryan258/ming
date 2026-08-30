#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"

# Prefer uv: `uv run --extra ui` syncs the UI dependency group before launching.
if command -v uv >/dev/null 2>&1; then
  if [ "${1:-}" = "--ui" ]; then
    shift
    exec uv run --python 3.12 --extra ui streamlit run ui/app.py "$@"
  fi
  exec uv run --python 3.12 --no-sync python -m ming "$@"
fi

# No uv: use a >=3.12 interpreter, preferring the project venv where deps live.
if [ -x .venv/bin/python ]; then
  PY=.venv/bin/python
elif python3 -c 'import sys; raise SystemExit(sys.version_info < (3, 12))' 2>/dev/null; then
  PY=python3
else
  echo "Ming requires Python 3.12+. Install it or install uv: https://docs.astral.sh/uv/" >&2
  exit 2
fi

if [ "${1:-}" = "--ui" ]; then
  shift
  exec "$PY" -m streamlit run ui/app.py "$@"
fi
exec "$PY" -m ming "$@"
