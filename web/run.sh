#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi
if [ ! -x .venv/bin/python ]; then
  echo 'Set up the Python environment first; see web/README.md.' >&2
  exit 1
fi
if [ ! -f frontend/dist/index.html ]; then
  echo 'Build the frontend first: npm --prefix frontend install && npm --prefix frontend run build' >&2
  exit 1
fi
exec .venv/bin/python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --no-access-log --timeout-graceful-shutdown 3
