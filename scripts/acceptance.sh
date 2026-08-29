#!/usr/bin/env bash
# The acceptance run. Everything here exists because something passed once and
# was not actually verified.
#
#   --no-skips   a skip is not a pass. `319 passed` and `319 passed, 7 skipped`
#                read identically to anyone scanning, and the second is missing
#                an acceptance criterion.
#   run twice    tests that pass once and fail on rerun are a real class -- the
#                restart tests did exactly that, resuming the first run's
#                conversation and seeing six turns instead of three.
#   host + container
#                a platform-specific fix was itself platform-asymmetric once,
#                passing on Windows and breaking collection on Linux. Running
#                both is what caught it.
#
# Usage:  POSTGRES_HOST_PORT=5433 scripts/acceptance.sh
set -euo pipefail

PY="${PY:-.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="${PY_FALLBACK:-.venv/bin/python}"

echo "=== ruff ==========================================================="
"$PY" -m ruff check .

echo "=== host, pass 1 ==================================================="
"$PY" -m pytest -q --no-skips

echo "=== host, pass 2 (idempotence) ====================================="
"$PY" -m pytest -q --no-skips

echo "=== golden set ====================================================="
"$PY" -m evals.validate --strict

echo "=== judge sets ====================================================="
"$PY" -m evals.validate judge || true   # INCONCLUSIVE while unlabelled is expected

echo "=== container ======================================================"
# --build, not a bare `up -d`. A stale image passes the host stage and then
# fails collection in the container on an import error, which reads like a
# platform bug and is really a dependency added since the last build.
docker compose up -d --build --wait
docker compose exec -T app python -m pytest -q --no-skips
docker compose exec -T app python -m app.cli doctor --no-llm

echo
echo "acceptance run complete"
