#!/usr/bin/env bash
# Launch the full House Prices stack locally with Docker Compose.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found on PATH." >&2
  exit 1
fi

if [ ! -f models/baseline_xgb_pipeline.pkl ]; then
  echo "Frozen baseline model is missing. Restore models/baseline_xgb_pipeline.pkl from the repository before deploying." >&2
  exit 1
fi

echo "Exporting the frozen Phase 2 model to the native runtime artifact..."
if command -v conda >/dev/null 2>&1; then
  if conda env list | grep -qE '^sys-304\s'; then
    conda run -n sys-304 python -m model_training.export_native
  else
    python -m model_training.export_native
  fi
else
  python -m model_training.export_native
fi

echo "Building and starting services..."
docker compose up --build -d

echo
echo "Stack is up:"
echo "  Frontend UI : http://localhost:3000"
echo "  Backend API : http://localhost:8000"
echo "  Health check: http://localhost:8000/health"
echo
echo "Tail logs with: docker compose logs -f"
echo "Stop with      : docker compose down"
