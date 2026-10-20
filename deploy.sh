#!/usr/bin/env bash
# Launch the persistent local House Prices prediction and monitoring stack.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required but was not found on PATH." >&2
  exit 1
fi

if ! docker info >/dev/null 2>&1; then
  echo "Docker's Linux container engine is unavailable. Start Docker Desktop or the Linux Docker service, then retry." >&2
  exit 1
fi

echo "Building and starting the stack. Compose runs the idempotent registry bootstrap before the API."
echo "The ADMIN_TOKEN fallback is only for a local demo; set ADMIN_TOKEN in the shell or ignored .env for a private value."
docker compose up --build --detach --wait

echo
echo "Stack is ready:"
echo "  Prediction form      : http://localhost:3000"
echo "  Operations dashboard : http://localhost:3000/dashboard.html"
echo "  Backend health       : http://localhost:8000/health"
echo
echo "Tail logs with: docker compose logs -f"
echo "Stop with      : docker compose down"