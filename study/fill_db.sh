#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONTAINER_NAME="${QDRANT_CONTAINER_NAME:-reflectra-qdrant}"
QDRANT_IMAGE="${QDRANT_IMAGE:-qdrant/qdrant:latest}"
QDRANT_STORAGE_DIR="${QDRANT_STORAGE_DIR:-$PROJECT_ROOT/qdrant_storage}"
VECTOR_DB_DIR="${VECTOR_DB_DIR:-$PROJECT_ROOT/data/vector_db}"
QDRANT_URL="${QDRANT_URL:-http://localhost:6333}"

echo "[INFO] Project root: $PROJECT_ROOT"
echo "[INFO] Qdrant container: $CONTAINER_NAME"
echo "[INFO] Qdrant URL: $QDRANT_URL"

mkdir -p "$QDRANT_STORAGE_DIR" "$VECTOR_DB_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] Docker is not installed or not available in PATH." >&2
  exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
  if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "[INFO] Qdrant container is already running."
  else
    echo "[INFO] Starting existing Qdrant container..."
    docker start "$CONTAINER_NAME" >/dev/null
  fi
else
  echo "[INFO] Pulling Qdrant image..."
  docker pull "$QDRANT_IMAGE"

  echo "[INFO] Creating Qdrant container..."
  docker run -d \
    --name "$CONTAINER_NAME" \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$QDRANT_STORAGE_DIR:/qdrant/storage" \
    "$QDRANT_IMAGE" >/dev/null
fi

if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

echo "[INFO] Waiting for Qdrant..."
QDRANT_READY=0
for _ in {1..60}; do
  if QDRANT_URL="$QDRANT_URL" python -c 'import os; from qdrant_client import QdrantClient; QdrantClient(url=os.environ["QDRANT_URL"]).get_collections()' >/dev/null 2>&1; then
    echo "[INFO] Qdrant is ready."
    QDRANT_READY=1
    break
  fi
  sleep 1
done

if [ "$QDRANT_READY" -ne 1 ]; then
  echo "[ERROR] Qdrant did not become ready within 60 seconds." >&2
  echo "[ERROR] Check logs with: docker logs $CONTAINER_NAME" >&2
  exit 1
fi

python -m study.fill_clap_audio_qdrant \
  --qdrant-url "$QDRANT_URL" \
  "$@"

COPY_DEST="$VECTOR_DB_DIR/qdrant_storage"
mkdir -p "$COPY_DEST"

echo "[INFO] Flushing Qdrant storage..."
docker exec "$CONTAINER_NAME" sh -c "sync" >/dev/null

echo "[INFO] Copying Qdrant storage to: $COPY_DEST"
docker cp "$CONTAINER_NAME:/qdrant/storage/." "$COPY_DEST"

echo "[INFO] Done. Copy is available at: $COPY_DEST"
