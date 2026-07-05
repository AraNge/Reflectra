#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "[INFO] Project root: $PROJECT_ROOT"

mkdir -p \
  data/audio/musiccaps \
  data/audio/audioset \
  data/audio/mtg_jamendo \
  data/audio/song_describer \
  data/images/coco_captions \
  data/images/flickr30k \
  data/images/emoset \
  data/metadata \
  data/embeddings/audio \
  data/embeddings/image \
  data/embeddings/text \
  data/hf_cache \
  results \
  qdrant_storage

if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment..."
  python -m venv .venv
else
  echo "[INFO] Virtual environment already exists."
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

if grep -q "\[project.optional-dependencies\]" pyproject.toml; then
  echo "[INFO] Installing project in editable mode with dev extras..."
  pip install -e ".[dev]"
else
  echo "[INFO] Installing project in editable mode..."
  pip install -e .
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[WARN] Docker is not installed or not available in PATH."
  echo "[WARN] Install Docker Desktop / Docker Engine, then run:"
  echo "       docker run -d --name reflectra-qdrant -p 6333:6333 -p 6334:6334 -v \"$(pwd)/qdrant_storage:/qdrant/storage\" qdrant/qdrant:latest"
  exit 0
fi

echo "[INFO] Pulling Qdrant Docker image..."
docker pull qdrant/qdrant:latest

if docker ps -a --format '{{.Names}}' | grep -q '^reflectra-qdrant$'; then
  if docker ps --format '{{.Names}}' | grep -q '^reflectra-qdrant$'; then
    echo "[INFO] Qdrant container already running: reflectra-qdrant"
  else
    echo "[INFO] Starting existing Qdrant container..."
    docker start reflectra-qdrant
  fi
else
  echo "[INFO] Creating and starting Qdrant container..."
  docker run -d \
    --name reflectra-qdrant \
    -p 6333:6333 \
    -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage" \
    qdrant/qdrant:latest
fi

echo "[INFO] Waiting for Qdrant to become available..."
for i in {1..30}; do
  if python - <<'PY'
from qdrant_client import QdrantClient
try:
    client = QdrantClient(url="http://localhost:6333")
    client.get_collections()
    print("Qdrant is ready")
except Exception:
    raise SystemExit(1)
PY
  then
    echo "[INFO] Setup complete."
    echo "[INFO] Activate environment with: source .venv/bin/activate"
    echo "[INFO] Qdrant URL: http://localhost:6333"
    exit 0
  fi
  sleep 1
done

echo "[WARN] Qdrant did not respond within 30 seconds. Check Docker logs:"
echo "       docker logs reflectra-qdrant"