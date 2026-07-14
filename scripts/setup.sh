#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
ACTION="${1:-start}"
if [[ "$ACTION" == "--remove" ]]; then
  ACTION="remove"
fi

echo "[INFO] Project root: $PROJECT_ROOT"

if [[ "$ACTION" == "stop" || "$ACTION" == "remove" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "[WARN] Docker is not installed or not available in PATH."
  else
    for container in reflectra-qdrant reflectra-jaeger; do
      if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "[INFO] Stopping ${container}..."
        docker stop "${container}" >/dev/null
      elif docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "[INFO] ${container} already stopped."
      else
        echo "[INFO] ${container} does not exist."
      fi

      if [[ "$ACTION" == "remove" ]] && docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "[INFO] Removing ${container}..."
        docker rm "${container}" >/dev/null
      fi
    done
  fi

  if [[ "$ACTION" == "remove" ]]; then
    echo "[INFO] Removing Qdrant storage..."
    rm -rf qdrant_storage
    mkdir -p qdrant_storage
    echo "[INFO] Reflectra containers and Qdrant data removed."
    exit 0
  fi

  echo "[INFO] Reflectra services stopped."
  exit 0
fi

if [[ "$ACTION" != "start" ]]; then
  echo "Usage: scripts/setup.sh [start|stop|remove|--remove]"
  exit 2
fi

mkdir -p \
  data/music \
  plots \
  qdrant_storage


if [ ! -d ".venv" ]; then
  echo "[INFO] Creating virtual environment..."
  python -m venv .venv
else
  echo "[INFO] Virtual environment already exists."
fi

source .venv/bin/activate

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

echo "[INFO] Pulling Jaeger Docker image..."
docker pull jaegertracing/all-in-one:1.21

if docker ps -a --format '{{.Names}}' | grep -q '^reflectra-jaeger$'; then
  if docker ps --format '{{.Names}}' | grep -q '^reflectra-jaeger$'; then
    echo "[INFO] Jaeger container already running: reflectra-jaeger"
  else
    echo "[INFO] Starting existing Jaeger container..."
    docker start reflectra-jaeger
  fi
else
  echo "[INFO] Creating and starting Jaeger container..."
  docker run -d \
    --name reflectra-jaeger \
    -p 16686:16686 \
    -p 6831:6831/udp \
    jaegertracing/all-in-one:1.21
fi

echo "[INFO] Waiting for Qdrant to become available..."
for i in {1..30}; do
  if python - <<'PY'
import os
import importlib
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
_import_module = importlib.import_module
def _protobuf_safe_import(name, package=None):
    if name == "google._upb._message":
        raise ImportError(name)
    return _import_module(name, package)
importlib.import_module = _protobuf_safe_import
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
    echo "[INFO] Jaeger UI: http://127.0.0.1:16686"
    echo "[INFO] Jaeger agent: localhost:6831"
    echo "[INFO] Put music files in: data/music"
    echo "[INFO] Index music with: python -m src.vector_db.index_clap_audio_qdrant --music-dir data/music"
    exit 0
  fi
  sleep 1
done

echo "[WARN] Qdrant did not respond within 30 seconds. Check Docker logs:"
echo "       docker logs reflectra-qdrant"
