#!/usr/bin/env bash
set -euo pipefail

N=100
MAX_AUDIOS=6
NUM_SHARDS=1
SHARD_INDEX=0
# Defaults for create_clap_benchmark.
BATCH_SIZE=4
QUERIES_PER_AUDIO=1
AUDIO_CLIP_SECONDS=15  # seconds
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SONG_DESCRIBER_METADATA="$PROJECT_ROOT/data/metadata/song_describer_metadata.jsonl"
OUTPUT_DIR="$PROJECT_ROOT/data/clap_benchmark"

cd "$PROJECT_ROOT"

usage() {
  echo "Usage: $0 [--audio-samples N] [--max-audios N] [--batch-size N] [--queries-per-audio N] [--audio-clip-seconds SECONDS]" >&2
  echo "  Builds/resumes the CLAP caption-to-audio benchmark in data/clap_benchmark." >&2
  echo "  --max-audios is total audio records to score per caption query." >&2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --audio-samples)
      if [ "$#" -lt 2 ]; then
        usage
        exit 1
      fi
      N="$2"
      shift 2
      ;;
    --max-audios)
      if [ "$#" -lt 2 ]; then
        usage
        exit 1
      fi
      MAX_AUDIOS="$2"
      shift 2
      ;;
    --queries-per-audio)
      if [ "$#" -lt 2 ]; then
        usage
        exit 1
      fi
      QUERIES_PER_AUDIO="$2"
      shift 2
      ;;
    --batch-size)
      if [ "$#" -lt 2 ]; then
        usage
        exit 1
      fi
      BATCH_SIZE="$2"
      shift 2
      ;;
    --audio-clip-seconds)
      if [ "$#" -lt 2 ]; then
        usage
        exit 1
      fi
      AUDIO_CLIP_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 2 ]; then
  usage
  exit 1
fi

if ! [[ "$MAX_AUDIOS" =~ ^[0-9]+$ ]] || [ "$MAX_AUDIOS" -lt 2 ]; then
  usage
  exit 1
fi

if ! [[ "$QUERIES_PER_AUDIO" =~ ^[0-9]+$ ]] || [ "$QUERIES_PER_AUDIO" -lt 1 ]; then
  usage
  exit 1
fi

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
  usage
  exit 1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Gated repo download may fail with 401 Unauthorized." >&2
fi

metadata_count() {
  local path="$1"
  if [ -f "$path" ]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

if [ "$(metadata_count "$SONG_DESCRIBER_METADATA")" -lt "$N" ]; then
  python -m src.datasets.downloaders.download_song_describer --number "$N"
fi

LLAMA_CPP_DIR="$HOME/llama-cpp"
LLAMA_SERVER="$LLAMA_CPP_DIR/llama-server"
MODEL_REPO="ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M"
MODEL_ALIAS="gemma-4-E4B-it"

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "llama-server not found or not executable: $LLAMA_SERVER" >&2
  exit 1
fi

# Check if llama-server is already running
SERVER_RUNNING=false
if curl -s -f "http://localhost:8080/health" > /dev/null 2>&1; then
  echo "llama-server is already running on port 8080"
  SERVER_RUNNING=true
else
  echo "llama-server is not running, starting it now..."
  
  "$LLAMA_SERVER" \
    -hf "$MODEL_REPO" \
    --alias "$MODEL_ALIAS" \
    -c 4096 \
    --port 8080 \
    -ngl 12 \
    -fit off &

  SERVER_PID=$!
  # Only set trap if we started the server
  trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT

  echo "Waiting for llama-server to download models and become ready..."
  MAX_ATTEMPTS=60
  ATTEMPT=0
  while ! curl -s -f "http://localhost:8080/health" > /dev/null; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "Error: llama-server process died unexpectedly before initialization." >&2
      exit 1
    fi
    ATTEMPT=$((ATTEMPT + 1))
    if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
      echo "Error: llama-server timed out waiting to start." >&2
      exit 1
    fi
    sleep 5
  done
  echo "llama-server is up and running successfully!"
fi

python -m src.benchmark.create_clap_benchmark \
  --audio_metadata "$SONG_DESCRIBER_METADATA" \
  --audio_samples "$N" \
  --batch_size "$BATCH_SIZE" \
  --queries_per_audio "$QUERIES_PER_AUDIO" \
  --max_audios "$MAX_AUDIOS" \
  --num_shards "$NUM_SHARDS" \
  --shard_index "$SHARD_INDEX" \
  --audio_clip_seconds "$AUDIO_CLIP_SECONDS" \
  --max_output_tokens 512 \
  --model "$MODEL_ALIAS" \
  --output_dir "$OUTPUT_DIR"
