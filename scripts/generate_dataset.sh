#!/usr/bin/env bash
set -euo pipefail

N=1000
NUM_SHARDS=4
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLICKR_METADATA="$PROJECT_ROOT/data/metadata/flickr30k_metadata.jsonl"
SONG_DESCRIBER_METADATA="$PROJECT_ROOT/data/metadata/song_describer_metadata.jsonl"

cd "$PROJECT_ROOT"

usage() {
  echo "Usage: $0 [-s SHARD_INDEX]" >&2
  echo "  Omit -s for single-PC mode: build all pairs and merge parquet tables." >&2
  echo "  Use -s with shard index 0..$((NUM_SHARDS - 1)) to build one shard only." >&2
}

SHARD_INDEX=""
while getopts ":s:h" opt; do
  case "$opt" in
    s)
      SHARD_INDEX="$OPTARG"
      ;;
    h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [ "$OPTIND" -le "$#" ]; then
  usage
  exit 1
fi

if [ -n "$SHARD_INDEX" ]; then
  if ! [[ "$SHARD_INDEX" =~ ^[0-9]+$ ]] || [ "$SHARD_INDEX" -ge "$NUM_SHARDS" ]; then
    usage
    exit 1
  fi
  BENCHMARK_NUM_SHARDS="$NUM_SHARDS"
  BENCHMARK_SHARD_INDEX="$SHARD_INDEX"
  RUN_MERGE=0
else
  BENCHMARK_NUM_SHARDS=1
  BENCHMARK_SHARD_INDEX=0
  RUN_MERGE=1
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "WARNING: HF_TOKEN is not set. Gated repo download may fail with 401 Unauthorized." >&2
fi

if [ "$RUN_MERGE" -eq 1 ]; then
  rm -f data/benchmark/manifest_shard_*.json data/benchmark/benchmark_shard_*.jsonl || true
fi

metadata_count() {
  local path="$1"
  if [ -f "$path" ]; then
    wc -l < "$path"
  else
    echo 0
  fi
}

if [ "$(metadata_count "$FLICKR_METADATA")" -lt "$N" ]; then
  python -m src.datasets.downloaders.download_flickr30k --number "$N"
fi

if [ "$(metadata_count "$SONG_DESCRIBER_METADATA")" -lt "$N" ]; then
  python -m src.datasets.downloaders.download_song_describer --number "$N"
fi

LLAMA_CPP_DIR="$HOME/llama-cpp"
LLAMA_SERVER="$LLAMA_CPP_DIR/llama-server"
# Changed repository to 4B and forced the Q4_K_M quantization layout
MODEL_REPO="ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M"
MODEL_ALIAS="gemma-4-E4B-it"

if [ ! -x "$LLAMA_SERVER" ]; then
  echo "llama-server not found or not executable: $LLAMA_SERVER" >&2
  exit 1
fi

# Set to 12 layers to safely leverage your 4GB VRAM without hitting OOM limits
"$LLAMA_SERVER" \
  -hf "$MODEL_REPO" \
  --alias "$MODEL_ALIAS" \
  -c 4096 \
  --port 8080 \
  -ngl 12 \
  -fit off &

SERVER_PID=$!
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

python -m src.benchmark.create_benchmark \
  --mode build \
  --image_samples "$N" \
  --audio_samples "$N" \
  --batch_size 4 \
  --max_output_tokens 128 \
  --model "$MODEL_ALIAS" \
  --num_shards "$BENCHMARK_NUM_SHARDS" \
  --shard_index "$BENCHMARK_SHARD_INDEX" \
  --image_metadata "$FLICKR_METADATA" \
  --audio_metadata "$SONG_DESCRIBER_METADATA"

if [ "$RUN_MERGE" -eq 1 ]; then
  python -m src.benchmark.create_benchmark \
    --mode merge \
    --image_samples "$N" \
    --audio_samples "$N" \
    --batch_size 4 \
    --max_output_tokens 128 \
    --model "$MODEL_ALIAS" \
    --num_shards "$BENCHMARK_NUM_SHARDS" \
    --image_metadata "$FLICKR_METADATA" \
    --audio_metadata "$SONG_DESCRIBER_METADATA"
else
  echo "[INFO] Built shard $BENCHMARK_SHARD_INDEX/$BENCHMARK_NUM_SHARDS."
  echo "[INFO] Merge after all shards are available with:"
  echo "       python -m src.benchmark.create_benchmark --mode merge --image_samples $N --audio_samples $N --model $MODEL_ALIAS --num_shards $BENCHMARK_NUM_SHARDS --image_metadata $FLICKR_METADATA --audio_metadata $SONG_DESCRIBER_METADATA"
fi
