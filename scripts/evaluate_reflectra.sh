#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

REFLECTRA_BENCHMARK_DIR="$PROJECT_ROOT/data/benchmark"
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results"
BATCH_SIZE=8
RELEVANCE_THRESHOLD=0.0
CHECKPOINT=""

usage() {
  cat >&2 <<'EOF'
Usage: scripts/evaluate_reflectra.sh [options]

Downloads/unpacks the Reflectra Hugging Face benchmark, then runs:
  - python -m src.evaluation.evaluate_reflectra

Options:
  --checkpoint PATH          Reflectra projection checkpoint.
  -h, --help                 Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint)
      if [ "$#" -lt 2 ]; then usage; exit 1; fi
      CHECKPOINT="$2"
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

if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
  usage
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Downloading and unpacking Reflectra benchmark..."
python -m src.datasets.downloaders.download_reflectra_benchmark \
  --output-dir "$REFLECTRA_BENCHMARK_DIR" \
  --batch-size "$BATCH_SIZE"

reflectra_args=()
if [ -n "$CHECKPOINT" ]; then
  reflectra_args+=(--checkpoint "$CHECKPOINT")
fi

echo "[INFO] Running Reflectra evaluation..."
python -m src.evaluation.evaluate_reflectra \
  --benchmark "$REFLECTRA_BENCHMARK_DIR" \
  --batch-size "$BATCH_SIZE" \
  --relevance-threshold "$RELEVANCE_THRESHOLD" \
  --output "$OUTPUT_DIR/reflectra_eval_results.json" \
  "${reflectra_args[@]}"

echo "[INFO] Done."
echo "[INFO] Reflectra results: $OUTPUT_DIR/reflectra_eval_results.json"
