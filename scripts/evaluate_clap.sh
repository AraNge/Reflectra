#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CLAP_BENCHMARK_DIR="$PROJECT_ROOT/data/clap_benchmark"
OUTPUT_DIR="$PROJECT_ROOT/evaluation_results"
BATCH_SIZE=8
RELEVANCE_THRESHOLD=0.0
MODEL_NAME="laion/clap-htsat-unfused"

usage() {
  cat >&2 <<'EOF'
Usage: scripts/evaluate_clap.sh [options]

Downloads/unpacks the CLAP Hugging Face benchmark, then runs:
  - python -m src.evaluation.evaluate_clap

Options:
  -h, --help                 Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
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

mkdir -p "$OUTPUT_DIR"

echo "[INFO] Downloading and unpacking CLAP benchmark..."
python -m src.datasets.downloaders.download_clap_benchmark \
  --output-dir "$CLAP_BENCHMARK_DIR" \
  --batch-size "$BATCH_SIZE"

echo "[INFO] Running CLAP evaluation..."
python -m src.evaluation.evaluate_clap \
  --benchmark_dir "$CLAP_BENCHMARK_DIR" \
  --batch-size "$BATCH_SIZE" \
  --model-name "$MODEL_NAME" \
  --relevance-threshold "$RELEVANCE_THRESHOLD" \
  --output "$OUTPUT_DIR/clap_eval_results.json"

echo "[INFO] Done."
echo "[INFO] CLAP results: $OUTPUT_DIR/clap_eval_results.json"
