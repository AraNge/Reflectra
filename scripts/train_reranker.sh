#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

usage() {
  echo "Usage: $0 [-p PROJECTION_CHECKPOINT] [-b BENCHMARK_PATH] [-t reranker_type] [-e epochs] [-r train_size] [-s test_size]" >&2
  echo "  -p is optional: if omitted, the most recently modified checkpoints/*projection*.pt is used." >&2
}

PROJECTION_CHECKPOINT=""
BENCHMARK_PATH="data/benchmark/image_audio_scores.parquet"
RERANKER_TYPE="mlp"
EPOCHS=15
TRAIN_SIZE=700
TEST_SIZE=300

while getopts ":p:b:t:e:r:s:" opt; do
  case "$opt" in
    p) PROJECTION_CHECKPOINT="$OPTARG" ;;
    b) BENCHMARK_PATH="$OPTARG" ;;
    t) RERANKER_TYPE="$OPTARG" ;;
    e) EPOCHS="$OPTARG" ;;
    r) TRAIN_SIZE="$OPTARG" ;;
    s) TEST_SIZE="$OPTARG" ;;
    *) usage; exit 1 ;;
  esac
done

ARGS=(
  --benchmark "$BENCHMARK_PATH"
  --reranker-type "$RERANKER_TYPE"
  --epochs "$EPOCHS"
  --train-size "$TRAIN_SIZE"
  --test-size "$TEST_SIZE"
)

if [ -n "$PROJECTION_CHECKPOINT" ]; then
  ARGS+=(--projection-checkpoint "$PROJECTION_CHECKPOINT")
fi

python -m src.training.train_reranker "${ARGS[@]}"

echo "[INFO] Reranker training done. The exact checkpoint filename was printed above (it is timestamped, not fixed)."
echo "[INFO] Evaluate with (replace <checkpoint> with the printed path):"
echo "       python -m src.evaluation.evaluate_reflectra --checkpoint <projection_checkpoint> --output evaluation_results/reflectra_eval_baseline.json"
echo "       python -m src.evaluation.evaluate_reflectra --checkpoint <projection_checkpoint> --use_reranker --reranker_checkpoint <checkpoint> --output evaluation_results/reflectra_eval_reranked.json"
