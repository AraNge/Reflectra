#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FLICKR_METADATA="$PROJECT_ROOT/data/metadata/flickr30k_metadata.jsonl"

cd "$PROJECT_ROOT"

usage() {
  echo "Usage: $0 -n NUM_SAMPLES" >&2
}

N=""
while getopts ":n:" opt; do
  case "$opt" in
    n)
      N="$OPTARG"
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

if [ -z "$N" ] || ! [[ "$N" =~ ^[0-9]+$ ]] || [ "$N" -lt 1 ]; then
  usage
  exit 1
fi

metadata_count() {
  if [ -f "$FLICKR_METADATA" ]; then
    wc -l < "$FLICKR_METADATA"
  else
    echo 0
  fi
}

if [ "$(metadata_count)" -lt "$N" ]; then
  python -m src.datasets.downloaders.download_flickr30k --number "$N"
fi

python -m src.training.train_projection \
  --image_metadata "$FLICKR_METADATA" \
  --train-split test \
  --max-train-samples "$N" \
  --max-val-samples 100 \
  --output-name "reflectra_projection_flickr30k_${N}.pt"
