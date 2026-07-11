from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.cxc_sits import load_cxc_sits_retrieval_data
from src.metrics.retrieval_metrics import (
    sparse_retrieval_metrics,
    transpose_sparse_relevance,
)
from src.models.clip_encoder import PretrainedCLIPEncoder
from src.utils.batch_encoding import encode_in_batches_clip
from src.utils.json import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation_results" / "clip_eval_results.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CLIP image-to-caption retrieval with CxC SITS graded "
            "caption relevance labels."
        )
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to merged CxC JSON, e.g. data/metadata/coco_karpathy_cxc_sits_val.json.",
    )
    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help="Root folder containing COCO images.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.0,
        help="Scores above this threshold are relevant for MRR/mAP/recall/precision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    image_paths, captions, image_to_caption_relevance = load_cxc_sits_retrieval_data(
        metadata_path=metadata_path,
        image_root=image_root,
        max_images=args.max_images,
    )
    num_nonzero_relevance = sum(len(row) for row in image_to_caption_relevance)

    if not image_paths:
        raise RuntimeError("No valid images found. Check --image-root.")
    if not captions:
        raise RuntimeError("No captions found in metadata.")
    if num_nonzero_relevance == 0:
        raise RuntimeError("No CxC relevance labels found. Check merged metadata.")

    model = PretrainedCLIPEncoder(
        model_name=args.model_name,
        freeze=True,
    )
    image_embeddings, text_embeddings = encode_in_batches_clip(
        model=model,
        image_paths=image_paths,
        texts=captions,
        batch_size=args.batch_size,
    )
    similarity = image_embeddings @ text_embeddings.T
    similarity = similarity.numpy()

    caption_to_image_relevance = transpose_sparse_relevance(
        relevance=image_to_caption_relevance,
        num_targets=len(captions),
    )
    image_to_caption_metrics = sparse_retrieval_metrics(
        similarity=similarity,
        relevance=image_to_caption_relevance,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )
    caption_to_image_metrics = sparse_retrieval_metrics(
        similarity=similarity.T,
        relevance=caption_to_image_relevance,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )

    results = {
        "metadata": str(metadata_path),
        "image_root": str(image_root),
        "model_name": args.model_name,
        "num_images": len(image_paths),
        "num_captions": len(captions),
        "num_nonzero_relevance": num_nonzero_relevance,
        "metric_notes": {
            "ndcg": "Uses raw CxC score as graded relevance.",
            "binary_metrics": (
                "MRR, mAP, recall, and precision treat scores above "
                f"{args.relevance_threshold} as relevant."
            ),
        },
        "image_to_caption": image_to_caption_metrics,
        "caption_to_image": caption_to_image_metrics,
    }

    print(json.dumps(results, indent=2))
    write_json(output_path, results)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
