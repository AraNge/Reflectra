import argparse
import json
from pathlib import Path
from typing import Dict
from src.utils.batch_encoding import encode_in_batches_clip

from src.datasets.loaders.image_metadata import ImageTextMetadataLoader
from src.datasets.evaluation_inputs import build_sparse_grouped_retrieval_inputs
from src.datasets.preprocessing.sampling import (
    sample_by_dataset_fractions,
    sample_by_dataset_counts,
    limit_total,
)
from src.metrics.retrieval_metrics import (
    balanced_edge_retrieval_metrics,
    sparse_binary_retrieval_metrics,
    sparse_compute_metrics,
    transpose_sparse_relevance,
    validate_binary_metrics,
)
from src.models.clip_encoder import PretrainedCLIPEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"


DEFAULT_IMAGE_METADATA_PATHS = [
    METADATA_DIR / "coco_captions_metadata.jsonl",
    METADATA_DIR / "flickr30k_metadata.jsonl",
    METADATA_DIR / "emoset_train_metadata.jsonl",
    METADATA_DIR / "emoset_test_metadata.jsonl",
]


def parse_dataset_fractions(value: str | None) -> Dict[str, float]:
    """
    Example:
        "coco_karpathy=0.5,nlphuji/flickr30k=0.8"
    """

    if not value:
        return {}

    result = {}

    for item in value.split(","):
        dataset_name, fraction = item.split("=")
        result[dataset_name.strip()] = float(fraction)

    return result


def parse_dataset_counts(value: str | None) -> Dict[str, int]:
    """
    Example:
        "coco_karpathy=50000,nlphuji/flickr30k=10000"
    """

    if not value:
        return {}

    result = {}

    for item in value.split(","):
        dataset_name, count = item.split("=")
        result[dataset_name.strip()] = int(count)

    return result


def parse_comma_separated_set(value: str | None) -> set[str]:
    if not value:
        return set()

    return {
        item.strip()
        for item in value.split(",")
        if item.strip()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=[str(path) for path in DEFAULT_IMAGE_METADATA_PATHS],
        help="JSONL metadata files. Defaults to local image-text metadata files.",
    )
    parser.add_argument(
        "--text-types",
        type=str,
        default="caption",
        help='Comma-separated text types to evaluate. Defaults to "caption".',
    )

    parser.add_argument(
        "--dataset-fractions",
        type=str,
        default=None,
        help='Example: "coco_karpathy=0.5,nlphuji/flickr30k=0.8"',
    )

    parser.add_argument(
        "--dataset-counts",
        type=str,
        default=None,
        help='Example: "coco_karpathy=5000,nlphuji/flickr30k=1000"',
    )

    args = parser.parse_args()

    loader = ImageTextMetadataLoader(
        metadata_paths=[Path(path) for path in args.metadata],
        project_root=PROJECT_ROOT,
        require_image_exists=True,
    )

    records = loader.load()

    text_types = parse_comma_separated_set(args.text_types)

    if text_types:
        records = [
            record for record in records
            if record["text_type"] in text_types
        ]

    if args.split:
        records = [
            record for record in records
            if record["split"] == args.split
        ]

    dataset_fractions = parse_dataset_fractions(args.dataset_fractions)
    dataset_counts = parse_dataset_counts(args.dataset_counts)

    if dataset_fractions:
        records = sample_by_dataset_fractions(
            records=records,
            fractions=dataset_fractions,
        )

    if dataset_counts:
        records = sample_by_dataset_counts(
            records=records,
            counts=dataset_counts,
        )

    records = limit_total(records, args.max_samples)

    print(f"Loaded text records: {len(records)}")

    if len(records) == 0:
        raise RuntimeError("No image-text records found. Check metadata paths, split, filters, and downloaded files.")

    image_paths, texts, image_to_text_relevance, input_stats = build_sparse_grouped_retrieval_inputs(
        records=records,
        media_id_field="image_id",
        media_path_field="image_path",
    )

    num_positive_edges = input_stats["num_positive_edges"]

    print(f"Unique images: {len(image_paths)}")
    print(f"Caption/query targets: {len(texts)}")
    print(f"Sparse positive links: {num_positive_edges}")

    if not image_paths or not texts or num_positive_edges == 0:
        raise RuntimeError("No valid grouped image-text retrieval inputs found.")

    model = PretrainedCLIPEncoder(
        model_name=args.model_name,
        freeze=True,
    )

    image_embeddings, text_embeddings = encode_in_batches_clip(
        model=model,
        image_paths=image_paths,
        texts=texts,
        batch_size=args.batch_size,
    )

    similarity = image_embeddings @ text_embeddings.T
    similarity = similarity.numpy()

    text_to_image_relevance = transpose_sparse_relevance(
        relevance=image_to_text_relevance,
        num_targets=len(texts),
    )

    image_to_text_binary = sparse_binary_retrieval_metrics(
        similarity=similarity,
        relevance=image_to_text_relevance,
    )
    text_to_image_binary = sparse_binary_retrieval_metrics(
        similarity=similarity.T,
        relevance=text_to_image_relevance,
    )

    validate_binary_metrics(image_to_text_binary, "image_to_text")
    validate_binary_metrics(text_to_image_binary, "text_to_image")

    results = {
        "num_records": len(records),
        "num_images": len(image_paths),
        "num_texts": len(texts),
        "num_positive_edges": num_positive_edges,
        "model_name": args.model_name,
        "metric": "sparse_grouped_binary_retrieval",
        "metric_notes": {
            "binary": "hit/recall/mrr treat every caption attached to the same image as relevant.",
            "binary_ndcg": "ndcg uses binary relevance scores because normal COCO/Flickr metadata has no graded labels.",
            "relevance_storage": "sparse dictionaries; dense zero-filled relevance matrix is not materialized.",
            "balanced_pairwise": "Each positive edge is evaluated against one positive plus up to the requested sampled negatives.",
        },
        "text_types": sorted(text_types),
        "image_to_text": {
            "binary_ndcg": sparse_compute_metrics(
                similarity=similarity,
                relevance=image_to_text_relevance,
                exponential_gain=False,
            ),
            "binary_retrieval": image_to_text_binary,
        },
        "text_to_image": {
            "binary_ndcg": sparse_compute_metrics(
                similarity=similarity.T,
                relevance=text_to_image_relevance,
                exponential_gain=False,
            ),
            "binary_retrieval": text_to_image_binary,
        },
        "balanced_pairwise": {
            "requested_num_negatives": 999,
            "image_to_text": balanced_edge_retrieval_metrics(
                similarity=similarity,
                relevance=image_to_text_relevance,
                num_negatives=999,
                seed=0,
            ),
            "text_to_image": balanced_edge_retrieval_metrics(
                similarity=similarity.T,
                relevance=text_to_image_relevance,
                num_negatives=999,
                seed=0,
            ),
        },
    }

    print(json.dumps(results, indent=2))

    output_dir = PROJECT_ROOT / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "clip_eval_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")


"""
python -m src.evaluation.evaluate_clip \
  --dataset-fractions "nlphuji/flickr30k=0.8" \
  --max-samples 50000

python -m src.evaluation.evaluate_clip --max-samples 1000

python -m src.evaluation.evaluate_clip \
  --text-types "caption" \
  --dataset-fractions "nlphuji/flickr30k=0.8,LiangJian24/EmoSet=1.0" \
  --max-samples 50000
"""

if __name__ == "__main__":
    main()
