import argparse
import json
from pathlib import Path
from src.models.clip_encoder import PretrainedCLIPEncoder
from src.utils.batch_encoding import encode_in_batches_clip
from src.datasets.evaluation_inputs import load_cxc_image_caption_records
from src.metrics.retrieval_metrics import (
    sparse_binary_retrieval_metrics,
    sparse_compute_metrics,
    transpose_sparse_relevance,
    balanced_edge_retrieval_metrics,
    validate_binary_metrics
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CLIP image-to-caption retrieval with CxC SITS graded labels."
    )

    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to merged CxC JSON, e.g. data/metadata/coco_karpathy_cxc_sits_val.json",
    )

    parser.add_argument(
        "--image-root",
        type=str,
        required=True,
        help=(
            "Root folder containing COCO images. "
            "Expected paths like image_root/val2014/COCO_val2014_000000391895.jpg"
        ),
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Limit number of images for debugging.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="openai/clip-vit-base-patch32",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "evaluation_results" / "clip_cxc_eval.json"),
    )

    parser.add_argument(
        "--exponential-gain",
        action="store_true",
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()



def main():
    args = parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    image_root = Path(args.image_root).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    image_paths, captions, relevance = load_cxc_image_caption_records(
        metadata_path=metadata_path,
        image_root=image_root,
        max_images=args.max_images,
    )

    print(f"Images with CxC labels and existing files: {len(image_paths)}")
    print(f"Caption targets: {len(captions)}")
    num_nonzero_relevance = sum(len(query_relevance) for query_relevance in relevance)
    print(f"Sparse relevance rows: {len(relevance)}")
    print(f"Nonzero relevance labels: {num_nonzero_relevance}")

    if len(image_paths) == 0:
        raise RuntimeError("No valid images found. Check --image-root.")

    if len(captions) == 0:
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
        relevance=relevance,
        num_targets=len(captions),
    )

    results = {
        "metadata": str(metadata_path),
        "image_root": str(image_root),
        "model_name": args.model_name,
        "num_images": len(image_paths),
        "num_captions": len(captions),
        "num_nonzero_relevance": num_nonzero_relevance,
        "metrics": {
            "graded": "ndcg uses raw CxC relevance scores",
            "binary": "hit/recall/mrr treat every CxC score > 0 as relevant",
            "relevance_storage": "sparse dictionaries; dense zero-filled relevance matrix is not materialized",
        },
        "image_to_caption": {
            "graded_ndcg": sparse_compute_metrics(
                similarity=similarity,
                relevance=relevance,
                exponential_gain=False,
            ),
            "graded_ndcg_exponential_gain": sparse_compute_metrics(
                similarity=similarity,
                relevance=relevance,
                exponential_gain=True,
            ),
            "binary_retrieval": sparse_binary_retrieval_metrics(
                similarity=similarity,
                relevance=relevance,
            ),
        },
        "caption_to_image": {
            "graded_ndcg": sparse_compute_metrics(
                similarity=similarity.T,
                relevance=caption_to_image_relevance,
                exponential_gain=False,
            ),
            "graded_ndcg_exponential_gain": sparse_compute_metrics(
                similarity=similarity.T,
                relevance=caption_to_image_relevance,
                exponential_gain=True,
            ),
            "binary_retrieval": sparse_binary_retrieval_metrics(
                similarity=similarity.T,
                relevance=caption_to_image_relevance,
            ),
        },
        "balanced_pairwise": {
            "requested_num_negatives": 999,
            "image_to_caption": balanced_edge_retrieval_metrics(
                similarity=similarity,
                relevance=relevance,
                num_negatives=999,
                seed=0,
            ),
            "caption_to_image": balanced_edge_retrieval_metrics(
                similarity=similarity.T,
                relevance=caption_to_image_relevance,
                num_negatives=999,
                seed=0,
            ),
        },
    }


    validate_binary_metrics(
        results["image_to_caption"]["binary_retrieval"],
        "image_to_caption",
    )

    validate_binary_metrics(
        results["caption_to_image"]["binary_retrieval"],
        "caption_to_image",
    )

    print(json.dumps(results, indent=2))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
