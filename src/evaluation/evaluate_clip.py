import argparse
import json
from pathlib import Path
from typing import Dict
import torch
from tqdm import tqdm

from src.datasets.loaders.image_metadata import ImageTextMetadataLoader
from src.datasets.preprocessing.sampling import (
    sample_by_dataset_fractions,
    sample_by_dataset_counts,
    limit_total,
)
from metrics.retrieval_metrics import retrieval_metrics
from src.models.clip_encoder import PretrainedCLIPEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


DEFAULT_IMAGE_METADATA_PATHS = [
    DATA_DIR / "coco_captions_metadata.jsonl",
    DATA_DIR / "flickr30k_metadata.jsonl",
    DATA_DIR / "emoset_train_metadata.jsonl",
    DATA_DIR / "emoset_test_metadata.jsonl",
]


def parse_dataset_fractions(value: str | None) -> Dict[str, float]:
    """
    Example:
        "whyen-wang/coco_captions=0.5,nlphuji/flickr30k=0.8"
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
        "whyen-wang/coco_captions=50000,nlphuji/flickr30k=10000"
    """

    if not value:
        return {}

    result = {}

    for item in value.split(","):
        dataset_name, count = item.split("=")
        result[dataset_name.strip()] = int(count)

    return result


def encode_in_batches(
    model: PretrainedCLIPEncoder,
    image_paths: list[str],
    texts: list[str],
    batch_size: int,
):
    image_embeddings = []
    text_embeddings = []

    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding CLIP"):
        end = start + batch_size

        batch_images = image_paths[start:end]
        batch_texts = texts[start:end]

        image_emb = model.encode_image(batch_images)
        text_emb = model.encode_text(batch_texts)

        image_embeddings.append(image_emb.cpu())
        text_embeddings.append(text_emb.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    return image_embeddings, text_embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-name", type=str, default="openai/clip-vit-base-patch32")

    parser.add_argument(
        "--dataset-fractions",
        type=str,
        default=None,
        help='Example: "whyen-wang/coco_captions=0.5,nlphuji/flickr30k=0.8"',
    )

    parser.add_argument(
        "--dataset-counts",
        type=str,
        default=None,
        help='Example: "whyen-wang/coco_captions=5000,nlphuji/flickr30k=1000"',
    )

    args = parser.parse_args()

    loader = ImageTextMetadataLoader(
        metadata_paths=DEFAULT_IMAGE_METADATA_PATHS,
        project_root=PROJECT_ROOT,
        require_image_exists=True,
        expand_captions=True
    )

    records = loader.load()

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

    print(f"Evaluation samples: {len(records)}")

    image_paths = [record["image_path"] for record in records]
    texts = [record["text"] for record in records]

    model = PretrainedCLIPEncoder(
        model_name=args.model_name,
        freeze=True,
    )

    image_embeddings, text_embeddings = encode_in_batches(
        model=model,
        image_paths=image_paths,
        texts=texts,
        batch_size=args.batch_size,
    )

    similarity = image_embeddings @ text_embeddings.T
    similarity = similarity.numpy()

    image_to_text = retrieval_metrics(similarity)
    text_to_image = retrieval_metrics(similarity.T)

    results = {
        "num_samples": len(records),
        "model_name": args.model_name,
        "image_to_text": image_to_text,
        "text_to_image": text_to_image,
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
  --dataset-fractions "whyen-wang/coco_captions=0.5,nlphuji/flickr30k=0.8,LiangJian24/EmoSet=1.0" \
  --max-samples 50000

python -m src.evaluation.evaluate_clip \
  --dataset-fractions "whyen-wang/coco_captions=0.5,nlphuji/flickr30k=0.8,LiangJian24/EmoSet=1.0" \
  --max-samples 50000

python -m src.evaluation.evaluate_clip --max-samples 1000
"""

if __name__ == "__main__":
    main()