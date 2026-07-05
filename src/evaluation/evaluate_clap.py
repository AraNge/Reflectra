import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from src.datasets.loaders.audio_metadata import AudioMetadataLoader
from src.datasets.preprocessing.sampling import (
    sample_by_dataset_fractions,
    sample_by_dataset_counts,
    limit_total,
)
from src.metrics.retrieval_metrics import retrieval_metrics
from src.models.clap_encoder import PretrainedCLAPEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


DEFAULT_AUDIO_METADATA_PATHS = [
    DATA_DIR / "musiccaps_metadata.jsonl",
    DATA_DIR / "audioset_metadata.jsonl",
    DATA_DIR / "song_describer_metadata.jsonl",
    DATA_DIR / "mtg_jamendo_metadata" / "mtg_jamendo_train_metadata.jsonl",
    DATA_DIR / "mtg_jamendo_metadata" / "mtg_jamendo_validation_metadata.jsonl",
]


def parse_dataset_fractions(value: str | None) -> Dict[str, float]:
    if not value:
        return {}

    result = {}

    for item in value.split(","):
        dataset_name, fraction = item.split("=")
        result[dataset_name.strip()] = float(fraction)

    return result


def parse_dataset_counts(value: str | None) -> Dict[str, int]:
    if not value:
        return {}

    result = {}

    for item in value.split(","):
        dataset_name, count = item.split("=")
        result[dataset_name.strip()] = int(count)

    return result


def encode_in_batches(
    model: PretrainedCLAPEncoder,
    audio_paths: list[str],
    texts: list[str],
    batch_size: int,
):
    audio_embeddings = []
    text_embeddings = []

    for start in tqdm(range(0, len(audio_paths), batch_size), desc="Encoding CLAP"):
        end = start + batch_size

        batch_audio_paths = audio_paths[start:end]
        batch_texts = texts[start:end]

        audio_emb = model.encode_audio(batch_audio_paths)
        text_emb = model.encode_text(batch_texts)

        audio_embeddings.append(audio_emb.cpu())
        text_embeddings.append(text_emb.cpu())

    audio_embeddings = torch.cat(audio_embeddings, dim=0)
    text_embeddings = torch.cat(text_embeddings, dim=0)

    return audio_embeddings, text_embeddings


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-name", type=str, default="laion/clap-htsat-unfused")

    parser.add_argument(
        "--dataset-fractions",
        type=str,
        default=None,
        help='Example: "google/MusicCaps=1.0,agkphysics/AudioSet=0.5"',
    )

    parser.add_argument(
        "--dataset-counts",
        type=str,
        default=None,
        help='Example: "google/MusicCaps=5000,rkstgr/mtg-jamendo=10000"',
    )

    args = parser.parse_args()

    loader = AudioMetadataLoader(
        metadata_paths=DEFAULT_AUDIO_METADATA_PATHS,
        project_root=PROJECT_ROOT,
        require_audio_exists=True,
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

    audio_paths = [record["audio_path"] for record in records]
    texts = [record["text"] for record in records]

    model = PretrainedCLAPEncoder(
        model_name=args.model_name,
        freeze=True,
    )

    audio_embeddings, text_embeddings = encode_in_batches(
        model=model,
        audio_paths=audio_paths,
        texts=texts,
        batch_size=args.batch_size,
    )

    similarity = text_embeddings @ audio_embeddings.T
    similarity = similarity.numpy()

    text_to_audio = retrieval_metrics(similarity)
    audio_to_text = retrieval_metrics(similarity.T)

    results = {
        "num_samples": len(records),
        "model_name": args.model_name,
        "text_to_audio": text_to_audio,
        "audio_to_text": audio_to_text,
    }

    print(json.dumps(results, indent=2))

    output_dir = PROJECT_ROOT / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "clap_eval_results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to: {output_path}")



"""
python -m src.evaluation.evaluate_clap --max-samples 1000

python -m src.evaluation.evaluate_clap \
  --dataset-fractions "google/MusicCaps=1.0,agkphysics/AudioSet=0.5,rkstgr/mtg-jamendo=0.8" \
  --max-samples 50000


python -m src.evaluation.evaluate_clap \
  --dataset-counts "google/MusicCaps=5000,rkstgr/mtg-jamendo=10000,agkphysics/AudioSet=20000"
"""

if __name__ == "__main__":
    main()