import argparse
import json
from pathlib import Path
from typing import Dict
from src.utils.batch_encoding import encode_in_batches_clap

from src.datasets.loaders.audio_metadata import AudioMetadataLoader
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
from src.models.clap_encoder import PretrainedCLAPEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
METADATA_DIR = DATA_DIR / "metadata"


DEFAULT_AUDIO_METADATA_PATHS = [
    METADATA_DIR / "musiccaps_metadata.jsonl",
    METADATA_DIR / "audioset_metadata.jsonl",
    METADATA_DIR / "audioset_balanced_metadata.jsonl",
    METADATA_DIR / "audioset_unbalanced_metadata.jsonl",
    METADATA_DIR / "song_describer_metadata.jsonl",
    METADATA_DIR / "mtg_jamendo_train_metadata.jsonl",
    METADATA_DIR / "mtg_jamendo_validation_metadata.jsonl",
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


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-name", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument(
        "--metadata",
        type=str,
        nargs="*",
        default=[str(path) for path in DEFAULT_AUDIO_METADATA_PATHS],
        help="JSONL metadata files. Defaults to local audio-text metadata files.",
    )

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
        metadata_paths=[Path(path) for path in args.metadata],
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

    print(f"Loaded text records: {len(records)}")

    if len(records) == 0:
        raise RuntimeError("No valid audio-text records after filtering.")

    audio_paths, texts, audio_to_text_relevance, input_stats = build_sparse_grouped_retrieval_inputs(
        records=records,
        media_id_field="audio_id",
        media_path_field="audio_path",
    )

    num_positive_edges = input_stats["num_positive_edges"]

    print(f"Unique audio files: {len(audio_paths)}")
    print(f"Caption/query targets: {len(texts)}")
    print(f"Sparse positive links: {num_positive_edges}")

    if not audio_paths or not texts or num_positive_edges == 0:
        raise RuntimeError("No valid grouped audio-text retrieval inputs found.")

    model = PretrainedCLAPEncoder(
        model_name=args.model_name,
        freeze=True,
    )

    audio_embeddings, text_embeddings = encode_in_batches_clap(
        model=model,
        audio_paths=audio_paths,
        texts=texts,
        batch_size=args.batch_size,
    )

    similarity = audio_embeddings @ text_embeddings.T
    similarity = similarity.numpy()

    text_to_audio_relevance = transpose_sparse_relevance(
        relevance=audio_to_text_relevance,
        num_targets=len(texts),
    )

    audio_to_text_binary = sparse_binary_retrieval_metrics(
        similarity=similarity,
        relevance=audio_to_text_relevance,
    )
    text_to_audio_binary = sparse_binary_retrieval_metrics(
        similarity=similarity.T,
        relevance=text_to_audio_relevance,
    )

    validate_binary_metrics(audio_to_text_binary, "audio_to_text")
    validate_binary_metrics(text_to_audio_binary, "text_to_audio")

    results = {
        "num_records": len(records),
        "num_audio": len(audio_paths),
        "num_texts": len(texts),
        "num_positive_edges": num_positive_edges,
        "model_name": args.model_name,
        "metric_notes": {
            "binary": "hit/recall/mrr treat every caption attached to the same audio file as relevant.",
            "binary_ndcg": "ndcg uses binary relevance scores because these audio datasets have no graded labels.",
            "relevance_storage": "sparse dictionaries; dense zero-filled relevance matrix is not materialized.",
            "balanced_pairwise": "Each positive edge is evaluated against one positive plus up to the requested sampled negatives.",
        },
        "audio_to_text": {
            "binary_ndcg": sparse_compute_metrics(
                similarity=similarity,
                relevance=audio_to_text_relevance,
                exponential_gain=False,
            ),
            "binary_retrieval": audio_to_text_binary,
        },
        "text_to_audio": {
            "binary_ndcg": sparse_compute_metrics(
                similarity=similarity.T,
                relevance=text_to_audio_relevance,
                exponential_gain=False,
            ),
            "binary_retrieval": text_to_audio_binary,
        },
        "balanced_pairwise": {
            "requested_num_negatives": 999,
            "audio_to_text": balanced_edge_retrieval_metrics(
                similarity=similarity,
                relevance=audio_to_text_relevance,
                num_negatives=999,
                seed=0,
            ),
            "text_to_audio": balanced_edge_retrieval_metrics(
                similarity=similarity.T,
                relevance=text_to_audio_relevance,
                num_negatives=999,
                seed=0,
            ),
        },
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
