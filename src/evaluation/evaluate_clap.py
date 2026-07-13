from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.datasets.loaders import (
    load_clap_audio_metadata,
    load_clap_benchmark_rows,
    resolve_clap_benchmark_paths,
)
from src.metrics.retrieval_metrics import SparseRelevance, sparse_retrieval_metrics
from src.models.clap_encoder import PretrainedCLAPEncoder
from src.utils.batch_encoding import encode_in_batches_clap
from src.utils.json import write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "clap_benchmark"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation_results" / "clap_eval_results.json"


def validate_benchmark_rows(
    rows: list[dict[str, Any]],
    audio_by_id: dict[str, dict[str, Any]],
) -> None:
    if not rows:
        raise RuntimeError("No CLAP benchmark rows found.")

    missing_audio_ids = sorted(
        {
            str(audio_id)
            for row in rows
            for audio_id in row.get("audio_ids", [])
            if str(audio_id) not in audio_by_id
        }
    )

    if missing_audio_ids:
        preview = ", ".join(missing_audio_ids[:10])
        raise RuntimeError(
            "Benchmark references audio IDs missing from audio_table.jsonl: "
            f"{preview}"
        )

    audio_counts = {
        len(row.get("audio_ids", []))
        for row in rows
    }
    if len(audio_counts) != 1:
        raise RuntimeError(
            "All benchmark rows must have the same number of audio_ids. "
            f"Found counts: {sorted(audio_counts)}"
        )


def build_similarity_and_relevance(
    rows: list[dict[str, Any]],
    audio_by_id: dict[str, dict[str, Any]],
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, SparseRelevance, list[list[int]], dict[str, Any]]:
    audio_ids = sorted(
        {
            str(audio_id)
            for row in rows
            for audio_id in row["audio_ids"]
        }
    )
    audio_index_by_id = {
        audio_id: index
        for index, audio_id in enumerate(audio_ids)
    }
    audio_paths = [
        audio_by_id[audio_id]["audio_path"]
        for audio_id in audio_ids
    ]
    texts = [str(row["caption"]) for row in rows]

    model = PretrainedCLAPEncoder(
        model_name=model_name,
        freeze=True,
    )
    audio_embeddings, text_embeddings = encode_in_batches_clap(
        model=model,
        audio_paths=audio_paths,
        texts=texts,
        batch_size=batch_size,
    )
    global_similarity = text_embeddings @ audio_embeddings.T
    global_similarity = global_similarity.numpy()

    max_candidates = max(len(row["audio_ids"]) for row in rows)
    relevance: SparseRelevance = []
    candidate_indices: list[list[int]] = []

    for row in rows:
        query_relevance = {}
        query_candidates = []

        for candidate_idx, audio_id in enumerate(row["audio_ids"]):
            audio_id = str(audio_id)
            global_audio_idx = audio_index_by_id[audio_id]
            query_candidates.append(global_audio_idx)

            relevance_by_audio_id = row.get("relevance", {})
            score = float(
                relevance_by_audio_id.get(
                    audio_id,
                    row["scores"][candidate_idx],
                )
            )
            if score > 0:
                query_relevance[global_audio_idx] = score

        relevance.append(query_relevance)
        candidate_indices.append(query_candidates)

    stats = {
        "num_queries": len(rows),
        "num_unique_audio": len(audio_ids),
        "max_candidates_per_query": max_candidates,
        "num_relevance_labels": sum(len(row) for row in relevance),
    }
    return global_similarity, relevance, candidate_indices, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CLAP on an LLM-scored caption-to-audio benchmark "
            "unpacked by src.datasets.downloaders.download_clap_benchmark."
        )
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=str(DEFAULT_BENCHMARK_DIR),
        help="Directory containing clap_llm_benchmark.jsonl and audio_table.jsonl.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--model-name", type=str, default="laion/clap-htsat-unfused")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=0.0,
        help="Scores above this threshold are relevant for MRR/mAP/recall/precision.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    benchmark_path, audio_table_path = resolve_clap_benchmark_paths(benchmark_dir)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_clap_benchmark_rows(benchmark_path)
    audio_by_id = load_clap_audio_metadata(audio_table_path)
    validate_benchmark_rows(rows, audio_by_id)

    similarity, relevance, candidate_indices, stats = build_similarity_and_relevance(
        rows=rows,
        audio_by_id=audio_by_id,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )

    metrics = sparse_retrieval_metrics(
        similarity=similarity,
        relevance=relevance,
        candidate_indices=candidate_indices,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )

    results = {
        "benchmark_dir": str(benchmark_dir),
        "benchmark": str(benchmark_path),
        "audio_metadata": str(audio_table_path),
        "model_name": args.model_name,
        "metric_notes": {
            "ndcg": "Uses the LLM 0..10 score as graded relevance.",
            "binary_metrics": (
                "MRR, mAP, recall, and precision treat scores above "
                f"{args.relevance_threshold} as relevant."
            ),
            "candidate_scope": (
                "Metrics are computed only over audios scored for each query; "
                "unevaluated audios are not treated as irrelevant."
            ),
        },
        **stats,
        "text_to_audio": metrics,
    }

    print(json.dumps(results, indent=2))
    write_json(output_path, results)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
