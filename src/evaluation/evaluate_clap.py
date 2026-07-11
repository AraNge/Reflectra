from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.metrics.retrieval_metrics import SparseRelevance, sparse_retrieval_metrics
from src.models.clap_encoder import PretrainedCLAPEncoder
from src.utils.batch_encoding import encode_in_batches_clap
from src.utils.benchmark_tables import (
    load_clap_benchmark_rows,
    resolve_clap_benchmark_paths,
)
from src.utils.json import write_json
from src.utils.media_tables import load_audio_table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
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
            for audio_id in row.get("candidate_audio_ids", [])
            if str(audio_id) not in audio_by_id
        }
    )

    if missing_audio_ids:
        preview = ", ".join(missing_audio_ids[:10])
        raise RuntimeError(
            "Benchmark references audio IDs missing from audio_table.parquet: "
            f"{preview}"
        )

    candidate_counts = {
        len(row.get("candidate_audio_ids", []))
        for row in rows
    }
    if len(candidate_counts) != 1:
        raise RuntimeError(
            "All benchmark rows must have the same number of candidates. "
            f"Found counts: {sorted(candidate_counts)}"
        )


def build_similarity_and_relevance(
    rows: list[dict[str, Any]],
    audio_by_id: dict[str, dict[str, Any]],
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, SparseRelevance, dict[str, Any]]:
    audio_ids = sorted(
        {
            str(audio_id)
            for row in rows
            for audio_id in row["candidate_audio_ids"]
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

    max_candidates = max(len(row["candidate_audio_ids"]) for row in rows)
    relevance: SparseRelevance = []

    for row in rows:
        query_relevance = {}

        for candidate_idx, audio_id in enumerate(row["candidate_audio_ids"]):
            audio_id = str(audio_id)
            global_audio_idx = audio_index_by_id[audio_id]

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

    stats = {
        "num_queries": len(rows),
        "num_unique_audio": len(audio_ids),
        "max_candidates_per_query": max_candidates,
        "num_relevance_labels": sum(len(row) for row in relevance),
    }
    return global_similarity, relevance, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CLAP on an LLM-scored caption-to-audio benchmark "
            "created by src.benchmark.create_clap_benchmark."
        )
    )
    parser.add_argument(
        "--benchmark_dir",
        type=str,
        default=str(DEFAULT_BENCHMARK_DIR),
        help="Directory containing clap_llm_benchmark.parquet and audio_table.parquet.",
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
    temp_parent = output_path.parent / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="clap_eval_",
        dir=temp_parent,
    ) as temp_dir:
        rows = load_clap_benchmark_rows(benchmark_path)
        required_audio_ids = {
            str(audio_id)
            for row in rows
            for audio_id in row["candidate_audio_ids"]
        }
        audio_by_id = load_audio_table(
            path=audio_table_path,
            materialize_dir=Path(temp_dir),
            project_root=PROJECT_ROOT,
            required_ids=required_audio_ids,
        )
        validate_benchmark_rows(rows, audio_by_id)

        similarity, relevance, stats = build_similarity_and_relevance(
            rows=rows,
            audio_by_id=audio_by_id,
            model_name=args.model_name,
            batch_size=args.batch_size,
        )
        
    metrics = sparse_retrieval_metrics(
        similarity=similarity,
        relevance=relevance,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )

    results = {
        "benchmark_dir": str(benchmark_dir),
        "benchmark": str(benchmark_path),
        "audio_table": str(audio_table_path),
        "model_name": args.model_name,
        "metric_notes": {
            "ndcg": "Uses the LLM 0..10 score as graded relevance.",
            "binary_metrics": (
                "MRR, mAP, recall, and precision treat scores above "
                f"{args.relevance_threshold} as relevant."
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
