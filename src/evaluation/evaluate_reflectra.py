from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.config import get_nested, load_config
from src.datasets.paths import PROJECT_ROOT
from src.metrics.retrieval_metrics import SparseRelevance, sparse_retrieval_metrics
from src.models.reflectra_model import ReflectraModel
from src.utils.benchmark_tables import (
    load_reflectra_score_rows,
    referenced_reflectra_media_ids,
    resolve_reflectra_benchmark_paths,
)
from src.utils.json import write_json
from src.utils.media_tables import load_unpacked_media_index


DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation_results" / "reflectra_eval_results.json"


def build_eval_inputs(
    score_rows: list[dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    audio_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], SparseRelevance]:
    image_ids = [row["image_id"] for row in score_rows]
    audio_ids = sorted(
        {
            audio_id
            for row in score_rows
            for audio_id in row["audio_ids"]
        }
    )

    missing_images = sorted(set(image_ids) - set(images_by_id))
    missing_audio = sorted(set(audio_ids) - set(audio_by_id))

    if missing_images:
        raise RuntimeError(f"Missing image_table.jsonl rows for IDs: {missing_images[:10]}")
    if missing_audio:
        raise RuntimeError(f"Missing audio_table.jsonl rows for IDs: {missing_audio[:10]}")

    audio_index_by_id = {
        audio_id: index
        for index, audio_id in enumerate(audio_ids)
    }
    relevance: SparseRelevance = []

    for row in score_rows:
        query_relevance = {}
        for audio_id, score in zip(row["audio_ids"], row["scores"]):
            if float(score) > 0:
                query_relevance[audio_index_by_id[audio_id]] = float(score)
        relevance.append(query_relevance)

    image_paths = [
        images_by_id[image_id]["image_path"]
        for image_id in image_ids
    ]
    audio_paths = [
        audio_by_id[audio_id]["audio_path"]
        for audio_id in audio_ids
    ]
    return image_paths, audio_paths, relevance


def load_reflectra_model(args: argparse.Namespace) -> ReflectraModel:
    model = ReflectraModel(
        clip_model_name=args.clip_model,
        clap_model_name=args.clap_model,
        projection_type=args.projection_type,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        device=args.device,
        projection_checkpoint=args.checkpoint,
        use_reranker=args.use_reranker,
        reranker_type=args.reranker_type,
        reranker_hidden_dim=args.reranker_hidden_dim,
        reranker_checkpoint=args.reranker_checkpoint,
        reranker_top_k=args.top_k,
    )

    model.eval()
    return model


def encode_in_batches(encode_fn, paths: list[str], batch_size: int) -> torch.Tensor:
    batches = []

    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        with torch.no_grad():
            batches.append(encode_fn(batch_paths).cpu())

    return torch.cat(batches, dim=0)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Evaluate Reflectra image-to-audio retrieval on the unpacked benchmark.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK_PATH),
        help=(
            "Unpacked benchmark directory containing image_audio_scores.jsonl, "
            "image_table.jsonl, audio_table.jsonl, images/, and audio/. "
            "Run python -m src.datasets.downloaders.download_reflectra_benchmark first."
        ),
    )
    parser.add_argument(
        "--clip_model",
        default=get_nested(config, "models", "clip", "openai/clip-vit-base-patch32"),
    )
    parser.add_argument(
        "--clap_model",
        default=get_nested(config, "models", "clap", "laion/clap-htsat-unfused"),
    )
    parser.add_argument(
        "--projection_type",
        choices=["mlp", "linear"],
        default="mlp",
    )
    parser.add_argument("--projection_hidden_dim", type=int, default=1024)
    parser.add_argument("--projection_dropout", type=float, default=0.1)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Projection checkpoint filename (auto-resolved under checkpoints/) or full path.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--relevance-threshold", type=float, default=0.0)

    # --- Reranker (second stage) flags. Run this script once without
    # --use_reranker (baseline) and once with it, to compare directly. ---
    parser.add_argument(
        "--use_reranker",
        action="store_true",
        help="Enable the cross-encoder reranker on top of bi-encoder retrieval.",
    )
    parser.add_argument(
        "--reranker_checkpoint",
        default=None,
        help="Trained reranker checkpoint filename or path (required if --use_reranker is set).",
    )
    parser.add_argument(
        "--reranker_type",
        default=get_nested(config, "reranker", "type", "mlp"),
        choices=["mlp", "attention"],
    )
    parser.add_argument(
        "--reranker_hidden_dim",
        type=int,
        default=get_nested(config, "reranker", "hidden_dim", 512),
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=get_nested(config, "reranker", "top_k", 20),
        help="Number of top bi-encoder candidates per image that get rescored by the reranker.",
    )

    args = parser.parse_args()

    if args.use_reranker and not args.reranker_checkpoint:
        parser.error("--use_reranker requires --reranker_checkpoint")

    return args


def main() -> None:
    args = parse_args()
    benchmark_path, benchmark_dir = resolve_reflectra_benchmark_paths(
        Path(args.benchmark).expanduser().resolve()
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_rows = load_reflectra_score_rows(benchmark_path)
    required_image_ids, required_audio_ids = referenced_reflectra_media_ids(score_rows)

    images_by_id = load_unpacked_media_index(
        path=benchmark_dir / "image_table.jsonl",
        id_column="image_id",
        path_column="image_path",
        dataset_dir=benchmark_dir,
        required_ids=required_image_ids,
    )
    audio_by_id = load_unpacked_media_index(
        path=benchmark_dir / "audio_table.jsonl",
        id_column="audio_id",
        path_column="audio_path",
        dataset_dir=benchmark_dir,
        required_ids=required_audio_ids,
    )

    image_paths, audio_paths, relevance = build_eval_inputs(
        score_rows=score_rows,
        images_by_id=images_by_id,
        audio_by_id=audio_by_id,
    )

    if not image_paths or not audio_paths:
        raise RuntimeError("No valid benchmark media found.")

    model = load_reflectra_model(args)

    image_embeddings = encode_in_batches(model.encode_image, image_paths, args.batch_size)
    audio_embeddings = encode_in_batches(
        lambda batch: model.encode_audio(batch, normalize=True),
        audio_paths,
        args.batch_size,
    )

    # Bi-encoder similarity is always computed first, regardless of whether
    # the reranker is enabled, so a baseline run and a reranked run are
    # directly comparable.
    similarity = image_embeddings @ audio_embeddings.T

    if args.use_reranker and model.reranker is not None:
        from src.models.reranker import rerank_topk_similarity

        similarity = rerank_topk_similarity(
            similarity=similarity,
            query_embeds=image_embeddings,
            candidate_embeds=audio_embeddings,
            reranker=model.reranker,
            top_k=args.top_k,
        )

    similarity = similarity.numpy()

    metrics = sparse_retrieval_metrics(
        similarity=similarity,
        relevance=relevance,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )

    results = {
        "benchmark": str(benchmark_dir),
        "num_images": len(image_paths),
        "num_audio": len(audio_paths),
        "num_relevance_labels": sum(len(row) for row in relevance),
        "clip_model": args.clip_model,
        "clap_model": args.clap_model,
        "checkpoint": args.checkpoint,
        "reranker_used": bool(args.use_reranker),
        "reranker_checkpoint": args.reranker_checkpoint if args.use_reranker else None,
        "reranker_type": args.reranker_type if args.use_reranker else None,
        "reranker_top_k": args.top_k if args.use_reranker else None,
        "metric_notes": {
            "ndcg": "Uses benchmark LLM scores as graded relevance.",
            "binary_metrics": (
                "MRR, mAP, recall, and precision treat scores above "
                f"{args.relevance_threshold} as relevant."
            ),
        },
        "image_to_audio": metrics,
    }

    print(json.dumps(results, indent=2))
    write_json(output_path, results)
    print(f"Saved results to: {output_path}")


"""
Run twice to compare:

# 1) baseline, bi-encoder only
python -m src.evaluation.evaluate_reflectra \
  --checkpoint reflectra_projection_flickr30k_1000.pt \
  --output evaluation_results/reflectra_eval_baseline.json

# 2) with the trained reranker
python -m src.evaluation.evaluate_reflectra \
  --checkpoint reflectra_projection_flickr30k_1000.pt \
  --use_reranker \
  --reranker_checkpoint reflectra_reranker_mlp_20260714_153000.pt \
  --output evaluation_results/reflectra_eval_reranked.json
"""

if __name__ == "__main__":
    main()
