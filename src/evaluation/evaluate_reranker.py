from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import get_nested, load_config
from src.metrics.retrieval_metrics import SparseRelevance, sparse_retrieval_metrics
from src.models.reflectra_model import ReflectraModel
from src.models.reranker import build_reranker
from src.utils.benchmark_tables import (
    load_reflectra_score_rows,
    referenced_reflectra_media_ids,
    resolve_reflectra_benchmark_paths,
)
from src.utils.json import write_json
from src.utils.media_tables import load_media_tables


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark" / "image_audio_scores.parquet"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation_results" / "reranker_eval_results.json"


def build_eval_inputs(
    score_rows: list[dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    audio_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], SparseRelevance]:
    image_ids = [row["image_id"] for row in score_rows]
    audio_ids = sorted(
        {audio_id for row in score_rows for audio_id in row["audio_ids"]}
    )

    missing_images = sorted(set(image_ids) - set(images_by_id))
    missing_audio = sorted(set(audio_ids) - set(audio_by_id))

    if missing_images:
        raise RuntimeError(f"Missing image_table.parquet rows for IDs: {missing_images[:10]}")
    if missing_audio:
        raise RuntimeError(f"Missing audio_table.parquet rows for IDs: {missing_audio[:10]}")

    audio_index_by_id = {audio_id: index for index, audio_id in enumerate(audio_ids)}
    relevance: SparseRelevance = []

    for row in score_rows:
        query_relevance = {}
        for audio_id, score in zip(row["audio_ids"], row["scores"]):
            if float(score) > 0:
                query_relevance[audio_index_by_id[audio_id]] = float(score)
        relevance.append(query_relevance)

    image_paths = [images_by_id[image_id]["image_path"] for image_id in image_ids]
    audio_paths = [audio_by_id[audio_id]["audio_path"] for audio_id in audio_ids]

    return image_paths, audio_paths, relevance


def _min_max_scale(values: np.ndarray) -> np.ndarray:
    value_min, value_max = float(values.min()), float(values.max())

    if value_max - value_min < 1e-8:
        return np.zeros_like(values)

    return (values - value_min) / (value_max - value_min)


def rerank_similarity(
    bi_encoder_similarity: np.ndarray,
    reranker: torch.nn.Module,
    image_embeddings: torch.Tensor,
    audio_embeddings: torch.Tensor,
    top_k: int,
    device: torch.device,
) -> np.ndarray:
    """
    Two-stage rerank: for every query, keep the bi-encoder ranking beyond
    top_k untouched, but replace the ordering *within* the top_k candidates
    with reranker scores, scaled so the whole reranked block sits above
    everything outside top_k. This mirrors how the reranker is actually used
    at serving time (rerank only what Qdrant returned).
    """

    reranked = bi_encoder_similarity.copy()
    reranker.eval()

    with torch.no_grad():
        for query_idx in range(bi_encoder_similarity.shape[0]):
            row = bi_encoder_similarity[query_idx]
            top_indices = np.argsort(-row)[:top_k]

            query_embed = image_embeddings[query_idx].to(device)
            candidate_embeds = audio_embeddings[top_indices].to(device)
            query_expanded = query_embed.unsqueeze(0).expand(candidate_embeds.size(0), -1)

            rerank_scores = reranker(query_expanded, candidate_embeds).cpu().numpy()

            floor = float(reranked[query_idx].min()) - 1.0
            reranked[query_idx, top_indices] = floor + 1.0 + _min_max_scale(rerank_scores)

    return reranked


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Compare bi-encoder-only vs bi-encoder+reranker retrieval on the LLM-graded benchmark.",
        parents=[config_parser],
    )
    parser.add_argument("--benchmark", default=str(DEFAULT_BENCHMARK_PATH))
    parser.add_argument("--clip_model", default=get_nested(config, "models", "clip", "openai/clip-vit-base-patch32"))
    parser.add_argument("--clap_model", default=get_nested(config, "models", "clap", "laion/clap-htsat-unfused"))
    parser.add_argument("--projection_type", choices=["mlp", "linear"], default="mlp")
    parser.add_argument("--projection_hidden_dim", type=int, default=1024)
    parser.add_argument("--projection_checkpoint", default=None)
    parser.add_argument("--reranker_checkpoint", required=True)
    parser.add_argument("--reranker_type", default=get_nested(config, "reranker", "type", "mlp"), choices=["mlp", "attention"])
    parser.add_argument("--reranker_hidden_dim", type=int, default=get_nested(config, "reranker", "hidden_dim", 512))
    parser.add_argument("--top_k", type=int, default=get_nested(config, "reranker", "top_k", 20))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--relevance-threshold", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if args.device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    benchmark_path, benchmark_dir = resolve_reflectra_benchmark_paths(
        Path(args.benchmark).expanduser().resolve()
    )
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_parent = output_path.parent / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="reranker_eval_", dir=temp_parent) as temp_dir:
        score_rows = load_reflectra_score_rows(benchmark_path)
        required_image_ids, required_audio_ids = referenced_reflectra_media_ids(score_rows)
        images_by_id, audio_by_id = load_media_tables(
            dataset_dir=benchmark_dir,
            materialize_dir=Path(temp_dir),
            project_root=PROJECT_ROOT,
            required_image_ids=required_image_ids,
            required_audio_ids=required_audio_ids,
        )
        image_paths, audio_paths, relevance = build_eval_inputs(
            score_rows=score_rows,
            images_by_id=images_by_id,
            audio_by_id=audio_by_id,
        )

        if not image_paths or not audio_paths:
            raise RuntimeError("No valid benchmark media found.")

        model = ReflectraModel(
            clip_model_name=args.clip_model,
            clap_model_name=args.clap_model,
            projection_type=args.projection_type,
            projection_hidden_dim=args.projection_hidden_dim,
            device=str(device),
        )
        if args.projection_checkpoint:
            projection_checkpoint = torch.load(args.projection_checkpoint, map_location=device)
            model.image_projection.load_state_dict(projection_checkpoint["projection_state_dict"])
        model.eval()

        reranker_checkpoint = torch.load(args.reranker_checkpoint, map_location=device)
        reranker = build_reranker(
            reranker_type=args.reranker_type,
            embed_dim=model.clap_dim,
            hidden_dim=args.reranker_hidden_dim,
        ).to(device)
        reranker.load_state_dict(reranker_checkpoint["reranker_state_dict"])
        reranker.eval()

        image_embeddings, audio_embeddings = [], []

        with torch.no_grad():
            for start in range(0, len(image_paths), args.batch_size):
                batch = image_paths[start:start + args.batch_size]
                image_embeddings.append(model.encode_image(batch).cpu())

            for start in range(0, len(audio_paths), args.batch_size):
                batch = audio_paths[start:start + args.batch_size]
                audio_embeddings.append(model.encode_audio(batch).cpu())

        image_embeddings = torch.cat(image_embeddings, dim=0)
        audio_embeddings = torch.cat(audio_embeddings, dim=0)

    bi_encoder_similarity = (image_embeddings @ audio_embeddings.T).numpy()

    reranked_similarity = rerank_similarity(
        bi_encoder_similarity=bi_encoder_similarity,
        reranker=reranker,
        image_embeddings=image_embeddings,
        audio_embeddings=audio_embeddings,
        top_k=args.top_k,
        device=device,
    )

    bi_encoder_metrics = sparse_retrieval_metrics(
        similarity=bi_encoder_similarity,
        relevance=relevance,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )
    reranked_metrics = sparse_retrieval_metrics(
        similarity=reranked_similarity,
        relevance=relevance,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )

    results = {
        "benchmark": str(benchmark_path),
        "num_images": len(image_paths),
        "num_audio": len(audio_paths),
        "top_k": args.top_k,
        "clip_model": args.clip_model,
        "clap_model": args.clap_model,
        "projection_checkpoint": args.projection_checkpoint,
        "reranker_checkpoint": args.reranker_checkpoint,
        "reranker_type": args.reranker_type,
        "metric_notes": {
            "ndcg": "Uses benchmark LLM scores as graded relevance.",
            "binary_metrics": (
                "MRR, mAP, recall, and precision treat scores above "
                f"{args.relevance_threshold} as relevant."
            ),
            "reranking": (
                f"Only the top_k={args.top_k} bi-encoder candidates per query are "
                "rescored; ranking beyond top_k is unchanged."
            ),
        },
        "bi_encoder_only": bi_encoder_metrics,
        "bi_encoder_plus_reranker": reranked_metrics,
    }

    print(json.dumps(results, indent=2))
    write_json(output_path, results)
    print(f"Saved results to: {output_path}")


"""
python -m src.evaluation.evaluate_reranker \
  --benchmark data/benchmark/image_audio_scores.parquet \
  --projection_checkpoint checkpoints/reflectra_projection_flickr30k_1000.pt \
  --reranker_checkpoint checkpoints/reflectra_reranker.pt \
  --top_k 20
"""

if __name__ == "__main__":
    main()
