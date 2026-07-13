from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.config import get_nested, load_config
from src.metrics.retrieval_metrics import SparseRelevance, sparse_retrieval_metrics
from src.models.reflectra_model import ReflectraModel
from src.utils.benchmark_tables import (
    load_reflectra_score_rows,
    referenced_reflectra_media_ids,
    resolve_reflectra_benchmark_paths,
)
from src.utils.json import read_jsonl, write_json
from src.utils.media_tables import resolve_media_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "data" / "benchmark"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "evaluation_results" / "reflectra_eval_results.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATTERNS = ("*.pt", "*.pth")


def build_eval_inputs(
    score_rows: list[dict[str, Any]],
    images_by_id: dict[str, dict[str, Any]],
    audio_by_id: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str], SparseRelevance, list[list[int]]]:
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
        raise RuntimeError(f"Missing benchmark image rows for IDs: {missing_images[:10]}")
    if missing_audio:
        raise RuntimeError(f"Missing benchmark audio rows for IDs: {missing_audio[:10]}")

    audio_index_by_id = {
        audio_id: index
        for index, audio_id in enumerate(audio_ids)
    }
    relevance: SparseRelevance = []
    candidate_indices: list[list[int]] = []

    for row in score_rows:
        query_relevance = {}
        query_candidates = []
        for audio_id, score in zip(row["audio_ids"], row["scores"]):
            audio_index = audio_index_by_id[audio_id]
            query_candidates.append(audio_index)
            if float(score) > 0:
                query_relevance[audio_index] = float(score)
        relevance.append(query_relevance)
        candidate_indices.append(query_candidates)

    image_paths = [
        images_by_id[image_id]["image_path"]
        for image_id in image_ids
    ]
    audio_paths = [
        audio_by_id[audio_id]["audio_path"]
        for audio_id in audio_ids
    ]
    return image_paths, audio_paths, relevance, candidate_indices


def load_unpacked_media_index(
    path: Path,
    id_column: str,
    path_column: str,
    dataset_dir: Path,
    required_ids: set[str],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing unpacked benchmark index: {path}. "
            "Run python -m src.datasets.downloaders.download_reflectra_benchmark first."
        )

    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        media_id = str(row[id_column])
        if media_id not in required_ids:
            continue

        if path_column not in row:
            raise ValueError(f"{path.name} row {media_id} is missing {path_column}.")

        media_path = resolve_media_path(
            path_value=str(row[path_column]),
            project_root=PROJECT_ROOT,
            dataset_dir=dataset_dir,
        )
        if not media_path.exists():
            raise FileNotFoundError(f"Unpacked media file not found: {media_path}")

        records[media_id] = {
            id_column: media_id,
            path_column: str(media_path),
            "captions": row.get("captions", []),
        }

    return records


def checkpoint_has_projection_head(path: Path) -> bool:
    try:
        checkpoint = torch.load(path, map_location="cpu")
        checkpoint["projection_state_dict"]
    except Exception as error:
        print(f"Skipping checkpoint without projection_state_dict {path}: {error}")
        return False

    return True


def find_default_projection_checkpoint() -> Path | None:
    if not DEFAULT_CHECKPOINT_DIR.exists():
        return None

    candidates: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        candidates.extend(DEFAULT_CHECKPOINT_DIR.glob(pattern))

    for path in sorted(set(candidates)):
        if checkpoint_has_projection_head(path):
            return path

    return None


def load_reflectra_model(args: argparse.Namespace) -> ReflectraModel:
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_default_projection_checkpoint()
        if checkpoint is None:
            print(f"No projection checkpoint found in {DEFAULT_CHECKPOINT_DIR}.")
        else:
            print(f"Using projection checkpoint: {checkpoint}")
            args.checkpoint = str(checkpoint)

    model = ReflectraModel(
        clip_model_name=args.clip_model,
        clap_model_name=args.clap_model,
        projection_type=args.projection_type,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        projection_checkpoint=checkpoint,
    )

    model.eval()
    return model


def encode_audio_in_batches(
    model: ReflectraModel,
    audio_paths: list[str],
    batch_size: int,
) -> torch.Tensor:
    batches = []

    for start in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[start:start + batch_size]
        with torch.no_grad():
            batches.append(model.encode_audio(batch_paths, normalize=True).cpu())

    return torch.cat(batches, dim=0)


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="Evaluate Reflectra image-to-audio retrieval on the created benchmark.",
        parents=[config_parser],
    )
    parser.add_argument(
        "--benchmark",
        default=str(DEFAULT_BENCHMARK_PATH),
        help=(
            "Path to an unpacked benchmark directory containing images/, audio/, "
            "image_audio_scores.jsonl, image_table.jsonl, and audio_table.jsonl."
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
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--relevance-threshold", type=float, default=0.0)
    return parser.parse_args()


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
    image_paths, audio_paths, relevance, candidate_indices = build_eval_inputs(
        score_rows=score_rows,
        images_by_id=images_by_id,
        audio_by_id=audio_by_id,
    )

    if not image_paths or not audio_paths:
        raise RuntimeError("No valid benchmark media found.")

    model = load_reflectra_model(args)
    audio_embeddings = encode_audio_in_batches(
        model=model,
        audio_paths=audio_paths,
        batch_size=args.batch_size,
    )

    similarities = []

    for start in range(0, len(image_paths), args.batch_size):
        batch_image_paths = image_paths[start:start + args.batch_size]
        with torch.no_grad():
            image_embeddings = model.encode_image(batch_image_paths).cpu()
            batch_similarity = image_embeddings @ audio_embeddings.T
        similarities.append(batch_similarity.cpu())

    similarity = torch.cat(similarities, dim=0).numpy()
    metrics = sparse_retrieval_metrics(
        similarity=similarity,
        relevance=relevance,
        candidate_indices=candidate_indices,
        threshold=args.relevance_threshold,
        exponential_gain=True,
    )

    results = {
        "benchmark": str(benchmark_path),
        "num_images": len(image_paths),
        "num_audio": len(audio_paths),
        "num_relevance_labels": sum(len(row) for row in relevance),
        "clip_model": args.clip_model,
        "clap_model": args.clap_model,
        "checkpoint": str(model.resolve_checkpoint_path(args.checkpoint))
        if args.checkpoint is not None
        else None,
        "metric_notes": {
            "ndcg": "Uses benchmark LLM scores as graded relevance.",
            "binary_metrics": (
                "MRR, mAP, recall, and precision treat scores above "
                f"{args.relevance_threshold} as relevant."
            ),
            "candidate_scope": (
                "Metrics are computed only over audios scored for each image; "
                "unevaluated audios are not treated as irrelevant."
            ),
        },
        "image_to_audio": metrics,
    }

    print(json.dumps(results, indent=2))
    write_json(output_path, results)
    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
