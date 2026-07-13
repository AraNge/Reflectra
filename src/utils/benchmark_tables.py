from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.json import read_jsonl


def resolve_reflectra_benchmark_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        score_path = path / "image_audio_scores.jsonl"
        benchmark_dir = path
    else:
        score_path = path
        benchmark_dir = path.parent

    if not score_path.exists():
        raise FileNotFoundError(f"Benchmark scores not found: {score_path}")
    if score_path.suffix != ".jsonl":
        raise ValueError(
            "Reflectra evaluation expects unpacked image_audio_scores.jsonl. "
            "Run python -m src.datasets.downloaders.download_reflectra_benchmark first."
        )

    return score_path, benchmark_dir


def normalize_reflectra_score_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    normalized_rows = []
    for row in rows:
        audio_ids = [str(audio_id) for audio_id in row["audio_ids"]]
        scores = [float(score) for score in row["scores"]]
        if len(audio_ids) != len(scores):
            raise ValueError(f"Mismatched audio_ids/scores for image {row['image_id']}")
        normalized_rows.append(
            {
                "image_id": str(row["image_id"]),
                "audio_ids": audio_ids,
                "scores": scores,
            }
        )

    return normalized_rows


def load_reflectra_score_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = read_jsonl(path)
        if not rows:
            return []
        columns = set(rows[0])
        if {"image_id", "audio_ids", "scores"}.issubset(columns):
            return normalize_reflectra_score_rows(rows)
        raise ValueError(
            "JSONL benchmark scores must contain image_id/audio_ids/scores fields."
        )

    raise ValueError("Reflectra benchmark scores must be unpacked .jsonl.")


def referenced_reflectra_media_ids(
    score_rows: list[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    image_ids = {row["image_id"] for row in score_rows}
    audio_ids = {
        audio_id
        for row in score_rows
        for audio_id in row["audio_ids"]
    }
    return image_ids, audio_ids
