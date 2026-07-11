from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.json import read_jsonl
from src.utils.media_tables import as_list


def resolve_reflectra_benchmark_paths(path: Path) -> tuple[Path, Path]:
    if path.is_dir():
        score_path = path / "image_audio_scores.parquet"
        benchmark_dir = path
    else:
        score_path = path
        benchmark_dir = path.parent

    if not score_path.exists():
        raise FileNotFoundError(f"Benchmark scores not found: {score_path}")

    return score_path, benchmark_dir


def resolve_clap_benchmark_paths(path: Path) -> tuple[Path, Path]:
    benchmark_dir = path
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(
            f"CLAP benchmark must be a directory containing benchmark parquet files: {path}"
        )

    benchmark_path = benchmark_dir / "clap_llm_benchmark.parquet"
    audio_table_path = benchmark_dir / "audio_table.parquet"

    if not benchmark_path.exists():
        raise FileNotFoundError(f"CLAP benchmark table not found: {benchmark_path}")
    if not audio_table_path.exists():
        raise FileNotFoundError(f"Audio table not found: {audio_table_path}")

    return benchmark_path, audio_table_path


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


def group_reflectra_pair_score_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[str, float]]] = {}

    for row in rows:
        grouped.setdefault(str(row["image_id"]), []).append(
            (str(row["audio_id"]), float(row["score"]))
        )

    return [
        {
            "image_id": image_id,
            "audio_ids": [audio_id for audio_id, _ in sorted(pairs)],
            "scores": [score for _, score in sorted(pairs)],
        }
        for image_id, pairs in sorted(grouped.items())
    ]


def load_reflectra_score_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
        rows = frame.to_dict(orient="records")
        columns = set(frame.columns)
        if {"image_id", "audio_ids", "scores"}.issubset(columns):
            return normalize_reflectra_score_rows(rows)
        if {"image_id", "audio_id", "score"}.issubset(columns):
            return normalize_reflectra_score_rows(
                group_reflectra_pair_score_rows(rows)
            )
        raise ValueError(
            "Parquet benchmark scores must contain either image_id/audio_ids/scores "
            "or image_id/audio_id/score columns."
        )
    if path.suffix == ".jsonl":
        return normalize_reflectra_score_rows(
            group_reflectra_pair_score_rows(read_jsonl(path))
        )

    raise ValueError("Reflectra benchmark scores must be a .parquet or .jsonl file.")


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


def load_clap_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".parquet":
        raw_rows = pd.read_parquet(path).to_dict(orient="records")
    elif path.suffix == ".jsonl":
        raw_rows = read_jsonl(path)
    else:
        raise ValueError("CLAP benchmark must be a .jsonl or .parquet file.")

    rows = []
    for row in raw_rows:
        candidate_audio_ids = [
            str(audio_id)
            for audio_id in as_list(row["candidate_audio_ids"])
        ]
        scores = [int(score) for score in as_list(row["scores"])]
        relevance = row.get("relevance", {})
        if not isinstance(relevance, dict):
            relevance = {
                audio_id: score
                for audio_id, score in zip(candidate_audio_ids, scores)
            }

        rows.append(
            {
                "query_id": str(row["query_id"]),
                "caption": str(row["caption"]),
                "positive_audio_id": str(row["positive_audio_id"]),
                "candidate_audio_ids": candidate_audio_ids,
                "scores": scores,
                "relevance": {
                    str(audio_id): int(score)
                    for audio_id, score in relevance.items()
                },
            }
        )

    return rows
