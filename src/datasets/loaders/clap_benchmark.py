from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets.paths import PROJECT_ROOT
from src.utils.json import read_jsonl
from src.utils.media_tables import as_list


CLAP_BENCHMARK_NAME = "clap_llm_benchmark.jsonl"
CLAP_AUDIO_TABLE_NAME = "audio_table.jsonl"


def resolve_clap_benchmark_paths(path: Path) -> tuple[Path, Path]:
    benchmark_dir = path
    if not benchmark_dir.is_dir():
        raise FileNotFoundError(
            f"CLAP benchmark must be an unpacked directory: {path}"
        )

    benchmark_path = benchmark_dir / CLAP_BENCHMARK_NAME
    audio_table_path = benchmark_dir / CLAP_AUDIO_TABLE_NAME

    if not benchmark_path.exists():
        raise FileNotFoundError(f"CLAP benchmark metadata not found: {benchmark_path}")
    if not audio_table_path.exists():
        raise FileNotFoundError(f"CLAP audio metadata not found: {audio_table_path}")

    return benchmark_path, audio_table_path


def load_clap_benchmark_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix != ".jsonl":
        raise ValueError("CLAP benchmark metadata must be a .jsonl file.")

    rows = []
    for row in read_jsonl(path):
        audio_ids = [
            str(audio_id)
            for audio_id in as_list(row["audio_ids"])
        ]
        scores = [int(score) for score in as_list(row["scores"])]
        if not audio_ids:
            raise ValueError(f"No audio_ids for {row['query_id']}.")
        if len(audio_ids) != len(scores):
            raise ValueError(f"Mismatched audio_ids/scores for {row['query_id']}.")

        relevance = row.get("relevance", {})
        if not isinstance(relevance, dict):
            relevance = {
                audio_id: score
                for audio_id, score in zip(audio_ids, scores)
            }

        rows.append(
            {
                "query_id": str(row["query_id"]),
                "caption": str(row["caption"]),
                "audio_ids": audio_ids,
                "scores": scores,
                "relevance": {
                    str(audio_id): int(score)
                    for audio_id, score in relevance.items()
                },
            }
        )

    return rows


def resolve_path(path_value: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()

    candidate = project_root / path_value
    if candidate.exists():
        return candidate.resolve()

    return path


def load_clap_audio_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix != ".jsonl":
        raise ValueError("CLAP audio metadata must be a .jsonl file.")

    audio_by_id = {}
    for row in read_jsonl(path):
        audio_id = str(row["audio_id"])
        audio_path = resolve_path(str(row["audio_path"]))
        audio_by_id[audio_id] = {
            "audio_id": audio_id,
            "captions": as_list(row.get("captions")),
            "audio_path": str(audio_path),
        }

    return audio_by_id
