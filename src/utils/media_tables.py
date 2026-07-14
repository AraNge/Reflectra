from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from src.utils.json import read_jsonl


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


def bytes_value(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    raise TypeError(f"Expected bytes-like value, got {type(value)!r}")


def resolve_media_path(
    path_value: str,
    project_root: Path,
    dataset_dir: Path | None = None,
) -> Path:
    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()

    if dataset_dir is not None:
        candidate = dataset_dir / path_value
        if candidate.exists():
            return candidate.resolve()

    candidate = project_root / path_value
    if candidate.exists():
        return candidate.resolve()

    return path


def image_suffix(image_bytes: bytes, fallback_path: str | None) -> str:
    if fallback_path:
        suffix = Path(fallback_path).suffix
        if suffix:
            return suffix

    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as image:
            image_format = (image.format or "").lower()
        if image_format == "jpeg":
            return ".jpg"
        if image_format:
            return f".{image_format}"
    except Exception:
        pass

    return ".img"


def load_unpacked_media_index(
    path: Path,
    id_column: str,
    path_column: str,
    dataset_dir: Path,
    required_ids: set[str] | None = None,
    project_root: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Load one of the unpacked benchmark index files written by
    src.datasets.downloaders.download_reflectra_benchmark
    (image_table.jsonl / audio_table.jsonl).

    Each JSONL row looks like:
        {"image_id": "...", "captions": [...], "image_path": "data/benchmark/images/xyz.jpg"}

    Paths in the file are relative to the project root, but this also
    tolerates already-absolute paths or paths relative to dataset_dir.
    """

    if project_root is None:
        from src.datasets.paths import PROJECT_ROOT as project_root  # noqa: N813

    records: dict[str, dict[str, Any]] = {}

    for row in read_jsonl(path):
        media_id = str(row[id_column])

        if required_ids is not None and media_id not in required_ids:
            continue

        resolved_path = resolve_media_path(
            str(row[path_column]),
            project_root=project_root,
            dataset_dir=dataset_dir,
        )

        records[media_id] = {
            id_column: media_id,
            "captions": as_list(row.get("captions")),
            path_column: str(resolved_path),
        }

    return records
