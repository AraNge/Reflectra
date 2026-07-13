from __future__ import annotations

import io
from pathlib import Path
from typing import Any


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
