from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_hash_id(
    *parts: Any,
    prefix: str = "",
    length: int = 32,
) -> str:
    """Return a deterministic, JSON-canonical SHA-256 identifier."""
    if length < 8 or length > 64:
        raise ValueError("length must be between 8 and 64")

    payload = json.dumps(
        parts,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}{digest}"


def portable_path_key(
    path: str | Path,
    project_root: str | Path | None = None,
) -> str:
    """
    Produce a stable path key.

    Paths inside project_root are represented relative to it, so collaborators
    can keep the same IDs even when their project is checked out elsewhere.
    """
    resolved_path = Path(path).expanduser().resolve()

    if project_root is not None:
        resolved_root = Path(project_root).expanduser().resolve()

        try:
            return resolved_path.relative_to(resolved_root).as_posix()
        except ValueError:
            pass

    return resolved_path.as_posix()


def make_global_media_id(
    media_type: str,
    source_dataset: str,
    source_id: str,
    media_path: str | Path,
    project_root: str | Path | None = None,
) -> str:
    """
    Create a namespace-safe ID that replaces a dataset-local media ID.

    The original dataset ID is still preserved separately as source_*_id.
    """
    media_type = str(media_type).strip().lower()

    if media_type not in {"image", "audio"}:
        raise ValueError("media_type must be 'image' or 'audio'")

    return stable_hash_id(
        media_type,
        str(source_dataset).strip(),
        str(source_id).strip(),
        portable_path_key(media_path, project_root),
        prefix=f"{media_type}_",
    )


def make_pair_id(image_id: str, audio_id: str) -> str:
    return stable_hash_id(
        "image_audio_pair",
        str(image_id),
        str(audio_id),
        prefix="pair_",
    )
