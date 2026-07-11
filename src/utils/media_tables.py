from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd


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


def resolve_media_path(path_value: str, project_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.exists():
        return path.resolve()

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


def materialize_image_record(
    row: dict[str, Any],
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    image_id = str(row["image_id"])
    source_path = row.get("image_path")

    if row.get("image") is None:
        if not source_path:
            raise ValueError(f"Image row has no image bytes or image_path: {image_id}")
        image_path = resolve_media_path(str(source_path), project_root)
    else:
        image_bytes = bytes_value(row["image"])
        suffix = image_suffix(image_bytes, source_path)
        image_path = output_dir / f"{image_id}{suffix}"
        image_path.write_bytes(image_bytes)

    return {
        "image_id": image_id,
        "captions": as_list(row.get("captions")),
        "image_path": str(image_path),
    }


def materialize_audio_record(
    row: dict[str, Any],
    output_dir: Path,
    project_root: Path,
) -> dict[str, Any]:
    audio_id = str(row["audio_id"])
    source_path = row.get("audio_path")

    if row.get("audio") is None:
        if not source_path:
            raise ValueError(f"Audio row has no audio bytes or audio_path: {audio_id}")
        audio_path = resolve_media_path(str(source_path), project_root)
    else:
        audio_bytes = bytes_value(row["audio"])
        suffix = Path(str(source_path)).suffix if source_path else ""

        if suffix:
            audio_path = output_dir / f"{audio_id}{suffix}"
            audio_path.write_bytes(audio_bytes)
        else:
            try:
                import soundfile as sf

                samples, sample_rate = sf.read(io.BytesIO(audio_bytes), always_2d=False)
                audio_path = output_dir / f"{audio_id}.wav"
                sf.write(audio_path, samples, sample_rate, format="WAV")
            except Exception:
                audio_path = output_dir / f"{audio_id}.bin"
                audio_path.write_bytes(audio_bytes)

    return {
        "audio_id": audio_id,
        "captions": as_list(row.get("captions")),
        "audio_path": str(audio_path),
    }


def media_table_paths(dataset_dir: Path) -> tuple[Path, Path]:
    return dataset_dir / "image_table.parquet", dataset_dir / "audio_table.parquet"


def has_media_tables(dataset_dir: Path) -> bool:
    image_table_path, audio_table_path = media_table_paths(dataset_dir)
    return image_table_path.exists() and audio_table_path.exists()


def load_image_table(
    path: Path,
    materialize_dir: Path,
    project_root: Path,
    required_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    output_dir = materialize_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in pd.read_parquet(path).to_dict(orient="records"):
        if required_ids is not None and str(row["image_id"]) not in required_ids:
            continue
        records.append(materialize_image_record(row, output_dir, project_root))

    return {record["image_id"]: record for record in records}


def load_audio_table(
    path: Path,
    materialize_dir: Path,
    project_root: Path,
    required_ids: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    output_dir = materialize_dir / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for row in pd.read_parquet(path).to_dict(orient="records"):
        if required_ids is not None and str(row["audio_id"]) not in required_ids:
            continue
        records.append(materialize_audio_record(row, output_dir, project_root))

    return {record["audio_id"]: record for record in records}


def load_media_tables(
    dataset_dir: Path,
    materialize_dir: Path,
    project_root: Path,
    required_image_ids: set[str] | None = None,
    required_audio_ids: set[str] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    image_table_path, audio_table_path = media_table_paths(dataset_dir)

    if not image_table_path.exists() or not audio_table_path.exists():
        raise FileNotFoundError(
            "Could not find compact parquet media tables. Expected "
            f"{image_table_path} and {audio_table_path}."
        )

    return (
        load_image_table(
            image_table_path,
            materialize_dir,
            project_root,
            required_ids=required_image_ids,
        ),
        load_audio_table(
            audio_table_path,
            materialize_dir,
            project_root,
            required_ids=required_audio_ids,
        ),
    )
