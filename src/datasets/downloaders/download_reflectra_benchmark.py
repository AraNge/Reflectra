from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm

from src.datasets.paths import DATA_DIR, HF_CACHE_DIR, PROJECT_ROOT, ensure_data_dirs
from src.utils.json import read_json, write_json, write_jsonl
from src.utils.media_tables import as_list, bytes_value, image_suffix


REFLECTRA_BENCHMARK_REPO_ID = "AraNge/reflectra-benchmark"
DEFAULT_OUTPUT_DIR = DATA_DIR / "benchmark"

SCORE_TABLE_NAME = "image_audio_scores.parquet"
IMAGE_TABLE_NAME = "image_table.parquet"
AUDIO_TABLE_NAME = "audio_table.parquet"
SCORE_INDEX_NAME = "image_audio_scores.jsonl"
IMAGE_INDEX_NAME = "image_table.jsonl"
AUDIO_INDEX_NAME = "audio_table.jsonl"
MANIFEST_NAME = "benchmark_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Reflectra image/audio benchmark from Hugging Face "
            "and unpack embedded media into data/benchmark."
        )
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, or commit.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Local output directory. Default: {DEFAULT_OUTPUT_DIR}.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Parquet rows to unpack at a time. Default: 64.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite media files that already exist.",
    )
    return parser.parse_args()


def safe_filename(value: Any) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(",", "_")
    )


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def download_benchmark_file(
    repo_id: str,
    filename: str,
    revision: str | None,
) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            revision=revision,
            cache_dir=str(HF_CACHE_DIR),
        )
    )


def maybe_download_manifest(
    repo_id: str,
    output_dir: Path,
    revision: str | None,
) -> dict[str, Any]:
    try:
        path = download_benchmark_file(
            repo_id=repo_id,
            filename=MANIFEST_NAME,
            revision=revision,
        )
    except EntryNotFoundError:
        return {}

    manifest = read_json(path)
    shutil.copy2(path, output_dir / MANIFEST_NAME)
    return manifest


def ensure_embedded_media_columns(
    path: Path,
    id_column: str,
    bytes_column: str,
) -> None:
    columns = set(pq.read_schema(path).names)
    required_columns = {id_column, bytes_column}
    missing_columns = required_columns - columns

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing {sorted(missing_columns)}. "
            "Upload byte-backed benchmark Parquet tables so the downloader "
            "can unpack media from Parquet-only files."
        )


def audio_suffix(row: dict[str, Any]) -> str:
    source_path = row.get("audio_path")
    if source_path:
        suffix = Path(str(source_path)).suffix
        if suffix:
            return suffix
    return ".wav"


def unpack_media_table(
    source_path: Path,
    output_index_path: Path,
    media_dir: Path,
    id_column: str,
    bytes_column: str,
    path_column: str,
    batch_size: int,
    overwrite: bool,
) -> int:
    parquet_file = pq.ParquetFile(source_path)
    media_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    progress = tqdm(
        total=parquet_file.metadata.num_rows,
        desc=f"Unpack {bytes_column}",
        unit="row",
    )
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            media_id = str(row[id_column])
            media_bytes = row.get(bytes_column)
            source_media_path = row.get(path_column)

            if bytes_column == "image":
                suffix = image_suffix(
                    bytes_value(media_bytes) if media_bytes is not None else b"",
                    str(source_media_path) if source_media_path else None,
                )
            else:
                suffix = audio_suffix(row)

            media_path = media_dir / f"{safe_filename(media_id)}{suffix}"
            if media_bytes is not None and (overwrite or not media_path.exists()):
                media_path.write_bytes(bytes_value(media_bytes))

            if media_bytes is None and source_media_path:
                media_path = Path(str(source_media_path)).expanduser()

            records.append(
                {
                    id_column: media_id,
                    "captions": as_list(row.get("captions")),
                    path_column: relative_to_project(media_path),
                }
            )
            progress.update(1)

    progress.close()
    write_jsonl(output_index_path, records)
    return len(records)


def write_score_index(source_path: Path, output_path: Path) -> int:
    frame = pd.read_parquet(source_path)
    columns = set(frame.columns)

    if {"image_id", "audio_ids", "scores"}.issubset(columns):
        records = [
            {
                "image_id": str(row["image_id"]),
                "audio_ids": [str(audio_id) for audio_id in as_list(row["audio_ids"])],
                "scores": [float(score) for score in as_list(row["scores"])],
            }
            for row in frame.to_dict(orient="records")
        ]
    elif {"image_id", "audio_id", "score"}.issubset(columns):
        grouped: dict[str, list[tuple[str, float]]] = {}
        for row in frame.to_dict(orient="records"):
            grouped.setdefault(str(row["image_id"]), []).append(
                (str(row["audio_id"]), float(row["score"]))
            )
        records = [
            {
                "image_id": image_id,
                "audio_ids": [audio_id for audio_id, _ in sorted(pairs)],
                "scores": [score for _, score in sorted(pairs)],
            }
            for image_id, pairs in sorted(grouped.items())
        ]
    else:
        raise ValueError(
            f"{source_path.name} must contain either image_id/audio_ids/scores "
            "or image_id/audio_id/score columns."
        )

    write_jsonl(output_path, records)
    return len(records)


def download_reflectra_benchmark(
    output_dir: Path,
    revision: str | None,
    batch_size: int,
    overwrite: bool,
) -> None:
    ensure_data_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading Reflectra benchmark from: {REFLECTRA_BENCHMARK_REPO_ID}")
    score_path = download_benchmark_file(
        REFLECTRA_BENCHMARK_REPO_ID,
        SCORE_TABLE_NAME,
        revision,
    )
    image_table_path = download_benchmark_file(
        REFLECTRA_BENCHMARK_REPO_ID,
        IMAGE_TABLE_NAME,
        revision,
    )
    audio_table_path = download_benchmark_file(
        REFLECTRA_BENCHMARK_REPO_ID,
        AUDIO_TABLE_NAME,
        revision,
    )

    manifest = maybe_download_manifest(
        REFLECTRA_BENCHMARK_REPO_ID,
        output_dir,
        revision,
    )
    ensure_embedded_media_columns(
        image_table_path,
        id_column="image_id",
        bytes_column="image",
    )
    ensure_embedded_media_columns(
        audio_table_path,
        id_column="audio_id",
        bytes_column="audio",
    )
    score_count = write_score_index(score_path, output_dir / SCORE_INDEX_NAME)

    image_count = unpack_media_table(
        source_path=image_table_path,
        output_index_path=output_dir / IMAGE_INDEX_NAME,
        media_dir=output_dir / "images",
        id_column="image_id",
        bytes_column="image",
        path_column="image_path",
        batch_size=batch_size,
        overwrite=overwrite,
    )
    audio_count = unpack_media_table(
        source_path=audio_table_path,
        output_index_path=output_dir / AUDIO_INDEX_NAME,
        media_dir=output_dir / "audio",
        id_column="audio_id",
        bytes_column="audio",
        path_column="audio_path",
        batch_size=batch_size,
        overwrite=overwrite,
    )

    manifest.update(
        {
            "source_repo_id": REFLECTRA_BENCHMARK_REPO_ID,
            "source_revision": revision,
            "unpacked": True,
            "image_count": image_count,
            "audio_count": audio_count,
            "score_row_count": score_count,
        }
    )
    write_json(output_dir / MANIFEST_NAME, manifest)

    for path in (
        output_dir / SCORE_TABLE_NAME,
        output_dir / IMAGE_TABLE_NAME,
        output_dir / AUDIO_TABLE_NAME,
    ):
        path.unlink(missing_ok=True)

    print("\nDownload summary:")
    print(f"Scores: {output_dir / SCORE_INDEX_NAME} ({score_count})")
    print(f"Images: {output_dir / 'images'} ({image_count})")
    print(f"Audio: {output_dir / 'audio'} ({audio_count})")
    print(f"Image index: {output_dir / IMAGE_INDEX_NAME}")
    print(f"Audio index: {output_dir / AUDIO_INDEX_NAME}")
    print(f"Manifest: {output_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    args = parse_args()
    download_reflectra_benchmark(
        output_dir=Path(args.output_dir).expanduser(),
        revision=args.revision,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )
