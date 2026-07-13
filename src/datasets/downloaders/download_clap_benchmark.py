from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.datasets.paths import DATA_DIR, HF_CACHE_DIR, PROJECT_ROOT, ensure_data_dirs
from src.utils.json import write_json, write_jsonl
from src.utils.media_tables import as_list, bytes_value


CLAP_BENCHMARK_REPO_ID = "AraNge/reflectra-clap-benchmark"
DEFAULT_OUTPUT_DIR = DATA_DIR / "clap_benchmark"

BENCHMARK_TABLE_NAME = "clap_llm_benchmark.parquet"
AUDIO_TABLE_NAME = "audio_table.parquet"
BENCHMARK_METADATA_NAME = "clap_llm_benchmark.jsonl"
AUDIO_METADATA_NAME = "audio_table.jsonl"
MANIFEST_NAME = "clap_benchmark_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the CLAP LLM benchmark from Hugging Face and unpack "
            "embedded audio into data/clap_benchmark."
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
        help="Rewrite audio files that already exist.",
    )
    parser.add_argument(
        "--benchmark-table",
        default=None,
        help="Optional local clap_llm_benchmark.parquet path for offline unpacking.",
    )
    parser.add_argument(
        "--audio-table",
        default=None,
        help="Optional local audio_table.parquet path for offline unpacking.",
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


def resolve_table_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.benchmark_table or args.audio_table:
        if not args.benchmark_table or not args.audio_table:
            raise ValueError(
                "Pass both --benchmark-table and --audio-table for offline unpacking."
            )
        return (
            Path(args.benchmark_table).expanduser().resolve(),
            Path(args.audio_table).expanduser().resolve(),
        )

    return (
        download_benchmark_file(
            repo_id=CLAP_BENCHMARK_REPO_ID,
            filename=BENCHMARK_TABLE_NAME,
            revision=args.revision,
        ),
        download_benchmark_file(
            repo_id=CLAP_BENCHMARK_REPO_ID,
            filename=AUDIO_TABLE_NAME,
            revision=args.revision,
        ),
    )


def normalize_benchmark_row(row: dict[str, Any]) -> dict[str, Any]:
    audio_ids = [
        str(audio_id)
        for audio_id in as_list(row["audio_ids"])
    ]
    scores = [int(score) for score in as_list(row["scores"])]

    if not audio_ids:
        raise ValueError(f"No audio_ids for {row['query_id']}.")
    if len(audio_ids) != len(scores):
        raise ValueError(f"Mismatched audio_ids/scores for {row['query_id']}.")

    return {
        "query_id": str(row["query_id"]),
        "caption": str(row["caption"]),
        "audio_ids": audio_ids,
        "scores": scores,
    }


def write_benchmark_metadata(source_path: Path, output_path: Path) -> int:
    frame = pd.read_parquet(source_path)
    required_columns = {
        "query_id",
        "caption",
        "audio_ids",
        "scores",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{source_path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    records = [
        normalize_benchmark_row(row)
        for row in frame.to_dict(orient="records")
    ]
    write_jsonl(output_path, records)
    return len(records)


def audio_suffix(row: dict[str, Any]) -> str:
    source_path = row.get("audio_path")
    if source_path:
        suffix = Path(str(source_path)).suffix
        if suffix:
            return suffix
    return ".wav"


def unpack_audio_table(
    source_path: Path,
    output_path: Path,
    audio_dir: Path,
    batch_size: int,
    overwrite: bool,
) -> int:
    columns = set(pq.read_schema(source_path).names)
    required_columns = {"audio_id", "captions", "audio"}
    missing_columns = required_columns - columns
    if missing_columns:
        raise ValueError(
            f"{source_path.name} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    parquet_file = pq.ParquetFile(source_path)
    audio_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    progress = tqdm(
        total=parquet_file.metadata.num_rows,
        desc="Unpack audio",
        unit="row",
    )
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            audio_id = str(row["audio_id"])
            audio_path = audio_dir / f"{safe_filename(audio_id)}{audio_suffix(row)}"
            audio_bytes = row.get("audio")

            if audio_bytes is None:
                raise ValueError(f"Audio row has no embedded bytes: {audio_id}")
            if overwrite or not audio_path.exists():
                audio_path.write_bytes(bytes_value(audio_bytes))

            records.append(
                {
                    "audio_id": audio_id,
                    "captions": as_list(row.get("captions")),
                    "audio_path": relative_to_project(audio_path),
                }
            )
            progress.update(1)

    progress.close()
    write_jsonl(output_path, records)
    return len(records)


def download_clap_benchmark(
    output_dir: Path,
    revision: str | None,
    batch_size: int,
    overwrite: bool,
    benchmark_table: str | None = None,
    audio_table: str | None = None,
) -> None:
    ensure_data_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        revision=revision,
        benchmark_table=benchmark_table,
        audio_table=audio_table,
    )

    print(f"Downloading CLAP benchmark from: {CLAP_BENCHMARK_REPO_ID}")
    benchmark_path, audio_table_path = resolve_table_paths(args)

    benchmark_count = write_benchmark_metadata(
        source_path=benchmark_path,
        output_path=output_dir / BENCHMARK_METADATA_NAME,
    )
    audio_count = unpack_audio_table(
        source_path=audio_table_path,
        output_path=output_dir / AUDIO_METADATA_NAME,
        audio_dir=output_dir / "audio",
        batch_size=batch_size,
        overwrite=overwrite,
    )
    write_json(
        output_dir / MANIFEST_NAME,
        {
            "source_repo_id": CLAP_BENCHMARK_REPO_ID,
            "source_revision": revision,
            "unpacked": True,
            "query_count": benchmark_count,
            "audio_count": audio_count,
        },
    )

    for path in (
        output_dir / BENCHMARK_TABLE_NAME,
        output_dir / AUDIO_TABLE_NAME,
    ):
        path.unlink(missing_ok=True)

    print("\nDownload summary:")
    print(f"Benchmark metadata: {output_dir / BENCHMARK_METADATA_NAME} ({benchmark_count})")
    print(f"Audio: {output_dir / 'audio'} ({audio_count})")
    print(f"Audio metadata: {output_dir / AUDIO_METADATA_NAME}")
    print(f"Manifest: {output_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    parsed_args = parse_args()
    download_clap_benchmark(
        output_dir=Path(parsed_args.output_dir).expanduser(),
        revision=parsed_args.revision,
        batch_size=parsed_args.batch_size,
        overwrite=parsed_args.overwrite,
        benchmark_table=parsed_args.benchmark_table,
        audio_table=parsed_args.audio_table,
    )
