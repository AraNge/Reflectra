from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from src.config import get_nested, load_config
from src.datasets.selection import (
    DEFAULT_AUDIO_METADATA_PATHS,
    DEFAULT_IMAGE_METADATA_PATHS,
    deterministic_sample,
    load_audio_metadata,
    load_image_metadata,
)
from src.utils.hashing import make_pair_id, stable_hash_id
from src.utils.json import append_jsonl, read_json, read_jsonl, write_json, write_jsonl
from src.utils.openai_client import create_openai_client


PROMPT = """
Return only minified JSON. No markdown. No explanation.
Task: score how well each audio item matches the image from 0 to 10, if you are 
not sure do not put high score, so better underestimate than overestimate.
Use mood, atmosphere, emotion, energy, style, aesthetics, and overall feeling.
Keys must be exactly the audio IDs listed below. Values must be integers.
Image captions:
{image_captions}
Audio captions:
{music_descriptions}
Example shape: {{"audio_id":8}}
"""


SCORE_COLUMNS = [
    "pair_id",
    "image_id",
    "audio_id",
    "score",
]

AUDIO_TABLE_COLUMNS = ["audio_id", "captions", "audio", "audio_path"]
IMAGE_TABLE_COLUMNS = ["image_id", "captions", "image", "image_path"]
MATCH_TABLE_COLUMNS = ["image_id", "audio_ids", "scores"]
HF_TABLE_COLUMNS = [
    "pair_id",
    "image_id",
    "audio_id",
    "score",
    "image_captions",
    "audio_captions",
    "image",
    "audio",
]


def dataset_fingerprint(
    images: list[dict[str, Any]],
    audio: list[dict[str, Any]],
    seed: int,
) -> str:
    image_signature = [
        {
            "image_id": record["image_id"],
            "captions": record["captions"],
            "image_path": record["image_path"],
            "source_dataset": record["source_dataset"],
            "split": record["split"],
        }
        for record in images
    ]
    audio_signature = [
        {
            "audio_id": record["audio_id"],
            "captions": record["captions"],
            "audio_path": record["audio_path"],
            "source_dataset": record["source_dataset"],
            "split": record["split"],
        }
        for record in audio
    ]

    return stable_hash_id(
        "benchmark_dataset",
        image_signature,
        audio_signature,
        seed,
        prefix="dataset_",
        length=48,
    )


def shard_for_pair(pair_id: str, num_shards: int) -> int:
    digest = pair_id.removeprefix("pair_")
    return int(digest, 16) % num_shards


def audio_for_image(
    image: dict[str, Any],
    audio: list[dict[str, Any]],
    max_samples: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_samples is None or max_samples >= len(audio):
        return audio

    ranked = sorted(
        audio,
        key=lambda record: (
            stable_hash_id(
                "benchmark_audio_per_image",
                seed,
                image["image_id"],
                record["audio_id"],
                length=64,
            ),
            record["audio_id"],
        ),
    )
    return ranked[:max_samples]


def iter_shard_pairs(
    images: list[dict[str, Any]],
    audio: list[dict[str, Any]],
    num_shards: int,
    shard_index: int,
    max_samples: int | None,
    seed: int,
) -> Iterator[tuple[dict[str, Any], dict[str, Any], str]]:
    for image in images:
        for audio_record in audio_for_image(
            image=image,
            audio=audio,
            max_samples=max_samples,
            seed=seed,
        ):
            pair_id = make_pair_id(
                image_id=image["image_id"],
                audio_id=audio_record["audio_id"],
            )

            if shard_for_pair(pair_id, num_shards) == shard_index:
                yield image, audio_record, pair_id


def count_shard_pairs(
    images: list[dict[str, Any]],
    audio: list[dict[str, Any]],
    num_shards: int,
    shard_index: int,
    max_samples: int | None,
    seed: int,
) -> int:
    return sum(
        1
        for _ in iter_shard_pairs(
            images=images,
            audio=audio,
            num_shards=num_shards,
            shard_index=shard_index,
            max_samples=max_samples,
            seed=seed,
        )
    )


def format_captions(captions: list[str]) -> str:
    return "\n".join(
        f"{index + 1}. {caption}"
        for index, caption in enumerate(captions)
    )


def build_prompt(
    image_captions: list[str],
    audio_batch: list[dict[str, Any]],
) -> str:
    music_descriptions = "\n\n".join(
        (
            f"{audio_record['audio_id']}:\n"
            f"{format_captions(audio_record['captions'])}"
        )
        for audio_record in audio_batch
    )

    return PROMPT.format(
        image_captions=format_captions(image_captions),
        music_descriptions=music_descriptions,
    )


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("The model response must be a JSON object.")

    return value


def score_audio_batch(
    client: OpenAI,
    image_captions: list[str],
    audio_batch: list[dict[str, Any]],
    model: str,
    max_attempts: int,
    max_output_tokens: int,
) -> dict[str, int]:
    pending = {
        audio_record["audio_id"]: audio_record
        for audio_record in audio_batch
    }
    scores: dict[str, int] = {}

    for attempt in range(max_attempts):
        if not pending:
            break

        prompt = build_prompt(
            image_captions=image_captions,
            audio_batch=list(pending.values()),
        )

        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            result = parse_json_object(response.output_text)

            for audio_id, raw_score in result.items():
                audio_id = str(audio_id)

                if audio_id not in pending:
                    print(
                        "[WARN] Ignoring an unknown audio ID returned by "
                        f"the model: {audio_id}"
                    )
                    continue

                try:
                    score = int(raw_score)
                except (TypeError, ValueError):
                    print(
                        f"[WARN] Invalid score for {audio_id}: {raw_score}"
                    )
                    continue

                if not 0 <= score <= 10:
                    print(
                        f"[WARN] Score outside 0..10 for {audio_id}: {score}"
                    )
                    continue

                scores[audio_id] = score
                pending.pop(audio_id)

        except Exception as exc:
            print(f"Retry {attempt + 1}/{max_attempts}: {exc}")

        if pending and attempt + 1 < max_attempts:
            time.sleep(2**attempt)

    if pending:
        print(
            "[WARN] The current batch still has "
            f"{len(pending)} unscored audio item(s). Re-running the shard "
            "will retry only the missing pairs."
        )

    return scores


def make_benchmark_row(
    image: dict[str, Any],
    audio_record: dict[str, Any],
    pair_id: str,
    score: int,
) -> dict[str, Any]:
    # Shard checkpoints contain only foreign keys and the score. Media and
    # captions are stored once in the final normalized tables.
    return {
        "pair_id": pair_id,
        "image_id": image["image_id"],
        "audio_id": audio_record["audio_id"],
        "score": int(score),
    }


def csv_safe_dataframe(
    records: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> pd.DataFrame:
    dataframe = pd.DataFrame(records, columns=columns)

    for column in dataframe.columns:
        dataframe[column] = dataframe[column].map(
            lambda value: json.dumps(value, ensure_ascii=False)
            if isinstance(value, (dict, list))
            else value
        )

    return dataframe


def write_huggingface_pair_table(
    benchmark: list[dict[str, Any]],
    images: list[dict[str, Any]],
    audio: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    try:
        from datasets import Audio, Dataset, Features, Image, Sequence, Value
    except Exception as exc:
        raise RuntimeError(
            "Could not create the Hugging Face benchmark table. Install the "
            "datasets package."
        ) from exc

    images_by_id = {record["image_id"]: record for record in images}
    audio_by_id = {record["audio_id"]: record for record in audio}
    records = []

    for row in benchmark:
        image = images_by_id[row["image_id"]]
        audio_record = audio_by_id[row["audio_id"]]
        image_path = Path(image["image_path"])
        audio_path = Path(audio_record["audio_path"])

        records.append(
            {
                "pair_id": row["pair_id"],
                "image_id": row["image_id"],
                "audio_id": row["audio_id"],
                "score": int(row["score"]),
                "image_captions": image["captions"],
                "audio_captions": audio_record["captions"],
                "image": {
                    "bytes": image_path.read_bytes(),
                    "path": image_path.name,
                },
                "audio": {
                    "bytes": audio_path.read_bytes(),
                    "path": audio_path.name,
                },
            }
        )

    features = Features(
        {
            "pair_id": Value("string"),
            "image_id": Value("string"),
            "audio_id": Value("string"),
            "score": Value("int64"),
            "image_captions": Sequence(Value("string")),
            "audio_captions": Sequence(Value("string")),
            "image": Image(),
            "audio": Audio(),
        }
    )

    dataset = Dataset.from_list(records, features=features)
    dataset.to_parquet(output_dir / "benchmark_hf.parquet")


def shard_prefix(shard_index: int, num_shards: int) -> str:
    width = max(3, len(str(num_shards - 1)))
    return (
        f"shard_{shard_index:0{width}d}"
        f"-of-{num_shards:0{width}d}"
    )


def write_or_validate_manifest(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    if path.exists():
        existing = read_json(path)

        keys = [
            "dataset_fingerprint",
            "num_shards",
            "shard_index",
            "image_count",
            "audio_count",
            "max_samples",
            "random_seed",
            "model",
        ]
        conflicts = {
            key: (existing.get(key), manifest.get(key))
            for key in keys
            if existing.get(key) != manifest.get(key)
        }

        if conflicts:
            raise ValueError(
                f"Existing manifest {path} does not match this run: "
                f"{conflicts}"
            )

        return

    write_json(path, manifest)


@contextmanager
def shard_lock(path: Path):
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY,
        )
    except FileExistsError as exc:
        raise RuntimeError(
            f"Shard lock already exists: {path}. Another process may be "
            "building the same shard. Remove the lock only when no such "
            "process is running."
        ) from exc

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(f"{os.getpid()}\n")

        yield
    finally:
        path.unlink(missing_ok=True)


def build_shard(args: argparse.Namespace) -> None:
    images = deterministic_sample(
        records=load_image_metadata(args.image_metadata),
        count=args.image_samples,
        seed=args.random_seed,
        id_field="image_id",
    )
    audio = deterministic_sample(
        records=load_audio_metadata(args.audio_metadata),
        count=args.audio_samples,
        seed=args.random_seed,
        id_field="audio_id",
    )

    fingerprint = dataset_fingerprint(
        images=images,
        audio=audio,
        seed=args.random_seed,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = shard_prefix(
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    shard_path = output_dir / f"benchmark_{prefix}.jsonl"
    shard_csv_path = output_dir / f"benchmark_{prefix}.csv"
    manifest_path = output_dir / f"manifest_{prefix}.json"
    lock_path = output_dir / f"benchmark_{prefix}.lock"

    assigned_pair_count = count_shard_pairs(
        images=images,
        audio=audio,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        max_samples=args.max_samples,
        seed=args.random_seed,
    )
    total_pair_count = sum(
        len(
            audio_for_image(
                image=image,
                audio=audio,
                max_samples=args.max_samples,
                seed=args.random_seed,
            )
        )
        for image in images
    )

    write_or_validate_manifest(
        path=manifest_path,
        manifest={
            "dataset_fingerprint": fingerprint,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "image_count": len(images),
            "audio_count": len(audio),
            "max_samples": args.max_samples,
            "total_pair_count": total_pair_count,
            "assigned_pair_count": assigned_pair_count,
            "random_seed": args.random_seed,
            "image_samples": args.image_samples,
            "audio_samples": args.audio_samples,
            "model": args.model,
        },
    )

    with shard_lock(lock_path):
        existing_rows = read_jsonl(shard_path, missing_ok=True)
        existing_by_pair_id: dict[str, dict[str, Any]] = {}

        for row in existing_rows:
            pair_id = str(row.get("pair_id", ""))

            if not pair_id:
                raise ValueError(
                    f"An existing row in {shard_path} has no pair_id."
                )

            if shard_for_pair(pair_id, args.num_shards) != args.shard_index:
                raise ValueError(
                    f"Pair {pair_id} does not belong to shard "
                    f"{args.shard_index}."
                )

            previous = existing_by_pair_id.get(pair_id)

            if previous is not None and previous != row:
                raise ValueError(
                    f"Conflicting duplicate pair {pair_id} in {shard_path}."
                )

            existing_by_pair_id[pair_id] = row

        completed_pair_ids = set(existing_by_pair_id)
        print(
            f"Shard {args.shard_index}/{args.num_shards}: "
            f"{len(completed_pair_ids)}/{assigned_pair_count} pairs complete, "
            f"{assigned_pair_count - len(completed_pair_ids)} remaining."
        )
        client = create_openai_client(
            api_key=args.api_key,
            api_key_env=args.api_key_env,
            base_url=args.base_url,
            config=args.config_data,
        )

        with tqdm(
            total=assigned_pair_count,
            initial=len(completed_pair_ids),
            desc=f"Shard {args.shard_index}/{args.num_shards}",
            unit="pair",
        ) as progress:
            progress.set_postfix(
                completed=len(completed_pair_ids),
                remaining=assigned_pair_count - len(completed_pair_ids),
            )
            for image in images:
                pending_for_image: list[
                    tuple[dict[str, Any], str]
                ] = []

                for audio_record in audio_for_image(
                    image=image,
                    audio=audio,
                    max_samples=args.max_samples,
                    seed=args.random_seed,
                ):
                    pair_id = make_pair_id(
                        image_id=image["image_id"],
                        audio_id=audio_record["audio_id"],
                    )

                    if (
                        shard_for_pair(pair_id, args.num_shards)
                        != args.shard_index
                        or pair_id in completed_pair_ids
                    ):
                        continue

                    pending_for_image.append((audio_record, pair_id))

                for start in range(
                    0,
                    len(pending_for_image),
                    args.batch_size,
                ):
                    batch = pending_for_image[
                        start:start + args.batch_size
                    ]
                    audio_batch = [
                        audio_record
                        for audio_record, _ in batch
                    ]
                    pair_ids = {
                        audio_record["audio_id"]: pair_id
                        for audio_record, pair_id in batch
                    }

                    scores = score_audio_batch(
                        client=client,
                        image_captions=image["captions"],
                        audio_batch=audio_batch,
                        model=args.model,
                        max_attempts=args.max_attempts,
                        max_output_tokens=args.max_output_tokens,
                    )

                    rows_to_append: list[dict[str, Any]] = []

                    for audio_record in audio_batch:
                        audio_id = audio_record["audio_id"]

                        if audio_id not in scores:
                            continue

                        pair_id = pair_ids[audio_id]

                        if pair_id in completed_pair_ids:
                            continue

                        rows_to_append.append(
                            make_benchmark_row(
                                image=image,
                                audio_record=audio_record,
                                pair_id=pair_id,
                                score=scores[audio_id],
                            )
                        )
                        completed_pair_ids.add(pair_id)

                    append_jsonl(
                        path=shard_path,
                        records=rows_to_append,
                    )
                    progress.update(len(rows_to_append))
                    progress.set_postfix(
                        completed=len(completed_pair_ids),
                        remaining=assigned_pair_count - len(completed_pair_ids),
                    )

        final_rows_by_pair_id = {
            row["pair_id"]: row
            for row in read_jsonl(shard_path, missing_ok=True)
        }
        final_rows = sorted(
            final_rows_by_pair_id.values(),
            key=lambda row: row["pair_id"],
        )

        csv_safe_dataframe(
            final_rows,
            columns=SCORE_COLUMNS,
        ).to_csv(
            shard_csv_path,
            index=False,
        )

        completed_count = len(final_rows)
        print(
            f"Shard {args.shard_index}: "
            f"{completed_count}/{assigned_pair_count} pairs complete."
        )

        if completed_count < assigned_pair_count:
            print(
                "[WARN] Run the same command again to retry only the "
                "missing pairs."
            )


def load_manifests(
    output_dir: Path,
    num_shards: int,
) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []

    for shard_index in range(num_shards):
        prefix = shard_prefix(
            shard_index=shard_index,
            num_shards=num_shards,
        )
        path = output_dir / f"manifest_{prefix}.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing manifest for shard {shard_index}: {path}"
            )

        manifests.append(read_json(path))

    reference = manifests[0]
    keys = [
        "dataset_fingerprint",
        "num_shards",
        "image_count",
        "audio_count",
        "max_samples",
        "total_pair_count",
        "random_seed",
        "model",
    ]

    for manifest in manifests[1:]:
        conflicts = {
            key: (reference.get(key), manifest.get(key))
            for key in keys
            if reference.get(key) != manifest.get(key)
        }

        if conflicts:
            raise ValueError(
                "Shard manifests were created from different benchmark "
                f"configurations: {conflicts}"
            )

    return manifests


def merge_shards(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    manifests = load_manifests(
        output_dir=output_dir,
        num_shards=args.num_shards,
    )

    rows_by_pair_id: dict[str, dict[str, Any]] = {}

    for shard_index in range(args.num_shards):
        prefix = shard_prefix(
            shard_index=shard_index,
            num_shards=args.num_shards,
        )
        path = output_dir / f"benchmark_{prefix}.jsonl"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing benchmark file for shard {shard_index}: {path}"
            )

        for row in read_jsonl(path):
            pair_id = row["pair_id"]
            existing = rows_by_pair_id.get(pair_id)

            if existing is not None and existing != row:
                raise ValueError(
                    f"Conflicting duplicate benchmark pair: {pair_id}"
                )

            rows_by_pair_id[pair_id] = row

    benchmark = sorted(
        rows_by_pair_id.values(),
        key=lambda row: row["pair_id"],
    )
    expected_count = int(manifests[0]["total_pair_count"])
    actual_count = len(benchmark)

    if (
        actual_count != expected_count
        and not args.allow_incomplete_merge
    ):
        raise ValueError(
            f"Expected {expected_count} pairs but found {actual_count}. "
            "Finish the missing shards or use --allow_incomplete_merge."
        )

    images = deterministic_sample(
        records=load_image_metadata(args.image_metadata),
        count=args.image_samples,
        seed=args.random_seed,
        id_field="image_id",
    )
    audio = deterministic_sample(
        records=load_audio_metadata(args.audio_metadata),
        count=args.audio_samples,
        seed=args.random_seed,
        id_field="audio_id",
    )

    fingerprint = dataset_fingerprint(images, audio, args.random_seed)
    if fingerprint != manifests[0]["dataset_fingerprint"]:
        raise ValueError(
            "The metadata/sample configuration used for merge does not match "
            "the shard manifests."
        )

    audio_table = [
        {
            "audio_id": record["audio_id"],
            "captions": record["captions"],
            "audio": Path(record["audio_path"]).read_bytes(),
            "audio_path": record["audio_path"],
        }
        for record in audio
    ]
    image_table = [
        {
            "image_id": record["image_id"],
            "captions": record["captions"],
            "image": Path(record["image_path"]).read_bytes(),
            "image_path": record["image_path"],
        }
        for record in images
    ]

    scores_by_image: dict[str, list[tuple[str, int]]] = {}
    for row in benchmark:
        scores_by_image.setdefault(row["image_id"], []).append(
            (row["audio_id"], int(row["score"]))
        )

    match_table = []
    for image in images:
        pairs = sorted(scores_by_image.get(image["image_id"], []))
        match_table.append(
            {
                "image_id": image["image_id"],
                "audio_ids": [audio_id for audio_id, _ in pairs],
                "scores": [score for _, score in pairs],
            }
        )

    if not args.write_parquet:
        raise ValueError(
            "The compact byte tables require Parquet. Use --write_parquet."
        )

    try:
        pd.DataFrame(audio_table, columns=AUDIO_TABLE_COLUMNS).to_parquet(
            output_dir / "audio_table.parquet", index=False
        )
        pd.DataFrame(image_table, columns=IMAGE_TABLE_COLUMNS).to_parquet(
            output_dir / "image_table.parquet", index=False
        )
        pd.DataFrame(match_table, columns=MATCH_TABLE_COLUMNS).to_parquet(
            output_dir / "image_audio_scores.parquet", index=False
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not write compact Parquet tables. Install pyarrow or "
            "fastparquet."
        ) from exc

    if args.write_hf:
        write_huggingface_pair_table(
            benchmark=benchmark,
            images=images,
            audio=audio,
            output_dir=output_dir,
        )

    write_json(
        output_dir / "benchmark_manifest.json",
        {
            "dataset_fingerprint": manifests[0][
                "dataset_fingerprint"
            ],
            "num_shards": args.num_shards,
            "max_samples": manifests[0].get("max_samples"),
            "expected_pair_count": expected_count,
            "actual_pair_count": actual_count,
            "complete": actual_count == expected_count,
            "huggingface_pair_table": args.write_hf,
        },
    )

    print(
        f"Merged {actual_count}/{expected_count} pairs into benchmark tables "
        f"in {output_dir}."
    )


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=None,
        help="Path to TOML config. Default: configs/reflectra.toml.",
    )
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)

    parser = argparse.ArgumentParser(
        description=(
            "Build or merge deterministic, non-overlapping image/audio "
            "benchmark shards."
        ),
        parents=[config_parser],
    )
    benchmark_config = config.get("benchmark", {})

    parser.add_argument(
        "--mode",
        choices=["build", "merge"],
        default="build",
    )
    parser.add_argument(
        "--image_metadata",
        type=str,
        nargs="*",
        default=[
            str(path)
            for path in DEFAULT_IMAGE_METADATA_PATHS
        ],
    )
    parser.add_argument(
        "--audio_metadata",
        type=str,
        nargs="*",
        default=[
            str(path)
            for path in DEFAULT_AUDIO_METADATA_PATHS
        ],
    )
    parser.add_argument(
        "--image_samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--audio_samples",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--max-samples",
        "--max_samples",
        dest="max_samples",
        type=int,
        default=None,
        help=(
            "Maximum number of audio candidates to score per image. "
            "If omitted, every sampled audio item is scored for every image."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(get_nested(config, "benchmark", "batch_size", 4)),
    )
    parser.add_argument(
        "--model",
        default=get_nested(
            config,
            "benchmark",
            "model",
            "",
        ),
    )
    parser.add_argument(
        "--base_url",
        default=get_nested(config, "llm", "base_url", "") or None,
        help="OpenAI-compatible API base URL. Defaults to [llm].base_url.",
    )
    parser.add_argument(
        "--api_key",
        default=get_nested(config, "llm", "api_key", "") or None,
        help=(
            "API key for the OpenAI-compatible client. Prefer "
            "--api_key_env or [llm].api_key_env for real secrets."
        ),
    )
    parser.add_argument(
        "--api_key_env",
        default=get_nested(config, "llm", "api_key_env", "OPENAI_API_KEY"),
        help="Environment variable that contains the API key.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=int(get_nested(config, "llm", "max_output_tokens", 128)),
        help="Maximum tokens the LLM may generate for each JSON scoring reply.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=int(
            get_nested(
                config,
                "benchmark",
                "random_seed",
                42,
            )
        ),
        help=(
            "All collaborators must use the same seed, metadata, sample "
            "counts, model and num_shards."
        ),
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--output_dir",
        default=str(benchmark_config.get("output_dir", "data/benchmark")),
    )
    parser.add_argument(
        "--write_parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write_hf",
        action=argparse.BooleanOptionalAction,
        default=bool(benchmark_config.get("write_hf", True)),
        help=(
            "Write benchmark_hf.parquet with Hugging Face Image/Audio "
            "features for preview/playback."
        ),
    )
    parser.add_argument(
        "--write_wide",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--allow_incomplete_merge",
        action="store_true",
    )

    args = parser.parse_args()

    if args.image_samples < 1:
        parser.error("--image_samples must be at least 1.")

    if args.audio_samples < 1:
        parser.error("--audio_samples must be at least 1.")

    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be at least 1 when provided.")

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")

    if args.max_attempts < 1:
        parser.error("--max_attempts must be at least 1.")

    if args.max_output_tokens < 16:
        parser.error("--max_output_tokens must be at least 16.")

    if not args.model:
        parser.error("A model is required via --model or configs/reflectra.toml.")

    if args.num_shards < 1:
        parser.error("--num_shards must be at least 1.")

    if not 0 <= args.shard_index < args.num_shards:
        parser.error(
            "--shard_index must satisfy "
            "0 <= shard_index < num_shards."
        )

    if args.mode == "build":
        if not args.image_metadata:
            parser.error(
                "At least one --image_metadata path is required."
            )

        if not args.audio_metadata:
            parser.error(
                "At least one --audio_metadata path is required."
            )

    args.config_data = config
    return args


def main(args: argparse.Namespace) -> None:
    if args.mode == "build":
        build_shard(args)
    else:
        merge_shards(args)


if __name__ == "__main__":
    main(parse_args())
