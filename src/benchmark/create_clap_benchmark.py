from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from src.config import get_nested, load_config
from src.datasets.selection import deterministic_sample, load_audio_metadata
from src.utils.hashing import stable_hash_id
from src.utils.json import read_jsonl, write_json, write_jsonl
from src.utils.openai_client import create_openai_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_METADATA_PATH = (
    PROJECT_ROOT / "data" / "metadata" / "song_describer_metadata.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clap_benchmark"
AUDIO_TABLE_COLUMNS = ["audio_id", "captions", "audio"]
SCORE_TABLE_COLUMNS = ["query_id", "caption", "audio_ids", "scores"]
PAIR_TABLE_COLUMNS = [
    "query_id",
    "caption",
    "audio_id",
    "score",
]

PROMPT = """
Return only minified JSON. No markdown. No explanation.
Task: score each audio item against this caption from 0 to 10, if you are 
not sure do not put high score, so better underestimate than overestimate.
Keys must be exactly the supplied audio IDs. Values must be integers.
Use string keys. Example key: "song_describer_123".
Use the audio descriptions as the evidence for mood, genre, instruments, 
energy, atmosphere, and semantic fit.
Caption: {caption}
Audio descriptions:
{audio_descriptions}
Example shape: {{"song_describer_123":8}}
"""


def response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    return str(response)


def parse_scores(text: str, expected_audio_ids: set[str]) -> dict[str, int]:
    text = text.strip()
    if not text:
        raise ValueError("Model response was empty.")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        value = json.loads(text[start:end + 1])

    if not isinstance(value, dict):
        raise ValueError("Model response must be a JSON object.")

    scores = {}
    for audio_id, raw_score in value.items():
        audio_id = str(audio_id)
        if audio_id not in expected_audio_ids:
            continue

        score = int(raw_score)
        if not 0 <= score <= 10:
            raise ValueError(f"Score for {audio_id} is outside 0..10: {score}")
        scores[audio_id] = score

    missing = expected_audio_ids - set(scores)
    if missing:
        raise ValueError(f"Missing scores for audio IDs: {sorted(missing)}")

    return scores


def format_captions(captions: Any) -> str:
    if captions is None:
        return ""
    if not isinstance(captions, list):
        captions = [captions]

    lines = []
    for index, caption in enumerate(captions):
        caption = str(caption).strip()
        if caption:
            lines.append(f"{index + 1}. {caption}")

    return "\n".join(lines)


def build_prompt(caption: str, audio_records: list[dict[str, Any]]) -> str:
    audio_descriptions = "\n\n".join(
        (
            f"{audio_record['audio_id']}:\n"
            f"{format_captions(audio_record.get('captions', []))}"
        )
        for audio_record in audio_records
    )
    return PROMPT.format(
        caption=caption,
        audio_descriptions=audio_descriptions,
    )


def score_audio_records(
    client: OpenAI,
    caption: str,
    audio_records: list[dict[str, Any]],
    model: str,
    max_attempts: int,
    audio_clip_seconds: float | None,
    max_output_tokens: int,
) -> dict[str, int] | None:
    _ = audio_clip_seconds
    expected_audio_ids = {record["audio_id"] for record in audio_records}
    prompt = build_prompt(caption=caption, audio_records=audio_records)
    last_error: Exception | None = None
    last_text = ""

    for attempt in range(max_attempts):
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )
            last_text = response_text(response)
            return parse_scores(last_text, expected_audio_ids)
        except Exception as exc:
            last_error = exc
            if attempt + 1 < max_attempts:
                time.sleep(2**attempt)

    preview = last_text.strip().replace("\n", " ")[:200]
    tqdm.write(
        "[WARN] Skipping CLAP query after "
        f"{max_attempts} failed scoring attempt(s): {last_error}. "
        f"Response preview: {preview!r}"
    )
    return None


def iter_queries(
    records: list[dict[str, Any]],
    queries_per_audio: int,
) -> list[tuple[dict[str, Any], str, int]]:
    queries = []

    for record in records:
        raw_captions = record.get("captions", [])
        if raw_captions is None:
            captions = []
        elif isinstance(raw_captions, list):
            captions = raw_captions
        else:
            captions = [raw_captions]

        caption_limit = min(len(captions), queries_per_audio)
        for caption_index, caption in enumerate(captions[:caption_limit]):
            caption = str(caption).strip()
            if caption:
                queries.append((record, caption, caption_index))

    return queries


def make_query_id(audio_id: str, caption: str, caption_index: int) -> str:
    return stable_hash_id(
        "clap_llm_query",
        audio_id,
        caption_index,
        caption,
        prefix="query_",
    )


def benchmark_row(
    query_record: dict[str, Any],
    caption: str,
    caption_index: int,
    audio_records: list[dict[str, Any]],
    scores: dict[str, int],
) -> dict[str, Any]:
    audio_ids = [record["audio_id"] for record in audio_records]
    return {
        "query_id": make_query_id(
            query_record["audio_id"],
            caption,
            caption_index,
        ),
        "caption": caption,
        "audio_ids": audio_ids,
        "scores": [int(scores[audio_id]) for audio_id in audio_ids],
    }


def dataset_fingerprint(
    records: list[dict[str, Any]],
    queries_per_audio: int,
    seed: int,
) -> str:
    signature = [
        {
            "audio_id": record["audio_id"],
            "captions": record["captions"],
            "audio_path": record["audio_path"],
            "source_dataset": record.get("source_dataset"),
            "split": record.get("split"),
        }
        for record in records
    ]
    return stable_hash_id(
        "clap_benchmark_dataset",
        signature,
        queries_per_audio,
        seed,
        prefix="dataset_",
        length=48,
    )


def shard_for_query(query_id: str, num_shards: int) -> int:
    digest = query_id.removeprefix("query_")
    return int(digest, 16) % num_shards


def accepted_query_rows(path: Path, force: bool) -> list[dict[str, Any]]:
    if force:
        return []

    rows = []
    for row in read_jsonl(path, missing_ok=True):
        scores = [int(score) for score in row.get("scores", [])]
        if scores and any(score > 0 for score in scores):
            rows.append(row)

    return rows


def build_benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_audio_records = load_audio_metadata(args.audio_metadata)
    build_benchmark_from_records(args, all_audio_records, output_dir)


def shard_prefix(shard_index: int, num_shards: int) -> str:
    width = max(3, len(str(num_shards - 1)))
    return (
        f"shard_{shard_index:0{width}d}"
        f"-of-{num_shards:0{width}d}"
    )


def audio_for_query(
    query_record: dict[str, Any],
    caption: str,
    caption_index: int,
    records: list[dict[str, Any]],
    max_audios: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    if max_audios is None or max_audios >= len(records):
        audio_records = records
    else:
        audio_records = sorted(
            records,
            key=lambda record: (
                stable_hash_id(
                    "clap_audio_per_query",
                    seed,
                    query_record["audio_id"],
                    caption_index,
                    caption,
                    record["audio_id"],
                    length=64,
                ),
                record["audio_id"],
            ),
        )[:max_audios]

    return sorted(
        audio_records,
        key=lambda record: (
            stable_hash_id(
                "clap_audio_pair_order",
                seed,
                query_record["audio_id"],
                caption_index,
                caption,
                record["audio_id"],
                length=64,
            ),
            record["audio_id"],
        ),
    )


def write_audio_table(records: list[dict[str, Any]], output_dir: Path) -> None:
    rows = [
        {
            "audio_id": record["audio_id"],
            "captions": record["captions"],
            "audio": Path(record["audio_path"]).read_bytes(),
        }
        for record in records
    ]
    pd.DataFrame(rows, columns=AUDIO_TABLE_COLUMNS).to_parquet(
        output_dir / "audio_table.parquet",
        index=False,
    )


def write_score_table(rows: list[dict[str, Any]], output_dir: Path) -> None:
    normalized_rows = [
        {
            "query_id": str(row["query_id"]),
            "caption": str(row["caption"]),
            "audio_ids": [str(audio_id) for audio_id in row["audio_ids"]],
            "scores": [int(score) for score in row["scores"]],
        }
        for row in rows
    ]
    pd.DataFrame(normalized_rows, columns=SCORE_TABLE_COLUMNS).to_parquet(
        output_dir / "clap_llm_benchmark.parquet",
        index=False,
    )


def pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs = []
    for row in rows:
        for audio_id, score in zip(row["audio_ids"], row["scores"]):
            pairs.append(
                {
                    "query_id": row["query_id"],
                    "caption": row["caption"],
                    "audio_id": audio_id,
                    "score": int(score),
                }
            )
    return pairs


def write_pair_csv(path: Path, pairs: list[dict[str, Any]]) -> None:
    pd.DataFrame(pairs, columns=PAIR_TABLE_COLUMNS).to_csv(
        path,
        index=False,
    )


def build_benchmark_from_records(
    args: argparse.Namespace,
    all_audio_records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    records = deterministic_sample(
        records=all_audio_records,
        count=args.audio_samples,
        seed=args.random_seed,
        id_field="audio_id",
    )
    if len(records) < args.audio_samples:
        print(
            "[WARN] Requested "
            f"{args.audio_samples} audio samples, but only {len(records)} "
            "usable records are available."
        )

    if len(records) < 2:
        raise ValueError("At least two audio records are required.")
    if args.max_audios is not None and len(records) < args.max_audios:
        raise ValueError(
            "Not enough audio records for --max_audios: "
            f"need {args.max_audios}, found {len(records)}."
        )

    fingerprint = dataset_fingerprint(
        records=records,
        queries_per_audio=args.queries_per_audio,
        seed=args.random_seed,
    )

    client = create_openai_client(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        config=args.config_data,
    )

    prefix = shard_prefix(
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    shard_path = output_dir / f"clap_benchmark_{prefix}.jsonl"
    shard_csv_path = output_dir / f"clap_benchmark_{prefix}.csv"
    manifest_path = output_dir / f"clap_manifest_{prefix}.json"

    rows_to_write = []
    all_rows = accepted_query_rows(shard_path, force=args.force)
    completed = {
        str(row["query_id"])
        for row in all_rows
        if row.get("query_id")
    }

    target_queries = iter_queries(records, args.queries_per_audio)
    if args.max_queries is not None:
        target_queries = target_queries[:args.max_queries]
    query_pool = iter_queries(
        deterministic_sample(
            records=all_audio_records,
            count=len(all_audio_records),
            seed=args.random_seed,
            id_field="audio_id",
        ),
        args.queries_per_audio,
    )

    total_pair_count = sum(
        len(
            audio_for_query(
                query_record=query_record,
                caption=caption,
                caption_index=caption_index,
                records=records,
                max_audios=args.max_audios,
                seed=args.random_seed,
            )
        )
        for query_record, caption, caption_index in target_queries
    )
    assigned_pair_count = sum(
        len(
            audio_for_query(
                query_record=query_record,
                caption=caption,
                caption_index=caption_index,
                records=records,
                max_audios=args.max_audios,
                seed=args.random_seed,
            )
        )
        for query_record, caption, caption_index in target_queries
        if shard_for_query(
            make_query_id(query_record["audio_id"], caption, caption_index),
            args.num_shards,
        )
        == args.shard_index
    )

    write_json(
        manifest_path,
        {
            "dataset_fingerprint": fingerprint,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "audio_count": len(records),
            "max_audios": args.max_audios,
            "queries_per_audio": args.queries_per_audio,
            "total_pair_count": total_pair_count,
            "assigned_pair_count": assigned_pair_count,
            "random_seed": args.random_seed,
            "audio_samples": args.audio_samples,
            "model": args.model,
        },
    )
    if not shard_path.exists() or args.force:
        write_jsonl(shard_path, [])
    if not shard_csv_path.exists() or args.force:
        write_pair_csv(shard_csv_path, [])

    accepted_pair_count = len(pair_rows(all_rows))

    with tqdm(
        total=assigned_pair_count,
        initial=min(accepted_pair_count, assigned_pair_count),
        desc="Score CLAP benchmark",
        unit="pair",
    ) as progress:
        for start in range(0, len(query_pool), args.batch_size):
            if accepted_pair_count >= assigned_pair_count:
                break

            query_batch = query_pool[start:start + args.batch_size]

            for query_record, caption, caption_index in query_batch:
                if accepted_pair_count >= assigned_pair_count:
                    break

                query_id = make_query_id(
                    query_record["audio_id"],
                    caption,
                    caption_index,
                )
                audio_records = audio_for_query(
                    query_record=query_record,
                    caption=caption,
                    caption_index=caption_index,
                    records=records,
                    max_audios=args.max_audios,
                    seed=args.random_seed,
                )

                if shard_for_query(query_id, args.num_shards) != args.shard_index:
                    continue

                if query_id in completed and not args.force:
                    continue

                scores = score_audio_records(
                    client=client,
                    caption=caption,
                    audio_records=audio_records,
                    model=args.model,
                    max_attempts=args.max_attempts,
                    audio_clip_seconds=args.audio_clip_seconds,
                    max_output_tokens=args.max_output_tokens,
                )
                if scores is None:
                    continue

                if all(score == 0 for score in scores.values()):
                    tqdm.write(
                        "[WARN] Skipping CLAP query because all scored "
                        f"audio records were zero: {query_id}"
                    )
                    continue

                row = benchmark_row(
                    query_record=query_record,
                    caption=caption,
                    caption_index=caption_index,
                    audio_records=audio_records,
                    scores=scores,
                )
                rows_to_write.append(row)
                all_rows.append(row)
                completed.add(query_id)

                progress.update(len(audio_records))
                accepted_pair_count += len(audio_records)

            if len(rows_to_write) >= args.flush_every:
                write_jsonl(shard_path, all_rows)
                write_pair_csv(shard_csv_path, pair_rows(all_rows))
                rows_to_write = []

    write_jsonl(shard_path, all_rows)
    pairs = pair_rows(all_rows)
    write_pair_csv(shard_csv_path, pairs)

    try:
        if args.num_shards == 1:
            write_score_table(all_rows, output_dir)
            write_audio_table(records, output_dir)
    except Exception as exc:
        print(f"[WARN] Audio table output was skipped: {exc}")

    if args.num_shards == 1:
        write_json(
            output_dir / "clap_benchmark_manifest.json",
            {
                "dataset_fingerprint": fingerprint,
                "num_shards": args.num_shards,
                "max_audios": args.max_audios,
                "queries_per_audio": args.queries_per_audio,
                "expected_pair_count": total_pair_count,
                "actual_pair_count": len(pairs),
                "complete": len(pairs) == total_pair_count,
            },
        )

    if len(pairs) < assigned_pair_count:
        print(
            "[WARN] CLAP shard is incomplete after exhausting replacement "
            f"queries: {len(pairs)}/{assigned_pair_count} pairs."
        )
    print(f"Wrote CLAP LLM benchmark shard to {shard_path}")


def parse_args() -> argparse.Namespace:
    config = load_config()

    parser = argparse.ArgumentParser(
        description="Create an LLM-scored caption-to-audio benchmark for CLAP.",
    )
    parser.add_argument(
        "--audio_metadata",
        type=str,
        nargs="*",
        default=[str(DEFAULT_AUDIO_METADATA_PATH)],
        help=(
            "Audio metadata JSONL path(s). Defaults to Song Describer. "
            "Audio files are read from the audio_path values in metadata."
        ),
    )
    parser.add_argument("--audio_samples", type=int, default=100)
    parser.add_argument("--queries_per_audio", type=int, default=1)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=int(get_nested(config, "benchmark", "batch_size", 4)),
        help="Number of caption queries to process before flush checks.",
    )
    parser.add_argument(
        "--max-audios",
        "--max_audios",
        dest="max_audios",
        type=int,
        default=10,
        help=(
            "Maximum total audio records to score per caption query."
        ),
    )
    parser.add_argument(
        "--model",
        default=get_nested(config, "benchmark", "model", ""),
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=int(get_nested(config, "benchmark", "random_seed", 42)),
    )
    parser.add_argument(
        "--output_dir",
        default=str(DEFAULT_OUTPUT_DIR),
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
        "--base_url",
        default=get_nested(config, "llm", "base_url", "") or None,
    )
    parser.add_argument(
        "--api_key",
        default=get_nested(config, "llm", "api_key", "") or None,
    )
    parser.add_argument(
        "--api_key_env",
        default=get_nested(config, "llm", "api_key_env", "OPENAI_API_KEY"),
    )
    parser.add_argument("--max_attempts", type=int, default=5)
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=int(get_nested(config, "llm", "max_output_tokens", 128)),
        help="Maximum tokens the LLM may generate for each JSON scoring reply.",
    )
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument(
        "--audio_clip_seconds",
        type=float,
        default=15.0,
        help=(
            "Retained for compatibility; local text LLM scoring uses audio "
            "descriptions instead of raw audio."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.audio_samples < 2:
        parser.error("--audio_samples must be at least 2.")
    if args.queries_per_audio < 1:
        parser.error("--queries_per_audio must be at least 1.")
    if args.max_audios is not None and args.max_audios < 2:
        parser.error("--max_audios must be at least 2.")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")
    if args.num_shards < 1:
        parser.error("--num_shards must be at least 1.")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error(
            "--shard_index must satisfy 0 <= shard_index < num_shards."
        )
    if args.max_attempts < 1:
        parser.error("--max_attempts must be at least 1.")
    if args.max_output_tokens < 16:
        parser.error("--max_output_tokens must be at least 16.")
    if args.flush_every < 1:
        parser.error("--flush_every must be at least 1.")
    if not args.model:
        parser.error("A model is required via --model or configs/reflectra.toml.")
    if args.audio_clip_seconds <= 0:
        args.audio_clip_seconds = None

    args.config_data = config
    return args


def main() -> None:
    build_benchmark(parse_args())


if __name__ == "__main__":
    main()
