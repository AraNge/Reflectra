from __future__ import annotations

import argparse
import base64
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

from src.config import get_nested, load_config
from src.datasets.selection import deterministic_sample
from src.metrics.retrieval_metrics import sparse_retrieval_metrics
from src.utils.audio import audio_payload_for_llm
from src.utils.hashing import stable_hash_id
from src.utils.json import read_jsonl, write_json, write_jsonl
from src.utils.media_tables import load_audio_table
from src.utils.openai_client import create_openai_client


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIO_TABLE_PATH = PROJECT_ROOT / "data" / "benchmark" / "audio_table.parquet"

PROMPT = """
Return only minified JSON. No markdown. No explanation.
Task: score each audio clip against this caption from 0 to 10, if you are 
not sure do not put high score, so better underestimate than overestimate.
Keys must be exactly the supplied audio IDs. Values must be integers.
Caption: {caption}
Example shape: {{"audio_id":8}}
"""


def response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    return str(response)


def parse_scores(text: str, expected_audio_ids: set[str]) -> dict[str, int]:
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


def score_audio_candidates(
    client: OpenAI,
    caption: str,
    candidates: list[dict[str, Any]],
    model: str,
    max_attempts: int,
    audio_clip_seconds: float | None,
    max_output_tokens: int,
) -> dict[str, int]:
    content: list[dict[str, Any]] = [
        {
            "type": "input_text",
            "text": PROMPT.format(caption=caption),
        }
    ]

    for candidate in candidates:
        audio_bytes, audio_format = audio_payload_for_llm(
            candidate["audio_path"],
            clip_seconds=audio_clip_seconds,
        )
        content.append(
            {
                "type": "input_text",
                "text": f"Audio ID: {candidate['audio_id']}",
            }
        )
        content.append(
            {
                "type": "input_audio",
                "input_audio": {
                    "data": base64.b64encode(audio_bytes).decode("ascii"),
                    "format": audio_format,
                },
            }
        )

    expected_audio_ids = {candidate["audio_id"] for candidate in candidates}

    for attempt in range(max_attempts):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                max_output_tokens=max_output_tokens,
            )
            return parse_scores(response_text(response), expected_audio_ids)
        except Exception:
            if attempt + 1 >= max_attempts:
                raise
            time.sleep(2**attempt)

    raise RuntimeError("Scoring failed unexpectedly.")


def iter_queries(
    records: list[dict[str, Any]],
    queries_per_audio: int,
) -> list[tuple[dict[str, Any], str, int]]:
    queries = []

    for record in records:
        captions = record.get("captions", [])[:queries_per_audio]
        for caption_index, caption in enumerate(captions):
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
    candidates: list[dict[str, Any]],
    scores: dict[str, int],
) -> dict[str, Any]:
    audio_ids = [candidate["audio_id"] for candidate in candidates]
    return {
        "query_id": make_query_id(
            query_record["audio_id"],
            caption,
            caption_index,
        ),
        "caption": caption,
        "positive_audio_id": query_record["audio_id"],
        "candidate_audio_ids": audio_ids,
        "scores": [int(scores[audio_id]) for audio_id in audio_ids],
        "relevance": {
            audio_id: int(scores[audio_id])
            for audio_id in audio_ids
        },
    }


def existing_query_ids(path: Path) -> set[str]:
    return {
        str(row["query_id"])
        for row in read_jsonl(path, missing_ok=True)
        if row.get("query_id")
    }


def compute_oracle_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}

    max_candidates = max(len(row["candidate_audio_ids"]) for row in rows)
    similarity = []
    relevance = []

    for row in rows:
        scores = list(row["scores"])
        padded_scores = scores + [-1_000_000.0] * (max_candidates - len(scores))
        similarity.append(padded_scores)
        relevance.append(
            {
                index: float(score)
                for index, score in enumerate(scores)
                if float(score) > 0
            }
        )

    return sparse_retrieval_metrics(
        similarity=pd.DataFrame(similarity).to_numpy(dtype=float),
        relevance=relevance,
        exponential_gain=True,
    )


def build_benchmark(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    temp_parent = output_dir / "tmp"
    temp_parent.mkdir(parents=True, exist_ok=True)
    audio_table_path = Path(args.audio_table).expanduser().resolve()

    with tempfile.TemporaryDirectory(
        prefix="clap_benchmark_",
        dir=temp_parent,
    ) as temp_dir:
        all_audio_records = list(
            load_audio_table(
                path=audio_table_path,
                materialize_dir=Path(temp_dir),
                project_root=PROJECT_ROOT,
            ).values()
        )

        build_benchmark_from_records(args, all_audio_records, output_dir)


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

    if len(records) < 2:
        raise ValueError("At least two audio records are required.")
    if len(records) < args.num_negatives + 1:
        raise ValueError(
            "Not enough audio records for the requested negatives: "
            f"need at least {args.num_negatives + 1}, found {len(records)}."
        )

    client = create_openai_client(
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        config=args.config_data,
    )
    rng = random.Random(args.random_seed)
    output_path = output_dir / args.output_name
    completed = set() if args.force else existing_query_ids(output_path)
    rows_to_write = []
    all_rows = [] if args.force else read_jsonl(output_path, missing_ok=True)

    queries = iter_queries(records, args.queries_per_audio)
    if args.max_queries is not None:
        queries = queries[:args.max_queries]

    for query_record, caption, caption_index in tqdm(
        queries,
        desc="Score CLAP benchmark",
        unit="query",
    ):
        query_id = make_query_id(
            query_record["audio_id"],
            caption,
            caption_index,
        )
        if query_id in completed and not args.force:
            continue

        negatives = [
            record
            for record in records
            if record["audio_id"] != query_record["audio_id"]
        ]
        rng.shuffle(negatives)
        candidates = [query_record] + negatives[:args.num_negatives]
        rng.shuffle(candidates)

        scores = score_audio_candidates(
            client=client,
            caption=caption,
            candidates=candidates,
            model=args.model,
            max_attempts=args.max_attempts,
            audio_clip_seconds=args.audio_clip_seconds,
            max_output_tokens=args.max_output_tokens,
        )
        row = benchmark_row(
            query_record=query_record,
            caption=caption,
            caption_index=caption_index,
            candidates=candidates,
            scores=scores,
        )
        rows_to_write.append(row)
        all_rows.append(row)

        if len(rows_to_write) >= args.flush_every:
            write_jsonl(output_path, all_rows)
            rows_to_write = []

    write_jsonl(output_path, all_rows)
    pd.DataFrame(all_rows).to_csv(output_path.with_suffix(".csv"), index=False)

    try:
        pd.DataFrame(all_rows).to_parquet(
            output_path.with_suffix(".parquet"),
            index=False,
        )
    except Exception as exc:
        print(f"[WARN] Parquet output was skipped: {exc}")

    manifest = {
        "num_queries": len(all_rows),
        "audio_samples": len(records),
        "num_negatives": args.num_negatives,
        "queries_per_audio": args.queries_per_audio,
        "model": args.model,
        "random_seed": args.random_seed,
        "audio_clip_seconds": args.audio_clip_seconds,
        "audio_table": str(Path(args.audio_table).expanduser().resolve()),
        "oracle_metrics_from_llm_scores": compute_oracle_metrics(all_rows),
    }
    write_json(output_path.with_name("clap_llm_benchmark_manifest.json"), manifest)
    print(f"Wrote CLAP LLM benchmark to {output_path}")


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=None)
    config_args, _ = config_parser.parse_known_args()
    config = load_config(config_args.config)
    benchmark_config = config.get("benchmark", {})

    parser = argparse.ArgumentParser(
        description="Create an LLM-scored caption-to-audio benchmark for CLAP.",
        parents=[config_parser],
    )
    parser.add_argument("--audio_table", default=str(DEFAULT_AUDIO_TABLE_PATH))
    parser.add_argument("--audio_samples", type=int, default=100)
    parser.add_argument("--queries_per_audio", type=int, default=1)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--num_negatives", type=int, default=9)
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
        default=str(benchmark_config.get("output_dir", "data/benchmark")),
    )
    parser.add_argument("--output_name", default="clap_llm_benchmark.jsonl")
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
    parser.add_argument("--flush_every", type=int, default=10)
    parser.add_argument(
        "--audio_clip_seconds",
        type=float,
        default=15.0,
        help="Use only a middle clip when audio is longer than this. Use 0 for full audio.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.audio_samples < 2:
        parser.error("--audio_samples must be at least 2.")
    if args.queries_per_audio < 1:
        parser.error("--queries_per_audio must be at least 1.")
    if args.num_negatives < 1:
        parser.error("--num_negatives must be at least 1.")
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
