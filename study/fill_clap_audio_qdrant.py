import argparse
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tqdm import tqdm

from src.config import get_nested, load_config
from study.audio_parts import (
    DEFAULT_STUDY_DATASETS,
    AudioRecord,
    DownloadedAudio,
    download_part,
    load_state,
    save_state,
)
from src.datasets.paths import DATA_DIR, ensure_data_dirs
from src.models.clap_encoder import PretrainedCLAPEncoder
from src.vector_db.index_clap_audio_qdrant import (
    DEFAULT_AUDIO_EXTENSIONS,
    resolve_project_path,
)
from src.vector_db.qdrant_store import (
    QdrantClient,
    create_collection_if_not_exists,
    get_qdrant_client,
    upsert_vectors,
)


@dataclass
class DownloadedPart:
    records: list[DownloadedAudio]
    next_dataset_state: dict[str, Any]
    exhausted: bool
    part_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fill Qdrant with CLAP audio embeddings by downloading music audio "
            "datasets in disposable parts. MusicCaps is intentionally excluded."
        )
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--collection-name", type=str, default=None)
    parser.add_argument("--qdrant-url", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--vector-size", type=int, default=None)
    parser.add_argument("--target-samples", type=int, default=500_000)
    parser.add_argument(
        "--part-size",
        type=int,
        default=200,
        help="Downloaded samples per streaming part. Default: 200; final parts shrink to remaining target/dataset samples.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(DATA_DIR / "study_audio_parts"),
        help="Temporary directory for the current downloaded part.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_STUDY_DATASETS),
        help=(
            "Comma-separated dataset keys. Supported: song_describer, "
            "mtg_jamendo_train, mtg_jamendo_validation, "
            "audioset_balanced_train, audioset_balanced_test, "
            "audioset_unbalanced_train, audioset_unbalanced_test."
        ),
    )
    parser.add_argument(
        "--keep-parts",
        action="store_true",
        help="Keep downloaded part folders for debugging instead of deleting them.",
    )
    return parser.parse_args()


def download_part_to_dir(
    dataset_key: str,
    part_dir: Path,
    dataset_state: dict[str, Any],
    part_size: int,
) -> DownloadedPart:
    records, next_dataset_state, exhausted = download_part(
        dataset_key=dataset_key,
        part_dir=part_dir,
        dataset_state=dict(dataset_state),
        part_size=part_size,
        progress_position=1,
    )
    return DownloadedPart(
        records=records,
        next_dataset_state=next_dataset_state,
        exhausted=exhausted,
        part_dir=part_dir,
    )


def get_collection_count(client: QdrantClient, collection_name: str) -> int:
    result = client.count(collection_name=collection_name, exact=True)
    return int(result.count)


def build_payload(record: AudioRecord) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "audio_id": record.audio_id,
        "dataset_id": record.dataset_id,
        "source_dataset": record.source_dataset,
        "captions": record.captions,
    }
    optional_values = {
        "dataset_key": record.dataset_key,
        "dataset_split": record.dataset_split,
        "dataset_subset": record.dataset_subset,
        "archive_idx": record.archive_idx,
    }
    payload.update(
        {
            key: value
            for key, value in optional_values.items()
            if value is not None
        }
    )
    return payload


def index_records(
    records: list[DownloadedAudio],
    client: QdrantClient,
    collection_name: str,
    model: PretrainedCLAPEncoder,
    batch_size: int,
) -> int:
    indexed_count = 0

    for start in tqdm(
        range(0, len(records), batch_size),
        desc="Index CLAP embeddings",
        position=0,
    ):
        batch = records[start : start + batch_size]
        audio_embeddings = model.encode_audio([str(item.audio_path) for item in batch])
        vectors = audio_embeddings.cpu().numpy().tolist()
        ids = [item.record.audio_id for item in batch]
        payloads = [build_payload(item.record) for item in batch]

        upsert_vectors(
            client=client,
            collection_name=collection_name,
            ids=ids,
            vectors=vectors,
            payloads=payloads,
            batch_size=batch_size,
        )
        indexed_count += len(batch)

    return indexed_count


def main() -> None:
    args = parse_args()
    ensure_data_dirs()
    config = load_config(args.config)

    collection_name = args.collection_name or get_nested(
        config,
        "qdrant",
        "collection_name",
        "reflectra_music_clap",
    )
    qdrant_url = args.qdrant_url or get_nested(
        config,
        "qdrant",
        "url",
        "http://localhost:6333",
    )
    model_name = args.model_name or get_nested(
        config,
        "models",
        "clap",
        "laion/clap-htsat-unfused",
    )
    batch_size = args.batch_size or int(get_nested(config, "audio_index", "batch_size", 8))
    vector_size = args.vector_size or int(get_nested(config, "qdrant", "vector_size", 512))
    dataset_keys = [item.strip() for item in args.datasets.split(",") if item.strip()]
    work_dir = resolve_project_path(args.work_dir)
    state_path = work_dir / "study_fill_state.json"

    print(f"[INFO] Datasets: {', '.join(dataset_keys)}")
    print("[INFO] MusicCaps is excluded.")
    print(f"[INFO] Target collection samples: {args.target_samples}")
    print(f"[INFO] Part size: {args.part_size}")
    print(f"[INFO] Temporary work directory: {work_dir}")
    print(f"[INFO] Audio extensions supported by base indexer: {DEFAULT_AUDIO_EXTENSIONS}")

    client = get_qdrant_client(url=qdrant_url)
    create_collection_if_not_exists(
        client=client,
        collection_name=collection_name,
        vector_size=vector_size,
    )

    model = PretrainedCLAPEncoder(model_name=model_name, freeze=True)
    state = load_state(state_path)

    collection_count = get_collection_count(client, collection_name)
    print(f"[INFO] Existing Qdrant collection count: {collection_count}")

    for dataset_key in dataset_keys:
        if collection_count >= args.target_samples:
            break

        dataset_state = state["datasets"].setdefault(dataset_key, {})
        if dataset_state.get("exhausted"):
            print(f"[INFO] Skipping exhausted dataset: {dataset_key}")
            continue

        part_index = 0

        def submit_download(
            executor: ThreadPoolExecutor,
            state_snapshot: dict[str, Any],
            size: int,
        ) -> Future[DownloadedPart]:
            nonlocal part_index
            part_dir = work_dir / "stream_parts" / dataset_key / f"part_{part_index % 2}"
            part_index += 1
            print(
                f"[INFO] Queue download: dataset={dataset_key}, "
                f"size={size}, state={state_snapshot}"
            )
            return executor.submit(
                download_part_to_dir,
                dataset_key=dataset_key,
                part_dir=part_dir,
                dataset_state=state_snapshot,
                part_size=size,
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            future: Future[DownloadedPart] | None = submit_download(
                executor=executor,
                state_snapshot=dict(dataset_state),
                size=min(args.part_size, args.target_samples - collection_count),
            )

            while collection_count < args.target_samples and future is not None:
                downloaded_part = future.result()
                records = downloaded_part.records
                next_dataset_state = downloaded_part.next_dataset_state
                exhausted = downloaded_part.exhausted
                part_dir = downloaded_part.part_dir

                next_future: Future[DownloadedPart] | None = None
                expected_after_current = collection_count + len(records)
                if not exhausted and expected_after_current < args.target_samples:
                    print("[INFO] Prefetching next part while CLAP/Qdrant processes the current part.")
                    next_future = submit_download(
                        executor=executor,
                        state_snapshot=next_dataset_state,
                        size=min(args.part_size, args.target_samples - expected_after_current),
                    )

                if not records:
                    dataset_state.update(next_dataset_state)
                    dataset_state["exhausted"] = exhausted
                    save_state(state_path, state)
                    print(f"[WARN] No usable audio in part for {dataset_key}.")
                    if not args.keep_parts:
                        shutil.rmtree(part_dir, ignore_errors=True)
                    if exhausted:
                        break
                    future = next_future
                    continue

                indexed_count = index_records(
                    records=records,
                    client=client,
                    collection_name=collection_name,
                    model=model,
                    batch_size=batch_size,
                )

                dataset_state.update(next_dataset_state)
                dataset_state["exhausted"] = exhausted
                dataset_state["indexed_records"] = int(dataset_state.get("indexed_records", 0)) + indexed_count
                save_state(state_path, state)

                if not args.keep_parts:
                    shutil.rmtree(part_dir, ignore_errors=True)
                    print(f"[INFO] Removed processed part from disk: {part_dir}")

                collection_count = get_collection_count(client, collection_name)
                print(f"[INFO] Qdrant collection count: {collection_count}")

                if exhausted:
                    print(f"[INFO] Dataset exhausted: {dataset_key}")
                    break

                future = next_future

    final_count = get_collection_count(client, collection_name)
    if final_count < args.target_samples:
        print(
            f"[WARN] Finished available datasets with {final_count}/"
            f"{args.target_samples} Qdrant points."
        )
    else:
        print(f"[INFO] Target reached: {final_count} Qdrant points.")


if __name__ == "__main__":
    main()
