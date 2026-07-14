import json
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from datasets import Audio, load_dataset
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.datasets.downloaders.download_audioset import (
    is_music_related,
    make_captions_from_labels,
    safe_filename as safe_audioset_filename,
    save_audio as save_audioset_audio,
)
from src.datasets.downloaders.download_mtg_jamendo import (
    MTG_JAMENDO_REPO_ID,
    SPLIT_FILES,
    build_captions as build_mtg_captions,
    load_tracks,
    safe_filename as safe_mtg_filename,
    save_audio_file as save_mtg_audio_file,
)
from src.datasets.downloaders.download_song_describer import (
    get_audio_from_row as get_song_describer_audio,
    safe_filename as safe_song_describer_filename,
    save_audio as save_song_describer_audio,
)
from src.datasets.paths import HF_CACHE_DIR


DEFAULT_STUDY_DATASETS = [
    "song_describer",
    "mtg_jamendo_train",
    "mtg_jamendo_validation",
    "audioset_balanced_train",
    "audioset_unbalanced_train",
]
STATE_VERSION = 1


@dataclass
class AudioRecord:
    audio_id: str
    dataset_id: str
    source_dataset: str
    captions: list[str]


@dataclass
class DownloadedAudio:
    record: AudioRecord
    audio_path: Path


def make_global_audio_id(source_dataset: str, dataset_id: str) -> str:
    return f"{source_dataset}:{dataset_id}"


def load_state(state_path: Path) -> dict[str, Any]:
    if not state_path.exists():
        return {"version": STATE_VERSION, "datasets": {}}

    with state_path.open("r", encoding="utf-8") as state_file:
        state = json.load(state_file)

    if state.get("version") != STATE_VERSION:
        raise RuntimeError(
            f"Unsupported state version in {state_path}: {state.get('version')}"
        )

    state.setdefault("datasets", {})
    return state


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2, sort_keys=True)
    tmp_path.replace(state_path)


def normalize_captions(value: Any) -> list[str]:
    if value is None:
        return []
    values = value if isinstance(value, list) else [value]
    return [str(item).strip() for item in values if str(item).strip()]


def clear_part_dir(part_dir: Path) -> None:
    if part_dir.exists():
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)


def iter_song_describer_part(
    part_dir: Path,
    start_offset: int,
    part_size: int,
    progress_position: int = 0,
) -> tuple[list[DownloadedAudio], int, bool]:
    split = "train"
    ds = load_dataset(
        "renumics/song-describer-dataset",
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )
    ds = ds.cast_column("path", Audio())

    records: list[DownloadedAudio] = []
    next_offset = start_offset
    effective_part_size = min(part_size, max(len(ds) - start_offset, 0))

    with tqdm(
        total=effective_part_size,
        desc="Download song_describer part",
        unit="sample",
        position=progress_position,
        leave=False,
    ) as progress:
        for row_index in range(start_offset, len(ds)):
            if len(records) >= effective_part_size:
                break

            row = ds[row_index]
            audio = get_song_describer_audio(row)
            next_offset = row_index + 1

            if audio is None:
                continue

            dataset_id = str(row.get("__index_level_0__", row_index))
            audio_id = make_global_audio_id("renumics/song-describer-dataset", dataset_id)
            file_id = f"song_describer_{dataset_id}"
            audio_path = part_dir / f"{safe_song_describer_filename(file_id)}.wav"

            if not save_song_describer_audio(audio, str(audio_path)):
                continue

            records.append(
                DownloadedAudio(
                    audio_path=audio_path,
                    record=AudioRecord(
                        audio_id=audio_id,
                        dataset_id=dataset_id,
                        source_dataset="renumics/song-describer-dataset",
                        captions=normalize_captions(row.get("caption")),
                    ),
                )
            )
            progress.update(1)

    exhausted = next_offset >= len(ds) and len(records) < part_size
    return records, next_offset, exhausted


def download_mtg_repo_file(filename: str) -> str:
    return hf_hub_download(
        repo_id=MTG_JAMENDO_REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(HF_CACHE_DIR),
    )


def iter_mtg_audio_paths(
    split: str,
    part_dir: Path,
    start_archive: int,
) -> Iterable[tuple[int, Path]]:
    split_config = SPLIT_FILES[split]
    archive_dir = split_config["archive_dir"]
    num_archives = split_config["num_archives"]

    for archive_idx in range(start_archive, num_archives):
        archive_filename = f"data/{archive_dir}/{archive_idx}.tar"
        archive_path = download_mtg_repo_file(archive_filename)
        extract_dir = part_dir / "_mtg_extract" / str(archive_idx)
        extract_dir.mkdir(parents=True, exist_ok=True)

        with tarfile.open(archive_path, "r") as archive:
            archive.extractall(extract_dir)

        for audio_path in sorted(extract_dir.rglob("*.opus")):
            yield archive_idx, audio_path


def iter_mtg_jamendo_part(
    split: str,
    part_dir: Path,
    dataset_state: dict[str, Any],
    part_size: int,
    progress_position: int = 0,
) -> tuple[list[DownloadedAudio], dict[str, Any], bool]:
    tracks = load_tracks(split)
    start_archive = int(dataset_state.get("archive_idx", 0))
    skip_seen = int(dataset_state.get("archive_seen", 0))

    records: list[DownloadedAudio] = []
    next_state = {"archive_idx": start_archive, "archive_seen": skip_seen}
    current_archive = start_archive
    seen_in_archive = 0

    with tqdm(
        total=part_size,
        desc=f"Download mtg_jamendo_{split} part",
        unit="sample",
        position=progress_position,
        leave=False,
    ) as progress:
        for archive_idx, source_path in iter_mtg_audio_paths(split, part_dir, start_archive):
            if len(records) >= part_size:
                break

            if archive_idx != current_archive:
                current_archive = archive_idx
                seen_in_archive = 0
                skip_seen = 0

            seen_in_archive += 1
            if archive_idx == start_archive and seen_in_archive <= skip_seen:
                continue

            next_state = {"archive_idx": archive_idx, "archive_seen": seen_in_archive}

            track_id = int(source_path.stem)
            track = tracks.get(track_id)
            if track is None:
                continue

            dataset_id = str(track_id)
            audio_id = make_global_audio_id("rkstgr/mtg-jamendo", dataset_id)
            file_id = f"mtg_jamendo_{dataset_id}"
            audio_path = part_dir / f"{safe_mtg_filename(file_id)}.wav"
            if not save_mtg_audio_file(str(source_path), str(audio_path)):
                continue

            records.append(
                DownloadedAudio(
                    audio_path=audio_path,
                    record=AudioRecord(
                        audio_id=audio_id,
                        dataset_id=dataset_id,
                        source_dataset="rkstgr/mtg-jamendo",
                        captions=build_mtg_captions(
                            genres=track["genres"],
                            instruments=track["instruments"],
                            moods=track["moods"],
                        ),
                    ),
                )
            )
            progress.update(1)

    num_archives = SPLIT_FILES[split]["num_archives"]
    exhausted = current_archive >= num_archives - 1 and len(records) < part_size
    return records, next_state, exhausted


def parse_audioset_key(dataset_key: str) -> tuple[str, str]:
    parts = dataset_key.split("_")
    if len(parts) != 3 or parts[0] != "audioset":
        raise ValueError(f"Invalid AudioSet dataset key: {dataset_key}")
    return parts[1], parts[2]


def iter_audioset_part(
    subset: str,
    split: str,
    part_dir: Path,
    start_offset: int,
    part_size: int,
    progress_position: int = 0,
) -> tuple[list[DownloadedAudio], int, bool]:
    ds = load_dataset(
        "agkphysics/AudioSet",
        subset,
        split=split,
        cache_dir=str(HF_CACHE_DIR),
        streaming=True,
    )

    records: list[DownloadedAudio] = []
    next_offset = start_offset
    seen_count = 0

    with tqdm(
        total=part_size,
        desc=f"Download audioset_{subset}_{split} part",
        unit="sample",
        position=progress_position,
        leave=False,
    ) as progress:
        for row in ds:
            if seen_count < start_offset:
                seen_count += 1
                continue

            if len(records) >= part_size:
                break

            seen_count += 1
            next_offset = seen_count

            human_labels = row.get("human_labels") or []
            if not is_music_related(human_labels):
                continue

            dataset_id = str(row["video_id"])
            audio_id = make_global_audio_id("agkphysics/AudioSet", dataset_id)
            audio_path = part_dir / f"{safe_audioset_filename(dataset_id)}.wav"
            if not save_audioset_audio(row["audio"], str(audio_path)):
                continue

            records.append(
                DownloadedAudio(
                    audio_path=audio_path,
                    record=AudioRecord(
                        audio_id=audio_id,
                        dataset_id=dataset_id,
                        source_dataset="agkphysics/AudioSet",
                        captions=make_captions_from_labels(human_labels),
                    ),
                )
            )
            progress.update(1)

    exhausted = len(records) < part_size
    return records, next_offset, exhausted


def download_part(
    dataset_key: str,
    part_dir: Path,
    dataset_state: dict[str, Any],
    part_size: int,
    progress_position: int = 0,
) -> tuple[list[DownloadedAudio], dict[str, Any], bool]:
    clear_part_dir(part_dir)

    if dataset_key == "song_describer":
        records, next_offset, exhausted = iter_song_describer_part(
            part_dir=part_dir,
            start_offset=int(dataset_state.get("offset", 0)),
            part_size=part_size,
            progress_position=progress_position,
        )
        return records, {"offset": next_offset}, exhausted

    if dataset_key in {"mtg_jamendo_train", "mtg_jamendo_validation"}:
        split = dataset_key.replace("mtg_jamendo_", "")
        return iter_mtg_jamendo_part(
            split=split,
            part_dir=part_dir,
            dataset_state=dataset_state,
            part_size=part_size,
            progress_position=progress_position,
        )

    if dataset_key.startswith("audioset_"):
        subset, split = parse_audioset_key(dataset_key)
        records, next_offset, exhausted = iter_audioset_part(
            subset=subset,
            split=split,
            part_dir=part_dir,
            start_offset=int(dataset_state.get("offset", 0)),
            part_size=part_size,
            progress_position=progress_position,
        )
        return records, {"offset": next_offset}, exhausted

    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def download_song_describer_by_dataset_id(dataset_id: str, output_dir: Path) -> Path:
    split = "train"
    row_index = int(dataset_id)
    ds = load_dataset(
        "renumics/song-describer-dataset",
        split=split,
        cache_dir=str(HF_CACHE_DIR),
    )
    ds = ds.cast_column("path", Audio())

    row = ds[row_index]
    audio = get_song_describer_audio(row)
    if audio is None:
        raise RuntimeError(f"No Song Describer audio found for dataset_id={dataset_id}")

    audio_path = output_dir / f"{safe_song_describer_filename(f'song_describer_{dataset_id}')}.wav"
    if not save_song_describer_audio(audio, str(audio_path)):
        raise RuntimeError(f"Failed to save Song Describer audio for dataset_id={dataset_id}")

    return audio_path


def download_mtg_jamendo_by_dataset_id(dataset_id: str, output_dir: Path) -> Path:
    track_id = int(dataset_id)
    audio_path = output_dir / f"{safe_mtg_filename(f'mtg_jamendo_{dataset_id}')}.wav"

    for split in ("train", "validation"):
        for _, source_path in iter_mtg_audio_paths(
            split=split,
            part_dir=output_dir / "_mtg_lookup",
            start_archive=0,
        ):
            if source_path.stem != str(track_id):
                continue

            if not save_mtg_audio_file(str(source_path), str(audio_path)):
                raise RuntimeError(f"Failed to save MTG-Jamendo audio for dataset_id={dataset_id}")

            shutil.rmtree(output_dir / "_mtg_lookup", ignore_errors=True)
            return audio_path

    shutil.rmtree(output_dir / "_mtg_lookup", ignore_errors=True)
    raise RuntimeError(f"MTG-Jamendo track not found for dataset_id={dataset_id}")


def download_audioset_by_dataset_id(dataset_id: str, output_dir: Path) -> Path:
    audio_path = output_dir / f"{safe_audioset_filename(dataset_id)}.wav"

    for subset, split in (
        ("balanced", "train"),
        ("balanced", "test"),
        ("unbalanced", "train"),
        ("unbalanced", "test"),
    ):
        ds = load_dataset(
            "agkphysics/AudioSet",
            subset,
            split=split,
            cache_dir=str(HF_CACHE_DIR),
            streaming=True,
        )

        for row in tqdm(ds, desc=f"Find AudioSet {dataset_id} in {subset}/{split}"):
            if str(row.get("video_id")) != str(dataset_id):
                continue

            if not save_audioset_audio(row["audio"], str(audio_path)):
                raise RuntimeError(f"Failed to save AudioSet audio for dataset_id={dataset_id}")

            return audio_path

    raise RuntimeError(f"AudioSet clip not found for dataset_id={dataset_id}")


def download_audio_from_metadata(
    metadata: dict[str, Any],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = metadata.get("payload", metadata)
    source_dataset = str(payload.get("source_dataset", ""))
    dataset_id = str(payload.get("dataset_id", ""))

    if not source_dataset or not dataset_id:
        raise ValueError("Metadata must contain source_dataset and dataset_id.")

    if source_dataset == "renumics/song-describer-dataset":
        return download_song_describer_by_dataset_id(dataset_id, output_dir)

    if source_dataset == "rkstgr/mtg-jamendo":
        return download_mtg_jamendo_by_dataset_id(dataset_id, output_dir)

    if source_dataset == "agkphysics/AudioSet":
        return download_audioset_by_dataset_id(dataset_id, output_dir)

    raise ValueError(f"Unsupported source_dataset: {source_dataset}")
