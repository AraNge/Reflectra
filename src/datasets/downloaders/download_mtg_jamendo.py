import os
import json
import argparse
import ast
import csv
import tarfile
from typing import Any, Dict, List

import librosa
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from src.datasets.paths import DATA_DIR, HF_CACHE_DIR, METADATA_DIR, ensure_data_dirs


ensure_data_dirs()

hf_cache_dir = str(HF_CACHE_DIR)
audio_output_root = str(DATA_DIR / "mtg_jamendo_audio")
metadata_output_root = str(METADATA_DIR)
MTG_JAMENDO_REPO_ID = "rkstgr/mtg-jamendo"
SPLIT_FILES = {
    "train": {
        "tracks": "train.tsv",
        "archive_dir": "train",
        "num_archives": 200,
    },
    "validation": {
        "tracks": "valid.tsv",
        "archive_dir": "val",
        "num_archives": 22,
    },
}

os.makedirs(audio_output_root, exist_ok=True)
os.makedirs(metadata_output_root, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/save MTG-Jamendo audio samples from Hugging Face."
    )

    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=100,
        help="Number of MTG-Jamendo samples to save. Default: 100.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to use. Default: train.",
    )

    parser.add_argument(
        "--music-only",
        action="store_true",
        help="Optional. Keep only rows with at least one genre/mood/instrument tag.",
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


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

    if isinstance(value, str):
        return [value]

    return [str(value)]


def save_audio_file(input_path: str, output_path: str) -> bool:
    try:
        waveform, sample_rate = librosa.load(input_path, sr=None, mono=False)

        if waveform.ndim == 2:
            waveform = waveform.T

        if waveform.size == 0:
            return False

        sf.write(output_path, waveform, sample_rate)
        return os.path.exists(output_path)

    except Exception as e:
        print(f"-> Failed to save audio: {e}")
        return False


def clean_tag(tag: str) -> str:
    return (
        str(tag)
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def build_captions(
    genres: List[str],
    instruments: List[str],
    moods: List[str],
) -> List[str]:
    """
    Build multiple text descriptions for one audio track.

    One audio can have many valid captions:
    - one caption per genre
    - one caption per mood
    - one caption per instrument
    - optional combined caption
    """

    genres = [clean_tag(x) for x in genres]
    instruments = [clean_tag(x) for x in instruments]
    moods = [clean_tag(x) for x in moods]

    captions = []

    for genre in genres:
        captions.append(f"A {genre} music track.")

    for mood in moods:
        captions.append(f"A {mood} music track.")

    for instrument in instruments:
        captions.append(f"A music track featuring {instrument}.")

    parts = []

    if genres:
        parts.append(f"genres: {', '.join(genres)}")

    if moods:
        parts.append(f"moods: {', '.join(moods)}")

    if instruments:
        parts.append(f"instruments: {', '.join(instruments)}")

    if parts:
        captions.append("A music track with " + "; ".join(parts) + ".")

    if not captions:
        captions.append("A music track.")

    unique = []
    seen = set()

    for item in captions:
        key = item.lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return unique


def is_tagged_music_sample(genres: List[str], instruments: List[str], moods: List[str]) -> bool:
    return bool(genres or instruments or moods)


def parse_tag_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return normalize_list(value)

    value = str(value).strip()

    if not value:
        return []

    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        parsed = value

    return normalize_list(parsed)


def download_repo_file(filename: str) -> str:
    return hf_hub_download(
        repo_id=MTG_JAMENDO_REPO_ID,
        filename=filename,
        repo_type="dataset",
        cache_dir=hf_cache_dir,
    )


def load_tracks(split: str) -> Dict[int, Dict[str, Any]]:
    split_config = SPLIT_FILES[split]
    tracks_path = download_repo_file(split_config["tracks"])
    tracks: Dict[int, Dict[str, Any]] = {}

    with open(tracks_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:
            track_id = int(row["id"])
            tracks[track_id] = {
                "id": track_id,
                "genres": parse_tag_list(row.get("genres")),
                "instruments": parse_tag_list(row.get("instruments")),
                "moods": parse_tag_list(row.get("moods")),
            }

    return tracks


def extract_archive(archive_path: str, output_dir: str) -> None:
    marker_path = os.path.join(output_dir, ".extracted")

    if os.path.exists(marker_path):
        return

    os.makedirs(output_dir, exist_ok=True)

    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(output_dir)

    with open(marker_path, "w", encoding="utf-8") as f:
        f.write("ok")


def iter_archive_audio_paths(split: str):
    split_config = SPLIT_FILES[split]
    archive_dir = split_config["archive_dir"]
    num_archives = split_config["num_archives"]
    extract_root = os.path.join(audio_output_root, "_extracted", archive_dir)

    for archive_idx in range(num_archives):
        archive_filename = f"data/{archive_dir}/{archive_idx}.tar"
        archive_path = download_repo_file(archive_filename)
        shard_output_dir = os.path.join(extract_root, str(archive_idx))

        extract_archive(
            archive_path=archive_path,
            output_dir=shard_output_dir,
        )

        for root, _, filenames in os.walk(shard_output_dir):
            for filename in filenames:
                if filename.endswith(".opus"):
                    yield os.path.join(root, filename)


def download_mtg_jamendo_samples(number: int, split: str, music_only: bool):
    print("Loading MTG-Jamendo from Hugging Face...")
    tracks = load_tracks(split)

    split_audio_output_dir = os.path.join(audio_output_root, split)
    metadata_output_path = os.path.join(
        metadata_output_root,
        f"mtg_jamendo_{split}_metadata.jsonl",
    )

    os.makedirs(split_audio_output_dir, exist_ok=True)

    print("Dataset loader: direct Hugging Face files")
    print(f"Total rows in split metadata: {len(tracks)}")
    print(f"Requested samples: {number}")
    print(f"Split: {split}")
    print(f"Music-only tag filter: {music_only}")
    print(f"Audio clips will save to: {split_audio_output_dir}")
    print(f"Metadata will save to: {metadata_output_path}")
    print("Starting...")

    saved_count = 0
    seen_count = 0
    skipped_count = 0
    failed_count = 0

    with open(metadata_output_path, "w", encoding="utf-8") as meta_file:
        for audio_source_path in tqdm(iter_archive_audio_paths(split)):
            if saved_count >= number:
                break

            seen_count += 1

            track_id = int(os.path.splitext(os.path.basename(audio_source_path))[0])
            track = tracks.get(track_id)

            if track is None:
                skipped_count += 1
                continue

            genres = track["genres"]
            instruments = track["instruments"]
            moods = track["moods"]

            if music_only and not is_tagged_music_sample(genres, instruments, moods):
                skipped_count += 1
                continue

            audio_id = f"mtg_jamendo_{track_id}"
            file_name = f"{safe_filename(audio_id)}.wav"
            file_path = os.path.join(split_audio_output_dir, file_name)

            if os.path.exists(file_path):
                print(f"\n[{saved_count + 1}/{number}] Already exists: {file_path}")
                saved_count += 1
            else:
                print(f"\n[{saved_count + 1}/{number}] Saving MTG-Jamendo track: {track_id}")
                print(f"-> Genres: {genres}")
                print(f"-> Moods: {moods}")
                print(f"-> Instruments: {instruments}")

                success = save_audio_file(audio_source_path, file_path)

                if not success:
                    failed_count += 1
                    print(f"-> Failed: {track_id}")
                    continue

                print(f"-> Saved to: {file_path}")
                saved_count += 1

            caption_items = build_captions(
                genres=genres,
                instruments=instruments,
                moods=moods,
            )

            metadata = {
                "audio_id": audio_id,
                "audio_path": file_path,
                "captions": caption_items,
                "source_dataset": "rkstgr/mtg-jamendo",
                "split": split,
            }

            meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print("\nDownload summary:")
    print(f"Split: {split}")
    print(f"Rows seen: {seen_count}")
    print(f"Saved audio files: {saved_count}")
    print(f"Skipped rows: {skipped_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {split_audio_output_dir}")
    print(f"Metadata file: {metadata_output_path}")


if __name__ == "__main__":
    args = parse_args()

    download_mtg_jamendo_samples(
        number=args.number,
        split=args.split,
        music_only=args.music_only,
    )
