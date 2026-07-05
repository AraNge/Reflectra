import os
import json
import argparse
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


# Project paths
module_dir = os.path.dirname(os.path.abspath(__file__))  # src/modules
project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))

hf_cache_dir = os.path.join(project_root, "data", "hf_cache")
audio_output_root = os.path.join(project_root, "data", "mtg_jamendo_audio")
metadata_output_root = os.path.join(project_root, "data", "mtg_jamendo_metadata")

os.makedirs(hf_cache_dir, exist_ok=True)
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


def save_audio(audio: Dict[str, Any], output_path: str) -> bool:
    """
    Expected Hugging Face audio format:
        audio = {
            "path": "...",
            "array": np.ndarray,
            "sampling_rate": int
        }
    """

    try:
        array = np.asarray(audio["array"])
        sampling_rate = int(audio["sampling_rate"])

        if array.size == 0:
            return False

        # If audio is [channels, samples], convert to [samples, channels]
        if array.ndim == 2 and array.shape[0] < array.shape[1]:
            array = array.T

        sf.write(output_path, array, sampling_rate)
        return os.path.exists(output_path)

    except Exception as e:
        print(f"-> Failed to save audio: {e}")
        return False


def build_caption(genres: List[str], instruments: List[str], moods: List[str]) -> str:
    """
    MTG-Jamendo has tags, not natural captions.
    This creates a simple caption useful for CLAP text encoder.
    """

    parts = []

    if genres:
        parts.append(f"genres: {', '.join(genres)}")

    if moods:
        parts.append(f"moods: {', '.join(moods)}")

    if instruments:
        parts.append(f"instruments: {', '.join(instruments)}")

    if not parts:
        return "a music track"

    return "A music track with " + "; ".join(parts) + "."


def is_tagged_music_sample(genres: List[str], instruments: List[str], moods: List[str]) -> bool:
    return bool(genres or instruments or moods)


def download_mtg_jamendo_samples(number: int, split: str, music_only: bool):
    print("Loading MTG-Jamendo from Hugging Face...")

    ds = load_dataset(
        "rkstgr/mtg-jamendo",
        split=split,
        cache_dir=hf_cache_dir,
    )

    split_audio_output_dir = os.path.join(audio_output_root, split)
    metadata_output_path = os.path.join(
        metadata_output_root,
        f"mtg_jamendo_{split}_metadata.jsonl",
    )

    os.makedirs(split_audio_output_dir, exist_ok=True)

    print(f"Dataset columns: {ds.column_names}")
    print(f"Total rows in split: {len(ds)}")
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
        for row in tqdm(ds):
            if saved_count >= number:
                break

            seen_count += 1

            track_id = row["id"]
            artist_id = row["artist_id"]
            album_id = row["album_id"]
            duration_in_sec = row["duration_in_sec"]

            genres = normalize_list(row.get("genres"))
            instruments = normalize_list(row.get("instruments"))
            moods = normalize_list(row.get("moods"))

            if music_only and not is_tagged_music_sample(genres, instruments, moods):
                skipped_count += 1
                continue

            audio = row["audio"]

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
                print(f"-> Duration: {duration_in_sec}s")

                success = save_audio(audio, file_path)

                if not success:
                    failed_count += 1
                    print(f"-> Failed: {track_id}")
                    continue

                print(f"-> Saved to: {file_path}")
                saved_count += 1

            caption = build_caption(
                genres=genres,
                instruments=instruments,
                moods=moods,
            )

            metadata = {
                "audio_id": audio_id,
                "track_id": track_id,
                "artist_id": artist_id,
                "album_id": album_id,
                "duration_in_sec": duration_in_sec,
                "audio_path": file_path,
                "genres": genres,
                "instruments": instruments,
                "moods": moods,
                "caption": caption,
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