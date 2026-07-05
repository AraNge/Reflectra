import os
import json
import argparse
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm


# Project paths
module_dir = os.path.dirname(os.path.abspath(__file__))  # src/modules
project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))

hf_cache_dir = os.path.join(project_root, "data", "hf_cache")
audio_output_dir = os.path.join(project_root, "data", "song_describer_audio")
metadata_output_path = os.path.join(project_root, "data", "song_describer_metadata.jsonl")

os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(audio_output_dir, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/save Song Describer Dataset audio samples from Hugging Face."
    )

    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=100,
        help="Number of Song Describer samples to save. Default: 100.",
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


def get_audio_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    In this HF dataset, the audio column appears to be named 'path'.

    Expected format:
        row["path"] = {
            "path": "...",
            "array": np.ndarray,
            "sampling_rate": int
        }
    """

    if "path" in row and isinstance(row["path"], dict):
        audio = row["path"]

        if "array" in audio and "sampling_rate" in audio:
            return audio

    # fallback: search any audio-like dict
    for value in row.values():
        if isinstance(value, dict) and "array" in value and "sampling_rate" in value:
            return value

    return None


def get_caption_or_description(row: Dict[str, Any]) -> Optional[str]:
    """
    Some versions of Song Describer may expose captions/descriptions.
    Your shown columns do not include captions, but this keeps the script robust.
    """

    possible_keys = [
        "caption",
        "description",
        "descriptions",
        "text",
        "sentence",
        "summary",
        "prompt",
    ]

    for key in possible_keys:
        if key in row and row[key] is not None:
            value = row[key]

            if isinstance(value, list):
                return " ".join(str(v) for v in value)

            return str(value)

    return None


def save_audio(audio: Dict[str, Any], output_path: str) -> bool:
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


def download_song_describer_samples(number: int):
    split = "train"

    print("Loading Song Describer Dataset from Hugging Face...")

    ds = load_dataset(
        "renumics/song-describer-dataset",
        split=split,
        cache_dir=hf_cache_dir,
    )

    print(f"Dataset columns: {ds.column_names}")
    print(f"Total rows in split: {len(ds)}")
    print(f"Requested samples: {number}")
    print(f"Audio clips will save to: {audio_output_dir}")
    print(f"Metadata will save to: {metadata_output_path}")

    number = min(number, len(ds))

    saved_count = 0
    failed_count = 0

    with open(metadata_output_path, "w", encoding="utf-8") as meta_file:
        for idx, row in enumerate(tqdm(ds.select(range(number)))):
            audio = get_audio_from_row(row)

            if audio is None:
                failed_count += 1
                print(f"\n[{idx + 1}/{number}] Failed: no audio field found.")
                continue

            artist_id = row.get("artist_id")
            album_id = row.get("album_id")
            duration = row.get("duration")
            dataset_index = row.get("__index_level_0__", idx)

            audio_id = f"song_describer_{dataset_index}"
            file_name = f"{safe_filename(audio_id)}.wav"
            file_path = os.path.join(audio_output_dir, file_name)

            if os.path.exists(file_path):
                print(f"\n[{idx + 1}/{number}] Already exists: {file_path}")
                saved_count += 1
            else:
                print(f"\n[{idx + 1}/{number}] Saving Song Describer audio: {audio_id}")
                print(f"-> Artist ID: {artist_id}")
                print(f"-> Album ID: {album_id}")
                print(f"-> Duration: {duration}")

                success = save_audio(audio, file_path)

                if not success:
                    failed_count += 1
                    print(f"-> Failed: {audio_id}")
                    continue

                print(f"-> Saved to: {file_path}")
                saved_count += 1

            caption_or_description = get_caption_or_description(row)

            metadata = {
                "audio_id": audio_id,
                "audio_path": file_path,
                "artist_id": artist_id,
                "album_id": album_id,
                "duration": duration,
                "caption": caption_or_description,
                "source_dataset": "renumics/song-describer-dataset",
                "split": split,
                "dataset_index": dataset_index,
            }

            # Store all non-audio fields safely.
            for key, value in row.items():
                if key == "path":
                    continue

                try:
                    json.dumps(value)
                    metadata[key] = value
                except TypeError:
                    metadata[key] = str(value)

            meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print("\nDownload summary:")
    print(f"Split: {split}")
    print(f"Requested: {number}")
    print(f"Saved audio files: {saved_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {audio_output_dir}")
    print(f"Metadata file: {metadata_output_path}")


if __name__ == "__main__":
    args = parse_args()
    download_song_describer_samples(number=args.number)