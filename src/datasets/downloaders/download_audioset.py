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
audio_output_dir = os.path.join(project_root, "data", "audioset_audio")
metadata_output_path = os.path.join(project_root, "data", "audioset_metadata.jsonl")

os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(audio_output_dir, exist_ok=True)


MUSIC_KEYWORDS = [
    "music",
    "musical",
    "song",
    "singing",
    "singer",
    "vocal",
    "choir",
    "chant",
    "humming",
    "rapping",
    "hip hop",
    "rap",
    "pop",
    "rock",
    "jazz",
    "blues",
    "classical",
    "electronic",
    "dance",
    "techno",
    "house",
    "disco",
    "reggae",
    "country",
    "folk",
    "opera",
    "instrument",
    "guitar",
    "electric guitar",
    "bass guitar",
    "piano",
    "keyboard",
    "synthesizer",
    "violin",
    "cello",
    "flute",
    "saxophone",
    "trumpet",
    "drum",
    "drum kit",
    "percussion",
    "beat",
    "melody",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/save music-related AudioSet samples from Hugging Face."
    )

    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=100,
        help="Number of music-related AudioSet samples to save. Default: 100.",
    )

    parser.add_argument(
        "--subset",
        type=str,
        default="balanced",
        choices=["balanced", "unbalanced", "full"],
        help=(
            "AudioSet subset/config to use. "
            "Options: balanced (~35.8k), unbalanced (~1.76M), full (~1.77M). "
            "Default: balanced."
        ),
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use (train or test). Default: train.",
    )

    return parser.parse_args()


def safe_filename(value: str) -> str:
    return (
        str(value)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(",", "_")
    )


def is_music_related(human_labels: List[str]) -> bool:
    if not human_labels:
        return False

    label_text = " ".join(str(label).lower() for label in human_labels)

    return any(keyword in label_text for keyword in MUSIC_KEYWORDS)


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


def download_audioset_samples(number: int, subset: str, split: str):
    print("Loading AudioSet from Hugging Face...")
    print(f"Subset/config: {subset}")
    print(f"Split: {split}")

    ds = load_dataset(
        "agkphysics/AudioSet",
        subset,
        split=split,
        cache_dir=hf_cache_dir,
        streaming=True,
    )

    subset_output_dir = os.path.join(audio_output_dir, subset)
    subset_metadata_path = os.path.join(
        project_root,
        "data",
        f"audioset_{subset}_metadata.jsonl",
    )

    os.makedirs(subset_output_dir, exist_ok=True)

    print(f"Requested music-related samples: {number}")
    print(f"Audio clips will save to: {subset_output_dir}")
    print(f"Metadata will save to: {subset_metadata_path}")
    print("Starting...")

    saved_count = 0
    seen_count = 0
    skipped_non_music = 0
    failed_count = 0

    with open(subset_metadata_path, "w", encoding="utf-8") as meta_file:
        for row in tqdm(ds):
            if saved_count >= number:
                break

            seen_count += 1

            video_id = row["video_id"]
            audio = row["audio"]
            human_labels = row["human_labels"]

            if not is_music_related(human_labels):
                skipped_non_music += 1
                continue

            file_name = f"{safe_filename(video_id)}.wav"
            file_path = os.path.join(subset_output_dir, file_name)

            if os.path.exists(file_path):
                print(f"\n[{saved_count + 1}/{number}] Already exists: {file_path}")
                saved_count += 1
            else:
                print(f"\n[{saved_count + 1}/{number}] Saving AudioSet clip: {video_id}")
                print(f"-> Human labels: {human_labels}")

                success = save_audio(audio, file_path)

                if not success:
                    failed_count += 1
                    print(f"-> Failed: {video_id}")
                    continue

                print(f"-> Saved to: {file_path}")
                saved_count += 1

            metadata = {
                "audio_id": video_id,
                "audio_path": file_path,
                "human_labels": human_labels,
                "source_dataset": "agkphysics/AudioSet",
                "subset": subset,
                "split": split,
            }

            meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print("\nDownload summary:")
    print(f"Subset/config: {subset}")
    print(f"Split: {split}")
    print(f"Rows seen: {seen_count}")
    print(f"Saved music samples: {saved_count}")
    print(f"Skipped non-music rows: {skipped_non_music}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {subset_output_dir}")
    print(f"Metadata file: {subset_metadata_path}")


if __name__ == "__main__":
    args = parse_args()
    download_audioset_samples(
        number=args.number,
        subset=args.subset,
        split=args.split,
    )