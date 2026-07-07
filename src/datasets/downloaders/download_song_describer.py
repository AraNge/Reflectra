import os
import json
import argparse
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm


from src.datasets.paths import DATA_DIR, HF_CACHE_DIR, METADATA_DIR, ensure_data_dirs


ensure_data_dirs()

hf_cache_dir = str(HF_CACHE_DIR)
audio_output_dir = str(DATA_DIR / "song_describer_audio")
metadata_output_path = str(METADATA_DIR / "song_describer_metadata.jsonl")

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


def normalize_captions(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        values = value
    else:
        values = [value]

    captions = []

    for item in values:
        text = str(item).strip()

        if text:
            captions.append(text)

    return captions



def get_audio_from_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Supports both old HF audio dict format and new torchcodec AudioDecoder format.
    """

    audio = row.get("path")

    if isinstance(audio, dict):
        if "array" in audio and "sampling_rate" in audio:
            return {
                "array": np.asarray(audio["array"]),
                "sampling_rate": int(audio["sampling_rate"]),
            }

    if hasattr(audio, "get_all_samples"):
        try:
            samples = audio.get_all_samples()

            array = samples.data

            # torch.Tensor -> numpy
            if hasattr(array, "detach"):
                array = array.detach().cpu().numpy()
            else:
                array = np.asarray(array)

            sampling_rate = int(samples.sample_rate)

            return {
                "array": array,
                "sampling_rate": sampling_rate,
            }

        except Exception as e:
            print(f"-> Failed to decode AudioDecoder: {e}")
            return None

    return None



def save_audio(audio: Dict[str, Any], output_path: str) -> bool:
    try:
        array = np.asarray(audio["array"])
        sampling_rate = int(audio["sampling_rate"])

        if array.size == 0:
            return False

        # torchcodec often gives [channels, samples]
        # soundfile wants [samples, channels]
        if array.ndim == 2 and array.shape[0] <= 8 and array.shape[0] < array.shape[1]:
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

    ds = ds.cast_column("path", Audio())

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

            dataset_index = row.get("__index_level_0__", idx)

            audio_id = f"song_describer_{dataset_index}"
            file_name = f"{safe_filename(audio_id)}.wav"
            file_path = os.path.join(audio_output_dir, file_name)

            if os.path.exists(file_path):
                print(f"\n[{idx + 1}/{number}] Already exists: {file_path}")
                saved_count += 1
            else:
                print(f"\n[{idx + 1}/{number}] Saving Song Describer audio: {audio_id}")

                success = save_audio(audio, file_path)

                if not success:
                    failed_count += 1
                    print(f"-> Failed: {audio_id}")
                    continue

                print(f"-> Saved to: {file_path}")
                saved_count += 1

            metadata = {
                "audio_id": audio_id,
                "audio_path": file_path,
                "captions": [row.get("caption")],
                "source_dataset": "renumics/song-describer-dataset",
                "split": split,
            }

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
