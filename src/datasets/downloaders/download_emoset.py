import os
import json
import argparse
from typing import Any
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from src.datasets.paths import DATA_DIR, HF_CACHE_DIR, METADATA_DIR, ensure_data_dirs


ensure_data_dirs()

hf_cache_dir = str(HF_CACHE_DIR)
image_output_dir = str(DATA_DIR / "emoset_images")

os.makedirs(image_output_dir, exist_ok=True)


EMOTION_TO_MUSIC_QUERY = {
    "amusement": "fun playful upbeat pop music with happy energy",
    "anger": "intense powerful dark pop or rock music with aggressive energy",
    "awe": "cinematic atmospheric emotional pop music with spacious feeling",
    "contentment": "calm warm relaxing pop music with peaceful mood",
    "disgust": "dark tense experimental music with uncomfortable mood",
    "excitement": "energetic dance pop music with high energy and party mood",
    "fear": "dark suspenseful electronic music with tense atmosphere",
    "sadness": "sad emotional pop ballad with melancholic mood",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/save EmoSet image-emotion samples from Hugging Face."
    )

    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=100,
        help="Number of EmoSet samples to save. Default: 100.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use. Default: train.",
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
        .replace("<", "")
        .replace(">", "")
    )


def get_first_image(images_field: Any) -> Image.Image:
    """
    EmoSet column:
        images = [PIL.Image]
    """
    if isinstance(images_field, list) and len(images_field) > 0:
        image = images_field[0]
    else:
        image = images_field

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image.Image, got {type(image)}")

    return image


def save_image(image: Image.Image, output_path: str) -> bool:
    try:
        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95)
        return os.path.exists(output_path)

    except Exception as e:
        print(f"-> Failed to save image: {e}")
        return False


def make_music_query_from_emotion(emotion: str) -> str:
    emotion = str(emotion).strip().lower()

    return EMOTION_TO_MUSIC_QUERY.get(
        emotion,
        f"{emotion} mood music matching the visual emotion of the image",
    )


def download_emoset_samples(number: int, split: str):
    print("Loading EmoSet from Hugging Face...")

    ds = load_dataset(
        "LiangJian24/EmoSet",
        split=split,
        cache_dir=hf_cache_dir,
    )

    split_image_output_dir = os.path.join(image_output_dir, split)
    metadata_output_path = os.path.join(
        str(METADATA_DIR),
        f"emoset_{split}_metadata.jsonl",
    )

    os.makedirs(split_image_output_dir, exist_ok=True)

    print(f"Dataset columns: {ds.column_names}")
    print(f"Total rows in split: {len(ds)}")
    print(f"Requested samples: {number}")
    print(f"Images will save to: {split_image_output_dir}")
    print(f"Metadata will save to: {metadata_output_path}")
    print("Starting...")

    saved_count = 0
    failed_count = 0

    with open(metadata_output_path, "w", encoding="utf-8") as meta_file:
        for idx, row in enumerate(tqdm(ds)):
            if saved_count >= number:
                break

            try:
                image = get_first_image(row["images"])
                emotion = str(row["answer"]).strip().lower()
                filename = f"emoset_{split}_{idx}_{safe_filename(emotion)}.jpg"
                file_path = os.path.join(split_image_output_dir, filename)

                if os.path.exists(file_path):
                    print(f"\n[{saved_count + 1}/{number}] Already exists: {file_path}")
                else:
                    print(f"\n[{saved_count + 1}/{number}] Saving EmoSet image: {filename}")
                    print(f"-> Emotion: {emotion}")

                    success = save_image(image, file_path)

                    if not success:
                        failed_count += 1
                        print(f"-> Failed: {filename}")
                        continue

                    print(f"-> Saved to: {file_path}")

                metadata = {
                    "image_id": f"emoset_{split}_{idx}",
                    "image_path": file_path,
                    "captions": [make_music_query_from_emotion(emotion)],
                    "split": split,
                    "source_dataset": "LiangJian24/EmoSet",
                }

                meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")
                saved_count += 1

            except Exception as e:
                failed_count += 1
                print(f"\n-> Failed row {idx}: {e}")

    print("\nDownload summary:")
    print(f"Split: {split}")
    print(f"Saved samples: {saved_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {split_image_output_dir}")
    print(f"Metadata file: {metadata_output_path}")


if __name__ == "__main__":
    args = parse_args()
    download_emoset_samples(
        number=args.number,
        split=args.split,
    )
