import os
import json
import argparse
from typing import Any, List
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# Project paths
module_dir = os.path.dirname(os.path.abspath(__file__))  # src/modules
project_root = os.path.abspath(os.path.join(module_dir, "..", ".."))

hf_cache_dir = os.path.join(project_root, "data", "hf_cache")
image_output_dir = os.path.join(project_root, "data", "coco_captions_images")
metadata_output_path = os.path.join(project_root, "data", "coco_captions_metadata.jsonl")

os.makedirs(hf_cache_dir, exist_ok=True)
os.makedirs(image_output_dir, exist_ok=True)


MUSIC_RELATED_KEYWORDS = [
    # People / social scenes
    "people",
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "group",
    "crowd",
    "friends",
    "couple",
    "party",
    "festival",
    "concert",
    "stage",
    "performance",
    "performing",
    "dancing",
    "dance",
    "singing",
    "singer",
    "musician",
    "band",

    # Music objects
    "guitar",
    "piano",
    "drum",
    "violin",
    "microphone",
    "instrument",

    # Vibe / places useful for music matching
    "beach",
    "city",
    "street",
    "night",
    "club",
    "bar",
    "restaurant",
    "park",
    "rain",
    "snow",
    "mountain",
    "ocean",
    "car",
    "bike",
    "bicycle",
    "motorcycle",
    "skateboard",
    "running",
    "gym",
    "sport",
    "wedding",
    "romantic",
    "train",
    "bus",
    "airplane",
    "boat",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/save music-vibe-related COCO Captions samples from Hugging Face."
    )

    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=10,
        help="Number of COCO Captions samples to save. Default: 10.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation"],
        help="Dataset split to use (train or validation). Default: train.",
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


def normalize_captions(captions_field: Any) -> List[str]:
    if captions_field is None:
        return []

    if isinstance(captions_field, str):
        return [captions_field]

    if isinstance(captions_field, list):
        return [str(caption) for caption in captions_field]

    return [str(captions_field)]


def is_music_vibe_related(captions: List[str]) -> bool:
    """
    Keeps images whose captions contain scenes useful for image-to-music retrieval.
    This does not mean the image is literally about music.
    It keeps images that can map to a music mood/vibe.
    """
    if not captions:
        return False

    text = " ".join(captions).lower()
    return any(keyword in text for keyword in MUSIC_RELATED_KEYWORDS)


def save_image(image: Image.Image, output_path: str) -> bool:
    try:
        if not isinstance(image, Image.Image):
            return False

        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95)

        return os.path.exists(output_path)

    except Exception as e:
        print(f"-> Failed to save image: {e}")
        return False


def download_coco_captions_samples(number: int, split: str):
    print("Loading COCO Captions from Hugging Face...")

    dataset = load_dataset(
        "whyen-wang/coco_captions",
        cache_dir=hf_cache_dir,
    )

    if split not in dataset:
        available_splits = list(dataset.keys())
        raise ValueError(
            f"Split '{split}' not found. Available splits: {available_splits}"
        )

    ds = dataset[split]

    print(f"Dataset columns: {ds.column_names}")
    print(f"Total rows in split: {len(ds)}")
    print(f"Requested music-vibe-related samples: {number}")
    print(f"Images will save to: {image_output_dir}")
    print(f"Metadata will save to: {metadata_output_path}")
    print("Starting...")

    saved_count = 0
    seen_count = 0
    skipped_non_music_vibe = 0
    failed_count = 0

    with open(metadata_output_path, "w", encoding="utf-8") as meta_file:
        for idx, row in enumerate(tqdm(ds)):
            if saved_count >= number:
                break

            seen_count += 1

            image = row["image"]
            captions = normalize_captions(row["captions"])

            if not is_music_vibe_related(captions):
                skipped_non_music_vibe += 1
                continue

            filename = f"coco_{split}_{idx}.jpg"
            filename = safe_filename(filename)
            file_path = os.path.join(image_output_dir, filename)

            if os.path.exists(file_path):
                print(f"\n[{saved_count + 1}/{number}] Already exists: {file_path}")
                saved_count += 1
            else:
                print(f"\n[{saved_count + 1}/{number}] Saving COCO image: {filename}")
                print(f"-> Captions: {captions[:2]}")

                success = save_image(image, file_path)

                if not success:
                    failed_count += 1
                    print(f"-> Failed: {filename}")
                    continue

                print(f"-> Saved to: {file_path}")
                saved_count += 1

            metadata = {
                "image_id": f"coco_{split}_{idx}",
                "filename": filename,
                "image_path": file_path,
                "captions": captions,
                "split": split,
                "source_dataset": "whyen-wang/coco_captions",
            }

            meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print("\nDownload summary:")
    print(f"Split: {split}")
    print(f"Rows seen: {seen_count}")
    print(f"Saved samples: {saved_count}")
    print(f"Skipped non-music-vibe rows: {skipped_non_music_vibe}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {image_output_dir}")
    print(f"Metadata file: {metadata_output_path}")


if __name__ == "__main__":
    args = parse_args()
    download_coco_captions_samples(
        number=args.number,
        split=args.split,
    )