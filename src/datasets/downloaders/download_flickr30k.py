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
image_output_dir = os.path.join(project_root, "data", "flickr30k_images")
metadata_output_path = os.path.join(project_root, "data", "flickr30k_metadata.jsonl")

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
    "skateboard",
    "running",
    "gym",
    "sport",
    "wedding",
    "romantic",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download/save music-vibe-related Flickr30k test samples from Hugging Face."
    )

    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=10,
        help="Number of Flickr30k test samples to save. Default: 10.",
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


def normalize_captions(caption_field: Any) -> List[str]:
    if caption_field is None:
        return []

    if isinstance(caption_field, str):
        return [caption_field]

    if isinstance(caption_field, list):
        return [str(c) for c in caption_field]

    return [str(caption_field)]


def is_music_vibe_related(captions: List[str]) -> bool:
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


def download_flickr30k_samples(number: int):
    split = "test"

    print("Loading Flickr30k from Hugging Face...")

    ds = load_dataset(
        "nlphuji/flickr30k",
        split=split,
        cache_dir=hf_cache_dir,
    )

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
        for row in tqdm(ds):
            if saved_count >= number:
                break

            seen_count += 1

            image = row["image"]
            captions = normalize_captions(row["caption"])

            if not is_music_vibe_related(captions):
                skipped_non_music_vibe += 1
                continue

            filename = row.get("filename") or f"flickr30k_{row.get('img_id', seen_count)}.jpg"
            filename = safe_filename(filename)

            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                filename = f"{filename}.jpg"

            filename = os.path.splitext(filename)[0] + ".jpg"
            file_path = os.path.join(image_output_dir, filename)

            if os.path.exists(file_path):
                print(f"\n[{saved_count + 1}/{number}] Already exists: {file_path}")
                saved_count += 1
            else:
                print(f"\n[{saved_count + 1}/{number}] Saving Flickr30k image: {filename}")
                print(f"-> Captions: {captions[:2]}")

                success = save_image(image, file_path)

                if not success:
                    failed_count += 1
                    print(f"-> Failed: {filename}")
                    continue

                print(f"-> Saved to: {file_path}")
                saved_count += 1

            metadata = {
                "image_id": row.get("img_id", None),
                "filename": filename,
                "image_path": file_path,
                "captions": captions,
                "split": split,
                "source_dataset": "nlphuji/flickr30k",
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
    download_flickr30k_samples(number=args.number)