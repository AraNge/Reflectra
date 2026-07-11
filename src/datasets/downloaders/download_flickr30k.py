import os
import json
import argparse
import io
import tarfile
from pathlib import Path
from typing import Any, List
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

from src.datasets.paths import DATA_DIR, HF_CACHE_DIR, METADATA_DIR, ensure_data_dirs


ensure_data_dirs()

hf_cache_dir = str(HF_CACHE_DIR)
image_output_dir = str(DATA_DIR / "flickr30k_images")
metadata_output_path = str(METADATA_DIR / "flickr30k_metadata.jsonl")

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


def clean_caption(text: Any) -> str:
    return (
        str(text)
        .replace("<start>", "")
        .replace("<end>", "")
        .strip()
    )


def row_captions(row: dict[str, Any]) -> List[str]:
    return [
        clean_caption(caption)
        for caption in normalize_captions(
            row.get("caption")
            or row.get("captions")
            or row.get("sentences")
            or row.get("raw")
        )
        if clean_caption(caption)
    ]


def is_music_vibe_related(captions: List[str]) -> bool:
    if not captions:
        return False

    text = " ".join(captions).lower()
    return any(keyword in text for keyword in MUSIC_RELATED_KEYWORDS)


def coerce_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value

    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")

        if image_bytes:
            return Image.open(io.BytesIO(image_bytes))

        if image_path:
            return Image.open(image_path)

    return None


def row_image(row: dict[str, Any]) -> Any:
    for field in ("image", "jpg", "img"):
        if field in row:
            return row[field]

    return None


def load_flickr30k_split(split: str):
    try:
        return load_dataset(
            "nlphuji/flickr30k",
            split=split,
            cache_dir=hf_cache_dir,
        )
    except RuntimeError as exc:
        if "Dataset scripts are no longer supported" not in str(exc):
            raise

        print(
            "Dataset script loading is not supported by this datasets version; "
            "using train-ready Flickr30k fallback files."
        )
        return None


def save_image(image: Any, output_path: str) -> bool:
    try:
        image = coerce_image(image)
        if not isinstance(image, Image.Image):
            return False

        image = image.convert("RGB")
        image.save(output_path, format="JPEG", quality=95)

        return os.path.exists(output_path)

    except Exception as e:
        print(f"-> Failed to save image: {e}")
        return False


def download_flickr30k_trainready_samples(number: int, split: str) -> None:
    repo_id = "gondimjoaom/flickr30k-trainready"
    print(f"Loading fallback Flickr30k files from {repo_id}...")

    captions_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="dataset_flickr30k_allEN.json",
        cache_dir=hf_cache_dir,
    )
    images_tar_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename="flickr30k-images.tar.gz",
        cache_dir=hf_cache_dir,
    )

    with open(captions_path, "r", encoding="utf-8") as file:
        captions_by_filename = json.load(file)

    candidate_captions: dict[str, list[str]] = {}
    for filename, captions in captions_by_filename.items():
        normalized = [
            clean_caption(caption)
            for caption in normalize_captions(captions)
            if clean_caption(caption)
        ]
        if is_music_vibe_related(normalized):
            candidate_captions[str(filename)] = normalized

    print(f"Candidate music-vibe images: {len(candidate_captions)}")
    print(f"Requested music-vibe-related samples: {number}")
    print(f"Images will save to: {image_output_dir}")
    print(f"Metadata will save to: {metadata_output_path}")
    print("Extracting selected images from fallback tar...")

    saved_count = 0
    seen_count = 0
    failed_count = 0

    with (
        tarfile.open(images_tar_path, "r:gz") as tar,
        open(metadata_output_path, "w", encoding="utf-8") as meta_file,
    ):
        members = (member for member in tar if member.isfile())

        for member in tqdm(members):
            if saved_count >= number:
                break

            filename = Path(member.name).name
            captions = candidate_captions.get(filename)
            if not captions:
                continue

            seen_count += 1
            safe_name = os.path.splitext(safe_filename(filename))[0] + ".jpg"
            file_path = os.path.join(image_output_dir, safe_name)

            if os.path.exists(file_path):
                saved_count += 1
            else:
                extracted = tar.extractfile(member)
                if extracted is None:
                    failed_count += 1
                    continue

                try:
                    image = Image.open(extracted)
                    success = save_image(image, file_path)
                except Exception as exc:
                    print(f"-> Failed to decode {filename}: {exc}")
                    success = False

                if not success:
                    failed_count += 1
                    continue

                saved_count += 1

            metadata = {
                "image_id": os.path.splitext(safe_name)[0],
                "image_path": file_path,
                "captions": captions,
                "split": split,
                "source_dataset": "nlphuji/flickr30k",
                "source_image_id": os.path.splitext(filename)[0],
            }
            meta_file.write(json.dumps(metadata, ensure_ascii=False) + "\n")

    print("\nDownload summary:")
    print(f"Split: {split}")
    print(f"Candidate images seen in tar: {seen_count}")
    print(f"Saved samples: {saved_count}")
    print(f"Failed: {failed_count}")
    print(f"Output directory: {image_output_dir}")
    print(f"Metadata file: {metadata_output_path}")

    if saved_count < number:
        raise RuntimeError(
            f"Only saved {saved_count}/{number} Flickr30k fallback samples."
        )


def download_flickr30k_samples(number: int):
    split = "test"

    print("Loading Flickr30k from Hugging Face...")

    ds = load_flickr30k_split(split)
    if ds is None:
        download_flickr30k_trainready_samples(number=number, split=split)
        return

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

            image = row_image(row)
            captions = row_captions(row)

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
                "image_id": str(row.get("img_id") or os.path.splitext(filename)[0]),
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
