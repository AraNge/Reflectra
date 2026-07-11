import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def find_image_file(image_root: Path, image_record: Dict[str, Any]) -> Path | None:
    filename = image_record.get("filename")
    filepath = image_record.get("filepath")

    if not filename:
        return None

    filename = str(filename)
    candidates: List[Path] = []

    # Supports both --image-root /path/to/val2014 and --image-root /path/to/coco_images.
    candidates.append(image_root / filename)

    if filepath:
        filepath = str(filepath)
        candidates.append(image_root / filepath / filename)
        candidates.append(image_root.parent / filepath / filename)

        if image_root.name == filepath:
            candidates.append(image_root / filename)

    seen_candidates = set()

    for candidate in candidates:
        candidate_key = str(candidate)

        if candidate_key in seen_candidates:
            continue

        seen_candidates.add(candidate_key)

        if candidate.exists():
            return candidate.resolve()

    return None


def load_cxc_sits_retrieval_data(
    metadata_path: Path,
    image_root: Path,
    max_images: int | None = None,
) -> Tuple[List[str], List[str], List[Dict[int, float]]]:
    """
    Builds image paths, caption targets, and sparse graded relevance from merged CxC SITS JSON.

    Returns:
        image_paths: unique image paths
        captions: unique caption targets
        relevance: one dict per image query, mapping caption index -> CxC score
    """

    with open(metadata_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    images = data["images"]

    image_paths: List[str] = []
    captions: List[str] = []

    image_index_by_filename: Dict[str, int] = {}
    caption_index_by_sentence_id: Dict[int, int] = {}

    selected_records: List[Dict[str, Any]] = []

    missing_count = 0
    no_cxc_count = 0

    for image_record in images:
        if "cxc_scores" not in image_record:
            no_cxc_count += 1
            continue

        image_path = find_image_file(image_root, image_record)

        if image_path is None:
            missing_count += 1
            continue

        filename = str(image_record["filename"])

        if filename in image_index_by_filename:
            continue

        image_index_by_filename[filename] = len(image_paths)
        image_paths.append(str(image_path))
        selected_records.append(image_record)

        for sentence in image_record.get("sentences", []):
            sentence_id = sentence.get("sentid")
            caption = sentence.get("raw")

            if sentence_id is None or not caption:
                continue

            sentence_id = int(sentence_id)

            if sentence_id not in caption_index_by_sentence_id:
                caption_index_by_sentence_id[sentence_id] = len(captions)
                captions.append(str(caption).strip())

        if max_images is not None and len(image_paths) >= max_images:
            break

    relevance: List[Dict[int, float]] = [{} for _ in image_paths]

    for image_record in selected_records:
        filename = str(image_record["filename"])
        image_idx = image_index_by_filename[filename]

        for item in image_record.get("cxc_scores", []):
            if len(item) != 3:
                continue

            target_id, score, _rating_type = item

            try:
                sentence_id = int(target_id)
                score = float(score)
            except Exception:
                continue

            if sentence_id not in caption_index_by_sentence_id:
                continue

            caption_idx = caption_index_by_sentence_id[sentence_id]
            previous_score = relevance[image_idx].get(caption_idx, 0.0)
            relevance[image_idx][caption_idx] = max(previous_score, score)

    if len(image_paths) == 0:
        print(f"Debug: image_root = {image_root}")
        print(f"Debug: image_root exists = {image_root.exists()}")
        print(f"Debug: examples skipped because no cxc_scores = {no_cxc_count}")
        print(f"Debug: examples skipped because image file missing = {missing_count}")

        for image_record in images[:5]:
            filename = image_record.get("filename")
            filepath = image_record.get("filepath")
            print(
                "Debug example:",
                {
                    "filename": filename,
                    "filepath": filepath,
                    "direct_path": str(image_root / str(filename)) if filename else None,
                    "direct_exists": (image_root / str(filename)).exists() if filename else None,
                },
            )

    return image_paths, captions, relevance
