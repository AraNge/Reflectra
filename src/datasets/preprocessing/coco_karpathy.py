import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


COCO_SOURCE_DATASET = "coco_karpathy"


def resolve_coco_image_path(image_root: Path, example: Dict[str, Any]) -> Path:
    filename = example.get("filename")
    filepath = example.get("filepath")

    if not filename:
        raise ValueError("COCO example is missing filename")

    if filepath:
        return image_root / str(filepath) / str(filename)

    return image_root / str(filename)


def normalize_coco_captions(sentences: Iterable[Dict[str, Any]]) -> List[str]:
    captions = []

    for sentence in sentences:
        raw = str(sentence.get("raw", "")).strip()

        if raw:
            captions.append(raw)

    return captions


class CocoKarpathyPreprocessor:
    """
    Converts Karpathy COCO JSON into the JSONL row shape consumed by ImageTextMetadataLoader.
    """

    def __init__(
        self,
        metadata_path: str | Path,
        image_root: str | Path,
        require_image_exists: bool = False,
        splits: Optional[set[str]] = None,
    ):
        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root)
        self.require_image_exists = require_image_exists
        self.splits = splits

    def load_rows(self) -> List[Dict[str, Any]]:
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        rows = []

        for example in data.get("images", []):
            row = self._normalize_example(example)

            if row is not None:
                rows.append(row)

        return rows

    def _normalize_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        split = str(example.get("split", "unknown"))

        if self.splits is not None and split not in self.splits:
            return None

        filename = example.get("filename")
        filepath = example.get("filepath")

        if not filename or not filepath:
            return None

        captions = normalize_coco_captions(example.get("sentences", []))

        if not captions:
            return None

        image_path = resolve_coco_image_path(self.image_root, example)

        if self.require_image_exists and not image_path.exists():
            return None

        image_id = example.get("cocoid") or example.get("imgid") or Path(filename).stem

        return {
            "image_id": str(image_id),
            "image_path": str(image_path),
            "captions": captions,
            "split": split,
            "source_dataset": COCO_SOURCE_DATASET,
        }


def write_coco_jsonl(rows: Iterable[Dict[str, Any]], output_path: str | Path) -> int:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1

    return count
