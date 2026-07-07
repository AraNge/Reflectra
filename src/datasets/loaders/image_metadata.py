from pathlib import Path
from typing import Any, Dict, List
from src.datasets.loaders.jsonl import read_jsonl


class ImageTextMetadataLoader:
    """
    Loads image-text metadata for CLIP training/evaluation.

    Common required fields:
    - image_id
    - image_path
    - source_dataset

    Optional:
    - split

    Text fields:
    - captions
    """

    def __init__(
        self,
        metadata_paths: List[str | Path],
        project_root: str | Path | None = None,
        require_image_exists: bool = True,
    ):
        self.metadata_paths = [Path(p) for p in metadata_paths]
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.require_image_exists = require_image_exists

    def load(self) -> List[Dict[str, Any]]:
        records = []

        for metadata_path in self.metadata_paths:
            if not metadata_path.exists():
                print(f"[WARN] Metadata file not found, skipping: {metadata_path}")
                continue

            rows = read_jsonl(metadata_path)

            for row in rows:
                records.extend(self._normalize_row(row))

        return records

    def _normalize_row(self, row: Dict[str, Any]) -> List[Dict[str, Any]]:
        image_id = row.get("image_id")
        image_path = row.get("image_path")
        source_dataset = row.get("source_dataset")

        if not image_id or not image_path or not source_dataset:
            return []

        resolved_image_path = self._resolve_path(image_path)

        if self.require_image_exists and not resolved_image_path.exists():
            return []

        base_record = {
            "image_id": str(image_id),
            "image_path": str(resolved_image_path),
            "source_dataset": str(source_dataset),
            "split": str(row.get("split", "unknown")),
        }

        return self._records_from_captions(
            row=row,
            base_record=base_record,
        )

    def _records_from_captions(
        self,
        row: Dict[str, Any],
        base_record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        captions = row.get("captions")

        if not captions:
            return []

        if isinstance(captions, str):
            captions = [captions]

        captions = [str(c).strip() for c in captions if str(c).strip()]

        if not captions:
            return []

        image_id = base_record["image_id"]

        return [
            {
                **base_record,
                "sample_id": f"{image_id}:caption_{idx}",
                "text": caption,
                "text_type": "caption",
            }
            for idx, caption in enumerate(captions)
        ]

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)

        if path.exists():
            return path.resolve()

        candidate = self.project_root / path_value

        if candidate.exists():
            return candidate.resolve()

        return path
