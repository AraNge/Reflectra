from pathlib import Path
from typing import Any, Dict, List, Optional
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
    - captions       # COCO / Flickr30k
    - music_query    # EmoSet-derived music/music-mood query
    """

    def __init__(
        self,
        metadata_paths: List[str | Path],
        project_root: str | Path | None = None,
        require_image_exists: bool = True,
        expand_captions: bool = True,
    ):
        self.metadata_paths = [Path(p) for p in metadata_paths]
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.require_image_exists = require_image_exists
        self.expand_captions = expand_captions

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
            "raw": row,
        }

        records = []

        records.extend(
            self._records_from_captions(
                row=row,
                base_record=base_record,
            )
        )

        music_query_record = self._record_from_single_text(
            row=row,
            base_record=base_record,
            field_name="music_query",
            text_type="music_query",
        )

        if music_query_record is not None:
            records.append(music_query_record)

        return records

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

        if self.expand_captions:
            return [
                {
                    **base_record,
                    "sample_id": f"{image_id}:caption_{idx}",
                    "text": caption,
                    "text_type": "caption",
                }
                for idx, caption in enumerate(captions)
            ]

        merged_caption = " ".join(captions)

        return [
            {
                **base_record,
                "sample_id": f"{image_id}:captions",
                "text": merged_caption,
                "text_type": "captions",
            }
        ]

    def _record_from_single_text(
        self,
        row: Dict[str, Any],
        base_record: Dict[str, Any],
        field_name: str,
        text_type: str,
    ) -> Optional[Dict[str, Any]]:
        text = self._clean_text(row.get(field_name))

        if not text:
            return None

        image_id = base_record["image_id"]

        return {
            **base_record,
            "sample_id": f"{image_id}:{text_type}",
            "text": text,
            "text_type": text_type,
        }

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)

        if path.exists():
            return path.resolve()

        candidate = self.project_root / path_value

        if candidate.exists():
            return candidate.resolve()

        return path

    @staticmethod
    def _clean_text(value: Any) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, list):
            value = " ".join(str(v) for v in value)

        value = str(value).strip()

        if not value:
            return None

        return value