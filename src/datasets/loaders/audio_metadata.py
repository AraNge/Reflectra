from pathlib import Path
from typing import Any, Dict, List

from src.datasets.loaders.jsonl import read_jsonl


class AudioMetadataLoader:
    """
    Loads audio-text metadata for CLAP training/evaluation.

    Common required fields:
    - audio_id
    - audio_path
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
        require_audio_exists: bool = True,
    ):
        self.metadata_paths = [Path(p) for p in metadata_paths]
        self.project_root = Path(project_root).resolve() if project_root else Path.cwd()
        self.require_audio_exists = require_audio_exists

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
        audio_id = row.get("audio_id")
        audio_path = row.get("audio_path")
        source_dataset = row.get("source_dataset")

        if not audio_id or not audio_path or not source_dataset:
            return []

        resolved_audio_path = self._resolve_path(audio_path)

        if self.require_audio_exists and not resolved_audio_path.exists():
            return []

        captions = self._extract_captions(row)

        if not captions:
            return []

        base_record = {
            "audio_id": str(audio_id),
            "audio_path": str(resolved_audio_path),
            "source_dataset": str(source_dataset),
            "split": str(row.get("split", "")),
        }

        return [
            {
                **base_record,
                "sample_id": f"{audio_id}:caption_{caption_idx}",
                "text": caption,
                "text_type": "caption",
            }
            for caption_idx, caption in enumerate(captions)
        ]

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)

        if path.exists():
            return path.resolve()

        candidate = self.project_root / path_value

        if candidate.exists():
            return candidate.resolve()

        return path

    def _extract_captions(self, row: Dict[str, Any]) -> List[str]:
        return self._clean_list(row.get("captions"))

    @staticmethod
    def _clean_list(value: Any) -> List[str]:
        if value is None:
            return []

        if not isinstance(value, list):
            value = [value]

        captions = []

        for item in value:
            text = str(item).strip()

            if text:
                captions.append(text)

        return captions
