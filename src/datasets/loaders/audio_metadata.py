from pathlib import Path
from typing import Any, Dict, List, Optional

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

    Text fields priority:
    1. caption
    2. human_labels
    3. moods
    4. aspect_list
    5. genres + instruments
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
                record = self._normalize_row(row)

                if record is not None:
                    records.append(record)

        return records

    def _normalize_row(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        audio_id = row.get("audio_id")
        audio_path = row.get("audio_path")
        source_dataset = row.get("source_dataset")

        if not audio_id or not audio_path or not source_dataset:
            return None

        resolved_audio_path = self._resolve_path(audio_path)

        if self.require_audio_exists and not resolved_audio_path.exists():
            return None

        text = self._extract_text(row)

        if not text:
            return None

        return {
            "audio_id": str(audio_id),
            "audio_path": str(resolved_audio_path),
            "text": text,
            "source_dataset": str(source_dataset),
            "split": str(row.get("split", "")),
            "raw": row,
        }

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)

        if path.exists():
            return path.resolve()

        candidate = self.project_root / path_value

        if candidate.exists():
            return candidate.resolve()

        return path

    def _extract_text(self, row: Dict[str, Any]) -> Optional[str]:
        # 1. caption
        caption = self._clean_text(row.get("caption"))
        if caption:
            return caption

        # 2. human_labels
        human_labels = row.get("human_labels")
        if human_labels:
            return f"An audio clip with human labels: {self._list_to_text(human_labels)}."

        # 3. moods
        moods = row.get("moods")
        if moods:
            return f"A music track with moods: {self._list_to_text(moods)}."

        # 4. aspect_list
        aspect_list = row.get("aspect_list")
        if aspect_list:
            return f"A music clip described by aspects: {self._list_to_text(aspect_list)}."

        # 5. genres + instruments
        genres = row.get("genres")
        instruments = row.get("instruments")

        if genres or instruments:
            parts = []

            if genres:
                parts.append(f"genres: {self._list_to_text(genres)}")

            if instruments:
                parts.append(f"instruments: {self._list_to_text(instruments)}")

            return "A music track with " + "; ".join(parts) + "."

        return None

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

    @staticmethod
    def _list_to_text(value: Any) -> str:
        if isinstance(value, list):
            return ", ".join(str(v) for v in value)

        return str(value)