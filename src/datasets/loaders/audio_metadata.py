from pathlib import Path
from typing import Any, Dict, List
from src.utils.json import read_jsonl


class AudioMetadataLoader:
    """Load one normalized record per audio ID with merged captions."""

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
        by_audio_id: Dict[str, Dict[str, Any]] = {}

        for metadata_path in self.metadata_paths:
            if not metadata_path.exists():
                print(f"[WARN] Metadata file not found, skipping: {metadata_path}")
                continue

            for row in read_jsonl(metadata_path):
                record = self._normalize_row(row)
                if record is None:
                    continue

                audio_id = record["audio_id"]
                existing = by_audio_id.get(audio_id)
                if existing is None:
                    by_audio_id[audio_id] = record
                    continue

                self._validate_same_media(existing, record)
                existing["captions"] = self._merge_unique(
                    existing["captions"], record["captions"]
                )

        return list(by_audio_id.values())

    def _normalize_row(self, row: Dict[str, Any]) -> Dict[str, Any] | None:
        audio_id = row.get("audio_id")
        audio_path = row.get("audio_path")
        source_dataset = row.get("source_dataset")

        if not audio_id or not audio_path or not source_dataset:
            return None

        resolved_audio_path = self._resolve_path(str(audio_path))
        if self.require_audio_exists and not resolved_audio_path.exists():
            return None

        captions = self._clean_list(row.get("captions"))
        if not captions:
            return None

        return {
            "audio_id": str(audio_id),
            "captions": captions,
            "audio_path": str(resolved_audio_path),
            "source_dataset": str(source_dataset),
            "split": str(row.get("split", "")),
            "source_audio_id": str(row.get("source_audio_id", audio_id)),
        }

    @staticmethod
    def _validate_same_media(existing: Dict[str, Any], new: Dict[str, Any]) -> None:
        for field in ("audio_path", "source_dataset", "split"):
            if existing[field] != new[field]:
                raise ValueError(
                    f"Conflicting {field} for audio_id={existing['audio_id']}: "
                    f"{existing[field]!r} != {new[field]!r}"
                )

    def _resolve_path(self, path_value: str) -> Path:
        path = Path(path_value)
        if path.exists():
            return path.resolve()

        candidate = self.project_root / path_value
        if candidate.exists():
            return candidate.resolve()

        return path

    @staticmethod
    def _clean_list(value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            value = [value]
        return AudioMetadataLoader._merge_unique([], [str(v).strip() for v in value])

    @staticmethod
    def _merge_unique(left: List[str], right: List[str]) -> List[str]:
        seen = set(left)
        merged = list(left)
        for item in right:
            if item and item not in seen:
                seen.add(item)
                merged.append(item)
        return merged
