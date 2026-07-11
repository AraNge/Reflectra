from __future__ import annotations

from pathlib import Path
from typing import Any

from src.datasets.loaders.audio_metadata import AudioMetadataLoader
from src.datasets.loaders.image_metadata import ImageTextMetadataLoader
from src.datasets.paths import METADATA_DIR, PROJECT_ROOT


DEFAULT_IMAGE_METADATA_PATHS = [
    METADATA_DIR / "coco_captions_metadata.jsonl",
    METADATA_DIR / "flickr30k_metadata.jsonl",
    METADATA_DIR / "emoset_train_metadata.jsonl",
    METADATA_DIR / "emoset_test_metadata.jsonl",
]


DEFAULT_AUDIO_METADATA_PATHS = [
    METADATA_DIR / "musiccaps_metadata.jsonl",
    METADATA_DIR / "audioset_metadata.jsonl",
    METADATA_DIR / "audioset_balanced_metadata.jsonl",
    METADATA_DIR / "audioset_unbalanced_metadata.jsonl",
    METADATA_DIR / "song_describer_metadata.jsonl",
    METADATA_DIR / "mtg_jamendo_train_metadata.jsonl",
    METADATA_DIR / "mtg_jamendo_validation_metadata.jsonl",
]


def load_image_metadata(
    metadata_paths: list[str | Path] | None = None,
    project_root: str | Path = PROJECT_ROOT,
    require_image_exists: bool = True,
) -> list[dict[str, Any]]:
    records = ImageTextMetadataLoader(
        metadata_paths=metadata_paths or DEFAULT_IMAGE_METADATA_PATHS,
        project_root=project_root,
        require_image_exists=require_image_exists,
    ).load()

    if not records:
        raise ValueError(
            "ImageTextMetadataLoader returned no usable image records."
        )

    return records


def load_audio_metadata(
    metadata_paths: list[str | Path] | None = None,
    project_root: str | Path = PROJECT_ROOT,
    require_audio_exists: bool = True,
) -> list[dict[str, Any]]:
    records = AudioMetadataLoader(
        metadata_paths=metadata_paths or DEFAULT_AUDIO_METADATA_PATHS,
        project_root=project_root,
        require_audio_exists=require_audio_exists,
    ).load()

    if not records:
        raise ValueError(
            "AudioMetadataLoader returned no usable audio records."
        )

    return records
