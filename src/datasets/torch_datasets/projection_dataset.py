from pathlib import Path
from typing import Any, Dict, List, Optional

from torch.utils.data import Dataset, DataLoader

from src.datasets.loaders.image_metadata import ImageTextMetadataLoader
from src.datasets.preprocessing.sampling import (
    sample_by_dataset_fractions,
    sample_by_dataset_counts,
    limit_total,
)
from src.datasets.paths import METADATA_DIR, PROJECT_ROOT


DEFAULT_IMAGE_METADATA_PATHS = [
    METADATA_DIR / "coco_captions_metadata.jsonl",
    METADATA_DIR / "flickr30k_metadata.jsonl",
    METADATA_DIR / "emoset_train_metadata.jsonl",
    METADATA_DIR / "emoset_test_metadata.jsonl",
]


class ImageTextProjectionDataset(Dataset):
    """
    Dataset for training image -> CLAP projection.

    Each item contains:
    - image_path
    - text
    - image_id
    - source_dataset
    - split
    - text_type
    """

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        record = self.records[idx]
        return {
            "image_path": record["image_path"],
            "text": record["text"],
            "image_id": record["image_id"],
            "source_dataset": record["source_dataset"],
            "split": record["split"],
            "text_type": record["text_type"],
        }


def collate_projection_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "image_paths": [item["image_path"] for item in batch],
        "texts": [item["text"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "source_datasets": [item["source_dataset"] for item in batch],
        "splits": [item["split"] for item in batch],
        "text_types": [item["text_type"] for item in batch],
    }


def parse_dataset_fractions(value: Optional[str]) -> Dict[str, float]:
    """
    Example:
        "coco_karpathy=0.5,nlphuji/flickr30k=0.8"
    """

    if not value:
        return {}

    result = {}

    for item in value.split(","):
        dataset_name, fraction = item.split("=")
        result[dataset_name.strip()] = float(fraction)

    return result


def parse_dataset_counts(value: Optional[str]) -> Dict[str, int]:
    """
    Example:
        "coco_karpathy=50000,nlphuji/flickr30k=10000"
    """

    if not value:
        return {}

    result = {}

    for item in value.split(","):
        dataset_name, count = item.split("=")
        result[dataset_name.strip()] = int(count)

    return result


def load_projection_records(
    metadata_paths: Optional[List[str | Path]] = None,
    project_root: str | Path = PROJECT_ROOT,
    train_split: str = "train",
    val_split: Optional[str] = None,
    dataset_fractions: Optional[str] = None,
    dataset_counts: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    require_image_exists: bool = True,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads image-text records and splits them into train/val records.
    """
    if metadata_paths is None:
        metadata_paths = DEFAULT_IMAGE_METADATA_PATHS

    loader = ImageTextMetadataLoader(
        metadata_paths=metadata_paths,
        project_root=project_root,
        require_image_exists=require_image_exists,
    )

    records = loader.load()

    parsed_fractions = parse_dataset_fractions(dataset_fractions)
    parsed_counts = parse_dataset_counts(dataset_counts)

    if parsed_fractions:
        records = sample_by_dataset_fractions(
            records=records,
            fractions=parsed_fractions,
        )

    if parsed_counts:
        records = sample_by_dataset_counts(
            records=records,
            counts=parsed_counts,
        )

    train_records = [
        record for record in records
        if record["split"] == train_split
    ]

    if val_split is not None:
        val_records = [
            record for record in records
            if record["split"] == val_split
        ]
    else:
        val_records = []

    train_records = limit_total(train_records, max_train_samples)
    val_records = limit_total(val_records, max_val_samples)

    return train_records, val_records


def create_projection_dataloaders(
    train_records: List[Dict[str, Any]],
    val_records: Optional[List[Dict[str, Any]]] = None,
    batch_size: int = 32,
    num_workers: int = 0,
    drop_last_train: bool = True,
) -> tuple[DataLoader, Optional[DataLoader]]:
    """
    Creates train and validation DataLoaders.
    """

    train_dataset = ImageTextProjectionDataset(train_records)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_projection_batch,
        drop_last=drop_last_train,
    )

    val_loader = None

    if val_records:
        val_dataset = ImageTextProjectionDataset(val_records)

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_projection_batch,
            drop_last=False,
        )

    return train_loader, val_loader
