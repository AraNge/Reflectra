import random
from typing import Any, Dict, List, Optional


def filter_by_dataset(
    records: List[Dict[str, Any]],
    dataset_name: str,
) -> List[Dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("source_dataset") == dataset_name
    ]


def filter_by_split(
    records: List[Dict[str, Any]],
    split: str,
) -> List[Dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("split") == split
    ]


def sample_fraction(
    records: List[Dict[str, Any]],
    fraction: float,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if fraction <= 0:
        return []

    if fraction >= 1:
        return records

    rng = random.Random(seed)
    sample_size = int(len(records) * fraction)

    return rng.sample(records, sample_size)


def sample_n(
    records: List[Dict[str, Any]],
    n: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if n <= 0:
        return []

    if n >= len(records):
        return records

    rng = random.Random(seed)
    return rng.sample(records, n)


def sample_by_dataset_fractions(
    records: List[Dict[str, Any]],
    fractions: Dict[str, float],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Example:
        fractions = {
            "coco_karpathy": 0.5,
            "nlphuji/flickr30k": 0.8,
            "LiangJian24/EmoSet": 1.0,
        }
    """

    sampled = []

    for dataset_name, fraction in fractions.items():
        dataset_records = filter_by_dataset(records, dataset_name)

        sampled.extend(
            sample_fraction(
                records=dataset_records,
                fraction=fraction,
                seed=seed,
            )
        )

    return sampled


def sample_by_dataset_counts(
    records: List[Dict[str, Any]],
    counts: Dict[str, int],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Example:
        counts = {
            "coco_karpathy": 50000,
            "nlphuji/flickr30k": 10000,
        }
    """

    sampled = []

    for dataset_name, count in counts.items():
        dataset_records = filter_by_dataset(records, dataset_name)

        sampled.extend(
            sample_n(
                records=dataset_records,
                n=count,
                seed=seed,
            )
        )

    return sampled


def limit_total(
    records: List[Dict[str, Any]],
    max_samples: Optional[int],
    seed: int = 42,
) -> List[Dict[str, Any]]:
    if max_samples is None:
        return records

    return sample_n(records, max_samples, seed=seed)
