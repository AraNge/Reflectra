from __future__ import annotations

import random
from typing import Any

from src.utils.hashing import stable_hash_id


def filter_by_dataset(
    records: list[dict[str, Any]],
    dataset_name: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("source_dataset") == dataset_name
    ]


def filter_by_split(
    records: list[dict[str, Any]],
    split: str,
) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("split") == split
    ]


def deterministic_sample(
    records: list[dict[str, Any]],
    count: int,
    seed: int,
    id_field: str,
) -> list[dict[str, Any]]:
    """
    Select the same records on every machine without depending on input order.
    """
    ranked = sorted(
        records,
        key=lambda record: (
            stable_hash_id(
                "sample",
                seed,
                record[id_field],
                length=64,
            ),
            record[id_field],
        ),
    )
    return ranked[: min(count, len(ranked))]


def sample_fraction(
    records: list[dict[str, Any]],
    fraction: float,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if fraction <= 0:
        return []

    if fraction >= 1:
        return records

    rng = random.Random(seed)
    sample_size = int(len(records) * fraction)

    return rng.sample(records, sample_size)


def sample_n(
    records: list[dict[str, Any]],
    n: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if n <= 0:
        return []

    if n >= len(records):
        return records

    rng = random.Random(seed)
    return rng.sample(records, n)


def sample_by_dataset_fractions(
    records: list[dict[str, Any]],
    fractions: dict[str, float],
    seed: int = 42,
) -> list[dict[str, Any]]:
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
    records: list[dict[str, Any]],
    counts: dict[str, int],
    seed: int = 42,
) -> list[dict[str, Any]]:
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
    records: list[dict[str, Any]],
    max_samples: int | None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if max_samples is None:
        return records

    return sample_n(records, max_samples, seed=seed)
