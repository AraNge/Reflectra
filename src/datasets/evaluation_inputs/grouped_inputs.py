from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.metrics.retrieval_metrics import SparseRelevance


def make_record_key(record: Dict[str, Any], id_field: str, path_field: str) -> Tuple[str, str]:
    source_dataset = str(record.get("source_dataset", "unknown"))
    item_id = record.get(id_field) or record.get(path_field)

    return source_dataset, str(item_id)


def make_text_key(record: Dict[str, Any], fallback_idx: int) -> Tuple[str, str]:
    source_dataset = str(record.get("source_dataset", "unknown"))
    sample_id = record.get("sample_id")

    if sample_id is None:
        sample_id = f"{record.get('text', '')}:{fallback_idx}"

    return source_dataset, str(sample_id)


def build_sparse_grouped_retrieval_inputs(
    records: List[Dict[str, Any]],
    media_id_field: str,
    media_path_field: str,
) -> tuple[List[str], List[str], SparseRelevance, Dict[str, int]]:
    """
    Build unique media targets, unique text targets, and sparse binary relevance.

    Every text record belonging to the same media key is a positive caption/query for
    that media item. No dense zero-filled media x text relevance matrix is created.
    """

    media_paths: List[str] = []
    texts: List[str] = []
    media_index_by_key: Dict[Tuple[str, str], int] = {}
    text_index_by_key: Dict[Tuple[str, str], int] = {}
    media_to_text_indices: Dict[int, set[int]] = defaultdict(set)

    for record_idx, record in enumerate(records):
        text = str(record.get("text", "")).strip()
        media_path = record.get(media_path_field)

        if not text or not media_path:
            continue

        media_key = make_record_key(record, media_id_field, media_path_field)
        text_key = make_text_key(record, record_idx)

        if media_key not in media_index_by_key:
            media_index_by_key[media_key] = len(media_paths)
            media_paths.append(str(Path(media_path)))

        if text_key not in text_index_by_key:
            text_index_by_key[text_key] = len(texts)
            texts.append(text)

        media_idx = media_index_by_key[media_key]
        text_idx = text_index_by_key[text_key]
        media_to_text_indices[media_idx].add(text_idx)

    relevance: SparseRelevance = [
        {
            int(text_idx): 1.0
            for text_idx in sorted(media_to_text_indices.get(media_idx, set()))
        }
        for media_idx in range(len(media_paths))
    ]

    stats = {
        "num_records": len(records),
        "num_media": len(media_paths),
        "num_texts": len(texts),
        "num_positive_edges": sum(len(row) for row in relevance),
    }

    return media_paths, texts, relevance, stats
