from __future__ import annotations

from typing import Dict, List

import numpy as np


SparseRelevance = List[Dict[int, float]]


def _ranked_indices(scores: np.ndarray) -> list[int]:
    return [int(index) for index in np.argsort(-scores)]


def _relevant_set(
    query_relevance: dict[int, float],
    threshold: float,
) -> set[int]:
    return {
        int(target_idx)
        for target_idx, score in query_relevance.items()
        if float(score) > threshold
    }


def transpose_sparse_relevance(
    relevance: SparseRelevance,
    num_targets: int,
) -> SparseRelevance:
    transposed: SparseRelevance = [{} for _ in range(num_targets)]

    for query_idx, query_relevance in enumerate(relevance):
        for target_idx, score in query_relevance.items():
            target_idx = int(target_idx)
            previous_score = transposed[target_idx].get(query_idx, 0.0)
            transposed[target_idx][query_idx] = max(previous_score, float(score))

    return transposed


def dcg(relevance: list[float], exponential_gain: bool = False) -> float:
    values = np.asarray(relevance, dtype=np.float64)
    gains = np.power(2.0, values) - 1.0 if exponential_gain else values
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def sparse_ndcg_at_k(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    k: int,
    exponential_gain: bool = False,
) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        query_relevance = relevance[query_idx]
        if not query_relevance:
            continue

        ranked = _ranked_indices(similarity[query_idx])[:k]
        ranked_relevance = [
            float(query_relevance.get(target_idx, 0.0))
            for target_idx in ranked
        ]
        ideal_relevance = sorted(
            (float(score) for score in query_relevance.values()),
            reverse=True,
        )[:k]

        ideal_dcg = dcg(ideal_relevance, exponential_gain=exponential_gain)
        if ideal_dcg <= 0:
            continue

        values.append(
            dcg(ranked_relevance, exponential_gain=exponential_gain) / ideal_dcg
        )

    return float(np.mean(values)) if values else 0.0


def sparse_mrr(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    threshold: float = 0.0,
) -> float:
    values = []
    for query_idx in range(similarity.shape[0]):
        relevant = _relevant_set(relevance[query_idx], threshold)
        if not relevant:
            continue

        for rank, target_idx in enumerate(
            _ranked_indices(similarity[query_idx]),
            start=1,
        ):
            if target_idx in relevant:
                values.append(1.0 / rank)
                break

    return float(np.mean(values)) if values else 0.0


def sparse_average_precision(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    threshold: float = 0.0,
) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        relevant = _relevant_set(relevance[query_idx], threshold)
        if not relevant:
            continue

        hits = 0
        precision_sum = 0.0

        for rank, target_idx in enumerate(
            _ranked_indices(similarity[query_idx]),
            start=1,
        ):
            if target_idx not in relevant:
                continue

            hits += 1
            precision_sum += hits / rank

        values.append(precision_sum / len(relevant))

    return float(np.mean(values)) if values else 0.0


def sparse_recall_at_k(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    k: int,
    threshold: float = 0.0,
) -> float:
    values = []
    for query_idx in range(similarity.shape[0]):
        relevant = _relevant_set(relevance[query_idx], threshold)
        if not relevant:
            continue

        top_k = set(_ranked_indices(similarity[query_idx])[:k])
        values.append(len(top_k & relevant) / len(relevant))

    return float(np.mean(values)) if values else 0.0


def sparse_precision_at_k(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    k: int,
    threshold: float = 0.0,
) -> float:
    values = []
    for query_idx in range(similarity.shape[0]):
        relevant = _relevant_set(relevance[query_idx], threshold)
        if not relevant:
            continue

        ranked = _ranked_indices(similarity[query_idx])[:k]
        if not ranked:
            continue

        values.append(len(set(ranked) & relevant) / len(ranked))

    return float(np.mean(values)) if values else 0.0


def sparse_retrieval_metrics(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    ks: tuple[int, ...] = (1, 5, 10),
    threshold: float = 0.0,
    exponential_gain: bool = False,
) -> dict[str, float]:
    metrics = {
        "mrr": sparse_mrr(
            similarity=similarity,
            relevance=relevance,
            threshold=threshold,
        ),
        "mAP": sparse_average_precision(
            similarity=similarity,
            relevance=relevance,
            threshold=threshold,
        ),
        "num_queries": sum(1 for row in relevance if row),
    }

    for k in ks:
        metrics[f"ndcg@{k}"] = sparse_ndcg_at_k(
            similarity=similarity,
            relevance=relevance,
            k=k,
            exponential_gain=exponential_gain,
        )
        metrics[f"recall@{k}"] = sparse_recall_at_k(
            similarity=similarity,
            relevance=relevance,
            k=k,
            threshold=threshold,
        )
        metrics[f"precision@{k}"] = sparse_precision_at_k(
            similarity=similarity,
            relevance=relevance,
            k=k,
            threshold=threshold,
        )

    return metrics
