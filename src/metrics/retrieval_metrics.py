import numpy as np
from typing import Dict, List, Tuple


SparseRelevance = List[Dict[int, float]]


def compute_ranks(similarity_matrix: np.ndarray) -> List[int]:
    ranks = []

    for query_idx in range(similarity_matrix.shape[0]):
        scores = similarity_matrix[query_idx]
        sorted_indices = np.argsort(-scores)
        rank = int(np.where(sorted_indices == query_idx)[0][0]) + 1
        ranks.append(rank)

    return ranks


def retrieval_metrics(
    similarity_matrix: np.ndarray,
    recall_ks: List[int] | None = None,
) -> Dict[str, float]:
    if recall_ks is None:
        recall_ks = [1, 5, 10]

    ranks = compute_ranks(similarity_matrix)
    ranks_array = np.array(ranks)
    metrics = {}

    for k in recall_ks:
        metrics[f"recall@{k}"] = float(np.mean(ranks_array <= k))

    metrics["mrr"] = float(np.mean(1.0 / ranks_array))
    metrics["median_rank"] = float(np.median(ranks_array))
    metrics["mean_rank"] = float(np.mean(ranks_array))

    return metrics


def retrieval_benchmark_metrics(
    similarity_matrix: np.ndarray,
    recall_ks: List[int] | None = None,
) -> Dict[str, float]:
    """
    One-positive-pair retrieval metrics in the common audio retrieval benchmark format.

    Assumes query i matches target i. Recall and Geom are percentages, matching
    the R@K tables commonly used by audio-text retrieval benchmarks.
    """
    if recall_ks is None:
        recall_ks = [1, 5, 10, 50]

    ranks = compute_ranks(similarity_matrix)
    ranks_array = np.array(ranks)
    metrics = {}

    for k in recall_ks:
        metrics[f"R@{k}"] = float(np.mean(ranks_array <= k) * 100.0)

    metrics["MedR"] = float(np.median(ranks_array))
    metrics["MeanR"] = float(np.mean(ranks_array))
    metrics["MRR"] = float(np.mean(1.0 / ranks_array))

    if all(key in metrics for key in ["R@1", "R@5", "R@10"]):
        metrics["Geom"] = float(
            np.power(
                metrics["R@1"] * metrics["R@5"] * metrics["R@10"],
                1.0 / 3.0,
            )
        )

    return metrics


def recall_at_k(similarity: np.ndarray, relevance: np.ndarray, k: int) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx] > 0

        num_relevant = int(np.sum(query_relevance))

        if num_relevant == 0:
            continue

        ranked_indices = np.argsort(-query_scores)[:k]
        hits = int(np.sum(query_relevance[ranked_indices]))
        values.append(hits / num_relevant)

    if not values:
        return 0.0

    return float(np.mean(values))


def hit_rate_at_k(similarity: np.ndarray, relevance: np.ndarray, k: int) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx] > 0

        if not np.any(query_relevance):
            continue

        ranked_indices = np.argsort(-query_scores)[:k]
        values.append(float(np.any(query_relevance[ranked_indices])))

    if not values:
        return 0.0

    return float(np.mean(values))


def mean_rank(similarity: np.ndarray, relevance: np.ndarray) -> float:
    ranks = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx] > 0

        if not np.any(query_relevance):
            continue

        ranked_indices = np.argsort(-query_scores)
        relevant_ranks = np.where(query_relevance[ranked_indices])[0] + 1
        ranks.append(float(np.min(relevant_ranks)))

    if not ranks:
        return 0.0

    return float(np.mean(ranks))


def median_rank(similarity: np.ndarray, relevance: np.ndarray) -> float:
    ranks = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx] > 0

        if not np.any(query_relevance):
            continue

        ranked_indices = np.argsort(-query_scores)
        relevant_ranks = np.where(query_relevance[ranked_indices])[0] + 1
        ranks.append(float(np.min(relevant_ranks)))

    if not ranks:
        return 0.0

    return float(np.median(ranks))


def mean_reciprocal_rank(similarity: np.ndarray, relevance: np.ndarray) -> float:
    reciprocal_ranks = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx] > 0

        if not np.any(query_relevance):
            continue

        ranked_indices = np.argsort(-query_scores)
        relevant_ranks = np.where(query_relevance[ranked_indices])[0] + 1
        reciprocal_ranks.append(1.0 / float(np.min(relevant_ranks)))

    if not reciprocal_ranks:
        return 0.0

    return float(np.mean(reciprocal_ranks))


def binary_retrieval_metrics(
    similarity: np.ndarray,
    relevance: np.ndarray,
) -> Dict[str, float]:
    return {
        "hit@1": hit_rate_at_k(similarity, relevance, 1),
        "hit@5": hit_rate_at_k(similarity, relevance, 5),
        "hit@10": hit_rate_at_k(similarity, relevance, 10),
        "recall@1": recall_at_k(similarity, relevance, 1),
        "recall@5": recall_at_k(similarity, relevance, 5),
        "recall@10": recall_at_k(similarity, relevance, 10),
        "mrr": mean_reciprocal_rank(similarity, relevance),
        "median_rank": median_rank(similarity, relevance),
        "mean_rank": mean_rank(similarity, relevance),
    }


def transpose_sparse_relevance(
    relevance: SparseRelevance,
    num_targets: int,
) -> SparseRelevance:
    transposed: SparseRelevance = [
        {}
        for _ in range(num_targets)
    ]

    for query_idx, query_relevance in enumerate(relevance):
        for target_idx, score in query_relevance.items():
            previous_score = transposed[target_idx].get(query_idx, 0.0)
            transposed[target_idx][query_idx] = max(previous_score, float(score))

    return transposed


def sparse_recall_at_k(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    k: int,
) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx]

        if not query_relevance:
            continue

        ranked_indices = np.argsort(-query_scores)[:k]
        hits = sum(1 for idx in ranked_indices if int(idx) in query_relevance)
        values.append(hits / len(query_relevance))

    if not values:
        return 0.0

    return float(np.mean(values))


def sparse_hit_rate_at_k(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    k: int,
) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx]

        if not query_relevance:
            continue

        ranked_indices = np.argsort(-query_scores)[:k]
        values.append(float(any(int(idx) in query_relevance for idx in ranked_indices)))

    if not values:
        return 0.0

    return float(np.mean(values))


def sparse_first_relevant_ranks(
    similarity: np.ndarray,
    relevance: SparseRelevance,
) -> List[float]:
    ranks = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx]

        if not query_relevance:
            continue

        ranked_indices = np.argsort(-query_scores)

        for rank, target_idx in enumerate(ranked_indices, start=1):
            if int(target_idx) in query_relevance:
                ranks.append(float(rank))
                break

    return ranks


def sparse_binary_retrieval_metrics(
    similarity: np.ndarray,
    relevance: SparseRelevance,
) -> Dict[str, float]:
    ks = [1, 5, 10]

    hit_values = {k: [] for k in ks}
    recall_values = {k: [] for k in ks}
    first_ranks = []

    num_targets = similarity.shape[1]

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance_raw = relevance[query_idx]

        valid_relevant = {
            int(target_idx)
            for target_idx, score in query_relevance_raw.items()
            if 0 <= int(target_idx) < num_targets and float(score) > 0
        }

        if not valid_relevant:
            continue

        ranked_indices = np.argsort(-query_scores)
        ranked_indices = [int(idx) for idx in ranked_indices]

        first_rank = None
        for rank, target_idx in enumerate(ranked_indices, start=1):
            if target_idx in valid_relevant:
                first_rank = rank
                break

        if first_rank is None:
            continue

        first_ranks.append(first_rank)

        for k in ks:
            top_k = set(ranked_indices[:k])
            hits = len(top_k & valid_relevant)

            hit_values[k].append(float(hits > 0))
            recall_values[k].append(hits / len(valid_relevant))

    if not first_ranks:
        return {
            "hit@1": 0.0,
            "hit@5": 0.0,
            "hit@10": 0.0,
            "recall@1": 0.0,
            "recall@5": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0,
            "median_rank": 0.0,
            "mean_rank": 0.0,
            "num_evaluated_queries": 0,
        }

    ranks_array = np.array(first_ranks, dtype=np.float64)

    return {
        "hit@1": float(np.mean(hit_values[1])),
        "hit@5": float(np.mean(hit_values[5])),
        "hit@10": float(np.mean(hit_values[10])),
        "recall@1": float(np.mean(recall_values[1])),
        "recall@5": float(np.mean(recall_values[5])),
        "recall@10": float(np.mean(recall_values[10])),
        "mrr": float(np.mean(1.0 / ranks_array)),
        "median_rank": float(np.median(ranks_array)),
        "mean_rank": float(np.mean(ranks_array)),
        "num_evaluated_queries": int(len(first_ranks)),
    }


def dcg_at_k(relevance: np.ndarray, exponential_gain: bool = False) -> float:
    relevance = np.asarray(relevance, dtype=np.float32)

    if exponential_gain:
        gains = np.power(2.0, relevance) - 1.0
    else:
        gains = relevance

    ranks = np.arange(1, len(gains) + 1)
    discounts = np.log2(ranks + 1)

    return float(np.sum(gains / discounts))


def ndcg_at_k(
    similarity: np.ndarray,
    relevance: np.ndarray,
    k: int,
    exponential_gain: bool = False,
) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx]

        if np.max(query_relevance) <= 0:
            continue

        ranked_indices = np.argsort(-query_scores)[:k]
        ranked_relevance = query_relevance[ranked_indices]

        ideal_indices = np.argsort(-query_relevance)[:k]
        ideal_relevance = query_relevance[ideal_indices]

        dcg = dcg_at_k(ranked_relevance, exponential_gain=exponential_gain)
        idcg = dcg_at_k(ideal_relevance, exponential_gain=exponential_gain)

        if idcg <= 0:
            continue

        values.append(dcg / idcg)

    if not values:
        return 0.0

    return float(np.mean(values))

def compute_metrics(
    similarity: np.ndarray,
    relevance: np.ndarray,
    exponential_gain: bool = False,
) -> Dict[str, float]:
    return {
        "ndcg@1": ndcg_at_k(similarity, relevance, 1, exponential_gain),
        "ndcg@5": ndcg_at_k(similarity, relevance, 5, exponential_gain),
        "ndcg@10": ndcg_at_k(similarity, relevance, 10, exponential_gain),
    }


def sparse_ndcg_at_k(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    k: int,
    exponential_gain: bool = False,
) -> float:
    values = []

    for query_idx in range(similarity.shape[0]):
        query_scores = similarity[query_idx]
        query_relevance = relevance[query_idx]

        if not query_relevance:
            continue

        ranked_indices = np.argsort(-query_scores)[:k]
        ranked_relevance = np.array(
            [
                query_relevance.get(int(target_idx), 0.0)
                for target_idx in ranked_indices
            ],
            dtype=np.float32,
        )
        ideal_relevance = np.array(
            sorted(query_relevance.values(), reverse=True)[:k],
            dtype=np.float32,
        )

        dcg = dcg_at_k(ranked_relevance, exponential_gain=exponential_gain)
        idcg = dcg_at_k(ideal_relevance, exponential_gain=exponential_gain)

        if idcg <= 0:
            continue

        values.append(dcg / idcg)

    if not values:
        return 0.0

    return float(np.mean(values))


def sparse_compute_metrics(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    exponential_gain: bool = False,
) -> Dict[str, float]:
    return {
        "ndcg@1": sparse_ndcg_at_k(similarity, relevance, 1, exponential_gain),
        "ndcg@5": sparse_ndcg_at_k(similarity, relevance, 5, exponential_gain),
        "ndcg@10": sparse_ndcg_at_k(similarity, relevance, 10, exponential_gain),
    }


def validate_binary_metrics(metrics: Dict[str, float], name: str) -> None:
    hit1 = metrics["hit@1"]
    hit5 = metrics["hit@5"]
    hit10 = metrics["hit@10"]
    mrr = metrics["mrr"]
    median_rank = metrics["median_rank"]

    if not (hit1 <= hit5 <= hit10):
        raise RuntimeError(f"{name}: hit@k is not monotonic")

    if median_rank == 1.0 and hit1 < 0.5:
        raise RuntimeError(
            f"{name}: impossible metrics: median_rank=1 but hit@1={hit1}"
        )

    max_mrr = (
        hit1
        + (hit5 - hit1) / 2.0
        + (hit10 - hit5) / 6.0
        + (1.0 - hit10) / 11.0
    )

    if mrr > max_mrr + 1e-9:
        raise RuntimeError(
            f"{name}: impossible MRR={mrr}; max possible from hit@k is {max_mrr}"
        )


def balanced_edge_retrieval_metrics(
    similarity: np.ndarray,
    relevance: SparseRelevance,
    num_negatives: int = 999,
    recall_ks: Tuple[int, ...] = (1, 5, 10),
    seed: int = 0,
) -> Dict[str, float]:
    """
    Balanced one-positive retrieval evaluation.

    Each positive edge becomes one retrieval query:
        query -> 1 positive target + up to num_negatives negative targets

    This removes bias from different target-space sizes.
    """

    rng = np.random.default_rng(seed)

    num_queries, num_targets = similarity.shape

    ranks = []
    candidate_counts = []
    requested_candidates = int(num_negatives + 1)

    all_target_indices = np.arange(num_targets)

    for query_idx in range(num_queries):
        query_relevance = relevance[query_idx]

        if not query_relevance:
            continue

        positive_targets = [
            int(target_idx)
            for target_idx, score in query_relevance.items()
            if 0 <= int(target_idx) < num_targets and float(score) > 0
        ]

        if not positive_targets:
            continue

        positive_set = set(positive_targets)

        negative_pool = np.array(
            [
                target_idx
                for target_idx in all_target_indices
                if int(target_idx) not in positive_set
            ],
            dtype=np.int64,
        )

        actual_num_negatives = min(num_negatives, len(negative_pool))

        if actual_num_negatives <= 0:
            continue

        for positive_target in positive_targets:
            negatives = rng.choice(
                negative_pool,
                size=actual_num_negatives,
                replace=False,
            )

            candidate_indices = np.concatenate(
                [
                    np.array([positive_target], dtype=np.int64),
                    negatives,
                ]
            )

            candidate_scores = similarity[query_idx, candidate_indices]

            sorted_candidate_positions = np.argsort(-candidate_scores)
            ranked_candidate_indices = candidate_indices[sorted_candidate_positions]

            positive_rank = int(
                np.where(ranked_candidate_indices == positive_target)[0][0]
            ) + 1

            ranks.append(positive_rank)
            candidate_counts.append(int(actual_num_negatives + 1))

    if not ranks:
        result = {
            f"hit@{k}": 0.0
            for k in recall_ks
        }
        result.update(
            {
                "mrr": 0.0,
                "median_rank": 0.0,
                "mean_rank": 0.0,
                "num_eval_edges": 0,
                "requested_candidates_per_query": requested_candidates,
                "mean_candidates_per_query": 0.0,
                "min_candidates_per_query": 0,
                "max_candidates_per_query": 0,
            }
        )
        return result

    ranks_array = np.asarray(ranks, dtype=np.float64)
    candidate_counts_array = np.asarray(candidate_counts, dtype=np.float64)

    result = {}

    for k in recall_ks:
        result[f"hit@{k}"] = float(np.mean(ranks_array <= k))

    result["mrr"] = float(np.mean(1.0 / ranks_array))
    result["median_rank"] = float(np.median(ranks_array))
    result["mean_rank"] = float(np.mean(ranks_array))
    result["num_eval_edges"] = int(len(ranks))
    result["requested_candidates_per_query"] = requested_candidates
    result["mean_candidates_per_query"] = float(np.mean(candidate_counts_array))
    result["min_candidates_per_query"] = int(np.min(candidate_counts_array))
    result["max_candidates_per_query"] = int(np.max(candidate_counts_array))

    return result
