from typing import Dict, List

import numpy as np


def compute_ranks(similarity_matrix: np.ndarray) -> List[int]:
    """
    similarity_matrix shape: [num_queries, num_targets]

    Assumes correct target for query i is target i.
    Returns 1-based rank for each query.
    """

    ranks = []

    for i in range(similarity_matrix.shape[0]):
        scores = similarity_matrix[i]

        sorted_indices = np.argsort(-scores)

        rank = int(np.where(sorted_indices == i)[0][0]) + 1
        ranks.append(rank)

    return ranks


def retrieval_metrics(
    similarity_matrix: np.ndarray,
    recall_ks: List[int] = [1, 5, 10],
) -> Dict[str, float]:
    ranks = compute_ranks(similarity_matrix)
    ranks_array = np.array(ranks)

    metrics = {}

    for k in recall_ks:
        metrics[f"recall@{k}"] = float(np.mean(ranks_array <= k))

    metrics["mrr"] = float(np.mean(1.0 / ranks_array))
    metrics["median_rank"] = float(np.median(ranks_array))
    metrics["mean_rank"] = float(np.mean(ranks_array))

    return metrics