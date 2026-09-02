import numpy as np


def recall_at_k(
    exact_ids: np.ndarray,
    approximate_ids: np.ndarray,
) -> float:
    """
    Compute Recall@K between exact and approximate search results.

    Recall@K measures the fraction of true nearest neighbors
    recovered by the approximate search.

    Parameters
    ----------
    exact_ids : np.ndarray
        IDs returned by exact search.

    approximate_ids : np.ndarray
        IDs returned by approximate search.

    Returns
    -------
    float
        Recall@K in the range [0.0, 1.0].
    """

    exact_ids = np.asarray(exact_ids)
    approximate_ids = np.asarray(approximate_ids)

    if exact_ids.ndim != 1:
        raise ValueError("exact_ids must be a 1D array")

    if approximate_ids.ndim != 1:
        raise ValueError("approximate_ids must be a 1D array")

    if len(exact_ids) == 0:
        raise ValueError("exact_ids cannot be empty")

    if len(approximate_ids) == 0:
        raise ValueError("approximate_ids cannot be empty")

    exact_set = set(exact_ids.tolist())
    approximate_set = set(approximate_ids.tolist())

    matches = len(exact_set.intersection(approximate_set))

    return matches / len(exact_set)