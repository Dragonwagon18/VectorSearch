from __future__ import annotations

import numpy as np


def batch_search(
    vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact batch nearest-neighbor search.

    Parameters
    ----------
    vectors:
        Database vectors with shape (N, D).

    queries:
        Query vectors with shape (B, D).

    ids:
        Item IDs with shape (N,).

    k:
        Number of nearest neighbors.

    Returns
    -------
    result_ids:
        Nearest-neighbor IDs with shape (B, k).

    result_distances:
        Squared L2 distances with shape (B, k).
    """

    vectors = np.asarray(vectors, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    ids = np.asarray(ids)

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")

    if queries.ndim != 2:
        raise ValueError("queries must be a 2D array")

    if ids.ndim != 1:
        raise ValueError("ids must be a 1D array")

    if vectors.shape[1] != queries.shape[1]:
        raise ValueError("query dimension does not match vector dimension")

    if len(ids) != len(vectors):
        raise ValueError("number of ids must match number of vectors")

    if len(vectors) == 0:
        raise ValueError("cannot search an empty index")

    if not 1 <= k <= len(vectors):
        raise ValueError("k must satisfy 1 <= k <= number of vectors")

    if not np.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values")

    if not np.isfinite(queries).all():
        raise ValueError("queries must contain only finite values")

    # ---------------------------------------------------------
    # Efficient exact squared-L2 distance
    #
    # ||q - x||²
    # = ||q||² + ||x||² - 2q·x
    #
    # queries: (B, D)
    # vectors: (N, D)
    #
    # queries @ vectors.T:
    #            (B, N)
    # ---------------------------------------------------------

    query_norms = np.sum(
        queries * queries,
        axis=1,
        keepdims=True,
    )

    vector_norms = np.sum(
        vectors * vectors,
        axis=1,
        keepdims=True,
    ).T

    distances = (
        query_norms
        + vector_norms
        - 2.0 * (queries @ vectors.T)
    )

    # Floating-point arithmetic can occasionally produce
    # tiny negative values such as -1e-7 for distances that
    # mathematically should be zero.
    distances = np.maximum(distances, 0.0)

    # ---------------------------------------------------------
    # Partial selection
    # ---------------------------------------------------------

    candidate_indices = np.argpartition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    candidate_distances = np.take_along_axis(
        distances,
        candidate_indices,
        axis=1,
    )

    candidate_ids = ids[candidate_indices]

    # ---------------------------------------------------------
    # Deterministic ordering:
    #
    #     (distance, item_id)
    #
    # We only sort the k selected candidates.
    # ---------------------------------------------------------

    result_ids = np.empty_like(candidate_ids)
    result_distances = np.empty_like(candidate_distances)

    for i in range(len(queries)):
        order = np.lexsort(
            (
                candidate_ids[i],
                candidate_distances[i],
            )
        )

        result_ids[i] = candidate_ids[i][order]
        result_distances[i] = candidate_distances[i][order]

    return result_ids, result_distances