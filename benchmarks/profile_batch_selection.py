from __future__ import annotations

import time

import numpy as np


def main() -> None:
    rng = np.random.default_rng(42)

    n = 100_000
    d = 128
    queries = 256
    k = 10

    vectors = rng.random((n, d), dtype=np.float32)
    query_vectors = rng.random((queries, d), dtype=np.float32)
    ids = np.arange(n, dtype=np.int64)

    # Precompute distances once.
    query_norms = np.sum(
        query_vectors * query_vectors,
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
        - 2.0 * (query_vectors @ vectors.T)
    )

    distances = np.maximum(distances, 0.0)

    print("Dataset:")
    print(f"  N = {n:,}")
    print(f"  D = {d}")
    print(f"  queries = {queries}")
    print(f"  k = {k}")
    print()

    print("Profiling top-k selection...")
    print()

    for batch_size in [1, 8, 16, 32, 64, 128]:
        batch_distances = distances[:batch_size]

        # ---------------------------------------------
        # argpartition
        # ---------------------------------------------

        t0 = time.perf_counter()

        candidate_indices = np.argpartition(
            batch_distances,
            kth=k - 1,
            axis=1,
        )[:, :k]

        candidate_distances = np.take_along_axis(
            batch_distances,
            candidate_indices,
            axis=1,
        )

        t1 = time.perf_counter()

        # ---------------------------------------------
        # ID lookup
        # ---------------------------------------------

        candidate_ids = ids[candidate_indices]

        t2 = time.perf_counter()

        # ---------------------------------------------
        # Final deterministic sorting
        # ---------------------------------------------

        for i in range(batch_size):
            np.lexsort(
                (
                    candidate_ids[i],
                    candidate_distances[i],
                )
            )

        t3 = time.perf_counter()

        partition_ms = (t1 - t0) * 1000
        id_lookup_ms = (t2 - t1) * 1000
        sorting_ms = (t3 - t2) * 1000
        total_ms = (t3 - t0) * 1000

        print(f"batch={batch_size:3d}")
        print(f"  argpartition: {partition_ms:8.3f} ms")
        print(f"  id lookup:    {id_lookup_ms:8.3f} ms")
        print(f"  final sort:   {sorting_ms:8.3f} ms")
        print(f"  total:        {total_ms:8.3f} ms")
        print()


if __name__ == "__main__":
    main()