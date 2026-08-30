import time

import numpy as np

from vector_search.batch import batch_search


def benchmark_batch(
    vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray,
    k: int,
    batch_size: int,
) -> None:
    total = len(queries)

    batches = [
        queries[i : i + batch_size]
        for i in range(0, total, batch_size)
    ]

    distance_times = []
    selection_times = []
    total_times = []

    for batch in batches:
        t0 = time.perf_counter()

        # Distance computation
        query_norms = np.sum(
            batch * batch,
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
            - 2.0 * (batch @ vectors.T)
        )

        distances = np.maximum(distances, 0.0)

        t1 = time.perf_counter()

        # Top-k selection
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

        for i in range(len(batch)):
            order = np.lexsort(
                (
                    candidate_ids[i],
                    candidate_distances[i],
                )
            )

        t2 = time.perf_counter()

        distance_times.append(t1 - t0)
        selection_times.append(t2 - t1)
        total_times.append(t2 - t0)

    distance_total = sum(distance_times)
    selection_total = sum(selection_times)
    total_time = sum(total_times)

    print(f"batch={batch_size}")
    print(
        f"  distance:  {distance_total * 1000:.3f} ms"
    )
    print(
        f"  selection: {selection_total * 1000:.3f} ms"
    )
    print(
        f"  total:     {total_time * 1000:.3f} ms"
    )
    print(
        f"  per-query: {total_time / total * 1000:.3f} ms"
    )
    print(
        f"  QPS:       {total / total_time:.2f}"
    )
    print()


def main() -> None:
    rng = np.random.default_rng(42)

    n = 100_000
    d = 128
    total_queries = 256
    k = 10

    vectors = rng.random(
        (n, d),
        dtype=np.float32,
    )

    queries = rng.random(
        (total_queries, d),
        dtype=np.float32,
    )

    ids = np.arange(n, dtype=np.int64)

    print("Dataset:")
    print(f"  N = {n:,}")
    print(f"  D = {d}")
    print(f"  queries = {total_queries}")
    print(f"  k = {k}")
    print()

    print("Profiling batch search...")
    print()

    for batch_size in [1, 8, 16, 32, 64, 128]:
        benchmark_batch(
            vectors,
            queries,
            ids,
            k,
            batch_size,
        )


if __name__ == "__main__":
    main()