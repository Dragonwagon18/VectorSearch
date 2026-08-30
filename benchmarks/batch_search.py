import time
from pathlib import Path

import numpy as np

from vector_search.exact_numpy import ExactNumPyIndex


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
        Database vectors of shape (N, D).

    queries:
        Query vectors of shape (B, D).

    ids:
        Item IDs of shape (N,).

    k:
        Number of nearest neighbors to return.

    Returns
    -------
    result_ids:
        Array of shape (B, k).

    result_distances:
        Array of shape (B, k).
    """

    vectors = np.asarray(vectors, dtype=np.float32)
    queries = np.asarray(queries, dtype=np.float32)
    ids = np.asarray(ids)

    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")

    if queries.ndim != 2:
        raise ValueError("queries must be a 2D array")

    if vectors.shape[1] != queries.shape[1]:
        raise ValueError("query dimension does not match vector dimension")

    if ids.ndim != 1:
        raise ValueError("ids must be a 1D array")

    if len(ids) != len(vectors):
        raise ValueError("number of ids must match number of vectors")

    if not np.isfinite(vectors).all():
        raise ValueError("vectors must contain only finite values")

    if not np.isfinite(queries).all():
        raise ValueError("queries must contain only finite values")

    n = len(vectors)

    if n == 0:
        raise ValueError("cannot search an empty index")

    if not 1 <= k <= n:
        raise ValueError("k must satisfy 1 <= k <= number of vectors")

    # ---------------------------------------------------------
    # Compute all query-to-vector distances.
    #
    # queries: (B, D)
    # vectors: (N, D)
    #
    # result:  (B, N)
    # ---------------------------------------------------------
    distances = np.sum(
        (queries[:, None, :] - vectors[None, :, :]) ** 2,
        axis=2,
    )

    # ---------------------------------------------------------
    # Select top-k candidates for every query.
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
    # (distance, item_id)
    #
    # We sort each query's k candidates.
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


def main() -> None:
    rng = np.random.default_rng(42)

    # ---------------------------------------------------------
    # Benchmark configuration
    # ---------------------------------------------------------
    n = 100_000
    d = 128
    total_queries = 256
    k = 10

    batch_sizes = [1, 8, 16, 32, 64, 128]

    print("Generating dataset...")

    vectors = rng.random(
        (n, d),
        dtype=np.float32,
    )

    queries = rng.random(
        (total_queries, d),
        dtype=np.float32,
    )

    ids = np.arange(n, dtype=np.int64)

    print()
    print("Dataset:")
    print(f"  N = {n:,}")
    print(f"  D = {d}")
    print(f"  queries = {total_queries}")
    print(f"  k = {k}")

    print()
    print("Benchmarking batch search...")

    results = []

    # ---------------------------------------------------------
    # Warm-up
    # ---------------------------------------------------------
    batch_search(
        vectors,
        queries[:1],
        ids,
        k,
    )

    # ---------------------------------------------------------
    # Benchmark each batch size
    # ---------------------------------------------------------
    for batch_size in batch_sizes:
        num_batches = total_queries // batch_size

        batch_times = []

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size

            batch_queries = queries[start:end]

            t0 = time.perf_counter()

            batch_search(
                vectors,
                batch_queries,
                ids,
                k,
            )

            elapsed = time.perf_counter() - t0

            batch_times.append(elapsed)

        batch_times = np.asarray(batch_times)

        total_time = float(np.sum(batch_times))

        per_query_latency = total_time / total_queries

        qps = total_queries / total_time

        p50 = float(np.percentile(batch_times / batch_size * 1000, 50))
        p95 = float(np.percentile(batch_times / batch_size * 1000, 95))
        p99 = float(np.percentile(batch_times / batch_size * 1000, 99))

        result = {
            "batch_size": batch_size,
            "total_queries": total_queries,
            "total_time_seconds": total_time,
            "per_query_latency_ms": per_query_latency * 1000,
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "qps": qps,
        }

        results.append(result)

        print(
            f"  batch={batch_size:3d} "
            f"per_query={per_query_latency * 1000:8.3f} ms "
            f"QPS={qps:8.2f}"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    output_dir = Path("benchmarks/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "batch.json"

    import json

    output = {
        "configuration": {
            "N": n,
            "D": d,
            "queries": total_queries,
            "k": k,
            "dtype": "float32",
            "batch_sizes": batch_sizes,
        },
        "results": results,
    }

    with output_file.open("w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    main()