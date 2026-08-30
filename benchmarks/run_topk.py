import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from vector_search.exact_numpy import ExactNumPyIndex


RESULTS_PATH = Path("benchmarks/results/topk.json")

DIMENSION = 128
K = 10
QUERY_COUNT = 200
SEED = 2026

DATASET_SIZES = [
    1_000,
    10_000,
    100_000,
    1_000_000,
]


def percentile(values, p):
    return float(np.percentile(values, p))


def search_argsort(vectors, ids, query, k):
    """
    Exact top-k search using full sorting.

    Complexity:
        Distance computation: O(ND)
        Selection: O(N log N)
    """
    delta = vectors - query
    distances = np.einsum("ij,ij->i", delta, delta)

    # Full deterministic sort by:
    #   1. distance
    #   2. ID
    order = np.lexsort((ids, distances))

    rows = order[:k]

    return ids[rows], distances[rows].astype(np.float32)


def search_argpartition(vectors, ids, query, k):
    """
    Exact top-k search using partial selection.

    First selects the k smallest distances with argpartition,
    then sorts only those k candidates deterministically.

    Complexity:
        Distance computation: O(ND)
        Selection: O(N)
        Final sort: O(k log k)
    """
    delta = vectors - query
    distances = np.einsum("ij,ij->i", delta, delta)

    # Find the k smallest distances without fully sorting N elements.
    candidate_rows = np.argpartition(distances, k - 1)[:k]

    candidate_distances = distances[candidate_rows]
    candidate_ids = ids[candidate_rows]

    # Deterministic ordering among the selected k candidates.
    order = np.lexsort(
        (
            candidate_ids,
            candidate_distances,
        )
    )

    rows = candidate_rows[order]

    return ids[rows], distances[rows].astype(np.float32)


def benchmark_search(vectors, ids, queries, method):
    latencies = []

    if method == "argsort":
        search_fn = search_argsort
    elif method == "argpartition":
        search_fn = search_argpartition
    else:
        raise ValueError(f"Unknown method: {method}")

    # Warm-up
    for query in queries[:10]:
        search_fn(vectors, ids, query, K)

    # Timed queries
    for query in queries:
        start = time.perf_counter()

        search_fn(vectors, ids, query, K)

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.asarray(latencies, dtype=np.float64)

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    mean_latency = float(np.mean(latencies))

    return {
        "method": method,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "qps": 1000.0 / mean_latency,
    }


def verify_results(vectors, ids, queries):
    """
    Verify that argsort and argpartition return identical
    deterministic top-k results.
    """
    for query in queries:
        ids_argsort, distances_argsort = search_argsort(
            vectors,
            ids,
            query,
            K,
        )

        ids_argpartition, distances_argpartition = search_argpartition(
            vectors,
            ids,
            query,
            K,
        )

        if not np.array_equal(
            ids_argsort,
            ids_argpartition,
        ):
            raise AssertionError(
                "Top-k IDs differ between argsort and argpartition"
            )

        if not np.allclose(
            distances_argsort,
            distances_argpartition,
            rtol=1e-5,
            atol=1e-6,
        ):
            raise AssertionError(
                "Top-k distances differ between argsort and argpartition"
            )


def main():
    rng = np.random.default_rng(SEED)

    results = []

    for n in DATASET_SIZES:
        print(f"Benchmarking N={n:,}...")

        vectors = rng.random(
            (n, DIMENSION),
            dtype=np.float32,
        )

        queries = rng.random(
            (QUERY_COUNT, DIMENSION),
            dtype=np.float32,
        )

        ids = np.arange(
            n,
            dtype=np.int64,
        )

        # Construct the existing ExactNumPyIndex.
        # This validates that the dataset itself is valid.
        build_start = time.perf_counter()

        index = ExactNumPyIndex(
            vectors,
            ids,
        )

        build_ms = (time.perf_counter() - build_start) * 1000

        # Verify correctness before benchmarking.
        verify_results(
            index.vectors,
            index.ids,
            queries,
        )

        argsort_result = benchmark_search(
            index.vectors,
            index.ids,
            queries,
            method="argsort",
        )

        argpartition_result = benchmark_search(
            index.vectors,
            index.ids,
            queries,
            method="argpartition",
        )

        speedup = (
            argsort_result["p50_ms"]
            / argpartition_result["p50_ms"]
        )

        results.append(
            {
                "n": n,
                "d": DIMENSION,
                "k": K,
                "queries": QUERY_COUNT,
                "build_ms": build_ms,
                "argsort": argsort_result,
                "argpartition": argpartition_result,
                "p50_speedup": speedup,
            }
        )

        print(
            f"  argsort       "
            f"p50={argsort_result['p50_ms']:.3f} ms "
            f"QPS={argsort_result['qps']:.2f}"
        )

        print(
            f"  argpartition  "
            f"p50={argpartition_result['p50_ms']:.3f} ms "
            f"QPS={argpartition_result['qps']:.2f}"
        )

        print(
            f"  speedup={speedup:.2f}x"
        )

    payload = {
        "experiment": "topk_selection",
        "configuration": {
            "dimension": DIMENSION,
            "k": K,
            "queries": QUERY_COUNT,
            "dtype": "float32",
            "seed": SEED,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
        },
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        )
    )

    print()
    print(f"Results written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()