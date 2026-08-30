from __future__ import annotations

import time

import numpy as np

from vector_search.exact_numpy import ExactNumPyIndex


N = 100_000
D = 128
NUM_QUERIES = 256
K = 10
REPEATS = 5

DTYPES = [
    np.float32,
    np.float64,
]


def benchmark(
    vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray,
) -> tuple[float, float, float]:
    index = ExactNumPyIndex(
        vectors=vectors,
        ids=ids,
    )

    # Warm-up
    index.search(queries[0], k=K)

    latencies = []

    for query in queries:
        start = time.perf_counter()

        index.search(
            query,
            k=K,
        )

        latencies.append(
            time.perf_counter() - start
        )

    latencies = np.asarray(latencies)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    return (
        p50 * 1000,
        p95 * 1000,
        p99 * 1000,
    )


def main() -> None:
    print("Dataset:")
    print(f"  N = {N:,}")
    print(f"  D = {D}")
    print(f"  queries = {NUM_QUERIES}")
    print(f"  k = {K}")
    print(f"  repeats = {REPEATS}")
    print()

    rng = np.random.default_rng(42)

    ids = np.arange(
        N,
        dtype=np.int64,
    )

    for dtype in DTYPES:
        print(f"========== {np.dtype(dtype)} ==========")

        vectors = rng.random(
            (N, D),
            dtype=dtype,
        )

        queries = rng.random(
            (NUM_QUERIES, D),
            dtype=dtype,
        )

        p50_values = []
        p95_values = []
        p99_values = []

        for run in range(1, REPEATS + 1):
            p50, p95, p99 = benchmark(
                vectors,
                queries,
                ids,
            )

            p50_values.append(p50)
            p95_values.append(p95)
            p99_values.append(p99)

            print(
                f"run={run} "
                f"p50={p50:8.3f} ms "
                f"p95={p95:8.3f} ms "
                f"p99={p99:8.3f} ms"
            )

        print()

        print("Summary:")
        print(
            f"  median p50: "
            f"{np.median(p50_values):.3f} ms"
        )

        print(
            f"  median p95: "
            f"{np.median(p95_values):.3f} ms"
        )

        print(
            f"  median p99: "
            f"{np.median(p99_values):.3f} ms"
        )

        print(
            f"  memory: "
            f"{vectors.nbytes / (1024 * 1024):.2f} MB"
        )

        print()


if __name__ == "__main__":
    main()