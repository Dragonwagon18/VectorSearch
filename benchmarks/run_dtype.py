from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from vector_search.exact_numpy import ExactNumPyIndex


N = 100_000
D = 128
NUM_QUERIES = 256
K = 10

DTYPES = [
    np.float32,
    np.float64,
]


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def benchmark_dtype(
    dtype: np.dtype,
    vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray,
) -> dict:

    build_start = time.perf_counter()

    index = ExactNumPyIndex(
        vectors=vectors,
        ids=ids,
    )

    build_time = time.perf_counter() - build_start

    # Warm-up
    index.search(
        queries[0],
        k=K,
    )

    latencies = []

    for query in queries:
        start = time.perf_counter()

        index.search(
            query,
            k=K,
        )

        elapsed = time.perf_counter() - start
        latencies.append(elapsed)

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    total_time = sum(latencies)
    qps = NUM_QUERIES / total_time

    memory_bytes = vectors.nbytes

    return {
        "dtype": str(np.dtype(dtype)),
        "N": N,
        "D": D,
        "queries": NUM_QUERIES,
        "k": K,
        "memory_bytes": memory_bytes,
        "memory_mb": memory_bytes / (1024 * 1024),
        "build_time_seconds": build_time,
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "p99_ms": p99 * 1000,
        "qps": qps,
    }


def main() -> None:
    rng = np.random.default_rng(42)

    print("Dataset:")
    print(f"  N = {N:,}")
    print(f"  D = {D}")
    print(f"  queries = {NUM_QUERIES}")
    print(f"  k = {K}")
    print()

    results = []

    for dtype in DTYPES:
        print(f"Benchmarking dtype={np.dtype(dtype)}...")

        vectors = rng.random(
            (N, D),
            dtype=dtype,
        )

        queries = rng.random(
            (NUM_QUERIES, D),
            dtype=dtype,
        )

        ids = np.arange(
            N,
            dtype=np.int64,
        )

        result = benchmark_dtype(
            dtype,
            vectors,
            queries,
            ids,
        )

        results.append(result)

        print(
            f"  memory={result['memory_mb']:.2f} MB "
            f"p50={result['p50_ms']:.3f} ms "
            f"p95={result['p95_ms']:.3f} ms "
            f"p99={result['p99_ms']:.3f} ms "
            f"QPS={result['qps']:.2f}"
        )

        print()

    output_path = Path(
        "benchmarks/results/dtype.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"Results written to {output_path}"
    )


if __name__ == "__main__":
    main()