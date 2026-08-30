import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from vector_search.exact_numpy import ExactNumPyIndex


RESULTS_PATH = Path("benchmarks/results/dimension.json")

DATASET_SIZE = 100_000
K = 10
QUERY_COUNT = 200
SEED = 2026

DIMENSIONS = [
    64,
    128,
    384,
    768,
    1536,
]


def percentile(values, p):
    return float(np.percentile(values, p))


def benchmark_search(index, queries):
    latencies = []

    # Warm-up
    for query in queries[:10]:
        index.search(query, k=K)

    # Timed queries
    for query in queries:
        start = time.perf_counter()

        index.search(query, k=K)

        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    latencies = np.asarray(latencies, dtype=np.float64)

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    mean_latency = float(np.mean(latencies))

    return {
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "qps": 1000.0 / mean_latency,
    }


def main():
    rng = np.random.default_rng(SEED)

    results = []

    for dimension in DIMENSIONS:
        print(f"Benchmarking D={dimension}...")

        vectors = rng.random(
            (DATASET_SIZE, dimension),
            dtype=np.float32,
        )

        queries = rng.random(
            (QUERY_COUNT, dimension),
            dtype=np.float32,
        )

        ids = np.arange(
            DATASET_SIZE,
            dtype=np.int64,
        )

        # Build index.
        build_start = time.perf_counter()

        index = ExactNumPyIndex(
            vectors,
            ids,
        )

        build_ms = (time.perf_counter() - build_start) * 1000

        # Actual vector memory footprint.
        memory_mb = (
            index.vectors.nbytes / (1024 * 1024)
        )

        benchmark = benchmark_search(
            index,
            queries,
        )

        result = {
            "n": DATASET_SIZE,
            "d": dimension,
            "k": K,
            "queries": QUERY_COUNT,
            "memory_mb": memory_mb,
            "build_ms": build_ms,
            "p50_ms": benchmark["p50_ms"],
            "p95_ms": benchmark["p95_ms"],
            "p99_ms": benchmark["p99_ms"],
            "qps": benchmark["qps"],
        }

        results.append(result)

        print(
            f"  memory={memory_mb:.2f} MB "
            f"p50={benchmark['p50_ms']:.3f} ms "
            f"p95={benchmark['p95_ms']:.3f} ms "
            f"QPS={benchmark['qps']:.2f}"
        )

    payload = {
        "experiment": "dimension_scaling",
        "configuration": {
            "dataset_size": DATASET_SIZE,
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