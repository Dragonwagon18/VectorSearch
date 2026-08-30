import json
import platform
import time
from pathlib import Path

import numpy as np

from vector_search.exact_numpy import ExactNumPyIndex


D = 128
K = 10
NUM_QUERIES = 200

DATASET_SIZES = [
    256,
    512,
    1_000,
    2_000,
    4_000,
    8_000,
    16_000,
    32_000,
    64_000,
    128_000,
    256_000,
    512_000,
    1_000_000,
]


def benchmark_search(index, queries):
    latencies = []

    # Warm-up
    for query in queries[:10]:
        index.search(query, K)

    for query in queries:
        start = time.perf_counter_ns()

        index.search(query, K)

        end = time.perf_counter_ns()

        latencies.append((end - start) / 1_000_000)

    latencies = np.asarray(latencies)

    return {
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "qps": float(1000.0 / np.mean(latencies)),
    }


def main():
    rng = np.random.default_rng(2026)

    queries = rng.normal(
        size=(NUM_QUERIES, D)
    ).astype(np.float32)

    results = []

    for n in DATASET_SIZES:
        print(f"Benchmarking N={n:,}...")

        vectors = rng.normal(
            size=(n, D)
        ).astype(np.float32)

        memory_mb = vectors.nbytes / (1024 * 1024)

        start = time.perf_counter_ns()

        index = ExactNumPyIndex(vectors)

        end = time.perf_counter_ns()

        build_ms = (end - start) / 1_000_000

        search_results = benchmark_search(index, queries)

        results.append(
            {
                "n": n,
                "d": D,
                "k": K,
                "memory_mb": memory_mb,
                "build_ms": build_ms,
                "queries": NUM_QUERIES,
                **search_results,
            }
        )

        print(
            f"  memory={memory_mb:.2f} MB "
            f"p50={search_results['p50_ms']:.3f} ms "
            f"p95={search_results['p95_ms']:.3f} ms "
            f"QPS={search_results['qps']:.2f}"
        )

        del vectors
        del index

    output = {
        "experiment": "cache_memory",
        "configuration": {
            "dimension": D,
            "k": K,
            "queries": NUM_QUERIES,
            "dtype": "float32",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": __import__("os").cpu_count(),
        },
        "results": results,
    }

    output_path = Path("benchmarks/results/cache.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Results written to {output_path}")


if __name__ == "__main__":
    main()