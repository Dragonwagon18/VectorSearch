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
REPEATS = 5


def percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


def benchmark_layout(
    layout_name: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray,
) -> dict:
    print(f"Benchmarking layout={layout_name}...")

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

    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    p99 = percentile(latencies, 99)

    total_time = sum(latencies)
    qps = NUM_QUERIES / total_time

    return {
        "layout": layout_name,
        "N": N,
        "D": D,
        "queries": NUM_QUERIES,
        "k": K,
        "c_contiguous": bool(vectors.flags["C_CONTIGUOUS"]),
        "f_contiguous": bool(vectors.flags["F_CONTIGUOUS"]),
        "strides": list(vectors.strides),
        "memory_mb": vectors.nbytes / (1024 * 1024),
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "p99_ms": p99 * 1000,
        "qps": qps,
    }


def main() -> None:
    print("Dataset:")
    print(f"  N = {N:,}")
    print(f"  D = {D}")
    print(f"  queries = {NUM_QUERIES}")
    print(f"  k = {K}")
    print(f"  repeats = {REPEATS}")
    print()

    rng = np.random.default_rng(42)

    # Generate one canonical dataset.
    base_vectors = rng.random(
        (N, D),
        dtype=np.float32,
    )

    queries = rng.random(
        (NUM_QUERIES, D),
        dtype=np.float32,
    )

    ids = np.arange(
        N,
        dtype=np.int64,
    )

    # ---------------------------------------------------------
    # Layout 1: C-contiguous
    # ---------------------------------------------------------
    c_vectors = np.ascontiguousarray(
        base_vectors
    )

    # ---------------------------------------------------------
    # Layout 2: F-contiguous
    # ---------------------------------------------------------
    f_vectors = np.asfortranarray(
        base_vectors
    )

    layouts = [
        ("C-contiguous", c_vectors),
        ("F-contiguous", f_vectors),
    ]

    results = []

    for layout_name, vectors in layouts:
        print(
            f"{layout_name}: "
            f"C={vectors.flags['C_CONTIGUOUS']} "
            f"F={vectors.flags['F_CONTIGUOUS']} "
            f"strides={vectors.strides}"
        )

        p50_runs = []
        p95_runs = []
        p99_runs = []
        qps_runs = []

        for run in range(1, REPEATS + 1):
            result = benchmark_layout(
                layout_name,
                vectors,
                queries,
                ids,
            )

            p50_runs.append(result["p50_ms"])
            p95_runs.append(result["p95_ms"])
            p99_runs.append(result["p99_ms"])
            qps_runs.append(result["qps"])

            print(
                f"  run={run} "
                f"p50={result['p50_ms']:.3f} ms "
                f"p95={result['p95_ms']:.3f} ms "
                f"p99={result['p99_ms']:.3f} ms "
                f"QPS={result['qps']:.2f}"
            )

        summary = {
            "layout": layout_name,
            "N": N,
            "D": D,
            "queries": NUM_QUERIES,
            "k": K,
            "c_contiguous": bool(
                vectors.flags["C_CONTIGUOUS"]
            ),
            "f_contiguous": bool(
                vectors.flags["F_CONTIGUOUS"]
            ),
            "strides": list(vectors.strides),
            "memory_mb": vectors.nbytes / (1024 * 1024),
            "median_p50_ms": float(
                np.median(p50_runs)
            ),
            "median_p95_ms": float(
                np.median(p95_runs)
            ),
            "median_p99_ms": float(
                np.median(p99_runs)
            ),
            "median_qps": float(
                np.median(qps_runs)
            ),
            "runs": {
                "p50_ms": p50_runs,
                "p95_ms": p95_runs,
                "p99_ms": p99_runs,
                "qps": qps_runs,
            },
        }

        results.append(summary)

        print()
        print("Summary:")
        print(
            f"  median p50: "
            f"{summary['median_p50_ms']:.3f} ms"
        )
        print(
            f"  median p95: "
            f"{summary['median_p95_ms']:.3f} ms"
        )
        print(
            f"  median p99: "
            f"{summary['median_p99_ms']:.3f} ms"
        )
        print(
            f"  median QPS: "
            f"{summary['median_qps']:.2f}"
        )
        print(
            f"  memory: "
            f"{summary['memory_mb']:.2f} MB"
        )
        print()

    # ---------------------------------------------------------
    # Compare layouts
    # ---------------------------------------------------------
    c_result = results[0]
    f_result = results[1]

    p50_speedup = (
        f_result["median_p50_ms"]
        / c_result["median_p50_ms"]
    )

    qps_ratio = (
        c_result["median_qps"]
        / f_result["median_qps"]
    )

    print("Layout comparison:")
    print(
        f"  C-contiguous p50: "
        f"{c_result['median_p50_ms']:.3f} ms"
    )
    print(
        f"  F-contiguous p50: "
        f"{f_result['median_p50_ms']:.3f} ms"
    )
    print(
        f"  C/F p50 ratio: "
        f"{p50_speedup:.2f}x"
    )
    print(
        f"  C/F QPS ratio: "
        f"{qps_ratio:.2f}x"
    )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------
    output_path = Path(
        "benchmarks/results/layout.json"
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

    print()
    print(
        f"Results written to {output_path}"
    )


if __name__ == "__main__":
    main()