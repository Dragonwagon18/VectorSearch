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


def benchmark(
    name: str,
    vectors: np.ndarray,
    queries: np.ndarray,
    ids: np.ndarray,
) -> dict:
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

    total_time = np.sum(latencies)

    return {
        "layout": name,
        "c_contiguous": bool(vectors.flags["C_CONTIGUOUS"]),
        "f_contiguous": bool(vectors.flags["F_CONTIGUOUS"]),
        "strides": list(vectors.strides),
        "memory_mb": vectors.nbytes / (1024 * 1024),
        "p50_ms": float(np.percentile(latencies, 50) * 1000),
        "p95_ms": float(np.percentile(latencies, 95) * 1000),
        "p99_ms": float(np.percentile(latencies, 99) * 1000),
        "qps": float(NUM_QUERIES / total_time),
    }


def create_layouts(
    base: np.ndarray,
) -> list[tuple[str, np.ndarray]]:
    """
    Create arrays with different memory-access patterns.

    All arrays contain the same logical vectors.
    """

    layouts = []

    # ---------------------------------------------------------
    # 1. Normal C-contiguous representation
    # ---------------------------------------------------------
    contiguous = np.ascontiguousarray(base)

    layouts.append(
        ("contiguous", contiguous)
    )

    # ---------------------------------------------------------
    # 2. Stride x2
    #
    # Every vector is separated by one unused row.
    # ---------------------------------------------------------
    padded_2 = np.empty(
        (N * 2, D),
        dtype=np.float32,
    )

    padded_2[::2] = base

    stride_2 = padded_2[::2]

    layouts.append(
        ("stride_x2", stride_2)
    )

    # ---------------------------------------------------------
    # 3. Stride x4
    # ---------------------------------------------------------
    padded_4 = np.empty(
        (N * 4, D),
        dtype=np.float32,
    )

    padded_4[::4] = base

    stride_4 = padded_4[::4]

    layouts.append(
        ("stride_x4", stride_4)
    )

    # ---------------------------------------------------------
    # 4. Stride x8
    # ---------------------------------------------------------
    padded_8 = np.empty(
        (N * 8, D),
        dtype=np.float32,
    )

    padded_8[::8] = base

    stride_8 = padded_8[::8]

    layouts.append(
        ("stride_x8", stride_8)
    )

    return layouts


def main() -> None:
    print("Dataset:")
    print(f"  N = {N:,}")
    print(f"  D = {D}")
    print(f"  queries = {NUM_QUERIES}")
    print(f"  k = {K}")
    print(f"  repeats = {REPEATS}")
    print()

    rng = np.random.default_rng(42)

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

    layouts = create_layouts(base_vectors)

    all_results = []

    for name, vectors in layouts:
        print("=" * 60)
        print(f"Layout: {name}")

        print(
            f"  shape:       {vectors.shape}"
        )

        print(
            f"  strides:     {vectors.strides}"
        )

        print(
            f"  C-contig:    "
            f"{vectors.flags['C_CONTIGUOUS']}"
        )

        print(
            f"  F-contig:    "
            f"{vectors.flags['F_CONTIGUOUS']}"
        )

        print(
            f"  memory:      "
            f"{vectors.nbytes / (1024 * 1024):.2f} MB"
        )

        runs = []

        for run in range(1, REPEATS + 1):
            result = benchmark(
                name,
                vectors,
                queries,
                ids,
            )

            runs.append(result)

            print(
                f"  run={run} "
                f"p50={result['p50_ms']:.3f} ms "
                f"p95={result['p95_ms']:.3f} ms "
                f"p99={result['p99_ms']:.3f} ms "
                f"QPS={result['qps']:.2f}"
            )

        summary = {
            "layout": name,
            "strides": list(vectors.strides),
            "memory_mb": vectors.nbytes / (1024 * 1024),
            "c_contiguous": bool(
                vectors.flags["C_CONTIGUOUS"]
            ),
            "f_contiguous": bool(
                vectors.flags["F_CONTIGUOUS"]
            ),
            "median_p50_ms": float(
                np.median(
                    [r["p50_ms"] for r in runs]
                )
            ),
            "median_p95_ms": float(
                np.median(
                    [r["p95_ms"] for r in runs]
                )
            ),
            "median_p99_ms": float(
                np.median(
                    [r["p99_ms"] for r in runs]
                )
            ),
            "median_qps": float(
                np.median(
                    [r["qps"] for r in runs]
                )
            ),
            "runs": runs,
        }

        all_results.append(summary)

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

    # ---------------------------------------------------------
    # Relative comparison
    # ---------------------------------------------------------

    baseline = all_results[0]

    print()
    print("=" * 60)
    print("Relative to contiguous baseline")
    print("=" * 60)

    for result in all_results:
        p50_ratio = (
            result["median_p50_ms"]
            / baseline["median_p50_ms"]
        )

        qps_ratio = (
            result["median_qps"]
            / baseline["median_qps"]
        )

        print(
            f"{result['layout']:12s} "
            f"p50={p50_ratio:.2f}x "
            f"QPS={qps_ratio:.2f}x"
        )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    output_path = Path(
        "benchmarks/results/strided.json"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w") as f:
        json.dump(
            all_results,
            f,
            indent=2,
        )

    print()
    print(
        f"Results written to {output_path}"
    )


if __name__ == "__main__":
    main()