import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil

from vector_search.datasets import synthetic_dataset
from vector_search.exact_loop import ExactLoopIndex
from vector_search.exact_numpy import ExactNumPyIndex


def percentile_ms(samples_ns, p):
    return float(np.percentile(samples_ns, p) / 1_000_000)


def benchmark(index, queries, k):
    # Warm-up
    for q in queries[:20]:
        index.search(q, k=k)

    samples = []

    for q in queries[20:]:
        started = time.perf_counter_ns()
        index.search(q, k=k)
        samples.append(time.perf_counter_ns() - started)

    mean_ns = float(np.mean(samples))

    return {
        "queries": len(samples),
        "p50_ms": percentile_ms(samples, 50),
        "p95_ms": percentile_ms(samples, 95),
        "p99_ms": percentile_ms(samples, 99),
        "qps": 1e9 / mean_ns,
    }


rows = []

for n in [1_000, 10_000, 100_000]:
    vectors, queries = synthetic_dataset(
        n,
        d=128,
        query_count=220,
    )

    # ---------------------------------------------------------
    # Exact Python loop
    # ---------------------------------------------------------
    loop_index = ExactLoopIndex(128)

    started = time.perf_counter_ns()

    for item_id, vector in enumerate(vectors):
        loop_index.add(vector, item_id)

    loop_build_ms = (
        time.perf_counter_ns() - started
    ) / 1_000_000

    loop_result = benchmark(
        loop_index,
        queries,
        k=10,
    )

    # ---------------------------------------------------------
    # Exact NumPy
    # ---------------------------------------------------------
    started = time.perf_counter_ns()

    numpy_index = ExactNumPyIndex(vectors)

    numpy_build_ms = (
        time.perf_counter_ns() - started
    ) / 1_000_000

    numpy_result = benchmark(
        numpy_index,
        queries,
        k=10,
    )

    # ---------------------------------------------------------
    # Correctness check
    # ---------------------------------------------------------
    for q in queries[:10]:
        expected = loop_index.search(q, k=10)
        actual = numpy_index.search(q, k=10)

        np.testing.assert_array_equal(
            actual.ids,
            expected.ids,
        )

        np.testing.assert_allclose(
            actual.distances,
            expected.distances,
            rtol=1e-5,
        )

    rows.append(
        {
            "n": n,
            "d": 128,
            "k": 10,
            "loop": {
                **loop_result,
                "build_ms": loop_build_ms,
            },
            "numpy": {
                **numpy_result,
                "build_ms": numpy_build_ms,
            },
            "speedup_qps": (
                numpy_result["qps"]
                / loop_result["qps"]
            ),
            "speedup_p50": (
                loop_result["p50_ms"]
                / numpy_result["p50_ms"]
            ),
        }
    )


payload = {
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_bytes": psutil.virtual_memory().total,
    },
    "results": rows,
}


output = Path("benchmarks/results/exact_comparison.json")
output.parent.mkdir(parents=True, exist_ok=True)

output.write_text(
    json.dumps(payload, indent=2)
)

print(json.dumps(payload, indent=2))