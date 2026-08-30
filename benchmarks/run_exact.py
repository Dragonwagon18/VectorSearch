import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil

from vector_search.datasets import synthetic_dataset
from vector_search.exact_numpy import ExactNumPyIndex


def percentile_ms(samples_ns, p):
    return float(np.percentile(samples_ns, p) / 1_000_000)


rows = []

for n in [1_000, 10_000, 100_000]:
    vectors, queries = synthetic_dataset(
        n,
        d=128,
        query_count=220,
    )

    started = time.perf_counter_ns()
    index = ExactNumPyIndex(vectors)
    build_ms = (time.perf_counter_ns() - started) / 1_000_000

    # Warm-up
    for q in queries[:20]:
        index.search(q, k=10)

    # Actual measurements
    samples = []

    for q in queries[20:]:
        started = time.perf_counter_ns()
        index.search(q, k=10)
        samples.append(time.perf_counter_ns() - started)

    rows.append(
        {
            "n": n,
            "d": 128,
            "k": 10,
            "queries": len(samples),
            "p50_ms": percentile_ms(samples, 50),
            "p95_ms": percentile_ms(samples, 95),
            "p99_ms": percentile_ms(samples, 99),
            "qps": 1e9 / float(np.mean(samples)),
            "build_ms": build_ms,
            "vector_bytes": int(vectors.nbytes),
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

output = Path("benchmarks/results/exact.json")
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, indent=2))

print(json.dumps(payload, indent=2))