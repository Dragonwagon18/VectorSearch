import time

import numpy as np

from vector_search.exact_numpy import ExactNumPyIndex


N = 100_000
D = 128
NUM_QUERIES = 256
K = 10
REPEATS = 5


def percentile(values, p):
    return float(np.percentile(values, p))


def benchmark_distance(vectors, queries):
    timings = {
        "subtraction": [],
        "einsum": [],
        "distance": [],
    }

    for query in queries:
        # ---------------------------------------------------------
        # 1. Subtraction only
        # ---------------------------------------------------------
        start = time.perf_counter()

        delta = vectors - query

        elapsed = (time.perf_counter() - start) * 1000
        timings["subtraction"].append(elapsed)

        # ---------------------------------------------------------
        # 2. Einsum only
        # ---------------------------------------------------------
        start = time.perf_counter()

        distances = np.einsum("ij,ij->i", delta, delta)

        elapsed = (time.perf_counter() - start) * 1000
        timings["einsum"].append(elapsed)

        # ---------------------------------------------------------
        # 3. Complete distance computation
        # ---------------------------------------------------------
        start = time.perf_counter()

        delta = vectors - query
        distances = np.einsum("ij,ij->i", delta, delta)

        elapsed = (time.perf_counter() - start) * 1000
        timings["distance"].append(elapsed)

    return timings


def benchmark_selection(distances, ids, k):
    timings = {
        "argpartition": [],
        "final_sort": [],
        "selection": [],
    }

    for distance in distances:
        # ---------------------------------------------------------
        # argpartition
        # ---------------------------------------------------------
        start = time.perf_counter()

        candidate_indices = np.argpartition(distance, k - 1)[:k]

        elapsed = (time.perf_counter() - start) * 1000
        timings["argpartition"].append(elapsed)

        # ---------------------------------------------------------
        # Final deterministic sort
        # ---------------------------------------------------------
        start = time.perf_counter()

        candidate_distances = distance[candidate_indices]
        candidate_ids = ids[candidate_indices]

        order = np.lexsort(
            (
                candidate_ids,
                candidate_distances,
            )
        )

        candidate_indices = candidate_indices[order]

        elapsed = (time.perf_counter() - start) * 1000
        timings["final_sort"].append(elapsed)

        # ---------------------------------------------------------
        # Complete selection
        # ---------------------------------------------------------
        start = time.perf_counter()

        candidate_indices = np.argpartition(distance, k - 1)[:k]

        candidate_distances = distance[candidate_indices]
        candidate_ids = ids[candidate_indices]

        order = np.lexsort(
            (
                candidate_ids,
                candidate_distances,
            )
        )

        candidate_indices = candidate_indices[order]

        elapsed = (time.perf_counter() - start) * 1000
        timings["selection"].append(elapsed)

    return timings


def summarize(name, values):
    print(
        f"  {name:<14}"
        f" p50={percentile(values, 50):8.3f} ms"
        f" p95={percentile(values, 95):8.3f} ms"
        f" p99={percentile(values, 99):8.3f} ms"
    )


def main():
    rng = np.random.default_rng(42)

    vectors = rng.random(
        (N, D),
        dtype=np.float32,
    )

    queries = rng.random(
        (NUM_QUERIES, D),
        dtype=np.float32,
    )

    ids = np.arange(N, dtype=np.int64)

    print("Dataset:")
    print(f"  N = {N:,}")
    print(f"  D = {D}")
    print(f"  queries = {NUM_QUERIES}")
    print(f"  k = {K}")
    print(f"  repeats = {REPEATS}")

    print()
    print("Profiling distance computation...")

    # Warmup
    warmup_query = queries[0]

    delta = vectors - warmup_query
    _ = np.einsum("ij,ij->i", delta, delta)

    for repeat in range(REPEATS):
        timings = benchmark_distance(
            vectors,
            queries,
        )

        print()
        print(f"========== distance run={repeat + 1} ==========")

        summarize("subtraction", timings["subtraction"])
        summarize("einsum", timings["einsum"])
        summarize("distance", timings["distance"])

    # -------------------------------------------------------------
    # Generate distances once for profiling top-k selection.
    # -------------------------------------------------------------
    all_distances = []

    for query in queries:
        delta = vectors - query
        distance = np.einsum(
            "ij,ij->i",
            delta,
            delta,
        )
        all_distances.append(distance)

    all_distances = np.asarray(all_distances)

    print()
    print("Profiling top-k selection...")

    for repeat in range(REPEATS):
        timings = benchmark_selection(
            all_distances,
            ids,
            K,
        )

        print()
        print(f"========== selection run={repeat + 1} ==========")

        summarize(
            "argpartition",
            timings["argpartition"],
        )

        summarize(
            "final_sort",
            timings["final_sort"],
        )

        summarize(
            "selection",
            timings["selection"],
        )

    # -------------------------------------------------------------
    # Full search using the actual index.
    # -------------------------------------------------------------
    print()
    print("Profiling complete search...")

    index = ExactNumPyIndex(
        vectors,
        ids,
    )

    search_times = []

    for query in queries:
        start = time.perf_counter()

        _ = index.search(query, K)

        elapsed = (time.perf_counter() - start) * 1000
        search_times.append(elapsed)

    print()
    summarize("full search", search_times)

    p50 = percentile(search_times, 50)

    if p50 > 0:
        qps = 1000.0 / p50
        print(f"  estimated QPS: {qps:.2f}")


if __name__ == "__main__":
    main()