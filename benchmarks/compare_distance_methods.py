import time

import numpy as np


N = 100_000
D = 128
NUM_QUERIES = 256
REPEATS = 5


def percentile(values, p):
    return float(np.percentile(values, p))


def method_subtract_einsum(vectors, query):
    """
    Method A:

        delta = vectors - query
        distances = sum(delta * delta)

    This creates an intermediate (N, D) array.
    """
    delta = vectors - query
    distances = np.einsum("ij,ij->i", delta, delta)

    return distances


def method_dot_product(vectors, query, vector_norms):
    """
    Method B:

        ||x - q||² = ||x||² + ||q||² - 2(x · q)

    This avoids explicitly creating the (N, D) delta matrix.
    """
    query_norm = np.dot(query, query)

    distances = (
        vector_norms
        + query_norm
        - 2.0 * np.dot(vectors, query)
    )

    return distances


def benchmark_method(method, vectors, queries, vector_norms=None):
    timings = []

    for query in queries:
        start = time.perf_counter()

        if vector_norms is None:
            _ = method(vectors, query)
        else:
            _ = method(vectors, query, vector_norms)

        elapsed = (time.perf_counter() - start) * 1000
        timings.append(elapsed)

    return timings


def summarize(name, timings):
    p50 = percentile(timings, 50)
    p95 = percentile(timings, 95)
    p99 = percentile(timings, 99)

    qps = 1000.0 / p50

    print(
        f"  {name:<22}"
        f" p50={p50:8.3f} ms"
        f" p95={p95:8.3f} ms"
        f" p99={p99:8.3f} ms"
        f" QPS={qps:8.2f}"
    )

    return p50


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

    # Precompute ||x||² once.
    #
    # This cost is paid during index construction rather than
    # during every search.
    vector_norms = np.einsum(
        "ij,ij->i",
        vectors,
        vectors,
    )

    print("Dataset:")
    print(f"  N = {N:,}")
    print(f"  D = {D}")
    print(f"  queries = {NUM_QUERIES}")
    print(f"  repeats = {REPEATS}")

    # ---------------------------------------------------------
    # Correctness check
    # ---------------------------------------------------------

    print()
    print("Checking numerical equivalence...")

    for query in queries[:10]:
        distances_a = method_subtract_einsum(
            vectors,
            query,
        )

        distances_b = method_dot_product(
            vectors,
            query,
            vector_norms,
        )

        if not np.allclose(
            distances_a,
            distances_b,
            rtol=1e-5,
            atol=1e-5,
        ):
            max_error = np.max(
                np.abs(distances_a - distances_b)
            )

            raise AssertionError(
                "Distance methods are not equivalent. "
                f"Maximum absolute error: {max_error}"
            )

    print("✓ Both methods produce equivalent distances")

    # ---------------------------------------------------------
    # Warmup
    # ---------------------------------------------------------

    query = queries[0]

    _ = method_subtract_einsum(
        vectors,
        query,
    )

    _ = method_dot_product(
        vectors,
        query,
        vector_norms,
    )

    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    print()
    print("Comparing distance computation methods...")

    subtract_results = []
    dot_results = []

    for repeat in range(REPEATS):
        subtract_timings = benchmark_method(
            method_subtract_einsum,
            vectors,
            queries,
        )

        dot_timings = benchmark_method(
            method_dot_product,
            vectors,
            queries,
            vector_norms,
        )

        subtract_results.append(
            percentile(subtract_timings, 50)
        )

        dot_results.append(
            percentile(dot_timings, 50)
        )

        print()
        print(f"========== run={repeat + 1} ==========")

        summarize(
            "subtract + einsum",
            subtract_timings,
        )

        summarize(
            "dot-product identity",
            dot_timings,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    median_subtract = float(
        np.median(subtract_results)
    )

    median_dot = float(
        np.median(dot_results)
    )

    speedup = median_subtract / median_dot

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(
        f"subtract + einsum median p50: "
        f"{median_subtract:.3f} ms"
    )

    print(
        f"dot-product identity median p50: "
        f"{median_dot:.3f} ms"
    )

    print(
        f"speedup: {speedup:.2f}x"
    )

    print()
    print("Memory behavior:")
    print()
    print("Subtract + einsum:")
    print("  vectors → temporary (N, D) delta → distances")
    print()
    print("Dot-product identity:")
    print("  vectors → dot product → distances")
    print()
    print(
        "The dot-product method avoids explicitly "
        "materializing the (N, D) delta matrix."
    )


if __name__ == "__main__":
    main()