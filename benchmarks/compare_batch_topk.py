from __future__ import annotations

import time

import numpy as np


def argsort_topk(
    distances: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(
        distances,
        axis=1,
        kind="stable",
    )[:, :k]

    top_distances = np.take_along_axis(
        distances,
        order,
        axis=1,
    )

    return order, top_distances


def argpartition_topk(
    distances: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.argpartition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    top_distances = np.take_along_axis(
        distances,
        indices,
        axis=1,
    )

    # Sort only k candidates.
    order = np.argsort(
        top_distances,
        axis=1,
    )

    indices = np.take_along_axis(
        indices,
        order,
        axis=1,
    )

    top_distances = np.take_along_axis(
        top_distances,
        order,
        axis=1,
    )

    return indices, top_distances


def benchmark(
    distances: np.ndarray,
    k: int,
    fn,
    name: str,
) -> float:
    # Warm-up
    fn(distances, k)

    start = time.perf_counter()
    fn(distances, k)
    elapsed = time.perf_counter() - start

    print(
        f"{name:<15} "
        f"{elapsed * 1000:9.3f} ms"
    )

    return elapsed


def main() -> None:
    rng = np.random.default_rng(42)

    n = 100_000
    d = 128
    queries = 128
    k = 10

    vectors = rng.random(
        (n, d),
        dtype=np.float32,
    )

    query_vectors = rng.random(
        (queries, d),
        dtype=np.float32,
    )

    # Compute distances once.
    query_norms = np.sum(
        query_vectors * query_vectors,
        axis=1,
        keepdims=True,
    )

    vector_norms = np.sum(
        vectors * vectors,
        axis=1,
        keepdims=True,
    ).T

    distances = (
        query_norms
        + vector_norms
        - 2.0 * (query_vectors @ vectors.T)
    )

    distances = np.maximum(distances, 0.0)

    print("Dataset:")
    print(f"  N = {n:,}")
    print(f"  D = {d}")
    print(f"  queries = {queries}")
    print(f"  k = {k}")
    print()

    print("Comparing batch top-k methods...")
    print()

    arg_indices, arg_distances = argpartition_topk(
        distances,
        k,
    )

    sort_indices, sort_distances = argsort_topk(
        distances,
        k,
    )

    # ---------------------------------------------------------
    # Correctness check
    # ---------------------------------------------------------

    for i in range(queries):
        expected = set(sort_indices[i])
        actual = set(arg_indices[i])

        if expected != actual:
            raise AssertionError(
                f"Mismatch for query {i}"
            )

    print("✓ Both methods return identical top-k sets")
    print()

    for batch_size in [1, 8, 16, 32, 64, 128]:
        batch = distances[:batch_size]

        print(f"batch={batch_size:3d}")

        argsort_time = benchmark(
            batch,
            k,
            argsort_topk,
            "  argsort",
        )

        argpartition_time = benchmark(
            batch,
            k,
            argpartition_topk,
            "  argpartition",
        )

        speedup = argsort_time / argpartition_time

        print(f"  speedup:       {speedup:6.2f}x")
        print()


if __name__ == "__main__":
    main()