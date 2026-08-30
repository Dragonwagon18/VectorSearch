import numpy as np


def synthetic_dataset(
    n: int,
    d: int,
    query_count: int,
    seed: int = 2026,
):
    rng = np.random.default_rng(seed)

    vectors = rng.normal(size=(n, d)).astype(np.float32)
    queries = rng.normal(size=(query_count, d)).astype(np.float32)

    return vectors, queries