import numpy as np
import pytest

from vector_search.exact_loop import ExactLoopIndex
from vector_search.exact_numpy import ExactNumPyIndex


def test_known_neighbors_are_sorted():
    vectors = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = ExactNumPyIndex(vectors)

    result = index.search([0.0, 0.0], k=3)

    assert result.ids.tolist() == [0, 2, 1]
    assert np.all(result.distances[:-1] <= result.distances[1:])


def test_ties_are_deterministic_by_id():
    vectors = np.array(
        [
            [1.0],
            [-1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    ids = np.array([20, 10, 30])

    index = ExactNumPyIndex(vectors, ids)

    result = index.search([0.0], k=3)

    # IDs 10 and 20 are tied at distance 1.
    assert result.ids.tolist() == [10, 20, 30]


def test_rejects_wrong_dimension():
    vectors = np.zeros((3, 4), dtype=np.float32)

    index = ExactNumPyIndex(vectors)

    with pytest.raises(ValueError):
        index.search(np.zeros(3), k=1)


def test_rejects_non_finite_vector():
    vectors = np.array(
        [
            [0.0, 1.0],
            [np.nan, 2.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        ExactNumPyIndex(vectors)


def test_rejects_non_finite_query():
    index = ExactNumPyIndex(np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(ValueError):
        index.search([np.nan, 0.0], k=1)


def test_rejects_empty_index():
    with pytest.raises(ValueError):
        ExactNumPyIndex(np.empty((0, 4), dtype=np.float32))


def test_k_must_be_valid():
    index = ExactNumPyIndex(np.zeros((3, 2), dtype=np.float32))

    with pytest.raises(ValueError):
        index.search([0.0, 0.0], k=0)

    with pytest.raises(ValueError):
        index.search([0.0, 0.0], k=4)


def test_k_one_returns_one_neighbor():
    vectors = np.array(
        [
            [0.0, 0.0],
            [2.0, 2.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = ExactNumPyIndex(vectors)

    result = index.search([1.1, 1.1], k=1)

    assert result.ids.tolist() == [2]


def test_k_n_returns_all_neighbors():
    vectors = np.array(
        [
            [0.0],
            [2.0],
            [1.0],
        ],
        dtype=np.float32,
    )

    index = ExactNumPyIndex(vectors)

    result = index.search([0.0], k=3)

    assert result.ids.tolist() == [0, 2, 1]


def test_rejects_duplicate_ids():
    vectors = np.zeros((3, 2), dtype=np.float32)
    ids = np.array([1, 1, 2])

    with pytest.raises(ValueError):
        ExactNumPyIndex(vectors, ids)


def test_rejects_wrong_id_count():
    vectors = np.zeros((3, 2), dtype=np.float32)
    ids = np.array([1, 2])

    with pytest.raises(ValueError):
        ExactNumPyIndex(vectors, ids)


def test_numpy_matches_loop_on_seeded_random_data():
    rng = np.random.default_rng(2026)

    x = rng.normal(size=(200, 16)).astype(np.float32)
    q = rng.normal(size=16).astype(np.float32)

    loop = ExactLoopIndex(16)

    for item_id, vector in enumerate(x):
        loop.add(vector, item_id)

    fast = ExactNumPyIndex(x)

    expected = loop.search(q, k=10)
    actual = fast.search(q, k=10)

    np.testing.assert_array_equal(actual.ids, expected.ids)
    np.testing.assert_allclose(
        actual.distances,
        expected.distances,
        rtol=1e-5,
    )


def test_numpy_matches_loop_on_multiple_random_trials():
    rng = np.random.default_rng(2026)

    for n, d, k in [
        (10, 3, 1),
        (20, 8, 5),
        (50, 16, 10),
        (100, 32, 20),
        (200, 64, 50),
    ]:
        for _ in range(20):
            x = rng.normal(size=(n, d)).astype(np.float32)
            q = rng.normal(size=d).astype(np.float32)

            loop = ExactLoopIndex(d)

            for item_id, vector in enumerate(x):
                loop.add(vector, item_id)

            fast = ExactNumPyIndex(x)

            expected = loop.search(q, k=k)
            actual = fast.search(q, k=k)

            np.testing.assert_array_equal(actual.ids, expected.ids)
            np.testing.assert_allclose(
                actual.distances,
                expected.distances,
                rtol=1e-5,
            )

def test_ties_at_k_boundary_are_deterministic_by_id():
    vectors = np.array(
        [
            [1.0],
            [-1.0],
            [1.0],
            [-1.0],
            [2.0],
        ],
        dtype=np.float32,
    )

    ids = np.array([50, 10, 30, 20, 99])

    index = ExactNumPyIndex(vectors, ids)

    result = index.search([0.0], k=2)

    # Four vectors are tied at distance 1.
    # The two smallest IDs must win.
    assert result.ids.tolist() == [10, 20]