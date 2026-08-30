import numpy as np
import pytest

from vector_search.exact_numpy import ExactNumPyIndex


def test_add_stores_vector_and_id():
    index = ExactNumPyIndex(3)

    index.add([1, 2, 3], 10)

    np.testing.assert_array_equal(
        index._vectors[0],
        np.array([1, 2, 3], dtype=np.float32),
    )
    assert index._ids == [10]


def test_add_rejects_wrong_dimension():
    index = ExactNumPyIndex(3)

    with pytest.raises(ValueError):
        index.add([1, 2], 10)


def test_add_rejects_non_finite_vector():
    index = ExactNumPyIndex(3)

    with pytest.raises(ValueError):
        index.add([1, np.nan, 3], 10)

    with pytest.raises(ValueError):
        index.add([1, np.inf, 3], 11)


def test_add_rejects_duplicate_id():
    index = ExactNumPyIndex(3)

    index.add([1, 2, 3], 10)

    with pytest.raises(ValueError):
        index.add([4, 5, 6], 10)

def test_known_neighbors_are_sorted():
    index = ExactNumPyIndex(2)

    index.add([0, 0], 10)
    index.add([1, 0], 11)
    index.add([3, 0], 12)

    result = index.search([0.9, 0], k=2)

    assert result.ids.tolist() == [11, 10]
    np.testing.assert_allclose(
        result.distances,
        [0.01, 0.81],
        atol=1e-6,
    )


def test_ties_are_deterministic_by_id():
    index = ExactNumPyIndex(1)

    index.add([-1], 20)
    index.add([1], 10)

    result = index.search([0], k=2)

    assert result.ids.tolist() == [10, 20]


def test_rejects_wrong_dimension():
    index = ExactNumPyIndex(2)

    with pytest.raises(ValueError):
        index.add([1, 2, 3], 7)


def test_rejects_duplicate_id():
    index = ExactNumPyIndex(2)

    index.add([1, 2], 10)

    with pytest.raises(ValueError):
        index.add([3, 4], 10)


def test_rejects_empty_index():
    index = ExactNumPyIndex(2)

    with pytest.raises(ValueError):
        index.search([1, 2], k=1)


def test_k_must_be_valid():
    index = ExactNumPyIndex(2)
    index.add([1, 2], 10)

    with pytest.raises(ValueError):
        index.search([1, 2], k=0)

    with pytest.raises(ValueError):
        index.search([1, 2], k=2)


def test_k_one_returns_one_neighbor():
    index = ExactNumPyIndex(2)

    index.add([0, 0], 10)
    index.add([1, 0], 11)
    index.add([3, 0], 12)

    result = index.search([0.9, 0], k=1)

    assert result.ids.tolist() == [11]
    assert len(result.distances) == 1


def test_k_n_returns_all_neighbors():
    index = ExactNumPyIndex(2)

    index.add([0, 0], 10)
    index.add([1, 0], 11)
    index.add([3, 0], 12)

    result = index.search([0.9, 0], k=3)

    assert result.ids.tolist() == [11, 10, 12]
    assert len(result.distances) == 3


def test_rejects_wrong_query_dimension():
    index = ExactNumPyIndex(2)
    index.add([0, 0], 10)

    with pytest.raises(ValueError):
        index.search([1, 2, 3], k=1)


def test_rejects_non_finite_vector():
    index = ExactNumPyIndex(2)

    with pytest.raises(ValueError):
        index.add([1, np.nan], 10)

    with pytest.raises(ValueError):
        index.add([1, np.inf], 11)


def test_rejects_non_finite_query():
    index = ExactNumPyIndex(2)
    index.add([0, 0], 10)

    with pytest.raises(ValueError):
        index.search([np.nan, 0], k=1)

    with pytest.raises(ValueError):
        index.search([np.inf, 0], k=1)