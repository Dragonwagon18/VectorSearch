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