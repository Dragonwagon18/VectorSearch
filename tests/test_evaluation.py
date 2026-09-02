import numpy as np
import pytest

from vector_search.evaluation import recall_at_k


def test_perfect_recall():
    exact = np.array([1, 2, 3, 4, 5])
    approximate = np.array([1, 2, 3, 4, 5])

    assert recall_at_k(exact, approximate) == 1.0


def test_partial_recall():
    exact = np.array([1, 2, 3, 4, 5])
    approximate = np.array([1, 2, 10, 20, 30])

    assert recall_at_k(exact, approximate) == 0.4


def test_zero_recall():
    exact = np.array([1, 2, 3])
    approximate = np.array([4, 5, 6])

    assert recall_at_k(exact, approximate) == 0.0


def test_order_does_not_matter():
    exact = np.array([1, 2, 3])
    approximate = np.array([3, 2, 1])

    assert recall_at_k(exact, approximate) == 1.0


def test_rejects_empty_exact_ids():
    with pytest.raises(ValueError):
        recall_at_k(
            np.array([]),
            np.array([1, 2, 3]),
        )


def test_rejects_empty_approximate_ids():
    with pytest.raises(ValueError):
        recall_at_k(
            np.array([1, 2, 3]),
            np.array([]),
        )


def test_rejects_non_1d_exact_ids():
    with pytest.raises(ValueError):
        recall_at_k(
            np.array([[1, 2, 3]]),
            np.array([1, 2, 3]),
        )


def test_rejects_non_1d_approximate_ids():
    with pytest.raises(ValueError):
        recall_at_k(
            np.array([1, 2, 3]),
            np.array([[1, 2, 3]]),
        )