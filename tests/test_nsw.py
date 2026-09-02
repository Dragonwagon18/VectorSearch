import pytest

from vector_search.nsw import NSWIndex


def test_initialization():
    index = NSWIndex(
        dimension=3,
        max_neighbors=4,
    )

    assert index.dimension == 3
    assert index.max_neighbors == 4

    assert index.vectors == []
    assert index.ids == []

    assert index.graph == {}

    assert index.entry_point is None


def test_rejects_invalid_dimension():
    with pytest.raises(ValueError):
        NSWIndex(dimension=0)


def test_rejects_invalid_max_neighbors():
    with pytest.raises(ValueError):
        NSWIndex(
            dimension=3,
            max_neighbors=0,
        )