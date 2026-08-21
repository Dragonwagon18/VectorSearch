import numpy as np

from vector_search.metrics import squared_l2, normalize_rows


def test_squared_l2_known_answer():
    assert squared_l2(
        np.array([1, 2]),
        np.array([4, 6]),
    ) == 25.0


def test_normalized_rows_have_unit_norm():
    x = normalize_rows(
        np.array(
            [
                [3, 4],
                [5, 12],
            ],
            dtype=np.float32,
        )
    )

    np.testing.assert_allclose(
        np.linalg.norm(x, axis=1),
        1.0,
        rtol=1e-6,
    )