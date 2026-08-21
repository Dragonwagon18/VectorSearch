import numpy as np


def squared_l2(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(
            f"expected equal 1D shapes, got {a.shape} and {b.shape}"
        )

    delta = a - b
    return float(delta @ delta)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)

    norms = np.linalg.norm(x, axis=1, keepdims=True)

    if np.any(norms == 0):
        raise ValueError(
            "cosine similarity is undefined for zero vectors"
        )

    return x / norms