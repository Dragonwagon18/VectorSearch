import numpy as np

from .exact_loop import SearchResult


class ExactNumPyIndex:
    def __init__(self, dimension: int):
        if dimension <= 0:
            raise ValueError("dimension must be positive")

        self.dimension = dimension
        self._vectors: list[np.ndarray] = []
        self._ids: list[int] = []

    def add(self, vector, item_id: int) -> None:
        v = np.asarray(vector, dtype=np.float32)

        if v.shape != (self.dimension,):
            raise ValueError(
                f"expected shape {(self.dimension,)}, got {v.shape}"
            )

        if not np.all(np.isfinite(v)):
            raise ValueError("vector must contain only finite values")

        if item_id in self._ids:
            raise ValueError(f"duplicate item_id: {item_id}")

        self._vectors.append(v.copy())
        self._ids.append(item_id)

    def search(self, query, k: int) -> SearchResult:
        q = np.asarray(query, dtype=np.float32)

        if q.shape != (self.dimension,):
            raise ValueError(
                f"expected shape {(self.dimension,)}, got {q.shape}"
            )
        if not np.all(np.isfinite(q)):
            raise ValueError("query must contain only finite values")
        
        if not 1 <= k <= len(self._vectors):
            raise ValueError(
                f"k must be between 1 and {len(self._vectors)}, got {k}"
            )
        vectors = np.stack(self._vectors)

        distances = np.sum((vectors - q) ** 2, axis=1)

        order = np.lexsort((self._ids, distances))

        top = order[:k]

        return SearchResult( ids =np.asarray(self._ids)[top], distances=distances[top].astype(np.float32))