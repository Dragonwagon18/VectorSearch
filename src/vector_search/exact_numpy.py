import numpy as np

from .exact_loop import SearchResult


class ExactNumPyIndex:
    """
    Exact nearest-neighbor search using NumPy vectorization.

    vectors: shape (n, d)
    ids:     shape (n,)
    """

    def __init__(
        self,
        vectors: np.ndarray,
        ids: np.ndarray | None = None,
    ):
        x = np.asarray(vectors, dtype=np.float32)

        if x.ndim != 2 or len(x) == 0:
            raise ValueError("vectors must be a non-empty 2D matrix")

        if not np.all(np.isfinite(x)):
            raise ValueError("vectors must contain only finite values")

        self.vectors = np.ascontiguousarray(x)
        self.dimension = x.shape[1]

        if ids is None:
            self.ids = np.arange(len(x), dtype=np.int64)
        else:
            self.ids = np.asarray(ids, dtype=np.int64)

            if self.ids.shape != (len(x),):
                raise ValueError("ids must match vector count")

            if len(np.unique(self.ids)) != len(x):
                raise ValueError("ids must be unique")

    def search(self, query, k: int) -> SearchResult:
        q = np.asarray(query, dtype=np.float32)

        if q.shape != (self.dimension,):
            raise ValueError(
                f"expected shape {(self.dimension,)}, got {q.shape}"
            )

        if not np.all(np.isfinite(q)):
            raise ValueError("query must contain only finite values")

        if not 1 <= k <= len(self.vectors):
            raise ValueError(
                f"k must be between 1 and {len(self.vectors)}, got {k}"
            )

        # Broadcasting:
        # (n, d) - (d,) -> (n, d)
        delta = self.vectors - q

        # Squared L2 distance for every vector.
        distances = np.einsum("ij,ij->i", delta, delta)

        # Find the k smallest distances without sorting everything.
        # Deterministic ordering by (distance, ID).
        #
        # We currently sort all vectors because argpartition alone
        # cannot guarantee deterministic ID-based tie-breaking at the
        # kth boundary.
        order = np.lexsort(
    (
        self.ids,
        distances,
    )
)

        rows = order[:k]

        return SearchResult(
            ids=self.ids[rows],
            distances=distances[rows].astype(np.float32),
        )