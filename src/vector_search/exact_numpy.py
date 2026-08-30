import numpy as np

from .exact_loop import SearchResult


class ExactNumPyIndex:
    """
    Exact nearest-neighbor search using NumPy vectorization.

    vectors: shape (n, d)
    ids:     shape (n,)

    Distance computation uses:

        ||x - q||² = ||x||² + ||q||² - 2(x · q)

    Vector norms are precomputed during index construction to
    avoid materializing an (n, d) temporary delta matrix for
    every query.
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
            self.ids = np.arange(
                len(x),
                dtype=np.int64,
            )
        else:
            self.ids = np.asarray(
                ids,
                dtype=np.int64,
            )

            if self.ids.shape != (len(x),):
                raise ValueError("ids must match vector count")

            if len(np.unique(self.ids)) != len(x):
                raise ValueError("ids must be unique")

        # ---------------------------------------------------------
        # Precompute ||x||² for every database vector.
        #
        # This is computed once during index construction instead
        # of once for every query.
        # ---------------------------------------------------------
        self.vector_norms = np.einsum(
            "ij,ij->i",
            self.vectors,
            self.vectors,
        )

    def search(
        self,
        query,
        k: int,
    ) -> SearchResult:
        q = np.asarray(
            query,
            dtype=np.float32,
        )

        if q.shape != (self.dimension,):
            raise ValueError(
                f"expected shape {(self.dimension,)}, got {q.shape}"
            )

        if not np.all(np.isfinite(q)):
            raise ValueError(
                "query must contain only finite values"
            )

        if not 1 <= k <= len(self.vectors):
            raise ValueError(
                f"k must be between 1 and {len(self.vectors)}, got {k}"
            )

        # ---------------------------------------------------------
        # Squared L2 distance:
        #
        # ||x - q||²
        # = ||x||² + ||q||² - 2(x · q)
        #
        # This avoids creating:
        #
        #     delta = vectors - q
        #
        # which would allocate an (N, D) temporary array.
        # ---------------------------------------------------------

        query_norm = np.dot(q, q)

        distances = (
            self.vector_norms
            + query_norm
            - 2.0 * np.dot(self.vectors, q)
        )

        # ---------------------------------------------------------
        # Floating-point arithmetic can produce tiny negative
        # values even though squared distances cannot mathematically
        # be negative.
        # ---------------------------------------------------------

        distances = np.maximum(
            distances,
            0.0,
        )

        # ---------------------------------------------------------
        # Deterministic ordering by:
        #
        #     (distance, item_id)
        #
        # We currently sort all vectors because argpartition alone
        # cannot guarantee deterministic ID-based tie-breaking at
        # the kth boundary.
        # ---------------------------------------------------------

        order = np.lexsort(
            (
                self.ids,
                distances,
            )
        )

        rows = order[:k]

        return SearchResult(
            ids=self.ids[rows],
            distances=distances[rows].astype(
                np.float32
            ),
        )