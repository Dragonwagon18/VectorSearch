from __future__ import annotations

import numpy as np


class NSWIndex:
    """
    Single-layer Navigable Small World graph.

    This is the foundation for understanding HNSW.
    """

    def __init__(
        self,
        dimension: int,
        max_neighbors: int = 8,
    ) -> None:

        if dimension <= 0:
            raise ValueError("dimension must be positive")

        if max_neighbors <= 0:
            raise ValueError("max_neighbors must be positive")

        self.dimension = dimension
        self.max_neighbors = max_neighbors

        self.vectors: list[np.ndarray] = []
        self.ids: list[int] = []

        # internal node index -> neighboring node indices
        self.graph: dict[int, set[int]] = {}

        self.id_to_index: dict[int, int] = {}

        self.entry_point: int | None = None