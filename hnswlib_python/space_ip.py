"""Port of ``hnswlib/space_ip.h`` (inner product / cosine space)."""

from __future__ import annotations

import math
from array import array


class InnerProductSpace:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def distance(self, a, b) -> float:
        return 1.0 - sum(x * y for x, y in zip(a, b))

    def normalize(self, vector) -> array:
        values = [float(x) for x in vector]
        norm = math.sqrt(sum(x * x for x in values))
        if norm == 0.0:
            return array("f", values)
        return array("f", [x / norm for x in values])


class CosineSpace(InnerProductSpace):
    """Cosine space normalizes both indexed and query vectors."""


__all__ = ["InnerProductSpace", "CosineSpace"]
