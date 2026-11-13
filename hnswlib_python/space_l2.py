"""Port of ``hnswlib/space_l2.h``."""

from __future__ import annotations

from array import array
from operator import sub


class L2Space:
    def __init__(self, dim: int) -> None:
        self.dim = int(dim)

    def distance(self, a, b) -> float:
        return sum(d * d for d in map(sub, a, b))

    def normalize(self, vector) -> array:
        return array("f", [float(x) for x in vector])


__all__ = ["L2Space"]
