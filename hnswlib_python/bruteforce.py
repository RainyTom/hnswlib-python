"""Port of ``hnswlib/bruteforce.h``."""

from __future__ import annotations

import heapq

from .hnswlib import AlgorithmInterface, labeltype
from .space_ip import CosineSpace, InnerProductSpace
from .space_l2 import L2Space


class BruteforceSearch(AlgorithmInterface):
    def __init__(self, space: str, dim: int, max_elements: int) -> None:
        if space == "l2":
            self.space = L2Space(dim)
        elif space == "ip":
            self.space = InnerProductSpace(dim)
        elif space == "cosine":
            self.space = CosineSpace(dim)
        else:
            raise ValueError("space must be one of 'cosine', 'ip', 'l2'")
        self.max_elements = int(max_elements)
        self.vectors: list = []
        self.labels: list[labeltype] = []

    def add_point(self, vector, label: labeltype) -> None:
        if len(self.vectors) >= self.max_elements:
            raise RuntimeError("The number of elements exceeds the specified limit")
        self.vectors.append(self.space.normalize(vector))
        self.labels.append(int(label))

    def search_knn(self, query, k: int) -> list[tuple[labeltype, float]]:
        q = self.space.normalize(query)
        distances = [
            (self.space.distance(q, vector), index)
            for index, vector in enumerate(self.vectors)
        ]
        best = heapq.nsmallest(k, distances)
        return [(self.labels[index], distance) for distance, index in best]


__all__ = ["BruteforceSearch"]
