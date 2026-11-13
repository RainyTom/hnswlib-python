"""Port of the shared types in ``hnswlib/hnswlib.h``."""

from __future__ import annotations

from abc import ABC, abstractmethod

labeltype = int
tableint = int
dist_t = float

kInvalidInternalId = 0xFFFFFFFF


class AlgorithmInterface(ABC):
    """Minimal interface mirrored from ``hnswlib/hnswlib.h``."""

    @abstractmethod
    def add_point(self, vector, label: labeltype) -> None:
        raise NotImplementedError

    @abstractmethod
    def search_knn(self, query, k: int) -> list[tuple[labeltype, dist_t]]:
        raise NotImplementedError


class BaseFilterFunctor:
    def __call__(self, label: labeltype) -> bool:
        raise NotImplementedError


class BaseSearchStopCondition(ABC):
    pass


__all__ = [
    "AlgorithmInterface",
    "BaseFilterFunctor",
    "BaseSearchStopCondition",
    "dist_t",
    "kInvalidInternalId",
    "labeltype",
    "tableint",
]
