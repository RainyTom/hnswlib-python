"""Port of ``python_bindings/LazyIndex.py``."""

from __future__ import annotations

from .bindings import Index


class LazyIndex:
    """Create the HNSW index on first use, mirroring hnswlib's LazyIndex."""

    def __init__(
        self,
        space: str,
        dim: int,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 100,
        level_rng: str = "mt19937",
    ) -> None:
        self._index = Index(space=space, dim=dim)
        self._index.init_index(
            max_elements,
            M=M,
            ef_construction=ef_construction,
            random_seed=random_seed,
            level_rng=level_rng,
        )

    def __getattr__(self, name):
        return getattr(self._index, name)


__all__ = ["LazyIndex"]
