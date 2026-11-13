"""Python binding surface mirroring ``python_bindings/bindings.cpp``."""

from __future__ import annotations

from hnswlib_python.hnswalg import HierarchicalNSW


class Index:
    def __init__(self, space: str = "l2", dim: int = 0) -> None:
        if space not in ("cosine", "ip", "l2"):
            raise ValueError("space must be one of 'cosine', 'ip', 'l2'")
        self.space = space
        self.dim = int(dim)
        self._index: HierarchicalNSW | None = None

    def init_index(
        self,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 100,
        allow_replace_deleted: bool = False,
        level_rng: str = "mt19937",
        build_backend: str = "python",
    ) -> None:
        self._index = HierarchicalNSW(
            dim=self.dim,
            space=self.space,
            max_elements=int(max_elements),
            M=int(M),
            ef_construction=int(ef_construction),
            random_seed=int(random_seed),
            level_rng=level_rng,
        )
        self._index.build_numpy = build_backend == "numpy"

    def set_ef(self, ef: int) -> None:
        self._require().ef = int(ef)

    def set_num_threads(self, num_threads: int) -> None:
        return None

    def add_items(self, data, ids=None, num_threads: int = -1, **kwargs) -> None:
        index = self._require()
        index.set_build_data(data)
        if ids is None:
            ids = list(range(index.count, index.count + len(data)))
        for vector, label in zip(data, ids):
            index.add_point(vector, int(label))

    def knn_query(self, data, k: int = 1, num_threads: int = -1, **kwargs):
        index = self._require()
        if len(data) > 0 and not hasattr(data[0], "__len__"):
            data = [data]
        labels = []
        distances = []
        for vector in data:
            found = index.search_knn(vector, int(k))
            labels.append([label for label, _ in found])
            distances.append([distance for _, distance in found])
        return labels, distances

    def get_current_count(self) -> int:
        return self._require().count

    def get_max_elements(self) -> int:
        return self._require().max_elements

    def save_index(self, path: str) -> None:
        import pickle

        with open(path, "wb") as handle:
            pickle.dump(self._require().state(), handle)

    def load_index(self, path: str) -> None:
        import pickle

        with open(path, "rb") as handle:
            state = pickle.load(handle)
        self._index = HierarchicalNSW.from_state(state)

    def _require(self) -> HierarchicalNSW:
        if self._index is None:
            raise RuntimeError("call init_index() before using the index")
        return self._index


__all__ = ["Index"]
