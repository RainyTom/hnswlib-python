"""Port of ``hnswlib/hnswalg.h``: HierarchicalNSW."""

from __future__ import annotations

import heapq
import math
import pickle
from array import array

from .hnswlib import AlgorithmInterface
from .space_ip import CosineSpace, InnerProductSpace
from .space_l2 import L2Space


class MT19937:
    """C++ ``std::mt19937`` seeded through ``init_genrand``."""

    def __init__(self, seed: int) -> None:
        self.mt = [0] * 624
        self.index = 624
        self.mt[0] = seed & 0xFFFFFFFF
        for i in range(1, 624):
            self.mt[i] = (
                1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i
            ) & 0xFFFFFFFF

    def _twist(self) -> None:
        for i in range(624):
            y = (self.mt[i] & 0x80000000) | (self.mt[(i + 1) % 624] & 0x7FFFFFFF)
            self.mt[i] = (
                self.mt[(i + 397) % 624]
                ^ (y >> 1)
                ^ (0x9908B0DF if y & 1 else 0)
            )
        self.index = 0

    def next_uint32(self) -> int:
        if self.index >= 624:
            self._twist()
        y = self.mt[self.index]
        self.index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & 0xFFFFFFFF

    def uniform(self) -> float:
        a = self.next_uint32()
        b = self.next_uint32()
        return ((a >> 5) * 67108864.0 + (b >> 6)) / 9007199254740992.0


class MinstdRand0:
    """C++ libstdc++ ``std::minstd_rand0``."""

    def __init__(self, seed: int) -> None:
        self.state = (seed % 2147483647) or 1

    def next_uint32(self) -> int:
        self.state = (16807 * self.state) % 2147483647
        return self.state

    def uniform(self) -> float:
        r = 2147483646.0
        a = self.next_uint32() - 1
        b = self.next_uint32() - 1
        return (a * r + b) / (r * r)


def _make_rng(name: str, seed: int):
    if name == "minstd_rand0":
        return MinstdRand0(seed)
    if name == "mt19937":
        return MT19937(seed)
    raise ValueError("level_rng must be 'mt19937' or 'minstd_rand0'")


def _make_space(space: str, dim: int):
    if space == "l2":
        return L2Space(dim)
    if space == "ip":
        return InnerProductSpace(dim)
    if space == "cosine":
        return CosineSpace(dim)
    raise ValueError("space must be one of 'cosine', 'ip', 'l2'")


class HierarchicalNSW(AlgorithmInterface):
    def __init__(
        self,
        dim: int,
        space: str,
        max_elements: int,
        M: int = 16,
        ef_construction: int = 200,
        random_seed: int = 100,
        level_rng: str = "mt19937",
    ) -> None:
        self.dim = dim
        self.space = space
        self.space_obj = _make_space(space, dim)
        self.max_elements = int(max_elements)
        self.M = int(M)
        self.maxM = self.M
        self.maxM0 = 2 * self.M
        self.ef_construction = int(ef_construction)
        self.ef = 10
        self.mult = 1.0 / math.log(self.M)
        self.level_rng = level_rng
        self.rng = _make_rng(level_rng, random_seed)
        self.vectors: list[array] = []
        self.labels: list[int] = []
        self.label_to_id: dict[int, int] = {}
        self.levels: list[int] = []
        self.graph: list[dict[int, list[int]]] = [{}]
        self.entry: int | None = None
        self.max_level = -1

    @property
    def count(self) -> int:
        return len(self.vectors)

    def _distance(self, a, b) -> float:
        return self.space_obj.distance(a, b)

    def _random_level(self) -> int:
        u = max(self.rng.uniform(), 1e-12)
        return int(-math.log(u) * self.mult)

    def _greedy(self, query, entry: int, layer: int) -> int:
        current = entry
        current_dist = self._distance(query, self.vectors[current])
        changed = True
        while changed:
            changed = False
            for candidate in self.graph[layer].get(current, []):
                distance = self._distance(query, self.vectors[candidate])
                if distance < current_dist:
                    current, current_dist = candidate, distance
                    changed = True
        return current

    def _search_layer(
        self, query, entry: int, ef: int, layer: int, construction: bool
    ) -> list[tuple[float, int]]:
        visited = {entry}
        distance = self._distance(query, self.vectors[entry])
        candidates = [(distance, entry)]
        top = [(-distance, entry)]
        while candidates:
            current_dist, current = heapq.heappop(candidates)
            if not construction and current_dist > -top[0][0]:
                break
            for neighbour in self.graph[layer].get(current, []):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                nd = self._distance(query, self.vectors[neighbour])
                if len(top) < ef or nd < -top[0][0]:
                    heapq.heappush(candidates, (nd, neighbour))
                    heapq.heappush(top, (-nd, neighbour))
                    if len(top) > ef:
                        heapq.heappop(top)
        return sorted((-d, node) for d, node in top)

    def _heuristic(
        self, candidates: list[tuple[float, int]], degree: int
    ) -> list[tuple[float, int]]:
        """Port of ``getNeighborsByHeuristic2``."""
        if len(candidates) < degree:
            return sorted(candidates)
        selected: list[tuple[float, int]] = []
        for distance, node in sorted(candidates):
            if len(selected) >= degree:
                break
            good = True
            for _, other in selected:
                if self._distance(self.vectors[node], self.vectors[other]) < distance:
                    good = False
                    break
            if good:
                selected.append((distance, node))
        return selected

    def _connect(
        self, node: int, candidates: list[tuple[float, int]], layer: int
    ) -> int:
        selected = self._heuristic(candidates, self.M)
        self.graph[layer][node] = [other for _, other in selected]
        max_degree = self.maxM0 if layer == 0 else self.maxM
        for _, neighbour in selected:
            links = self.graph[layer].setdefault(neighbour, [])
            if node in links:
                continue
            if len(links) < max_degree:
                links.append(node)
            else:
                merged = [
                    (self._distance(self.vectors[node], self.vectors[neighbour]), node)
                ]
                merged += [
                    (self._distance(self.vectors[other], self.vectors[neighbour]), other)
                    for other in links
                ]
                kept = self._heuristic(merged, max_degree)
                self.graph[layer][neighbour] = [other for _, other in kept]
        return selected[0][1] if selected else node

    def add_point(self, vector, label: int) -> None:
        if self.count >= self.max_elements:
            raise RuntimeError("The number of elements exceeds the specified limit")
        if label in self.label_to_id:
            raise RuntimeError("Label already exists")
        stored = self.space_obj.normalize(vector)
        node = len(self.vectors)
        self.vectors.append(stored)
        self.labels.append(int(label))
        self.label_to_id[int(label)] = node
        level = self._random_level()
        self.levels.append(level)
        while len(self.graph) <= level:
            self.graph.append({})
        for layer in range(level + 1):
            self.graph[layer].setdefault(node, [])

        if self.entry is None:
            self.entry = node
            self.max_level = level
            return

        query = stored
        current = self.entry
        if level < self.max_level:
            for layer in range(self.max_level, level, -1):
                current = self._greedy(query, current, layer)
        for layer in range(min(level, self.max_level), -1, -1):
            candidates = self._search_layer(
                query, current, self.ef_construction, layer, construction=True
            )
            current = self._connect(node, candidates, layer)
        if level > self.max_level:
            self.entry = node
            self.max_level = level

    def search_knn(self, query, k: int) -> list[tuple[int, float]]:
        if self.entry is None or k <= 0:
            return []
        q = self.space_obj.normalize(query)
        current = self.entry
        current_dist = self._distance(q, self.vectors[current])
        for layer in range(self.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                for candidate in self.graph[layer].get(current, []):
                    distance = self._distance(q, self.vectors[candidate])
                    if distance < current_dist:
                        current, current_dist = candidate, distance
                        changed = True
        candidates = self._search_layer(
            q, current, max(self.ef, k), 0, construction=False
        )
        candidates.sort()
        return [(self.labels[node], distance) for distance, node in candidates[:k]]

    def state(self) -> dict:
        return {
            "dim": self.dim,
            "space": self.space,
            "max_elements": self.max_elements,
            "M": self.M,
            "ef_construction": self.ef_construction,
            "ef": self.ef,
            "level_rng": self.level_rng,
            "vectors": self.vectors,
            "labels": self.labels,
            "label_to_id": self.label_to_id,
            "levels": self.levels,
            "graph": self.graph,
            "entry": self.entry,
            "max_level": self.max_level,
        }

    @classmethod
    def from_state(cls, state: dict) -> "HierarchicalNSW":
        obj = cls(
            dim=state["dim"],
            space=state["space"],
            max_elements=state["max_elements"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            random_seed=100,
            level_rng=state["level_rng"],
        )
        obj.ef = state["ef"]
        obj.vectors = state["vectors"]
        obj.labels = state["labels"]
        obj.label_to_id = state["label_to_id"]
        obj.levels = state["levels"]
        obj.graph = state["graph"]
        obj.entry = state["entry"]
        obj.max_level = state["max_level"]
        return obj


def load_state(path: str) -> HierarchicalNSW:
    with open(path, "rb") as handle:
        return HierarchicalNSW.from_state(pickle.load(handle))


__all__ = ["HierarchicalNSW", "MT19937", "MinstdRand0", "load_state"]
