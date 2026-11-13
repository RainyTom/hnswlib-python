"""Port of ``hnswlib/visited_list_pool.h``.

The Python graph search uses a plain ``set`` per query; this module keeps the
upstream abstraction for API parity.
"""

from __future__ import annotations


class VisitedList:
    def __init__(self) -> None:
        self.mass: set[int] = set()

    def reset(self) -> None:
        self.mass.clear()


class VisitedListPool:
    def __init__(self, init_max_items: int, num_lists: int) -> None:
        self._lists = [VisitedList() for _ in range(max(1, num_lists))]

    def get_free_visited_list(self) -> VisitedList:
        item = self._lists.pop()
        item.reset()
        return item

    def release_visited_list(self, visited_list: VisitedList) -> None:
        self._lists.append(visited_list)


__all__ = ["VisitedList", "VisitedListPool"]
