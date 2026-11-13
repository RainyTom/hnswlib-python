"""Port of the interface in ``hnswlib/stop_condition.h``.

Stop conditions are not used by this port; the abstract surface is kept so the
module layout matches upstream.
"""

from __future__ import annotations

from .hnswlib import BaseSearchStopCondition


class StopConditionInterface(BaseSearchStopCondition):
    def should_stop_search(self, dist_to_query: float, lower_bound: float) -> bool:
        raise NotImplementedError

    def should_consider_candidate(
        self, dist_to_query: float, lower_bound: float
    ) -> bool:
        raise NotImplementedError

    def should_remove_extra(self) -> bool:
        raise NotImplementedError

    def add_point_to_result(self, label, data, dist: float) -> None:
        raise NotImplementedError

    def remove_point_from_result(self, label, data, dist: float) -> None:
        raise NotImplementedError

    def filter_results(self, result) -> None:
        raise NotImplementedError


__all__ = ["StopConditionInterface"]
