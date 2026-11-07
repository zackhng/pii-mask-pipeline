"""A set of non-overlapping [start, end) intervals with O(log n) overlap checks."""
from __future__ import annotations

import bisect


class IntervalSet:
    def __init__(self, intervals=()):
        self._starts: list[int] = []
        self._ends: list[int] = []
        for s, e in sorted(intervals):
            self.add(s, e)

    def is_free(self, start: int, end: int) -> bool:
        """True if [start, end) overlaps no stored interval."""
        i = bisect.bisect_right(self._starts, start)
        if i > 0 and self._ends[i - 1] > start:
            return False
        return not (i < len(self._starts) and self._starts[i] < end)

    def add(self, start: int, end: int) -> None:
        i = bisect.bisect_right(self._starts, start)
        self._starts.insert(i, start)
        self._ends.insert(i, end)
