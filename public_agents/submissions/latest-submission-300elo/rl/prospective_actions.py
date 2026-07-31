"""Shared bounded enumeration for prospective legal-action candidates."""
from __future__ import annotations

import itertools
import math
from typing import Any

from rl.encoder.encoding import MAX_OPTIONS


PROSPECTIVE_ACTION_CANDIDATE_VERSION = "2.0.0"


def _combination_from_rank(count: int, size: int, rank: int) -> tuple[int, ...]:
    """Unrank one lexicographic combination without enumerating its prefix."""

    total = math.comb(count, size)
    if not 0 <= rank < total:
        raise ValueError("combination rank is outside the legal domain")
    values: list[int] = []
    lower = 0
    for remaining in range(size, 0, -1):
        for value in range(lower, count):
            suffixes = math.comb(count - value - 1, remaining - 1)
            if rank < suffixes:
                values.append(value)
                lower = value + 1
                break
            rank -= suffixes
    return tuple(values)


def _action_from_rank(
    count: int,
    min_count: int,
    max_count: int,
    rank: int,
) -> tuple[int, ...]:
    for size in range(min_count, max_count + 1):
        size_total = math.comb(count, size)
        if rank < size_total:
            return _combination_from_rank(count, size, rank)
        rank -= size_total
    raise ValueError("action rank is outside the legal domain")


def enumerate_prospective_actions(
    select: dict[str, Any],
    *,
    max_branches: int,
) -> tuple[tuple[int, ...], ...]:
    """Return all legal actions when possible, otherwise full-domain coverage.

    A bounded candidate set is sampled at evenly spaced lexicographic ranks,
    including both endpoints. This avoids the former prefix bias while staying
    deterministic and identical between offline sidecars and PyTorch runtime.
    """

    if max_branches <= 0:
        raise ValueError("max_branches must be positive")
    count = len(select.get("option") or [])
    if count > MAX_OPTIONS:
        raise ValueError("decision has more options than the encoder supports")
    min_count = max(0, int(select.get("minCount", 1) or 0))
    max_count = min(count, max(min_count, int(select.get("maxCount", 1) or 0)))
    if min_count > count:
        raise ValueError("decision minCount exceeds its option count")
    total = sum(math.comb(count, size) for size in range(min_count, max_count + 1))
    if total <= max_branches:
        return tuple(
            tuple(int(index) for index in action)
            for size in range(min_count, max_count + 1)
            for action in itertools.combinations(range(count), size)
        )
    ranks = [
        (index * (total - 1)) // (max_branches - 1)
        for index in range(max_branches)
    ] if max_branches > 1 else [total // 2]
    return tuple(
        _action_from_rank(count, min_count, max_count, rank)
        for rank in ranks
    )
