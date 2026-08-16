from __future__ import annotations

import pytest

from scripts.rl.run_ar021 import deck_relative_group_advantages


def _collection(deck: int, matchup: int, group: int, returns: list[float]) -> dict:
    return {
        "learner_deck_index": deck,
        "matchup_index": matchup,
        "group_index": group,
        "returns": returns,
    }


def test_deck_credit_is_paired_within_same_opponent_and_group_seed() -> None:
    collections = [
        _collection(0, 0, 0, [1.0, 1.0]),
        _collection(0, 0, 1, [-1.0, -1.0]),
        _collection(1, 0, 0, [-1.0, -1.0]),
        _collection(1, 0, 1, [1.0, 1.0]),
    ]
    advantages, cohorts = deck_relative_group_advantages(collections)
    assert advantages == pytest.approx([1.0, -1.0, -1.0, 1.0])
    assert len(cohorts) == 2
    assert all(cohort["zero_variance"] is False for cohort in cohorts)


def test_deck_credit_is_zero_when_decks_tie() -> None:
    collections = [
        _collection(0, 2, 3, [1.0, -1.0]),
        _collection(1, 2, 3, [1.0, -1.0]),
    ]
    advantages, cohorts = deck_relative_group_advantages(collections)
    assert advantages == [0.0, 0.0]
    assert cohorts[0]["zero_variance"] is True
