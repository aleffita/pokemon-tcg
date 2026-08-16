from __future__ import annotations

import json

import pytest
import torch

import scripts.rl.run_ar021 as run_ar021_module
from scripts.rl.run_ar021 import (
    _save_completed_epoch_checkpoint,
    deck_relative_group_advantages,
)


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


def test_deck_credit_prefers_dense_policy_scores_over_tied_terminal_returns() -> None:
    left = _collection(0, 0, 0, [-1.0, -1.0])
    right = _collection(1, 0, 0, [-1.0, -1.0])
    left["policy_scores"] = [-0.6, -0.7]
    right["policy_scores"] = [-1.2, -1.1]
    advantages, cohorts = deck_relative_group_advantages([left, right])
    assert advantages == pytest.approx([1.0, -1.0])
    assert cohorts[0]["zero_variance"] is False


def test_completed_epoch_checkpoint_validates_before_atomic_promotion(
    tmp_path, monkeypatch
) -> None:
    latest_path = tmp_path / "candidate.latest.pt"
    latest_path.write_bytes(b"previous-complete-epoch")
    observed: dict[str, object] = {}

    def fake_save(path, _model, _metadata, **kwargs):
        assert path.name == "candidate.latest.pt.incomplete"
        path.write_bytes(b"next-complete-epoch")
        observed["diagnostics"] = kwargs["diagnostics"]
        observed["config"] = kwargs["config"]
        return "a" * 64

    def fake_validate(path, *, approved_root_sha256):
        assert path.read_bytes() == b"next-complete-epoch"
        assert latest_path.read_bytes() == b"previous-complete-epoch"
        observed["approved_root_sha256"] = approved_root_sha256
        return {}

    monkeypatch.setattr(run_ar021_module, "save_grpo_candidate_checkpoint", fake_save)
    monkeypatch.setattr(
        run_ar021_module, "validate_candidate_provenance", fake_validate
    )
    metadata = _save_completed_epoch_checkpoint(
        output_dir=tmp_path,
        model=torch.nn.Linear(1, 1),
        model_metadata={},
        root_sha256="b" * 64,
        sample_manifest_sha256="c" * 64,
        bundle_sha256="d" * 64,
        sample_manifest_content_sha256="e" * 64,
        config={"update_epochs": 12},
        diagnostics={"epoch_metrics": [{"epoch": 1}, {"epoch": 2}]},
        experiment="AR-test",
        completed_epoch=2,
    )

    assert latest_path.read_bytes() == b"next-complete-epoch"
    assert not (tmp_path / "candidate.latest.pt.incomplete").exists()
    assert metadata["completed_update_epochs"] == 2
    assert metadata["validated"] is True
    assert observed["diagnostics"]["optimizer_steps"] == 2
    assert observed["config"]["checkpoint_kind"] == "completed_epoch"
    sidecar = json.loads((tmp_path / "candidate.latest.json").read_text())
    assert sidecar["candidate_sha256"] == "a" * 64
    assert sidecar["completed_update_epochs"] == 2
