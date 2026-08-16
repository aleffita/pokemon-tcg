from __future__ import annotations

import copy
import json

import pytest
import torch

from scripts.rl.trajectory_group_grpo import (
    _masked_distribution,
    expand_group_advantages,
    flatten_provenance_bundle,
    normalize_group_returns,
    recompute_logprobs_by_decision,
    save_grpo_candidate_checkpoint,
    trajectory_group_grpo_update,
)
from scripts.rl.ppo_micro_update import (
    build_sample_manifest,
    save_compressed_bundle,
    sha256_file,
    validate_candidate_provenance,
)
from scripts.rl.trajectory_probe import APPROVED_STAGE4_ROOT_SHA256


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1], dtype=torch.float32))

    def logits_value(self, model_input, memory_in=None):
        batch = model_input["action_mask"].shape[0]
        logits = self.logits.expand(batch, -1)
        return logits, torch.zeros(batch), memory_in


def _sample(model: _ToyPolicy, action: int, mask: list[float], decision: int, substep: int) -> dict:
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    model_input = {"action_mask": mask_tensor.reshape(1, -1).clone()}
    distribution = _masked_distribution(model.logits.detach().reshape(1, -1), mask_tensor.reshape(1, -1))
    logprob = float(distribution.log_prob(torch.tensor([action])).item())
    return {
        "episode_id": "synthetic",
        "env_step": decision * 2 + substep,
        "decision_index": decision,
        "substep": substep,
        "model_input": model_input,
        "action_mask": mask_tensor,
        "memory_input": torch.zeros(1, 1, 2),
        "action": action,
        "behavior_logprob": logprob,
        "logical_action_logprob": None,
    }


def _trajectory_group(model: _ToyPolicy) -> list[dict]:
    first = [
        _sample(model, 0, [1, 1, 0], 0, 0),
        _sample(model, 2, [1, 0, 1], 0, 1),
        _sample(model, 0, [1, 1, 0], 1, 0),
    ]
    second = [
        _sample(model, 1, [1, 1, 0], 0, 0),
        _sample(model, 1, [1, 1, 0], 1, 0),
    ]
    for samples in (first, second):
        for decision_index in sorted({sample["decision_index"] for sample in samples}):
            decision = [sample for sample in samples if sample["decision_index"] == decision_index]
            logical = sum(sample["behavior_logprob"] for sample in decision)
            for sample in decision:
                sample["logical_action_logprob"] = logical
    return [
        {
            "episode_id": "first",
            "terminal_return": 1.0,
            "decisions": (tuple(first[:2]), (first[2],)),
            "logical_decisions": 2,
            "substeps": 3,
        },
        {
            "episode_id": "second",
            "terminal_return": -1.0,
            "decisions": ((second[0],), (second[1],)),
            "logical_decisions": 2,
            "substeps": 2,
        },
    ]


def _provenance_sample(index: int, *, episode_id: str, done: bool) -> dict:
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    model_input = {
        "action_mask": mask.reshape(1, -1).clone(),
        "feature": torch.tensor([[float(index)]], dtype=torch.float32),
    }
    return {
        "sample_index": index,
        "episode_id": episode_id,
        "env_step": index,
        "decision_index": 0,
        "substep": index,
        "model_input": model_input,
        "action_mask": mask.clone(),
        "memory_input": torch.zeros(1, 1, 2, dtype=torch.float32),
        "action": 0,
        "behavior_logprob": -0.1,
        "logical_action_logprob": -0.1,
        "value": 0.0,
        "reward": 1.0 if done else 0.0,
        "done": done,
    }


def test_flatten_provenance_bundle_assigns_global_sample_order():
    first = _provenance_sample(0, episode_id="first", done=True)
    second = _provenance_sample(0, episode_id="second", done=True)
    trajectories = [
        {
            "episode_id": "first",
            "decisions": ((first,),),
            "provenance_bundle": (first,),
        },
        {
            "episode_id": "second",
            "decisions": ((second,),),
            "provenance_bundle": (second,),
        },
    ]
    flattened = flatten_provenance_bundle(trajectories)
    assert [sample["sample_index"] for sample in flattened] == [0, 1]
    assert [sample["episode_id"] for sample in flattened] == ["first", "second"]


def test_ar019_serializer_passes_existing_candidate_provenance_preflight(tmp_path):
    model = _ToyPolicy()
    model.card_feat = None
    metadata = {"inference_config": {}, "training_config": {}}
    bundle = [
        _provenance_sample(0, episode_id="fixture", done=False),
        _provenance_sample(1, episode_id="fixture", done=True),
    ]
    manifest = build_sample_manifest(
        bundle,
        root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        metadata_date="2026-08-12",
        deck_content_sha256="d" * 64,
        deck_source_file_sha256="e" * 64,
    )
    manifest_path = tmp_path / "sample.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    bundle_path = tmp_path / "trajectory_bundle.pt.gz"
    bundle_hash = save_compressed_bundle(bundle_path, bundle, manifest)
    candidate_path = tmp_path / "candidate.pt"
    save_grpo_candidate_checkpoint(
        candidate_path,
        model,
        metadata,
        root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        sample_manifest_sha256=sha256_file(manifest_path),
        bundle_sha256=bundle_hash,
        sample_manifest_content_sha256=manifest["sha256"],
        config={"algorithm": "trajectory_group_grpo"},
        diagnostics={"loss": 0.0},
    )

    artifacts = validate_candidate_provenance(
        candidate_path,
        approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
    )
    assert artifacts["sample_manifest"] == manifest_path
    assert artifacts["trajectory_bundle"] == bundle_path
    payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
    assert payload["autoresearch"]["artifacts"] == {
        "sample_manifest": "sample.manifest.json",
        "trajectory_bundle": "trajectory_bundle.pt.gz",
    }
    assert payload["autoresearch"]["sample_manifest_content_sha256"] == manifest["sha256"]


def test_relative_terminal_normalization_uses_group_mean_and_population_std():
    advantages, stats = normalize_group_returns([1.0, -1.0, 0.0, 1.0])
    assert stats["zero_variance"] is False
    assert stats["return_mean"] == pytest.approx(0.25)
    assert stats["return_std"] == pytest.approx((0.6875) ** 0.5)
    assert advantages.mean().item() == pytest.approx(0.0, abs=1e-7)


def test_zero_variance_group_fails_closed_to_zero_advantages():
    advantages, stats = normalize_group_returns([1.0, 1.0, 1.0, 1.0])
    assert stats["zero_variance"] is True
    assert torch.equal(advantages, torch.zeros(4))


def test_legal_mask_excludes_illegal_logits_and_rejects_empty_fiber():
    distribution = _masked_distribution(
        torch.tensor([[0.0, 100.0, 1.0]]),
        torch.tensor([[1.0, 0.0, 1.0]]),
    )
    assert distribution.probs[0, 1].item() == 0.0
    assert distribution.probs[0, 0].item() > 0.0
    with pytest.raises(ValueError, match="all-illegal"):
        _masked_distribution(torch.zeros(1, 3), torch.zeros(1, 3))


def test_complete_logprob_sums_substeps_and_equal_snapshots_have_identity_ratio():
    behavior = _ToyPolicy()
    learner = copy.deepcopy(behavior)
    trajectories = _trajectory_group(behavior)
    learner_logprob, behavior_logprob, _decision_map, _substep_map = recompute_logprobs_by_decision(
        learner, trajectories
    )
    expected = torch.tensor(
        [
            sum(sample["behavior_logprob"] for sample in trajectories[0]["decisions"][0]),
            trajectories[0]["decisions"][1][0]["behavior_logprob"],
            trajectories[1]["decisions"][0][0]["behavior_logprob"],
            trajectories[1]["decisions"][1][0]["behavior_logprob"],
        ]
    )
    assert torch.allclose(behavior_logprob, expected)
    assert torch.allclose(learner_logprob, behavior_logprob)
    assert torch.allclose(torch.exp(learner_logprob - behavior_logprob), torch.ones(4))


def test_group_credit_is_shared_by_logical_decisions_and_all_substeps():
    model = _ToyPolicy()
    trajectories = _trajectory_group(model)
    advantages, _stats = normalize_group_returns([1.0, -1.0])
    decision_advantages, substep_advantages, mapping = expand_group_advantages(
        trajectories, advantages
    )
    assert decision_advantages.tolist() == [pytest.approx(advantages[0].item())] * 2 + [
        pytest.approx(advantages[1].item())
    ] * 2
    assert mapping.tolist() == [0, 0, 1, 1]
    assert substep_advantages.tolist() == [advantages[0].item()] * 3 + [advantages[1].item()] * 2


def test_tiny_policy_only_update_mutates_parameters_and_has_finite_metrics():
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = trajectory_group_grpo_update(
        model,
        root,
        _trajectory_group(model),
        learning_rate=1e-2,
    )
    assert metrics["optimizer_steps"] == 1
    assert metrics["value_loss"] == 0.0
    assert metrics["zero_variance_group"] is False
    assert metrics["changed_parameters"] > 0
    assert all(torch.isfinite(value).all() for value in model.parameters())
    assert all(not torch.equal(before[name], value) for name, value in model.state_dict().items())
    for key, value in metrics.items():
        if isinstance(value, float):
            assert torch.isfinite(torch.tensor(value))


def test_zero_variance_update_does_not_mutate_parameters():
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    trajectories = _trajectory_group(model)
    trajectories[1]["terminal_return"] = 1.0
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    metrics = trajectory_group_grpo_update(model, root, trajectories, learning_rate=1e-2)
    assert metrics["zero_variance_group"] is True
    assert metrics["optimizer_steps"] == 0
    assert metrics["changed_parameters"] == 0
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
