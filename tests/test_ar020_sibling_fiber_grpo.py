from __future__ import annotations

import copy

import pytest
import torch

from scripts.rl.sibling_fiber_grpo import (
    _branch_candidates,
    sibling_fiber_grpo_update,
    sibling_fiber_grpo_update_groups,
)
from scripts.rl.trajectory_group_grpo import (
    _masked_distribution,
    recompute_logprobs_by_decision,
)


class _ToyPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1, -0.1], dtype=torch.float32))

    def logits_value(self, model_input, memory_in=None):
        batch = model_input["action_mask"].shape[0]
        return self.logits.expand(batch, -1), torch.zeros(batch), memory_in


def _sample(model: _ToyPolicy, action: int, mask: list[float], decision: int) -> dict:
    mask_tensor = torch.tensor(mask, dtype=torch.float32)
    model_input = {"action_mask": mask_tensor.reshape(1, -1).clone()}
    distribution = _masked_distribution(model.logits.detach().reshape(1, -1), model_input["action_mask"])
    sample = {
        "episode_id": f"fiber-{decision}",
        "env_step": decision,
        "decision_index": decision,
        "substep": 0,
        "model_input": model_input,
        "action_mask": mask_tensor,
        "memory_input": torch.zeros(1, 1, 2),
        "action": action,
        "behavior_logprob": float(distribution.log_prob(torch.tensor([action])).item()),
        "logical_action_logprob": None,
    }
    sample["logical_action_logprob"] = sample["behavior_logprob"]
    return sample


def _trajectory(model: _ToyPolicy, branch_action: int, terminal_return: float) -> dict:
    branch = _sample(model, branch_action, [1.0, 1.0, 1.0, 0.0], 0)
    continuation = _sample(model, 0, [1.0, 0.0, 1.0, 1.0], 1)
    base = {
        "action_mask_sha256": "same-mask",
        "memory_input_sha256": "same-memory",
        "model_input_digests": [{"name": "action_mask", "sha256": "same-input"}],
    }
    return {
        "episode_id": f"fiber-{branch_action}",
        "terminal_return": terminal_return,
        "decisions": ((branch,), (continuation,)),
        "branch": {"decision_index": 0, "substep": 0, "action": branch_action},
        "branch_base": base,
        "logical_decisions": 2,
        "substeps": 2,
    }


def test_dynamic_k_is_capped_by_the_real_legal_fiber() -> None:
    generator = torch.Generator(device="cpu").manual_seed(20)
    two = _branch_candidates(
        torch.tensor([[0.0, 1.0, -1.0, -2.0]]),
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        requested_k=4,
        generator=generator,
    )
    assert len(two) == 2
    assert len(set(two)) == 2


def test_dynamic_k_uniform_mixture_preserves_legal_distinct_fibers() -> None:
    generator = torch.Generator(device="cpu").manual_seed(21)
    fibers = _branch_candidates(
        torch.tensor([[5.0, -5.0, 0.0, -1.0]]),
        torch.tensor([[1.0, 1.0, 1.0, 0.0]]),
        requested_k=4,
        generator=generator,
        uniform_mix=1.0,
    )
    assert len(fibers) == 3
    assert len(set(fibers)) == 3
    assert set(fibers) <= {0, 1, 2}


def test_equal_snapshots_have_identity_logprob_ratio() -> None:
    behavior = _ToyPolicy()
    learner = copy.deepcopy(behavior)
    trajectories = [_trajectory(behavior, 0, 1.0), _trajectory(behavior, 1, -1.0)]
    learner_logprob, behavior_logprob, _mapping, _substeps = recompute_logprobs_by_decision(
        learner, trajectories
    )
    assert torch.allclose(torch.exp(learner_logprob - behavior_logprob), torch.ones(4))


def test_future_credit_updates_branch_and_continuation() -> None:
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    metrics = sibling_fiber_grpo_update(
        model,
        root,
        [_trajectory(model, 0, 1.0), _trajectory(model, 1, -1.0)],
        learning_rate=1e-2,
        credit_scope="branch_and_continuation",
    )
    assert metrics["optimizer_steps"] == 1
    assert metrics["credited_logical_actions"] == 4
    assert metrics["continuation_credit"] is True
    assert metrics["continuation_discount"] == pytest.approx(0.97)
    assert all(torch.isfinite(value).all() for value in model.parameters())


def test_branch_only_remains_available_as_control() -> None:
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    metrics = sibling_fiber_grpo_update(
        model,
        root,
        [_trajectory(model, 0, 1.0), _trajectory(model, 1, -1.0)],
        learning_rate=1e-2,
        credit_scope="branch_only",
    )
    assert metrics["credited_logical_actions"] == 2
    assert metrics["continuation_credit"] is False
    assert metrics["continuation_credit_sum"] == 0.0


def test_grouped_update_normalizes_each_dynamic_base_once() -> None:
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    groups = [
        [_trajectory(model, 0, 1.0), _trajectory(model, 1, -1.0)],
        [_trajectory(model, 2, 1.0), _trajectory(model, 0, -1.0)],
    ]
    metrics = sibling_fiber_grpo_update_groups(
        model,
        root,
        groups,
        learning_rate=1e-2,
        credit_scope="branch_and_continuation",
    )
    assert metrics["optimizer_steps"] == 1
    assert metrics["group_count"] == 2
    assert metrics["group_sizes"] == [2, 2]
    assert metrics["credited_logical_actions"] == 8
    assert metrics["continuation_credit"] is True
    assert all(torch.isfinite(value).all() for value in model.parameters())


def test_grouped_all_zero_variance_emits_root_equivalent_noop() -> None:
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    before = [value.detach().clone() for value in model.parameters()]
    groups = [
        [_trajectory(model, 0, 1.0), _trajectory(model, 1, 1.0)],
        [_trajectory(model, 2, -1.0), _trajectory(model, 0, -1.0)],
    ]
    metrics = sibling_fiber_grpo_update_groups(model, root, groups)
    assert metrics["optimizer_steps"] == 0
    assert metrics["zero_variance_groups"] == 2
    assert metrics["no_update_reason"] == "all_groups_zero_variance"
    assert all(
        torch.equal(before_value, after_value)
        for before_value, after_value in zip(before, model.parameters())
    )
