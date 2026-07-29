"""Unit invariants for causal prospective-planner training integration."""
from __future__ import annotations

import ast
from pathlib import Path
import tempfile

import mlx.core as mx
import numpy as np

from rl.prospective_input_adapter import ProspectivePlannerNumpyBatch
from scripts.bc.bc_train_mlx import (
    _prospective_batch_geometry,
    _prospective_context_work_plan,
    _prospective_physical_group_batches,
    _prospective_sibling_policy_terms,
    _prospective_training_group_indices,
    _reconstruct_prospective_trunk_context,
)

ROOT = Path(__file__).resolve().parents[2]


class _RecordingTrunk:
    """Small deterministic trunk double exposing memory and picked leakage."""

    scratch_tokens = 1
    d = 2

    def __init__(self) -> None:
        self.learned_init = mx.zeros((self.scratch_tokens, self.d), mx.float16)
        self.memory_inputs: list[np.ndarray] = []

    def _encode(self, observation, *, memory_in):
        picked = observation["picked"].astype(mx.float16)
        self.memory_inputs.append(np.asarray(memory_in))
        cls_out = mx.broadcast_to(picked[:, None], (len(picked), self.d))
        pooled = cls_out * mx.array(10, dtype=mx.float16)
        memory_out = memory_in + picked[:, None, None] + mx.array(
            1, dtype=mx.float16
        )
        return cls_out, None, pooled, None, memory_out


def _two_decision_multiselect_fixture():
    metadata = np.zeros(
        4,
        dtype=[
            ("episode_id", "U8"),
            ("side", "i4"),
            ("step_id", "i4"),
        ],
    )
    metadata["episode_id"] = "episode"
    metadata["side"] = 0
    metadata["step_id"] = [0, 0, 1, 1]

    nodes = np.zeros(
        2,
        dtype=[
            ("episode_id", "U8"),
            ("side", "i4"),
            ("step_id", "i4"),
        ],
    )
    nodes["episode_id"] = "episode"
    nodes["side"] = 0
    nodes["step_id"] = [0, 1]

    batch = ProspectivePlannerNumpyBatch(
        context=np.zeros((2, 1, 2), dtype=np.float16),
        branch_tokens=np.zeros((2, 1, 2), dtype=np.float16),
        coordinates=np.zeros((2, 2, 4), dtype=np.int32),
        attention_mask=np.zeros((2, 1, 2, 2), dtype=np.float32),
        branch_valid=np.ones((2, 1), dtype=np.bool_),
        parent_index=np.full((2, 1), -1, dtype=np.int32),
        node_index=np.asarray([[0], [1]], dtype=np.int64),
        group_keys=(("g0", 0, "d0"), ("g1", 0, "d1")),
    )
    observations = {
        # The first row of each decision is pre-action. The second row carries
        # a behavior-derived subselection that must never enter its own target
        # context, but must be committed for the next decision's memory.
        "picked": np.asarray([0, 99, 0, 77], dtype=np.float16),
    }
    return metadata, nodes, batch, observations


def test_context_replay_is_linear_and_uses_pre_action_subrow() -> None:
    metadata, nodes, batch, observations = _two_decision_multiselect_fixture()
    target_keys = tuple(
        (
            str(node["episode_id"]),
            int(node["side"]),
            int(node["step_id"]),
        )
        for node in nodes
    )
    work_plan, work_total = _prospective_context_work_plan(
        metadata,
        target_keys,
        row_limit=len(metadata),
    )
    assert work_total == 2
    assert len(work_plan) == 1

    trunk = _RecordingTrunk()
    progress: list[int] = []
    cache_parent = ROOT / "model" / "checkpoint" / "smoke"
    cache_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=cache_parent, prefix=".prospective-context-test-"
    ) as cache_dir:
        destination = np.memmap(
            Path(cache_dir) / "context.fp16",
            mode="w+",
            dtype=np.float16,
            shape=(2, 3, 2),
        )
        context = _reconstruct_prospective_trunk_context(
            trunk,
            observations,
            set(),
            metadata,
            target_keys,
            row_limit=len(metadata),
            work_plan=work_plan,
            context_destination=destination,
            on_decision_complete=progress.append,
        )
        assert isinstance(context, np.memmap)
        context_copy = np.asarray(context).copy()
        del context
        del destination
    context = context_copy
    coordinates, attention_mask = _prospective_batch_geometry(
        batch, context_length=context.shape[1]
    )

    assert progress == [1, 2]
    assert len(trunk.memory_inputs) == 2
    np.testing.assert_array_equal(
        trunk.memory_inputs[0][0], trunk.memory_inputs[0][1]
    )
    np.testing.assert_array_equal(
        trunk.memory_inputs[1][0], trunk.memory_inputs[1][1]
    )
    # Step 0 context uses picked=0 -> memory_out=1, not picked=99 -> 100.
    np.testing.assert_array_equal(context[0, 0], [0, 0])
    np.testing.assert_array_equal(context[0, 1], [0, 0])
    np.testing.assert_array_equal(context[0, 2], [1, 1])
    # The final step-0 subrow commits memory=100 exactly once; step 1's
    # pre-action context therefore observes memory_out=101.
    np.testing.assert_array_equal(trunk.memory_inputs[1][0, 0], [100, 100])
    np.testing.assert_array_equal(context[1, 2], [101, 101])
    assert coordinates.shape == (2, 4, 4)
    assert attention_mask.shape == (2, 1, 4, 4)


def test_prospective_groups_respect_bc_train_split() -> None:
    training_groups = _prospective_training_group_indices(
        np.asarray([0, 2], dtype=np.int64),
        train_stop=2,
    )
    np.testing.assert_array_equal(training_groups, [0])


def test_training_uses_only_bounded_sidecar_materialization() -> None:
    corpus = np.arange(100_003, dtype=np.int64)
    batches = list(_prospective_physical_group_batches(corpus, 8192))
    assert batches[0][0:2] == (0, 8192)
    assert batches[-1][1] == len(corpus)
    assert max(len(indices) for _, _, indices in batches) == 8192
    np.testing.assert_array_equal(
        np.concatenate([indices for _, _, indices in batches]), corpus
    )

    trainer_path = ROOT / "scripts" / "bc" / "bc_train_mlx.py"
    tree = ast.parse(trainer_path.read_text(encoding="utf-8"))
    adapter_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "rl.prospective_input_adapter"
        for alias in node.names
    }
    assert "load_real_prospective_planner_batch" not in adapter_imports
    assert "load_real_prospective_planner_index" in adapter_imports
    assert "materialize_real_prospective_planner_batch" in adapter_imports


def test_policy_terms_compare_only_valid_siblings() -> None:
    scores = mx.array([[3, 1, 2, 10, 8, 7]], dtype=mx.float32)
    returns = mx.array([[5, 1, -2, 2, 9, 4]], dtype=mx.float32)
    # Roots 0/1 are siblings. Nodes 2/3 share parent 0. Nodes 4 and 5 each
    # have no sibling and therefore remain supervised-only.
    parents = mx.array([[-1, -1, 0, 0, 1, 3]], dtype=mx.int32)
    valid = mx.ones((1, 6), dtype=mx.bool_)
    logprobs, advantages, policy_valid = _prospective_sibling_policy_terms(
        scores,
        returns,
        parents,
        valid,
    )
    mx.eval(logprobs, advantages, policy_valid)

    np.testing.assert_array_equal(
        np.asarray(policy_valid), [[True, True, True, True, False, False]]
    )
    expected_root = np.asarray([3, 1], dtype=np.float32)
    expected_root -= np.log(np.exp(expected_root).sum())
    expected_children = np.asarray([2, 10], dtype=np.float32)
    expected_children -= np.log(np.exp(expected_children).sum())
    np.testing.assert_allclose(
        np.asarray(logprobs)[0, :2], expected_root, rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(logprobs)[0, 2:4],
        expected_children,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(advantages)[0],
        [1, -1, -1, 1, 0, 0],
        rtol=2e-6,
        atol=2e-6,
    )


def test_policy_terms_exclude_equal_return_siblings() -> None:
    scores = mx.array([[3, 1, 2, 10]], dtype=mx.float32)
    returns = mx.array([[0, 0, -1, 1]], dtype=mx.float32)
    parents = mx.array([[-1, -1, 0, 0]], dtype=mx.int32)
    valid = mx.ones((1, 4), dtype=mx.bool_)
    _, advantages, policy_valid = _prospective_sibling_policy_terms(
        scores, returns, parents, valid
    )
    mx.eval(advantages, policy_valid)
    np.testing.assert_array_equal(
        np.asarray(policy_valid), [[False, False, True, True]]
    )
    np.testing.assert_allclose(
        np.asarray(advantages), [[0, 0, -1, 1]], rtol=2e-6, atol=2e-6
    )


def main() -> None:
    test_context_replay_is_linear_and_uses_pre_action_subrow()
    print("PASS: linear causal replay and pre-action multi-select context")
    test_prospective_groups_respect_bc_train_split()
    print("PASS: prospective supervision is isolated to the BC train split")
    test_training_uses_only_bounded_sidecar_materialization()
    print("PASS: trainer uses index plus bounded physical materialization")
    test_policy_terms_compare_only_valid_siblings()
    print("PASS: group-relative objective is scoped to valid sibling sets")
    test_policy_terms_exclude_equal_return_siblings()
    print("PASS: equal-return sibling sets do not dilute policy supervision")


if __name__ == "__main__":
    main()
