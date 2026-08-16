import json

import mlx.core as mx
import numpy as np

from rl.packed_data import (
    PACKED_FORMAT_VERSION,
    PackedArrayStore,
    TRAINER_ORDER_COLUMNS,
    _digest_array,
    _digest_columns,
    sha256_file,
)
from scripts.bc.bc_train_mlx import (
    _batched_sequential_tbptt_loss,
    _build_tbptt_decision_groups,
    _build_tbptt_plan,
    _load_temporal_batch,
)


def _write_store(root):
    arrays = {
        "episode_id": np.array([10, 10, 11, 11, 11, 12], dtype=np.int64),
        "side": np.array([0, 0, 0, 0, 0, 1], dtype=np.int32),
        "step_id": np.array([1, 1, 1, 2, 2, 1], dtype=np.int32),
        "decision_id": np.array([100, 100, 110, 111, 111, 120], dtype=np.int32),
        "substep": np.array([0, 1, 0, 0, 1, 0], dtype=np.int32),
        "action_mask": np.arange(6 * 3, dtype=np.float32).reshape(6, 3),
        "y": np.array([1, 0, 2, 1, 0, 1], dtype=np.int32),
        "opt_group": np.array(
            [[1, 0, 2], [0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1], [2, 1, 0]],
            dtype=np.int32,
        ),
        "is_attack": np.array([True, False, True, False, True, False]),
        "aux_ko": np.array([0, 1, 0, 1, 0, 1], dtype=np.float32),
        "aux_prize_delta": np.arange(6, dtype=np.float32),
        "aux_terminal": np.zeros(6, dtype=np.float32),
        "aux_return": np.ones(6, dtype=np.float32),
        "aux_valid": np.ones(6, dtype=np.float32),
    }
    specs, digests = {}, {}
    columns_dir = root / "columns"
    columns_dir.mkdir()
    for name, array in arrays.items():
        path = columns_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        specs[name] = {
            "file": f"columns/{name}.npy", "dtype": str(array.dtype),
            "shape": list(array.shape), "nbytes": int(array.nbytes),
            "file_sha256": sha256_file(path),
        }
        digests[name] = _digest_array(array)
    order = list(TRAINER_ORDER_COLUMNS)
    manifest = {
        "format": "fixed-width-npy-mmap", "format_version": PACKED_FORMAT_VERSION,
        "source_sha256": "synthetic", "selected_rows": 6,
        "columns": list(arrays), "column_specs": specs, "column_digests": digests,
        "data_digest": _digest_columns(arrays, list(arrays)),
        "selection": {
            "max_rows": 0,
            "val_frac": 0.25,
            "selected_episode_ids": [10, 11, 12],
            "val_episode_ids": [10],
            "train_episode_ids": [11, 12],
            "val_rows": 2,
            "train_rows": 4,
        },
        "row_order": {
            "columns": order, "val_rows": 2, "train_rows": 4,
            "val_digest": _digest_columns({name: arrays[name][:2] for name in order}, order),
            "train_digest": _digest_columns({name: arrays[name][2:] for name in order}, order),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return arrays


class _SequentialProbeModel:
    scratch_tokens = 1
    d = 1
    learned_init = mx.zeros((1, 1, 1), dtype=mx.float32)

    def logits_value(self, observations, *, memory_in):
        rows = int(observations["action_mask"].shape[0])
        return (
            mx.zeros((rows, 3), dtype=mx.float32),
            mx.zeros((rows, 1), dtype=mx.float32),
            memory_in + mx.ones_like(memory_in),
        )


def test_packed_production_temporal_loader_relabels_and_consumes_tbptt_sequentially(
    tmp_path,
):
    arrays = _write_store(tmp_path)
    store = PackedArrayStore(tmp_path, columns=list(arrays))
    meta = {name: arrays[name] for name in ("episode_id", "side", "step_id")}
    groups = _build_tbptt_decision_groups(meta, len(arrays["episode_id"]))
    plan = _build_tbptt_plan(groups, chunk_size=1, row_budget=3)
    assert len(plan) == 3

    seen_rows = []
    seen_labels = []
    memories = {}
    model = _SequentialProbeModel()
    for temporal_batch in plan:
        loaded = _load_temporal_batch(
            temporal_batch,
            keys=["action_mask"],
            int_keys=set(),
            aux_active=False,
            source="tbptt-cache",
            cache_backend=(
                store,
                np.zeros(len(arrays["y"]), dtype=np.int32),
                np.zeros(len(arrays["y"]), dtype=np.int32),
                np.arange(len(arrays["y"]), dtype=np.int32),
            ),
            emit_fetched=True,
        )
        lane_observations, lane_labels, decision_lengths, lane_rows, _, fetched = loaded
        lane_memories = [
            None if chunk.is_new_group else memories[chunk.group_index]
            for chunk in temporal_batch
        ]
        _, memory_out = _batched_sequential_tbptt_loss(
            model,
            lane_observations,
            lane_labels,
            decision_lengths,
            lane_memories,
        )
        mx.eval(memory_out)
        for lane_index, chunk in enumerate(temporal_batch):
            memories[chunk.group_index] = memory_out[lane_index : lane_index + 1]
            rows = lane_rows[lane_index]
            seen_rows.extend(rows.tolist())
            seen_labels.extend(np.asarray(lane_labels[lane_index]).tolist())
            assert fetched[lane_index]["action_mask"].shape[0] == len(rows)
            np.testing.assert_array_equal(
                fetched[lane_index]["y"], np.array([0, 0, 1, 2, 0, 1])[rows]
            )
    assert seen_rows == [0, 1, 2, 5, 3, 4]
    assert seen_labels == [0, 0, 1, 1, 2, 0]
    assert sorted(int(np.asarray(value).reshape(-1)[0]) for value in memories.values()) == [1, 1, 2]
