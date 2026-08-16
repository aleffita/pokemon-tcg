import json

import numpy as np

from rl.packed_data import (
    PACKED_FORMAT_VERSION,
    PackedArrayStore,
    TRAINER_ORDER_COLUMNS,
    _digest_array,
    _digest_columns,
    sha256_file,
)
from scripts.bc.bc_train_mlx import _build_tbptt_decision_groups, _build_tbptt_plan


def _write_store(root):
    arrays = {
        "episode_id": np.array([10, 10, 11, 11, 11, 12], dtype=np.int64),
        "side": np.array([0, 0, 0, 0, 0, 1], dtype=np.int32),
        "step_id": np.array([1, 1, 1, 2, 2, 1], dtype=np.int32),
        "decision_id": np.array([100, 100, 110, 111, 111, 120], dtype=np.int32),
        "substep": np.array([0, 1, 0, 0, 1, 0], dtype=np.int32),
        "action_mask": np.arange(6 * 3, dtype=np.float32).reshape(6, 3),
        "y": np.arange(6, dtype=np.int32),
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


def test_packed_store_feeds_tbptt_plan_with_masks_and_aux_targets(tmp_path):
    arrays = _write_store(tmp_path)
    store = PackedArrayStore(tmp_path, columns=list(arrays))
    meta = {name: arrays[name] for name in ("episode_id", "side", "step_id")}
    groups = _build_tbptt_decision_groups(meta, len(arrays["episode_id"]))
    plan = _build_tbptt_plan(groups, chunk_size=2, row_budget=4)
    assert len(plan) == 2

    seen = []
    for temporal_batch in plan:
        for chunk in temporal_batch:
            rows = np.concatenate(chunk.decisions)
            fetched = store.read_rows(rows)
            seen.extend(fetched["episode_id"].tolist())
            assert fetched["action_mask"].shape[0] == len(rows)
            assert fetched["aux_valid"].shape == fetched["aux_return"].shape
            np.testing.assert_array_equal(fetched["y"], arrays["y"][rows])
    assert seen == arrays["episode_id"].tolist()
