import json

import numpy as np
import pytest

from rl.packed_data import (
    PACKED_FORMAT_VERSION,
    PackedArrayStore,
    TRAINER_ORDER_COLUMNS,
    _digest_array,
    _digest_columns,
    required_trainer_columns,
    sha256_file,
    split_episode_ids,
    validate_selection,
)


def test_split_episode_ids_matches_first_appearance_cap():
    episode_ids = np.array([10, 10, 11, 12, 12, 12, 13], dtype=np.int64)
    selected, train, val = split_episode_ids(episode_ids, max_rows=4, val_frac=0.1)
    assert selected.tolist() == [10, 11, 12]
    assert val.tolist() == [10]
    assert train.tolist() == [11, 12]


def test_split_max_rows_zero_is_seed_independent():
    ids = np.array([20, 20, 21, 22, 22, 23], dtype=np.int64)
    first = split_episode_ids(ids, max_rows=0, val_frac=0.25)
    second = split_episode_ids(ids, max_rows=0, val_frac=0.25)
    assert [part.tolist() for part in first] == [part.tolist() for part in second]


def test_required_contract_is_independent_and_includes_masks_aux_and_tbptt():
    inputs = ["action_mask", "self_hand_mask", "state"]
    required = required_trainer_columns(inputs)
    assert required[:3] == ["action_mask", "self_hand_mask", "state"]
    assert required[-19:] == [
        "y", "is_attack", "opt_group", "aux_ko", "aux_prize_delta",
        "aux_terminal", "aux_return", "aux_valid", "episode_id", "side",
        "step_id", "decision_id", "substep", "new_episode", "terminal",
        "reward", "outcome", "is_self", "day_id",
    ]
    with pytest.raises(ValueError, match="legal/action mask"):
        required_trainer_columns(["state"])


def test_packed_store_is_mmap_read_only_and_contract_checked(tmp_path):
    arrays = {
        "episode_id": np.array([10, 10, 11], dtype=np.int64),
        "side": np.array([0, 1, 0], dtype=np.int32),
        "step_id": np.array([1, 1, 2], dtype=np.int32),
        "y": np.array([3, 4, 5], dtype=np.int32),
    }
    columns = list(arrays)
    specs = {}
    digests = {}
    column_dir = tmp_path / "columns"
    column_dir.mkdir()
    for name, array in arrays.items():
        path = column_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        specs[name] = {
            "file": f"columns/{name}.npy",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "nbytes": int(array.nbytes),
            "file_sha256": sha256_file(path),
        }
        digests[name] = _digest_array(array)
    manifest = {
        "format": "fixed-width-npy-mmap",
        "format_version": PACKED_FORMAT_VERSION,
        "source_sha256": "source-digest",
        "selected_rows": 3,
        "logical_bytes": sum(array.nbytes for array in arrays.values()),
        "columns": columns,
        "column_specs": specs,
        "column_digests": digests,
        "selection": {"max_rows": 3, "val_frac": 0.1},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    store = PackedArrayStore(tmp_path, row_start=1, row_stop=3, columns=["y"])
    assert store.array("y").flags.writeable is False
    with pytest.raises(ValueError, match="missing required trainer columns"):
        PackedArrayStore(tmp_path, columns=["y"], required_columns=["opt_group"])
    np.testing.assert_array_equal(
        store.read_rows(np.array([1, 0], dtype=np.int64))["y"], [5, 4]
    )
    validate_selection(
        PackedArrayStore(tmp_path, columns=["episode_id"]),
        source_sha256="source-digest",
        max_rows=3,
        val_frac=0.1,
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_selection(
            PackedArrayStore(tmp_path, columns=["episode_id"]),
            source_sha256="other-source",
            max_rows=3,
            val_frac=0.1,
        )
