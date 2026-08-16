import json

import numpy as np
import pytest

from rl.packed_data import (
    PACKED_FORMAT_VERSION,
    PackedArrayStore,
    _digest_array,
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
        "selection": {"max_rows": 3, "val_frac": 0.1, "seed": 7},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    store = PackedArrayStore(tmp_path, row_start=1, row_stop=3, columns=["y"])
    assert store.array("y").flags.writeable is False
    np.testing.assert_array_equal(
        store.read_rows(np.array([1, 0], dtype=np.int64))["y"], [5, 4]
    )
    validate_selection(
        PackedArrayStore(tmp_path, columns=["episode_id"]),
        source_sha256="source-digest",
        max_rows=3,
        val_frac=0.1,
        seed=7,
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_selection(
            PackedArrayStore(tmp_path, columns=["episode_id"]),
            source_sha256="other-source",
            max_rows=3,
            val_frac=0.1,
            seed=7,
        )
