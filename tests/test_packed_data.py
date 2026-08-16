import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from rl.packed_data import (
    APPROVED_STAGE4_ROOT_SHA256,
    PACKED_BACKEND_NAME,
    PACKED_DEDUP_CONTRACT,
    PACKED_FORMAT_VERSION,
    PACKED_TBPTT_CONTRACT,
    PackedArrayStore,
    TRAINER_ORDER_COLUMNS,
    approved_stage4_root_matches,
    build_resume_identity,
    _digest_array,
    _digest_columns,
    required_trainer_columns,
    sha256_file,
    split_episode_ids,
    source_digest,
    validate_packed_tbptt_compatibility,
    validate_resume_identity,
    validate_selection,
)


def test_split_episode_ids_matches_first_appearance_cap():
    episode_ids = np.array([10, 10, 11, 12, 12, 12, 13], dtype=np.int64)
    selected, train, val = split_episode_ids(episode_ids, max_rows=4, val_frac=0.1)
    assert selected.tolist() == [10, 11, 12]
    assert val.tolist() == [10]
    assert train.tolist() == [11, 12]


def test_split_episode_ids_is_deterministic_without_a_seed_argument():
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


def _write_strict_store(root):
    """Write a complete small v2 store for runtime-contract tests."""
    root.mkdir(parents=True, exist_ok=True)
    n_rows = 6
    arrays = {
        "action_mask": np.arange(n_rows * 2, dtype=np.float32).reshape(n_rows, 2),
        "episode_id": np.array([10, 10, 11, 11, 11, 12], dtype=np.int64),
        "side": np.array([0, 0, 0, 0, 0, 1], dtype=np.int32),
        "step_id": np.array([1, 1, 1, 2, 2, 1], dtype=np.int32),
        "decision_id": np.array([100, 100, 110, 111, 111, 120], dtype=np.int32),
        "substep": np.array([0, 1, 0, 0, 1, 0], dtype=np.int32),
        "y": np.arange(n_rows, dtype=np.int32),
        "is_attack": np.array([True, False, True, False, True, False]),
        "opt_group": np.zeros((n_rows, 2), dtype=np.int32),
        "aux_ko": np.zeros(n_rows, dtype=np.int32),
        "aux_prize_delta": np.zeros(n_rows, dtype=np.float32),
        "aux_terminal": np.zeros(n_rows, dtype=bool),
        "aux_return": np.zeros(n_rows, dtype=np.float32),
        "aux_valid": np.ones(n_rows, dtype=np.int32),
        "new_episode": np.zeros(n_rows, dtype=bool),
        "terminal": np.zeros(n_rows, dtype=bool),
        "reward": np.zeros(n_rows, dtype=np.float32),
        "outcome": np.zeros(n_rows, dtype=np.float32),
        "is_self": np.ones(n_rows, dtype=bool),
        "day_id": np.ones(n_rows, dtype=np.float32),
    }
    columns = required_trainer_columns(["action_mask"])
    assert set(columns) == set(arrays)
    specs, digests = {}, {}
    column_dir = root / "columns"
    column_dir.mkdir()
    for name in columns:
        array = arrays[name]
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
    order = list(TRAINER_ORDER_COLUMNS)
    manifest = {
        "format": PACKED_BACKEND_NAME,
        "format_version": PACKED_FORMAT_VERSION,
        "backend": {
            "name": PACKED_BACKEND_NAME,
            "format": PACKED_BACKEND_NAME,
            "format_version": PACKED_FORMAT_VERSION,
        },
        "source_sha256": "synthetic-source",
        "selected_rows": n_rows,
        "logical_bytes": sum(array.nbytes for array in arrays.values()),
        "columns": columns,
        "column_specs": specs,
        "column_digests": digests,
        "data_digest": _digest_columns(arrays, columns),
        "selection": {
            "max_rows": 0,
            "val_frac": 0.25,
            "selected_episode_ids": [10, 11, 12],
            "val_episode_ids": [10],
            "train_episode_ids": [11, 12],
            "val_rows": 2,
            "train_rows": 4,
        },
        "required_contract": {
            "model_input_columns": ["action_mask"],
            "labels": ["y", "is_attack", "opt_group"],
            "auxiliary": [
                "aux_ko",
                "aux_prize_delta",
                "aux_terminal",
                "aux_return",
                "aux_valid",
            ],
            "columns": columns,
            "order": order,
            "dedup": PACKED_DEDUP_CONTRACT,
            "tbptt": PACKED_TBPTT_CONTRACT,
        },
        "row_order": {
            "columns": order,
            "val_rows": 2,
            "train_rows": 4,
            "val_digest": _digest_columns(
                {name: arrays[name][:2] for name in order}, order
            ),
            "train_digest": _digest_columns(
                {name: arrays[name][2:] for name in order}, order
            ),
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return arrays


def _strict_store(root):
    arrays = _write_strict_store(root)
    columns = required_trainer_columns(["action_mask"])
    return PackedArrayStore(
        root,
        columns=columns,
        required_columns=columns,
        strict_contract=True,
    ), arrays


def test_strict_runtime_opens_all_order_columns_and_validates_split(tmp_path):
    store, arrays = _strict_store(tmp_path)
    assert set(TRAINER_ORDER_COLUMNS).issubset(store.columns)
    assert store.selected_rows == len(arrays["episode_id"])


def test_tampered_boundary_and_order_fail_closed(tmp_path):
    _write_strict_store(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["row_order"]["val_rows"] = 3
    manifest["row_order"]["train_rows"] = 3
    manifest["selection"]["val_rows"] = 3
    manifest["selection"]["train_rows"] = 3
    manifest_path.write_text(json.dumps(manifest))
    columns = required_trainer_columns(["action_mask"])
    with pytest.raises(ValueError, match="packed val rows|row-order digest"):
        PackedArrayStore(
            tmp_path,
            columns=columns,
            required_columns=columns,
            strict_contract=True,
        )

    _write_strict_store(tmp_path / "order")
    order_root = tmp_path / "order"
    side_path = order_root / "columns" / "side.npy"
    side = np.load(side_path, allow_pickle=False).copy()
    side[0] = 1
    np.save(side_path, side, allow_pickle=False)
    order_manifest_path = order_root / "manifest.json"
    order_manifest = json.loads(order_manifest_path.read_text())
    order_manifest["column_specs"]["side"]["file_sha256"] = sha256_file(side_path)
    order_manifest["column_digests"]["side"] = _digest_array(side)
    order_manifest_path.write_text(json.dumps(order_manifest))
    with pytest.raises(ValueError, match="row-order digest"):
        PackedArrayStore(
            order_root,
            columns=columns,
            required_columns=columns,
            strict_contract=True,
        )


def test_required_order_column_missing_is_rejected(tmp_path):
    _write_strict_store(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["columns"].remove("decision_id")
    manifest["required_contract"]["columns"].remove("decision_id")
    manifest_path.write_text(json.dumps(manifest))
    columns = required_trainer_columns(["action_mask"])
    with pytest.raises(ValueError, match="missing required trainer columns"):
        PackedArrayStore(
            tmp_path,
            columns=columns,
            required_columns=columns,
            strict_contract=True,
        )


@pytest.mark.parametrize("field", ["dedup", "tbptt"])
def test_dedup_and_tbptt_metadata_are_required(tmp_path, field):
    _write_strict_store(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["required_contract"][field]
    manifest_path.write_text(json.dumps(manifest))
    columns = required_trainer_columns(["action_mask"])
    with pytest.raises(ValueError, match="metadata|contract"):
        PackedArrayStore(
            tmp_path,
            columns=columns,
            required_columns=columns,
            strict_contract=True,
        )


def test_inverted_source_order_is_rejected(tmp_path):
    source_a = tmp_path / "a.parquet"
    source_b = tmp_path / "b.parquet"
    source_a.write_bytes(b"source-a")
    source_b.write_bytes(b"source-b")
    store_root = tmp_path / "store"
    _write_strict_store(store_root)
    manifest_path = store_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["source_sha256"] = source_digest([source_a, source_b])
    manifest_path.write_text(json.dumps(manifest))
    store = PackedArrayStore(store_root, columns=["action_mask"])
    validate_selection(
        store,
        source_sha256=source_digest([source_a, source_b]),
        max_rows=0,
        val_frac=0.25,
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        validate_selection(
            store,
            source_sha256=source_digest([source_b, source_a]),
            max_rows=0,
            val_frac=0.25,
        )


def test_split_membership_is_deterministic_without_a_seed_argument():
    ids = np.array([20, 20, 21, 22, 22, 23], dtype=np.int64)
    first = split_episode_ids(ids, max_rows=0, val_frac=0.25)
    second = split_episode_ids(ids, max_rows=0, val_frac=0.25)
    assert [part.tolist() for part in first] == [part.tolist() for part in second]


def _resume_identity(seed=7):
    return build_resume_identity(
        source_sha256="source",
        selection={"max_rows": 0, "selected_episode_ids": [1, 2]},
        split={"val_rows": 2, "train_rows": 4, "val_episode_ids": [1], "train_episode_ids": [2]},
        backend={"name": PACKED_BACKEND_NAME, "data_digest": "packed"},
        seed=seed,
        dedup=True,
        tbptt_chunk=16,
    )


@pytest.mark.parametrize("field", ["source", "selection", "split", "seed", "dedup", "tbptt", "backend"])
def test_resume_identity_rejects_each_identity_mismatch(field):
    identity = _resume_identity()
    changed = copy.deepcopy(identity)
    if field == "source":
        changed["source"]["sha256"] = "other-source"
    elif field == "selection":
        changed["selection"]["max_rows"] = 1
    elif field == "split":
        changed["split"]["val_rows"] = 3
    elif field == "seed":
        changed["trainer"]["seed"] = 8
    elif field == "dedup":
        changed["trainer"]["dedup"] = False
    elif field == "tbptt":
        changed["trainer"]["tbptt_chunk"] = 8
    elif field == "backend":
        changed["backend"]["data_digest"] = "other-packed"
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity(identity, changed, packed=True)


def test_resume_identity_rejects_partial_identity():
    identity = _resume_identity()
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity({"version": 1, "source": identity["source"]}, identity, packed=False)


def test_resume_identity_rejects_legacy_without_production_artifact_policy(tmp_path):
    identity = _resume_identity()
    non_root = tmp_path / "stage4_root.pkl"
    non_root.write_bytes(b"not-the-approved-root")
    with pytest.raises(ValueError, match="explicit Stage 4 warm-start"):
        validate_resume_identity(
            None,
            identity,
            packed=False,
            resume_path=non_root,
            optimizer_state="reset",
            scheduler_state="reset",
        )


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("canonical-relative", True),
        ("canonical-absolute", True),
        ("regular-copy", False),
        ("hardlink", False),
        ("symlink", False),
    ],
)
def test_approved_stage4_root_accepts_only_canonical_path_and_sha256(
    tmp_path, monkeypatch, case, expected
):
    """Relative candidates are interpreted against the verified process cwd."""
    root = Path(__file__).resolve().parents[1] / "experiments/autoresearch/root/stage4_root.pkl"
    project_root = root.parents[3]
    assert root.is_file(), "the frozen Stage 4 root must be present for this policy test"
    if case == "canonical-relative":
        monkeypatch.chdir(project_root)
        candidate = Path("experiments/autoresearch/root/stage4_root.pkl")
    elif case == "canonical-absolute":
        candidate = root
    elif case == "regular-copy":
        candidate = tmp_path / "stage4_root-copy.pkl"
        shutil.copyfile(root, candidate)
    elif case == "hardlink":
        candidate = tmp_path / "stage4_root-hardlink.pkl"
        candidate.hardlink_to(root)
    else:
        candidate = tmp_path / "stage4_root-symlink.pkl"
        candidate.symlink_to(root)
    assert approved_stage4_root_matches(candidate, repo_root=project_root) is expected
    assert APPROVED_STAGE4_ROOT_SHA256 == "b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b"


def test_approved_stage4_root_rejects_project_relative_candidate_from_wrong_cwd(
    tmp_path, monkeypatch
):
    root = Path(__file__).resolve().parents[1] / "experiments/autoresearch/root/stage4_root.pkl"
    project_root = root.parents[3]
    monkeypatch.chdir(tmp_path)
    candidate = Path("experiments/autoresearch/root/stage4_root.pkl")
    assert approved_stage4_root_matches(candidate, repo_root=project_root) is False


def test_resume_identity_rejects_mismatch_and_handles_legacy_policy():
    identity = _resume_identity()
    assert validate_resume_identity(identity, identity, packed=True) == "validated"
    changed = copy.deepcopy(identity)
    changed["backend"] = {"name": PACKED_BACKEND_NAME, "data_digest": "other"}
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_resume_identity(identity, changed, packed=True)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        validate_resume_identity(None, identity, packed=True)
    with pytest.raises(ValueError, match="legacy checkpoint"):
        validate_resume_identity(
            None,
            identity,
            packed=True,
            resume_path=Path("experiments/autoresearch/root/stage4_root.pkl"),
            optimizer_state="reset",
            scheduler_state="reset",
        )
    with pytest.raises(ValueError, match="only an explicit Stage 4 warm-start"):
        validate_resume_identity(None, identity, packed=False)


def test_packed_without_tbptt_is_rejected(tmp_path):
    store, _ = _strict_store(tmp_path)
    with pytest.raises(ValueError, match="no Parquet fallback"):
        validate_packed_tbptt_compatibility(store, 0)
    validate_packed_tbptt_compatibility(store, 16)


def test_resume_identity_seed_variants_are_distinct_but_keep_split_identity():
    first = _resume_identity(seed=7)
    second = _resume_identity(seed=8)
    assert first["selection"] == second["selection"]
    assert first["split"] == second["split"]
    assert first["trainer"]["seed"] != second["trainer"]["seed"]
    assert first != second
