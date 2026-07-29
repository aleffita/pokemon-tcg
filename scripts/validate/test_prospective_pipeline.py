"""Validate BC -> prospective sidecar integration on the real two-episode smoke corpus."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from rl.encoder.encoding import MAX_OPTIONS
from rl.prospective_input_adapter import (
    ACTION_ATTR_AGGREGATE_VERSION,
    ACTION_SET_FEATURE_VERSION,
    BRANCH_FEATURE_LAYOUT_VERSION,
    PROSPECTIVE_INPUT_ADAPTER_VERSION,
    aggregate_action_opt_attr,
)
from rl.train_config import load_config
from scripts.bc.build_bc_from_zips import _ensure_prospective_sidecar
from scripts.validate.test_prospective_groups import validate


ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "data" / "bc_data" / "bc_smoke_pipeline_2026_07_28"
SIDECAR = DATASET / "prospective_v1"
REPLAY_ZIP = ROOT / "data" / "bc_replay_zip" / "2026-07-28.zip"
CONFIG = ROOT / "configs" / "smoke.json"


def main() -> None:
    validate(SIDECAR, REPLAY_ZIP, CONFIG)
    dataset_manifest = json.loads(
        (DATASET / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    sidecar_manifest = json.loads(
        (SIDECAR / "prospective_manifest.json").read_text(encoding="utf-8")
    )
    prospective = dataset_manifest["prospective"]
    assert prospective["enabled"] is True
    assert prospective["status"] in {"built", "reused"}
    assert prospective["sidecar_name"] == "prospective_v1"
    assert prospective["fingerprint"] == sidecar_manifest["fingerprint"]
    assert prospective["input_adapter_version"] == PROSPECTIVE_INPUT_ADAPTER_VERSION
    assert (
        prospective["action_attr_aggregate_version"]
        == ACTION_ATTR_AGGREGATE_VERSION
    )
    assert (
        prospective["branch_feature_layout_version"]
        == BRANCH_FEATURE_LAYOUT_VERSION
    )
    assert (
        prospective["action_set_feature_version"]
        == ACTION_SET_FEATURE_VERSION
    )
    assert prospective["node_rows"] == sidecar_manifest["outputs"]["node_rows"]
    assert prospective["branch_rows"] == sidecar_manifest["outputs"]["branch_rows"]
    assert dataset_manifest["config"]["prospective_enabled"] is True
    assert dataset_manifest["config"]["prospective_sidecar_name"] == "prospective_v1"
    assert dataset_manifest["config"]["prospective_max_groups"] == 4
    assert dataset_manifest["config"]["prospective_max_branches"] == 4
    assert dataset_manifest["config"]["prospective_trials"] == 1
    assert dataset_manifest["config"]["prospective_horizon"] == 2
    assert dataset_manifest["config"]["prospective_gamma"] == 1.0
    assert not Path(f"{DATASET}_shards").exists()

    nodes = np.load(SIDECAR / "prospective_nodes.npy", allow_pickle=False)
    branches = np.load(SIDECAR / "prospective_branches.npy", allow_pickle=False)
    assert len(nodes) == 24 and int(nodes["valid"].sum()) == 24
    assert len(branches) == 12 and int(branches["valid"].sum()) == 12

    metadata = np.load(DATASET / "episode_meta.npy", allow_pickle=False)
    labels = np.load(DATASET / "__labels__.npy", allow_pickle=False)
    option_attributes = np.load(DATASET / "opt_attr.npy", allow_pickle=False)
    rows_by_decision: dict[tuple[str, int, int], list[int]] = {}
    for row_index, row in enumerate(metadata):
        key = (
            str(row["episode_id"]),
            int(row["side"]),
            int(row["step_id"]),
        )
        rows_by_decision.setdefault(key, []).append(row_index)
    real_multi_select_checked = False
    for row_indices in rows_by_decision.values():
        selected = [
            int(labels[index])
            for index in row_indices
            if 0 <= int(labels[index]) < MAX_OPTIONS
        ]
        if len(selected) < 2 or len(set(selected)) < 2:
            continue
        selected = list(dict.fromkeys(selected))
        source = np.asarray(option_attributes[row_indices[0]], dtype=np.float32)
        combined = aggregate_action_opt_attr(source, selected)
        expected = np.mean(source[selected], axis=0, dtype=np.float32)
        np.testing.assert_array_equal(combined, expected)
        real_multi_select_checked = True
        break
    assert real_multi_select_checked

    disabled_out = DATASET.parent / "bc_smoke_pipeline_disabled_probe"
    assert not disabled_out.exists()
    disabled = replace(load_config(config_path=str(CONFIG)), prospective_enabled=False)
    disabled_contract = _ensure_prospective_sidecar(
        disabled_out,
        cfg=disabled,
        config_source=CONFIG,
        zip_paths=[REPLAY_ZIP],
        sources=[],
    )
    assert disabled_contract["status"] == "disabled"
    assert not disabled_out.exists()

    mismatched = replace(
        load_config(config_path=str(CONFIG)),
        prospective_gamma=0.9,
    )
    try:
        _ensure_prospective_sidecar(
            DATASET,
            cfg=mismatched,
            config_source="configs/smoke.json",
            zip_paths=[REPLAY_ZIP],
            sources=dataset_manifest["sources"],
        )
    except RuntimeError as exc:
        assert "different config contract" in str(exc)
    else:
        raise AssertionError("unsafe prospective resume was not rejected")
    print(
        "[test-prospective-pipeline] "
        "real BC corpus + sidecar manifest + safe whole-sidecar resume PASS"
    )


if __name__ == "__main__":
    main()
