"""Validate BC -> prospective sidecar integration on the real two-episode smoke corpus."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

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
