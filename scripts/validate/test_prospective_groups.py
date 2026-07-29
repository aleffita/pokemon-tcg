"""Validate the prospective planner sidecar against the real 2026-07-28 smoke corpus.

No observations or search states are mocked here. The sidecar under validation
must have been built from the two real episodes selected by ``configs/smoke.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from rl.prospective_schema import (
    MAX_ENCODED_POSITION,
    MAX_ENCODED_VERB,
    MIN_ENCODED_POSITION,
    MIN_ENCODED_VERB,
    PROSPECTIVE_COORD_SCHEMA_VERSION,
    pack_entity_zone_relation,
    prospective_coordinate_schema,
)


REQUIRED_NODE_FIELDS = {
    "episode_id", "side", "step_id", "group_id", "branch_id",
    "parent_branch_id", "has_parent", "depth", "trial_index", "trial_seed",
    "determination_id", "root_branch_order", "branch_order", "sibling_index",
    "action_index", "has_action_index", "action_json", "subselection_count",
    "src_pos", "tgt_pos", "verb", "entity_zone_relation_id",
    "behavior_selected", "has_behavior_logprob", "behavior_logprob",
    "has_reference_logprob", "reference_logprob", "valid", "failure_stage",
    "failure_type", "reward_terminal", "reward_ko", "reward_prize",
    "scalar_reward", "scalar_return", "terminal", "ko_signal",
    "prizes_taken", "player_name", "opponent_name", "is_self", "outcome",
}

REQUIRED_BRANCH_FIELDS = {
    "episode_id", "side", "step_id", "group_id", "branch_id",
    "parent_branch_id", "has_parent", "depth", "branch_order",
    "sibling_index", "action_index", "has_action_index", "action_json",
    "src_pos", "tgt_pos", "verb", "entity_zone_relation_id",
    "behavior_selected", "has_behavior_logprob", "behavior_logprob",
    "has_reference_logprob", "reference_logprob", "requested_trials",
    "valid_trials", "failed_trials", "valid", "mean_scalar_return",
    "terminal_rate", "win_rate", "ko_rate", "expected_prizes_taken",
    "player_name", "opponent_name", "is_self", "outcome",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate(sidecar: Path, replay_zip: Path, config_path: Path) -> None:
    manifest = json.loads(
        (sidecar / "prospective_manifest.json").read_text(encoding="utf-8")
    )
    nodes = np.load(sidecar / "prospective_nodes.npy", allow_pickle=False)
    branches = np.load(sidecar / "prospective_branches.npy", allow_pickle=False)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["planner_version"] == 1
    assert (
        manifest["prospective_coord_schema_version"]
        == PROSPECTIVE_COORD_SCHEMA_VERSION
    )
    assert manifest["prospective_coordinate_schema"] == prospective_coordinate_schema()
    assert manifest["rope_nd_axis_order"] == [
        "match_time", "rollout_depth", "branch_action", "entity_zone_relation"
    ]
    assert manifest["config"]["workers"] == 1
    assert int(config["max_episodes"]) == 2
    assert int(config["bc_workers"]) == 1
    assert manifest["source"]["sha256"] == _sha256(replay_zip)
    assert manifest["semantics"]["hidden_opponent_deck_used"] is False
    assert manifest["semantics"]["synthetic_fill_allowed"] is False
    assert manifest["audit"]["synthetic_fill_rejections"] == 0
    assert manifest["audit"]["episodes_read"] == 2

    with zipfile.ZipFile(replay_zip) as archive:
        expected_episode_ids = [
            Path(name).stem
            for name in archive.namelist()
            if name.endswith(".json")
        ][:2]
    assert manifest["source"]["episode_ids"] == expected_episode_ids
    assert sorted(set(nodes["episode_id"].tolist())) == sorted(expected_episode_ids)

    assert REQUIRED_NODE_FIELDS <= set(nodes.dtype.names or ())
    assert REQUIRED_BRANCH_FIELDS <= set(branches.dtype.names or ())
    assert len(nodes) == manifest["outputs"]["node_rows"]
    assert len(branches) == manifest["outputs"]["branch_rows"]
    assert len(nodes) > 0 and len(branches) > 0

    for array in (nodes, branches):
        for field in array.dtype.names or ():
            if np.issubdtype(array.dtype[field], np.floating):
                assert np.isfinite(array[field]).all(), field

    for array in (nodes, branches):
        assert np.all(
            (MIN_ENCODED_POSITION <= array["src_pos"])
            & (array["src_pos"] <= MAX_ENCODED_POSITION)
        )
        assert np.all(
            (MIN_ENCODED_POSITION <= array["tgt_pos"])
            & (array["tgt_pos"] <= MAX_ENCODED_POSITION)
        )
        assert np.all(
            (MIN_ENCODED_VERB <= array["verb"])
            & (array["verb"] <= MAX_ENCODED_VERB)
        )
        expected_relation = pack_entity_zone_relation(
            array["src_pos"], array["tgt_pos"], array["verb"]
        )
        assert np.array_equal(
            array["entity_zone_relation_id"].astype(np.int64),
            expected_relation,
        )

    assert np.all(nodes["depth"] >= 1)
    assert np.all(branches["depth"] == 1)
    assert np.all(nodes["has_parent"] == (nodes["depth"] > 1))
    assert not np.any(branches["has_parent"])
    assert np.all(branches["requested_trials"] == (
        branches["valid_trials"] + branches["failed_trials"]
    ))
    assert np.count_nonzero(branches["valid"]) > 0

    node_ids = set(nodes["branch_id"].tolist())
    for node in nodes[nodes["has_parent"]]:
        assert str(node["parent_branch_id"]) in node_ids

    for group_id in sorted(set(nodes["group_id"].tolist())):
        group = nodes[nodes["group_id"] == group_id]
        root = group[group["depth"] == 1]
        assert len(set(root["root_branch_order"].tolist())) >= 2
        for trial_index in set(root["trial_index"].tolist()):
            trial = root[root["trial_index"] == trial_index]
            assert len(set(trial["trial_seed"].tolist())) == 1
            assert len(set(trial["determination_id"].tolist())) == 1
            assert len(set(trial["determination_mode"].tolist())) == 1
            assert len(set(trial["determination_candidates"].tolist())) == 1

    assert not np.any(nodes["has_behavior_logprob"])
    assert not np.any(nodes["has_reference_logprob"])
    assert np.all(nodes["behavior_logprob"] == 0)
    assert np.all(nodes["reference_logprob"] == 0)
    assert not any(
        "Synthetic" in str(value) for value in nodes["failure_type"].tolist()
    )

    print(
        "[test-prospective] "
        f"episodes={manifest['audit']['episodes_read']} "
        f"groups={manifest['audit']['groups_emitted']} "
        f"branches={len(branches)} nodes={len(nodes)} PASS"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        default="data/bc_data/bc_smoke_2026_07_28/prospective_v1",
    )
    parser.add_argument(
        "--zip",
        dest="replay_zip",
        default="data/bc_replay_zip/2026-07-28.zip",
    )
    parser.add_argument("--config", default="configs/smoke.json")
    args = parser.parse_args()
    validate(Path(args.sidecar), Path(args.replay_zip), Path(args.config))


if __name__ == "__main__":
    main()
