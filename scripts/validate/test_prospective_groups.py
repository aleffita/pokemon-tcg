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

from rl.prospective_actions import PROSPECTIVE_ACTION_CANDIDATE_VERSION
from rl.prospective_actions import enumerate_prospective_actions
from rl.prospective_input_adapter import (
    ACTION_ATTR_AGGREGATE_VERSION,
    ACTION_ATTR_WIDTH,
    ACTION_SET_FEATURE_VERSION,
    ACTION_SET_FEATURE_WIDTH,
    ACTION_SET_FOURIER_FREQUENCIES,
    ACTION_SET_MOMENT_ORDER,
    BRANCH_FEATURE_LAYOUT,
    BRANCH_FEATURE_LAYOUT_VERSION,
    PROSPECTIVE_INPUT_ADAPTER_VERSION,
    aggregate_action_opt_attr,
    aggregate_action_set_features,
    decode_prospective_action,
    load_real_prospective_planner_index,
)
from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.prospective_schema import (
    MAX_ENCODED_POSITION,
    MAX_ENCODED_VERB,
    MIN_ENCODED_POSITION,
    MIN_ENCODED_VERB,
    PROSPECTIVE_COORD_SCHEMA_VERSION,
    pack_entity_zone_relation,
    prospective_coordinate_schema,
)
from scripts.bc.build_prospective_groups import (
    BRANCH_DTYPE,
    GROUP_DTYPE,
    NODE_DTYPE,
    SCHEMA_VERSION,
    _load_materialized_root_opt_attr,
    _iter_real_decisions,
    _legal_actions,
)


REQUIRED_NODE_FIELDS = {
    "parent_index", "depth", "root_branch_order", "branch_order",
    "sibling_index", "action_index", "has_action_index", "action_offset",
    "action_count", "subselection_count", "action_attr_mean",
    "action_set_features",
    "src_pos", "tgt_pos", "verb", "entity_zone_relation_id",
    "behavior_selected", "has_behavior_logprob", "behavior_logprob",
    "has_reference_logprob", "reference_logprob", "valid",
    "failure_stage_code", "failure_type_code",
    "reward_terminal", "reward_ko", "reward_prize",
    "scalar_reward", "scalar_return", "terminal", "ko_signal",
    "prizes_taken",
}

REQUIRED_BRANCH_FIELDS = {
    "episode_side_index", "step_id", "target_row", "group_id",
    "branch_order", "sibling_index", "action_index", "has_action_index",
    "action_offset", "action_count",
    "action_attr_mean",
    "action_set_features",
    "src_pos", "tgt_pos", "verb", "entity_zone_relation_id",
    "behavior_selected", "has_behavior_logprob", "behavior_logprob",
    "has_reference_logprob", "reference_logprob", "requested_trials",
    "valid_trials", "failed_trials", "valid", "mean_scalar_return",
    "terminal_rate", "win_rate", "ko_rate", "expected_prizes_taken",
}


def test_candidate_enumeration_covers_the_legal_domain() -> None:
    simple = {
        "option": list(range(51)),
        "minCount": 1,
        "maxCount": 1,
    }
    assert enumerate_prospective_actions(simple, max_branches=64) == tuple(
        (index,) for index in range(51)
    )
    bounded = {
        "option": list(range(20)),
        "minCount": 2,
        "maxCount": 2,
    }
    actions = enumerate_prospective_actions(bounded, max_branches=4)
    assert len(actions) == 4
    assert actions[0] == (0, 1)
    assert actions[-1] == (18, 19)
    assert actions == enumerate_prospective_actions(
        bounded, max_branches=4
    )


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
    actions = np.load(sidecar / "prospective_actions.npy", allow_pickle=False)
    groups = np.load(sidecar / "prospective_groups.npy", allow_pickle=False)
    offsets = np.load(
        sidecar / "prospective_group_offsets.npy", allow_pickle=False
    )
    episode_sides = np.load(
        sidecar / "prospective_episode_sides.npy", allow_pickle=False
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["planner_version"] == 2
    assert (
        manifest["action_candidate_version"]
        == PROSPECTIVE_ACTION_CANDIDATE_VERSION
    )
    assert manifest["input_adapter_version"] == PROSPECTIVE_INPUT_ADAPTER_VERSION
    assert (
        manifest["prospective_coord_schema_version"]
        == PROSPECTIVE_COORD_SCHEMA_VERSION
    )
    assert manifest["prospective_coordinate_schema"] == prospective_coordinate_schema()
    action_schema = manifest["action_feature_schema"]
    assert action_schema["aggregate_version"] == ACTION_ATTR_AGGREGATE_VERSION
    assert (
        action_schema["branch_feature_layout_version"]
        == BRANCH_FEATURE_LAYOUT_VERSION
    )
    assert action_schema["field"] == "action_attr_mean"
    assert action_schema["dtype"] == "float32"
    assert action_schema["shape"] == [ACTION_ATTR_WIDTH]
    action_slice = BRANCH_FEATURE_LAYOUT["action_attr_mean"]
    assert isinstance(action_slice, slice)
    assert action_schema["adapter_slots"] == [
        action_slice.start,
        action_slice.stop,
    ]
    assert (
        action_schema["action_set_feature_version"]
        == ACTION_SET_FEATURE_VERSION
    )
    assert action_schema["action_set_field"] == "action_set_features"
    assert action_schema["action_set_shape"] == [ACTION_SET_FEATURE_WIDTH]
    set_slice = BRANCH_FEATURE_LAYOUT["action_set_features"]
    assert isinstance(set_slice, slice)
    assert action_schema["action_set_adapter_slots"] == [
        set_slice.start,
        set_slice.stop,
    ]
    assert action_schema["action_set_moment_order"] == list(
        ACTION_SET_MOMENT_ORDER
    )
    assert action_schema["action_set_fourier_frequencies"] == list(
        ACTION_SET_FOURIER_FREQUENCIES
    )
    assert manifest["rope_nd_axis_order"] == [
        "match_time", "rollout_depth", "branch_action", "entity_zone_relation"
    ]
    assert manifest["config"]["workers"] == 1
    assert int(config["max_episodes"]) == 2
    assert int(config["bc_workers"]) == 1
    assert manifest["source"]["sha256"] == _sha256(replay_zip)
    assert manifest["semantics"]["hidden_opponent_deck_used"] is False
    assert manifest["semantics"]["synthetic_fill_allowed"] is False
    assert "never synthesized" in manifest["semantics"]["unmaterialized_root"]
    assert manifest["audit"]["synthetic_fill_rejections"] == 0
    assert manifest["audit"]["episodes_read"] == 2
    assert manifest["audit"]["candidate_roots"] >= manifest["audit"]["groups_emitted"]
    assert manifest["audit"]["skipped_unmaterialized_roots"] >= 0
    assert manifest["audit"]["sides_with_unmaterialized_roots"] >= 0
    assert manifest["storage"]["version"] == "compact-sharded-v1"
    assert manifest["outputs"]["node_itemsize"] == NODE_DTYPE.itemsize < 512
    assert manifest["outputs"]["branch_itemsize"] == BRANCH_DTYPE.itemsize < 512
    assert groups.dtype == GROUP_DTYPE
    assert actions.dtype == np.dtype("i2")
    flush_groups = int(manifest["storage"]["flush_groups"])
    expected_shards = (
        int(manifest["outputs"]["group_rows"]) + flush_groups - 1
    ) // flush_groups
    assert manifest["audit"]["shards_emitted"] == expected_shards >= 1

    with zipfile.ZipFile(replay_zip) as archive:
        expected_episode_ids = [
            Path(name).stem
            for name in archive.namelist()
            if name.endswith(".json")
        ][:2]
    assert manifest["source"]["episode_ids"] == expected_episode_ids
    assert sorted(set(episode_sides["episode_id"].tolist())) == sorted(
        expected_episode_ids
    )

    assert REQUIRED_NODE_FIELDS <= set(nodes.dtype.names or ())
    assert REQUIRED_BRANCH_FIELDS <= set(branches.dtype.names or ())
    assert len(nodes) == manifest["outputs"]["node_rows"]
    assert len(branches) == manifest["outputs"]["branch_rows"]
    assert len(actions) == manifest["outputs"]["action_rows"]
    assert len(groups) == manifest["outputs"]["group_rows"]
    assert len(offsets) == len(groups) + 1
    assert len(nodes) > 0 and len(branches) > 0
    assert nodes["action_attr_mean"].shape == (len(nodes), ACTION_ATTR_WIDTH)
    assert branches["action_attr_mean"].shape == (
        len(branches),
        ACTION_ATTR_WIDTH,
    )
    assert nodes["action_set_features"].shape == (
        len(nodes),
        ACTION_SET_FEATURE_WIDTH,
    )
    assert branches["action_set_features"].shape == (
        len(branches),
        ACTION_SET_FEATURE_WIDTH,
    )

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
    assert np.all(branches["requested_trials"] == (
        branches["valid_trials"] + branches["failed_trials"]
    ))
    assert np.count_nonzero(branches["valid"]) > 0

    assert np.array_equal(offsets[:-1], groups["node_start"])
    assert int(offsets[-1]) == len(nodes)
    assert np.array_equal(
        groups["node_start"] + groups["node_count"],
        offsets[1:],
    )
    for group in groups:
        start = int(group["node_start"])
        stop = start + int(group["node_count"])
        tree = nodes[start:stop]
        root = tree[tree["depth"] == 1]
        assert len(set(root["root_branch_order"].tolist())) >= 2
        for local_index, node in enumerate(tree):
            parent = int(node["parent_index"])
            assert -1 <= parent < local_index
            assert (parent == -1) == (int(node["depth"]) == 1)
            assert len(decode_prospective_action(actions, node)) == int(
                node["action_count"]
            )

    assert not np.any(nodes["has_behavior_logprob"])
    assert not np.any(nodes["has_reference_logprob"])
    assert np.all(nodes["behavior_logprob"] == 0)
    assert np.all(nodes["reference_logprob"] == 0)
    empty_actions = nodes[nodes["subselection_count"] == 0]
    if len(empty_actions):
        assert np.all(empty_actions["action_attr_mean"] == 0)
        assert np.all(empty_actions["action_set_features"] == 0)
    combined_actions = nodes[nodes["subselection_count"] > 1]
    if len(combined_actions):
        assert np.any(combined_actions["action_set_features"] != 0)

    dataset_dir = sidecar.parent
    metadata = np.load(dataset_dir / "episode_meta.npy", allow_pickle=False)
    materialized_opt_attr = np.load(
        dataset_dir / "opt_attr.npy", allow_pickle=False
    )
    index = load_real_prospective_planner_index(dataset_dir)
    for group_index, group in enumerate(groups):
        decision_key = index.group_target_keys[group_index]
        target_row = int(group["target_row"])
        target = metadata[target_row]
        assert decision_key == (
            str(target["episode_id"]),
            int(target["side"]),
            int(target["step_id"]),
        )
        start = int(group["node_start"])
        stop = start + int(group["node_count"])
        for node in nodes[start:stop]:
            if int(node["depth"]) != 1:
                continue
            action = decode_prospective_action(actions, node)
            expected = aggregate_action_opt_attr(
                materialized_opt_attr[target_row],
                action,
            )
            np.testing.assert_array_equal(node["action_attr_mean"], expected)

    root_lookup = _load_materialized_root_opt_attr(dataset_dir)
    sample_key = index.group_target_keys[0]
    assert np.shares_memory(root_lookup.get(sample_key), root_lookup.values)
    assert not root_lookup.get(sample_key).flags["OWNDATA"]

    encoder = TokenEncoder(get_card_table())
    real_sibling_set_checked = False
    with zipfile.ZipFile(replay_zip) as archive:
        members = [name for name in archive.namelist() if name.endswith(".json")][:2]
        for member in members:
            episode = json.loads(archive.read(member))
            for decision in _iter_real_decisions(
                episode,
                Path(member).stem,
                both_sides=True,
                self_aliases=frozenset(),
            ):
                select = decision.observation.get("select") or {}
                if int(select.get("maxCount", 1) or 0) <= 1:
                    continue
                actions = _legal_actions(select, max_branches=16)
                actions = [action for action in actions if len(action) > 1]
                if len(actions) < 2:
                    continue
                encoded = encoder.encode(
                    decision.observation,
                    picked=set(),
                    self_deck=list(decision.own_deck),
                    tracker=None,
                    ability_slots=None,
                )
                features = [
                    aggregate_action_set_features(
                        encoded["opt_src_pos"],
                        encoded["opt_tgt_pos"],
                        encoded["opt_verb"],
                        action,
                    )
                    for action in actions
                ]
                assert len({feature.tobytes() for feature in features}) == len(actions)
                real_sibling_set_checked = True
                break
            if real_sibling_set_checked:
                break
    assert real_sibling_set_checked

    print(
        "[test-prospective] "
        f"episodes={manifest['audit']['episodes_read']} "
        f"groups={manifest['audit']['groups_emitted']} "
        f"branches={len(branches)} nodes={len(nodes)} PASS"
    )


def main() -> None:
    test_candidate_enumeration_covers_the_legal_domain()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sidecar",
        default="data/bc_data/bc_smoke_2026_07_28/prospective_v2",
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
