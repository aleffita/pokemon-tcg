"""Validate runtime prospective search using only real smoke replay states."""
from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.prospective_input_adapter import (
    decode_prospective_action,
    load_real_prospective_planner_index,
)
from rl.prospective_runtime import (
    ProspectiveRuntimeConfig,
    build_runtime_prospective_tree,
    enumerate_legal_action_combinations,
)
from rl.prospective_schema import (
    ENTITY_ZONE_RELATION_AXIS,
)
from scripts.bc.build_prospective_groups import _iter_real_decisions
from scripts.validate.test_prospective_planner_torch import (
    _real_sidecar_batch,
)


ROOT = Path(__file__).resolve().parents[2]
REPLAY_ZIP = ROOT / "data" / "bc_replay_zip" / "2026-07-28.zip"
SMOKE_DATA = ROOT / "data" / "bc_data" / "bc_smoke_2026_07_28"
SIDECAR = SMOKE_DATA / "prospective_v2"
SMOKE_CONFIG = ROOT / "configs" / "smoke.json"


def _real_decisions():
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    aliases = frozenset(config["bc_self_aliases"])
    decisions = []
    with zipfile.ZipFile(REPLAY_ZIP) as archive:
        names = [name for name in archive.namelist() if name.endswith(".json")][
            : int(config["max_episodes"])
        ]
        for name in names:
            episode = json.loads(archive.read(name))
            decisions.extend(
                _iter_real_decisions(
                    episode,
                    Path(name).stem,
                    both_sides=bool(config["bc_both_sides"]),
                    self_aliases=aliases,
                )
            )
    return decisions


def test_runtime_rebuilds_real_sidecar_root_contract() -> None:
    config = json.loads(SMOKE_CONFIG.read_text(encoding="utf-8"))
    sidecar_index = load_real_prospective_planner_index(SMOKE_DATA)
    first_group = sidecar_index.groups[0]
    start = int(first_group["node_start"])
    stop = start + int(first_group["node_count"])
    group_nodes = sidecar_index.nodes[start:stop]
    roots = group_nodes[group_nodes["depth"] == 1]
    roots = np.sort(roots, order="root_branch_order")
    target_key = sidecar_index.group_target_keys[0]
    decision = next(
        item
        for item in _real_decisions()
        if (
            item.episode_id == target_key[0]
            and item.side == target_key[1]
            and item.step_id == target_key[2]
        )
    )

    encoder = TokenEncoder(get_card_table())
    planner_config, planner_sidecar_batch = _real_sidecar_batch()
    planner_group_index = 0
    runtime = ProspectiveRuntimeConfig(
        trials=int(config["prospective_trials"]),
        horizon=int(config["prospective_horizon"]),
        max_branches=int(config["prospective_max_branches"]),
        gamma=float(config["prospective_gamma"]),
        seed=int(config["seed"]),
    )
    tree = build_runtime_prospective_tree(
        decision.observation,
        list(decision.own_deck),
        encoder,
        planner_sidecar_batch.context[planner_group_index],
        planner_config=planner_config,
        runtime_config=runtime,
        match_time=decision.step_id,
    )
    assert tree is not None
    assert tree.audit["synthetic_fill_rejections"] == 0
    assert tree.audit["determination_failures"] == 0
    assert tree.audit["search_failures"] == 0
    assert tree.audit["trials_used"] == runtime.trials

    sidecar_actions = tuple(
        decode_prospective_action(sidecar_index.actions, row) for row in roots
    )
    assert tree.root_actions == sidecar_actions
    runtime_roots = [
        node
        for node in tree.nodes[0]
        if int(node["depth"]) == 1
    ]
    assert len(runtime_roots) == len(roots)
    for runtime_root, sidecar_root in zip(runtime_roots, roots):
        assert runtime_root["root_branch_order"] == int(
            sidecar_root["root_branch_order"]
        )
        assert runtime_root["entity_zone_relation_id"] == int(
            sidecar_root["entity_zone_relation_id"]
        )
        np.testing.assert_allclose(
            runtime_root["action_attr_mean"],
            sidecar_root["action_attr_mean"],
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            runtime_root["action_set_features"],
            sidecar_root["action_set_features"],
            rtol=0.0,
            atol=0.0,
        )
        local_index = tree.nodes[0].index(runtime_root)
        context_length = tree.batch.context.shape[1]
        assert (
            tree.batch.coordinates[
                0,
                context_length + local_index,
                ENTITY_ZONE_RELATION_AXIS,
            ]
            == int(sidecar_root["entity_zone_relation_id"])
        )
    print(
        "  PASS: runtime rebuilt real sidecar roots with no hidden/synthetic fill"
    )


def test_real_multiselect_combination_enumeration() -> None:
    decision = next(
        item
        for item in _real_decisions()
        if int(item.observation["select"].get("maxCount", 1) or 0) > 1
    )
    select = decision.observation["select"]
    actions = enumerate_legal_action_combinations(select, max_branches=4)
    assert len(actions) == 4
    option_count = len(select["option"])
    minimum = int(select.get("minCount", 1) or 0)
    maximum = int(select.get("maxCount", 1) or 0)
    for action in actions:
        assert minimum <= len(action) <= maximum
        assert len(set(action)) == len(action)
        assert all(0 <= index < option_count for index in action)
    assert any(len(action) != 1 for action in actions)
    print("  PASS: real multi-select decision enumerated as legal combinations")


def main() -> None:
    print("=== Prospective runtime validation ===")
    test_runtime_rebuilds_real_sidecar_root_contract()
    test_real_multiselect_combination_enumeration()
    print("ALL PASSED")


if __name__ == "__main__":
    main()


__all__ = [
    "test_real_multiselect_combination_enumeration",
    "test_runtime_rebuilds_real_sidecar_root_contract",
]
