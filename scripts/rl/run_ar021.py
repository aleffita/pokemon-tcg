"""Run AR-021: grouped dynamic-K prospective sibling-fiber GRPO."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from scripts.rl.ppo_micro_update import (
    build_sample_manifest,
    save_compressed_bundle,
    validate_bundle,
    validate_candidate_provenance,
)
from scripts.rl.sibling_fiber_grpo import (
    DEFAULT_OUTPUT as AR020_OUTPUT,
    SIBLING_FORMAT,
    collect_sibling_fiber_group,
    load_external_opponent,
    sibling_fiber_grpo_update_groups,
)
from scripts.rl.trajectory_group_grpo import (
    _git_commit,
    flatten_provenance_bundle,
    normalize_group_returns,
    save_grpo_candidate_checkpoint,
)
from scripts.rl.trajectory_probe import (
    APPROVED_STAGE4_ROOT_SHA256,
    DateBoundEncoder,
    deck_content_sha256,
    load_deck,
    load_stage4,
    sha256_file,
    validate_meta_date,
)


EXPERIMENT = "AR-021"
DEFAULT_OUTPUT = AR020_OUTPUT.parent / EXPERIMENT


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "sample_index": item["sample_index"],
            "episode_id": item["episode_id"],
            "env_step": item["env_step"],
            "legal_action_mask_digest": item["action_mask_digest"],
            "memory_input_digest": item["memory_input_digest"],
            "model_input_digests": item["model_input_digests"],
            "done": item["done"],
        }
        for item in manifest["order"]
    ]


def _write_report(
    output_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    experiment: str,
) -> None:
    report = f"""# {experiment} - grouped dynamic-K sibling-fiber GRPO

Captured on {manifest['captured_at']} from frozen Stage 4 root `{manifest['root_sha256']}`.

## Result

The collector created multiple exact recurrent sibling bases per matchup.
Each base selected its own effective K from the legal action set, each matchup
was normalized independently, and all groups were combined in one FP32
policy-only optimizer step with terminal credit through discounted future
logical decisions. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | {manifest['group_count']} / {manifest['group_size']} |
| Effective K per base | `{manifest['effective_group_sizes']}` |
| Logical decisions / substeps | {manifest['logical_decisions']} / {manifest['substeps']} |
| Collection seconds / decisions/s | {manifest['collection_seconds']} / {manifest['collection_decisions_per_second']} |
| One grouped optimizer step | {metrics['optimizer_steps']} |
| Update seconds | {metrics['update_seconds']} |
| Loss / gradient norm | {metrics['loss']} / {metrics['gradient_norm']} |
| Candidate parameter L2 delta | {metrics['parameter_l2']} |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses one optimizer step over all groups, while each group's
  sibling-relative credit remains separate.
- All rollouts run to terminal completion and continuation credit uses discount
  `{metrics['continuation_discount']}` without duplicating conditional substeps.
- Candidate preflight passed: `{manifest['candidate_preflight']['passed']}`.

## Limitations and next gate

This is a bounded grouped prospective update, not a strength estimate. The
collection remains serial and the recurrent learner boundary is detached. Run
the controlled same-deck candidate-vs-root gate and the multi-opponent panel
before interpreting or promoting the candidate.

## Provenance

- Code commit at execution: `{manifest['code_commit']}`
- Candidate SHA-256: `{manifest['candidate_sha256']}`
- Sample manifest SHA-256: `{manifest['sample_manifest_sha256']}`
- Trajectory bundle SHA-256: `{manifest['bundle_sha256']}`
- Tournament gate: pending
"""
    (output_dir / "report.md").write_text(report)


def _write_capsule(
    root: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    experiment: str,
) -> None:
    capsule = f"""# State Capsule {experiment[-3:]} - grouped dynamic-K sibling-fiber GRPO

Captured {manifest['captured_at']}.

## Current state

- Frozen Stage 4 root remains fallback: `{manifest['root_sha256']}`.
- AR-021 collected `{manifest['group_count']}` exact recurrent sibling groups
  and `{manifest['group_size']}` fibers with effective K
  `{manifest['effective_group_sizes']}`.
- One grouped FP32 policy-only update applied independent group-relative
  terminal credit through future continuation with discount
  `{metrics['continuation_discount']}`.
- Candidate: `{manifest['candidate_sha256']}`; preflight passed.
- Tournament is pending; no promotion, RoPE-ND, MoE, or historical
  ETL/Parquet/packed-data path was run.

## Evidence

- `experiments/autoresearch/{experiment}/report.md`
- `experiments/autoresearch/{experiment}/manifest.json`
- `experiments/autoresearch/{experiment}/metrics.json`
- `experiments/autoresearch/{experiment}/sample.manifest.json`
- `experiments/autoresearch/{experiment}/trajectory_bundle.pt.gz`
- `experiments/autoresearch/{experiment}/candidate.pt`

## Metrics

- Collection: `{manifest['collection_seconds']}` s,
  `{manifest['collection_decisions_per_second']}` decisions/s.
- Update: `{metrics['update_seconds']}` s; one optimizer step.
- Credited logical actions: `{metrics['credited_logical_actions']}`.
- Parameter L2 delta: `{metrics['parameter_l2']}`;
  gradient norm `{metrics['gradient_norm']}`.

## Next control point

Run the controlled same-deck candidate-vs-root and multi-opponent panel gate.
Keep the root fallback unless grouped sibling-fiber evidence wins that gate.
"""
    (root / f"STATE_CAPSULE_{experiment[-3:]}.md").write_text(capsule)


def run_ar021(
    *,
    checkpoint: Path,
    deck_path: Path,
    meta_date: str,
    output_dir: Path = DEFAULT_OUTPUT,
    opponent_deck_paths: list[Path] | None = None,
    opponent_agent_paths: list[Path] | None = None,
    groups_per_matchup: int = 2,
    k_max: int = 4,
    seed: int = 21021,
    experiment: str = EXPERIMENT,
) -> dict[str, Any]:
    if groups_per_matchup < 1:
        raise ValueError("--groups-per-matchup must be at least one")
    if k_max < 2:
        raise ValueError("--k-max must be at least two")
    validate_meta_date(meta_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    deck = load_deck(deck_path)
    card_table = get_card_table()
    model, metadata = load_stage4(checkpoint, card_table)
    model = model.float()
    root_reference = copy.deepcopy(model).eval()
    encoder = DateBoundEncoder(TokenEncoder(card_table), meta_date)
    root_hash = sha256_file(checkpoint)
    deck_hash = deck_content_sha256(deck)
    deck_source_hash = sha256_file(deck_path)
    opponent_paths = opponent_deck_paths or [deck_path]
    if opponent_agent_paths is not None and len(opponent_agent_paths) != len(opponent_paths):
        raise ValueError("--opponent-agent must be repeated once per --opponent-deck")
    agent_paths = opponent_agent_paths or [None] * len(opponent_paths)
    grouped_trajectories: list[list[dict[str, Any]]] = []
    collections: list[dict[str, Any]] = []
    for matchup_index, opponent_path in enumerate(opponent_paths):
        opponent_deck = load_deck(opponent_path)
        opponent_hash = deck_content_sha256(opponent_deck)
        opponent_source_hash = sha256_file(opponent_path)
        for group_index in range(groups_per_matchup):
            group_seed = seed + matchup_index * 100000 + group_index * 1000
            episode_prefix = f"ar021-m{matchup_index}-g{group_index}"
            opponent_agent_path = agent_paths[matchup_index]
            opponent_factory = (
                None
                if opponent_agent_path is None
                else lambda path=opponent_agent_path: load_external_opponent(path)
            )
            opponent_mode = (
                "current_vs_external_policy_true_recurrent"
                if opponent_agent_path is not None
                else "current_vs_current_true_recurrent"
            )
            trajectories, collection = collect_sibling_fiber_group(
                model=model,
                encoder=encoder,
                deck=deck,
                deck_content_hash=deck_hash,
                deck_source_file_hash=deck_source_hash,
                model_hash=root_hash,
                opponent_deck=opponent_deck,
                opponent_deck_content_hash=opponent_hash,
                opponent_deck_source_file_hash=opponent_source_hash,
                games=k_max,
                seed=group_seed,
                episode_prefix=episode_prefix,
                opponent_factory=opponent_factory,
                opponent_agent_path=(
                    str(opponent_agent_path) if opponent_agent_path is not None else None
                ),
                opponent_mode=opponent_mode,
            )
            collection.update(
                {
                    "matchup_index": matchup_index,
                    "group_index": group_index,
                    "group_id": episode_prefix,
                "opponent_deck": str(opponent_path),
                "opponent_agent_path": (
                    str(opponent_agent_path) if opponent_agent_path is not None else None
                ),
                    "requested_seed": group_seed,
                }
            )
            grouped_trajectories.append(trajectories)
            collections.append(collection)

    all_trajectories = [item for group in grouped_trajectories for item in group]
    provenance_bundle = flatten_provenance_bundle(all_trajectories)
    sample_manifest = build_sample_manifest(
        provenance_bundle,
        root_sha256=root_hash,
        metadata_date=meta_date,
        deck_content_sha256=deck_hash,
        deck_source_file_sha256=deck_source_hash,
    )
    validate_bundle(provenance_bundle, _manifest_rows(sample_manifest))
    sample_manifest_path = output_dir / "sample.manifest.json"
    trajectory_bundle_path = output_dir / "trajectory_bundle.pt.gz"
    _write_json(sample_manifest_path, sample_manifest)
    bundle_hash = save_compressed_bundle(trajectory_bundle_path, provenance_bundle, sample_manifest)
    sample_manifest_file_hash = sha256_file(sample_manifest_path)

    metrics = sibling_fiber_grpo_update_groups(
        model,
        root_reference,
        grouped_trajectories,
        clip_epsilon=0.2,
        learning_rate=1e-5,
        credit_scope="branch_and_continuation",
        continuation_discount=0.97,
    )
    config = {
        "algorithm": "sibling_fiber_grpo_grouped",
        "group_size_cap": k_max,
        "groups_per_matchup": groups_per_matchup,
        "matchup_count": len(opponent_paths),
        "opponent_agent_paths": [
            str(path) if path is not None else None for path in agent_paths
        ],
        "group_count": len(grouped_trajectories),
        "effective_group_sizes": [collection["games"] for collection in collections],
        "clip_epsilon": 0.2,
        "learning_rate": 1e-5,
        "advantage_epsilon": 1e-8,
        "precision": "FP32",
        "value_loss": 0.0,
        "selfplay_mode": "current_vs_current_true_recurrent_shared_base",
        "behavior_snapshot": "frozen Stage4 root",
        "credit_scope": "branch_and_continuation",
        "continuation_discount": 0.97,
        "matchup_normalization": "independent_group_relative_returns",
        "optimizer_aggregation": "one_step_over_independent_groups",
        "rollout_storage": "compact bounded provenance bundle persisted adjacent to candidate",
    }
    if any(path is not None for path in agent_paths):
        config["selfplay_mode"] = "current_vs_external_policy_true_recurrent_shared_base"
    candidate_path = output_dir / "candidate.pt"
    candidate_hash = save_grpo_candidate_checkpoint(
        candidate_path,
        model,
        metadata,
        root_sha256=root_hash,
        sample_manifest_sha256=sample_manifest_file_hash,
        bundle_sha256=bundle_hash,
        sample_manifest_content_sha256=sample_manifest["sha256"],
        config=config,
        diagnostics=metrics,
        experiment=experiment,
    )
    preflight_artifacts = validate_candidate_provenance(
        candidate_path,
        approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
    )
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    return_statistics = []
    for collection in collections:
        advantages, group_stats = normalize_group_returns(collection["returns"])
        return_statistics.append(
            {
                "group_id": collection["group_id"],
                "returns": collection["returns"],
                "advantages": advantages.tolist(),
                **group_stats,
            }
        )
    all_collections_seconds = sum(float(item["collection_seconds"]) for item in collections)
    all_fibers = sum(int(item["games"]) for item in collections)
    all_decisions = sum(int(item["logical_decisions"]) for item in collections)
    all_substeps = sum(int(item["substeps"]) for item in collections)
    manifest: dict[str, Any] = {
        "format": SIBLING_FORMAT,
        "experiment": experiment,
        "captured_at": captured_at,
        "code_commit": _git_commit(),
        "root_checkpoint": str(checkpoint),
        "root_sha256": root_hash,
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "candidate_bytes": candidate_path.stat().st_size,
        "candidate_preflight": {
            "passed": True,
            "approved_root_sha256": APPROVED_STAGE4_ROOT_SHA256,
            "artifacts": {name: str(path) for name, path in preflight_artifacts.items()},
        },
        "sample_manifest": str(sample_manifest_path),
        "sample_manifest_sha256": sample_manifest_file_hash,
        "sample_manifest_content_sha256": sample_manifest["sha256"],
        "trajectory_bundle": str(trajectory_bundle_path),
        "bundle_sha256": bundle_hash,
        "metadata_date": meta_date,
        "deck": str(deck_path),
        "deck_content_sha256": deck_hash,
        "deck_source_file_sha256": deck_source_hash,
        "group_count": len(grouped_trajectories),
        "groups_per_matchup": groups_per_matchup,
        "matchup_count": len(opponent_paths),
        "group_size": all_fibers,
        "requested_group_size": k_max,
        "effective_group_sizes": [collection["games"] for collection in collections],
        "branch_actions": [collection["branch_actions"] for collection in collections],
        "branch_base": [collection["branch_base"] for collection in collections],
        "agent_sides": [int(item["agent_side"]) for item in all_trajectories],
        "group_returns": [collection["returns"] for collection in collections],
        "return_mean": [item["return_mean"] for item in return_statistics],
        "return_std": [item["return_std"] for item in return_statistics],
        "returns_advantages": [item["advantages"] for item in return_statistics],
        "logical_decisions": all_decisions,
        "substeps": all_substeps,
        "collection_seconds": round(all_collections_seconds, 6),
        "collection_games_per_second": all_fibers / all_collections_seconds if all_collections_seconds else None,
        "collection_decisions_per_second": all_decisions / all_collections_seconds if all_collections_seconds else None,
        "collection_substeps_per_second": all_substeps / all_collections_seconds if all_collections_seconds else None,
        "update_seconds": metrics["update_seconds"],
        "trajectory_summaries": [
            summary
            for collection in collections
            for summary in collection["trajectory_summaries"]
        ],
        "groups": [
            {
                "group_id": collection["group_id"],
                "matchup_index": collection["matchup_index"],
                "group_index": collection["group_index"],
                "opponent_deck": collection["opponent_deck"],
                "opponent_agent_path": collection.get("opponent_agent_path"),
                "opponent_mode": collection.get("opponent_mode"),
                "opponent_deck_content_sha256": collection.get("opponent_deck_content_sha256"),
                "opponent_deck_source_file_sha256": collection.get("opponent_deck_source_file_sha256"),
                "effective_group_size": collection["games"],
                "branch_actions": collection["branch_actions"],
                "returns": collection["returns"],
                "collection_seconds": collection["collection_seconds"],
            }
            for collection in collections
        ],
        "metrics": metrics,
        "config": config,
        "rollout_persistence": "compact bounded provenance bundle persisted; no unbounded rollout buffer",
        "tournament": {"status": "pending"},
        "invariants": {
            "behavior_snapshot_is_frozen_stage4_root": root_hash == APPROVED_STAGE4_ROOT_SHA256,
            "shared_branch_base_per_group": True,
            "distinct_legal_branch_actions_per_group": True,
            "common_seed_per_group": True,
            "independent_recurrent_lanes": True,
            "terminal_returns_in_minus_one_zero_plus_one": True,
            "complete_behavior_logprob_retained": True,
            "branch_credit_only": False,
            "continuation_credit_disabled": False,
            "future_continuation_credit": True,
            "dynamic_k_per_base": True,
            "matchup_stratification": True,
            "opponent_policy_stratification": any(path is not None for path in agent_paths),
            "independent_group_normalization": True,
            "single_grouped_optimizer_step": metrics["optimizer_steps"] == 1,
            "candidate_preflight_passed": True,
            "stage4_root_preserved": True,
            "requested_group_size_was_four": k_max == 4,
            "effective_group_uses_distinct_legal_actions": all(
                collection["games"] >= 2 for collection in collections
            ),
            "no_tournament": True,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics.json", metrics)
    _write_report(output_dir, manifest, metrics, experiment)
    _write_capsule(Path("experiments/autoresearch"), manifest, metrics, experiment)
    (output_dir / "logs" / "run.log").write_text(
        "\n".join(
            [
                f"{experiment} grouped dynamic-K sibling-fiber GRPO",
                f"group_count={manifest['group_count']}",
                f"effective_group_sizes={manifest['effective_group_sizes']}",
                f"returns={manifest['group_returns']}",
                f"logical_decisions={manifest['logical_decisions']}",
                f"substeps={manifest['substeps']}",
                f"collection_seconds={manifest['collection_seconds']}",
                f"update_seconds={metrics['update_seconds']}",
                f"candidate_sha256={candidate_hash}",
                "candidate_preflight=passed",
                "tournament=pending",
            ]
        )
        + "\n"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("experiments/autoresearch/root/stage4_root.pkl"))
    parser.add_argument("--agent-deck", type=Path, default=Path("agent/deck.csv"))
    parser.add_argument("--meta-date", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--opponent-deck",
        type=Path,
        action="append",
        default=None,
        help="Opponent deck CSV; repeat for independent matchup strata.",
    )
    parser.add_argument(
        "--opponent-agent",
        type=Path,
        action="append",
        default=None,
        help="Optional tournament opponent main.py, paired by order with --opponent-deck.",
    )
    parser.add_argument("--groups-per-matchup", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=21021)
    parser.add_argument("--experiment", type=str, default=EXPERIMENT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_ar021(
        checkpoint=args.checkpoint,
        deck_path=args.agent_deck,
        meta_date=args.meta_date,
        output_dir=args.output_dir,
        opponent_deck_paths=args.opponent_deck,
        opponent_agent_paths=args.opponent_agent,
        groups_per_matchup=args.groups_per_matchup,
        k_max=args.k_max,
        seed=args.seed,
        experiment=args.experiment,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
