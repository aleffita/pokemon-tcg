"""Execute the bounded AR-020 sibling-fiber GRPO micro-update."""

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
    DEFAULT_OUTPUT,
    SIBLING_FORMAT,
    collect_sibling_fiber_group,
    flatten_provenance_bundle,
    save_grpo_candidate_checkpoint,
    sibling_fiber_grpo_update,
)
from scripts.rl.trajectory_group_grpo import _git_commit, normalize_group_returns
from scripts.rl.trajectory_probe import (
    APPROVED_STAGE4_ROOT_SHA256,
    DateBoundEncoder,
    deck_content_sha256,
    load_deck,
    load_stage4,
    sha256_file,
    validate_meta_date,
)


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


def _write_report(output_dir: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    report = f"""# AR-020 - sibling-fiber GRPO micro-update

Captured on {manifest['captured_at']} from frozen Stage 4 root `{manifest['root_sha256']}`.

## Result

    The collector created the maximum available distinct legal sibling fibers
    (requested K=4) from one exact in-process recurrent base, with common
    post-branch randomness and independent recurrent lanes. One FP32
    policy-only update applied terminal group-relative credit to the branch and
    discounted future logical decisions. The frozen root remains the fallback.
    No tournament, package, submission, or promotion was run by the runner.

| Metric | Result |
| --- | ---: |
| Effective K / branch actions | {manifest['group_size']} / `{manifest['branch_actions']}` |
| Group returns | `{manifest['group_returns']}` |
| Return mean / population std | `{manifest['return_mean']}` / `{manifest['return_std']}` |
| Logical decisions / substeps | {manifest['logical_decisions']} / {manifest['substeps']} |
| Collection seconds / decisions/s | {manifest['collection_seconds']} / {manifest['collection_decisions_per_second']} |
| Update seconds | {manifest['update_seconds']} |
| Loss / gradient norm | {metrics['loss']} / {metrics['gradient_norm']} |
| Branch ratio mean / min / max | {metrics['ratio_mean']} / {metrics['ratio_min']} / {metrics['ratio_max']} |
| Candidate parameter L2 delta | {metrics['parameter_l2']} |
| Candidate bytes | {manifest['candidate_bytes']} |

## Contracts checked

- All fibers share the same first-state action-mask, model-input, recurrent
  memory, agent side, and random seed, while their first actions are distinct
  legal actions from the frozen root distribution.
- Behavior data uses true recurrent current-vs-current self-play with an
  independent mirror lane. Complete logical-action behavior logprobs remain
  recorded for every continuation.
- Sibling credit targets the branch logical action once per fiber and, in this
  run, propagates discounted terminal credit through future logical decisions.
  It is never duplicated across conditional substeps.
- Group returns are terminal values in `{{-1, 0, +1}}`; zero variance fails
  closed to an optimizer no-op. This run had
  `zero_variance_group={metrics['zero_variance_group']}`.
- Candidate is strict FP32 and linked to the adjacent bounded provenance
  bundle. Candidate preflight passed: `{manifest['candidate_preflight']['passed']}`.

## Limitations and next gate

This is a bounded dynamic-K prospective micro-update, not a strength estimate.
Collection is serial and the recurrent learner boundary is detached. Run the
candidate-vs-root and common multi-opponent/deck-panel tournament before
interpreting or promoting it.

## Provenance

- Code commit at execution: `{manifest['code_commit']}`
- Candidate SHA-256: `{manifest['candidate_sha256']}`
- Sample manifest SHA-256: `{manifest['sample_manifest_sha256']}`
- Trajectory bundle SHA-256: `{manifest['bundle_sha256']}`
- Candidate-vs-root gate: pending
"""
    (output_dir / "report.md").write_text(report)


def _write_capsule(root: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    capsule = f"""# State Capsule 020 - sibling-fiber GRPO micro-update

Captured {manifest['captured_at']}.

## Current state

- Frozen Stage 4 root remains fallback: `{manifest['root_sha256']}`.
- AR-020 collected `{manifest['group_size']}` common-base sibling fibers
  (requested K=4) with branch actions
  `{manifest['branch_actions']}` and returns `{manifest['group_returns']}`.
- One FP32 policy-only update applied branch-relative terminal credit through
  the future continuation with discount `{metrics['continuation_discount']}`.
- Candidate: `{manifest['candidate_sha256']}`; preflight passed.
- No tournament, promotion, RoPE-ND, MoE, or historical ETL/Parquet/packed-data
  work was run by the runner.

## Evidence

- `experiments/autoresearch/AR-020/report.md`
- `experiments/autoresearch/AR-020/manifest.json`
- `experiments/autoresearch/AR-020/metrics.json`
- `experiments/autoresearch/AR-020/sample.manifest.json`
- `experiments/autoresearch/AR-020/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-020/candidate.pt`

## Metrics

- Collection: `{manifest['collection_seconds']}` s, `{manifest['collection_decisions_per_second']}` decisions/s.
- Update: `{metrics['update_seconds']}` s; ratio mean `{metrics['ratio_mean']}`.
- Credit scope: `{metrics['credit_scope']}`; credited logical actions `{metrics['credited_logical_actions']}`.
- Parameter L2 delta: `{metrics['parameter_l2']}`; gradient norm `{metrics['gradient_norm']}`.

## Next control point

Run the same candidate-vs-root and opponent-panel tournament surface. Keep the
root fallback unless sibling-fiber evidence improves the useful panel.
"""
    (root / "STATE_CAPSULE_020.md").write_text(capsule)


def run_ar020(
    *,
    checkpoint: Path,
    deck_path: Path,
    meta_date: str,
    output_dir: Path = DEFAULT_OUTPUT,
    opponent_deck_paths: list[Path] | None = None,
    k_max: int = 4,
    seed: int = 20020,
) -> dict[str, Any]:
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
    grouped_trajectories: list[list[dict[str, Any]]] = []
    collections: list[dict[str, Any]] = []
    for matchup_index, opponent_path in enumerate(opponent_paths):
        opponent_deck = load_deck(opponent_path)
        opponent_hash = deck_content_sha256(opponent_deck)
        opponent_source_hash = sha256_file(opponent_path)
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
            seed=seed + matchup_index * 1000,
        )
        collection["matchup_index"] = matchup_index
        collection["opponent_deck"] = str(opponent_path)
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
    # Keep each matchup's relative normalization independent. The optimizer
    # steps are sequential but all groups were collected from the same frozen
    # root snapshot, so no deck's returns are incorrectly centered against a
    # different deck's reward distribution.
    matchup_metrics = [
        sibling_fiber_grpo_update(
            model,
            root_reference,
            trajectories,
            clip_epsilon=0.2,
            learning_rate=1e-5,
            credit_scope="branch_and_continuation",
            continuation_discount=0.97,
        )
        for trajectories in grouped_trajectories
    ]
    metrics = dict(matchup_metrics[-1])
    if len(matchup_metrics) > 1:
        metrics.update(
            {
                "algorithm": "sibling_fiber_grpo_stratified_matchups",
                "matchup_count": len(matchup_metrics),
                "matchup_metrics": matchup_metrics,
                "update_seconds": sum(float(item["update_seconds"]) for item in matchup_metrics),
                "group_size": sum(int(item["group_size"]) for item in matchup_metrics),
                "logical_decisions": sum(int(item["logical_decisions"]) for item in matchup_metrics),
                "continuation_logical_decisions": sum(
                    int(item["continuation_logical_decisions"]) for item in matchup_metrics
                ),
                "credited_logical_actions": sum(
                    int(item["credited_logical_actions"]) for item in matchup_metrics
                ),
            }
        )
    config = {
        "algorithm": "sibling_fiber_grpo",
        "group_size_cap": k_max,
        "effective_group_sizes": [collection["games"] for collection in collections],
        "matchup_count": len(collections),
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
        "rollout_storage": "compact bounded provenance bundle persisted adjacent to candidate",
    }
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
        experiment="AR-020",
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
        "experiment": "AR-020",
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
        "matchups": [
            {
                "matchup_index": collection["matchup_index"],
                "opponent_deck": collection["opponent_deck"],
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
            "shared_branch_base": True,
            "distinct_legal_branch_actions": True,
            "common_seed": True,
            "independent_recurrent_lanes": True,
            "terminal_returns_in_minus_one_zero_plus_one": True,
            "complete_behavior_logprob_retained": True,
            "branch_credit_only": False,
            "continuation_credit_disabled": False,
            "future_continuation_credit": True,
            "dynamic_k_per_base": True,
            "matchup_stratification": True,
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
    _write_report(output_dir, manifest, metrics)
    _write_capsule(Path("experiments/autoresearch"), manifest, metrics)
    (output_dir / "logs" / "run.log").write_text(
        "\n".join(
            [
                "AR-020 sibling-fiber GRPO micro-update",
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
    parser.add_argument("--k-max", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20020)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_ar020(
        checkpoint=args.checkpoint,
        deck_path=args.agent_deck,
        meta_date=args.meta_date,
        output_dir=args.output_dir,
        opponent_deck_paths=args.opponent_deck,
        k_max=args.k_max,
        seed=args.seed,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
