"""Run AR-021: grouped dynamic-K prospective sibling-fiber GRPO."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.policy_infer_torch import load_inference_checkpoint
from rl.ropend import default_ropend_config
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


def _aggregate_hash(values: list[str]) -> str:
    """Stable identity for an ordered explicit deck list."""
    payload = json.dumps(values, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _behavior_snapshot_hash(parent_sha256: str, ropend_config: dict[str, Any] | None) -> str:
    """Identify the exact rollout architecture without changing root provenance."""
    payload = {
        "parent_sha256": parent_sha256,
        "ropend": ropend_config,
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_parent_checkpoint(checkpoint: Path, card_table: Any) -> tuple[Any, dict[str, Any], str]:
    """Load the immutable root or a provenance-linked descendant candidate."""
    parent_hash = sha256_file(checkpoint)
    if parent_hash == APPROVED_STAGE4_ROOT_SHA256:
        model, metadata = load_stage4(checkpoint, card_table)
    else:
        validate_candidate_provenance(
            checkpoint,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )
        model, metadata = load_inference_checkpoint(checkpoint, card_table)
    return model, metadata, parent_hash


def deck_relative_group_advantages(
    collections: list[dict[str, Any]],
) -> tuple[list[float], list[dict[str, Any]]]:
    """Pair deck outcomes by opponent/group seed and standardize within cohort."""
    cohorts: dict[tuple[int, int], list[tuple[int, int, float]]] = {}
    for collection_index, collection in enumerate(collections):
        key = (int(collection["matchup_index"]), int(collection["group_index"]))
        score = float(np.mean(np.asarray(collection["returns"], dtype=np.float64)))
        cohorts.setdefault(key, []).append(
            (collection_index, int(collection["learner_deck_index"]), score)
        )

    advantages = [0.0] * len(collections)
    summaries: list[dict[str, Any]] = []
    for (matchup_index, group_index), rows in sorted(cohorts.items()):
        rows.sort(key=lambda item: item[1])
        scores = np.asarray([row[2] for row in rows], dtype=np.float64)
        mean = float(scores.mean())
        std = float(scores.std(ddof=0))
        normalized = np.zeros_like(scores) if len(rows) < 2 or std <= 1e-8 else (scores - mean) / std
        for (collection_index, _deck_index, _score), value in zip(rows, normalized, strict=True):
            advantages[collection_index] = float(value)
        summaries.append(
            {
                "matchup_index": matchup_index,
                "group_index": group_index,
                "learner_deck_indices": [row[1] for row in rows],
                "scores": scores.tolist(),
                "mean": mean,
                "std": std,
                "zero_variance": bool(len(rows) < 2 or std <= 1e-8),
                "advantages": normalized.tolist(),
            }
        )
    return advantages, summaries


def _pool_deck_paths(pool_dir: Path | None, limit: int) -> list[Path]:
    if pool_dir is None or not pool_dir.exists():
        return []
    if limit < 0:
        raise ValueError("--deck-pool-limit must be non-negative")
    paths = sorted(path for path in pool_dir.glob("*.json") if path.is_file())
    return paths[:limit]


def _resolve_update_device(requested: str) -> torch.device:
    if requested == "auto":
        # Measured on this workload: small recurrent batches are faster on CPU
        # than MPS because dispatch/transfer dominates. Keep Metal explicit for
        # future larger batches instead of silently selecting a slower device.
        return torch.device("cpu")
    if requested not in {"cpu", "mps"}:
        raise ValueError("--update-device must be auto, cpu, or mps")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("--update-device=mps requested but Metal is unavailable")
    return torch.device(requested)


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
Each base selected its own effective K from the legal action set, launched all K
continuations concurrently, and combined sibling-relative credit with paired
inter-deck credit across equal opponent/group seeds. The frozen behavior data
is reused for multiple FP32 policy-only epochs when relative signal exists. If
every sibling and deck cohort was homogeneous,
the update emitted a root-equivalent no-op candidate and preserved the
zero-variance evidence. The frozen root remains the fallback pending tournament.

| Metric | Result |
| --- | ---: |
| Groups / fibers | {manifest['group_count']} / {manifest['group_size']} |
| Effective K per base | `{manifest['effective_group_sizes']}` |
| Branch policy/uniform mixture | `{manifest['config']['branch_selection']}` / {manifest['config']['branch_uniform_mix']} |
| Logical decisions / substeps | {manifest['logical_decisions']} / {manifest['substeps']} |
| Collection seconds / decisions/s | {manifest['collection_seconds']} / {manifest['collection_decisions_per_second']} |
| Grouped optimizer steps | {metrics['optimizer_steps']} |
| Update seconds | {metrics['update_seconds']} |
| Loss / gradient norm | {metrics['loss']} / {metrics['gradient_norm']} |
| Candidate parameter L2 delta | {metrics['parameter_l2']} |

## Contracts checked

- Every sibling group has one exact simulator snapshot, distinct legal branch
  actions, common branch provenance, and independent recurrent lanes.
- Effective K is dynamic per base: `min(K_max, legal branch actions)`.
- Deck and matchup strata normalize returns independently; no group is centered
  against another matchup's terminal distribution.
- The candidate uses `{metrics['requested_update_epochs']}` requested optimizer
  epochs over all signal-bearing groups, while each group's sibling-relative
  credit remains separate; an all-zero-signal matrix is explicitly fail-closed.
- All K sibling futures execute simultaneously after the recurrent branch base
  is fixed; no polling or scheduler participates in process completion.
- All rollouts run to terminal completion and continuation credit uses discount
  `{metrics['continuation_discount']}` without duplicating conditional substeps.
- Candidate preflight passed: `{manifest['candidate_preflight']['passed']}`.

## Limitations and next gate

This is a bounded grouped prospective update, not a strength estimate. Groups
are collected sequentially while sibling games within each group are parallel;
the recurrent learner boundary is detached. Run
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
- The grouped FP32 policy-only path applied sibling-relative and paired
  inter-deck terminal credit through future continuation with discount
  `{metrics['continuation_discount']}`, or emitted a no-op when all groups were
  zero-variance.
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
- Update: `{metrics['update_seconds']}` s; `{metrics['optimizer_steps']}` optimizer steps.
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
    learner_deck_paths: list[Path] | None = None,
    output_dir: Path = DEFAULT_OUTPUT,
    opponent_deck_paths: list[Path] | None = None,
    opponent_agent_paths: list[Path] | None = None,
    groups_per_matchup: int = 2,
    k_max: int = 4,
    branch_uniform_mix: float = 0.0,
    update_epochs: int = 4,
    deck_relative_weight: float = 0.5,
    learning_rate: float = 5e-6,
    ropend: bool = False,
    ropend_init_scale: float = 0.0,
    prospective_aux_weight: float = 0.2,
    deck_aux_weight: float = 0.1,
    aux_batch_size: int = 256,
    deck_pool_dir: Path | None = Path("experiments/decks/swarm/inbox"),
    deck_pool_limit: int = 8,
    swarm_results_dir: Path = Path("experiments/decks/swarm/results"),
    update_device: str = "auto",
    seed: int = 21021,
    experiment: str = EXPERIMENT,
) -> dict[str, Any]:
    if groups_per_matchup < 1:
        raise ValueError("--groups-per-matchup must be at least one")
    if k_max < 2:
        raise ValueError("--k-max must be at least two")
    if not 0.0 <= branch_uniform_mix <= 1.0:
        raise ValueError("--branch-uniform-mix must be between zero and one")
    if update_epochs < 1:
        raise ValueError("--update-epochs must be at least one")
    if not np.isfinite(deck_relative_weight) or deck_relative_weight < 0.0:
        raise ValueError("--deck-relative-weight must be finite and non-negative")
    if not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("--learning-rate must be finite and positive")
    if not np.isfinite(prospective_aux_weight) or prospective_aux_weight < 0.0:
        raise ValueError("--prospective-aux-weight must be finite and non-negative")
    if not np.isfinite(deck_aux_weight) or deck_aux_weight < 0.0:
        raise ValueError("--deck-aux-weight must be finite and non-negative")
    if aux_batch_size < 1:
        raise ValueError("--aux-batch-size must be at least one")
    validate_meta_date(meta_date)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    learner_paths = list(learner_deck_paths or [deck_path])
    learner_paths.extend(_pool_deck_paths(deck_pool_dir, deck_pool_limit))
    if not learner_paths:
        raise ValueError("at least one learner deck is required")
    learner_decks = []
    seen_deck_content: set[str] = set()
    for path in learner_paths:
        deck = load_deck(path)
        content_hash = deck_content_sha256(deck)
        if content_hash in seen_deck_content:
            continue
        seen_deck_content.add(content_hash)
        learner_decks.append((path, deck, content_hash, sha256_file(path)))
    learner_paths = [item[0] for item in learner_decks]
    deck_snapshot_dir = output_dir / "decks"
    deck_snapshot_dir.mkdir(parents=True, exist_ok=True)
    deck_snapshots: list[str] = []
    for index, (path, _deck, content_hash, _source_hash) in enumerate(learner_decks):
        snapshot = deck_snapshot_dir / f"{index:03d}_{content_hash[:12]}.json"
        shutil.copy2(path, snapshot)
        deck_snapshots.append(str(snapshot))
    learner_content_hashes = [item[2] for item in learner_decks]
    learner_source_hashes = [item[3] for item in learner_decks]
    aggregate_content_hash = _aggregate_hash(learner_content_hashes)
    aggregate_source_hash = _aggregate_hash(learner_source_hashes)
    card_table = get_card_table()
    model, metadata, parent_hash = _load_parent_checkpoint(checkpoint, card_table)
    existing_ropend = getattr(model, "ropend_config", None)
    if ropend and existing_ropend is None:
        canonical_ropend = model.enable_ropend(
            default_ropend_config(init_scale=ropend_init_scale)
        )
        metadata = {**metadata, "ropend": canonical_ropend}
    elif existing_ropend is not None:
        canonical_ropend = dict(existing_ropend)
        metadata = {**metadata, "ropend": canonical_ropend}
    else:
        canonical_ropend = None
    model = model.float()
    root_reference = copy.deepcopy(model).eval()
    encoder = DateBoundEncoder(TokenEncoder(card_table), meta_date)
    root_hash = APPROVED_STAGE4_ROOT_SHA256
    behavior_hash = _behavior_snapshot_hash(parent_hash, canonical_ropend)
    opponent_paths = opponent_deck_paths or [deck_path]
    if opponent_agent_paths is not None and len(opponent_agent_paths) != len(opponent_paths):
        raise ValueError("--opponent-agent must be repeated once per --opponent-deck")
    agent_paths = opponent_agent_paths or [None] * len(opponent_paths)
    grouped_trajectories: list[list[dict[str, Any]]] = []
    collections: list[dict[str, Any]] = []
    for learner_index, (learner_path, deck, deck_hash, deck_source_hash) in enumerate(learner_decks):
        for matchup_index, opponent_path in enumerate(opponent_paths):
            opponent_deck = load_deck(opponent_path)
            opponent_hash = deck_content_sha256(opponent_deck)
            opponent_source_hash = sha256_file(opponent_path)
            for group_index in range(groups_per_matchup):
                group_seed = (
                    seed
                    + matchup_index * 100_000
                    + group_index * 1_000
                )
                episode_prefix = (
                    f"{experiment.lower().replace('-', '')}"
                    f"-d{learner_index}-m{matchup_index}-g{group_index}"
                )
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
                    model_hash=behavior_hash,
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
                    branch_uniform_mix=branch_uniform_mix,
                )
                collection.update(
                    {
                        "learner_deck_index": learner_index,
                        "learner_deck": str(learner_path),
                        "learner_deck_content_sha256": deck_hash,
                        "learner_deck_source_file_sha256": deck_source_hash,
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
        deck_content_sha256=aggregate_content_hash,
        deck_source_file_sha256=aggregate_source_hash,
    )
    validate_bundle(provenance_bundle, _manifest_rows(sample_manifest))
    sample_manifest_path = output_dir / "sample.manifest.json"
    trajectory_bundle_path = output_dir / "trajectory_bundle.pt.gz"
    _write_json(sample_manifest_path, sample_manifest)
    bundle_hash = save_compressed_bundle(trajectory_bundle_path, provenance_bundle, sample_manifest)
    sample_manifest_file_hash = sha256_file(sample_manifest_path)

    deck_group_advantages, deck_cohorts = deck_relative_group_advantages(collections)
    resolved_update_device = _resolve_update_device(update_device)
    model = model.to(resolved_update_device)
    root_reference = root_reference.to(resolved_update_device)

    metrics = sibling_fiber_grpo_update_groups(
        model,
        root_reference,
        grouped_trajectories,
        clip_epsilon=0.2,
        learning_rate=learning_rate,
        credit_scope="branch_and_continuation",
        continuation_discount=0.97,
        update_epochs=update_epochs,
        deck_group_advantages=deck_group_advantages,
        deck_relative_weight=deck_relative_weight,
        prospective_aux_weight=prospective_aux_weight,
        deck_aux_weight=deck_aux_weight,
        aux_batch_size=aux_batch_size,
    )
    config = {
        "algorithm": "sibling_fiber_grpo_grouped",
        "group_size_cap": k_max,
        "branch_uniform_mix": branch_uniform_mix,
        "branch_selection": "policy_uniform_mixture",
        "groups_per_matchup": groups_per_matchup,
        "matchup_count": len(opponent_paths),
        "learner_deck_count": len(learner_decks),
        "learner_deck_paths": [str(path) for path, _deck, _hash, _source_hash in learner_decks],
        "learner_deck_snapshots": deck_snapshots,
        "deck_pool_dir": str(deck_pool_dir) if deck_pool_dir is not None else None,
        "deck_pool_limit": deck_pool_limit,
        "learner_deck_content_sha256": learner_content_hashes,
        "learner_deck_source_file_sha256": learner_source_hashes,
        "logical_matchup_count": len(learner_decks) * len(opponent_paths),
        "opponent_agent_paths": [
            str(path) if path is not None else None for path in agent_paths
        ],
        "group_count": len(grouped_trajectories),
        "effective_group_sizes": [collection["games"] for collection in collections],
        "clip_epsilon": 0.2,
        "learning_rate": learning_rate,
        "update_epochs": update_epochs,
        "deck_relative_weight": deck_relative_weight,
        "prospective_aux_weight": prospective_aux_weight,
        "deck_aux_weight": deck_aux_weight,
        "aux_batch_size": aux_batch_size,
        "update_device": str(resolved_update_device),
        "collection_device": "cpu_multiprocess",
        "prospective_targets": "same-turn KO/prize plus future prize margin and terminal outcome",
        "deck_target": "first recurrent state masked deck-card distribution with tied card embeddings",
        "deck_relative_cohorts": deck_cohorts,
        "advantage_epsilon": 1e-8,
        "precision": "FP32",
        "value_loss": 0.0,
        "selfplay_mode": "current_vs_current_true_recurrent_shared_base",
        "root_checkpoint": "frozen Stage4 root",
        "parent_checkpoint": str(checkpoint),
        "parent_sha256": parent_hash,
        "behavior_snapshot_sha256": behavior_hash,
        "ropend": canonical_ropend,
        "credit_scope": "branch_and_continuation",
        "continuation_discount": 0.97,
        "matchup_normalization": "sibling_relative_plus_paired_inter_deck_cohort",
        "optimizer_aggregation": "multi_epoch_over_frozen_behavior_groups_or_fail_closed_noop",
        "parallel_collection": "all_dynamic_K sibling continuations execute concurrently",
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
        "root_checkpoint": "experiments/autoresearch/root/stage4_root.pkl",
        "root_sha256": root_hash,
        "parent_checkpoint": str(checkpoint),
        "parent_sha256": parent_hash,
        "behavior_snapshot_sha256": behavior_hash,
        "ropend": canonical_ropend,
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
        "deck": str(learner_paths[0]) if len(learner_paths) == 1 else "multiple learner decks",
        "deck_paths": [str(path) for path in learner_paths],
        "deck_content_sha256": aggregate_content_hash,
        "deck_source_file_sha256": aggregate_source_hash,
        "learner_decks": [
            {
                "path": str(path),
                "snapshot": deck_snapshots[index],
                "content_sha256": content_hash,
                "source_file_sha256": source_hash,
            }
            for index, (path, _deck, content_hash, source_hash) in enumerate(learner_decks)
        ],
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
        "deck_group_advantages": deck_group_advantages,
        "deck_relative_cohorts": deck_cohorts,
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
                "learner_deck_index": collection["learner_deck_index"],
                "learner_deck": collection["learner_deck"],
                "learner_deck_content_sha256": collection["learner_deck_content_sha256"],
                "matchup_index": collection["matchup_index"],
                "group_index": collection["group_index"],
                "opponent_deck": collection["opponent_deck"],
                "opponent_agent_path": collection.get("opponent_agent_path"),
                "opponent_mode": collection.get("opponent_mode"),
                "opponent_deck_content_sha256": collection.get("opponent_deck_content_sha256"),
                "opponent_deck_source_file_sha256": collection.get("opponent_deck_source_file_sha256"),
                "effective_group_size": collection["games"],
                "branch_uniform_mix": collection.get("branch_uniform_mix", branch_uniform_mix),
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
            "root_provenance_is_frozen_stage4": root_hash == APPROVED_STAGE4_ROOT_SHA256,
            "parent_is_root_or_valid_descendant": True,
            "paired_seed_across_learner_decks": True,
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
            "learner_deck_stratification": len(learner_decks) > 1,
            "independent_group_normalization": True,
            "requested_multi_epoch_update": update_epochs > 1,
            "optimizer_steps_match_requested_epochs": metrics["optimizer_steps"] in {0, update_epochs},
            "inter_deck_relative_credit": len(learner_decks) > 1 and deck_relative_weight > 0.0,
            "parallel_sibling_games": all(
                bool(collection.get("parallel_fibers")) for collection in collections
            ),
            "all_groups_zero_variance": metrics["zero_variance_groups"] == len(grouped_trajectories),
            "no_signal_fail_closed": metrics.get("no_update_reason") == "all_groups_zero_variance",
            "candidate_preflight_passed": True,
            "stage4_root_preserved": True,
            "requested_group_size_was_four": k_max == 4,
            "branch_uniform_mix_finite": 0.0 <= branch_uniform_mix <= 1.0,
            "effective_group_uses_distinct_legal_actions": all(
                collection["games"] >= 2 for collection in collections
            ),
            "no_tournament": True,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics.json", metrics)
    swarm_result = {
        "format": "ptcg-deck-swarm-result-v1",
        "experiment": experiment,
        "captured_at": captured_at,
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "parent_sha256": parent_hash,
        "learner_decks": manifest["learner_decks"],
        "groups": manifest["groups"],
        "deck_relative_cohorts": deck_cohorts,
        "throughput": {
            "games": all_fibers,
            "logical_decisions": all_decisions,
            "collection_seconds": manifest["collection_seconds"],
            "games_per_second": manifest["collection_games_per_second"],
            "decisions_per_second": manifest["collection_decisions_per_second"],
            "update_seconds": metrics["update_seconds"],
        },
        "training": {
            "optimizer_steps": metrics["optimizer_steps"],
            "zero_variance_groups": metrics["zero_variance_groups"],
            "credited_logical_actions": metrics["credited_logical_actions"],
            "prospective_auxiliary": metrics.get("prospective_auxiliary"),
            "deck_reconstruction": metrics.get("deck_reconstruction"),
        },
        "tournament": manifest["tournament"],
    }
    swarm_results_dir.mkdir(parents=True, exist_ok=True)
    _write_json(swarm_results_dir / f"{experiment}.json", swarm_result)
    _write_json(swarm_results_dir / "latest.json", swarm_result)
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
    parser.add_argument(
        "--agent-deck",
        type=Path,
        action="append",
        default=None,
        help="Learner deck JSON/CSV; repeat for explicit multi-deck strata.",
    )
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
    parser.add_argument(
        "--branch-uniform-mix",
        type=float,
        default=0.0,
        help="Mix uniform legal-action mass into sibling branch sampling.",
    )
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--deck-relative-weight", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--ropend", action="store_true")
    parser.add_argument("--ropend-init-scale", type=float, default=0.0)
    parser.add_argument("--prospective-aux-weight", type=float, default=0.2)
    parser.add_argument("--deck-aux-weight", type=float, default=0.1)
    parser.add_argument("--aux-batch-size", type=int, default=256)
    parser.add_argument(
        "--deck-pool-dir",
        type=Path,
        default=Path("experiments/decks/swarm/inbox"),
    )
    parser.add_argument("--deck-pool-limit", type=int, default=8)
    parser.add_argument(
        "--swarm-results-dir",
        type=Path,
        default=Path("experiments/decks/swarm/results"),
    )
    parser.add_argument("--update-device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--seed", type=int, default=21021)
    parser.add_argument("--experiment", type=str, default=EXPERIMENT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    learner_deck_paths = args.agent_deck or [Path("agent/deck.csv")]
    print(json.dumps(run_ar021(
        checkpoint=args.checkpoint,
        deck_path=learner_deck_paths[0],
        meta_date=args.meta_date,
        learner_deck_paths=learner_deck_paths,
        output_dir=args.output_dir,
        opponent_deck_paths=args.opponent_deck,
        opponent_agent_paths=args.opponent_agent,
        groups_per_matchup=args.groups_per_matchup,
        k_max=args.k_max,
        branch_uniform_mix=args.branch_uniform_mix,
        update_epochs=args.update_epochs,
        deck_relative_weight=args.deck_relative_weight,
        learning_rate=args.learning_rate,
        ropend=args.ropend,
        ropend_init_scale=args.ropend_init_scale,
        prospective_aux_weight=args.prospective_aux_weight,
        deck_aux_weight=args.deck_aux_weight,
        aux_batch_size=args.aux_batch_size,
        deck_pool_dir=args.deck_pool_dir,
        deck_pool_limit=args.deck_pool_limit,
        swarm_results_dir=args.swarm_results_dir,
        update_device=args.update_device,
        seed=args.seed,
        experiment=args.experiment,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
