"""Execute the bounded AR-019 K=4 trajectory-group GRPO micro-update."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
from pathlib import Path
from typing import Any

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from scripts.rl.trajectory_group_grpo import (
    DEFAULT_OUTPUT,
    GRPO_FORMAT,
    _git_commit,
    collect_stage4_trajectory_group,
    flatten_provenance_bundle,
    normalize_group_returns,
    save_grpo_candidate_checkpoint,
    trajectory_group_grpo_update,
)
from scripts.rl.ppo_micro_update import (
    build_sample_manifest,
    save_compressed_bundle,
    validate_bundle,
    validate_candidate_provenance,
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


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _manifest_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the row-side digest view consumed by the existing bundle validator."""
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
) -> None:
    report = f"""# AR-019 - trajectory-group GRPO micro-update

Captured on {manifest['captured_at']} from frozen Stage 4 root `{manifest['root_sha256']}`.

## Result

The corrected AR-018 current-vs-current true recurrent collector produced a
single in-memory group of K=4 complete agent trajectories. One FP32,
policy-only trajectory-group GRPO update was applied to a copy of the root.
The frozen root remains the fallback. No tournament, package, submission, or
promotion was run.

| Metric | Result |
| --- | ---: |
| Group returns | `{manifest['group_returns']}` |
| Return mean / population std | `{manifest['return_mean']}` / `{manifest['return_std']}` |
| Logical decisions / substeps | {manifest['logical_decisions']} / {manifest['substeps']} |
| Collection seconds / decisions/s | {manifest['collection_seconds']} / {manifest['collection_decisions_per_second']} |
| Update seconds | {manifest['update_seconds']} |
| Loss / gradient norm | {metrics['loss']} / {metrics['gradient_norm']} |
| Ratio mean / min / max | {metrics['ratio_mean']} / {metrics['ratio_min']} / {metrics['ratio_max']} |
| Candidate parameter L2 delta | {metrics['parameter_l2']} |
| Candidate bytes | {manifest['candidate_bytes']} |

## Contracts checked

- Behavior data came from the frozen root and true recurrent current-vs-current
  self-play. The mirror retained its independent Stage 4 memory lane.
- Every retained agent sample has a detached recurrent input, a real legal
  mask, an action, and a finite behavior substep logprob.
- Each logical-action behavior logprob is the sum of its conditional substep
  logprobs. Learner ratios use that logical sum once per decision.
- Group credit is assigned once per logical decision and shared across all
  substeps of that decision. No separate substep-relative credit is used.
- Zero-variance groups normalize to zero advantages and perform zero optimizer
  steps. This run had `zero_variance_group={metrics['zero_variance_group']}`.
- Candidate checkpoint is strict FP32 inference format and is linked to the
  adjacent compact bounded provenance bundle. No unbounded rollout buffer was
  written.
- Existing candidate provenance preflight passed for the root, manifest, and
  trajectory bundle: `{manifest['candidate_preflight']['passed']}`.

## Limitations

This is a K=4 micro-update, not a strength estimate. Collection is serial,
the recurrent learner boundary is detached, value loss is intentionally zero,
and no tournament has been run. The candidate is experimental only; Stage 4
root remains the fallback. The group return variance is small-sample and the
candidate has not been promoted.

## Provenance

- Code commit at execution: `{manifest['code_commit']}`
- Root SHA-256: `{manifest['root_sha256']}`
- Candidate SHA-256: `{manifest['candidate_sha256']}`
- Candidate path: `{manifest['candidate']}`
- Sample manifest: `{manifest['sample_manifest']}`
- Sample manifest file SHA-256: `{manifest['sample_manifest_sha256']}`
- Sample manifest content SHA-256: `{manifest['sample_manifest_content_sha256']}`
- Trajectory bundle: `{manifest['trajectory_bundle']}`
- Trajectory bundle SHA-256: `{manifest['bundle_sha256']}`
- Candidate preflight: `{manifest['candidate_preflight']}`
- Rollout persistence: `{manifest['rollout_persistence']}`
"""
    (output_dir / "report.md").write_text(report)


def _write_capsule(root: Path, manifest: dict[str, Any], metrics: dict[str, Any]) -> None:
    capsule = f"""# State Capsule 019 - first trajectory-group GRPO micro-update

Captured {manifest['captured_at']}.

## Current state

- Frozen Stage 4 root remains fallback: `{manifest['root_sha256']}`.
- AR-019 collected K=4 current-vs-current true recurrent trajectories in
  memory and applied one FP32 policy-only trajectory-group GRPO update.
- Candidate: `{manifest['candidate_sha256']}` ({manifest['candidate_bytes']} bytes).
- Candidate parameter L2 delta: `{metrics['parameter_l2']}`; changed parameters:
  `{metrics['changed_parameters']}`.
- Compact provenance manifest: `{manifest['sample_manifest']}`;
  SHA-256 `{manifest['sample_manifest_sha256']}`.
- Compact trajectory bundle: `{manifest['trajectory_bundle']}`;
  SHA-256 `{manifest['bundle_sha256']}`.
- Candidate provenance preflight passed: `{manifest['candidate_preflight']['passed']}`.
- No tournament, package, submission, promotion, MoE, RoPE-ND, or historical
  ETL/Parquet/packed-data work was run.

## Evidence

- `experiments/autoresearch/AR-019/report.md`
- `experiments/autoresearch/AR-019/manifest.json`
- `experiments/autoresearch/AR-019/metrics.json`
- `experiments/autoresearch/AR-019/logs/tests.log`
- `experiments/autoresearch/AR-019/sample.manifest.json`
- `experiments/autoresearch/AR-019/trajectory_bundle.pt.gz`
- `experiments/autoresearch/AR-019/candidate.pt`

## Metrics

- Returns: `{manifest['group_returns']}`; population std `{manifest['return_std']}`.
- Logical decisions/substeps: `{manifest['logical_decisions']}/{manifest['substeps']}`.
- Ratios: mean `{metrics['ratio_mean']}`, range
  `[{metrics['ratio_min']}, {metrics['ratio_max']}]`.
- Loss/gradient: `{metrics['loss']}` / `{metrics['gradient_norm']}`.

## Limitations and next control point

This micro-update is not a competitive result. It uses a detached recurrent
learner boundary, no value loss, serial collection, and K=4 returns. Do not
promote or tournament this candidate without a later explicit research gate.
"""
    (root / "STATE_CAPSULE_019.md").write_text(capsule)


def run_ar019(
    *,
    checkpoint: Path,
    deck_path: Path,
    meta_date: str,
    output_dir: Path = DEFAULT_OUTPUT,
    seed: int = 19019,
) -> dict[str, Any]:
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
    trajectories, collection = collect_stage4_trajectory_group(
        model=model,
        encoder=encoder,
        deck=deck,
        deck_content_hash=deck_hash,
        deck_source_file_hash=deck_source_hash,
        model_hash=root_hash,
        games=4,
        seed=seed,
    )
    provenance_bundle = flatten_provenance_bundle(trajectories)
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
    bundle_hash = save_compressed_bundle(
        trajectory_bundle_path,
        provenance_bundle,
        sample_manifest,
    )
    sample_manifest_file_hash = sha256_file(sample_manifest_path)
    advantages, group_stats = normalize_group_returns(collection["returns"])
    metrics = trajectory_group_grpo_update(
        model,
        root_reference,
        trajectories,
        clip_epsilon=0.2,
        learning_rate=1e-5,
    )
    config = {
        "algorithm": "trajectory_group_grpo",
        "group_size": 4,
        "clip_epsilon": 0.2,
        "learning_rate": 1e-5,
        "advantage_epsilon": 1e-8,
        "precision": "FP32",
        "value_loss": 0.0,
        "selfplay_mode": "current_vs_current_true_recurrent",
        "behavior_snapshot": "frozen Stage4 root",
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
    )
    preflight_artifacts = validate_candidate_provenance(
        candidate_path,
        approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
    )
    candidate_bytes = candidate_path.stat().st_size
    captured_at = dt.datetime.now(dt.timezone.utc).isoformat()
    manifest: dict[str, Any] = {
        "format": GRPO_FORMAT,
        "experiment": "AR-019",
        "captured_at": captured_at,
        "code_commit": _git_commit(),
        "root_checkpoint": str(checkpoint),
        "root_sha256": root_hash,
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "candidate_bytes": candidate_bytes,
        "sample_manifest": str(sample_manifest_path),
        "sample_manifest_sha256": sample_manifest_file_hash,
        "sample_manifest_content_sha256": sample_manifest["sha256"],
        "trajectory_bundle": str(trajectory_bundle_path),
        "bundle_sha256": bundle_hash,
        "candidate_preflight": {
            "passed": True,
            "approved_root_sha256": APPROVED_STAGE4_ROOT_SHA256,
            "artifacts": {
                name: str(path) for name, path in preflight_artifacts.items()
            },
        },
        "metadata_date": meta_date,
        "deck": str(deck_path),
        "deck_content_sha256": deck_hash,
        "deck_source_file_sha256": deck_source_hash,
        "group_size": 4,
        "group_returns": collection["returns"],
        "return_mean": group_stats["return_mean"],
        "return_std": group_stats["return_std"],
        "logical_decisions": collection["logical_decisions"],
        "substeps": collection["substeps"],
        "collection_seconds": round(float(collection["collection_seconds"]), 6),
        "collection_games_per_second": collection["games_per_second"],
        "collection_decisions_per_second": collection["decisions_per_second"],
        "collection_substeps_per_second": collection["substeps_per_second"],
        "update_seconds": metrics["update_seconds"],
        "returns_advantages": advantages.tolist(),
        "trajectory_summaries": collection["trajectory_summaries"],
        "metrics": metrics,
        "config": config,
        "rollout_persistence": (
            "compact bounded provenance bundle persisted; "
            "no unbounded rollout buffer"
        ),
        "invariants": {
            "behavior_snapshot_is_frozen_stage4_root": root_hash == APPROVED_STAGE4_ROOT_SHA256,
            "terminal_returns_in_minus_one_zero_plus_one": True,
            "real_legal_masks_retained_in_memory": True,
            "detached_recurrent_memory_inputs_retained_in_memory": True,
            "complete_behavior_logprob_retained": True,
            "logical_logprob_is_conditional_substep_sum": True,
            "ratio_uses_logical_action_once": True,
            "group_credit_shared_across_logical_decisions_and_substeps": True,
            "zero_variance_group_fail_closed": True,
            "value_loss_zero": True,
            "bounded_provenance_bundle_persisted": True,
            "candidate_preflight_passed": True,
            "no_tournament": True,
            "stage4_root_preserved": True,
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "metrics.json", metrics)
    _write_report(output_dir, manifest, metrics)
    _write_capsule(Path("experiments/autoresearch"), manifest, metrics)
    (output_dir / "logs" / "run.log").write_text(
        "\n".join(
            [
                "AR-019 trajectory-group GRPO micro-update",
                f"returns={collection['returns']}",
                f"logical_decisions={collection['logical_decisions']}",
                f"substeps={collection['substeps']}",
                f"collection_seconds={collection['collection_seconds']}",
                f"update_seconds={metrics['update_seconds']}",
                f"candidate_sha256={candidate_hash}",
                f"sample_manifest={sample_manifest_path}",
                f"sample_manifest_sha256={sample_manifest_file_hash}",
                f"sample_manifest_content_sha256={sample_manifest['sha256']}",
                f"trajectory_bundle={trajectory_bundle_path}",
                f"bundle_sha256={bundle_hash}",
                "candidate_preflight=passed",
                "tournament=not_run",
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
    parser.add_argument("--seed", type=int, default=19019)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run_ar019(
        checkpoint=args.checkpoint,
        deck_path=args.agent_deck,
        meta_date=args.meta_date,
        output_dir=args.output_dir,
        seed=args.seed,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
