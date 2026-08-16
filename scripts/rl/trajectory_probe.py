"""Collect a tiny, strict on-policy trajectory from the frozen Stage 4 root.

This module deliberately does not import the packed-data path. The selected
Parquet file is inspected for provenance only; environment observations are
encoded directly from ``CabtEnv``.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from rl.encoder.card_features import CardTable, get_card_table
from rl.encoder.encoding import SUBMIT_ACTION, TokenEncoder, build_mask
from rl.encoder.meta_lookup import get_meta_lookup
from rl.env.env import CabtEnv, random_opponent
from rl.policy_infer_torch import load_inference_checkpoint
from scripts.rl.ppo_micro_update import (
    build_sample_manifest,
    ppo_micro_update,
    save_candidate_checkpoint,
    save_compressed_bundle,
    validate_bundle,
)


DEFAULT_CHECKPOINT = Path("experiments/autoresearch/root/stage4_root.pkl")
DEFAULT_DECK = Path("agent/deck.csv")
DEFAULT_OUTPUT = Path("experiments/autoresearch/AR-009/logs")
APPROVED_STAGE4_ROOT_SHA256 = "b59daeab12cd9224a14f85989b5aa5821b5f27453092f7e3f408c24a166b840b"
MAX_GAMES_PER_MODE = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_tensor(value: torch.Tensor | None) -> str:
    if value is None:
        raise ValueError("trajectory memory digest cannot be computed from None")
    tensor = value.detach().to(device="cpu").contiguous()
    header = f"{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
    return sha256_bytes(header + tensor.numpy().tobytes(order="C"))


def load_deck(path: Path) -> list[int]:
    cards = [int(line.strip().rstrip(",")) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"agent deck must contain exactly 60 cards, got {len(cards)} from {path}")
    return cards


def deck_content_sha256(cards: list[int]) -> str:
    """Hash the normalized parsed 60-card content, not the source file bytes."""
    return sha256_bytes(json.dumps(cards, separators=(",", ":")).encode("ascii"))


def validate_meta_date(meta_date: str) -> str:
    try:
        parsed = dt.date.fromisoformat(meta_date)
    except ValueError as exc:
        raise ValueError(f"--meta-date must be a complete YYYY-MM-DD date, got {meta_date!r}") from exc
    normalized = parsed.isoformat()
    if normalized != meta_date:
        raise ValueError(f"--meta-date must use complete YYYY-MM-DD form, got {meta_date!r}")
    lookup = get_meta_lookup()
    day_id = lookup.resolve_day_id(meta_date)
    if day_id is None:
        raise ValueError(f"metadata date {meta_date!r} did not resolve to a catalog day")
    # This intentionally raises for the current catalog's incomplete day 31.
    lookup.day_index_norm(day_id)
    return meta_date


class DateBoundEncoder:
    """Inject the explicit metadata date when the engine omits it.

    The wrapper refuses a conflicting engine date. It is local to this probe and
    does not alter the process-wide MetaLookup singleton.
    """

    def __init__(self, base: TokenEncoder, meta_date: str) -> None:
        self.base = base
        self.meta_date = validate_meta_date(meta_date)

    @property
    def int_keys(self) -> set[str]:
        return self.base.int_keys

    @property
    def shapes(self) -> dict[str, tuple[int, ...]]:
        return self.base.shapes

    def encode(
        self,
        obs: dict[str, Any],
        picked: set[int] | None = None,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        current = obs.get("current")
        if not isinstance(current, dict):
            raise ValueError("engine observation has no complete current metadata object")
        observed_dates = {
            field: current.get(field)
            for field in ("date", "archive_date")
            if current.get(field) is not None
        }
        conflicting = {
            field: value
            for field, value in observed_dates.items()
            if value != self.meta_date
        }
        if conflicting:
            raise ValueError(
                f"engine metadata date fields {conflicting!r} conflicts with explicit "
                f"--meta-date {self.meta_date!r}"
            )
        if len(set(observed_dates.values())) > 1:
            raise ValueError(
                f"engine current.date and current.archive_date disagree: {observed_dates!r}"
            )
        if not observed_dates:
            current = dict(current)
            current["date"] = self.meta_date
            current["archive_date"] = self.meta_date
            bound_obs = dict(obs)
            bound_obs["current"] = current
        else:
            bound_obs = obs
        return self.base.encode(bound_obs, picked, **kwargs)


def inspect_parquet_provenance(path: Path) -> dict[str, Any]:
    """Read Parquet metadata only. No rows are loaded and no packed store is used."""
    if path.suffix != ".parquet":
        raise ValueError(f"Parquet context must have a .parquet suffix, got {path}")
    if not path.is_file():
        raise FileNotFoundError(f"default Parquet context not found: {path}")
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(parquet.metadata.num_rows),
        "columns": list(parquet.schema_arrow.names),
        "used_for_model_input": False,
        "used_as_metadata_provenance_only": True,
        "packed": False,
    }


def load_stage4(checkpoint: Path, card_table: CardTable):
    """Load the root through the one approved strict inference entry point."""
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage4 checkpoint not found: {checkpoint}")
    observed_hash = sha256_file(checkpoint)
    if observed_hash != APPROVED_STAGE4_ROOT_SHA256:
        raise ValueError(
            "Stage4 checkpoint hash is not the approved frozen root: "
            f"{observed_hash} != {APPROVED_STAGE4_ROOT_SHA256}"
        )
    # Do not catch this call: load_inference_checkpoint is strict and must fail loudly.
    return load_inference_checkpoint(checkpoint, card_table)


def as_model_input(encoded: dict[str, np.ndarray], int_keys: set[str]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        array = np.asarray(value)
        dtype = np.int64 if key in int_keys else np.float32
        result[key] = torch.as_tensor(array.astype(dtype, copy=False)).reshape(1, *array.shape)
    return result


def initial_memory(model: Any) -> torch.Tensor:
    learned_init = getattr(model, "learned_init", None)
    if learned_init is None:
        raise AttributeError("Stage4 model has no learned_init recurrent state")
    return learned_init.detach().to(dtype=torch.float32).unsqueeze(0).clone()


def _masked_distribution(logits: torch.Tensor, action_mask: np.ndarray) -> torch.distributions.Categorical:
    flat = logits.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    legal = torch.as_tensor(np.asarray(action_mask).reshape(-1) >= 0.5, dtype=torch.bool)
    if legal.numel() != flat.numel():
        raise ValueError(f"model logits/action mask size mismatch: {flat.numel()} != {legal.numel()}")
    if not bool(legal.any()):
        raise RuntimeError("environment emitted an all-illegal action mask")
    masked = flat.masked_fill(~legal, float("-inf"))
    return torch.distributions.Categorical(logits=masked)


def _sample_distribution(
    distribution: torch.distributions.Categorical,
    generator: torch.Generator,
) -> int:
    """Sample on a local generator so unrelated engine code cannot advance it."""
    return int(torch.multinomial(distribution.probs, 1, generator=generator).item())


def _mirror_opponent(
    model: Any,
    encoder: DateBoundEncoder,
    rng: np.random.Generator,
) -> Callable[..., list[int]]:
    """Build a frozen-weight mirror with an explicit no-memory boundary."""

    def choose(raw_obs: dict[str, Any], _rng: random.Random, *, deck=None, tracker=None, ability_slots=None, **_) -> list[int]:
        select = raw_obs["select"]
        options = select.get("option") or []
        picked: set[int] = set()
        result: list[int] = []
        for _substep in range(int(select.get("maxCount", 1))):
            encoded = encoder.encode(
                raw_obs,
                picked=picked,
                self_deck=deck,
                tracker=tracker,
                ability_slots=ability_slots,
            )
            model_input = as_model_input(encoded, encoder.int_keys)
            with torch.inference_mode():
                logits, _value, _memory_out = model.logits_value(model_input, memory_in=None)
            distribution = _masked_distribution(logits, encoded["action_mask"])
            probabilities = distribution.probs.numpy()
            action = int(rng.choice(len(probabilities), p=probabilities))
            if action == SUBMIT_ACTION:
                break
            if action >= len(options) or action in picked:
                raise AssertionError("mirror sampled an illegal action")
            picked.add(action)
            result.append(action)
            if len(result) >= int(select.get("maxCount", 1)):
                break
        min_count = int(select.get("minCount", 1))
        if len(result) < min_count:
            raise RuntimeError("mirror policy submitted fewer than the engine minimum")
        return result

    return choose


def collect_episode(
    env: CabtEnv,
    model: Any,
    encoder: DateBoundEncoder,
    episode_id: str,
    opponent_mode: str,
    reset_seed: int,
    deck_content_hash: str,
    deck_source_file_hash: str,
    model_hash: str,
    action_generator: torch.Generator,
    bundle: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    observation, reset_info = env.reset(seed=reset_seed)
    side = int(reset_info["agent_index"])
    memory = initial_memory(model)
    initial_memory_digest = digest_tensor(memory)
    rows: list[dict[str, Any]] = []
    env_step = 0
    decision_index = 0
    picked: set[int] = set()
    decision_memory_in = memory
    decision_substep = 0
    done = False

    while not done:
        encoded = observation
        action_mask = np.asarray(encoded["action_mask"], dtype=np.float32).copy()
        mask_digest = sha256_bytes(action_mask.tobytes(order="C"))
        model_input = as_model_input(encoded, encoder.int_keys)
        memory_input = decision_memory_in
        with torch.inference_mode():
            logits, value, memory_out = model.logits_value(model_input, memory_in=memory_input)
        distribution = _masked_distribution(logits, action_mask)
        action = _sample_distribution(distribution, action_generator)
        legal = bool(action_mask[action] >= 0.5)
        if not legal:
            raise AssertionError(f"sampled illegal action {action}")
        logprob = float(distribution.log_prob(torch.tensor(action)).item())
        entropy = float(distribution.entropy().item())
        if not math.isfinite(logprob):
            raise AssertionError("sampled action has no finite logprob")

        select = getattr(env, "_obs", {}).get("select") or {}
        max_count = int(select.get("maxCount", 1))
        selected_action = action != SUBMIT_ACTION
        if selected_action:
            picked.add(action)
        next_observation, reward, terminated, truncated, _info = env.step(action)
        done = bool(terminated or truncated)
        decision_complete = action == SUBMIT_ACTION or len(picked) >= max_count or done
        sample_index = len(bundle)
        rows.append(
            {
                "sample_index": sample_index,
                "episode_id": episode_id,
                "env_step": env_step,
                "decision_index": decision_index,
                "substep": decision_substep,
                "side": side,
                "action": action,
                "legal_action_mask_digest": mask_digest,
                "action_logprob": logprob,
                "entropy": entropy,
                "value": float(value.detach().reshape(-1)[0].item()),
                "reward": float(reward),
                "terminal": bool(terminated),
                "done": bool(done),
                "truncated": bool(truncated),
                "memory_input_digest": digest_tensor(memory_input),
                "memory_output_digest": digest_tensor(memory_out),
                "model_input_digests": [
                    {"name": key, "sha256": digest_tensor(value)}
                    for key, value in model_input.items()
                ],
                "opponent_mode": opponent_mode,
                "deck_content_sha256": deck_content_hash,
                "deck_source_file_sha256": deck_source_file_hash,
                "model_hash": model_hash,
                "metadata_date": encoder.meta_date,
            }
        )
        bundle.append(
            {
                "sample_index": sample_index,
                "episode_id": episode_id,
                "env_step": env_step,
                "decision_index": decision_index,
                "substep": decision_substep,
                "model_input": {
                    key: value.detach().to(device="cpu").clone()
                    for key, value in model_input.items()
                },
                "action_mask": torch.as_tensor(action_mask, dtype=torch.float32).clone(),
                "memory_input": memory_input.detach().to(device="cpu", dtype=torch.float32).clone(),
                "action": action,
                "behavior_logprob": logprob,
                "value": float(value.detach().reshape(-1)[0].item()),
                "reward": float(reward),
                "done": bool(done),
            }
        )
        env_step += 1
        observation = next_observation
        if truncated:
            raise RuntimeError(f"episode {episode_id} truncated before terminal reward")
        if decision_complete:
            memory = memory_out.detach().to(dtype=torch.float32).clone()
            decision_memory_in = memory
            picked = set()
            decision_index += 1
            decision_substep = 0
        else:
            decision_substep += 1

    terminal_rows = [row for row in rows if row["terminal"]]
    if len(terminal_rows) != 1:
        raise AssertionError(f"episode {episode_id} has {len(terminal_rows)} terminal rows, expected one")
    if rows[-1]["done"] is not True:
        raise AssertionError(f"episode {episode_id} did not end with done=True")
    if initial_memory_digest != rows[0]["memory_input_digest"]:
        raise AssertionError(f"episode {episode_id} did not reset recurrent memory at boundary")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise AssertionError("trajectory is empty")
    episodes = Counter(row["episode_id"] for row in rows)
    for episode_id, count in episodes.items():
        episode_rows = [row for row in rows if row["episode_id"] == episode_id]
        if count != len(episode_rows):
            raise AssertionError("episode row count mismatch")
        if sum(bool(row["terminal"]) for row in episode_rows) != 1:
            raise AssertionError(f"episode {episode_id} has no unique terminal row")
        if not episode_rows[-1]["done"]:
            raise AssertionError(f"episode {episode_id} does not end in done=True")
        for row in episode_rows:
            if not row["legal_action_mask_digest"]:
                raise AssertionError("missing action-mask digest")
            if not math.isfinite(float(row["action_logprob"])):
                raise AssertionError("missing action logprob")
            if not row["memory_input_digest"] or not row["memory_output_digest"]:
                raise AssertionError("missing recurrent memory digest")
        if not any(row["terminal"] and "reward" in row for row in episode_rows):
            raise AssertionError(f"episode {episode_id} has no terminal reward row")
    for episode_id in episodes:
        decisions: dict[int, list[dict[str, Any]]] = {}
        for row in [item for item in rows if item["episode_id"] == episode_id]:
            decisions.setdefault(int(row["decision_index"]), []).append(row)
        for decision, decision_rows in decisions.items():
            expected = list(range(len(decision_rows)))
            actual = [int(row["substep"]) for row in decision_rows]
            if actual != expected:
                raise AssertionError(f"episode {episode_id} decision {decision} has broken substep order")


def write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    manifest = dict(manifest)
    manifest["row_count"] = len(rows)
    manifest["trajectory_sha256"] = sha256_bytes(jsonl.encode("utf-8"))
    (output_dir / "trajectory.jsonl").write_text(jsonl)
    (output_dir / "trajectory.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    lines = [
        f"{manifest.get('experiment', 'AR-009')} Stage4 PPO micro-update probe",
        f"metadata_date={manifest['metadata_date']}",
        f"model_sha256={manifest['model_sha256']}",
        f"deck_content_sha256={manifest.get('deck_content_sha256', manifest.get('deck_sha256', ''))}",
        f"deck_source_file_sha256={manifest.get('deck_source_file_sha256', '')}",
        f"rows={manifest['row_count']}",
        f"trajectory_sha256={manifest['trajectory_sha256']}",
        f"sample_manifest_sha256={manifest.get('sample_manifest_sha256', '')}",
        f"sample_manifest_content_sha256={manifest.get('sample_manifest_content_sha256', '')}",
        f"bundle_sha256={manifest.get('bundle_sha256', '')}",
        f"candidate_sha256={manifest.get('candidate_sha256', '')}",
    ]
    for mode, counts in manifest["counts_by_mode"].items():
        lines.append(f"mode={mode} episodes={counts['episodes']} rows={counts['rows']} terminals={counts['terminals']}")
    (output_dir / "trajectory.log").write_text("\n".join(lines) + "\n")


def run_probe(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    deck_path: Path = DEFAULT_DECK,
    meta_date: str,
    output_dir: Path = DEFAULT_OUTPUT,
    games_per_mode: int = 1,
    seed: int = 8008,
    experiment: str = "AR-009",
) -> dict[str, Any]:
    if games_per_mode < 1 or games_per_mode > MAX_GAMES_PER_MODE:
        raise ValueError(f"games_per_mode must be between 1 and {MAX_GAMES_PER_MODE} for this bounded probe")
    validate_meta_date(meta_date)
    deck = load_deck(deck_path)
    parquet_path = Path("data/bc_data") / f"{meta_date}.parquet"
    parquet_provenance = inspect_parquet_provenance(parquet_path)
    card_table = get_card_table()
    model, model_metadata = load_stage4(checkpoint, card_table)
    encoder = DateBoundEncoder(TokenEncoder(card_table), meta_date)
    model_hash = sha256_file(checkpoint)
    deck_content_hash = deck_content_sha256(deck)
    deck_source_file_hash = sha256_file(deck_path)

    np.random.seed(seed)
    random.seed(seed)
    rows: list[dict[str, Any]] = []
    bundle: list[dict[str, Any]] = []
    counts_by_mode: dict[str, dict[str, int]] = {}
    started = time.perf_counter()
    for mode_index, mode in enumerate(("random", "mirror_no_memory")):
        mode_rows: list[dict[str, Any]] = []
        mirror = _mirror_opponent(model, encoder, np.random.default_rng(seed + 1000 + mode_index))
        for game_index in range(games_per_mode):
            opponent_fn = random_opponent if mode == "random" else mirror
            env = CabtEnv(
                agent_deck=deck,
                opponent_deck=deck,
                opponent_fn=opponent_fn,
                encoder=encoder,
                seed=seed + mode_index * 100 + game_index,
                max_steps=4000,
            )
            try:
                action_generator = torch.Generator(device="cpu").manual_seed(
                    seed + mode_index * 100 + game_index
                )
                episode_rows = collect_episode(
                    env,
                    model,
                    encoder,
                    f"{mode}-{game_index:03d}",
                    mode,
                    seed + mode_index * 100 + game_index,
                    deck_content_hash,
                    deck_source_file_hash,
                    model_hash,
                    action_generator,
                    bundle,
                )
            finally:
                env.close()
            mode_rows.extend(episode_rows)
        rows.extend(mode_rows)
        counts_by_mode[mode] = {
            "episodes": games_per_mode,
            "rows": len(mode_rows),
            "terminals": sum(bool(row["terminal"]) for row in mode_rows),
        }
    validate_rows(rows)
    validate_bundle(bundle, rows)
    elapsed = time.perf_counter() - started
    sample_manifest = build_sample_manifest(
        bundle,
        root_sha256=model_hash,
        metadata_date=meta_date,
        deck_content_sha256=deck_content_hash,
        deck_source_file_sha256=deck_source_file_hash,
    )
    experiment_dir = output_dir.parent
    sample_manifest_path = experiment_dir / "sample.manifest.json"
    sample_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sample_manifest_path.write_text(json.dumps(sample_manifest, indent=2, sort_keys=True) + "\n")
    sample_manifest_file_hash = sha256_file(sample_manifest_path)
    bundle_path = experiment_dir / "trajectory_bundle.pt.gz"
    bundle_hash = save_compressed_bundle(bundle_path, bundle, sample_manifest)
    root_reference = copy.deepcopy(model).eval()
    ppo_config: dict[str, Any] = {
        "algorithm": "PPO",
        "epochs": 1,
        "gamma": 1.0,
        "clip_epsilon": 0.2,
        "learning_rate": 1e-5,
        "value_coefficient": 0.5,
        "entropy_coefficient": 0.0,
        "precision": "FP32",
        "sample_modes": ["random", "mirror_no_memory"],
        "truncated_bptt_boundary": "memory input detached per environment substep",
    }
    ppo_metrics = ppo_micro_update(
        model,
        root_reference,
        bundle,
        gamma=float(ppo_config["gamma"]),
        clip_epsilon=float(ppo_config["clip_epsilon"]),
        learning_rate=float(ppo_config["learning_rate"]),
        value_coefficient=float(ppo_config["value_coefficient"]),
        entropy_coefficient=float(ppo_config["entropy_coefficient"]),
    )
    candidate_path = experiment_dir / "candidate.pt"
    candidate_hash = save_candidate_checkpoint(
        candidate_path,
        model,
        model_metadata,
        root_sha256=model_hash,
        sample_manifest_sha256=sample_manifest_file_hash,
        bundle_sha256=bundle_hash,
        config=ppo_config,
        diagnostics=ppo_metrics,
        sample_manifest_content_sha256=str(sample_manifest["sha256"]),
        experiment=experiment,
    )
    manifest = {
        "format": "ptcg-stage4-ppo-micro-update-v1",
        "experiment": experiment,
        "metadata_date": meta_date,
        "checkpoint": str(checkpoint),
        "model_sha256": model_hash,
        "deck": str(deck_path),
        "deck_content_sha256": deck_content_hash,
        "deck_source_file_sha256": deck_source_file_hash,
        "opponent_modes": ["random", "mirror_no_memory"],
        "counts_by_mode": counts_by_mode,
        "episodes": games_per_mode * 2,
        "collection_seconds": round(elapsed, 6),
        "rows_per_second": round(len(rows) / elapsed, 3) if elapsed else None,
        "parquet_provenance": parquet_provenance,
        "packed_used": False,
        "sample_manifest_sha256": sample_manifest_file_hash,
        "sample_manifest_content_sha256": sample_manifest["sha256"],
        "sample_manifest": str(sample_manifest_path),
        "bundle": str(bundle_path),
        "bundle_sha256": bundle_hash,
        "candidate": str(candidate_path),
        "candidate_sha256": candidate_hash,
        "ppo_config": ppo_config,
        "ppo_metrics": ppo_metrics,
        "model_metadata": {
            "arch_version": model_metadata.get("arch_version"),
            "token_schema_version": model_metadata.get("token_schema_version"),
            "scratch_registers": model_metadata.get("scratch_registers"),
        },
        "invariants": {
            "actions_legal": True,
            "multi_select_substeps_logged": True,
            "finite_logprobs": True,
            "memory_reset_at_episode_boundary": True,
            "terminal_reward_present": True,
            "policy_action_rng_isolated": True,
            "bundle_has_model_inputs": True,
            "bundle_has_real_action_masks": True,
            "bundle_has_detached_memory_inputs": True,
            "ppo_terminal_return_propagation": True,
            "ppo_finite_normalized_advantages": True,
            "ppo_clipped_ratio_objective": True,
            "ppo_one_epoch_only": True,
            "candidate_strict_inference_checkpoint": True,
            "fixed_seed_reproducibility_scope": "conditional_on_identical_engine_observations",
        },
        "determinism": {
            "policy_action_generator": "torch.Generator seeded independently per episode",
            "full_episode_replay": "not guaranteed by CabtEnv seed; engine BattleStart has no exposed seed",
        },
    }
    write_outputs(output_dir, rows, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--agent-deck", type=Path, default=DEFAULT_DECK)
    parser.add_argument("--meta-date", required=True, help="complete YYYY-MM-DD metadata date")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--games-per-mode",
        type=int,
        default=1,
        choices=range(1, MAX_GAMES_PER_MODE + 1),
        help="episodes per opponent mode (1-4; modes and opponent semantics are unchanged)",
    )
    parser.add_argument("--seed", type=int, default=8008)
    parser.add_argument("--experiment", default="AR-009")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = run_probe(
        checkpoint=args.checkpoint,
        deck_path=args.agent_deck,
        meta_date=args.meta_date,
        output_dir=args.output_dir,
        games_per_mode=args.games_per_mode,
        seed=args.seed,
        experiment=args.experiment,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
