"""Bounded FP32 PPO update helpers for the Stage4 autoresearch probe.

The bundle is deliberately independent of the AR-008 JSONL.  It retains the
actual tensors consumed by the policy, the real legal masks, detached
recurrent inputs, and all PPO scalars in collection order.  The update uses a
detached recurrent state at every sample, which is the explicit truncated-BPTT
boundary for this micro-experiment.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import torch

from rl.policy_infer_torch import TORCH_INFERENCE_FORMAT


BUNDLE_FORMAT = "ptcg-stage4-ppo-bundle-v1"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_digest(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    header = f"{tuple(tensor.shape)}|{tensor.dtype}".encode("ascii")
    return _sha256_bytes(header + tensor.numpy().tobytes(order="C"))


def _mask_digest(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return _sha256_bytes(tensor.numpy().tobytes(order="C"))


def _model_input_digests(model_input: dict[str, torch.Tensor]) -> list[dict[str, str]]:
    """Digest every model input in the encoder's insertion/collection order."""
    return [
        {"name": name, "sha256": _tensor_digest(tensor)}
        for name, tensor in model_input.items()
    ]


def _manifest_content_sha256(manifest: dict[str, Any]) -> str:
    content = dict(manifest)
    content.pop("sha256", None)
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(canonical)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: torch.Tensor | float) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all().item())
    return math.isfinite(float(value))


def validate_bundle(bundle: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    """Validate tensor shape, collection order, legality, and row digests."""
    if not bundle:
        raise ValueError("PPO bundle is empty")
    if len(bundle) != len(rows):
        raise ValueError(f"PPO bundle/row count mismatch: {len(bundle)} != {len(rows)}")

    for index, (sample, row) in enumerate(zip(bundle, rows, strict=True)):
        if sample.get("sample_index") != index:
            raise ValueError(f"PPO bundle order mismatch at sample {index}")
        for key in (
            "model_input",
            "action_mask",
            "memory_input",
            "action",
            "behavior_logprob",
            "value",
            "reward",
            "done",
        ):
            if key not in sample:
                raise ValueError(f"PPO bundle sample {index} is missing {key!r}")

        model_input = sample["model_input"]
        if not isinstance(model_input, dict) or not model_input:
            raise ValueError(f"PPO bundle sample {index} has no model input")
        batch_sizes = set()
        for name, tensor in model_input.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"PPO model input {name!r} is not a tensor")
            if tensor.ndim < 1:
                raise ValueError(f"PPO model input {name!r} has no batch dimension")
            batch_sizes.add(int(tensor.shape[0]))
        if batch_sizes != {1}:
            raise ValueError(f"PPO model inputs must retain batch size 1, got {batch_sizes}")

        action_mask = sample["action_mask"]
        memory_input = sample["memory_input"]
        if not isinstance(action_mask, torch.Tensor) or action_mask.ndim != 1:
            raise ValueError(f"PPO action mask {index} must be a 1D tensor")
        if not isinstance(memory_input, torch.Tensor) or memory_input.ndim != 3:
            raise ValueError(f"PPO memory input {index} must be [1, scratch, d]")
        if memory_input.shape[0] != 1:
            raise ValueError(f"PPO memory input {index} must retain batch size 1")
        input_mask = model_input.get("action_mask")
        if input_mask is None or not torch.equal(
            input_mask.reshape(-1).to(dtype=torch.float32), action_mask
        ):
            raise ValueError(f"PPO sample {index} real action mask disagrees with model input")
        action = int(sample["action"])
        if action < 0 or action >= action_mask.numel():
            raise ValueError(f"PPO sample {index} action is outside the mask")
        if float(action_mask[action].item()) < 0.5:
            raise ValueError(f"PPO sample {index} contains an illegal action")
        for scalar_name in ("behavior_logprob", "value", "reward"):
            if not _finite(float(sample[scalar_name])):
                raise ValueError(f"PPO sample {index} has a non-finite {scalar_name}")
        if not isinstance(sample["done"], bool):
            raise TypeError(f"PPO sample {index} done flag is not bool")

        if sample["episode_id"] != row["episode_id"] or sample["env_step"] != row["env_step"]:
            raise ValueError(f"PPO sample {index} is not linked to its row")
        if row.get("sample_index") != index:
            raise ValueError(f"trajectory row order mismatch at sample {index}")
        if row.get("legal_action_mask_digest") != _mask_digest(action_mask):
            raise ValueError(f"PPO sample {index} action-mask digest mismatch")
        if row.get("memory_input_digest") != _tensor_digest(memory_input):
            raise ValueError(f"PPO sample {index} memory-input digest mismatch")
        expected_input_digests = _model_input_digests(model_input)
        if row.get("model_input_digests") != expected_input_digests:
            raise ValueError(f"PPO sample {index} model-input digest mismatch")
        if bool(row.get("done")) != sample["done"]:
            raise ValueError(f"PPO sample {index} done flag disagrees with row")


def build_sample_manifest(
    bundle: list[dict[str, Any]],
    *,
    root_sha256: str,
    metadata_date: str,
    deck_content_sha256: str,
    deck_source_file_sha256: str,
) -> dict[str, Any]:
    """Create a deterministic, tensor-digest-linked sample manifest."""
    samples = []
    for sample in bundle:
        samples.append(
            {
                "sample_index": int(sample["sample_index"]),
                "episode_id": sample["episode_id"],
                "env_step": int(sample["env_step"]),
                "decision_index": int(sample["decision_index"]),
                "substep": int(sample["substep"]),
                "action": int(sample["action"]),
                "action_mask_digest": _mask_digest(sample["action_mask"]),
                "memory_input_digest": _tensor_digest(sample["memory_input"]),
                "model_input_digests": _model_input_digests(sample["model_input"]),
                "behavior_logprob": float(sample["behavior_logprob"]),
                "value": float(sample["value"]),
                "reward": float(sample["reward"]),
                "done": bool(sample["done"]),
            }
        )
    manifest = {
        "format": BUNDLE_FORMAT,
        "root_sha256": root_sha256,
        "metadata_date": metadata_date,
        "deck_content_sha256": deck_content_sha256,
        "deck_source_file_sha256": deck_source_file_sha256,
        "sample_count": len(samples),
        "order": samples,
        "truncated_bptt": {
            "memory_input": "detached per collected environment substep",
            "gradient_boundary": "no gradient crosses recurrent memory input",
        },
    }
    manifest["sha256"] = _manifest_content_sha256(manifest)
    return manifest


def save_compressed_bundle(path: Path, bundle: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    """Persist the complete PPO input bundle as deterministic gzip-compressed torch data."""
    payload = {
        "format": BUNDLE_FORMAT,
        "manifest": manifest,
        "samples": bundle,
    }
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    compressed = gzip.compress(buffer.getvalue(), compresslevel=9, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(compressed)
    return _sha256_bytes(compressed)


def discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Propagate terminal returns backwards, resetting at every done flag."""
    if rewards.ndim != 1 or dones.ndim != 1 or rewards.shape != dones.shape:
        raise ValueError("rewards and dones must be equal-length 1D tensors")
    result = torch.zeros_like(rewards, dtype=torch.float32)
    running = torch.tensor(0.0, dtype=torch.float32)
    for index in range(rewards.numel() - 1, -1, -1):
        if bool(dones[index].item()):
            running = torch.tensor(0.0, dtype=torch.float32)
        running = rewards[index].to(torch.float32) + float(gamma) * running
        result[index] = running
    if not _finite(result):
        raise ValueError("discounted returns are non-finite")
    return result


def _batch_bundle(bundle: list[dict[str, Any]]) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    keys = tuple(bundle[0]["model_input"])
    if any(tuple(sample["model_input"]) != keys for sample in bundle):
        raise ValueError("PPO model input key order differs across samples")
    model_input = {
        key: torch.cat([sample["model_input"][key] for sample in bundle], dim=0)
        for key in keys
    }
    masks = torch.stack([sample["action_mask"] for sample in bundle])
    memories = torch.cat([sample["memory_input"] for sample in bundle], dim=0)
    actions = torch.tensor([int(sample["action"]) for sample in bundle], dtype=torch.long)
    old_logprobs = torch.tensor(
        [float(sample["behavior_logprob"]) for sample in bundle], dtype=torch.float32
    )
    values = torch.tensor([float(sample["value"]) for sample in bundle], dtype=torch.float32)
    rewards = torch.tensor([float(sample["reward"]) for sample in bundle], dtype=torch.float32)
    dones = torch.tensor([bool(sample["done"]) for sample in bundle], dtype=torch.bool)
    if not _finite(old_logprobs) or not _finite(values) or not _finite(rewards):
        raise ValueError("PPO scalars are non-finite")
    return model_input, masks, memories, actions, old_logprobs, values, rewards, dones


def _masked_distribution(logits: torch.Tensor, masks: torch.Tensor) -> torch.distributions.Categorical:
    if logits.shape != masks.shape:
        raise ValueError(f"PPO logits/masks shape mismatch: {logits.shape} != {masks.shape}")
    masked = logits.masked_fill(masks < 0.5, float("-inf"))
    if not bool((masks >= 0.5).any(dim=1).all().item()):
        raise ValueError("PPO batch contains an all-illegal action mask")
    return torch.distributions.Categorical(logits=masked)


def ppo_micro_update(
    model: torch.nn.Module,
    root_reference: torch.nn.Module,
    bundle: list[dict[str, Any]],
    *,
    gamma: float = 1.0,
    clip_epsilon: float = 0.2,
    learning_rate: float = 1e-5,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.0,
) -> dict[str, float | int | str]:
    """Run exactly one full-bundle FP32 PPO optimizer epoch."""
    model_input, masks, memories, actions, old_logprobs, old_values, rewards, dones = _batch_bundle(bundle)
    returns = discounted_returns(rewards, dones, gamma)
    raw_advantages = returns - old_values
    advantage_std = raw_advantages.std(unbiased=False)
    if float(advantage_std.item()) > 1e-8:
        advantages = (raw_advantages - raw_advantages.mean()) / advantage_std
    else:
        advantages = torch.zeros_like(raw_advantages)
    if not _finite(advantages) or not _finite(returns):
        raise ValueError("PPO returns or normalized advantages are non-finite")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    logits, values, _memory_out = model.logits_value(model_input, memory_in=memories.detach())
    distribution = _masked_distribution(logits, masks)
    logprobs = distribution.log_prob(actions)
    ratio = torch.exp(logprobs - old_logprobs)
    surrogate_one = ratio * advantages.detach()
    surrogate_two = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages.detach()
    policy_loss = -torch.minimum(surrogate_one, surrogate_two).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns.detach())
    entropy = distribution.entropy().mean()
    loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
    if not _finite(loss):
        raise ValueError("PPO loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not _finite(grad_norm):
        raise ValueError("PPO gradient norm is non-finite")
    optimizer.step()

    model.eval()
    root_reference.eval()
    with torch.no_grad():
        root_logits, _root_values, _ = root_reference.logits_value(
            model_input, memory_in=memories.detach()
        )
        candidate_logits, _candidate_values, _ = model.logits_value(
            model_input, memory_in=memories.detach()
        )
        root_distribution = _masked_distribution(root_logits, masks)
        candidate_distribution = _masked_distribution(candidate_logits, masks)
        root_kl = torch.distributions.kl_divergence(root_distribution, candidate_distribution)

    squared_delta = 0.0
    max_abs_delta = 0.0
    changed_parameters = 0
    parameter_count = 0
    for candidate, root in zip(model.parameters(), root_reference.parameters(), strict=True):
        delta = candidate.detach().to(torch.float32) - root.detach().to(torch.float32)
        squared_delta += float(torch.sum(delta * delta).item())
        max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        changed_parameters += int(torch.count_nonzero(delta).item())
        parameter_count += delta.numel()

    metrics: dict[str, float | int | str] = {
        "samples": int(len(bundle)),
        "epochs": 1,
        "gamma": float(gamma),
        "clip_epsilon": float(clip_epsilon),
        "learning_rate": float(learning_rate),
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "value_loss": float(value_loss.detach().item()),
        "entropy": float(entropy.detach().item()),
        "gradient_norm": float(grad_norm.detach().item()),
        "ratio_mean": float(ratio.detach().mean().item()),
        "ratio_min": float(ratio.detach().min().item()),
        "ratio_max": float(ratio.detach().max().item()),
        "return_min": float(returns.min().item()),
        "return_max": float(returns.max().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()),
        "root_reference_kl_mean": float(root_kl.mean().item()),
        "root_reference_parameter_l2": math.sqrt(squared_delta),
        "root_reference_parameter_max_abs": max_abs_delta,
        "root_reference_changed_parameters": int(changed_parameters),
        "root_reference_parameter_count": int(parameter_count),
        "memory_boundary": "detached recurrent state input per sample; no cross-step gradient",
    }
    if not all(_finite(value) for value in metrics.values() if isinstance(value, (float, int))):
        raise ValueError("PPO diagnostics are non-finite")
    return metrics


def save_candidate_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_metadata: dict[str, Any],
    *,
    root_sha256: str,
    sample_manifest_sha256: str,
    bundle_sha256: str,
    config: dict[str, Any],
    diagnostics: dict[str, Any],
    sample_manifest_content_sha256: str | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> str:
    """Save a strict portable inference checkpoint with experiment provenance."""
    arch_config = {
        key: value
        for key, value in model_metadata.items()
        if key not in {"inference_config", "training_config", "static_feature_contract"}
    }
    model_state = {
        key: value.detach().to(device="cpu", dtype=torch.float32).clone()
        for key, value in model.state_dict().items()
    }
    static_features = getattr(model, "card_feat", None)
    payload = {
        "format": TORCH_INFERENCE_FORMAT,
        "arch_config": arch_config,
        "inference_config": dict(model_metadata["inference_config"]),
        "training_config": dict(model_metadata.get("training_config", {})),
        "static_card_features": (
            static_features.detach().to(device="cpu", dtype=torch.float32).clone()
            if static_features is not None
            else None
        ),
        "static_feature_contract": model_metadata.get("static_feature_contract"),
        "state_dict": model_state,
        "autoresearch": {
            "experiment": "AR-009",
            "root_sha256": root_sha256,
            # These are file-byte hashes. The manifest's own ``sha256`` is a
            # separate canonical-content digest recorded below.
            "sample_manifest_sha256": sample_manifest_sha256,
            "bundle_sha256": bundle_sha256,
            "sample_manifest_content_sha256": sample_manifest_content_sha256,
            "artifacts": dict(artifact_paths or {
                "sample_manifest": "sample.manifest.json",
                "trajectory_bundle": "trajectory_bundle.pt.gz",
            }),
            "config": config,
            "diagnostics": diagnostics,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return _sha256_bytes(path.read_bytes())


def _resolve_artifact_path(candidate_path: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"candidate provenance is missing {label} path")
    path = Path(value)
    return path if path.is_absolute() else candidate_path.parent / path


def _load_bundle_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"candidate provenance artifact is missing: {path}")
    try:
        raw = gzip.decompress(path.read_bytes())
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"candidate trajectory bundle cannot be loaded: {path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != BUNDLE_FORMAT:
        raise ValueError("candidate trajectory bundle format is unsupported")
    return payload


def validate_candidate_provenance(
    candidate_path: Path,
    *,
    approved_root_sha256: str,
) -> dict[str, Path]:
    """Fail closed before model loading if AR candidate evidence is detached."""
    candidate_path = Path(candidate_path)
    if not candidate_path.is_file():
        raise FileNotFoundError(f"candidate checkpoint does not exist: {candidate_path}")
    try:
        payload = torch.load(candidate_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"candidate checkpoint cannot be loaded: {candidate_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("candidate checkpoint payload must be an object")
    provenance = payload.get("autoresearch")
    if not isinstance(provenance, dict):
        raise ValueError("candidate checkpoint has no autoresearch provenance")
    if provenance.get("root_sha256") != approved_root_sha256:
        raise ValueError(
            "candidate root_sha256 does not match the approved frozen Stage4 root: "
            f"{provenance.get('root_sha256')!r} != {approved_root_sha256!r}"
        )
    sample_manifest_hash = provenance.get("sample_manifest_sha256")
    bundle_hash = provenance.get("bundle_sha256")
    if not isinstance(sample_manifest_hash, str) or len(sample_manifest_hash) != 64:
        raise ValueError("candidate provenance is missing the sample-manifest file hash")
    if not isinstance(bundle_hash, str) or len(bundle_hash) != 64:
        raise ValueError("candidate provenance is missing the trajectory-bundle file hash")
    artifacts = provenance.get("artifacts")
    if artifacts is None:
        artifacts = {}
    if not isinstance(artifacts, dict):
        raise ValueError("candidate provenance artifacts must be an object")
    sample_manifest_path = _resolve_artifact_path(
        candidate_path,
        artifacts.get("sample_manifest", "sample.manifest.json"),
        "sample-manifest",
    )
    bundle_path = _resolve_artifact_path(
        candidate_path,
        artifacts.get("trajectory_bundle", "trajectory_bundle.pt.gz"),
        "trajectory-bundle",
    )
    if not sample_manifest_path.is_file():
        raise FileNotFoundError(f"candidate provenance sample manifest is missing: {sample_manifest_path}")
    if not bundle_path.is_file():
        raise FileNotFoundError(f"candidate provenance trajectory bundle is missing: {bundle_path}")
    if sha256_file(sample_manifest_path) != sample_manifest_hash:
        raise ValueError("candidate sample-manifest file hash mismatch")
    if sha256_file(bundle_path) != bundle_hash:
        raise ValueError("candidate trajectory-bundle file hash mismatch")
    try:
        manifest = json.loads(sample_manifest_path.read_text())
    except Exception as exc:
        raise ValueError("candidate sample manifest is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != BUNDLE_FORMAT:
        raise ValueError("candidate sample manifest format is unsupported")
    if manifest.get("root_sha256") != approved_root_sha256:
        raise ValueError("candidate sample manifest root provenance mismatch")
    if manifest.get("sha256") != _manifest_content_sha256(manifest):
        raise ValueError("candidate sample manifest content hash mismatch")
    content_hash = provenance.get("sample_manifest_content_sha256")
    if content_hash != manifest["sha256"]:
        raise ValueError("candidate metadata is not linked to sample-manifest content")

    bundle_payload = _load_bundle_payload(bundle_path)
    if bundle_payload.get("manifest") != manifest:
        raise ValueError("candidate trajectory bundle is not linked to the sample manifest")
    samples = bundle_payload.get("samples")
    order = manifest.get("order")
    if not isinstance(samples, list) or not isinstance(order, list):
        raise ValueError("candidate provenance bundle or manifest order is invalid")
    if len(samples) != len(order) or manifest.get("sample_count") != len(order):
        raise ValueError("candidate provenance sample counts disagree")
    rows = []
    for index, item in enumerate(order):
        if not isinstance(item, dict):
            raise ValueError(f"candidate sample manifest row {index} is invalid")
        rows.append({
            "sample_index": item.get("sample_index"),
            "episode_id": item.get("episode_id"),
            "env_step": item.get("env_step"),
            "legal_action_mask_digest": item.get("action_mask_digest"),
            "memory_input_digest": item.get("memory_input_digest"),
            "model_input_digests": item.get("model_input_digests"),
            "done": item.get("done"),
        })
    validate_bundle(samples, rows)
    for index, sample in enumerate(samples):
        if _model_input_digests(sample["model_input"]) != order[index]["model_input_digests"]:
            raise ValueError(f"candidate model-input digest mismatch at sample {index}")
    return {"sample_manifest": sample_manifest_path, "trajectory_bundle": bundle_path}


__all__ = [
    "BUNDLE_FORMAT",
    "build_sample_manifest",
    "discounted_returns",
    "ppo_micro_update",
    "save_candidate_checkpoint",
    "save_compressed_bundle",
    "sha256_file",
    "validate_candidate_provenance",
    "validate_bundle",
]
