from __future__ import annotations

import gzip
import hashlib
import io
import json
import runpy
from pathlib import Path

import pytest
import torch

from rl.encoder.card_features import get_card_table
from rl.policy_infer_torch import load_inference_checkpoint
from scripts.rl.ppo_micro_update import (
    _model_input_digests,
    build_sample_manifest,
    save_candidate_checkpoint,
    save_compressed_bundle,
    sha256_file,
    validate_bundle,
    validate_candidate_provenance,
)
from scripts.rl.trajectory_probe import APPROVED_STAGE4_ROOT_SHA256, digest_tensor


ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "experiments/autoresearch/root/stage4_root.pkl"
PUBLIC_AGENT = ROOT / "public_agents/submissions/latest-submission-300elo/main.py"


def _provenance_fixture(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    card_table = get_card_table()
    model, metadata = load_inference_checkpoint(FROZEN_ROOT, card_table)
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    memory = torch.zeros(1, 1, 2, dtype=torch.float32)
    bundle = []
    rows = []
    for index, (action, done) in enumerate(((0, False), (2, True))):
        model_input = {
            "action_mask": mask.reshape(1, -1).clone(),
            "feature": torch.tensor([[float(index)]], dtype=torch.float32),
        }
        sample = {
            "sample_index": index,
            "episode_id": "fixture",
            "env_step": index,
            "decision_index": index,
            "substep": 0,
            "model_input": model_input,
            "action_mask": mask.clone(),
            "memory_input": memory.clone(),
            "action": action,
            "behavior_logprob": -0.1,
            "value": 0.0,
            "reward": 1.0 if done else 0.0,
            "done": done,
        }
        bundle.append(sample)
        rows.append(
            {
                "sample_index": index,
                "episode_id": "fixture",
                "env_step": index,
                "legal_action_mask_digest": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                "memory_input_digest": digest_tensor(memory),
                "model_input_digests": _model_input_digests(model_input),
                "done": done,
            }
        )
    validate_bundle(bundle, rows)
    manifest = build_sample_manifest(
        bundle,
        root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        metadata_date="2026-08-12",
        deck_content_sha256="d" * 64,
        deck_source_file_sha256="e" * 64,
    )
    manifest_path = tmp_path / "sample.manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    bundle_path = tmp_path / "trajectory_bundle.pt.gz"
    bundle_hash = save_compressed_bundle(bundle_path, bundle, manifest)
    candidate_path = tmp_path / "candidate.pt"
    save_candidate_checkpoint(
        candidate_path,
        model,
        metadata,
        root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        sample_manifest_sha256=sha256_file(manifest_path),
        bundle_sha256=bundle_hash,
        sample_manifest_content_sha256=manifest["sha256"],
        config={"algorithm": "PPO", "epochs": 1},
        diagnostics={"root_reference_kl_mean": 0.0},
    )
    return candidate_path


def test_candidate_provenance_and_strict_load(tmp_path):
    candidate = _provenance_fixture(tmp_path)
    artifacts = validate_candidate_provenance(
        candidate,
        approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
    )
    assert artifacts["sample_manifest"].name == "sample.manifest.json"
    loaded, metadata = load_inference_checkpoint(candidate, get_card_table())
    assert next(loaded.parameters()).dtype == torch.float32
    assert metadata["inference_config"]["prospective_planner"]["enabled"] is False


def test_candidate_provenance_rejects_root_tamper(tmp_path):
    candidate = _provenance_fixture(tmp_path)
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    payload["autoresearch"]["root_sha256"] = "f" * 64
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match="root_sha256"):
        validate_candidate_provenance(
            candidate,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sample_manifest", "/tmp/sample.manifest.json", "relative"),
        ("trajectory_bundle", "../trajectory_bundle.pt.gz", "traversal"),
    ],
)
def test_candidate_provenance_rejects_absolute_and_traversal_paths(
    tmp_path, field, value, message
):
    candidate = _provenance_fixture(tmp_path)
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    payload["autoresearch"]["artifacts"][field] = value
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match=message):
        validate_candidate_provenance(
            candidate,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )


def test_candidate_provenance_rejects_symlink_artifact(tmp_path):
    candidate = _provenance_fixture(tmp_path)
    manifest_path = tmp_path / "sample.manifest.json"
    real_manifest = tmp_path / "real.manifest.json"
    real_manifest.write_bytes(manifest_path.read_bytes())
    manifest_path.unlink()
    manifest_path.symlink_to(real_manifest)
    with pytest.raises(ValueError, match="sample manifest.*symlink"):
        validate_candidate_provenance(
            candidate,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )


def test_candidate_provenance_rejects_manifest_content_tamper(tmp_path):
    candidate = _provenance_fixture(tmp_path)
    manifest_path = tmp_path / "sample.manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["deck_content_sha256"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    payload["autoresearch"]["sample_manifest_sha256"] = sha256_file(manifest_path)
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match="content hash"):
        validate_candidate_provenance(
            candidate,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )


def test_candidate_provenance_rejects_bundle_manifest_mismatch(tmp_path):
    candidate = _provenance_fixture(tmp_path)
    bundle_path = tmp_path / "trajectory_bundle.pt.gz"
    manifest = json.loads((tmp_path / "sample.manifest.json").read_text())
    with gzip.GzipFile(fileobj=io.BytesIO(bundle_path.read_bytes())) as handle:
        bundle_payload = torch.load(io.BytesIO(handle.read()), map_location="cpu", weights_only=True)
    detached_manifest = dict(manifest)
    detached_manifest["metadata_date"] = "2026-08-13"
    bundle_hash = save_compressed_bundle(
        bundle_path,
        bundle_payload["samples"],
        detached_manifest,
    )
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    payload["autoresearch"]["bundle_sha256"] = bundle_hash
    torch.save(payload, candidate)
    with pytest.raises(ValueError, match="not linked"):
        validate_candidate_provenance(
            candidate,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )


def test_candidate_provenance_rejects_missing_or_tampered_artifacts(tmp_path):
    candidate = _provenance_fixture(tmp_path)
    (tmp_path / "trajectory_bundle.pt.gz").unlink()
    with pytest.raises(FileNotFoundError, match="trajectory bundle"):
        validate_candidate_provenance(
            candidate,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )

    candidate = _provenance_fixture(tmp_path / "tampered")
    bundle_path = tmp_path / "tampered/trajectory_bundle.pt.gz"
    bundle_path.write_bytes(bundle_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bundle file hash"):
        validate_candidate_provenance(
            candidate,
            approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        )


def test_opt_in_candidate_executes_first_choose_decision(tmp_path, monkeypatch):
    candidate = _provenance_fixture(tmp_path)
    monkeypatch.setenv("PTCG_MODEL_PATH", str(candidate))
    monkeypatch.setenv("PTCG_EXPECTED_MODEL_SHA256", sha256_file(candidate))
    namespace = runpy.run_path(str(PUBLIC_AGENT), run_name="ptcg_ar010_candidate_smoke")
    assert namespace["_RUNTIME_DATA"]["prospective_planner"]["enabled"] is False
    model = namespace["_LOADED_MODEL"]
    calls = []
    original = type(model).logits_value

    def counted(self, *args, **kwargs):
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(model), "logits_value", counted)
    result = namespace["choose"](
        {
            "option": [{"type": 0}],
            "minCount": 1,
            "maxCount": 1,
            "type": 0,
            "context": 0,
        },
        {
            "yourIndex": 0,
            "turn": 1,
            "turnActionCount": 0,
            "firstPlayer": 0,
            "date": "2026-08-12",
            "players": [{}, {}],
        },
        logs=[],
    )
    assert calls, "choose() returned without a model forward"
    assert result == [0]


def test_opt_in_candidate_tensor_tamper_is_rejected_by_expected_sha(
    tmp_path, monkeypatch
):
    candidate = _provenance_fixture(tmp_path)
    expected_sha256 = sha256_file(candidate)
    payload = torch.load(candidate, map_location="cpu", weights_only=True)
    state_key = next(iter(payload["state_dict"]))
    payload["state_dict"][state_key] = payload["state_dict"][state_key].clone()
    payload["state_dict"][state_key].view(-1)[0] += 1.0
    torch.save(payload, candidate)

    monkeypatch.setenv("PTCG_MODEL_PATH", str(candidate))
    monkeypatch.setenv("PTCG_EXPECTED_MODEL_SHA256", expected_sha256)
    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        runpy.run_path(str(PUBLIC_AGENT), run_name="ptcg_ar012_tensor_tamper")


@pytest.mark.parametrize(
    ("expected_sha256", "message"),
    [(None, "required"), ("0" * 64, "candidate SHA-256 mismatch")],
)
def test_opt_in_candidate_requires_matching_expected_sha(
    tmp_path, monkeypatch, expected_sha256, message
):
    candidate = _provenance_fixture(tmp_path)
    monkeypatch.setenv("PTCG_MODEL_PATH", str(candidate))
    if expected_sha256 is None:
        monkeypatch.delenv("PTCG_EXPECTED_MODEL_SHA256", raising=False)
    else:
        monkeypatch.setenv("PTCG_EXPECTED_MODEL_SHA256", expected_sha256)
    with pytest.raises(ValueError, match=message):
        runpy.run_path(str(PUBLIC_AGENT), run_name="ptcg_ar012_expected_sha")


def test_opt_in_candidate_rejects_relative_model_path_before_load_or_provenance(
    tmp_path, monkeypatch
):
    candidate = _provenance_fixture(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PTCG_MODEL_PATH", candidate.name)
    monkeypatch.setenv("PTCG_EXPECTED_MODEL_SHA256", sha256_file(candidate))
    with pytest.raises(ValueError, match="absolute path"):
        runpy.run_path(str(PUBLIC_AGENT), run_name="ptcg_ar013_relative_path")


def test_opt_in_candidate_rejects_malformed_expected_sha(tmp_path, monkeypatch):
    candidate = _provenance_fixture(tmp_path)
    monkeypatch.setenv("PTCG_MODEL_PATH", str(candidate))
    monkeypatch.setenv("PTCG_EXPECTED_MODEL_SHA256", "not-a-sha256")
    with pytest.raises(ValueError, match="64-character hexadecimal"):
        runpy.run_path(str(PUBLIC_AGENT), run_name="ptcg_ar013_malformed_sha")


def test_default_public_agent_behavior_does_not_enter_candidate_gate(monkeypatch):
    monkeypatch.delenv("PTCG_MODEL_PATH", raising=False)
    monkeypatch.setenv("PTCG_EXPECTED_MODEL_SHA256", "not-used-by-default-path")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("default public behavior invoked candidate provenance")

    monkeypatch.setattr(
        "scripts.rl.ppo_micro_update.validate_candidate_provenance",
        fail_if_called,
    )
    class _DefaultModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))

    def default_loader(*args, **kwargs):
        return _DefaultModel(), {
            "nlayers": 0,
            "scratch_registers": 0,
            "inference_config": {
                "bc_would_ko": False,
                "prospective_planner": {"enabled": False},
            },
        }

    monkeypatch.setattr("rl.policy_infer_torch.load_inference_checkpoint", default_loader)
    namespace = runpy.run_path(str(PUBLIC_AGENT), run_name="ptcg_ar010_default_smoke")
    assert namespace["_OPT_IN_MODEL_PATH"] is None
    assert "_OPT_IN_PROVENANCE" not in namespace
