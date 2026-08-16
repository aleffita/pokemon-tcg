from __future__ import annotations

import rl.policy_infer_torch as policy_infer_torch
from scripts.build_submission import _materialize_submission_checkpoint


def test_materialize_preserves_validated_torch_candidate(tmp_path, monkeypatch) -> None:
    source = tmp_path / "candidate.pt"
    destination = tmp_path / "portable.pt.building"
    source.write_bytes(b"autoresearch-provenance-and-fp32-weights")
    observed = {}

    def fake_load(path, card_table):
        observed["path"] = str(path)
        observed["card_table"] = card_table
        return object(), {"nlayers": 4, "scratch_registers": 4}

    monkeypatch.setattr(policy_infer_torch, "load_torch_inference_checkpoint", fake_load)
    metadata, action = _materialize_submission_checkpoint(
        str(source), str(destination), "cards"
    )

    assert destination.read_bytes() == source.read_bytes()
    assert metadata["nlayers"] == 4
    assert action == "validated PyTorch FP32"
    assert observed == {"path": str(source), "card_table": "cards"}


def test_materialize_converts_mlx_checkpoint(tmp_path, monkeypatch) -> None:
    source = tmp_path / "candidate.pkl"
    destination = tmp_path / "portable.pt.building"
    source.write_bytes(b"mlx")

    def fake_save(path, output, card_table):
        assert path == str(source)
        assert output == str(destination)
        assert card_table == "cards"
        destination.write_bytes(b"converted-fp32")
        return {"nlayers": 4, "scratch_registers": 4}

    monkeypatch.setattr(policy_infer_torch, "save_torch_inference_checkpoint", fake_save)
    _metadata, action = _materialize_submission_checkpoint(
        str(source), str(destination), "cards"
    )

    assert destination.read_bytes() == b"converted-fp32"
    assert action == "converted MLX -> PyTorch FP32"
