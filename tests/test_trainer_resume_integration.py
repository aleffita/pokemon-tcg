"""Production trainer resume-policy integration fixtures.

These tests call the setup helpers used by ``bc_train_mlx.main``.  They load a
checkpoint and initialize/reset phase state, but never enter the training loop.
"""

import copy
import pickle
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from rl.packed_data import (
    PACKED_BACKEND_NAME,
    approved_stage4_root_matches,
    build_resume_identity,
)
from scripts.bc.bc_train_mlx import (
    _apply_optimizer_step,
    _build_optimizer,
    _load_resume_setup,
    _prepare_training_phases,
)


ROOT = Path(__file__).resolve().parents[1]
APPROVED_ROOT = ROOT / "experiments/autoresearch/root/stage4_root.pkl"


class _RecordingModel:
    def __init__(self, config):
        self.config = config
        self.updated = None

    def update(self, params):
        self.updated = params

    def get_config(self):
        return self.config

    def trainable_parameters(self):
        return {"weight": 0}


class _RecordingOptimizer:
    def __init__(self):
        self.state = {"loaded": True}
        self.init_calls = 0

    def init(self, params):
        assert params == {"weight": 0}
        self.init_calls += 1
        self.state = {"fresh": True}


class _TinyUpdateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 2)


def _micro_optimizer_config():
    return SimpleNamespace(
        optimizer="muon_adamw",
        lr=1e-3,
        lr_schedule="linear",
        lr_min_ratio=0.1,
        muon_momentum=0.95,
        muon_weight_decay=0.01,
        adamw_betas=[0.9, 0.999],
        adamw_eps=1e-8,
        adamw_weight_decay=0.01,
        structured_weight_decay=0.1,
    )


def _tree_snapshots(tree):
    mx.eval(tree)
    return [np.asarray(value).copy() for _, value in nn.utils.tree_flatten(tree)]


def _identity(*, seed=17, dedup=True, tbptt_chunk=16, backend=None):
    return build_resume_identity(
        source_sha256="source-a-b",
        selection={"max_rows": 0, "selected_episode_ids": [10, 11]},
        split={
            "val_rows": 2,
            "train_rows": 4,
            "val_episode_ids": [10],
            "train_episode_ids": [11],
        },
        backend=backend or {"name": PACKED_BACKEND_NAME, "data_digest": "packed-a"},
        seed=seed,
        dedup=dedup,
        tbptt_chunk=tbptt_chunk,
    )


def _write_checkpoint(path, *, data_identity=None):
    state = {
        "model": {"weight": 1},
        "arch_config": {"kind": "tiny"},
        "epoch": 7,
        "gstep": 99,
        "val_acc": 0.7,
        "best_val_acc": 0.8,
        "optimizer": {"loaded": True},
        "optimizer_contract": {"name": "tiny"},
        "optimizer_phase_step": 12,
        "scheduler_phase_step": 13,
        "scheduler_total_steps": 20,
        "scheduler_contract": {"name": "tiny"},
    }
    if data_identity is not None:
        state["data_identity"] = data_identity
    with path.open("wb") as handle:
        pickle.dump(state, handle)
    return state


def _root_model():
    with APPROVED_ROOT.open("rb") as handle:
        state = pickle.load(handle)
    return _RecordingModel(state["arch_config"])


def test_production_resume_setup_allows_only_exact_root_warmstart_and_resets_phases():
    assert approved_stage4_root_matches(APPROVED_ROOT)
    current = _identity()
    model = _root_model()
    setup = _load_resume_setup(
        APPROVED_ROOT,
        model=model,
        data_identity=current,
        packed=False,
        optimizer_state="reset",
        scheduler_state="reset",
    )
    assert setup.compatibility == "legacy-stage4-warmstart-no-data-identity"
    assert setup.state["optimizer"]

    optimizer = _RecordingOptimizer()
    phase = _prepare_training_phases(
        optimizer,
        model,
        state=setup.state,
        resume_path=APPROVED_ROOT,
        optimizer_state="reset",
        scheduler_state="reset",
        optimizer_contract={"name": "tiny"},
        configured_total_steps=4,
        run_optimizer_steps=2,
    )
    assert phase.optimizer_resumed is False
    assert optimizer.init_calls == 1
    assert optimizer.state == {"fresh": True}
    assert phase.optimizer_phase_step == 0
    assert (phase.scheduler_phase_step, phase.scheduler_total_steps) == (0, 4)


def test_production_resume_setup_continues_matching_identity_optimizer_and_scheduler(tmp_path):
    current = _identity()
    checkpoint = tmp_path / "matching.pkl"
    _write_checkpoint(checkpoint, data_identity=current)
    model = _RecordingModel({"kind": "tiny"})
    setup = _load_resume_setup(
        checkpoint,
        model=model,
        data_identity=current,
        packed=True,
        optimizer_state="resume",
        scheduler_state="resume",
    )
    optimizer = _RecordingOptimizer()
    phase = _prepare_training_phases(
        optimizer,
        model,
        state=setup.state,
        resume_path=checkpoint,
        optimizer_state="resume",
        scheduler_state="resume",
        optimizer_contract={"name": "tiny"},
        configured_total_steps=20,
        run_optimizer_steps=2,
    )
    assert setup.compatibility == "validated"
    assert phase.optimizer_resumed is True
    assert optimizer.init_calls == 0
    assert optimizer.state == {"loaded": True}
    assert phase.optimizer_phase_step == 12
    assert (phase.scheduler_phase_step, phase.scheduler_total_steps) == (13, 20)


@pytest.mark.parametrize("mode", ["optimizer", "scheduler"])
def test_production_resume_setup_rejects_phase_resume_for_legacy_root(mode):
    with pytest.raises(ValueError, match="explicit Stage 4 warm-start"):
        _load_resume_setup(
            APPROVED_ROOT,
            model=_root_model(),
            data_identity=_identity(),
            packed=False,
            optimizer_state="resume" if mode == "optimizer" else "reset",
            scheduler_state="resume" if mode == "scheduler" else "reset",
        )


def test_production_resume_setup_rejects_non_root_legacy_filename_and_packed_root(tmp_path):
    non_root = tmp_path / "stage4_root.pkl"
    _write_checkpoint(non_root)
    for path, packed in ((non_root, False), (APPROVED_ROOT, True)):
        with pytest.raises(ValueError, match="legacy checkpoint|explicit Stage 4 warm-start"):
            _load_resume_setup(
                path,
                model=_root_model() if path == APPROVED_ROOT else _RecordingModel({"kind": "tiny"}),
                data_identity=_identity(),
                packed=packed,
                optimizer_state="reset",
                scheduler_state="reset",
            )


def test_production_resume_setup_rejects_partial_identity(tmp_path):
    checkpoint = tmp_path / "partial.pkl"
    _write_checkpoint(checkpoint, data_identity={"version": 1, "source": {"sha256": "source-a-b"}})
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_resume_setup(
            checkpoint,
            model=_RecordingModel({"kind": "tiny"}),
            data_identity=_identity(),
            packed=False,
            optimizer_state="reset",
            scheduler_state="reset",
        )


@pytest.mark.parametrize("phase_setup", ["fresh", "reset"])
def test_production_optimizer_scheduler_micro_update_mutates_state_and_advances(
    tmp_path, phase_setup
):
    cfg = _micro_optimizer_config()
    model = _TinyUpdateModel()
    model.set_dtype(mx.float32)
    optimizer = _build_optimizer(cfg)
    reset_checkpoint = None
    if phase_setup == "reset":
        reset_checkpoint = tmp_path / "reset.pkl"
        reset_checkpoint.write_bytes(b"phase-reset-fixture")

    phase = _prepare_training_phases(
        optimizer,
        model,
        state={},
        resume_path=reset_checkpoint,
        optimizer_state="reset",
        scheduler_state="reset",
        optimizer_contract={"name": "micro"},
        configured_total_steps=4,
        run_optimizer_steps=2,
    )
    assert phase.optimizer_resumed is False
    assert (phase.optimizer_phase_step, phase.scheduler_phase_step) == (0, 0)
    optimizer.learning_rate = cfg.lr
    before_parameters = _tree_snapshots(model.parameters())
    before_state = _tree_snapshots(optimizer.state)
    gradients = nn.utils.tree_map(
        lambda parameter: mx.ones(parameter.shape, dtype=mx.float32),
        model.trainable_parameters(),
    )

    first = _apply_optimizer_step(
        optimizer,
        model,
        gradients,
        2,
        gstep=0,
        optimizer_phase_step=phase.optimizer_phase_step,
        scheduler_phase_step=phase.scheduler_phase_step,
        scheduler_total_steps=phase.scheduler_total_steps,
        lr=cfg.lr,
        lr_schedule=cfg.lr_schedule,
        warmup_steps=0,
        lr_min_ratio=cfg.lr_min_ratio,
        max_grad_norm=10.0,
    )
    second = _apply_optimizer_step(
        optimizer,
        model,
        gradients,
        2,
        gstep=first.gstep,
        optimizer_phase_step=first.optimizer_phase_step,
        scheduler_phase_step=first.scheduler_phase_step,
        scheduler_total_steps=phase.scheduler_total_steps,
        lr=cfg.lr,
        lr_schedule=cfg.lr_schedule,
        warmup_steps=0,
        lr_min_ratio=cfg.lr_min_ratio,
        max_grad_norm=10.0,
    )

    assert (first.gstep, first.optimizer_phase_step, first.scheduler_phase_step) == (
        1,
        1,
        1,
    )
    assert (second.gstep, second.optimizer_phase_step, second.scheduler_phase_step) == (
        2,
        2,
        2,
    )
    assert second.learning_rate != first.learning_rate
    after_parameters = _tree_snapshots(model.parameters())
    after_state = _tree_snapshots(optimizer.state)
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(before_parameters, after_parameters)
    )
    assert any(
        not np.array_equal(before, after)
        for before, after in zip(before_state, after_state)
    )


@pytest.mark.parametrize("field", ["source", "selection", "split", "seed", "dedup", "tbptt", "backend"])
def test_production_resume_setup_rejects_identity_mismatch(tmp_path, field):
    current = _identity()
    saved = copy.deepcopy(current)
    if field == "source":
        saved["source"]["sha256"] = "other-source"
    elif field == "selection":
        saved["selection"]["max_rows"] = 1
    elif field == "split":
        saved["split"]["val_rows"] = 3
    elif field == "seed":
        saved["trainer"]["seed"] = 18
    elif field == "dedup":
        saved["trainer"]["dedup"] = False
    elif field == "tbptt":
        saved["trainer"]["tbptt_chunk"] = 8
    elif field == "backend":
        saved["backend"]["data_digest"] = "other-packed"
    checkpoint = tmp_path / f"{field}.pkl"
    _write_checkpoint(checkpoint, data_identity=saved)
    with pytest.raises(ValueError, match="identity mismatch"):
        _load_resume_setup(
            checkpoint,
            model=_RecordingModel({"kind": "tiny"}),
            data_identity=current,
            packed=True,
            optimizer_state="reset",
            scheduler_state="reset",
        )
