"""Production trainer resume-policy integration fixtures.

These tests call the setup helpers used by ``bc_train_mlx.main``.  They load a
checkpoint and initialize/reset phase state, but never enter the training loop.
"""

import copy
import pickle
from pathlib import Path

import pytest

from rl.packed_data import (
    PACKED_BACKEND_NAME,
    approved_stage4_root_matches,
    build_resume_identity,
)
from scripts.bc.bc_train_mlx import (
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
