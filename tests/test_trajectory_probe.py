from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from rl.encoder.encoding import SUBMIT_ACTION, build_mask
from rl.encoder.card_features import get_card_table
from rl.env.env import CabtEnv
from rl.policy_infer_torch import load_inference_checkpoint
from scripts.rl.ppo_micro_update import (
    _model_input_digests,
    ppo_micro_update,
    save_candidate_checkpoint,
    validate_bundle,
)
from scripts.rl.trajectory_probe import (
    DateBoundEncoder,
    _StatefulMirror,
    _masked_distribution,
    digest_tensor,
    APPROVED_STAGE4_ROOT_SHA256,
    behavior_importance_ratio,
    composite_behavior_logprob,
    initial_memory,
    inspect_parquet_provenance,
    load_stage4,
    build_parser,
    run_probe,
    validate_meta_date,
    validate_rows,
    write_outputs,
)


class _Encoder:
    int_keys: set[str] = set()

    def encode(self, obs, picked=None, **kwargs):
        current = obs["current"]
        assert current["date"] == "2026-08-12"
        select = obs["select"]
        return {
            "action_mask": build_mask(select, picked),
            "feature": np.array([len(picked)], dtype=np.float32),
        }


class _Model:
    def __init__(self):
        self.learned_init = torch.zeros(2, 3)


def test_date_bound_encoder_requires_explicit_complete_date_and_injects_missing_engine_date(monkeypatch):
    class _Lookup:
        def resolve_day_id(self, value):
            assert value == "2026-08-12"
            return 30

        def day_index_norm(self, value):
            assert value == 30
            return 0.5

    monkeypatch.setattr("scripts.rl.trajectory_probe.get_meta_lookup", lambda: _Lookup())
    wrapped = DateBoundEncoder(_Encoder(), "2026-08-12")
    wrapped.encode(
        {"current": {"turn": 1}, "select": {"option": [], "minCount": 0, "maxCount": 1}},
        picked=set(),
    )
    with pytest.raises(ValueError, match="conflicts"):
        wrapped.encode(
            {"current": {"date": "2026-08-11"}, "select": {"option": [], "minCount": 0, "maxCount": 1}},
            picked=set(),
        )
    with pytest.raises(ValueError, match="conflict"):
        wrapped.encode(
            {"current": {"date": "2026-08-12", "archive_date": "2026-08-11"}, "select": {"option": [], "minCount": 0, "maxCount": 1}},
            picked=set(),
        )


def test_initial_memory_digest_is_stable_and_non_null():
    model = _Model()
    first = initial_memory(model)
    second = initial_memory(model)
    assert first.shape == (1, 2, 3)
    assert digest_tensor(first) == digest_tensor(second)


def test_composite_behavior_logprob_sums_masked_substeps_and_identity_ratio():
    first = _masked_distribution(
        torch.tensor([[1.0, -4.0, 0.5, -2.0]]),
        np.array([1.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )
    second = _masked_distribution(
        torch.tensor([[-1.0, 2.0, 0.25, -3.0]]),
        np.array([1.0, 1.0, 0.0, 1.0], dtype=np.float32),
    )
    first_logprob = float(first.log_prob(torch.tensor(2)).item())
    second_logprob = float(second.log_prob(torch.tensor(1)).item())
    expected = first_logprob + second_logprob
    assert composite_behavior_logprob([first_logprob, second_logprob]) == pytest.approx(expected)
    assert behavior_importance_ratio(expected, expected) == pytest.approx(1.0)


class _SequenceRng:
    def __init__(self, actions):
        self.actions = list(actions)

    def choice(self, _size, p=None):
        assert p is not None
        return self.actions.pop(0)


class _MirrorEncoder:
    int_keys: set[str] = set()
    meta_date = "2026-08-12"

    def encode(self, obs, picked=None, **_kwargs):
        picked = set(picked or ())
        mask = np.zeros(SUBMIT_ACTION + 1, dtype=np.float32)
        if 0 not in picked:
            mask[0] = 1.0
        if picked and 1 not in picked:
            mask[1] = 1.0
        if len(picked) >= 1:
            mask[SUBMIT_ACTION] = 1.0
        return {"action_mask": mask}


class _StatefulToyModel:
    def __init__(self):
        self.learned_init = torch.zeros(1, 2)
        self.memory_inputs = []

    def logits_value(self, model_input, memory_in=None):
        assert memory_in is not None
        self.memory_inputs.append(memory_in.detach().clone())
        batch = model_input["action_mask"].shape[0]
        logits = torch.zeros(batch, SUBMIT_ACTION + 1, dtype=torch.float32)
        value = torch.zeros(batch, dtype=torch.float32)
        memory_out = memory_in + float(len(self.memory_inputs))
        return logits, value, memory_out


def _mirror_obs():
    return {
        "select": {"option": [10, 11], "minCount": 2, "maxCount": 2},
        "current": {"turn": 1, "yourIndex": 1},
    }


def test_stateful_mirror_lanes_are_independent_and_commit_last_substep_output():
    model = _StatefulToyModel()
    first = _StatefulMirror(model, _MirrorEncoder(), _SequenceRng([0, 1, 0, 1, 0, 1]))
    second = _StatefulMirror(model, _MirrorEncoder(), _SequenceRng([0, 1, 0, 1]))
    first.reset_episode("first")
    second.reset_episode("second")
    initial_digest = first.initial_memory_digest

    first(_mirror_obs(), object())
    second(_mirror_obs(), object())
    first(_mirror_obs(), object())
    second(_mirror_obs(), object())

    assert all(memory is not None for memory in model.memory_inputs)
    assert digest_tensor(model.memory_inputs[0]) == initial_digest
    assert digest_tensor(model.memory_inputs[1]) == initial_digest
    assert digest_tensor(model.memory_inputs[2]) == initial_digest
    assert digest_tensor(model.memory_inputs[3]) == initial_digest
    assert first.decisions[1]["memory_input_digest"] == first.decisions[0]["committed_memory_output_digest"]
    assert second.decisions[1]["memory_input_digest"] == second.decisions[0]["committed_memory_output_digest"]
    assert first.decisions[1]["memory_input_digest"] != second.decisions[1]["memory_input_digest"]
    assert first.events[0]["memory_input_digest"] == first.events[1]["memory_input_digest"]
    assert first.events[0]["decision_memory_output_digest"] == first.events[1]["decision_memory_output_digest"]
    assert first.events[1]["memory_output_digest"] == first.events[0]["decision_memory_output_digest"]

    first.reset_episode("first-reset")
    first(_mirror_obs(), object())
    assert first.decisions[0]["memory_input_digest"] == initial_digest


def test_reset_hook_reinitializes_opponent_before_each_cabt_retry():
    class _ResetAwareOpponent:
        def __init__(self):
            self.attempts = []
            self.calls = []
            self.state = None

        def reset_attempt(self, attempt):
            self.attempts.append(int(attempt))
            self.state = f"attempt-{attempt}"

        def __call__(self, _obs, _rng, **_kwargs):
            self.calls.append(self.state)
            return [0]

    class _RetryingGame:
        def __init__(self):
            self.starts = 0
            self.obs = None

        def battle_start(self, _deck0, _deck1):
            self.starts += 1
            your_index = 1 if self.starts == 1 else 0
            self.obs = {
                "select": {"option": [0], "minCount": 1, "maxCount": 1},
                "current": {"yourIndex": your_index, "result": -1},
            }
            return self.obs, object()

        def battle_select(self, picks):
            assert picks == [0]
            self.obs = {
                "select": {"option": [0], "minCount": 1, "maxCount": 1},
                "current": {"yourIndex": 0, "result": 1},
            }
            return self.obs

        def battle_finish(self):
            return None

    opponent = _ResetAwareOpponent()
    env = CabtEnv(
        agent_deck=[1],
        opponent_deck=[1],
        opponent_fn=opponent,
        randomize_side=False,
        reset_hook=opponent.reset_attempt,
    )
    env._game = _RetryingGame()
    env._tracker = None
    env._opp_tracker = None
    env._opp_ability = None
    env._encode = lambda: {"action_mask": np.ones(1, dtype=np.float32)}

    _observation, info = env.reset(seed=18018)
    assert info["agent_index"] == 0
    assert opponent.attempts == [0, 1]
    assert opponent.calls == ["attempt-0"]
    env.close()


def test_agent_selection_failure_records_mirror_terminal_reward():
    class _TerminalSpy:
        def __init__(self):
            self.agent_returns = []

        def on_terminal(self, agent_return):
            self.agent_returns.append(float(agent_return))

    class _BrokenGame:
        def battle_select(self, _picks):
            raise RuntimeError("synthetic engine rejection")

    spy = _TerminalSpy()
    env = CabtEnv(agent_deck=[1], opponent_deck=[1], opponent_fn=spy)
    env._game = _BrokenGame()
    env._ability = None
    env._obs = {
        "select": {"option": [0], "minCount": 1, "maxCount": 1},
        "current": {"yourIndex": 0, "result": -1},
    }

    reward, terminated = env._apply_selection([0])
    assert reward == -1.0
    assert terminated is True
    assert spy.agent_returns == [-1.0]
    env.close()


def test_identical_behavior_and_learner_snapshots_recompute_complete_logprob():
    behavior_model = _StatefulToyModel()
    learner_model = copy.deepcopy(behavior_model)
    behavior = _StatefulMirror(behavior_model, _MirrorEncoder(), _SequenceRng([0, 1, 0, 1]))
    learner = _StatefulMirror(learner_model, _MirrorEncoder(), _SequenceRng([0, 1, 0, 1]))
    behavior.reset_episode("behavior")
    learner.reset_episode("learner")
    raw_obs = _mirror_obs()
    behavior(raw_obs, object())
    learner(raw_obs, object())

    grouped = {}
    for record in behavior.events:
        grouped.setdefault(int(record["decision_index"]), []).append(record)
    memory = initial_memory(learner_model)
    recomputed = []
    for decision_index in sorted(grouped):
        picked = set()
        decision_memory = memory
        memory_out = memory
        substep_logprobs = []
        for record in grouped[decision_index]:
            encoded = learner.encoder.encode(raw_obs, picked=picked)
            model_input = {
                "action_mask": torch.as_tensor(encoded["action_mask"]).reshape(1, -1),
            }
            with torch.inference_mode():
                logits, _value, memory_out = learner_model.logits_value(
                    model_input,
                    memory_in=decision_memory,
                )
            distribution = _masked_distribution(logits, encoded["action_mask"])
            substep_logprobs.append(
                float(distribution.log_prob(torch.tensor(int(record["action"]))).item())
            )
            if int(record["action"]) != SUBMIT_ACTION:
                picked.add(int(record["action"]))
        memory = memory_out.detach().to(dtype=torch.float32).clone()
        recomputed.append(composite_behavior_logprob(substep_logprobs))

    observed = [float(decision["logical_action_logprob"]) for decision in behavior.decisions]
    assert recomputed == pytest.approx(observed)
    assert all(
        behavior_importance_ratio(learner_logprob, behavior_logprob) == pytest.approx(1.0)
        for learner_logprob, behavior_logprob in zip(recomputed, observed)
    )


def test_strict_stage4_loader_does_not_fallback(monkeypatch, tmp_path):
    checkpoint = tmp_path / "stage4_root.pkl"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    def strict_loader(path, card_table):
        calls.append((path, card_table))
        raise ValueError("strict mismatch")

    monkeypatch.setattr(
        "scripts.rl.trajectory_probe.sha256_file",
        lambda path: APPROVED_STAGE4_ROOT_SHA256,
    )
    monkeypatch.setattr("scripts.rl.trajectory_probe.load_inference_checkpoint", strict_loader)
    with pytest.raises(ValueError, match="strict mismatch"):
        load_stage4(checkpoint, object())
    assert calls == [(checkpoint, object())] or len(calls) == 1


def test_stage4_root_hash_is_rejected_before_loader(monkeypatch, tmp_path):
    checkpoint = tmp_path / "not_the_root.pkl"
    checkpoint.write_bytes(b"not the frozen root")
    calls = []
    monkeypatch.setattr("scripts.rl.trajectory_probe.load_inference_checkpoint", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="approved frozen root"):
        load_stage4(checkpoint, object())
    assert calls == []


def test_games_per_mode_preserves_two_and_accepts_four():
    parser = build_parser()
    assert parser.parse_args(["--meta-date", "2026-08-12", "--games-per-mode", "2"]).games_per_mode == 2
    assert parser.parse_args(["--meta-date", "2026-08-12", "--games-per-mode", "4"]).games_per_mode == 4
    assert "1-4" in parser.format_help()


def test_games_per_mode_rejects_five_before_probe_work():
    with pytest.raises(ValueError, match="between 1 and 4"):
        run_probe(meta_date="2026-08-12", games_per_mode=5)

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--meta-date", "2026-08-12", "--games-per-mode", "5"])


def test_rows_require_ordered_multiselect_substeps_and_terminal_reward():
    rows = [
        {
            "episode_id": "random-000", "decision_index": 0, "substep": 0,
            "terminal": False, "done": False, "legal_action_mask_digest": "m",
            "action_logprob": -0.1, "memory_input_digest": "i", "memory_output_digest": "o",
            "reward": 0.0,
        },
        {
            "episode_id": "random-000", "decision_index": 0, "substep": 1,
            "terminal": True, "done": True, "legal_action_mask_digest": "m2",
            "action_logprob": -0.2, "memory_input_digest": "i", "memory_output_digest": "o2",
            "reward": 1.0,
        },
    ]
    validate_rows(rows)
    rows[1]["substep"] = 2
    with pytest.raises(AssertionError, match="substep order"):
        validate_rows(rows)


def test_output_manifest_contains_exact_jsonl_digest(tmp_path):
    rows = [{"episode_id": "e", "action": 1}]
    manifest = {
        "metadata_date": "2026-08-12",
        "model_sha256": "model",
        "deck_sha256": "deck",
        "counts_by_mode": {"random": {"episodes": 1, "rows": 1, "terminals": 1}},
    }
    write_outputs(tmp_path, rows, manifest)
    jsonl = (tmp_path / "trajectory.jsonl").read_text()
    saved = json.loads((tmp_path / "trajectory.manifest.json").read_text())
    import hashlib
    assert saved["trajectory_sha256"] == hashlib.sha256(jsonl.encode()).hexdigest()
    assert (tmp_path / "trajectory.log").is_file()


def test_parquet_provenance_is_metadata_only():
    provenance = inspect_parquet_provenance(Path("data/bc_data/2026-08-12.parquet"))
    assert provenance["rows"] > 0
    assert provenance["used_for_model_input"] is False
    assert provenance["packed"] is False


def _sample_bundle():
    mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    memory = torch.zeros(1, 1, 2, dtype=torch.float32)
    samples = []
    rows = []
    for index, (action, reward, done, value) in enumerate(((0, 0.0, False, 0.0), (2, 1.0, True, 0.5))):
        model_input = {
            "action_mask": mask.reshape(1, -1).clone(),
            "feature": torch.tensor([[float(index)]], dtype=torch.float32),
        }
        sample = {
            "sample_index": index,
            "episode_id": "e",
            "env_step": index,
            "decision_index": index,
            "substep": 0,
            "model_input": model_input,
            "action_mask": mask.clone(),
            "memory_input": memory.clone(),
            "action": action,
            "behavior_logprob": float(torch.log_softmax(torch.tensor([0.2, -0.3, 0.1]), 0)[action]),
            "value": value,
            "reward": reward,
            "done": done,
        }
        samples.append(sample)
        rows.append(
            {
                "sample_index": index,
                "episode_id": "e",
                "env_step": index,
                "legal_action_mask_digest": hashlib.sha256(mask.numpy().tobytes()).hexdigest(),
                "memory_input_digest": digest_tensor(memory),
                "model_input_digests": _model_input_digests(model_input),
                "done": done,
            }
        )
    return samples, rows


def test_bundle_preserves_shape_order_and_digests():
    bundle, rows = _sample_bundle()
    validate_bundle(bundle, rows)
    rows[1]["legal_action_mask_digest"] = "wrong"
    with pytest.raises(ValueError, match="mask digest"):
        validate_bundle(bundle, rows)


def test_bundle_rejects_model_input_tensor_tamper():
    bundle, rows = _sample_bundle()
    validate_bundle(bundle, rows)
    bundle[1]["model_input"]["feature"][0, 0] = 99.0
    with pytest.raises(ValueError, match="model-input digest"):
        validate_bundle(bundle, rows)


class _ToyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.1], dtype=torch.float32))
        self.value = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    def logits_value(self, model_input, memory_in=None):
        batch = model_input["action_mask"].shape[0]
        return self.logits.expand(batch, -1), self.value.expand(batch), memory_in


def test_ppo_micro_update_mutates_policy_and_reports_root_reference():
    bundle, rows = _sample_bundle()
    validate_bundle(bundle, rows)
    model = _ToyPolicy()
    root = copy.deepcopy(model)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    metrics = ppo_micro_update(model, root, bundle, learning_rate=1e-2)
    assert metrics["epochs"] == 1
    assert metrics["advantage_std"] == pytest.approx(1.0)
    assert metrics["root_reference_changed_parameters"] > 0
    assert any(not torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_candidate_checkpoint_strict_loads_via_inference_entrypoint(tmp_path):
    card_table = get_card_table()
    model, metadata = load_inference_checkpoint(
        Path("experiments/autoresearch/root/stage4_root.pkl"), card_table
    )
    path = tmp_path / "candidate.pt"
    save_candidate_checkpoint(
        path,
        model,
        metadata,
        root_sha256=APPROVED_STAGE4_ROOT_SHA256,
        sample_manifest_sha256="a" * 64,
        bundle_sha256="b" * 64,
        config={"algorithm": "PPO", "epochs": 1},
        diagnostics={"root_reference_kl_mean": 0.0},
        sample_manifest_content_sha256="c" * 64,
    )
    loaded, loaded_metadata = load_inference_checkpoint(path, card_table)
    assert next(loaded.parameters()).dtype == torch.float32
    assert loaded_metadata["arch_version"] == metadata["arch_version"]
