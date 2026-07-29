"""Functional checks for PyTorch-only submission inference semantics."""
from __future__ import annotations

import torch

from agent import main as agent_main
from rl.encoder.enc_constants import N_ACTIONS, OPT_PICKED, OPT_STRUCT
from rl.encoder.encoding import MAX_OPTIONS, SUBMIT_ACTION, build_mask


class _ScriptedModel:
    def __init__(self) -> None:
        self.memory_inputs = []
        self.calls = 0

    def logits_value(self, observation, memory_in=None):
        self.memory_inputs.append(memory_in)
        logits = torch.full((1, N_ACTIONS), -100.0, dtype=torch.float16)
        if self.calls == 0:
            logits[0, 1] = 10.0
            logits[0, SUBMIT_ACTION] = 20.0  # illegal before min_count
        else:
            logits[0, SUBMIT_ACTION] = 20.0
        self.calls += 1
        memory_out = torch.full((1, 2, 3), float(self.calls), dtype=torch.float16)
        return logits, torch.zeros(1, dtype=torch.float16), memory_out


def test_multiselect_reencodes_and_advances_memory_once() -> None:
    select = {
        "option": [{"type": 3}, {"type": 3}, {"type": 3}],
        "minCount": 1,
        "maxCount": 3,
    }
    seen_picked: list[set[int]] = []

    def encode_step(picked: set[int]):
        seen_picked.append(set(picked))
        opt_attr = torch.zeros(MAX_OPTIONS, OPT_STRUCT, dtype=torch.float16).numpy()
        for index in picked:
            opt_attr[index, OPT_PICKED] = 1.0
        return {
            "action_mask": build_mask(select, picked),
            "opt_attr": opt_attr,
        }

    incoming = torch.full((1, 2, 3), 7.0, dtype=torch.float16)
    model = _ScriptedModel()
    actions, outgoing = agent_main._autoregressive_select(
        model,
        encode_step,
        int_keys=set(),
        options=select["option"],
        min_count=select["minCount"],
        max_count=select["maxCount"],
        memory_in=incoming,
    )

    assert actions == [1]
    assert seen_picked == [set(), {1}]
    assert all(value is incoming for value in model.memory_inputs)
    assert torch.equal(outgoing, torch.full((1, 2, 3), 2.0, dtype=torch.float16))

    first = encode_step(set())
    second = encode_step({1})
    assert first["action_mask"][SUBMIT_ACTION] == 0.0
    assert second["action_mask"][SUBMIT_ACTION] == 1.0
    assert second["action_mask"][1] == 0.0
    assert second["opt_attr"][1, OPT_PICKED] == 1.0
    print("  PASS: multi-select reencodes picked/mask/SUBMIT without memory chaining")


def test_agent_source_has_no_mlx_backend_switch() -> None:
    with open(agent_main.__file__, encoding="utf-8") as fh:
        source = fh.read()
    assert "PTCG_INFERENCE_BACKEND" not in source
    assert "policy_mlx" not in source
    assert "import mlx" not in source
    assert "train_config.json" not in source
    assert "load_config" not in source
    print("  PASS: agent inference has no MLX backend or environment switch")


def test_would_ko_runs_once_per_decision() -> None:
    import rl.search_agent as search_agent

    select = {
        "type": 0,
        "option": [{"type": 13, "attackId": 1}],
        "minCount": 1,
        "maxCount": 1,
    }
    calls = []

    class _Encoder:
        int_keys = set()

        def encode(self, observation, picked, **_kwargs):
            return {
                "action_mask": build_mask(select, picked),
                "opt_attr": torch.zeros(
                    MAX_OPTIONS, OPT_STRUCT, dtype=torch.float16
                ).numpy(),
            }

    class _OnePickModel:
        def logits_value(self, observation, memory_in=None):
            logits = torch.full((1, N_ACTIONS), -100.0, dtype=torch.float16)
            logits[0, 0] = 1.0
            return logits, torch.zeros(1, dtype=torch.float16), torch.ones(
                1, 2, 3, dtype=torch.float16
            )

    def annotate(observation, deck, encoder, n_var, rng):
        calls.append((observation, deck, encoder, n_var, rng))
        observation["select"]["option"][0]["would_ko"] = 1.0
        return {0: (1.0, 1.0, 0.0)}

    old_encoder = agent_main._ENCODER
    old_model = agent_main._LOADED_MODEL
    old_annotate = search_agent.annotate_would_ko
    old_enabled = agent_main._RUNTIME_CFG.bc_would_ko
    old_nvar = agent_main._RUNTIME_CFG.bc_wk_nvar
    try:
        agent_main._ENCODER = _Encoder()
        agent_main._LOADED_MODEL = _OnePickModel()
        agent_main._RUNTIME_CFG.bc_would_ko = True
        agent_main._RUNTIME_CFG.bc_wk_nvar = 10
        agent_main._TRACKERS.clear()
        search_agent.annotate_would_ko = annotate
        result = agent_main.choose(
            select,
            {"yourIndex": 0, "turn": 1},
            logs=[],
        )
    finally:
        agent_main._ENCODER = old_encoder
        agent_main._LOADED_MODEL = old_model
        agent_main._RUNTIME_CFG.bc_would_ko = old_enabled
        agent_main._RUNTIME_CFG.bc_wk_nvar = old_nvar
        agent_main._TRACKERS.clear()
        search_agent.annotate_would_ko = old_annotate

    assert result == [0]
    assert len(calls) == 1
    assert calls[0][3] == 10
    print("  PASS: would_ko is configured and annotated exactly once per decision")


def main() -> None:
    print("=== PyTorch-only agent inference validation ===")
    test_multiselect_reencodes_and_advances_memory_once()
    test_agent_source_has_no_mlx_backend_switch()
    test_would_ko_runs_once_per_decision()
    print("ALL PASSED")


if __name__ == "__main__":
    main()


__all__ = [
    "test_multiselect_reencodes_and_advances_memory_once",
    "test_agent_source_has_no_mlx_backend_switch",
    "test_would_ko_runs_once_per_decision",
]
