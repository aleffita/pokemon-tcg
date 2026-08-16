from __future__ import annotations

import json

import scripts.rl.run_continuous_grpo as continuous_module


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_continuous_grpo_refreshes_parent_and_generated_population(
    tmp_path, monkeypatch
) -> None:
    clock = _Clock()
    calls = []

    def fake_run_ar021(**kwargs):
        calls.append(kwargs)
        cycle_dir = kwargs["output_dir"]
        (cycle_dir / "decks").mkdir(parents=True)
        generated = cycle_dir / "decks" / "generated_turn0.json"
        generated.write_text("[1]")
        candidate = cycle_dir / "candidate.pt"
        candidate.write_bytes(b"candidate")
        clock.now += 100.0
        return {
            "candidate": str(candidate),
            "candidate_sha256": f"{len(calls):064x}",
            "group_size": 20,
            "logical_decisions": 1000,
            "metrics": {"optimizer_steps": 3},
        }

    monkeypatch.setattr(continuous_module, "run_ar021", fake_run_ar021)
    result = continuous_module.run_continuous_grpo(
        initial_checkpoint=tmp_path / "initial.pt",
        output_dir=tmp_path / "continuous",
        deck_path=tmp_path / "deck.json",
        meta_date="2026-08-12",
        opponent_deck_paths=[tmp_path / "opponent.csv"],
        opponent_agent_paths=[tmp_path / "main.py"],
        total_budget_seconds=1000.0,
        cycle_update_seconds=200.0,
        update_epochs_per_cycle=3,
        generated_population_size=2,
        minimum_cycle_seconds=1.0,
        finalization_reserve_seconds=1.0,
        max_cycles=3,
        clock=clock,
    )

    assert result["completed_cycle_count"] == 3
    assert calls[1]["checkpoint"] == calls[0]["output_dir"] / "candidate.pt"
    assert calls[1]["learner_deck_paths"][-1] == (
        calls[0]["output_dir"] / "decks" / "generated_turn0.json"
    )
    assert len(result["generated_population"]) == 2
    latest = json.loads((tmp_path / "continuous" / "latest.json").read_text())
    assert latest["completed_cycle_count"] == 3
    assert latest["latest"]["candidate_sha256"] == f"{3:064x}"
