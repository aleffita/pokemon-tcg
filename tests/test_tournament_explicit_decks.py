import json
from pathlib import Path

import pytest

from scripts.tournament import (
    _cell_checkpoint_path,
    _load_cell_checkpoint,
    _read_deck_file,
    _save_cell_checkpoint,
    _write_progress_report,
)


ROOT = Path(__file__).resolve().parents[1]


def test_explicit_deck_files_are_exact_and_json_parity_preserved():
    v0_path = ROOT / "experiments/decks/deck_supreme_60.json"
    v0 = json.loads(v0_path.read_text())
    assert _read_deck_file(str(v0_path)) == sorted(json.loads((ROOT / "agent/deck.json").read_text()))
    assert len(_read_deck_file(str(v0_path))) == 60

    expected_candidates = (
        ROOT / "experiments/decks/candidates/deck_v1_anti_lucario_tempo.json",
        ROOT / "experiments/decks/candidates/deck_v2_anti_control_lock.json",
    )
    for path in expected_candidates:
        deck = _read_deck_file(str(path))
        assert deck is not None, path
        assert len(deck) == 60, path


def test_tournament_cell_checkpoint_roundtrip_and_signature_guard(tmp_path):
    checkpoint = _cell_checkpoint_path(str(tmp_path), 2, 7, 0)
    signature = {
        "agent": "agent/main.py",
        "deck_file": "deck.json",
        "deck_id": 42,
        "opponent_label": "first",
        "opponent_path": "first",
        "opp_deck_id": None,
        "games": 2,
    }
    result = {
        "wins": 1,
        "losses": 1,
        "draws": 0,
        "elapsed_s": 0.25,
        "game_results": [
            {"game_index": 0, "our_side": 0, "result": 1, "replay_json": {}},
            {"game_index": 1, "our_side": 1, "result": -1, "replay_json": {}},
        ],
    }

    _save_cell_checkpoint(checkpoint, signature, result)
    assert _load_cell_checkpoint(checkpoint, signature) == result
    assert not Path(f"{checkpoint}.tmp").exists()

    incompatible = {**signature, "games": 3}
    with pytest.raises(RuntimeError, match="incompatible tournament checkpoint"):
        _load_cell_checkpoint(checkpoint, incompatible)


def test_tournament_progress_report_is_atomic_and_marks_running(tmp_path):
    report = tmp_path / "report.json"
    _write_progress_report(
        str(report),
        our_path="agent/main.py",
        custom_deck_paths=["deck-a.json", "deck-b.json"],
        do_sweep=True,
        sweep_source="local",
        games=50,
        note="durability test",
        structured_rows=[{"opponent_label": "first", "wins": 1, "losses": 0}],
        total_w=1,
        total_l=0,
        total_d=0,
        elapsed_s=1.5,
        completed_blocks=1,
        total_blocks=2,
        status="running",
    )

    payload = json.loads(report.read_text())
    assert payload["status"] == "running"
    assert payload["completed_blocks"] == 1
    assert payload["total_blocks"] == 2
    assert payload["overall"]["wr_pct"] == 100.0
    assert not Path(f"{report}.tmp").exists()
