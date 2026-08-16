import json
from pathlib import Path

from scripts.tournament import _read_deck_file


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
