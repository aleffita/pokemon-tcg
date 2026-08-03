"""Functional tests for the replay-log-derived aux RL targets added to the BC dataset writer
(scripts/bc/build_bc_dataset.py: _decision_prize_states / _compute_aux_targets).

These targets are read straight from each decision's obs['current']['players'][*]['prize']
counts -- no re-simulation. Two different deltas are computed on purpose (see the docstring on
_compute_aux_targets): a "rest of the current turn" lookahead (aux_prize_delta/aux_ko, which
intentionally repeats across every decision inside one turn) and a TRUE consecutive-decision
transition delta used for reward/aux_return (which must NOT repeat, or a single KO would get
summed into the return once per decision that shares its turn).
"""
from __future__ import annotations

import unittest

from scripts.bc import build_bc_dataset as B


def _obs(turn, me, my_prize, opp_prize, options=1):
    return {
        "select": {"type": 0, "maxCount": 1, "option": [{}] * options},
        "current": {
            "turn": turn,
            "yourIndex": me,
            "players": [
                {"prize": [None] * my_prize} if me == 0 else {"prize": [None] * opp_prize},
                {"prize": [None] * opp_prize} if me == 0 else {"prize": [None] * my_prize},
            ],
        },
    }


def _entry(obs, action):
    return {"observation": obs, "action": action}


class DecisionPrizeStatesTests(unittest.TestCase):
    def test_extracts_turn_and_prize_counts_for_valid_decisions(self):
        went = [
            _entry({"action": [int(c) for c in range(60)]}, [int(c) for c in range(60)]),
            _entry(_obs(turn=0, me=0, my_prize=6, opp_prize=6), [0]),
            _entry({}, [0]),  # response to decision 0 (off-by-one label pairing)
        ]
        states = B._decision_prize_states(went)
        self.assertEqual(len(states), 1)
        self.assertTrue(states[0]["valid"])
        self.assertEqual(states[0]["turn"], 0)
        self.assertEqual(states[0]["my_prize"], 6)
        self.assertEqual(states[0]["opp_prize"], 6)

    def test_missing_prize_or_turn_data_is_marked_invalid(self):
        went = [
            _entry({"action": [int(c) for c in range(60)]}, [int(c) for c in range(60)]),
            _entry({"select": {"type": 0, "maxCount": 1, "option": [{}]}, "current": {}}, [0]),
            _entry({}, [0]),
        ]
        states = B._decision_prize_states(went)
        self.assertEqual(len(states), 1)
        self.assertFalse(states[0]["valid"])

    def test_no_decision_before_a_deck_is_seen(self):
        went = [_entry(_obs(turn=0, me=0, my_prize=6, opp_prize=6), [0]), _entry({}, [0])]
        states = B._decision_prize_states(went)
        self.assertEqual(states, [])


class ComputeAuxTargetsTests(unittest.TestCase):
    def _states(self, spec):
        """spec: list of (i, turn, my_prize, opp_prize) or None for an invalid decision."""
        out = []
        for i, item in enumerate(spec):
            if item is None:
                out.append({"i": i, "valid": False})
                continue
            turn, my_prize, opp_prize = item
            out.append({
                "i": i, "valid": True, "turn": turn,
                "my_prize": my_prize, "opp_prize": opp_prize,
            })
        return out

    def test_ko_shared_across_a_turn_is_not_double_counted_in_return(self):
        # Turn 0 has 3 decisions. Between decision 1 and decision 2 the OPPONENT
        # takes a prize (opp goes 6 -> 5 means opp KO'd one of MY Pokemon and
        # took a prize card off their own prize stack). That's BAD for me, so
        # aux_prize_delta must be NEGATIVE (+prizes_i_took - prizes_opp_took).
        # The turn-window aux_prize_delta must repeat (-1) across all 3
        # decisions (spec-literal lookahead feature), but the return must only
        # ever "spend" that -1/6 once.
        states = self._states([
            (0, 6, 6),  # decision 0: turn0, my=6 opp=6
            (0, 6, 6),  # decision 1: turn0, my=6 opp=6 (no change yet)
            (0, 6, 5),  # decision 2: turn0, my=6 opp=5 (opp already took a prize here)
        ])
        aux = B._compute_aux_targets(states, outcome=1)
        # aux_prize_delta: "now vs end of this turn" -- end-of-turn state is
        # decision 2 itself (my=6, opp=5) for all three decisions since they
        # share turn 0. prizes_i_took=0-0=0, prizes_opp_took=6-5=1 -> -1.
        self.assertAlmostEqual(aux[0]["aux_prize_delta"], -1.0)
        self.assertAlmostEqual(aux[1]["aux_prize_delta"], -1.0)
        self.assertAlmostEqual(aux[2]["aux_prize_delta"], 0.0)  # decision 2 IS end-of-turn
        self.assertEqual(aux[0]["aux_ko"], 1)
        self.assertEqual(aux[1]["aux_ko"], 1)
        self.assertEqual(aux[2]["aux_ko"], 0)
        # reward/aux_return: the true transition happens between decision 1
        # and 2 only. Opp took a prize between them -> reward on decision 1 is
        # -1/6 (bad for me).
        self.assertAlmostEqual(aux[0]["reward"], 0.0)
        self.assertAlmostEqual(aux[1]["reward"], -1.0 / 6.0)
        # decision 2 is terminal (outcome=1 = I won) -> +1 bonus, no further
        # prize transition known.
        self.assertAlmostEqual(aux[2]["reward"], 1.0)
        # gamma=1.0 suffix sum: exactly one prize-swing contribution + terminal.
        # (This scenario is inherently inconsistent -- outcome=1 with opp taking
        # a prize -- but the point is that -1/6 only appears once in the sum.)
        self.assertAlmostEqual(aux[0]["aux_return"], -1.0 / 6.0 + 1.0)
        self.assertAlmostEqual(aux[1]["aux_return"], -1.0 / 6.0 + 1.0)
        self.assertAlmostEqual(aux[2]["aux_return"], 1.0)

    def test_terminal_bonus_sign_matches_outcome(self):
        win = B._compute_aux_targets(self._states([(0, 6, 6)]), outcome=1)
        loss = B._compute_aux_targets(self._states([(0, 6, 6)]), outcome=-1)
        self.assertAlmostEqual(win[0]["reward"], 1.0)
        self.assertAlmostEqual(loss[0]["reward"], -1.0)
        self.assertEqual(win[0]["aux_terminal"], 1)
        self.assertEqual(loss[0]["aux_terminal"], 1)

    def test_invalid_decision_gets_all_zero_aux_and_aux_valid_zero(self):
        states = self._states([(0, 6, 6), None, (0, 6, 5)])
        aux = B._compute_aux_targets(states, outcome=1)
        self.assertEqual(aux[1], {
            "aux_ko": 0, "aux_prize_delta": 0.0,
            "aux_terminal": 0, "aux_valid": 0,
            "reward": 0.0, "aux_return": 0.0,
        })
        self.assertEqual(aux[0]["aux_valid"], 1)
        self.assertEqual(aux[2]["aux_valid"], 1)


if __name__ == "__main__":
    unittest.main()
