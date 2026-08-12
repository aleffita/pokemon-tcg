"""Single-agent Gymnasium-style wrapper around the cabt engine for RL.

Key facts that shape this design (all verified against the engine):

* The engine itself is fully multi-battle per process (ApiBattleStart = new ApiData();
  proven by scripts/test_multibattle.py, 2026-07-13) -- the old one-battle-per-process
  rule was cg.sim's Python class-attr singleton. Each env instance owns its own
  battle pointer (see _native_handles), so K envs can share a worker process
  (vec_env_selfplay --envs-per-worker). ``close()`` frees the native battle.
* We drive the engine directly with ``battle_start/battle_select`` (decks given
  up front), so ``select`` is never None and there is no deck-selection step.
* The action space is dynamic. We expose a single masked ``Discrete(N_ACTIONS)``
  pick per step and BUFFER picks internally for multi-select decisions
  (``maxCount`` > 1), submitting to the engine only when the set is complete.
  Index ``SUBMIT_ACTION`` (== MAX_OPTIONS) ends an optional selection early.
* The opponent plays inside ``step``; the agent only ever sees its own turns.

Reward: +1 win / -1 loss / 0 draw at terminal (plus optional shaping hook).
"""

from __future__ import annotations

import ctypes
import logging
import os
import random

import numpy as np

logging.disable(logging.CRITICAL)  # silence kaggle_environments import chatter

from rl.encoder.encoding import TokenEncoder, SUBMIT_ACTION, build_mask
from rl.encoder.encoding import GameTracker, AbilityTracker, binary_to_obs
from rl.encoder.card_features import get_card_table


def _native_handles():
    """Per-INSTANCE native-engine handles: the shared ctypes lib (signatures declared once,
    idempotent) + a fresh battle-pointer holder. The engine is multi-battle per process
    (ApiBattleStart = new ApiData(); zero mutable statics -- scripts/test_multibattle.py); the
    old 'one battle per process' constraint was cg.sim's `Battle` CLASS-attribute singleton,
    which K in-process envs would trample. Shared by CabtEnv and TwoSidedSelfPlayEnv."""
    import types
    from cg.sim import lib as L
    L.GetBinaryObs.restype = ctypes.c_int
    L.GetBinaryObs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    L.AdvanceLog.restype = ctypes.c_int
    L.AdvanceLog.argtypes = [ctypes.c_void_p]
    return types.SimpleNamespace(battle_ptr=None, obs=None), L


class _BinEngine:
    """Stage-2 JSON-free engine shim. Same battle_start/select/finish interface as cg.game, but
    steps the engine WITHOUT GetBattleData (no JSON serialize) and returns the light binary_to_obs
    game-logic dict; the raw GetBinaryObs buffer is stashed on ``last_arr`` for the tracker/encode
    (which read it directly). ``advance_log`` consumes the current player's log delta (once/decision)."""

    def __init__(self, lib, Battle, cbuf, cbuf_np, cap):
        self.lib = lib; self.Battle = Battle; self.cbuf = cbuf; self.cbuf_np = cbuf_np
        self.cap = cap; self.last_arr = None

    def _getbin(self):
        n = self.lib.GetBinaryObs(self.Battle.battle_ptr, self.cbuf, self.cap)
        if n > self.cap:
            # WriteBinaryObs stops WRITING at cap but returns the full count -- parsing the
            # truncated buffer would silently corrupt the obs AND the tracker. Raise instead:
            # the vec_env worker recovery turns this into a clean reset episode.
            raise RuntimeError(f"GetBinaryObs overflow: obs needs {n} ints > cap {self.cap}")
        a = np.array(self.cbuf_np[:n], dtype=np.int32)          # copy: stable across the decision's buffering
        self.last_arr = a
        return a

    def battle_start(self, deck0, deck1):
        cards = list(deck0) + list(deck1)
        arg = (ctypes.c_int * len(cards))(*cards)
        sd = self.lib.BattleStart(arg)
        self.Battle.battle_ptr = sd.battlePtr
        if not self.Battle.battle_ptr:
            return None, sd
        return binary_to_obs(self._getbin()), sd

    def battle_select(self, select_list):
        parg = (ctypes.c_int * len(select_list))(*select_list)
        err = self.lib.Select(self.Battle.battle_ptr, parg, len(select_list))
        if err != 0:
            raise ValueError("battle_ptr broken.") if err == 30 else IndexError()
        return binary_to_obs(self._getbin())

    def battle_finish(self):
        self.lib.BattleFinish(self.Battle.battle_ptr)

    def advance_log(self):
        self.lib.AdvanceLog(self.Battle.battle_ptr)

_AGENT_DECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "deck.csv")


def load_deck(path: str = _AGENT_DECK) -> list[int]:
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


def random_opponent(raw_obs: dict, rng: random.Random, **_) -> list[int]:
    """Baseline opponent: a uniformly random legal selection (engine-native).

    ``**_`` swallows the deck/tracker context the env passes to policy opponents."""
    sel = raw_obs["select"]
    n, k = len(sel["option"]), sel["maxCount"]
    return rng.sample(range(n), min(k, n)) if n else []


class CabtEnv:
    """Gymnasium-style env. obs is a dict of numpy arrays (see Encoder.shapes)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        agent_deck: list[int] | None = None,
        opponent_deck: list[int] | None = None,
        agent_decks: list[list[int]] | None = None,    # POOL: sampled each episode
        opponent_decks: list[list[int]] | None = None,
        opponent_fn=None,                # (raw_obs, rng) -> list[int]; default random
        encoder: TokenEncoder | None = None,
        randomize_side: bool = True,     # alternate which player the agent is
        reward_shaping=None,             # (prev_state, new_state, agent_idx) -> float
        max_steps: int = 4000,
        would_ko: bool = False,          # annotate engine-simulated would-KO per attack option
        terminal_margin: float = 0.0,    # F19 graded terminal reward weight (0 = flat +/-1)
        native_encode: bool = False,     # collection speedup: cg engine + GetBinaryObs + Cython encode_native
        native_validate: bool = False,   # debug: run BOTH encodes each step and assert byte-identical
        seed: int | None = None,
    ):
        # A pool of decks (each side sampled per episode). Single-deck args are a
        # convenience that wraps into a 1-element pool.
        self.agent_decks = agent_decks or ([agent_deck] if agent_deck else [load_deck()])
        self.opponent_decks = opponent_decks or ([opponent_deck] if opponent_deck else list(self.agent_decks))
        self.opponent_fn = opponent_fn or random_opponent
        self.encoder = encoder or TokenEncoder(get_card_table())
        self.randomize_side = randomize_side
        self.reward_shaping = reward_shaping
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.would_ko = bool(would_ko)                    # engine-sim would-KO feature per attack option
        self.terminal_margin = float(terminal_margin)     # F19: scale the terminal reward by victory margin
        # SEPARATE per-side trackers, each fed ONLY that side's own decision obs -- because
        # obs['logs'] is an incremental delta and, at Kaggle inference, an agent only ever
        # receives the obs on its OWN turns (cabt interpreter writes logs to the player about
        # to move). So each side sees only the *last* opponent sub-action's reveals; feeding a
        # tracker every obs would over-inform it vs inference. Per-player attribution inside
        # GameTracker lets each side read revealed_for(its opponent).
        self._tracker = GameTracker()                        # learner's view (reads opp's reveals)
        self._opp_tracker = GameTracker()                    # self-play opponent's view (reads learner's reveals)
        self._last_tracked_obs = None                        # learner tracker updates once per new decision obs
        # per-side ability-used memory (each fed only that side's own ABILITY picks)
        self._ability = AbilityTracker()
        self._opp_ability = AbilityTracker()
        self._agent_deck: list[int] | None = None        # this episode's agent deck (for v2 decklist)
        self._opp_deck: list[int] | None = None           # this episode's opponent deck (for the opponent's v2 encode)
        self._obs = None            # current raw engine obs
        self._picked: list[int] = []  # buffered picks for the current decision
        self._agent_idx = 0
        self._steps = 0
        self._done = True
        self._result_override = None  # forced terminal reward (e.g. opponent forfeit)
        # engine handles resolved lazily (native path swaps in cg + GetBinaryObs; default is the
        # kaggle_environments cabt engine). train==test holds: at inference the official engine + the
        # Python encode run; the native encode is byte-identical to it (same source engine, same rules).
        self._native = bool(native_encode)
        self._validate = bool(native_validate)
        self._game = None
        self._Battle = None

    # -- engine handles (imported lazily so logging is disabled first) ------
    def _ensure_engine(self):
        if self._game is not None:
            return
        if self._native:
            if self.would_ko:
                raise ValueError("native_encode (Stage-2 JSON-free) is incompatible with would_ko "
                                 "-- the KO sim needs the JSON obs. Use --no-would-ko.")
            B, L = _native_handles()
            self._Battle, self._lib = B, L
            self._setup_native()                                  # builds the encode matrices + cbuf
            self._game = _BinEngine(L, B, self._cbuf, self._cbuf_np, self._CAP)
        else:
            from kaggle_environments.envs.cabt.cg import game as g
            from kaggle_environments.envs.cabt.cg.sim import Battle as B
            self._game, self._Battle = g, B

    def _engine(self):
        self._ensure_engine()
        return self._game

    # -- gym API ------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
        game = self._engine()
        # Retry until the AGENT actually has a decision: _advance_to_agent plays the
        # opponent's opening, which can occasionally END the game before the agent ever
        # acts (deck-out / instant decide). That would leave _done=True and make the next
        # step() raise "step() after episode end" -> a fresh battle dodges the dead start.
        for _attempt in range(32):
            self._safe_finish()   # free any prior battle this env instance owns
            self._agent_idx = self.rng.randint(0, 1) if self.randomize_side else 0
            agent_deck = self.rng.choice(self.agent_decks)      # sample decks this episode
            self._agent_deck = agent_deck
            opp_deck = self.rng.choice(self.opponent_decks)
            self._opp_deck = opp_deck
            d0 = agent_deck if self._agent_idx == 0 else opp_deck
            d1 = opp_deck if self._agent_idx == 0 else agent_deck
            obs, start = game.battle_start(d0, d1)
            if obs is None:
                raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")

            if self._tracker is not None:
                self._tracker.reset()
                self._opp_tracker.reset()
                self._ability.reset()
                self._opp_ability.reset()
            self._last_tracked_obs = None
            self._obs = obs
            self._picked = []
            self._steps = 0
            self._done = False
            self._result_override = None
            self._advance_to_agent()
            if not self._done:                  # live state where the agent must act
                break
        return self._encode(), {"agent_index": self._agent_idx}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() after episode end; call reset().")
        info: dict = {}
        sel = self._obs["select"]
        mask = build_mask(sel, set(self._picked))
        action = int(action)
        if action >= len(mask) or mask[action] == 0:  # guard illegal picks
            legal = [i for i, m in enumerate(mask) if m]
            action = legal[0]
            info["illegal_action"] = True

        submit = action == SUBMIT_ACTION
        if not submit:
            self._picked.append(action)

        # auto-submit once we've reached maxCount
        if submit or len(self._picked) >= sel["maxCount"]:
            reward, terminated = self._apply_selection(sorted(set(self._picked)))
            self._picked = []
            if terminated:
                self._done = True
                return self._encode(), reward, True, False, info
            self._advance_to_agent()
            if self._done:  # opponent move ended the game
                return self._encode(), self._terminal_reward(), True, False, info
            self._steps += 1
            truncated = self._steps >= self.max_steps
            self._done = truncated
            return self._encode(), reward, False, truncated, info

        # still buffering this multi-select: same decision, updated mask
        return self._encode(), 0.0, False, False, info

    def action_masks(self):
        """Current legal-action mask (CleanRL / MaskablePPO convention)."""
        return build_mask(self._obs["select"], set(self._picked))

    def close(self):
        self._safe_finish()

    def _safe_finish(self):
        """Finish the current native battle and NULL this instance's pointer.

        The engine's battle_finish frees the battle but leaves Battle.battle_ptr
        dangling; calling it again (e.g. a new CabtEnv after close()) double-frees
        and crashes the process. Guarding on the pointer makes finish idempotent.
        """
        if self._Battle is None or not self._Battle.battle_ptr:   # engine not yet resolved / no live battle
            return
        try:
            self._engine().battle_finish()
        except Exception:
            pass
        self._Battle.battle_ptr = None

    # -- internals ----------------------------------------------------------
    def _encode(self):
        # fold THIS decision's logs into the learner tracker, once per new decision
        # obs (buffering reuses the same obs -> guard on identity). This is exactly the
        # obs the Kaggle submission's agent() receives, so train == test.
        if self._native:                                          # Stage-2 JSON-free path
            arr = self._game.last_arr
            if self._obs is not self._last_tracked_obs:
                self._tracker.update_native(arr)                  # belief from the binary logs + visible cards
                self._game.advance_log()                          # consume logIndex[me] once/decision
                self._last_tracked_obs = self._obs
            self._ability.note_turn(self._obs["current"]["turn"])
            return self._encode_native()
        if self._obs is not self._last_tracked_obs:
            self._tracker.update(self._obs)
            if self.would_ko:                         # engine-sim KO per attack option (once/decision)
                from rl import search_agent as _SA2
                _SA2.annotate_would_ko(self._obs, self._agent_deck, self.encoder)
            self._last_tracked_obs = self._obs
        self._ability.note_turn((self._obs["current"] or {}).get("turn"))
        return self.encoder.encode(self._obs, set(self._picked),
                                   self_deck=self._agent_deck, tracker=self._tracker,
                                   ability_slots=self._ability.slots)

    # -- native (Cython) JSON-free encode path: byte-identical to encoder.encode ----------------
    def _setup_native(self):
        """Build the once-per-env native machinery: a shared NativeEncoder (matrices + B=1
        BatchEncoder) + the GetBinaryObs int buffer used by the _BinEngine shim."""
        from rl.native_enc import NativeEncoder
        self._nenc = NativeEncoder(self.encoder)
        self._CAP = 16384
        self._cbuf = (ctypes.c_int * self._CAP)()
        self._cbuf_np = np.ctypeslib.as_array(self._cbuf)        # zero-copy int32 view (ctypes slice -> list is slow)

    def _encode_native(self):
        return self._nenc.encode(self._game.last_arr, self._obs["select"], self._agent_deck,
                                 self._tracker, self._ability.slots, self._picked)

    def _state(self):
        return self._obs["current"]

    def _apply_selection(self, indices: list[int]):
        """Submit the agent's selection; return (reward, terminated)."""
        game = self._engine()
        prev = self._state()
        if self._ability is not None:                  # record OUR ability picks (sel before submit)
            self._ability.record(self._obs["select"], indices)
        try:
            self._obs = game.battle_select(indices)
        except Exception:
            # illegal selection -> agent forfeits
            return -1.0, True
        s = self._state()
        if s["result"] >= 0:
            return self._terminal_reward(), True
        shaped = 0.0
        if self.reward_shaping:
            shaped = self.reward_shaping(prev, s, self._agent_idx)
        return shaped, False

    def _advance_to_agent(self):
        """Play opponent decisions until it's the agent's turn or the game ends."""
        game = self._engine()
        while True:
            s = self._state()
            if s["result"] >= 0:
                self._done = True
                return
            if s["yourIndex"] == self._agent_idx:
                return  # agent's decision
            # the opponent encodes from ITS own perspective: its true deck + its OWN trackers,
            # fed only its decision obs (so it sees what it would at inference). It reads
            # revealed_for(the learner) internally via the obs's yourIndex.
            opp_slots = None
            if self._opp_tracker is not None:
                if self._native:
                    self._opp_tracker.update_native(self._game.last_arr)   # opp belief from binary
                    self._game.advance_log()                               # consume logIndex[opp]
                else:
                    self._opp_tracker.update(self._obs)
                self._opp_ability.note_turn((self._obs["current"] or {}).get("turn"))
                opp_slots = self._opp_ability.slots
            sel = self._obs["select"]
            if self._native:                                       # opponent may encode natively (task: 2-side)
                picks = self.opponent_fn(self._obs, self.rng, deck=self._opp_deck, tracker=self._opp_tracker,
                                         ability_slots=opp_slots, binary=self._game.last_arr)
            else:
                picks = self.opponent_fn(self._obs, self.rng, deck=self._opp_deck,
                                         tracker=self._opp_tracker, ability_slots=opp_slots)
            if self._opp_ability is not None:
                self._opp_ability.record(sel, picks)
            try:
                self._obs = game.battle_select(picks)
            except Exception:
                # opponent made an illegal move -> agent wins (result stays -1,
                # so force the reward rather than reading the unfinished state)
                self._result_override = 1.0
                self._done = True
                return

    def _terminal_reward(self) -> float:
        if self._result_override is not None:
            return self._result_override                  # forfeit -> flat +/-1 (no margin info)
        s = self._state()
        r = s["result"]
        if r == 2 or r < 0:
            return 0.0
        base = 1.0 if r == self._agent_idx else -1.0
        if self.terminal_margin <= 0.0:
            return base
        # F19: scale by victory margin = the LOSER's remaining prizes (6 = blowout, 1 = squeaker).
        # R = sign * (1 + alpha * loser_remaining/6); win term dominant, margin a graded bonus. Returns
        # then live in [-(1+alpha), 1+alpha] -- set --value-vmax >= 1+alpha for the categorical head.
        loser = (1 - self._agent_idx) if base > 0 else self._agent_idx
        rem = len((s["players"][loser].get("prize") or []))
        return base * (1.0 + self.terminal_margin * (rem / 6.0))


def prize_diff_shaping(scale: float = 0.1):
    """Optional dense reward based on prize cards taken.

    In PTCG you take a card from YOUR OWN prize pile when you KO an opponent's
    Pokemon, and you win when your pile is empty. So MY pile shrinking is good,
    the OPPONENT's pile shrinking is bad.
    """
    def shape(prev, new, agent_idx):
        opp = 1 - agent_idx
        def prizes(state, i):
            return len(state["players"][i].get("prize") or [])
        d_me = prizes(prev, agent_idx) - prizes(new, agent_idx)   # prizes I took
        d_opp = prizes(prev, opp) - prizes(new, opp)              # prizes opp took
        return scale * (d_me - d_opp)
    return shape