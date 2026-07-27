"""BC agent for the PTCG AI Battle Challenge.

MLX-only inference with autoregressive multi-select.
Loads a trained TokenTransformerMLX checkpoint and uses it to select actions.
Keeps per-side GameTracker + AbilityTracker for stateful encoding.

Usage:
  Used by evaluate.py, run_battle.py, and the Kaggle submission harness.
  The engine calls agent(obs) once per decision point.
"""
import os
import sys
from typing import Any


# ---- path setup (Kaggle-safe: __file__ is not defined when exec'd) ----
def _find_dir(filename: str = "deck.csv") -> str:
    """Find the directory containing `filename`, searching sys.path + common locations."""
    candidates = list(sys.path) + ["/kaggle_simulations/agent", os.getcwd()]
    for d in candidates:
        if d and os.path.exists(os.path.join(d, filename)):
            return d
    return os.getcwd()


# On Kaggle, agent dir = /kaggle_simulations/agent (deck.csv + rl/ all flat).
# Locally, agent dir = .../agent/ and project root is its parent.
_AGENT_DIR = _find_dir("deck.csv")
# If we found deck.csv in a dir that ALSO has rl/, we're on Kaggle (flat layout).
# Otherwise, the project root is the parent (local dev layout).
if os.path.isdir(os.path.join(_AGENT_DIR, "rl")):
    _PROJECT_ROOT = _AGENT_DIR
else:
    _PROJECT_ROOT = os.path.dirname(_AGENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# The Kaggle sandbox image ships no MLX, so the submission bundle carries its
# own unpacked wheels. Append (not insert): a locally installed MLX wins, which
# keeps development on the Metal backend. Must run before `import mlx`.
_VENDOR_DIR = os.path.join(_AGENT_DIR, "_vendor")
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.append(_VENDOR_DIR)

import numpy as np
import mlx.core as mx
import mlx.nn as nn

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION
from rl.encoder.enc_constants import MAX_OPTIONS
from rl.policy_mlx import build_token_net_mlx
from rl.train_config import TrainConfig

_DECK_PATH = os.path.join(_AGENT_DIR, "deck.csv")

# ---- model checkpoint search (MLX only) ----
# Use config dirs for search paths, with known fallbacks
_cfg_default = TrainConfig()
_MODEL_PATH = None
for candidate in [
    os.path.join(_PROJECT_ROOT, _cfg_default.model_dir, "bc_best_mlx_final.pkl"),
    os.path.join(_PROJECT_ROOT, _cfg_default.checkpoint_dir, "bc_best_mlx.pkl"),
    os.path.join(_PROJECT_ROOT, _cfg_default.model_dir, "bc_best_final.pkl"),
    os.path.join(_PROJECT_ROOT, _cfg_default.checkpoint_dir, "bc_best.pkl"),
]:
    if os.path.exists(candidate):
        _MODEL_PATH = candidate
        break

# ---- load once at import ----
if _MODEL_PATH is None:
    print("[bc-agent] WARNING: no checkpoint found, using random policy")
_CARD_TABLE = get_card_table()
_ENCODER = TokenEncoder(_CARD_TABLE)

# Build default architecture config from TrainConfig defaults
_DEFAULT_CFG: dict[str, Any] = {
    "d_model": TrainConfig.d_model,
    "nhead": TrainConfig.nhead,
    "nlayers": TrainConfig.nlayers,
    "static": TrainConfig.static,
    "split_heads": TrainConfig.split_heads,
    "structured": TrainConfig.structured,
    "scratch_registers": TrainConfig.scratch_registers,
    "value_atoms": TrainConfig.value_atoms,
    "value_vmax": TrainConfig.value_vmax,
}


def _load_model():
    """Load the BC model from MLX checkpoint."""
    if _MODEL_PATH is None:
        return None

    import pickle
    with open(_MODEL_PATH, "rb") as f:
        state = pickle.load(f)

    # Merge config: checkpoint arch_config (if present) overrides defaults
    ckpt_cfg = state.get("arch_config", state.get("config", state.get("net_config", {})))
    cfg = dict(_DEFAULT_CFG)
    # Override all architecture keys from checkpoint
    _ARCH_KEYS = ("d_model", "nhead", "nlayers", "ff_dim", "static", "split_heads",
                  "structured", "scratch_registers", "value_atoms", "value_vmax")
    for key in _ARCH_KEYS:
        if key in ckpt_cfg:
            cfg[key] = ckpt_cfg[key]
    # Map ff_dim → ff for build_token_net_mlx
    if "ff_dim" in ckpt_cfg:
        cfg["ff"] = ckpt_cfg["ff_dim"]

    net = build_token_net_mlx(_CARD_TABLE, cfg)
    model_state = state.get("model")
    if model_state is not None:
        # Flatten nested parameter dict (from model.parameters()) to key-value pairs
        if isinstance(model_state, dict):
            flat = nn.utils.tree_flatten(model_state)
        else:
            flat = model_state
        # Filter to only parameters the model actually has (skip numpy-backed like card_feat)
        model_param_keys = {k for k, _ in nn.utils.tree_flatten(net.parameters())}
        flat_filtered = [(k, v) for k, v in flat if k in model_param_keys]
        # Convert numpy arrays to MLX arrays
        flat_mlx = [(k, mx.array(v)) for k, v in flat_filtered]
        tree = nn.utils.tree_unflatten(flat_mlx)
        net.update(tree)
    net.eval()
    acc = state.get("val_acc", "?")
    print(f"[bc-agent] loaded MLX model {_MODEL_PATH} (val_acc={acc})")
    return net


_LOADED_MODEL = _load_model()

# ---- deck ----
def load_deck(path: str = _DECK_PATH) -> list[int]:
    with open(path) as f:
        return [int(line.strip().rstrip(",")) for line in f if line.strip()]

DECK: list[int] = load_deck()


def reload_deck(path: str = _DECK_PATH) -> list[int]:
    """Re-read deck.csv into DECK and return it.

    On Kaggle the deck never changes, so DECK is read once at import. Local
    tooling that swaps deck.csv between runs (the tournament deck sweep) must
    call this, or every swept deck plays the composition present at import.
    """
    global DECK
    DECK = load_deck(path)
    return DECK

# ---- per-side state (reset when deck is submitted) ----
_TRACKERS: dict[int, dict] = {}  # side -> {"tracker": GameTracker, "ability": AbilityTracker, "deck": list}


def _get_tracker(side: int):
    if side not in _TRACKERS:
        _TRACKERS[side] = {
            "tracker": GameTracker(),
            "ability": AbilityTracker(),
            "deck": None,
            "memory": None,  # F.1: persistent scratch register state
            "stale": True,   # needs a reset before its first decision
        }
    st = _TRACKERS[side]
    if st["stale"]:
        st["tracker"].reset()
        st["ability"].reset()
        st["deck"] = list(DECK)
        st["memory"] = None  # F.1: clean memory at match start
        st["stale"] = False
    return st


def _build_mlx_tensors(encoded: dict, int_keys: set) -> dict[str, mx.array]:
    """Convert encoded numpy arrays to MLX arrays with FP16 for numerics (E.4)."""
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        if k in int_keys:
            ob[k] = mx.array(arr.astype(np.int32)).reshape(1, *arr.shape)
        else:
            ob[k] = mx.array(arr.astype(np.float16)).reshape(1, *arr.shape)
    return ob


def _autoregressive_select(
    model,
    encoded: dict,
    int_keys: set,
    options: list,
    min_count: int,
    max_count: int,
    memory_in: mx.array | None = None,
) -> tuple[list[int], mx.array | None]:
    """Autoregressive multi-select: pick options one at a time, masking already-picked.

    For each substep:
      1. Encode with current picked set
      2. Forward pass through model
      3. Mask illegal + already-picked options
      4. Select best (argmax)
      5. If SUBMIT and enough picks -> break
      6. Add to picked set

    Returns (list of selected option indices, memory_out).
    """
    n = len(options)
    if n == 0:
        return [], memory_in

    if _LOADED_MODEL is None:
        count = max(min_count, min(max_count, n))
        return list(range(count)), memory_in

    picked_set: set[int] = set()
    results: list[int] = []
    current_memory = memory_in

    for _substep in range(max_count):
        # Build MLX tensors (FP16 for numerics)
        ob = _build_mlx_tensors(encoded, int_keys)

        # Forward pass with memory
        logits, _, memory_out = model.logits_value(ob, memory_in=current_memory)
        current_memory = memory_out
        logits_np = np.asarray(logits).flatten()

        # Mask illegal options
        action_mask = np.asarray(encoded["action_mask"]).flatten()
        logits_np[action_mask < 0.5] = -1e9

        # Mask already-picked options
        for p in picked_set:
            if p < len(logits_np):
                logits_np[p] = -1e9

        # Select best legal option
        action = int(np.argmax(logits_np))

        # SUBMIT is only accepted when min_count is satisfied
        if action == SUBMIT_ACTION and len(results) >= min_count:
            break

        # If SUBMIT was chosen but min_count not met, or if all options masked,
        # we cannot proceed further — return what we have
        if logits_np[action] <= -1e9:
            break

        picked_set.add(action)
        results.append(action)

        if len(results) >= max_count:
            break

    return results, current_memory


def choose(select: dict[str, Any], current: dict | None, logs: list | None = None) -> list[int]:
    """Pick option indices using the BC model with autoregressive multi-select.

    Args:
        select: the select dict from the engine (options, minCount, maxCount)
        current: the current game state
        logs: complete observation logs from the engine (E.2: never discard)
    """
    options = select.get("option") or []
    n = len(options)
    if n == 0:
        return []

    min_count = select.get("minCount", 0)
    max_count = select.get("maxCount", 1)

    # If no model loaded, fallback to baseline (first N legal options)
    if _LOADED_MODEL is None:
        count = max(min_count, min(max_count, n))
        return list(range(count))

    # Get side and trackers
    side = current.get("yourIndex", 0) if current else 0
    st = _get_tracker(side)
    tracker = st["tracker"]
    ability = st["ability"]
    deck = st["deck"]

    # E.2: Pass complete logs (never discard observation information)
    obs_for_encode = {"select": select, "current": current, "logs": logs or []}
    try:
        tracker.update(obs_for_encode)
        ability.note_turn(current.get("turn") if current else None)
    except Exception:
        pass

    # Encode observation
    try:
        encoded = _ENCODER.encode(
            obs_for_encode,
            picked=set(),
            self_deck=deck,
            tracker=tracker,
            ability_slots=ability.slots,
        )
    except Exception:
        count = max(min_count, min(max_count, n))
        return list(range(count))

    int_keys = _ENCODER.int_keys

    # F.1: Pass memory and store memory_out
    memory_in = st["memory"]
    results, memory_out = _autoregressive_select(
        _LOADED_MODEL, encoded, int_keys, options, min_count, max_count,
        memory_in=memory_in,
    )
    st["memory"] = memory_out  # F.1: persist memory for next decision

    return results


def agent(obs: dict[str, Any]) -> list[int]:
    """Engine entry point. Returns deck or chosen option indices."""
    select = obs.get("select")
    current = obs.get("current")

    # Deck submission (new match). The observation carries no side here, so
    # only mark state stale: _get_tracker resets each side lazily, on its first
    # decision. Eagerly clearing both sides is wrong whenever one process
    # serves both players (local self-play), because the second player's deck
    # submission would wipe the first player's live memory mid-match.
    if select is None:
        for st in _TRACKERS.values():
            st["stale"] = True
        return list(DECK)

    # E.2: Pass complete logs from observation to choose()
    logs = obs.get("logs", [])
    return choose(select, current, logs=logs)
