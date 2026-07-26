"""BC agent for the PTCG AI Battle Challenge.

Loads a trained TokenTransformer checkpoint and uses it to select actions.
Keeps per-side GameTracker + AbilityTracker for stateful encoding.

Usage:
  Used by evaluate.py, run_battle.py, and the Kaggle submission harness.
  The engine calls agent(obs) once per decision point.
"""
import os
import sys
from typing import Any

import numpy as np
import torch


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

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION
_DECK_PATH = os.path.join(_AGENT_DIR, "deck.csv")

# Detect MLX availability
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    from rl.policy_mlx import build_token_net_mlx
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False

from rl.policy import build_token_net  # PyTorch (always available for Kaggle)

# Model: search PyTorch (.pt) and MLX (.pkl) checkpoints
_MODEL_PATH = None
_MODEL_TYPE = None
for candidate, mtype in [
    # MLX checkpoints (priority — latest training)
    (os.path.join(_PROJECT_ROOT, "model", "bc_model", "bc_best_mlx_final.pkl"), "mlx"),
    (os.path.join(_PROJECT_ROOT, "model", "checkpoint", "bc_best_mlx.pkl"), "mlx"),
    # PyTorch checkpoints
    (os.path.join(_AGENT_DIR, "model", "bc_best_final.pt"), "torch"),
    (os.path.join(_PROJECT_ROOT, "model", "bc_model", "bc_best_final.pt"), "torch"),
    (os.path.join(_PROJECT_ROOT, "model", "checkpoint", "bc_best.pt"), "torch"),
]:
    if os.path.exists(candidate):
        _MODEL_PATH = candidate
        _MODEL_TYPE = mtype
        break

# ---- load once at import ----
if _MODEL_PATH is None:
    print(f"[bc-agent] WARNING: no checkpoint found, using random policy")
_CARD_TABLE = get_card_table()
_ENCODER = TokenEncoder(_CARD_TABLE)

def _load_model():
    """Load the BC model from checkpoint (MLX or PyTorch)."""
    if _MODEL_PATH is None:
        return None

    # MLX model
    if _MODEL_TYPE == "mlx" and _HAS_MLX:
        import pickle
        with open(_MODEL_PATH, "rb") as f:
            state = pickle.load(f)
        cfg = {"d_model": 128, "nhead": 4, "nlayers": 3,
               "static": True, "split_heads": True}
        net = build_token_net_mlx(_CARD_TABLE, cfg)
        if isinstance(state.get("model"), dict):
            net.update(state["model"])
        net.eval()
        acc = state.get("val_acc", "?")
        print(f"[bc-agent] loaded MLX model {_MODEL_PATH} (val_acc={acc})")
        return ("mlx", net)

    # PyTorch model
    if _MODEL_TYPE == "torch":
        ckpt = torch.load(_MODEL_PATH, map_location="cpu", weights_only=False)
        cfg = ckpt.get("net_config", {})
        net = build_token_net(_CARD_TABLE, cfg)
        net.load_state_dict(ckpt["net"])
        net.eval()
        acc = ckpt.get("bc_val_acc", "?")
        print(f"[bc-agent] loaded PyTorch model {_MODEL_PATH} (val_acc={acc})")
        return ("torch", net)

    return None

_LOADED = _load_model()

# ---- deck ----
def load_deck(path: str = _DECK_PATH) -> list[int]:
    with open(path) as f:
        return [int(line.strip().rstrip(",")) for line in f if line.strip()]

DECK: list[int] = load_deck()

# ---- per-side state (reset when deck is submitted) ----
_TRACKERS: dict[int, dict] = {}  # side -> {"tracker": GameTracker, "ability": AbilityTracker, "deck": list}


def _get_tracker(side: int):
    if side not in _TRACKERS:
        _TRACKERS[side] = {
            "tracker": GameTracker(),
            "ability": AbilityTracker(),
            "deck": None,
        }
    return _TRACKERS[side]


def choose(select: dict[str, Any], current: dict | None) -> list[int]:
    """Pick option indices using the BC model (or fallback to first-N)."""
    options = select.get("option") or []
    n = len(options)
    if n == 0:
        return []

    min_count = select.get("minCount", 0)
    max_count = select.get("maxCount", 1)

    # If no model loaded, fallback to baseline (first N legal options)
    if _LOADED is None:
        count = max(min_count, min(max_count, n))
        return list(range(count))

    model_type, model = _LOADED

    # Get side and trackers
    side = current.get("yourIndex", 0) if current else 0
    st = _get_tracker(side)
    tracker = st["tracker"]
    ability = st["ability"]
    deck = st["deck"]

    # Update tracker with current observation
    obs_for_encode = {"select": select, "current": current, "logs": []}
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

    # Build input tensors (MLX or PyTorch)
    int_keys = _ENCODER.int_keys
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        if model_type == "mlx":
            dtype = np.int32 if k in int_keys else np.float32
            ob[k] = mx.array(arr.astype(dtype)).reshape(1, *arr.shape)
        else:
            dtype = torch.long if k in int_keys else torch.float32
            ob[k] = torch.as_tensor(arr, dtype=dtype).unsqueeze(0)

    # Forward pass
    if model_type == "mlx":
        logits, _ = model.logits_value(ob)
        logits = logits.squeeze(0)
        action_mask = mx.array(encoded["action_mask"].astype(np.float32))
        logits = mx.where(action_mask < 0.5, -1e9, logits)
        logits_np = np.asarray(logits)
    else:
        with torch.no_grad():
            logits, _ = model.logits_value(ob)
        logits = logits.squeeze(0)
        action_mask = torch.as_tensor(encoded["action_mask"], dtype=torch.float32)
        logits = logits.masked_fill(action_mask < 0.5, -1e9)
        logits_np = logits.numpy()

    # Pick top-k legal options (numpy for simplicity)
    count = max(min_count, min(max_count, n))
    if count == 1:
        return [int(np.argmax(logits_np))]
    else:
        topk = np.argsort(-logits_np)[:count].tolist()
        return topk[:count]


def agent(obs: dict[str, Any]) -> list[int]:
    """Engine entry point. Returns deck or chosen option indices."""
    select = obs.get("select")
    current = obs.get("current")

    # Deck submission
    if select is None:
        # Reset trackers for both sides
        for side in (0, 1):
            st = _get_tracker(side)
            st["tracker"].reset()
            st["ability"].reset()
            st["deck"] = list(DECK)
        return list(DECK)

    return choose(select, current)
