"""BC agent for the PTCG AI Battle Challenge.

PyTorch-only inference with autoregressive multi-select.
Loads a strict FP16 PyTorch mirror of the MLX training checkpoint.
Keeps per-side GameTracker + AbilityTracker for stateful encoding.

Usage:
  Used by evaluate.py, run_battle.py, and the Kaggle submission harness.
  The engine calls agent(obs) once per decision point.
"""
import os
import random
import sys
from types import SimpleNamespace
from typing import Any


# ---- path setup (Kaggle-safe: __file__ is not defined when exec'd) ----
def _find_dir(filename: str = "deck.csv") -> str:
    """Find the directory containing `filename`, searching sys.path + common locations."""
    local_agent_dir = (
        os.path.dirname(os.path.abspath(__file__))
        if "__file__" in globals()
        else None
    )
    candidates = [
        local_agent_dir,
        "/kaggle_simulations/agent",
        *sys.path,
        os.getcwd(),
    ]
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
# An opt-in local candidate may be produced by the repository's strict FP32
# loader while this submission directory still carries the older FP16 mirror.
# Preload the repository package only for that explicit path, leaving the
# default submission import path and behavior unchanged.
_OPT_IN_MODEL_PATH = os.environ.get("PTCG_MODEL_PATH")
if _OPT_IN_MODEL_PATH:
    _REPOSITORY_ROOT = os.path.abspath(os.path.join(_PROJECT_ROOT, "../../.."))
    sys.path = [entry for entry in sys.path if os.path.abspath(entry or os.getcwd()) != _PROJECT_ROOT]
    sys.path.insert(0, _REPOSITORY_ROOT)

import numpy as np
import torch
from rl.policy_infer_torch import load_inference_checkpoint

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION

_DECK_PATH = os.path.join(_AGENT_DIR, "deck.csv")

# ---- model checkpoint search ----
# Prefer the pre-converted FP16 artifact in submissions. Local development can
# still load the MLX trainer checkpoint through the strict converter.
_MODEL_PATH = None
if _OPT_IN_MODEL_PATH:
    if not os.path.isfile(_OPT_IN_MODEL_PATH):
        raise FileNotFoundError(f"PTCG_MODEL_PATH does not exist: {_OPT_IN_MODEL_PATH}")
    _MODEL_PATH = _OPT_IN_MODEL_PATH
else:
    for candidate in [
        os.path.join(_PROJECT_ROOT, "model", "bc_model", "bc_best_torch_fp16.pt"),
        os.path.join(_PROJECT_ROOT, "model", "checkpoint", "bc_best_torch_fp16.pt"),
        os.path.join(_PROJECT_ROOT, "model", "bc_model", "bc_best_mlx_final.pkl"),
        os.path.join(_PROJECT_ROOT, "model", "checkpoint", "bc_best_mlx.pkl"),
        os.path.join(_PROJECT_ROOT, "model", "bc_model", "bc_best_final.pkl"),
        os.path.join(_PROJECT_ROOT, "model", "checkpoint", "bc_best.pkl"),
    ]:
        if os.path.exists(candidate):
            _MODEL_PATH = candidate
            break

if _OPT_IN_MODEL_PATH:
    # Candidate provenance is checked before the strict model loader. This is
    # deliberately opt-in only; the normal submission search and checkpoint
    # path remain unchanged.
    from scripts.rl.ppo_micro_update import validate_candidate_provenance
    from scripts.rl.trajectory_probe import APPROVED_STAGE4_ROOT_SHA256

    _OPT_IN_PROVENANCE = validate_candidate_provenance(
        _MODEL_PATH,
        approved_root_sha256=APPROVED_STAGE4_ROOT_SHA256,
    )

# ---- load once at import ----
if _MODEL_PATH is None:
    print("[bc-agent] WARNING: no checkpoint found, using random policy")
_CARD_TABLE = get_card_table()
_ENCODER = TokenEncoder(_CARD_TABLE)

def _load_model():
    """Load a strict FP16 PyTorch inference model.

    Loading is strict: a shape/parameter mismatch raises rather than falling
    back to a partial model, so a broken checkpoint is loud, not silent.
    """
    if _MODEL_PATH is None:
        return None, {}, {
            "version": 1,
            "seed": 0,
            "bc_would_ko": False,
            "bc_wk_nvar": 10,
            "provenance": "no-checkpoint",
            "prospective_planner": {
                "enabled": False,
                "config": None,
                "runtime": None,
                "trained_optimizer_steps": 0,
                "provenance": "no-checkpoint",
            },
        }
    net, metadata = load_inference_checkpoint(_MODEL_PATH, _CARD_TABLE)
    runtime_cfg = metadata["inference_config"]
    prospective_cfg = runtime_cfg.get("prospective_planner", {"enabled": False})
    print(f"[bc-agent] loaded PyTorch model {_MODEL_PATH} "
          f"(nlayers={metadata['nlayers']}, "
          f"scratch={metadata['scratch_registers']}, "
          f"dtype={next(net.parameters()).dtype}, "
          f"would_ko={runtime_cfg['bc_would_ko']}, "
          f"prospective={prospective_cfg['enabled']})")
    return net, metadata, runtime_cfg


_LOADED_MODEL, _MODEL_METADATA, _RUNTIME_DATA = _load_model()
_RUNTIME_CFG = SimpleNamespace(**_RUNTIME_DATA)
_PROSPECTIVE_MODEL = _MODEL_METADATA.get("prospective_planner_model")
_MODEL_DTYPE = (
    next(_LOADED_MODEL.parameters()).dtype
    if _LOADED_MODEL is not None
    else torch.float32
)

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
            "would_ko_rng": None,
            "decision_index": 0,
            "stale": True,   # needs a reset before its first decision
        }
    st = _TRACKERS[side]
    if st["stale"]:
        st["tracker"].reset()
        st["ability"].reset()
        st["deck"] = list(DECK)
        st["memory"] = None  # F.1: clean memory at match start
        st["would_ko_rng"] = random.Random(int(_RUNTIME_CFG.seed) + int(side))
        st["decision_index"] = 0
        st["stale"] = False
    return st


def _build_tensors(encoded: dict, int_keys: set) -> dict:
    """Convert encoded arrays to tensors matching the selected checkpoint."""
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        if k in int_keys:
            ob[k] = torch.as_tensor(arr.astype(np.int64)).reshape(1, *arr.shape)
        else:
            ob[k] = torch.as_tensor(arr.astype(np.float32)).to(_MODEL_DTYPE).reshape(1, *arr.shape)
    return ob


def _logits_to_numpy(logits) -> np.ndarray:
    """Flatten PyTorch logits to a float32 NumPy vector."""
    return logits.detach().to(torch.float32).numpy().flatten()


def _autoregressive_select(
    model,
    encode_step,
    int_keys: set,
    options: list,
    min_count: int,
    max_count: int,
    memory_in=None,
) -> tuple[list[int], Any]:
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
    memory_out = memory_in

    for _substep in range(max_count):
        # Re-encode the same decision with the actual buffered picks. This
        # updates action_mask, OPT_PICKED and SUBMIT exactly like BC rows.
        encoded = encode_step(set(picked_set))
        ob = _build_tensors(encoded, int_keys)

        # Every substep belongs to one engine decision. They all read the same
        # incoming recurrent state; only the final substep's state is committed
        # to the next decision.
        with torch.inference_mode():
            logits, _, memory_out = model.logits_value(ob, memory_in=memory_in)
        logits_np = _logits_to_numpy(logits)

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

        # SUBMIT chosen too early, or everything masked: take the best real
        # option instead of stopping. Breaking here returns fewer than
        # min_count picks, which the engine rejects as an INVALID action.
        if action == SUBMIT_ACTION or logits_np[action] <= -1e9:
            fallback = [i for i in range(n) if i not in picked_set]
            if not fallback:
                break
            legal = [i for i in fallback if i < len(action_mask) and action_mask[i] >= 0.5]
            pool = legal or fallback
            action = max(pool, key=lambda i: logits_np[i] if i < len(logits_np) else -1e9)

        picked_set.add(action)
        results.append(action)

        if len(results) >= max_count:
            break

    # Never hand back fewer picks than the engine demands.
    if len(results) < min_count:
        for i in range(n):
            if len(results) >= min_count:
                break
            if i not in picked_set:
                picked_set.add(i)
                results.append(i)

    return results, memory_out


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

    # The same prospective feature used by dataset construction is computed
    # once per real decision, before all autoregressive substeps.
    if _RUNTIME_CFG.bc_would_ko:
        try:
            from rl.search_agent import annotate_would_ko
            annotate_would_ko(
                obs_for_encode,
                deck,
                _ENCODER,
                n_var=int(_RUNTIME_CFG.bc_wk_nvar),
                rng=st["would_ko_rng"],
            )
        except Exception:
            # Inference remains legal if the engine cannot simulate an
            # incomplete observation; the encoder then emits the neutral trio.
            pass

    def encode_step(picked: set[int]) -> dict:
        return _ENCODER.encode(
            obs_for_encode,
            picked=picked,
            self_deck=deck,
            tracker=tracker,
            ability_slots=ability.slots,
        )

    int_keys = _ENCODER.int_keys

    # F.1: Pass memory and store memory_out
    memory_in = st["memory"]
    match_time = int(st["decision_index"])
    st["decision_index"] = match_time + 1

    # A trained, checkpoint-enabled prospective planner may select one complete
    # legal action (including multi-select combinations). Search failure is a
    # local fallback condition: the existing Transformer/TBPTT path below is
    # unchanged. The trunk advances exactly once when the planner succeeds.
    prospective_cfg = _RUNTIME_DATA["prospective_planner"]
    if prospective_cfg["enabled"] and _PROSPECTIVE_MODEL is not None:
        try:
            from rl.prospective_runtime import (
                ProspectiveRuntimeConfig,
                build_runtime_prospective_tree,
                rerank_with_prospective_planner,
            )
            from rl.prospective_schema import ProspectivePlannerConfig

            encoded_context = encode_step(set())
            context_observation = _build_tensors(encoded_context, int_keys)
            with torch.inference_mode():
                cls_out, _, pooled, _, planner_memory_out = (
                    _LOADED_MODEL._encode(
                        context_observation,
                        memory_in=memory_in,
                    )
                )
                trunk_context = torch.cat(
                    (
                        cls_out[:, None, :],
                        pooled[:, None, :],
                        planner_memory_out,
                    ),
                    dim=1,
                )[0].detach().cpu().numpy()
            tree = build_runtime_prospective_tree(
                obs_for_encode,
                list(deck),
                _ENCODER,
                trunk_context,
                planner_config=ProspectivePlannerConfig(
                    **prospective_cfg["config"]
                ),
                runtime_config=ProspectiveRuntimeConfig.from_dict(
                    prospective_cfg["runtime"]
                ),
                match_time=match_time,
            )
            reranked = (
                rerank_with_prospective_planner(_PROSPECTIVE_MODEL, tree)
                if tree is not None
                else None
            )
            if reranked is not None:
                results = list(reranked.action)
                if not (
                    min_count <= len(results) <= max_count
                    and len(set(results)) == len(results)
                    and all(0 <= index < n for index in results)
                ):
                    raise ValueError("prospective planner returned an illegal action")
                # The exact memory produced while constructing the trained
                # [CLS, pooled, scratch] planner context is committed once.
                st["memory"] = planner_memory_out
                ability.record(select, results)
                return results
        except Exception:
            # Planner search is strictly additive. Any unavailable engine
            # state, rejected determinization, or malformed result retains the
            # checkpoint-compatible Transformer behavior.
            pass

    try:
        results, memory_out = _autoregressive_select(
            _LOADED_MODEL, encode_step, int_keys, options, min_count, max_count,
            memory_in=memory_in,
        )
        st["memory"] = memory_out  # F.1: persist memory for next decision
    except Exception:
        results = []

    # Last line of defence before the engine: an action short of min_count is
    # INVALID and forfeits the game, so fall back to the first legal indices.
    if len(results) < min_count:
        results = list(range(max(min_count, min(max_count, n))))

    ability.record(select, results)
    return results


def _agent_impl(obs: dict[str, Any]) -> list[int]:
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


# Must stay the last callable defined in this module: the Kaggle harness picks
# whichever callable is defined last (kaggle_environments/agent.py).
def agent(obs: dict[str, Any]) -> list[int]:
    """Engine entry point. Returns deck or chosen option indices.

    Wraps the implementation so no exception can escape: the harness turns a
    raised exception into an INVALID action, which forfeits the game outright.
    The traceback goes to stdout, where the episode's agent logs capture it.
    """
    try:
        return _agent_impl(obs)
    except Exception:
        import traceback
        traceback.print_exc()

        select = obs.get("select") if hasattr(obs, "get") else getattr(obs, "select", None)
        if select is None:
            return list(DECK)
        get = (lambda k, d: select.get(k, d)) if hasattr(select, "get") \
            else (lambda k, d: getattr(select, k, d))
        n = len(get("option", []) or [])
        min_count = get("minCount", 1) or 1
        max_count = get("maxCount", 1) or 1
        return list(range(max(min_count, min(max_count, n))))
