"""BC agent for the PTCG AI Battle Challenge.

PyTorch-only inference with autoregressive multi-select.
Loads a strict FP16 PyTorch mirror of the MLX training checkpoint.
Keeps per-side GameTracker + AbilityTracker for stateful encoding.

Inference mode is selected via PTCG_INFERENCE_MODE env var:
  baseline  — autoregressive multi-select (default)
  b1        — Linha 2 B1: K latent TRM-style refinements before deciding
  b2        — Linha 2 B2: K deterministic latent perturbations, mean logits

Extra env vars:
  PTCG_LATENT_K        — iterations / perturbations for b1/b2 (default 3)
  PTCG_PERTURB_EPS     — perturbation magnitude for b2 (default 0.05)

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


_AGENT_DIR = _find_dir("deck.csv")
if os.path.isdir(os.path.join(_AGENT_DIR, "rl")):
    _PROJECT_ROOT = _AGENT_DIR
else:
    _PROJECT_ROOT = os.path.dirname(_AGENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch
from rl.policy_infer_torch import load_inference_checkpoint

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION

_DECK_PATH = os.path.join(_AGENT_DIR, "deck.csv")

# ---- inference mode ----
_INFERENCE_MODE = os.environ.get("PTCG_INFERENCE_MODE", "baseline").strip().lower()
_VALID_MODES = {"baseline", "b1", "b2"}
if _INFERENCE_MODE not in _VALID_MODES:
    print(f"[bc-agent] WARNING: unknown PTCG_INFERENCE_MODE={_INFERENCE_MODE!r}; falling back to baseline")
    _INFERENCE_MODE = "baseline"
_LATENT_K = int(os.environ.get("PTCG_LATENT_K", "3"))
_PERTURB_EPS = float(os.environ.get("PTCG_PERTURB_EPS", "0.05"))

# ---- model checkpoint search ----
_MODEL_PATH = None
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

if _MODEL_PATH is None:
    print("[bc-agent] WARNING: no checkpoint found, using random policy")
_CARD_TABLE = get_card_table()
_ENCODER = TokenEncoder(_CARD_TABLE)

def _load_model():
    if _MODEL_PATH is None:
        return None, {}, {
            "version": 1, "seed": 0, "bc_would_ko": False, "bc_wk_nvar": 10,
            "provenance": "no-checkpoint",
        }
    net, metadata = load_inference_checkpoint(_MODEL_PATH, _CARD_TABLE)
    runtime_cfg = metadata["inference_config"]
    print(f"[bc-agent] loaded PyTorch FP16 model {_MODEL_PATH} "
          f"(nlayers={metadata['nlayers']}, "
          f"scratch={metadata['scratch_registers']}, "
          f"would_ko={runtime_cfg['bc_would_ko']}, "
          f"mode={_INFERENCE_MODE})")
    return net, metadata, runtime_cfg


_LOADED_MODEL, _MODEL_METADATA, _RUNTIME_DATA = _load_model()
_RUNTIME_CFG = SimpleNamespace(**_RUNTIME_DATA)

# ---- deck ----
def load_deck(path: str = _DECK_PATH) -> list[int]:
    with open(path) as f:
        return [int(line.strip().rstrip(",")) for line in f if line.strip()]

DECK: list[int] = load_deck()


def reload_deck(path: str = _DECK_PATH) -> list[int]:
    global DECK
    DECK = load_deck(path)
    return DECK

# ---- per-side state ----
_TRACKERS: dict[int, dict] = {}


def _get_tracker(side: int):
    if side not in _TRACKERS:
        _TRACKERS[side] = {
            "tracker": GameTracker(),
            "ability": AbilityTracker(),
            "deck": None,
            "memory": None,
            "would_ko_rng": None,
            "decision_index": 0,
            "stale": True,
        }
    st = _TRACKERS[side]
    if st["stale"]:
        st["tracker"].reset()
        st["ability"].reset()
        st["deck"] = list(DECK)
        st["memory"] = None
        st["would_ko_rng"] = random.Random(int(_RUNTIME_CFG.seed) + int(side))
        st["decision_index"] = 0
        st["stale"] = False
    return st


def _build_tensors(encoded: dict, int_keys: set) -> dict:
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        if k in int_keys:
            ob[k] = torch.as_tensor(arr.astype(np.int64)).reshape(1, *arr.shape)
        else:
            ob[k] = torch.as_tensor(arr.astype(np.float16)).reshape(1, *arr.shape)
    return ob


def _logits_to_numpy(logits) -> np.ndarray:
    return logits.detach().to(torch.float32).numpy().flatten()


# ---------- forward variants per mode ----------

def _forward_baseline(ob, memory_in):
    """Standard single forward pass."""
    with torch.inference_mode():
        logits, _, memory_out = _LOADED_MODEL.logits_value(ob, memory_in=memory_in)
    return logits, memory_out


def _forward_b1(ob, memory_in):
    """B1: K TRM-style latent refinements. Same obs, evolve memory K-1 times, then read policy."""
    memory = memory_in
    with torch.inference_mode():
        for _ in range(max(0, _LATENT_K - 1)):
            _, _, memory = _LOADED_MODEL.logits_value(ob, memory_in=memory)
        logits, _, memory_out = _LOADED_MODEL.logits_value(ob, memory_in=memory)
    return logits, memory_out


def _forward_b2(ob, memory_in):
    """B2: K deterministic perturbations of memory, mean of logits, middle perturbation memory committed."""
    if memory_in is None:
        base = _LOADED_MODEL.learned_init.detach().unsqueeze(0).to(torch.float16)
    else:
        base = memory_in.detach().to(torch.float16)
    shape = tuple(base.shape)
    logits_stack = []
    mem_stack = []
    with torch.inference_mode():
        for k in range(_LATENT_K):
            rng = np.random.default_rng(seed=1000 + k)
            noise_np = (rng.standard_normal(shape) * _PERTURB_EPS).astype(np.float16)
            noise = torch.from_numpy(noise_np)
            perturbed = base + noise
            logits_k, _, mem_k = _LOADED_MODEL.logits_value(ob, memory_in=perturbed)
            logits_stack.append(logits_k.to(torch.float32))
            mem_stack.append(mem_k)
    mean_logits = torch.stack(logits_stack, dim=0).mean(dim=0).to(torch.float16)
    committed_memory = mem_stack[_LATENT_K // 2]
    return mean_logits, committed_memory


# ---------- autoregressive select with pluggable forward ----------

def _select_action_from_logits(logits_np, picked_set, action_mask, n, min_count, results):
    """Shared post-processing: mask picked, choose action, handle SUBMIT/illegal."""
    logits_np = logits_np.copy()
    logits_np[action_mask < 0.5] = -1e9
    for p in picked_set:
        if p < len(logits_np):
            logits_np[p] = -1e9
    action = int(np.argmax(logits_np))
    if action == SUBMIT_ACTION and len(results) >= min_count:
        return SUBMIT_ACTION, logits_np
    if action == SUBMIT_ACTION or logits_np[action] <= -1e9:
        fallback = [i for i in range(n) if i not in picked_set]
        if not fallback:
            return None, logits_np
        legal = [i for i in fallback if i < len(action_mask) and action_mask[i] >= 0.5]
        pool = legal or fallback
        action = max(pool, key=lambda i: logits_np[i] if i < len(logits_np) else -1e9)
    return action, logits_np


def _autoregressive_select_mode(
    encode_step,
    int_keys,
    options,
    min_count,
    max_count,
    memory_in,
    obs_for_encode,
    deck,
    match_time,
):
    """Autoregressive multi-select with mode-aware forward and optional sidecar mixing."""
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
        encoded = encode_step(set(picked_set))
        ob = _build_tensors(encoded, int_keys)
        action_mask = np.asarray(encoded["action_mask"]).flatten()

        if _INFERENCE_MODE == "b1":
            logits, memory_out = _forward_b1(ob, memory_in)
            logits_np = _logits_to_numpy(logits)
        elif _INFERENCE_MODE == "b2":
            logits, memory_out = _forward_b2(ob, memory_in)
            logits_np = _logits_to_numpy(logits)
        else:  # baseline
            logits, memory_out = _forward_baseline(ob, memory_in)
            logits_np = _logits_to_numpy(logits)

        action, _ = _select_action_from_logits(
            logits_np, picked_set, action_mask, n, min_count, results,
        )
        if action == SUBMIT_ACTION:
            break
        if action is None:
            break
        picked_set.add(action)
        results.append(action)
        if len(results) >= max_count:
            break

    if len(results) < min_count:
        for i in range(n):
            if len(results) >= min_count:
                break
            if i not in picked_set:
                picked_set.add(i)
                results.append(i)

    return results, memory_out


def choose(select: dict[str, Any], current: dict | None, logs: list | None = None) -> list[int]:
    options = select.get("option") or []
    n = len(options)
    if n == 0:
        return []
    min_count = select.get("minCount", 0)
    max_count = select.get("maxCount", 1)
    if _LOADED_MODEL is None:
        count = max(min_count, min(max_count, n))
        return list(range(count))

    side = current.get("yourIndex", 0) if current else 0
    st = _get_tracker(side)
    tracker = st["tracker"]
    ability = st["ability"]
    deck = st["deck"]

    obs_for_encode = {"select": select, "current": current, "logs": logs or []}
    try:
        tracker.update(obs_for_encode)
        ability.note_turn(current.get("turn") if current else None)
    except Exception:
        pass

    if _RUNTIME_CFG.bc_would_ko:
        try:
            from rl.search_agent import annotate_would_ko
            annotate_would_ko(
                obs_for_encode, deck, _ENCODER,
                n_var=int(_RUNTIME_CFG.bc_wk_nvar),
                rng=st["would_ko_rng"],
            )
        except Exception:
            pass

    def encode_step(picked: set[int]) -> dict:
        return _ENCODER.encode(
            obs_for_encode, picked=picked, self_deck=deck,
            tracker=tracker, ability_slots=ability.slots,
        )

    int_keys = _ENCODER.int_keys
    memory_in = st["memory"]
    match_time = int(st["decision_index"])
    st["decision_index"] = match_time + 1

    try:
        results, memory_out = _autoregressive_select_mode(
            encode_step=encode_step,
            int_keys=int_keys,
            options=options,
            min_count=min_count,
            max_count=max_count,
            memory_in=memory_in,
            obs_for_encode=obs_for_encode,
            deck=deck,
            match_time=match_time,
        )
        st["memory"] = memory_out
    except Exception:
        results = []

    if len(results) < min_count:
        results = list(range(max(min_count, min(max_count, n))))

    ability.record(select, results)
    return results


def _agent_impl(obs: dict[str, Any]) -> list[int]:
    select = obs.get("select")
    current = obs.get("current")
    if select is None:
        for st in _TRACKERS.values():
            st["stale"] = True
        return list(DECK)
    logs = obs.get("logs", [])
    return choose(select, current, logs=logs)


def agent(obs: dict[str, Any]) -> list[int]:
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
