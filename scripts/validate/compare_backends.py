"""Compare the MLX policy and the PyTorch inference mirror on REAL decisions.

Runs one local cabt battle to capture the exact observations the agent receives
(complete logs, per-side trackers, the same TokenEncoder), then replays every
captured decision through both backends and reports per-component divergence.

This does NOT use the replay JSON as ground truth: replays serialize some fields
differently from the live engine, so observations are captured straight from the
environment instead. FP32 is used for the numeric comparison to isolate model
equivalence from FP32 rounding; the shipped agent still runs FP32.

Usage::

    uv run python scripts/validate/compare_backends.py --decisions 40
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

import mlx.core as mx
import mlx.nn as mlx_nn

from scripts._common import make_env  # noqa: F401 — sets up kaggle_environments quietly
from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder, GameTracker, AbilityTracker, SUBMIT_ACTION
from rl.policy_mlx import build_token_net_mlx
from rl.policy_infer_torch import load_mlx_checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT = os.path.join(ROOT, "model", "bc_model", "bc_best_mlx_final.pkl")
DECK_PATH = os.path.join(ROOT, "agent", "deck.csv")


def _load_deck() -> list[int]:
    with open(DECK_PATH) as fh:
        return [int(line.strip().rstrip(",")) for line in fh if line.strip()]


def _build_mlx() -> mlx_nn.Module:
    import pickle

    with open(CHECKPOINT, "rb") as fh:
        state = pickle.load(fh)
    model = build_token_net_mlx(get_card_table(), state["arch_config"])
    leaves = mlx_nn.utils.tree_flatten(state["model"])
    keys = {k for k, _ in mlx_nn.utils.tree_flatten(model.parameters())}
    model.update(mlx_nn.utils.tree_unflatten([(k, mx.array(v)) for k, v in leaves if k in keys]))
    model.eval()
    return model


def _to_mlx(encoded: dict, int_keys: set) -> dict:
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        ob[k] = mx.array(arr.astype(np.int32 if k in int_keys else np.float32)).reshape(1, *arr.shape)
    return ob


def _to_torch(encoded: dict, int_keys: set) -> dict:
    ob = {}
    for k, v in encoded.items():
        arr = np.asarray(v)
        dtype = torch.int64 if k in int_keys else torch.float32
        ob[k] = torch.as_tensor(arr.astype(np.int64 if k in int_keys else np.float32), dtype=dtype).reshape(1, *arr.shape)
    return ob


class _CapturingAgent:
    """A cabt agent that plays with MLX but records every decision's observation."""

    def __init__(self, mlx_model, encoder: TokenEncoder, deck: list[int]) -> None:
        self._mlx = mlx_model
        self._encoder = encoder
        self._deck = deck
        self._sides: dict[int, dict] = {}
        self.records: list[dict] = []

    def _state(self, side: int) -> dict:
        if side not in self._sides:
            self._sides[side] = {
                "tracker": GameTracker(),
                "ability": AbilityTracker(),
                "memory": None,
            }
        return self._sides[side]

    def __call__(self, obs: dict, *_config) -> list[int]:
        # kaggle_environments passes (observation, configuration) to a callable
        # object; a plain function is called with just the observation. Accept
        # the extra positional arg so both invocation styles work.
        try:
            return self._decide(obs)
        except Exception:
            import traceback
            traceback.print_exc()
            raise

    def _decide(self, obs: dict) -> list[int]:
        select = obs.get("select")
        if select is None:
            for st in self._sides.values():
                st["tracker"].reset()
                st["ability"].reset()
                st["memory"] = None
            return list(self._deck)

        options = select.get("option") or []
        if not options:
            return []
        current = obs.get("current") or {}
        side = current.get("yourIndex", 0)
        st = self._state(side)
        obs_for_encode = {"select": select, "current": current, "logs": obs.get("logs", []) or []}
        st["tracker"].update(obs_for_encode)
        st["ability"].note_turn(current.get("turn"))
        encoded = self._encoder.encode(
            obs_for_encode, picked=set(), self_deck=self._deck,
            tracker=st["tracker"], ability_slots=st["ability"].slots,
        )

        memory_in = st["memory"]
        ob = _to_mlx(encoded, self._encoder.int_keys)
        logits, _, memory_out = self._mlx.logits_value(
            ob, memory_in=(mx.array(memory_in) if memory_in is not None else None))
        mx.eval(logits, memory_out)
        st["memory"] = np.asarray(memory_out)

        # Record this decision for the offline comparison.
        self.records.append({
            "encoded": {k: np.asarray(v).copy() for k, v in encoded.items()},
            "memory_in": None if memory_in is None else np.asarray(memory_in).copy(),
            "min_count": int(select.get("minCount", 0)),
            "max_count": int(select.get("maxCount", 1)),
            "n_options": len(options),
        })

        logits_np = np.asarray(logits).flatten()
        action_mask = np.asarray(encoded["action_mask"]).flatten()
        logits_np[action_mask < 0.5] = -1e9
        action = int(np.argmax(logits_np))
        min_count = int(select.get("minCount", 0))
        if action == SUBMIT_ACTION and 0 >= min_count:
            return [i for i in range(min(len(options), max(1, min_count)))]
        if action == SUBMIT_ACTION or logits_np[action] <= -1e9:
            legal = [i for i in range(len(options)) if action_mask[i] >= 0.5]
            action = legal[0] if legal else 0
        return [action]


# A mismatch is only a real policy divergence when the two disputed options are
# NOT tied: if MLX and PyTorch each see a clear (> TIE_TOL) winner and still
# disagree, that is a divergence. Options whose logits are equal within TIE_TOL
# are genuine ties where argmax tie-breaking is allowed to differ.
_TIE_TOL = 1e-3


def _capture_battle(mlx_model, encoder: TokenEncoder, deck: list[int]) -> list[dict]:
    from kaggle_environments import make
    env = make("cabt", configuration={})
    capture = _CapturingAgent(mlx_model, encoder, deck)
    env.run([capture, capture])
    statuses = [s.status for s in env.steps[-1]]
    print(f"[compare] battle: steps={len(env.steps)} statuses={statuses} "
          f"decisions={len(capture.records)}")
    return capture.records


def compare(decisions: int, battles: int) -> int:
    encoder = TokenEncoder(get_card_table())
    deck = _load_deck()

    # Capture real decisions with the MLX backend, then load the PyTorch mirror
    # only afterward. Keeping the two runtimes ordered mirrors how the shipped
    # agent selects a single backend per process.
    mlx_model = _build_mlx()
    records: list[dict] = []
    for _ in range(max(1, battles)):
        records.extend(_capture_battle(mlx_model, encoder, deck))
        if decisions > 0 and len(records) >= decisions:
            break
    if decisions > 0:
        records = records[:decisions]

    torch_model, cfg = load_mlx_checkpoint(CHECKPOINT, get_card_table(), dtype=torch.float32)
    print(f"[compare] comparing {len(records)} decisions")

    worst_logit = worst_value = worst_memory = 0.0
    benign_ties = 0
    real_divergences = 0
    int_keys = encoder.int_keys
    for i, rec in enumerate(records):
        ob_m = _to_mlx(rec["encoded"], int_keys)
        ob_t = _to_torch(rec["encoded"], int_keys)
        mem = rec["memory_in"]
        mem_m = mx.array(mem) if mem is not None else None
        mem_t = torch.as_tensor(mem, dtype=torch.float32) if mem is not None else None

        lm, vm, mm = mlx_model.logits_value(ob_m, memory_in=mem_m)
        mx.eval(lm, vm, mm)
        with torch.inference_mode():
            lt, vt, mt = torch_model.logits_value(ob_t, memory_in=mem_t)

        mask = np.asarray(rec["encoded"]["action_mask"]).flatten() > 0.5
        a = np.asarray(lm).flatten()
        b = lt.numpy().flatten()
        worst_logit = max(worst_logit, float(np.max(np.abs(a[mask] - b[mask]))) if mask.any() else 0.0)
        worst_value = max(worst_value, float(np.max(np.abs(np.asarray(vm) - vt.numpy()))))
        worst_memory = max(worst_memory, float(np.max(np.abs(np.asarray(mm) - mt.numpy()))))

        am_m = int(np.argmax(np.where(mask, a, -np.inf)))
        am_t = int(np.argmax(np.where(mask, b, -np.inf)))
        if am_m == am_t:
            continue
        # Disagreement: benign if the two options are tied in BOTH backends.
        tied = abs(a[am_m] - a[am_t]) <= _TIE_TOL and abs(b[am_m] - b[am_t]) <= _TIE_TOL
        if tied:
            benign_ties += 1
            print(f"  [decision {i}] benign tie mlx={am_m} torch={am_t} "
                  f"margin={abs(a[am_m]-a[am_t]):.2e}")
        else:
            real_divergences += 1
            print(f"  [decision {i}] DIVERGENCE mlx={am_m} torch={am_t} "
                  f"mlx[{am_m}]={a[am_m]:.5f} mlx[{am_t}]={a[am_t]:.5f} "
                  f"torch[{am_m}]={b[am_m]:.5f} torch[{am_t}]={b[am_t]:.5f}")

    print(f"[compare] worst legal-logit diff : {worst_logit:.3e}")
    print(f"[compare] worst value diff       : {worst_value:.3e}")
    print(f"[compare] worst memory diff      : {worst_memory:.3e}")
    print(f"[compare] benign ties            : {benign_ties}")
    print(f"[compare] real divergences       : {real_divergences}")

    ok = (worst_logit < 1e-2 and worst_value < 1e-2 and worst_memory < 1e-2
          and real_divergences == 0)
    print("[compare] PASS" if ok else "[compare] FAIL")
    return 0 if ok else 1


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--decisions", type=int, default=0, help="Max decisions to compare (0 = all)")
    p.add_argument("--battles", type=int, default=3, help="Self-play battles to sample decisions from")
    args = p.parse_args()
    raise SystemExit(compare(args.decisions, args.battles))


if __name__ == "__main__":
    main()
