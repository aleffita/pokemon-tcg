"""Generate correctly-shaped synthetic BC datasets for smoke testing.

Creates a directory of .npy files matching the TokenEncoder output schema.
All shape constants are imported from rl/encoder/enc_constants.py.

Usage:
  PYTHONPATH=. uv run python scripts/validate/make_synthetic_data.py \
      --rows 1000 --out data/bc_data/synthetic_1k
"""
from __future__ import annotations

import argparse
import os

import numpy as np

# --- shape constants from the single source of truth ---
from rl.encoder.enc_constants import (
    DECK_SIZE, N_PRIZE, MAX_HAND, MAX_DISCARD, N_STADIUM,
    N_BENCH, N_PREEVO, N_TOOLS, N_ENERGY_CARDS, UNIT_ATTR,
    MAX_OPTIONS, OPT_STRUCT, N_ACTIONS,
    G, N_STATE_TOKENS,
)
from rl.encoder.effect_data import N_ATTACK_FX


# --- vocab sizes ---
CARD_VOCAB = 1268          # EN_Card_Data rows + UNK
OPT_ATTR_DIM = OPT_STRUCT + N_ATTACK_FX


def make_dataset(n_rows: int, out_dir: str, seed: int = 42) -> None:
    """Write a directory of .npy files with n_rows synthetic observations."""
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)

    # helpers
    def _rand_ids(shape, vocab=CARD_VOCAB, lo=0):
        """Random card ids in [lo, vocab)."""
        return rng.integers(lo, vocab, size=shape, dtype=np.int32)

    def _rand_float(shape, scale=1.0):
        """Random normal clipped to [0, 1] (approx), then scaled."""
        return (rng.standard_normal(shape).astype(np.float32) * 0.2 + 0.5).clip(0.0, 1.0) * scale

    # --- __labels__ and __is_attack__ ---
    labels = np.zeros(n_rows, dtype=np.int32)
    is_attack = rng.random(n_rows) < 0.3

    # --- int keys (int32) ---
    int_arrays: dict[str, np.ndarray] = {
        "select_type":      rng.integers(0, 16, size=(n_rows, 1), dtype=np.int32),
        "select_context":   rng.integers(0, 64, size=(n_rows, 1), dtype=np.int32),
        "effect_id":        rng.integers(0, CARD_VOCAB, size=(n_rows, 2), dtype=np.int32),
        # card-list streams (ids in [0, CARD_VOCAB))
        "self_deck_id":     _rand_ids((n_rows, DECK_SIZE)),
        "opp_deck_id":      _rand_ids((n_rows, DECK_SIZE)),
        "self_prize_id":    _rand_ids((n_rows, N_PRIZE)),
        "opp_prize_id":     _rand_ids((n_rows, N_PRIZE)),
        "self_hand_id":     _rand_ids((n_rows, MAX_HAND)),
        "opp_hand_id":      _rand_ids((n_rows, MAX_HAND)),
        "self_discard_id":  _rand_ids((n_rows, MAX_DISCARD)),
        "opp_discard_id":   _rand_ids((n_rows, MAX_DISCARD)),
        "stadium_id":       _rand_ids((n_rows, N_STADIUM)),
        # unit ids
        "self_unit_top_id":     _rand_ids((n_rows, 1 + N_BENCH)),
        "self_unit_preevo_id":  _rand_ids((n_rows, 1 + N_BENCH, N_PREEVO)),
        "self_unit_tool_id":    _rand_ids((n_rows, 1 + N_BENCH, N_TOOLS)),
        "self_unit_energy_id":  _rand_ids((n_rows, 1 + N_BENCH, N_ENERGY_CARDS)),
        "opp_unit_top_id":      _rand_ids((n_rows, 1 + N_BENCH)),
        "opp_unit_preevo_id":   _rand_ids((n_rows, 1 + N_BENCH, N_PREEVO)),
        "opp_unit_tool_id":     _rand_ids((n_rows, 1 + N_BENCH, N_TOOLS)),
        "opp_unit_energy_id":   _rand_ids((n_rows, 1 + N_BENCH, N_ENERGY_CARDS)),
        # option ids
        "opt_src_pos":  rng.integers(-1, N_STATE_TOKENS, size=(n_rows, MAX_OPTIONS), dtype=np.int32),
        "opt_tgt_pos":  rng.integers(-1, N_STATE_TOKENS, size=(n_rows, MAX_OPTIONS), dtype=np.int32),
        "opt_src_card": _rand_ids((n_rows, MAX_OPTIONS)),
        "opt_tgt_card": _rand_ids((n_rows, MAX_OPTIONS)),
        "opt_verb":     rng.integers(0, 17, size=(n_rows, MAX_OPTIONS), dtype=np.int32),
        "opt_attack_id": _rand_ids((n_rows, MAX_OPTIONS), vocab=2048),
    }

    # --- float keys (float32) ---
    float_arrays: dict[str, np.ndarray] = {
        "cls_scalars":    _rand_float((n_rows, G)),
        "self_unit_attr": _rand_float((n_rows, 1 + N_BENCH, UNIT_ATTR)),
        "opp_unit_attr":  _rand_float((n_rows, 1 + N_BENCH, UNIT_ATTR)),
        "opt_attr":       _rand_float((n_rows, MAX_OPTIONS, OPT_ATTR_DIM)),
        "action_mask":    np.zeros((n_rows, N_ACTIONS), dtype=np.float32),
        "self_deck_flag": _rand_float((n_rows, DECK_SIZE)),
        "opp_deck_flag":  _rand_float((n_rows, DECK_SIZE)),
        "opp_hand_flag":  _rand_float((n_rows, MAX_HAND)),
        # binary masks (float32): 1 where present, 0 where padding
        "self_deck_mask":    _rand_float((n_rows, DECK_SIZE)),
        "opp_deck_mask":     _rand_float((n_rows, DECK_SIZE)),
        "self_prize_mask":   _rand_float((n_rows, N_PRIZE)),
        "opp_prize_mask":    _rand_float((n_rows, N_PRIZE)),
        "self_hand_mask":    _rand_float((n_rows, MAX_HAND)),
        "opp_hand_mask":     _rand_float((n_rows, MAX_HAND)),
        "self_discard_mask": _rand_float((n_rows, MAX_DISCARD)),
        "opp_discard_mask":  _rand_float((n_rows, MAX_DISCARD)),
        "stadium_mask":      _rand_float((n_rows, N_STADIUM)),
        "self_unit_mask":    _rand_float((n_rows, 1 + N_BENCH)),
        "opp_unit_mask":     _rand_float((n_rows, 1 + N_BENCH)),
    }

    # --- action_mask: ~20% legal, at least 1 legal per row ---
    am = float_arrays["action_mask"]
    legal_counts = rng.integers(1, max(2, MAX_OPTIONS // 5) + 1, size=n_rows)
    for i in range(n_rows):
        # random subset of option indices
        idx = rng.choice(MAX_OPTIONS, size=legal_counts[i], replace=False)
        am[i, idx] = 1.0
    # ensure at least one legal action (safety net; already satisfied by construction)

    # --- __labels__: pick a random legal option for each row ---
    for i in range(n_rows):
        legal = np.flatnonzero(am[i])
        if len(legal) == 0:
            # should not happen, but safety
            am[i, 0] = 1.0
            legal = np.array([0])
        labels[i] = int(rng.choice(legal))

    # --- save all arrays ---
    def _save(name: str, arr: np.ndarray) -> None:
        path = os.path.join(out_dir, f"{name}.npy")
        np.save(path, arr, allow_pickle=False)
        print(f"  {name:28s} {str(arr.shape):20s} {arr.dtype}")

    print(f"[make_synthetic] writing {n_rows} rows to {out_dir}")
    _save("__labels__", labels)
    _save("__is_attack__", is_attack)

    for name, arr in int_arrays.items():
        _save(name, arr)
    for name, arr in float_arrays.items():
        _save(name, arr)

    print(f"[make_synthetic] done — {len(int_arrays) + len(float_arrays) + 2} files written")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=1000, help="Number of synthetic observations")
    p.add_argument("--out", default="data/bc_data/synthetic_1k", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    a = p.parse_args()

    make_dataset(a.rows, a.out, a.seed)

    # --- verify: load like bc_train_mlx.py does ---
    print(f"\n[verify] loading {a.out} as trainer would...")
    d = {f[:-4]: np.load(os.path.join(a.out, f), mmap_mode="r")
         for f in sorted(os.listdir(a.out)) if f.endswith(".npy")}
    N = int(d["__labels__"].shape[0])
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]
    print(f"  N={N}  keys={len(keys)}")
    # basic sanity
    for k in keys:
        assert d[k].shape[0] == N, f"{k}: first dim {d[k].shape[0]} != {N}"
    legal = (d["action_mask"].sum(axis=1) > 0).all()
    assert legal, "some rows have no legal action"
    label_legal = (np.arange(N), d["__labels__"])
    assert (d["action_mask"][label_legal] > 0).all(), "some labels are illegal"
    print("[verify] PASSED")


if __name__ == "__main__":
    main()
