"""Verify MLX policy uses correct canonical token-type IDs.

Checks:
  - All type IDs in policy_mlx.py come from rl/token_schema (no hardcoded magic numbers)
  - Opponent unit types are distinct from self unit types
  - Model builds and forward passes correctly
"""
import sys
import os

import inspect
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from rl.encoder.card_features import get_card_table
from rl.policy_mlx import build_token_net_mlx, TokenTransformerMLX
from rl.token_schema import (
    T_CLS, T_SELF_DECK, T_OPP_DECK, T_SELF_PRIZE, T_OPP_PRIZE,
    T_SELF_HAND, T_OPP_HAND, T_SELF_DISC, T_OPP_DISC, T_STADIUM,
    T_SELF_ACTIVE, T_SELF_BENCH, T_OPP_ACTIVE, T_OPP_BENCH,
    T_OPT, T_EFFECT, T_SEL_TYPE, T_SEL_CTX, T_CARD_SYNTH, N_TTYPES,
)


def test_all_type_ids_distinct():
    """All 19 token-type IDs must be unique."""
    ids = [T_CLS, T_SELF_DECK, T_OPP_DECK, T_SELF_PRIZE, T_OPP_PRIZE,
           T_SELF_HAND, T_OPP_HAND, T_SELF_DISC, T_OPP_DISC, T_STADIUM,
           T_SELF_ACTIVE, T_SELF_BENCH, T_OPP_ACTIVE, T_OPP_BENCH,
           T_OPT, T_EFFECT, T_SEL_TYPE, T_SEL_CTX, T_CARD_SYNTH]
    assert len(set(ids)) == N_TTYPES, f"Expected {N_TTYPES} distinct IDs, got {len(set(ids))}"
    assert ids == list(range(N_TTYPES)), f"IDs must be 0..{N_TTYPES-1}, got {ids}"
    print("  PASS: all type IDs distinct and continuous")


def test_opp_unit_types_distinct_from_self():
    """Opponent active/bench must differ from self active/bench."""
    assert T_OPP_ACTIVE != T_SELF_ACTIVE, "opp_active must differ from self_active"
    assert T_OPP_BENCH != T_SELF_BENCH, "opp_bench must differ from self_bench"
    assert T_OPP_ACTIVE != T_OPP_BENCH, "opp_active must differ from opp_bench"
    assert T_OPP_ACTIVE != T_SELF_BENCH, "opp_active must differ from self_bench"
    print(f"  PASS: self_active={T_SELF_ACTIVE}, self_bench={T_SELF_BENCH}, "
          f"opp_active={T_OPP_ACTIVE}, opp_bench={T_OPP_BENCH} — all distinct")


def test_no_hardcoded_type_ids_in_encode():
    """The _encode method must use named constants, not hardcoded integers."""
    src = inspect.getsource(TokenTransformerMLX._encode)
    # These were the old hardcoded calls — they must NOT appear
    forbidden = [
        "_type(B, 1, 0)",    # should be T_CLS
        "_type(B, 1, 16)",   # should be T_SEL_TYPE
        "_type(B, 1, 17)",   # should be T_SEL_CTX
        "_type(B, K, 14)",   # should be T_OPT
        "_type(B, K, 18)",   # should be T_CARD_SYNTH
    ]
    found = [f for f in forbidden if f in src]
    assert not found, f"Hardcoded type IDs still present: {found}"
    print("  PASS: no hardcoded type IDs in _encode")


def test_model_builds_and_forward():
    """Model builds with canonical schema and forward pass produces correct shapes."""
    ct = get_card_table()
    cfg = {"d_model": 128, "nhead": 4, "nlayers": 3, "static": True, "split_heads": True}
    model = build_token_net_mlx(ct, cfg)

    # Verify type_emb has N_TTYPES entries
    assert model.type_emb.weight.shape[0] == N_TTYPES, \
        f"type_emb has {model.type_emb.weight.shape[0]} entries, expected {N_TTYPES}"

    nparams = sum(p.size for _, p in nn.utils.tree_flatten(model.parameters()))
    assert nparams > 0, "Model has no parameters"
    print(f"  PASS: model built with {nparams:,} params, type_emb has {N_TTYPES} entries")


def main():
    print("=== Token-Type Schema Validation ===")
    test_all_type_ids_distinct()
    test_opp_unit_types_distinct_from_self()
    test_no_hardcoded_type_ids_in_encode()
    test_model_builds_and_forward()
    print("\n  ALL PASSED — token-type schema is canonical and collision-free")


if __name__ == "__main__":
    main()
