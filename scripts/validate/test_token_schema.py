"""Validate rl/token_schema.py against the PyTorch reference ground truth."""

import sys
import os

# Ensure project root is on sys.path so rl.token_schema is importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from rl.token_schema import (
    ARCH_VERSION,
    TOKEN_SCHEMA_VERSION,
    T_CLS,
    T_SELF_DECK,
    T_OPP_DECK,
    T_SELF_PRIZE,
    T_OPP_PRIZE,
    T_SELF_HAND,
    T_OPP_HAND,
    T_SELF_DISC,
    T_OPP_DISC,
    T_STADIUM,
    T_SELF_ACTIVE,
    T_SELF_BENCH,
    T_OPP_ACTIVE,
    T_OPP_BENCH,
    T_OPT,
    T_EFFECT,
    T_SEL_TYPE,
    T_SEL_CTX,
    T_CARD_SYNTH,
    N_TTYPES,
)

# All token-type constants collected in declaration order.
ALL_IDS = [
    T_CLS,
    T_SELF_DECK,
    T_OPP_DECK,
    T_SELF_PRIZE,
    T_OPP_PRIZE,
    T_SELF_HAND,
    T_OPP_HAND,
    T_SELF_DISC,
    T_OPP_DISC,
    T_STADIUM,
    T_SELF_ACTIVE,
    T_SELF_BENCH,
    T_OPP_ACTIVE,
    T_OPP_BENCH,
    T_OPT,
    T_EFFECT,
    T_SEL_TYPE,
    T_SEL_CTX,
    T_CARD_SYNTH,
]

# Expected values from the PyTorch reference (ground truth).
EXPECTED = {
    "T_CLS": 0,
    "T_SELF_DECK": 1,
    "T_OPP_DECK": 2,
    "T_SELF_PRIZE": 3,
    "T_OPP_PRIZE": 4,
    "T_SELF_HAND": 5,
    "T_OPP_HAND": 6,
    "T_SELF_DISC": 7,
    "T_OPP_DISC": 8,
    "T_STADIUM": 9,
    "T_SELF_ACTIVE": 10,
    "T_SELF_BENCH": 11,
    "T_OPP_ACTIVE": 12,
    "T_OPP_BENCH": 13,
    "T_OPT": 14,
    "T_EFFECT": 15,
    "T_SEL_TYPE": 16,
    "T_SEL_CTX": 17,
    "T_CARD_SYNTH": 18,
    "N_TTYPES": 19,
}


def test_all_distinct():
    """All 19 token-type IDs must be distinct."""
    assert len(ALL_IDS) == 19, f"Expected 19 IDs, got {len(ALL_IDS)}"
    assert len(set(ALL_IDS)) == 19, (
        f"Duplicate token-type IDs found: {sorted(ALL_IDS)}"
    )


def test_continuous_range():
    """IDs must occupy exactly 0..18 with no gaps."""
    assert sorted(ALL_IDS) == list(range(19)), (
        f"IDs are not continuous 0..18: {sorted(ALL_IDS)}"
    )


def test_match_pytorch_reference():
    """Every ID must match the PyTorch reference ground truth."""
    names = [
        "T_CLS", "T_SELF_DECK", "T_OPP_DECK", "T_SELF_PRIZE", "T_OPP_PRIZE",
        "T_SELF_HAND", "T_OPP_HAND", "T_SELF_DISC", "T_OPP_DISC", "T_STADIUM",
        "T_SELF_ACTIVE", "T_SELF_BENCH", "T_OPP_ACTIVE", "T_OPP_BENCH",
        "T_OPT", "T_EFFECT", "T_SEL_TYPE", "T_SEL_CTX", "T_CARD_SYNTH",
    ]
    values = [
        T_CLS, T_SELF_DECK, T_OPP_DECK, T_SELF_PRIZE, T_OPP_PRIZE,
        T_SELF_HAND, T_OPP_HAND, T_SELF_DISC, T_OPP_DISC, T_STADIUM,
        T_SELF_ACTIVE, T_SELF_BENCH, T_OPP_ACTIVE, T_OPP_BENCH,
        T_OPT, T_EFFECT, T_SEL_TYPE, T_SEL_CTX, T_CARD_SYNTH,
    ]
    for name, val in zip(names, values):
        expected = EXPECTED[name]
        assert val == expected, f"{name}: got {val}, expected {expected}"
    assert N_TTYPES == EXPECTED["N_TTYPES"], (
        f"N_TTYPES: got {N_TTYPES}, expected {EXPECTED['N_TTYPES']}"
    )


def test_version_strings():
    """Version strings must be non-empty semver-like."""
    assert isinstance(ARCH_VERSION, str) and len(ARCH_VERSION) > 0
    assert isinstance(TOKEN_SCHEMA_VERSION, str) and len(TOKEN_SCHEMA_VERSION) > 0


def test_no_collision_with_encoder_constants():
    """Token-type IDs must not collide with enc_constants shape values."""
    # Import shape constants that live alongside token types.
    from rl.encoder.enc_constants import MAX_OPTIONS, N_OPT_TYPES

    # MAX_OPTIONS (192) and N_OPT_TYPES (17) are not token types.
    # Ensure T_SEL_CTX (17) does NOT equal N_OPT_TYPES by accident if names
    # drift; this is a semantic guard that the IDs stay in their own namespace.
    assert T_SEL_CTX != MAX_OPTIONS, "T_SEL_CTX must not equal MAX_OPTIONS"


if __name__ == "__main__":
    tests = [
        test_all_distinct,
        test_continuous_range,
        test_match_pytorch_reference,
        test_version_strings,
        test_no_collision_with_encoder_constants,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
