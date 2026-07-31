"""Canonical token-type IDs for the TokenTransformer.

Single source of truth shared by encoder, MLX policy, trainer, loader, and agent.
These constants match the PyTorch reference (reference/mikaelzinho-pytorch/rl/policy.py)
exactly. Do NOT duplicate these values elsewhere -- import from here.

Token-type IDs assign a semantic class to every position in the Transformer input
sequence so the type embedding can distinguish entity kinds (deck cards, bench units,
options, etc.) even when their positional/card features look similar.
"""

# Architecture and schema versioning — recorded in every checkpoint.
ARCH_VERSION = "1.0.0"
TOKEN_SCHEMA_VERSION = "1.0.0"

# ---- Token-type ID constants (PyTorch reference ground truth) ----
T_CLS = 0           # summary / CLS token
T_SELF_DECK = 1     # our deck list cards
T_OPP_DECK = 2      # opponent deck list cards (inferred)
T_SELF_PRIZE = 3    # our face-down prize cards
T_OPP_PRIZE = 4     # opponent prize cards (inferred)
T_SELF_HAND = 5     # our hand cards
T_OPP_HAND = 6      # opponent hand cards (public knowledge)
T_SELF_DISC = 7     # our discard pile cards
T_OPP_DISC = 8      # opponent discard pile cards
T_STADIUM = 9       # stadium slot(s)
T_SELF_ACTIVE = 10  # our active Pokemon unit
T_SELF_BENCH = 11   # our bench Pokemon units
T_OPP_ACTIVE = 12   # opponent active Pokemon unit
T_OPP_BENCH = 13    # opponent bench Pokemon units
T_OPT = 14          # option / action tokens
T_EFFECT = 15       # source-effect tokens
T_SEL_TYPE = 16     # select-type signal token
T_SEL_CTX = 17      # select-context signal token

T_CARD_SYNTH = 18   # synthesized card token (attached energy/tool, deck-search pick)

N_TTYPES = 19       # total number of distinct token types (0..18 inclusive)
