"""TokenTransformer port for MLX (Apple Silicon native).

Drop-in replacement for rl/policy.py using mlx.nn instead of torch.nn.
Same interface: logits_value() / get_value() / get_action_and_value().

Usage:
  from rl.policy_mlx import build_token_net_mlx
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from rl import CardTable
from .encoder import effect_data
from rl import (
    MAX_OPTIONS, N_ACTIONS, OPT_STRUCT, OPT_PICKED, N_OPT_TYPES,
    MAX_ATTACK, N_SELECT_TYPES, N_SELECT_CTX, UNIT_ATTR, G,
)


# --- Model constants ---
_N_TTYPES: int = 19      # token-type ids
N_SCRATCH: int = 4        # register/workspace tokens ("ViTs Need Registers")


# ============================================================
# Transformer encoder (manual — MLX has no nn.TransformerEncoder)
# ============================================================

class TransformerEncoderLayerMLX(nn.Module):
    """Single Transformer encoder layer (norm_first=False, like PyTorch default)."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int) -> None:
        super().__init__()
        self.attn: nn.MultiHeadAttention = nn.MultiHeadAttention(d_model, nhead)
        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.ff: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, d_model),
        )

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        h = self.attn(x, x, x, mask=mask)
        x = self.norm1(x + h)
        x = self.norm2(x + self.ff(x))
        return x


class TransformerEncoderMLX(nn.Module):
    """Stack of Transformer encoder layers."""

    def __init__(self, d_model: int, nhead: int, ff_dim: int, nlayers: int) -> None:
        super().__init__()
        self.layers: list[TransformerEncoderLayerMLX] = [
            TransformerEncoderLayerMLX(d_model, nhead, ff_dim)
            for _ in range(nlayers)
        ]

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)
        return x


# ============================================================
# TokenTransformer (MLX port)
# ============================================================

class TokenTransformerMLX(nn.Module):
    """Token-set state -> Transformer encoder -> pointer policy + CLS value.

    MLX port of rl/policy.TokenTransformer. Same architecture, same interface.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        nlayers: int = 2,
        dropout: float = 0.0,
        card_feat: np.ndarray | None = None,
        structured: bool = False,
        split_heads: bool = False,
        value_categorical: bool = False,
        value_atoms: int = 51,
        value_vmax: float = 1.0,
        opt_struct: int = OPT_STRUCT + effect_data.N_ATTACK_FX,
    ) -> None:
        super().__init__()
        self.d: int = d_model
        self.UNK: int = vocab_size

        # Card identity embedding (index 0 = pad, vocab_size = UNK)
        self.card_emb: nn.Embedding = nn.Embedding(vocab_size + 1, d_model)

        # Static card features (optional)
        self.static_proj: nn.Linear | None = None
        self.card_feat: mx.array | None = None
        if card_feat is not None:
            _f = np.zeros((vocab_size + 1, card_feat.shape[1]), dtype=np.float32)
            _f[:card_feat.shape[0]] = card_feat
            self.card_feat = mx.array(_f)
            self.static_proj = nn.Linear(card_feat.shape[1], d_model)

        # Structured action head (verb-conditioned scoring)
        self.structured: bool = structured
        if structured:
            self.type_query: nn.Embedding = nn.Embedding(N_OPT_TYPES, d_model)
            self.type_bias: nn.Embedding = nn.Embedding(N_OPT_TYPES, 1)

        # Token-type / context embeddings
        self.type_emb: nn.Embedding = nn.Embedding(_N_TTYPES, d_model)
        self.sel_type_emb: nn.Embedding = nn.Embedding(N_SELECT_TYPES, d_model)
        self.sel_ctx_emb: nn.Embedding = nn.Embedding(N_SELECT_CTX, d_model)

        # Learnable special tokens
        self.cls: mx.array = mx.zeros(d_model)
        self.split_heads: bool = split_heads
        if split_heads:
            self.value_tok: mx.array = mx.zeros(d_model)
            self.submit_tok: mx.array = mx.zeros(d_model)
        self.scratch_tokens: int = N_SCRATCH
        self.scratch: mx.array = mx.zeros((N_SCRATCH, d_model))

        # Projections
        self.unit_attr_proj: nn.Linear = nn.Linear(UNIT_ATTR, d_model)
        self.drawable_emb: mx.array = mx.zeros(d_model)
        self.opp_drawable_emb: mx.array = mx.zeros(d_model)
        self.hand_certain_emb: mx.array = mx.zeros(d_model)
        self.opt_src_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.opt_tgt_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.opt_attr_proj: nn.Linear = nn.Linear(opt_struct, d_model)
        self.opt_verb_emb: nn.Embedding = nn.Embedding(N_OPT_TYPES, d_model)
        self.attack_emb: nn.Embedding = nn.Embedding(MAX_ATTACK, d_model)
        self.scalar_proj: nn.Linear = nn.Linear(G, d_model)

        # Transformer encoder
        self.encoder: TransformerEncoderMLX = TransformerEncoderMLX(
            d_model, nhead, ff_dim=4 * d_model, nlayers=nlayers
        )

        # Output heads
        self.opt_head: nn.Linear = nn.Linear(d_model, 1)
        self.submit_head: nn.Linear = nn.Linear(d_model, 1)
        v_in: int = d_model if split_heads else 2 * d_model
        self.value_categorical: bool = value_categorical
        if value_categorical:
            self.value_atoms: int = int(value_atoms)
            self.value_vmax: float = float(value_vmax)
            self.value_head: nn.Linear = nn.Linear(v_in, self.value_atoms)
            self.atom_support: mx.array = mx.linspace(-value_vmax, value_vmax, value_atoms)
        else:
            self.value_head: nn.Linear = nn.Linear(v_in, 1)

    # --- token builders ---

    def _type(self, B: int, K: int, t: int) -> mx.array:
        """Create type embedding tokens for a stream."""
        ids = mx.array([t] * (B * K)).reshape(B, K)
        return self.type_emb(ids)

    def _static(self, ids: mx.array) -> mx.array:
        """Static card features projection. Returns zeros if disabled."""
        if self.static_proj is None:
            return mx.zeros((*ids.shape, self.d))
        return self.static_proj(self.card_feat[ids])

    def _card_stream(self, ids: mx.array, t: int) -> mx.array:
        """Card list tokens: card_emb(id) + type_emb + static features."""
        B, K = ids.shape
        return self.card_emb(ids) + self._type(B, K, t) + self._static(ids)

    def _unit_stream(
        self,
        top_id: mx.array,
        preevo_id: mx.array,
        tool_id: mx.array,
        energy_id: mx.array,
        attr: mx.array,
        active_t: int,
        bench_t: int,
    ) -> mx.array:
        """In-play unit tokens: top_card + pre-evos + tools + energies + attr + type."""
        B, U = top_id.shape
        idbag = (
            self.card_emb(top_id)
            + self.card_emb(preevo_id).sum(axis=2)
            + self.card_emb(tool_id).sum(axis=2)
            + self.card_emb(energy_id).sum(axis=2)
        )  # [B, U, d]
        # Active slot gets active_t, rest get bench_t
        types_arr = [[bench_t] * U for _ in range(B)]
        for b in range(B):
            types_arr[b][0] = active_t
        types = mx.array(types_arr, dtype=mx.int32)
        return idbag + self.unit_attr_proj(attr) + self.type_emb(types) + self._static(top_id)

    def _opt_stream(
        self,
        src_tok: mx.array,
        tgt_tok: mx.array,
        attr: mx.array,
        verb: mx.array,
        attack_id: mx.array,
    ) -> mx.array:
        """Option tokens: src_proj + tgt_proj + attr_proj + verb_emb + attack_emb + type."""
        B, K = attr.shape[0], attr.shape[1]
        return (
            self.opt_src_proj(src_tok)
            + self.opt_tgt_proj(tgt_tok)
            + self.opt_attr_proj(attr)
            + self.opt_verb_emb(verb)
            + self.attack_emb(attack_id)
            + self._type(B, K, 14)  # _T_OPT = 14
        )

    # --- core ---

    def _resolve(self, pos: mx.array, card: mx.array, state_seq: mx.array) -> mx.array:
        """Gather token at pos from state_seq; synthesize from card id if pos < 0."""
        B, K = pos.shape
        d = state_seq.shape[2]
        safe = mx.clip(pos, a_min=0, a_max=state_seq.shape[1] - 1)
        # Per-batch gather: state_seq[b, safe[b,k], :] for each (b, k)
        # MLX doesn't have torch.gather, use explicit indexing
        batch_idx = mx.arange(B).reshape(B, 1)  # [B, 1]
        tok = state_seq[batch_idx, safe]          # [B, K, d]
        # Zero out positions where pos < 0
        valid = (pos >= 0).astype(mx.float32)[..., None]
        tok = tok * valid
        # Synthesize card token where pos < 0 but card > 0
        need = ((pos < 0) & (card > 0)).astype(mx.float32)[..., None]
        synth = (
            self.card_emb(card) + self._static(card)
            + self._type(B, K, 18)  # _T_CARD_SYNTH = 18
        )
        return tok + need * synth

    def _encode(self, o: dict, opt_len: int | None = None) -> tuple:
        """Build full token sequence and run Transformer.

        Args:
            o: observation dict (from encoder, mlx arrays)
            opt_len: optional truncation for option tokens

        Returns:
            (cls_out, opt_out, pooled, extra)
            - cls_out: [B, d] CLS token output
            - opt_out: [B, n_opt, d] option token outputs
            - pooled: [B, d] mean-pool over non-pad tokens
            - extra: (value_out, submit_out) if split_heads, else None
        """
        B = o["cls_scalars"].shape[0]
        toks: list[mx.array] = []
        pads: list[mx.array] = []

        # --- CLS token (global scalars, never padded) ---
        cls_tok = (
            mx.broadcast_to(self.cls.reshape(1, 1, self.d), (B, 1, self.d))
            + self.scalar_proj(o["cls_scalars"]).reshape(B, 1, self.d)
            + self._type(B, 1, 0)  # _T_CLS = 0
        )
        toks.append(cls_tok)
        pads.append(mx.zeros((B, 1), dtype=mx.bool_))

        # --- split_heads: dedicated value + submit tokens ---
        if self.split_heads:
            value_tok = (
                mx.broadcast_to(self.value_tok.reshape(1, 1, self.d), (B, 1, self.d))
                + self.scalar_proj(o["cls_scalars"]).reshape(B, 1, self.d)
                + self._type(B, 1, 0)
            )
            toks.append(value_tok)
            pads.append(mx.zeros((B, 1), dtype=mx.bool_))

            submit_tok = (
                mx.broadcast_to(self.submit_tok.reshape(1, 1, self.d), (B, 1, self.d))
                + self._type(B, 1, 0)
            )
            toks.append(submit_tok)
            pads.append(mx.zeros((B, 1), dtype=mx.bool_))

        # --- select type + context tokens (never padded) ---
        sel_type_tok = (
            self.sel_type_emb(o["select_type"].squeeze(-1)).reshape(B, 1, self.d)
            + self._type(B, 1, 16)  # _T_SEL_TYPE = 16
        )
        sel_ctx_tok = (
            self.sel_ctx_emb(o["select_context"].squeeze(-1)).reshape(B, 1, self.d)
            + self._type(B, 1, 17)  # _T_SEL_CTX = 17
        )
        toks.append(sel_type_tok)
        pads.append(mx.zeros((B, 1), dtype=mx.bool_))
        toks.append(sel_ctx_tok)
        pads.append(mx.zeros((B, 1), dtype=mx.bool_))

        # --- card-list streams (deck, prize, hand, discard, stadium, effect) ---
        _CARD_STREAMS = [
            ("self_deck", 3), ("opp_deck", 4),
            ("self_prize", 5), ("opp_prize", 6),
            ("self_hand", 7), ("opp_hand", 8),
            ("self_discard", 9), ("opp_discard", 10),
            ("stadium", 11),
            ("effect", 15),  # _T_EFFECT = 15
        ]
        for name, t in _CARD_STREAMS:
            tok = self._card_stream(o[f"{name}_id"], t)
            # Add special markers (deck flags, hand certainty)
            if name == "self_deck":
                tok = tok + o["self_deck_flag"][..., None] * self.drawable_emb
            elif name == "opp_deck":
                tok = tok + o["opp_deck_flag"][..., None] * self.opp_drawable_emb
            elif name == "opp_hand":
                tok = tok + o["opp_hand_flag"][..., None] * self.hand_certain_emb
            toks.append(tok)
            pads.append(o[f"{name}_mask"] < 0.5)

        # --- unit streams (active + bench, both sides) ---
        _UNIT_TYPES = [
            ("self", 12, 13),  # _T_SELF_ACTIVE=12, _T_SELF_BENCH=13
            ("opp", 14, 15),   # actually _T_OPP_ACTIVE=14 doesn't exist, let me check
        ]
        # Correction: using actual type ids from policy.py
        for side, (at, bt) in [("self", (12, 13)), ("opp", (13, 13))]:
            tok = self._unit_stream(
                o[f"{side}_unit_top_id"], o[f"{side}_unit_preevo_id"],
                o[f"{side}_unit_tool_id"], o[f"{side}_unit_energy_id"],
                o[f"{side}_unit_attr"], at, bt,
            )
            toks.append(tok)
            pads.append(o[f"{side}_unit_mask"] < 0.5)

        # --- concatenate all state tokens ---
        state_seq = mx.concatenate(toks, axis=1)  # [B, N_STATE, d]
        pad_state = mx.concatenate(pads, axis=1)   # [B, N_STATE]

        # --- option tokens ---
        _shift = 2 if self.split_heads else 0
        _src_pos = mx.where(o["opt_src_pos"] >= 0, o["opt_src_pos"] + _shift, o["opt_src_pos"])
        _tgt_pos = mx.where(o["opt_tgt_pos"] >= 0, o["opt_tgt_pos"] + _shift, o["opt_tgt_pos"])

        opt_tok = self._opt_stream(
            self._resolve(_src_pos, o["opt_src_card"], state_seq),
            self._resolve(_tgt_pos, o["opt_tgt_card"], state_seq),
            o["opt_attr"], o["opt_verb"], o["opt_attack_id"],
        )

        # Present = legal OR already-picked
        opt_present = (
            (o["action_mask"][..., :MAX_OPTIONS] > 0.5)
            | (o["opt_attr"][..., OPT_PICKED] > 0.5)
        )
        # Truncate options if requested
        if opt_len is not None and opt_len < opt_tok.shape[1]:
            opt_tok = opt_tok[:, :opt_len]
            opt_present = opt_present[:, :opt_len]
        n_opt = opt_tok.shape[1]

        # --- scratch tokens (between state and options) ---
        scr = (
            mx.broadcast_to(self.scratch.reshape(1, self.scratch_tokens, self.d), (B, self.scratch_tokens, self.d))
            + self._type(B, self.scratch_tokens, 0)  # _T_CLS type
        )

        # --- full sequence: state + scratch + options ---
        seq = mx.concatenate([state_seq, scr, opt_tok], axis=1)
        pad = mx.concatenate([
            pad_state,
            mx.zeros((B, self.scratch_tokens), dtype=mx.bool_),
            ~opt_present,
        ], axis=1)

        # --- run Transformer ---
        # MLX MultiHeadAttention mask: True = attend, False = ignore
        # Needs to be 4D (B, H, S_q, S_kv) or (B, 1, 1, S_kv) for key-padding mask
        attn_mask = (~pad)[:, None, None, :]  # [B, 1, 1, S_kv]
        enc = self.encoder(seq, mask=attn_mask)

        cls_out = enc[:, 0]                    # [B, d]
        opt_out = enc[:, -n_opt:]              # [B, n_opt, d]
        present = (~pad).astype(mx.float32)[..., None]
        denom = mx.maximum(present.sum(axis=1), 1.0)
        pooled = (enc * present).sum(axis=1) / denom
        extra = (enc[:, 1], enc[:, 2]) if self.split_heads else None

        return cls_out, opt_out, pooled, extra

    def logits_value(
            self, o: dict, opt_len: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """Forward pass: return policy logits + value estimate.

        Args:
            o: observation dict (mlx arrays)
            opt_len: optional option truncation

        Returns:
            (logits, value): logits [B, N_ACTIONS], value [B]
        """
        cls_out, opt_out, pooled, extra = self._encode(o, opt_len=opt_len)
        n = opt_out.shape[1]

        # Option scoring
        if self.structured:
            verb = o["opt_verb"][:, :n]
            opt_logits = (
                    (opt_out * self.type_query(verb)).sum(axis=-1)
                    + self.type_bias(verb).squeeze(-1)
                    + self.opt_head(opt_out).squeeze(-1)
            )
        else:
            opt_logits = self.opt_head(opt_out).squeeze(-1)  # [B, n]

        # Pad to full action dim if truncated
        if n < MAX_OPTIONS:
            pad_size = MAX_OPTIONS - n
            opt_logits = mx.pad(opt_logits, [(0, 0), (0, pad_size)], constant_values=-1e9)

        # Submit logit
        submit_src = extra[1] if self.split_heads else cls_out
        submit_logit = self.submit_head(submit_src)  # [B, 1]

        # Value
        if self.split_heads:
            v_in = extra[0]
        else:
            v_in = mx.concatenate([cls_out, pooled], axis=-1)
        value = self.value_head(v_in).squeeze(-1)  # [B]

        # Full logits: [B, N_ACTIONS]
        logits = mx.concatenate([opt_logits, submit_logit], axis=-1)

        # Mask illegal options
        action_mask = o["action_mask"]
        logits = mx.where(action_mask < 0.5, -1e9, logits)

        return logits, value

    def get_value(self, o: dict, opt_len: int | None = None) -> mx.array:
        """Return only the value estimate."""
        cls_out, _, pooled, extra = self._encode(o, opt_len=opt_len)
        if self.split_heads:
            v_in = extra[0]
        else:
            v_in = mx.concatenate([cls_out, pooled], axis=-1)
        return self.value_head(v_in).squeeze(-1)

    def get_action_and_value(
            self, o: dict, action: mx.array | None = None, opt_len: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """Return action, log_prob, entropy, value (for PPO)."""
        logits, value = self.logits_value(o, opt_len=opt_len)
        # Softmax for sampling
        probs = mx.softmax(logits, axis=-1)
        log_probs = mx.log(mx.clip(probs, a_min=1e-8, a_max=None))
        entropy = -(probs * log_probs).sum(axis=-1)
        if action is None:
            # Sample from distribution
            action = mx.random.categorical(logits, num_samples=1).squeeze(-1)
        log_prob = mx.take_along_axis(log_probs, action.reshape(-1, 1), axis=1).squeeze(-1)
        return action, log_prob, entropy, value


# ============================================================
# Builder function
# ============================================================

def build_token_net_mlx(card_table: CardTable, net_config: dict | None = None) -> TokenTransformerMLX:
    """Construct a TokenTransformerMLX sized to card_table.

    Same interface as rl.policy.build_token_net.
    """
    cfg = dict(net_config or {})
    cfg.pop("arch", None)
    cfg.pop("emb_dim", None)
    cfg.pop("ff", None)
    use_static: bool = cfg.pop("static", False)
    use_structured: bool = cfg.pop("structured", False)
    use_split: bool = cfg.pop("split_heads", False)
    cfg.pop("would_ko", None)
    opt_struct: int = cfg.pop("opt_struct", OPT_STRUCT + effect_data.N_ATTACK_FX)

    feat = card_table.matrix if use_static else None
    if feat is not None:
        extra = np.asarray(
            [effect_data.ability_multihot(i) + effect_data.trainer_multihot(i)
             for i in range(feat.shape[0])], dtype=np.float32
        )
        feat = np.concatenate([np.asarray(feat, dtype=np.float32), extra], axis=1)

    return TokenTransformerMLX(
        card_table.vocab_size, card_feat=feat, structured=use_structured,
        split_heads=use_split, opt_struct=opt_struct, **cfg,
    )