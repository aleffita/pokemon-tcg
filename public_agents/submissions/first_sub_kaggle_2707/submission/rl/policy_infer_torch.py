"""PyTorch inference mirror for the current MLX checkpoint.

This module is intentionally separate from ``rl/policy.py`` and
``rl/policy_mlx.py``.  It uses the checkpoint's embedded ``arch_config`` as the
inference contract; no training JSON is needed at runtime.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.encoder import effect_data
from rl.encoder.card_features import CardTable
from rl.encoder.encoding import MAX_OPTIONS, OPT_PICKED
from rl.policy import TokenTransformer, build_token_net
from rl.token_schema import ARCH_VERSION, TOKEN_SCHEMA_VERSION


def _flatten_checkpoint(tree: Any, prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten the MLX pickle tree into parameter-name -> numpy arrays."""
    if isinstance(tree, dict):
        out: dict[str, np.ndarray] = {}
        for key, value in tree.items():
            name = f"{prefix}.{key}" if prefix else key
            out.update(_flatten_checkpoint(value, name))
        return out
    if isinstance(tree, (list, tuple)):
        out = {}
        for i, value in enumerate(tree):
            out.update(_flatten_checkpoint(value, f"{prefix}.{i}"))
        return out
    return {prefix: np.asarray(tree)}


def checkpoint_arch_config(state: dict[str, Any]) -> dict[str, Any]:
    cfg = state.get("arch_config")
    if not isinstance(cfg, dict):
        raise ValueError("MLX checkpoint has no arch_config; refusing ambiguous inference load")
    for key in ("arch_version", "token_schema_version"):
        if key not in cfg:
            raise ValueError(f"MLX checkpoint arch_config is missing {key}")
    if cfg["arch_version"] != ARCH_VERSION:
        raise ValueError(f"unsupported architecture version {cfg['arch_version']!r}; expected {ARCH_VERSION!r}")
    if cfg["token_schema_version"] != TOKEN_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported token schema {cfg['token_schema_version']!r}; expected {TOKEN_SCHEMA_VERSION!r}"
        )
    if int(cfg.get("max_options", MAX_OPTIONS)) != MAX_OPTIONS:
        raise ValueError(f"checkpoint max_options={cfg.get('max_options')} != encoder max_options={MAX_OPTIONS}")
    if not cfg.get("has_learned_init", False):
        raise ValueError("checkpoint has no learned_init; current recurrent inference contract requires it")
    return dict(cfg)


class TokenTransformerTorchInference(TokenTransformer):
    """Dynamic-scratch, memory-aware PyTorch mirror of ``TokenTransformerMLX``."""

    def __init__(self, card_table: CardTable, cfg: dict[str, Any]) -> None:
        d_model = int(cfg["d_model"])
        ff_dim = int(cfg.get("ff_dim", 4 * d_model))
        if ff_dim != 4 * d_model:
            raise ValueError(
                f"PyTorch inference mirror requires ff_dim=4*d_model, got {ff_dim} and {d_model}"
            )
        net_cfg = {
            "d_model": d_model,
            "nhead": int(cfg["nhead"]),
            "nlayers": int(cfg["nlayers"]),
            "static": bool(cfg["static"]),
            "split_heads": bool(cfg["split_heads"]),
            "structured": bool(cfg["structured"]),
            "value_categorical": bool(cfg.get("value_categorical", False)),
            "value_atoms": int(cfg.get("value_atoms", 51)),
            "value_vmax": float(cfg.get("value_vmax", 1.0)),
        }
        # ``build_token_net`` constructs the exact static feature table; the
        # parent constructor supplies the structured head and categorical value
        # buffers. Architecture values all come from ``arch_config`` above, and
        # the recurrent workspace is resized to the checkpoint below.
        base = build_token_net(card_table, net_cfg)
        super().__init__(
            card_table.vocab_size,
            d_model=base.d,
            nhead=int(cfg["nhead"]),
            nlayers=int(cfg["nlayers"]),
            card_feat=base.card_feat[:card_table.vocab_size].numpy() if base.card_feat is not None else None,
            structured=bool(cfg["structured"]),
            split_heads=bool(cfg["split_heads"]),
            value_categorical=bool(cfg.get("value_categorical", False)),
            value_atoms=int(cfg.get("value_atoms", 51)),
            value_vmax=float(cfg.get("value_vmax", 1.0)),
            opt_struct=2 + 4 + 5 + 3 + 1 + effect_data.N_ATTACK_FX,
        )
        # ``super`` rebuilt the same architecture. Keep the configured tables
        # and replace only the recurrent workspace dimensions.
        n_scratch = int(cfg["scratch_registers"])
        self.scratch_tokens = n_scratch
        self.scratch = torch.nn.Parameter(torch.zeros(n_scratch, self.d))
        self.learned_init = torch.nn.Parameter(torch.zeros(n_scratch, self.d))
        self._card_table = card_table

    def _card_emb(self, ids: torch.Tensor) -> torch.Tensor:
        emb = self.card_emb(ids)
        return emb * (ids != 0).unsqueeze(-1).to(emb.dtype)

    def _static(self, ids: torch.Tensor) -> torch.Tensor:
        if self.static_proj is None:
            return torch.zeros(*ids.shape, self.d, dtype=self.card_emb.weight.dtype, device=ids.device)
        return self.static_proj(self.card_feat[ids])

    def _card_stream(self, ids, t, device):
        b, k = ids.shape
        return self._card_emb(ids) + self._type(b, k, t, device) + self._static(ids)

    def _unit_stream(self, top_id, preevo_id, tool_id, energy_id, attr, active_t, bench_t, device):
        b, u = top_id.shape
        idbag = (self._card_emb(top_id) + self._card_emb(preevo_id).sum(2)
                 + self._card_emb(tool_id).sum(2) + self._card_emb(energy_id).sum(2))
        types = torch.full((b, u), bench_t, dtype=torch.long, device=device)
        types[:, 0] = active_t
        return idbag + self.unit_attr_proj(attr) + self.type_emb(types) + self._static(top_id)

    def _resolve(self, pos, card, state_seq):
        b, k = pos.shape
        safe = pos.clamp(min=0, max=state_seq.shape[1] - 1)
        tok = torch.gather(state_seq, 1, safe.unsqueeze(-1).expand(-1, -1, self.d))
        tok = tok * (pos >= 0).unsqueeze(-1).to(tok.dtype)
        need = ((pos < 0) & (card > 0)).unsqueeze(-1).to(tok.dtype)
        synth = self._card_emb(card) + self._static(card) + self._type(b, k, 18, card.device)
        return tok + need * synth

    @staticmethod
    def _bucket(m: int) -> int:
        for bucket in (32, 64, 128, MAX_OPTIONS):
            if m <= bucket:
                return bucket
        return MAX_OPTIONS

    def _encode(self, o, opt_len=None, memory_in=None):
        dev = o["cls_scalars"].device
        b = o["cls_scalars"].shape[0]
        toks, pads = [], []
        def nopad(): pads.append(torch.zeros(b, 1, dtype=torch.bool, device=dev))

        toks.append(self.cls.expand(b, 1, self.d) + self.scalar_proj(o["cls_scalars"]).unsqueeze(1)
                    + self._type(b, 1, 0, dev)); nopad()
        if self.split_heads:
            toks.append(self.value_tok.expand(b, 1, self.d) + self.scalar_proj(o["cls_scalars"]).unsqueeze(1)
                        + self._type(b, 1, 0, dev)); nopad()
            toks.append(self.submit_tok.expand(b, 1, self.d) + self._type(b, 1, 0, dev)); nopad()
        toks.append(self.sel_type_emb(o["select_type"].squeeze(-1)).unsqueeze(1) + self._type(b, 1, 16, dev)); nopad()
        toks.append(self.sel_ctx_emb(o["select_context"].squeeze(-1)).unsqueeze(1) + self._type(b, 1, 17, dev)); nopad()

        streams = (("self_deck", 1), ("opp_deck", 2), ("self_prize", 3), ("opp_prize", 4),
                   ("self_hand", 5), ("opp_hand", 6), ("self_discard", 7), ("opp_discard", 8),
                   ("stadium", 9), ("effect", 15))
        for name, typ in streams:
            tok = self._card_stream(o[f"{name}_id"], typ, dev)
            if name == "self_deck": tok = tok + o["self_deck_flag"].unsqueeze(-1) * self.drawable_emb
            if name == "opp_deck": tok = tok + o["opp_deck_flag"].unsqueeze(-1) * self.opp_drawable_emb
            if name == "opp_hand": tok = tok + o["opp_hand_flag"].unsqueeze(-1) * self.hand_certain_emb
            toks.append(tok); pads.append(o[f"{name}_mask"] < 0.5)
        for side, active, bench in (("self", 10, 11), ("opp", 12, 13)):
            toks.append(self._unit_stream(o[f"{side}_unit_top_id"], o[f"{side}_unit_preevo_id"],
                                          o[f"{side}_unit_tool_id"], o[f"{side}_unit_energy_id"],
                                          o[f"{side}_unit_attr"], active, bench, dev))
            pads.append(o[f"{side}_unit_mask"] < 0.5)

        state_seq = torch.cat(toks, 1)
        pad_state = torch.cat(pads, 1)
        n_full = state_seq.shape[1]
        keep = (~pad_state).any(0)
        keep[:3] = True  # CLS/select type/select context are always present
        if self.split_heads:
            keep[3:5] = True  # dedicated value/submit precede select type/context
        keep_idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        # Keep the same exact source/target address space as the MLX encoder:
        # absolute option positions are remapped only after state compaction.
        if keep_idx.numel() == n_full:
            remap = torch.arange(n_full, dtype=torch.long, device=dev)
        else:
            remap = torch.full((n_full,), -1, dtype=torch.long, device=dev)
            remap[keep_idx] = torch.arange(keep_idx.numel(), device=dev)
        state_seq = state_seq.index_select(1, keep_idx)
        pad_state = pad_state.index_select(1, keep_idx)
        shift = 2 if self.split_heads else 0
        src = o["opt_src_pos"]
        tgt = o["opt_tgt_pos"]
        src = torch.where(src >= 0, src + shift, src)
        tgt = torch.where(tgt >= 0, tgt + shift, tgt)
        src = torch.where(src >= 0, remap[src.clamp(0, n_full - 1)], src)
        tgt = torch.where(tgt >= 0, remap[tgt.clamp(0, n_full - 1)], tgt)
        opt_tok = self._opt_stream(self._resolve(src, o["opt_src_card"], state_seq),
                                   self._resolve(tgt, o["opt_tgt_card"], state_seq),
                                   o["opt_attr"], o["opt_verb"], o["opt_attack_id"], dev)
        present = ((o["action_mask"][..., :MAX_OPTIONS] > 0.5)
                   | (o["opt_attr"][..., OPT_PICKED] > 0.5))
        if opt_len is None:
            opt_len = self._bucket(int(present.sum(1).max().item()))
        if opt_len < opt_tok.shape[1]:
            opt_tok, present = opt_tok[:, :opt_len], present[:, :opt_len]
        if memory_in is None:
            mem = self.learned_init.unsqueeze(0).expand(b, -1, -1)
        else:
            mem = memory_in.reshape(-1, self.scratch_tokens, self.d).expand(b, -1, -1)
        scr = mem + self._type(b, self.scratch_tokens, 0, dev)
        seq = torch.cat((state_seq, scr, opt_tok), 1)
        pad = torch.cat((pad_state, torch.zeros(b, self.scratch_tokens, dtype=torch.bool, device=dev), ~present), 1)
        enc = self.encoder(seq, src_key_padding_mask=pad)
        cls_out, opt_out = enc[:, 0], enc[:, -opt_tok.shape[1]:]
        present_all = (~pad).unsqueeze(-1).to(enc.dtype)
        pooled = (enc * present_all).sum(1) / present_all.sum(1).clamp_min(1)
        extra = (enc[:, 1], enc[:, 2]) if self.split_heads else None
        scr_out = enc[:, state_seq.shape[1]:state_seq.shape[1] + self.scratch_tokens]
        return cls_out, opt_out, pooled, extra, scr_out

    def logits_value(self, o, opt_len=None, memory_in=None):
        cls_out, opt_out, pooled, extra, memory_out = self._encode(o, opt_len, memory_in)
        n = opt_out.shape[1]
        if self.structured:
            verb = o["opt_verb"][:, :n]
            opt_logits = (
                (opt_out * self.type_query(verb)).sum(-1)
                + self.type_bias(verb).squeeze(-1)
                + self.opt_head(opt_out).squeeze(-1)
            )
        else:
            opt_logits = self.opt_head(opt_out).squeeze(-1)
        if n < MAX_OPTIONS:
            opt_logits = torch.nn.functional.pad(opt_logits, (0, MAX_OPTIONS - n), value=-65504.0)
        submit = self.submit_head(extra[1] if self.split_heads else cls_out)
        vin = extra[0] if self.split_heads else torch.cat((cls_out, pooled), -1)
        if self.value_categorical:
            atom_logits = self.value_head(vin)
            value = (torch.softmax(atom_logits, dim=-1) * self.atom_support).sum(-1)
        else:
            value = self.value_head(vin).squeeze(-1)
        logits = torch.cat((opt_logits, submit), -1)
        # FP16 cannot represent -1e9; -65504 is the finite equivalent and
        # preserves the ordering/masking contract without overflowing.
        return logits.masked_fill(o["action_mask"] < 0.5, -65504.0), value, memory_out


def load_mlx_checkpoint(path: str | Path, card_table: CardTable, *, dtype=torch.float16):
    """Load an MLX checkpoint into a strict FP16 PyTorch inference model."""
    with open(path, "rb") as fh:
        state = pickle.load(fh)
    cfg = checkpoint_arch_config(state)
    model = TokenTransformerTorchInference(card_table, cfg)
    target = model.state_dict()
    source = _flatten_checkpoint(state["model"])
    converted: dict[str, torch.Tensor] = {}
    for key, expected in target.items():
        if key == "card_feat":
            continue
        mlx_key = key
        if key.startswith("encoder.layers."):
            parts = key.split(".")
            layer = ".".join(parts[:3])
            tail = ".".join(parts[3:])
            trans = {
                "self_attn.in_proj_weight": None,
                "self_attn.in_proj_bias": None,
                "self_attn.out_proj.weight": f"{layer}.attn.out_proj.weight",
                "self_attn.out_proj.bias": f"{layer}.attn.out_proj.bias",
                "linear1.weight": f"{layer}.ff.layers.0.weight",
                "linear1.bias": f"{layer}.ff.layers.0.bias",
                "linear2.weight": f"{layer}.ff.layers.2.weight",
                "linear2.bias": f"{layer}.ff.layers.2.bias",
                "norm1.weight": f"{layer}.norm1.weight", "norm1.bias": f"{layer}.norm1.bias",
                "norm2.weight": f"{layer}.norm2.weight", "norm2.bias": f"{layer}.norm2.bias",
            }
            if tail in ("self_attn.in_proj_weight", "self_attn.in_proj_bias"):
                proj = "weight" if tail.endswith("weight") else "bias"
                vals = [source[f"{layer}.attn.{p}_proj.{proj}"] for p in ("query", "key", "value")]
                tensor = torch.from_numpy(np.concatenate(vals, axis=0))
            else:
                mlx_key = trans[tail]
                tensor = torch.from_numpy(source[mlx_key])
        else:
            tensor = torch.from_numpy(source.get(mlx_key, source.get(key))) if (mlx_key in source or key in source) else None
        if tensor is None:
            raise ValueError(f"checkpoint missing parameter for {key}")
        if tuple(tensor.shape) != tuple(expected.shape):
            raise ValueError(f"shape mismatch for {key}: checkpoint {tuple(tensor.shape)} != model {tuple(expected.shape)}")
        converted[key] = tensor.to(dtype=dtype)
    missing_model = set(target) - set(converted)
    if missing_model:
        raise ValueError(f"checkpoint conversion omitted model parameters: {sorted(missing_model)}")
    model.load_state_dict(converted, strict=True)
    model = model.to(dtype=dtype)
    model.eval()
    return model, cfg


__all__ = [
    "TokenTransformerTorchInference",
    "checkpoint_arch_config",
    "load_mlx_checkpoint",
]


if __name__ == "__main__":
    from rl.encoder.card_features import get_card_table

    loaded, loaded_cfg = load_mlx_checkpoint(
        "model/bc_model/bc_best_mlx_final.pkl", get_card_table()
    )
    print(f"loaded {loaded_cfg['arch_version']} dtype={next(loaded.parameters()).dtype}")
    print(f"scratch={tuple(loaded.scratch.shape)} learned_init={tuple(loaded.learned_init.shape)}")
