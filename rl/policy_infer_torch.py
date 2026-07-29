"""PyTorch inference mirror for the current MLX checkpoint.

This module is intentionally separate from ``rl/policy.py`` and
``rl/policy_mlx.py``.  It uses the checkpoint's embedded ``arch_config`` as the
inference contract; no training JSON is needed at runtime.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rl.encoder import effect_data
from rl.encoder.card_features import CardTable
from rl.encoder.encoding import MAX_OPTIONS, OPT_PICKED
from rl.policy import TokenTransformer, build_token_net
from rl.prospective_input_adapter import PROSPECTIVE_INPUT_ADAPTER_VERSION
from rl.prospective_input_adapter import (
    ACTION_ATTR_AGGREGATE_VERSION,
    ACTION_SET_FEATURE_VERSION,
    BRANCH_FEATURE_LAYOUT_VERSION,
)
from rl.prospective_planner_torch import (
    ProspectivePlannerTorch,
    convert_mlx_prospective_payload,
    extract_mlx_prospective_payload,
    load_prospective_torch_payload,
    prospective_torch_checkpoint_payload,
)
from rl.prospective_schema import (
    PROSPECTIVE_COORD_SCHEMA_VERSION,
    PROSPECTIVE_PLANNER_VERSION,
    ProspectivePlannerConfig,
)
from rl.token_schema import ARCH_VERSION, TOKEN_SCHEMA_VERSION

TORCH_INFERENCE_FORMAT = "ptcg-torch-fp16-v1"


def _disabled_prospective_config(*, provenance: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "planner_version": PROSPECTIVE_PLANNER_VERSION,
        "coord_schema_version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "input_adapter_version": PROSPECTIVE_INPUT_ADAPTER_VERSION,
        "action_attr_aggregate_version": ACTION_ATTR_AGGREGATE_VERSION,
        "action_set_feature_version": ACTION_SET_FEATURE_VERSION,
        "branch_feature_layout_version": BRANCH_FEATURE_LAYOUT_VERSION,
        "config": None,
        "runtime": None,
        "trained_optimizer_steps": 0,
        "provenance": provenance,
    }


def _normalize_prospective_inference_config(
    value: Any,
    *,
    provenance: str,
) -> dict[str, Any]:
    if value is None:
        return _disabled_prospective_config(provenance=provenance)
    if not isinstance(value, dict):
        raise ValueError("inference prospective_planner must be an object")
    enabled = bool(value.get("enabled", False))
    if not enabled:
        return _disabled_prospective_config(
            provenance=str(value.get("provenance", provenance))
        )
    if value.get("planner_version") != PROSPECTIVE_PLANNER_VERSION:
        raise ValueError("inference prospective planner version is unsupported")
    if value.get("coord_schema_version") != PROSPECTIVE_COORD_SCHEMA_VERSION:
        raise ValueError("inference prospective coordinate schema is unsupported")
    if value.get("input_adapter_version") != PROSPECTIVE_INPUT_ADAPTER_VERSION:
        raise ValueError("inference prospective input adapter is unsupported")
    if (
        value.get("action_attr_aggregate_version")
        != ACTION_ATTR_AGGREGATE_VERSION
    ):
        raise ValueError("inference action attribute aggregate is unsupported")
    if (
        value.get("action_set_feature_version")
        != ACTION_SET_FEATURE_VERSION
    ):
        raise ValueError("inference action set feature is unsupported")
    if (
        int(value.get("branch_feature_layout_version", -1))
        != BRANCH_FEATURE_LAYOUT_VERSION
    ):
        raise ValueError("inference branch feature layout is unsupported")
    planner_config_raw = value.get("config")
    if not isinstance(planner_config_raw, dict):
        raise ValueError("enabled prospective planner has no config")
    planner_config = ProspectivePlannerConfig(**planner_config_raw)
    planner_config.validate()
    runtime = value.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError("enabled prospective planner has no runtime config")
    normalized_runtime = {
        "trials": int(runtime.get("trials", 0)),
        "horizon": int(runtime.get("horizon", 0)),
        "max_branches": int(runtime.get("max_branches", 0)),
        "gamma": float(runtime.get("gamma", -1.0)),
        "seed": int(runtime.get("seed", 0)),
    }
    if (
        normalized_runtime["trials"] <= 0
        or normalized_runtime["horizon"] <= 0
        or normalized_runtime["max_branches"] <= 0
        or not 0.0 <= normalized_runtime["gamma"] <= 1.0
    ):
        raise ValueError("prospective runtime config has invalid bounds")
    if normalized_runtime["max_branches"] > MAX_OPTIONS:
        raise ValueError("prospective max_branches exceeds encoder capacity")
    trained_optimizer_steps = int(value.get("trained_optimizer_steps", 0))
    if trained_optimizer_steps <= 0:
        raise ValueError(
            "enabled prospective planner has no completed optimizer step"
        )
    return {
        "enabled": True,
        "planner_version": PROSPECTIVE_PLANNER_VERSION,
        "coord_schema_version": PROSPECTIVE_COORD_SCHEMA_VERSION,
        "input_adapter_version": PROSPECTIVE_INPUT_ADAPTER_VERSION,
        "action_attr_aggregate_version": ACTION_ATTR_AGGREGATE_VERSION,
        "action_set_feature_version": ACTION_SET_FEATURE_VERSION,
        "branch_feature_layout_version": BRANCH_FEATURE_LAYOUT_VERSION,
        "config": planner_config.to_dict(),
        "runtime": normalized_runtime,
        "trained_optimizer_steps": trained_optimizer_steps,
        "provenance": str(value.get("provenance", provenance)),
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_static_features(
    state: dict[str, Any],
    card_table: CardTable,
) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    features = state.get("static_card_features")
    contract = state.get("static_feature_contract")
    if features is None and contract is None:
        return None, None
    if features is None or not isinstance(contract, dict):
        raise ValueError("checkpoint static feature payload is incomplete")
    array = np.asarray(features, dtype=np.float16)
    expected_shape = tuple(int(value) for value in contract.get("shape", ()))
    if tuple(array.shape) != expected_shape:
        raise ValueError(
            f"static feature shape {tuple(array.shape)} != {expected_shape}"
        )
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    if digest != contract.get("sha256"):
        raise ValueError("checkpoint static feature table hash mismatch")
    csv_digest = _sha256_file(card_table.csv_path)
    if csv_digest != contract.get("card_csv_sha256"):
        raise ValueError(
            "runtime card table does not match the checkpoint's encoder contract"
        )
    return array, dict(contract)


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


def checkpoint_inference_config(state: dict[str, Any]) -> dict[str, Any]:
    """Read the runtime contract embedded in a trainer/export checkpoint.

    Older checkpoints predate this contract. They are treated as models that
    were trained without prospective would-KO features; this is the only safe
    legacy default because enabling a new input feature at inference would
    create train/inference drift.
    """
    cfg = state.get("inference_config")
    if cfg is None:
        return {
            "version": 1,
            "seed": int(state.get("seed", 0)),
            "bc_would_ko": False,
            "bc_wk_nvar": 10,
            "provenance": "legacy-checkpoint-default",
            "prospective_planner": _disabled_prospective_config(
                provenance="legacy-checkpoint-default"
            ),
        }
    if not isinstance(cfg, dict):
        raise ValueError("checkpoint inference_config must be an object")
    required = ("version", "seed", "bc_would_ko", "bc_wk_nvar")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"checkpoint inference_config is missing {missing}")
    if int(cfg["version"]) != 1:
        raise ValueError(
            f"unsupported inference_config version {cfg['version']!r}"
        )
    if int(cfg["bc_wk_nvar"]) <= 0:
        raise ValueError("checkpoint bc_wk_nvar must be positive")
    return {
        "version": 1,
        "seed": int(cfg["seed"]),
        "bc_would_ko": bool(cfg["bc_would_ko"]),
        "bc_wk_nvar": int(cfg["bc_wk_nvar"]),
        "provenance": str(cfg.get("provenance", "trainer-checkpoint")),
        "prospective_planner": _normalize_prospective_inference_config(
            cfg.get("prospective_planner"),
            provenance=(
                "checkpoint-disabled"
                if "prospective_planner" in cfg
                else "legacy-checkpoint-default"
            ),
        ),
    }


def _load_mlx_prospective_planner(
    state: dict[str, Any],
    inference_config: dict[str, Any],
) -> tuple[ProspectivePlannerTorch | None, ProspectivePlannerConfig | None]:
    payload = extract_mlx_prospective_payload(state)
    enabled = bool(inference_config["prospective_planner"]["enabled"])
    if payload is None:
        if enabled:
            raise ValueError(
                "checkpoint enables prospective inference without planner state"
            )
        return None, None
    model, config = convert_mlx_prospective_payload(payload)
    if (
        payload.get("input_adapter_version")
        != PROSPECTIVE_INPUT_ADAPTER_VERSION
    ):
        raise ValueError("prospective planner input adapter is incompatible")
    inference_planner = inference_config["prospective_planner"]
    if enabled and inference_planner["config"] != config.to_dict():
        raise ValueError(
            "prospective planner model and inference config disagree"
        )
    if enabled:
        trained_steps = int(payload.get("optimizer_steps", 0))
        if trained_steps <= 0:
            raise ValueError("enabled MLX prospective planner is untrained")
        if trained_steps != int(inference_planner["trained_optimizer_steps"]):
            raise ValueError(
                "prospective optimizer step disagrees with inference config"
            )
    return model, config


def _load_torch_prospective_planner(
    payload: dict[str, Any],
    inference_config: dict[str, Any],
) -> tuple[ProspectivePlannerTorch | None, ProspectivePlannerConfig | None]:
    planner_payload = payload.get("prospective_planner")
    enabled = bool(inference_config["prospective_planner"]["enabled"])
    if planner_payload is None:
        if enabled:
            raise ValueError(
                "artifact enables prospective inference without planner state"
            )
        return None, None
    if not isinstance(planner_payload, dict):
        raise ValueError("artifact prospective_planner must be an object")
    model, config = load_prospective_torch_payload(planner_payload)
    inference_planner = inference_config["prospective_planner"]
    if enabled and inference_planner["config"] != config.to_dict():
        raise ValueError(
            "prospective planner artifact and inference config disagree"
        )
    if enabled:
        trained_steps = int(planner_payload.get("trained_optimizer_steps", 0))
        if trained_steps <= 0:
            raise ValueError("enabled Torch prospective planner is untrained")
        if trained_steps != int(inference_planner["trained_optimizer_steps"]):
            raise ValueError(
                "artifact prospective optimizer step disagrees with config"
            )
    return model, config


class TokenTransformerTorchInference(TokenTransformer):
    """Dynamic-scratch, memory-aware PyTorch mirror of ``TokenTransformerMLX``."""

    def __init__(
        self,
        card_table: CardTable,
        cfg: dict[str, Any],
        static_card_features: np.ndarray | None = None,
    ) -> None:
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
        if static_card_features is not None:
            static_tensor = torch.as_tensor(
                static_card_features, dtype=torch.float16
            )
            if self.card_feat is None:
                raise ValueError(
                    "checkpoint contains static features for a non-static model"
                )
            if tuple(static_tensor.shape) != tuple(self.card_feat.shape):
                raise ValueError(
                    f"static feature shape {tuple(static_tensor.shape)} != "
                    f"model {tuple(self.card_feat.shape)}"
                )
            self.card_feat = static_tensor
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
    arch_cfg = checkpoint_arch_config(state)
    static_features, static_contract = _checkpoint_static_features(
        state, card_table
    )
    inference_config = checkpoint_inference_config(state)
    prospective_model, prospective_config = _load_mlx_prospective_planner(
        state, inference_config
    )
    metadata = {
        **arch_cfg,
        "inference_config": inference_config,
        "training_config": dict(state.get("run_config", {})),
        "static_feature_contract": static_contract,
        "prospective_planner_model": prospective_model,
        "prospective_planner_config": prospective_config,
    }
    model = TokenTransformerTorchInference(
        card_table, arch_cfg, static_features
    )
    target = model.state_dict()
    source = _flatten_checkpoint(state["model"])
    converted: dict[str, torch.Tensor] = {}
    used_source: set[str] = set()
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
                source_keys = [
                    f"{layer}.attn.{p}_proj.{proj}"
                    for p in ("query", "key", "value")
                ]
                vals = [source[source_key] for source_key in source_keys]
                used_source.update(source_keys)
                tensor = torch.from_numpy(np.concatenate(vals, axis=0))
            else:
                mlx_key = trans[tail]
                used_source.add(mlx_key)
                tensor = torch.from_numpy(source[mlx_key])
        else:
            source_key = mlx_key if mlx_key in source else key
            tensor = (
                torch.from_numpy(source[source_key])
                if source_key in source
                else None
            )
            if tensor is not None:
                used_source.add(source_key)
        if tensor is None:
            raise ValueError(f"checkpoint missing parameter for {key}")
        if tuple(tensor.shape) != tuple(expected.shape):
            raise ValueError(f"shape mismatch for {key}: checkpoint {tuple(tensor.shape)} != model {tuple(expected.shape)}")
        converted[key] = tensor.to(dtype=dtype)
    missing_model = set(target) - set(converted)
    if missing_model:
        raise ValueError(f"checkpoint conversion omitted model parameters: {sorted(missing_model)}")
    unexpected_source = set(source) - used_source
    if unexpected_source:
        raise ValueError(
            "MLX checkpoint contains unmapped parameters: "
            f"{sorted(unexpected_source)}"
        )
    model.load_state_dict(converted, strict=True)
    model = model.to(dtype=dtype)
    model.eval()
    return model, metadata


def save_torch_inference_checkpoint(
    mlx_path: str | Path,
    torch_path: str | Path,
    card_table: CardTable,
) -> dict[str, Any]:
    """Strictly convert an MLX trainer checkpoint to portable PyTorch FP16."""
    model, metadata = load_mlx_checkpoint(
        mlx_path, card_table, dtype=torch.float16
    )
    state_dict = model.state_dict()
    bad_dtype = {
        key: str(value.dtype)
        for key, value in state_dict.items()
        if value.is_floating_point() and value.dtype != torch.float16
    }
    if bad_dtype:
        raise ValueError(f"FP16 conversion left non-FP16 tensors: {bad_dtype}")
    payload = {
        "format": TORCH_INFERENCE_FORMAT,
        "arch_config": {
            key: value
            for key, value in metadata.items()
            if key not in (
                "inference_config",
                "training_config",
                "prospective_planner_model",
                "prospective_planner_config",
            )
        },
        "inference_config": metadata["inference_config"],
        "training_config": metadata["training_config"],
        "static_card_features": (
            model.card_feat.detach().to(torch.float16)
            if metadata["static_feature_contract"] is not None
            else None
        ),
        "static_feature_contract": metadata["static_feature_contract"],
        "state_dict": state_dict,
        "prospective_planner": (
            prospective_torch_checkpoint_payload(
                metadata["prospective_planner_model"],
                trained_optimizer_steps=int(
                    metadata["inference_config"]["prospective_planner"][
                        "trained_optimizer_steps"
                    ]
                ),
            )
            if metadata["prospective_planner_model"] is not None
            else None
        ),
    }
    torch.save(payload, torch_path)
    return metadata


def load_torch_inference_checkpoint(
    path: str | Path,
    card_table: CardTable,
) -> tuple[TokenTransformerTorchInference, dict[str, Any]]:
    """Load a pre-converted PyTorch checkpoint with exact keys, shapes and FP16."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != TORCH_INFERENCE_FORMAT:
        raise ValueError(
            f"unsupported PyTorch inference checkpoint format "
            f"{payload.get('format') if isinstance(payload, dict) else type(payload).__name__!r}"
        )
    arch_cfg = checkpoint_arch_config(
        {"arch_config": payload.get("arch_config")}
    )
    static_features, static_contract = _checkpoint_static_features(
        payload, card_table
    )
    inference_config = checkpoint_inference_config(payload)
    prospective_model, prospective_config = _load_torch_prospective_planner(
        payload, inference_config
    )
    metadata = {
        **arch_cfg,
        "inference_config": inference_config,
        "training_config": dict(payload.get("training_config", {})),
        "static_feature_contract": static_contract,
        "prospective_planner_model": prospective_model,
        "prospective_planner_config": prospective_config,
    }
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("PyTorch inference checkpoint has no state_dict")

    model = TokenTransformerTorchInference(
        card_table, arch_cfg, static_features
    ).to(dtype=torch.float16)
    expected = model.state_dict()
    missing = set(expected) - set(state_dict)
    unexpected = set(state_dict) - set(expected)
    if missing or unexpected:
        raise ValueError(
            f"PyTorch inference state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"state_dict[{key!r}] is not a tensor")
        if tuple(tensor.shape) != tuple(expected[key].shape):
            raise ValueError(
                f"shape mismatch for {key}: checkpoint {tuple(tensor.shape)} "
                f"!= model {tuple(expected[key].shape)}"
            )
        if tensor.is_floating_point() and tensor.dtype != torch.float16:
            raise ValueError(f"state_dict[{key!r}] is {tensor.dtype}, expected torch.float16")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, metadata


def load_inference_checkpoint(
    path: str | Path,
    card_table: CardTable,
) -> tuple[TokenTransformerTorchInference, dict[str, Any]]:
    """Load either the submission FP16 artifact or a local MLX trainer pickle."""
    path = Path(path)
    if path.suffix == ".pt":
        return load_torch_inference_checkpoint(path, card_table)
    return load_mlx_checkpoint(path, card_table, dtype=torch.float16)


__all__ = [
    "TokenTransformerTorchInference",
    "TORCH_INFERENCE_FORMAT",
    "checkpoint_arch_config",
    "checkpoint_inference_config",
    "load_inference_checkpoint",
    "load_mlx_checkpoint",
    "load_torch_inference_checkpoint",
    "save_torch_inference_checkpoint",
]


if __name__ == "__main__":
    from rl.encoder.card_features import get_card_table

    loaded, loaded_cfg = load_mlx_checkpoint(
        "model/bc_model/bc_best_mlx_final.pkl", get_card_table()
    )
    print(f"loaded {loaded_cfg['arch_version']} dtype={next(loaded.parameters()).dtype}")
    print(f"scratch={tuple(loaded.scratch.shape)} learned_init={tuple(loaded.learned_init.shape)}")
