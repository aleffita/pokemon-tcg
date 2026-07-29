"""Centralized training and project configuration.

Hierarchy: CLI args > JSON config > defaults.
All scripts should read config through this module for consistency.

Usage:
    from rl.train_config import load_config
    cfg = load_config(cli_args_dict)  # CLI overrides
    cfg = load_config()               # JSON + defaults only
"""

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "configs/train_config.json"

# Fields that map to different CLI flag names
_FIELD_TO_FLAG = {
    "batch_size": "batch",
    "data_dir": "data",
    "replay_zip_dir": "replay-zip",
    "kaggle_episodes_prefix": "kaggle-episodes-prefix",
    "checkpoint_dir": "checkpoint-dir",
    "model_dir": "model-dir",
    "lr_min_ratio": "lr-min-ratio",
    "max_grad_norm": "max-grad-norm",
    "warmup_steps": "warmup-steps",
    "accum_steps": "accum-steps",
    "kaggle_competition": "kaggle-competition",
    "slab_rows": "slab-rows",
    "val_frac": "val-frac",
    "bc_workers": "workers",
    "bc_flush": "flush",
    "bc_ep_timeout": "ep-timeout",
    "max_episodes": "max-episodes",
    "max_rows": "max-rows",
}


@dataclass
class TrainConfig:
    # Model architecture
    d_model: int = 128
    nhead: int = 4
    nlayers: int = 3
    ff_dim: int = 512
    static: bool = True
    split_heads: bool = True
    structured: bool = False
    scratch_registers: int = 16

    # Trainer options
    prefetch: bool = False
    compile: bool = False
    log_interval: int = 100

    # Training
    epochs: int = 10
    batch_size: int = 128
    accum_steps: int = 1
    lr: float = 5e-4
    lr_schedule: str = "cosine"
    warmup_steps: int = 1500
    lr_min_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 0
    optimizer: str = "muon_adamw"
    optimizer_state: str = "reset"
    scheduler_state: str = "reset"
    scheduler_total_steps: int = 0
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.01
    adamw_betas: list[float] = field(default_factory=lambda: [0.9, 0.999])
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.01

    # Data
    data_dir: str = "data/bc_data"
    replay_zip_dir: str = "data/bc_replay_zip"
    slab_rows: int = 32768
    val_frac: float = 0.1
    val_batch_size: int = 128

    # Output
    checkpoint_dir: str = "model/checkpoint"
    model_dir: str = "model/bc_model"
    checkpoint_every_epochs: int = 1

    # Kaggle
    kaggle_competition: str = "pokemon-tcg-ai-battle"
    kaggle_episodes_prefix: str = "kaggle/pokemon-tcg-ai-battle-episodes"

    # BC pipeline build
    bc_workers: int = 8
    bc_flush: int = 200  # episodes per batch/shard
    bc_ep_timeout: int = 60  # seconds per episode
    bc_would_ko: bool = False
    bc_wk_nvar: int = 10
    bc_both_sides: bool = True
    bc_self_aliases: list[str] = field(
        default_factory=lambda: ["FitaLabs", "Alef Oliveira"]
    )

    # Training constraints (0 = all)
    max_episodes: int = 0  # 0 = all episodes (for smoke testing)
    max_rows: int = 0  # 0 = all rows (for smoke testing)

    # Categorical value (model)
    value_atoms: int = 51
    value_vmax: float = 1.0

    # F.3: TBPTT (0 = disabled, 8/16/32 for sequential training)
    tbptt_chunk: int = 0

    # Lateral prospective planner. The builder materializes a real rollout
    # sidecar; the trainer consumes it only when explicitly enabled.
    prospective_enabled: bool = False
    prospective_sidecar_name: str = "prospective_v2"
    prospective_max_groups: int = 0
    prospective_max_branches: int = 64
    prospective_trials: int = 1
    prospective_horizon: int = 2
    prospective_gamma: float = 1.0

    # Prospective planner architecture/training. d_model is shared with the
    # trunk so detached causal context can be passed without another bridge.
    prospective_batch_size: int = 128
    prospective_nhead: int = 4
    prospective_nlayers: int = 2
    prospective_ff_dim: int = 512
    prospective_rope_base: float = 10000.0
    prospective_uncertainty_floor: float = 1e-6
    prospective_lr: float = 2.457e-4
    prospective_scheduler_total_steps: int = 0
    prospective_clip_ratio: float = 0.2
    prospective_kl_coefficient: float = 0.0
    prospective_policy_weight: float = 1.0
    prospective_return_weight: float = 1.0
    prospective_value_weight: float = 1.0
    prospective_ko_weight: float = 1.0
    prospective_prize_weight: float = 1.0
    prospective_terminal_weight: float = 1.0
    prospective_uncertainty_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def get_default_config_path() -> str:
    """Find configs/train_config.json walking up from CWD toward project root.

    Walks up from os.getcwd() looking for ``configs/train_config.json``.
    Returns the first one found, or the bare name ``configs/train_config.json``
    (CWD relative) if none is discovered.
    """
    current = Path.cwd()
    for candidate in [current, *current.parents]:
        cfg = candidate / DEFAULT_CONFIG_PATH
        if cfg.is_file():
            return str(cfg)
    return DEFAULT_CONFIG_PATH


def _parse_cli_overrides(raw: dict[str, Any] | None) -> dict[str, Any]:
    """Normalise CLI overrides into field-name keys.

    Accepts argparse-style ``snake_case`` keys directly, **or** flags
    like ``batch`` or ``data`` that map to a field with a different name.
    Unrecognised keys are silently dropped so that extras (``--verbose``
    etc.) do not cause errors.
    """
    if not raw:
        return {}

    flag_to_field = {v: k for k, v in _FIELD_TO_FLAG.items()}
    out: dict[str, Any] = {}

    for key, value in raw.items():
        # Normalise leading dashes away
        clean = key.lstrip("-").replace("-", "_")
        if clean in flag_to_field:
            clean = flag_to_field[clean]
        if hasattr(TrainConfig, clean):
            out[clean] = value
    return out


def _coerce_value(field_name: str, raw_value: Any) -> Any:
    """Cast a JSON value to the correct type for the given dataclass field."""
    # Grab the field type from our dataclass definition
    import dataclasses
    for f in dataclasses.fields(TrainConfig):
        if f.name == field_name:
            expected = f.type
            break
    else:
        return raw_value

    if expected == "bool":
        return bool(raw_value)
    if expected == "int":
        return int(raw_value)
    if expected == "float":
        return float(raw_value)
    if expected == "str":
        return str(raw_value)
    return raw_value


def load_config(
    cli_overrides: dict[str, Any] | None = None,
    config_path: str | None = None,
) -> TrainConfig:
    """Build a TrainConfig with the hierarchy: CLI > specified config > CWD config.json > defaults.

    Parameters
    ----------
    cli_overrides:
        Argument dict from argparse or similar. Unrecognised keys are
        ignored so that extra flags don't break config loading.
    config_path:
        Path to a JSON config file (--config). When None, falls back to
        searching for configs/train_config.json walking up from CWD.
        If the file does not exist the defaults are used unchanged.
    """
    resolved_path = config_path or get_default_config_path()
    json_overrides: dict[str, Any] = {}

    if resolved_path and os.path.isfile(resolved_path):
        with open(resolved_path) as fh:
            json_overrides = json.load(fh) or {}

    # Parse CLI overrides (may be None or an argparse Namespace dict)
    cli = _parse_cli_overrides(cli_overrides)

    # Build kwargs: defaults < JSON < CLI
    kwargs: dict[str, Any] = {}
    for field_name in [f.name for f in TrainConfig.__dataclass_fields__.values()]:
        if field_name in json_overrides:
            kwargs[field_name] = _coerce_value(field_name, json_overrides[field_name])
        if field_name in cli:
            kwargs[field_name] = _coerce_value(field_name, cli[field_name])

    return TrainConfig(**kwargs)


def save_config(cfg: TrainConfig, path: str = "configs/train_config.json") -> None:
    """Persist a TrainConfig to a JSON file."""
    with open(path, "w") as fh:
        fh.write(cfg.to_json())
        fh.write("\n")
