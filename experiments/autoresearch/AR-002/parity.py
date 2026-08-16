"""Independent Parquet -> packed value, shape, dtype, and row-order parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.packed_data import (
    PackedArrayStore,
    TRAINER_ORDER_COLUMNS,
    _digest_array,
    _digest_columns,
    required_trainer_columns,
    sha256_file,
    source_digest,
    split_episode_ids,
)


def _reference_column(batch, name: str, enc: TokenEncoder, int_keys: set[str]) -> np.ndarray:
    """Decode a Parquet batch independently of the production cache helper."""
    column = batch.column(name)
    if name == "opt_group":
        shape, dtype = enc.shapes["action_mask"], np.int32
    elif name in enc.shapes:
        shape = enc.shapes[name]
        dtype = np.int32 if name in int_keys else np.float32
    else:
        metadata_dtypes = {
            "y": np.int32, "is_attack": np.bool_, "episode_id": np.int64,
            "side": np.int32, "step_id": np.int32, "decision_id": np.int32,
            "substep": np.int32, "new_episode": np.bool_, "terminal": np.bool_,
            "reward": np.float32, "aux_ko": np.int32,
            "aux_prize_delta": np.float32, "aux_terminal": np.bool_,
            "aux_return": np.float32, "aux_valid": np.int32,
            # These two columns are currently normalized by the trainer's
            # fallback metadata dtype, so the independent reference must
            # compare the effective runtime representation, not Arrow's
            # physical type.
            "outcome": np.float32, "is_self": np.float32, "day_id": np.float32,
        }
        shape, dtype = None, metadata_dtypes[name]
    if shape is None:
        return column.to_numpy(zero_copy_only=False).astype(dtype)
    if shape == (1,):
        return column.to_numpy(zero_copy_only=False).astype(dtype).reshape(-1, 1)
    flat = column.flatten().to_numpy(zero_copy_only=False).astype(dtype)
    width = int(np.prod(shape))
    if len(flat) != batch.num_rows * width:
        raise ValueError(f"reference column {name} has an invalid flattened width")
    return flat.reshape((batch.num_rows, *shape))


def _read_reference(
    sources: list[Path], columns: list[str], enc: TokenEncoder,
    selected: np.ndarray, train: np.ndarray, val: np.ndarray,
) -> tuple[dict[str, np.ndarray], int, int]:
    dataset = pads.dataset([str(path) for path in sources], format="parquet")
    row_filter = pads.field("episode_id").isin(selected.tolist())
    parts = {name: [] for name in columns}
    for batch in dataset.to_batches(columns=columns, filter=row_filter, batch_size=32768):
        if batch.num_rows:
            for name in columns:
                parts[name].append(_reference_column(batch, name, enc, set(enc.int_keys)))
    raw = {name: np.concatenate(values, axis=0) for name, values in parts.items()}
    val_mask = np.isin(raw["episode_id"], val)
    train_mask = np.isin(raw["episode_id"], train)
    ordered = {
        name: np.ascontiguousarray(
            np.concatenate([raw[name][val_mask], raw[name][train_mask]], axis=0)
        )
        for name in columns
    }
    return ordered, int(val_mask.sum()), int(train_mask.sum())


def run(
    sources: list[Path], packed_path: Path, *, max_rows: int, val_frac: float, seed: int
) -> dict:
    enc = TokenEncoder(get_card_table())
    columns = required_trainer_columns(enc.shapes)
    dataset = pads.dataset([str(path) for path in sources], format="parquet")
    eid_parts = [
        batch.column("episode_id").to_numpy(zero_copy_only=False)
        for batch in dataset.to_batches(columns=["episode_id"])
        if batch.num_rows
    ]
    if not eid_parts:
        raise ValueError("source Parquet set contains no rows")
    all_eids = np.concatenate(eid_parts).astype(np.int64, copy=False)
    selected, train_ids, val_ids = split_episode_ids(
        all_eids, max_rows=max_rows, val_frac=val_frac
    )
    baseline, val_rows, train_rows = _read_reference(
        sources, columns, enc, selected, train_ids, val_ids
    )
    candidate_store = PackedArrayStore(
        packed_path, columns=columns, required_columns=columns
    )
    candidate = candidate_store.read_rows(
        np.arange(candidate_store.selected_rows, dtype=np.int64)
    )
    per_column = {}
    mismatches = []
    for name in columns:
        left, right = baseline[name], candidate[name]
        equal = left.dtype == right.dtype and left.shape == right.shape and np.array_equal(
            left, right, equal_nan=True
        )
        per_column[name] = {
            "equal": bool(equal), "dtype": str(right.dtype), "shape": list(right.shape),
            "baseline_digest": _digest_array(left), "candidate_digest": _digest_array(right),
            "bytes": int(right.nbytes),
        }
        if not equal:
            mismatches.append({"column": name, "baseline": per_column[name]})
    groups = {
        "row_order": list(TRAINER_ORDER_COLUMNS),
        "labels_and_masks": ["y", "is_attack", "opt_group"],
        "auxiliary": ["aux_ko", "aux_prize_delta", "aux_terminal", "aux_return", "aux_valid"],
        "model_inputs": sorted(enc.shapes),
    }
    group_digests = {
        group: {
            "baseline": _digest_columns(baseline, names),
            "candidate": _digest_columns(candidate, names),
            "equal": _digest_columns(baseline, names) == _digest_columns(candidate, names),
        }
        for group, names in groups.items()
    }
    row_order = {
        "columns": list(TRAINER_ORDER_COLUMNS), "val_rows": val_rows, "train_rows": train_rows,
        "baseline": _digest_columns(baseline, list(TRAINER_ORDER_COLUMNS)),
        "candidate": _digest_columns(candidate, list(TRAINER_ORDER_COLUMNS)),
        "equal": all(np.array_equal(baseline[name], candidate[name]) for name in TRAINER_ORDER_COLUMNS),
    }
    result = {
        "sources": [str(path) for path in sources],
        "source_sha256": source_digest([str(path) for path in sources]),
        "source_file_sha256": [sha256_file(path) for path in sources],
        "packed_data_digest": candidate_store.manifest["data_digest"],
        "required_columns": columns,
        "selection": {
            "max_rows": max_rows, "val_frac": val_frac, "trainer_seed": seed,
            "seed_semantics": "seed does not participate in episode selection or split",
            "selected_episodes": int(len(selected)), "selected_rows": int(len(candidate["episode_id"])),
            "train_rows": train_rows, "val_rows": val_rows,
            "episode_ids": [int(x) for x in selected],
        },
        "row_count_equal": len(baseline["episode_id"]) == len(candidate["episode_id"]),
        "row_order": row_order, "per_column": per_column,
        "digest_groups": group_digests, "mismatches": mismatches,
        "parity": not mismatches and row_order["equal"] and all(
            item["equal"] for item in group_digests.values()
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--packed", required=True)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13971479023478)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(
        [Path(path) for path in args.source], Path(args.packed),
        max_rows=args.max_rows, val_frac=args.val_frac, seed=args.seed,
    )
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "parity": result["parity"], "selected_rows": result["selection"]["selected_rows"],
        "train_rows": result["selection"]["train_rows"], "val_rows": result["selection"]["val_rows"],
        "row_order_equal": result["row_order"]["equal"],
        "mismatch_count": len(result["mismatches"]),
    }, indent=2, sort_keys=True))
    if not result["parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
