"""AR-002 exact value, dtype, shape, and ordering parity proof."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.packed_data import PackedArrayStore, _digest_array, _digest_columns, sha256_file
from scripts.bc.bc_train_mlx import (
    _AUX_COLUMNS,
    _META_COLUMN_DTYPES,
    _ParquetRowGroupCache,
    _TBPTT_FILTER_CACHE,
    _scan_tbptt_locations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--packed", required=True)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13971479023478)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.source)
    packed_path = Path(args.packed)
    manifest = json.loads((packed_path / "manifest.json").read_text())
    enc = TokenEncoder(get_card_table())
    columns = list(manifest["columns"])
    int_keys = set(enc.int_keys)
    dataset = pads.dataset([str(source)], format="parquet")
    scan_parts = []
    for batch in dataset.to_batches(columns=["episode_id"]):
        if batch.num_rows:
            scan_parts.append(batch.column("episode_id").to_numpy(zero_copy_only=False))
    all_eids = np.concatenate(scan_parts).astype(np.int64, copy=False)
    unique, first, counts = np.unique(
        all_eids, return_index=True, return_counts=True
    )
    order = np.argsort(first)
    unique = unique[order]
    counts = counts[order]
    cutoff = min(int(np.searchsorted(np.cumsum(counts), args.max_rows) + 1), len(unique))
    selected = unique[:cutoff]
    n_val_eps = max(1, int(round(len(selected) * args.val_frac)))
    if n_val_eps >= len(selected):
        n_val_eps = len(selected) - 1
    val_ids = selected[:n_val_eps]
    train_ids = selected[n_val_eps:]

    baseline_parts: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    split_counts = {}
    for split_name, ids in (("val", val_ids), ("train", train_ids)):
        row_filter = pads.field("episode_id").isin(ids.tolist())
        _TBPTT_FILTER_CACHE[id(row_filter)] = np.asarray(ids, dtype=np.int64)
        meta, file_idx, rg_idx, offsets, paths = _scan_tbptt_locations(
            dataset, row_filter, enc.shapes, int_keys
        )
        n_rows = len(meta["episode_id"])
        cache = _ParquetRowGroupCache(
            paths, columns, enc.shapes, int_keys, ssd_spill_dir=None
        )
        fetched = cache.read_rows(
            np.arange(n_rows, dtype=np.int64), file_idx, rg_idx, offsets
        )
        split_counts[split_name] = n_rows
        for name in columns:
            baseline_parts[name].append(fetched[name])

    baseline = {
        name: np.concatenate(baseline_parts[name], axis=0) for name in columns
    }
    candidate_store = PackedArrayStore(packed_path)
    candidate = candidate_store.read_rows(np.arange(candidate_store.selected_rows))
    mismatches: list[dict] = []
    per_column = {}
    for name in columns:
        left = baseline[name]
        right = candidate[name]
        equal = (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
        per_column[name] = {
            "equal": bool(equal),
            "dtype": str(right.dtype),
            "shape": list(right.shape),
            "baseline_digest": _digest_array(left),
            "candidate_digest": _digest_array(right),
            "bytes": int(right.nbytes),
        }
        if not equal:
            mismatches.append({"column": name, "baseline": per_column[name]})

    digest_groups = {
        "episode_side_order": ["episode_id", "side", "step_id", "decision_id", "substep"],
        "labels_and_masks": ["y", "is_attack", "opt_group"],
        "auxiliary": list(_AUX_COLUMNS),
        "model_inputs": sorted(enc.shapes),
    }
    group_digests = {}
    for group, names in digest_groups.items():
        group_digests[group] = {
            "baseline": _digest_columns(baseline, names),
            "candidate": _digest_columns(candidate, names),
            "equal": _digest_columns(baseline, names) == _digest_columns(candidate, names),
        }

    result = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "packed_data_digest": manifest["data_digest"],
        "selection": {
            "max_rows": args.max_rows,
            "val_frac": args.val_frac,
            "seed": args.seed,
            "selected_episodes": int(len(selected)),
            "selected_rows": int(len(candidate["episode_id"])),
            "train_rows": int(split_counts["train"]),
            "val_rows": int(split_counts["val"]),
            "episode_ids": [int(x) for x in selected],
        },
        "row_count_equal": int(len(baseline["episode_id"])) == int(len(candidate["episode_id"])),
        "per_column": per_column,
        "digest_groups": group_digests,
        "mismatches": mismatches,
        "parity": not mismatches and all(item["equal"] for item in group_digests.values()),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "parity": result["parity"],
        "selected_rows": result["selection"]["selected_rows"],
        "train_rows": result["selection"]["train_rows"],
        "val_rows": result["selection"]["val_rows"],
        "mismatch_count": len(mismatches),
        "digest_groups": group_digests,
    }, indent=2, sort_keys=True))
    if not result["parity"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
