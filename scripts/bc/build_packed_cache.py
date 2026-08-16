"""Build a data-identical fixed-width mmap store for one BC Parquet selection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.packed_data import (
    PACKED_BACKEND_NAME,
    PACKED_DEDUP_CONTRACT,
    PACKED_FORMAT_VERSION,
    PACKED_TBPTT_CONTRACT,
    _digest_array,
    _digest_columns,
    required_trainer_columns,
    sha256_file,
    source_digest,
    split_episode_ids,
    TRAINER_ORDER_COLUMNS,
)


def _read_normalized_batch(batch, name, enc, int_keys):
    # Importing the trainer helper keeps the candidate's logical dtype/shape
    # contract byte-for-byte aligned with the live loader.
    from scripts.bc.bc_train_mlx import _read_batch_column

    return _read_batch_column(batch, name, enc.shapes, int_keys)


def build(
    source: Path | list[Path],
    output: Path,
    *,
    max_rows: int,
    val_frac: float,
    seed: int,
) -> dict:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite packed output: {output}")
    output.mkdir(parents=True)

    sources = [source] if isinstance(source, Path) else list(source)
    if not sources:
        raise ValueError("packed ETL requires at least one Parquet source")
    if any(path.suffix != ".parquet" for path in sources):
        raise ValueError("packed ETL supports only .parquet sources")

    enc = TokenEncoder(get_card_table())
    int_keys = set(enc.int_keys)
    columns = required_trainer_columns(enc.shapes)
    if len(columns) != len(set(columns)):
        raise ValueError("packed column list contains duplicates")

    source_hash = source_digest([str(path) for path in sources])
    source_pfs = [pq.ParquetFile(path) for path in sources]
    dataset = pads.dataset([str(path) for path in sources], format="parquet")
    scan_parts = []
    for batch in dataset.to_batches(columns=["episode_id"]):
        if batch.num_rows:
            scan_parts.append(batch.column("episode_id").to_numpy(zero_copy_only=False))
    if not scan_parts:
        raise ValueError("source Parquet contains no rows")
    all_episode_ids = np.concatenate(scan_parts).astype(np.int64, copy=False)
    selected_eids, train_eids, val_eids = split_episode_ids(
        all_episode_ids, max_rows=max_rows, val_frac=val_frac
    )
    row_filter = pads.field("episode_id").isin(selected_eids.tolist())

    parts: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    for batch in dataset.to_batches(columns=columns, filter=row_filter, batch_size=32768):
        if not batch.num_rows:
            continue
        for name in columns:
            parts[name].append(_read_normalized_batch(batch, name, enc, int_keys))
    raw_arrays: dict[str, np.ndarray] = {}
    for name in columns:
        if not parts[name]:
            raise ValueError(f"selected source has no rows for column {name}")
        raw_arrays[name] = np.ascontiguousarray(np.concatenate(parts[name], axis=0))
    raw_episode_ids = raw_arrays["episode_id"]
    selected_mask = np.isin(raw_episode_ids, selected_eids)
    val_mask = np.isin(raw_episode_ids, val_eids)
    train_mask = np.isin(raw_episode_ids, train_eids)
    if np.any(val_mask & train_mask) or not np.array_equal(
        np.flatnonzero(selected_mask), np.concatenate([np.flatnonzero(val_mask), np.flatnonzero(train_mask)])
    ):
        raise AssertionError("selected source rows are not partitioned into val then train")
    arrays = {
        name: np.ascontiguousarray(
            np.concatenate([raw_arrays[name][val_mask], raw_arrays[name][train_mask]], axis=0)
        )
        for name in columns
    }
    selected_rows = len(arrays[columns[0]])
    if selected_rows != int(np.isin(all_episode_ids, selected_eids).sum()):
        raise AssertionError("selected row count mismatch while building packed store")
    if not np.array_equal(np.unique(raw_episode_ids[selected_mask]), np.unique(selected_eids)):
        raise AssertionError("selected episode IDs mismatch while building packed store")

    columns_dir = output / "columns"
    columns_dir.mkdir()
    specs: dict[str, dict] = {}
    column_digests: dict[str, str] = {}
    for name in columns:
        path = columns_dir / f"{name}.npy"
        np.save(path, arrays[name], allow_pickle=False)
        specs[name] = {
            "file": str(path.relative_to(output)),
            "dtype": str(arrays[name].dtype),
            "shape": list(arrays[name].shape),
            "nbytes": int(arrays[name].nbytes),
            "file_sha256": sha256_file(path),
        }
        column_digests[name] = _digest_array(arrays[name])

    manifest = {
        "format": "fixed-width-npy-mmap",
        "format_version": PACKED_FORMAT_VERSION,
        "backend": {
            "name": PACKED_BACKEND_NAME,
            "format": "fixed-width-npy-mmap",
            "format_version": PACKED_FORMAT_VERSION,
        },
        "source_paths": [str(path) for path in sources],
        "source_sha256": source_hash,
        "source_file_sha256": [sha256_file(path) for path in sources],
        "source_manifest_sha256": [
            sha256_file(path.with_name(path.stem + ".manifest.json"))
            if path.with_name(path.stem + ".manifest.json").is_file()
            else None
            for path in sources
        ],
        "source_rows": int(sum(pf.metadata.num_rows for pf in source_pfs)),
        "source_row_groups": int(sum(pf.metadata.num_row_groups for pf in source_pfs)),
        "selected_rows": selected_rows,
        "logical_bytes": int(sum(spec["nbytes"] for spec in specs.values())),
        "columns": columns,
        "column_specs": specs,
        "column_digests": column_digests,
        "data_digest": _digest_columns(arrays, columns),
        "selection": {
            "max_rows": int(max_rows),
            "val_frac": float(val_frac),
            "seed_semantics": "not used for episode selection; trainer shuffle only",
            "selected_episode_ids": [int(x) for x in selected_eids],
            "train_episode_ids": [int(x) for x in train_eids],
            "val_episode_ids": [int(x) for x in val_eids],
            "train_rows": int(np.isin(arrays["episode_id"], train_eids).sum()),
            "val_rows": int(np.isin(arrays["episode_id"], val_eids).sum()),
        },
        "required_contract": {
            "model_input_columns": sorted(enc.shapes),
            "labels": ["y", "is_attack", "opt_group"],
            "auxiliary": [
                "aux_ko",
                "aux_prize_delta",
                "aux_terminal",
                "aux_return",
                "aux_valid",
            ],
            "columns": columns,
            "order": list(TRAINER_ORDER_COLUMNS),
            "dedup": PACKED_DEDUP_CONTRACT,
            "tbptt": PACKED_TBPTT_CONTRACT,
        },
        "row_order": {
            "columns": list(TRAINER_ORDER_COLUMNS),
            "val_rows": int(val_mask.sum()),
            "train_rows": int(train_mask.sum()),
            "val_digest": _digest_columns(
                {name: arrays[name][: int(val_mask.sum())] for name in TRAINER_ORDER_COLUMNS},
                list(TRAINER_ORDER_COLUMNS),
            ),
            "train_digest": _digest_columns(
                {name: arrays[name][int(val_mask.sum()) :] for name in TRAINER_ORDER_COLUMNS},
                list(TRAINER_ORDER_COLUMNS),
            ),
        },
        "build": {
            "seed": int(seed),
            "seed_semantics": "provenance only; no effect on episode selection or row order",
        },
    }
    manifest_path = output / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", required=True, help="Parquet source; repeat for consecutive days")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-rows", type=int, default=2048)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13971479023478)
    args = parser.parse_args()
    manifest = build(
        [Path(path) for path in args.source],
        Path(args.out),
        max_rows=args.max_rows,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(json.dumps({
        "output": args.out,
        "selected_rows": manifest["selected_rows"],
        "train_rows": manifest["selection"]["train_rows"],
        "val_rows": manifest["selection"]["val_rows"],
        "logical_bytes": manifest["logical_bytes"],
        "data_digest": manifest["data_digest"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
