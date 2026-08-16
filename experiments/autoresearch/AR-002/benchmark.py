"""AR-002 data-identical baseline/candidate load benchmark."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pyarrow.dataset as pads

from rl.encoder.card_features import get_card_table
from rl.encoder.encoding import TokenEncoder
from rl.packed_data import PackedArrayStore, sha256_file, split_episode_ids
from scripts.bc.bc_train_mlx import (
    _AUX_COLUMNS,
    _META_COLUMN_DTYPES,
    _ParquetRowGroupCache,
    _TBPTT_FILTER_CACHE,
    _scan_tbptt_locations,
)


def _rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes; this benchmark is intentionally local
    # to the target machine rather than pretending Linux units are universal.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _swap_bytes() -> dict[str, int]:
    import psutil

    swap = psutil.swap_memory()
    return {"sin": int(swap.sin), "sout": int(swap.sout)}


def _selected(source: Path, max_rows: int, val_frac: float):
    dataset = pads.dataset([str(source)], format="parquet")
    parts = []
    for batch in dataset.to_batches(columns=["episode_id"]):
        if batch.num_rows:
            parts.append(batch.column("episode_id").to_numpy(zero_copy_only=False))
    all_ids = np.concatenate(parts).astype(np.int64, copy=False)
    return dataset, all_ids, split_episode_ids(
        all_ids, max_rows=max_rows, val_frac=val_frac
    )


def _required_columns(enc: TokenEncoder) -> list[str]:
    # This is the exact default Stage-4 cache column set: dedup is disabled in
    # AR-001, while all five auxiliary targets are active.
    return sorted(enc.shapes) + ["y", "is_attack"] + list(_AUX_COLUMNS)


def baseline_load(source: Path, max_rows: int, val_frac: float) -> dict:
    dataset, all_ids, (selected, train_ids, val_ids) = _selected(
        source, max_rows, val_frac
    )
    enc = TokenEncoder(get_card_table())
    columns = _required_columns(enc)
    int_keys = set(enc.int_keys)
    result = {
        "selected_rows": int(np.isin(all_ids, selected).sum()),
        "train_rows": 0,
        "val_rows": 0,
        "columns": len(columns),
        "splits": {},
    }
    t0 = time.perf_counter()
    total_cache_bytes = 0
    total_rows = 0
    total_scan_s = 0.0
    total_read_s = 0.0
    for split_name, ids in (("val", val_ids), ("train", train_ids)):
        row_filter = pads.field("episode_id").isin(ids.tolist())
        _TBPTT_FILTER_CACHE[id(row_filter)] = np.asarray(ids, dtype=np.int64)
        scan_t0 = time.perf_counter()
        meta, file_idx, rg_idx, offsets, paths = _scan_tbptt_locations(
            dataset, row_filter, enc.shapes, int_keys
        )
        scan_s = time.perf_counter() - scan_t0
        n_rows = len(meta["episode_id"])
        cache = _ParquetRowGroupCache(
            paths, columns, enc.shapes, int_keys, ssd_spill_dir=None
        )
        read_t0 = time.perf_counter()
        cache.read_rows(np.arange(n_rows, dtype=np.int64), file_idx, rg_idx, offsets)
        read_s = time.perf_counter() - read_t0
        report = cache.report()
        result["splits"][split_name] = {
            "rows": n_rows,
            "scan_s": scan_s,
            "read_s": read_s,
            "cache_report": report,
        }
        result[f"{split_name}_rows"] = n_rows
        total_rows += n_rows
        total_cache_bytes += report["bytes_loaded"]
        total_scan_s += scan_s
        total_read_s += read_s
    result.update(
        {
            "load_s": time.perf_counter() - t0,
            "location_scan_s": total_scan_s,
            "rowgroup_read_s": total_read_s,
            "decoded_bytes": total_cache_bytes,
            "decoded_bytes_per_row": total_cache_bytes / total_rows,
            "rows_per_s": total_rows / total_read_s,
            "rss_peak_bytes": _rss_bytes(),
            "swap": _swap_bytes(),
            "etl_s": 0.0,
            "total_source_to_ready_s": time.perf_counter() - t0,
        }
    )
    return result


def candidate_load(source: Path, packed: Path, max_rows: int, val_frac: float, seed: int) -> dict:
    enc = TokenEncoder(get_card_table())
    columns = _required_columns(enc)
    manifest = json.loads((packed / "manifest.json").read_text())
    selected = manifest["selection"]
    if manifest["source_sha256"] != sha256_file(source):
        raise ValueError("candidate source hash does not match fixed Parquet")
    if selected["max_rows"] != max_rows or selected["val_frac"] != val_frac or selected["seed"] != seed:
        raise ValueError("candidate selection contract does not match benchmark")
    split_ranges = {
        "val": (0, int(selected["val_rows"])),
        "train": (
            int(selected["val_rows"]),
            int(selected["val_rows"] + selected["train_rows"]),
        ),
    }
    result = {
        "selected_rows": int(manifest["selected_rows"]),
        "train_rows": int(selected["train_rows"]),
        "val_rows": int(selected["val_rows"]),
        "columns": len(columns),
        "splits": {},
        "packed_data_digest": manifest["data_digest"],
        "packed_logical_bytes": int(manifest["logical_bytes"]),
    }
    t0 = time.perf_counter()
    total_rows = 0
    total_bytes = 0
    for split_name, (start, stop) in split_ranges.items():
        load_t0 = time.perf_counter()
        store = PackedArrayStore(
            packed, row_start=start, row_stop=stop, columns=columns
        )
        load_s = time.perf_counter() - load_t0
        read_t0 = time.perf_counter()
        store.read_rows(np.arange(stop - start, dtype=np.int64))
        read_s = time.perf_counter() - read_t0
        report = store.report()
        result["splits"][split_name] = {
            "rows": stop - start,
            "constructor_s": load_s,
            "read_s": read_s,
            "cache_report": report,
        }
        total_rows += stop - start
        total_bytes += store.logical_bytes
    result.update(
        {
            "load_s": time.perf_counter() - t0,
            "constructor_s": sum(v["constructor_s"] for v in result["splits"].values()),
            "mmap_read_s": sum(v["read_s"] for v in result["splits"].values()),
            "decoded_bytes": total_bytes,
            "decoded_bytes_per_row": total_bytes / total_rows,
            "rows_per_s": total_rows / sum(v["read_s"] for v in result["splits"].values()),
            "rss_peak_bytes": _rss_bytes(),
            "swap": _swap_bytes(),
            "etl_s": None,
            "total_source_to_ready_s": None,
        }
    )
    return result


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
    packed = Path(args.packed)
    result = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "selection": {
            "max_rows": args.max_rows,
            "val_frac": args.val_frac,
            "seed": args.seed,
        },
        "baseline": baseline_load(source, args.max_rows, args.val_frac),
        "candidate": candidate_load(
            source, packed, args.max_rows, args.val_frac, args.seed
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
