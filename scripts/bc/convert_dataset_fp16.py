"""Convert BC dataset from float32 to float16 (half precision).

Reduces disk size by ~50% and I/O time by ~50% for float feature arrays.
Integer arrays (card IDs, labels) stay as int32.

Usage:
  python scripts/bc/convert_dataset_fp16.py data/bc_data/bc_2026_07_21
  python scripts/bc/convert_dataset_fp16.py data/bc_data/bc_2026_07_21 --out data/bc_data/bc_2026_07_21_fp16
"""
import argparse
import os
import sys
import time

import numpy as np

# Keys that MUST stay as int32 (card IDs, indices, labels)
INT_KEYS = {
    "__labels__", "__group__", "__is_attack__",
    "self_deck_id", "opp_deck_id",
    "self_prize_id", "opp_prize_id",
    "self_hand_id", "opp_hand_id",
    "self_discard_id", "opp_discard_id",
    "stadium_id",
    "self_unit_top_id", "self_unit_preevo_id", "self_unit_tool_id", "self_unit_energy_id",
    "opp_unit_top_id", "opp_unit_preevo_id", "opp_unit_tool_id", "opp_unit_energy_id",
    "opt_src_pos", "opt_tgt_pos",
    "opt_src_card", "opt_tgt_card",
    "opt_verb", "opt_attack_id",
    "select_type", "select_context",
    "effect_id",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data", help="Input .npy directory (float32)")
    p.add_argument("--out", default=None,
                   help="Output directory (default: input + '_fp16')")
    p.add_argument("--verify", action="store_true",
                   help="Verify precision loss after conversion")
    a = p.parse_args()

    in_dir = os.path.abspath(a.data)
    out_dir = a.out or (in_dir.rstrip("/") + "_fp16")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(in_dir) if f.endswith(".npy"))
    print(f"[fp16] Converting {len(files)} files: {in_dir} -> {out_dir}", flush=True)

    total_in = 0
    total_out = 0
    converted = 0
    kept = 0

    for fname in files:
        key = fname[:-4]
        in_path = os.path.join(in_dir, fname)
        out_path = os.path.join(out_dir, fname)

        arr = np.load(in_path, mmap_mode="r")
        orig_dtype = arr.dtype
        orig_size = os.path.getsize(in_path)
        total_in += orig_size

        if key in INT_KEYS or arr.dtype in (np.int32, np.int64, np.int8):
            # Keep integer arrays as-is (chunked copy for large arrays)
            chunk_rows = max(1, 500_000_000 // max(int(np.prod(arr.shape[1:], dtype=np.int64)) * arr.dtype.itemsize, 1))
            if arr.shape[0] > chunk_rows:
                out_arr = np.lib.format.open_memmap(out_path, mode='w+',
                                                     dtype=arr.dtype, shape=arr.shape)
                for start in range(0, arr.shape[0], chunk_rows):
                    end = min(start + chunk_rows, arr.shape[0])
                    out_arr[start:end] = np.asarray(arr[start:end])
                out_arr.flush()
            else:
                np.save(out_path, np.asarray(arr))
            kept += 1
            out_size = os.path.getsize(out_path)
            total_out += out_size
            print(f"  {fname:30s} {str(arr.shape):20s} {orig_dtype} -> kept ({out_size / 1e6:.1f} MB)",
                  flush=True)
        elif arr.dtype == np.float32:
            # Convert to float16 in CHUNKS to avoid OOM on 8GB machines
            chunk_rows = max(1, 500_000_000 // (int(np.prod(arr.shape[1:], dtype=np.int64)) * 4))
            n_rows = arr.shape[0]
            max_diff = 0.0
            mean_diff_sum = 0.0
            mean_diff_count = 0

            # Create output memmap
            out_shape = arr.shape
            out_arr = np.lib.format.open_memmap(out_path, mode='w+',
                                                 dtype=np.float16, shape=out_shape)
            for start in range(0, n_rows, chunk_rows):
                end = min(start + chunk_rows, n_rows)
                chunk = np.asarray(arr[start:end])
                chunk_fp16 = chunk.astype(np.float16)
                out_arr[start:end] = chunk_fp16
                if a.verify:
                    diff = np.abs(chunk - chunk_fp16.astype(np.float32))
                    max_diff = max(max_diff, float(diff.max()))
                    mean_diff_sum += float(diff.sum())
                    mean_diff_count += diff.size
            out_arr.flush()
            converted += 1
            out_size = os.path.getsize(out_path)
            total_out += out_size

            if a.verify:
                mean_diff = mean_diff_sum / max(mean_diff_count, 1)
                print(f"  {fname:30s} {str(arr.shape):20s} fp32->fp16 "
                      f"({orig_size / 1e6:.1f} -> {out_size / 1e6:.1f} MB) "
                      f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}",
                      flush=True)
            else:
                print(f"  {fname:30s} {str(arr.shape):20s} fp32->fp16 "
                      f"({orig_size / 1e6:.1f} -> {out_size / 1e6:.1f} MB)",
                      flush=True)
        else:
            # Unknown dtype, copy as-is
            np.save(out_path, np.asarray(arr))
            out_size = os.path.getsize(out_path)
            total_out += out_size
            print(f"  {fname:30s} {str(arr.shape):20s} {orig_dtype} -> copied", flush=True)

    print(f"\n[fp16] Done!", flush=True)
    print(f"  Converted: {converted} files (float32 -> float16)", flush=True)
    print(f"  Kept: {kept} files (int32, unchanged)", flush=True)
    print(f"  Total in:  {total_in / 1e9:.2f} GB", flush=True)
    print(f"  Total out: {total_out / 1e9:.2f} GB", flush=True)
    print(f"  Reduction: {(1 - total_out / total_in) * 100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
