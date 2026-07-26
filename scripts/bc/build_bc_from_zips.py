"""Scalable BC-dataset build from ZIPPED Kaggle replay archives — STREAMING + RESUME edition.

Streams episode json straight out of .zip files (no full unzip), processes them in parallel, and
reuses the OFF-BY-ONE-FIXED rows_from_episode from build_bc_dataset.py (so the label fix + the
self-validating tripwire apply unchanged).

BATCH PROCESSING: episodes are processed in fixed-size batches (BC_FLUSH, default 200). Each
batch submits only its episodes to the Pool, collects all results, flushes to a numbered shard
directory, and frees memory BEFORE the next batch starts. This bounds peak RAM to one batch of
results (~30k rows) regardless of total dataset size.

RESUME: each completed shard gets a `.done` marker file. On re-run, existing completed shards
are skipped automatically — only incomplete/missing shards are reprocessed. This means a crash
at episode 4000 (shard 20) can be resumed by re-running the same command; shards 0-19 are reused
and only the remaining episodes are processed.

  PYTHONPATH=. python scripts/bc/build_bc_from_zips.py OUT ZIP1 [ZIP2 ...]
Env: BC_WORKERS     (default 8),  BC_CAP_EPS (default 0 = all),
     BC_FLUSH       (default 200 episodes per batch/shard),
     BC_EP_TIMEOUT  (default 60s per episode).
"""
import os
import sys
import json
import shutil
import zipfile
from multiprocessing import Pool

import numpy as np

# import the FIXED rows_from_episode + shared encoder from the sibling script (scripts/ on path).
# build_bc_dataset reads sys.argv AT IMPORT (MAX_EPS=int(sys.argv[3])), so hide our zip args during import.
# Add both scripts/bc/ (for sibling imports) and project root (for rl.encoder.*)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)
_argv = sys.argv
sys.argv = [_argv[0]]
import build_bc_dataset as B   # B.rows_from_episode (off-by-one fixed), B.enc
sys.argv = _argv

OUT = sys.argv[1]
ZIPS = sys.argv[2:]
WORKERS = int(os.environ.get("BC_WORKERS", "8"))
CAP = int(os.environ.get("BC_CAP_EPS", "0"))
FLUSH = int(os.environ.get("BC_FLUSH", "200"))


def _job(arg):
    """(zip_path, member) -> (rows, labels, attack) for that episode; never raises (bad episode -> [])."""
    zp, name = arg
    try:
        with zipfile.ZipFile(zp) as z:
            ep = json.loads(z.read(name))
        rows, labs, atk = [], [], []
        for r, l, ia in B.rows_from_episode(ep):
            rows.append(r); labs.append(l); atk.append(ia)
        return rows, labs, atk
    except Exception:
        return [], [], []


# ---------------------------------------------------------------------------
#  SHARD I/O
# ---------------------------------------------------------------------------

def _shard_path(shard_dir, idx):
    return os.path.join(shard_dir, f"shard_{idx:04d}")


def _shard_complete(shard_dir, idx):
    """A shard is complete only if its .done marker exists (written AFTER all .npy files)."""
    return os.path.exists(os.path.join(_shard_path(shard_dir, idx), ".done"))


def _shard_row_count(shard_dir, idx):
    """Read the row count from a completed shard without loading large arrays."""
    lab_path = os.path.join(_shard_path(shard_dir, idx), "__labels__.npy")
    lab = np.load(lab_path)
    n = len(lab)
    del lab
    return n


def _flush_shard(shard_dir, shard_idx, rows, labels, attack, int_keys):
    """Write accumulated rows to a numbered shard directory on disk.

    Writes all .npy files first, then a .done marker LAST as an atomic completeness signal.
    If the process crashes mid-flush, the shard lacks .done and will be reprocessed on resume.
    """
    if not rows:
        return 0
    path = _shard_path(shard_dir, shard_idx)
    # clean up any partial shard from a previous crash
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    for k in rows[0].keys():
        dt = np.int32 if (k in int_keys or k == "__group__") else np.float32
        np.save(os.path.join(path, f"{k}.npy"), np.stack([r[k] for r in rows]).astype(dt))
    np.save(os.path.join(path, "__labels__.npy"), np.array(labels, dtype=np.int64))
    np.save(os.path.join(path, "__is_attack__.npy"), np.array(attack, dtype=np.int8))
    # .done marker LAST — the shard is only valid if this file exists
    open(os.path.join(path, ".done"), "w").close()
    return len(rows)


def _discover_shards(shard_dir):
    """Count completed shards in shard_dir (for merge)."""
    if not os.path.isdir(shard_dir):
        return 0
    n = 0
    while _shard_complete(shard_dir, n):
        n += 1
    return n


def _merge_shards(shard_dir, out_path, n_shards):
    """Merge shard directories into final output using memmap. Peak RAM = O(one key × one shard).

    For each key: pre-allocate a memmap output file with the total row count, then copy each shard's
    data into it one at a time. The memmap is flushed and closed before moving to the next key, so
    only one key's worth of shard data is in memory at any point.
    """
    s0_path = _shard_path(shard_dir, 0)
    keys = sorted(f[:-4] for f in os.listdir(s0_path) if f.endswith(".npy"))

    shard_rows = []
    for i in range(n_shards):
        shard_rows.append(_shard_row_count(shard_dir, i))
    total = sum(shard_rows)

    os.makedirs(out_path, exist_ok=True)

    for ki, k in enumerate(keys):
        sample = np.load(os.path.join(s0_path, f"{k}.npy"))
        shape = (total,) + sample.shape[1:]
        dt = sample.dtype
        del sample

        out_mm = np.lib.format.open_memmap(
            os.path.join(out_path, f"{k}.npy"),
            mode="w+", dtype=dt, shape=shape,
        )
        offset = 0
        for i in range(n_shards):
            chunk = np.load(os.path.join(_shard_path(shard_dir, i), f"{k}.npy"))
            out_mm[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
            del chunk
        out_mm.flush()
        del out_mm

        if (ki + 1) % 10 == 0:
            print(f"[bc-zips]   merged {ki + 1}/{len(keys)} keys ...", flush=True)

    return total


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    tasks = []
    for zp in ZIPS:
        with zipfile.ZipFile(zp) as z:
            tasks.extend((zp, n) for n in z.namelist() if n.endswith(".json"))
    if CAP:
        tasks = tasks[:CAP]

    # Split tasks into fixed-size batches (one batch = one shard)
    n_batches = (len(tasks) + FLUSH - 1) // FLUSH
    print(f"[bc-zips] {len(tasks)} episodes from {len(ZIPS)} zip(s); "
          f"workers={WORKERS}, batch_size={FLUSH}, batches={n_batches}", flush=True)

    int_keys = set(B.enc.int_keys)
    shard_dir = (OUT.rstrip(".npz") if OUT.endswith(".npz") else OUT) + "_shards"
    TIMEOUT = float(os.environ.get("BC_EP_TIMEOUT", "60"))

    # ---- RESUME: count already-completed shards ----
    resumed = 0
    resumed_rows = 0
    while _shard_complete(shard_dir, resumed):
        resumed_rows += _shard_row_count(shard_dir, resumed)
        resumed += 1
    if resumed:
        print(f"[bc-zips] RESUME: found {resumed} completed shards ({resumed_rows} rows), "
              f"skipping first {resumed * FLUSH} episodes", flush=True)

    # ---- process remaining batches ----
    total_rows = resumed_rows
    skipped = 0

    # One Pool for all batches (workers stay warm — no re-import per batch)
    with Pool(WORKERS) as pool:
        for batch_idx in range(resumed, n_batches):
            batch_start = batch_idx * FLUSH
            batch_end = min(batch_start + FLUSH, len(tasks))
            batch = tasks[batch_start:batch_end]

            # Clean up any partial shard from a previous crash (no .done marker)
            partial = _shard_path(shard_dir, batch_idx)
            if os.path.isdir(partial) and not _shard_complete(shard_dir, batch_idx):
                shutil.rmtree(partial)

            # Submit ONLY this batch to the pool — bounded memory
            asyncs = [pool.apply_async(_job, (t,)) for t in batch]

            buf_rows, buf_labels, buf_attack = [], [], []
            batch_skipped = 0
            for a in asyncs:
                try:
                    r, l, ia = a.get(timeout=TIMEOUT)
                except Exception:
                    r, l, ia = [], [], []; batch_skipped += 1
                buf_rows.extend(r); buf_labels.extend(l); buf_attack.extend(ia)

            # Flush this batch to disk and free memory
            n = _flush_shard(shard_dir, batch_idx, buf_rows, buf_labels, buf_attack, int_keys)
            total_rows += n
            skipped += batch_skipped
            del buf_rows, buf_labels, buf_attack

            print(f"[bc-zips]   batch {batch_idx + 1}/{n_batches} "
                  f"(eps {batch_start}-{batch_end - 1}) -> {n} rows "
                  f"[total: {total_rows}, skipped: {skipped}]", flush=True)

    if skipped:
        print(f"[bc-zips] skipped {skipped} hung/failed episodes (timeout {TIMEOUT}s each)", flush=True)

    n_shards = _discover_shards(shard_dir)
    print(f"[bc-zips] {total_rows} rows from {len(tasks)} episodes in {n_shards} shards", flush=True)
    if total_rows == 0:
        print("[bc-zips] NO ROWS"); return

    # ---- merge shards into final output (memory-efficient, one key at a time) ----
    print(f"[bc-zips] merging {n_shards} shards ({total_rows} rows) -> {OUT} ...", flush=True)
    if OUT.endswith(".npz"):
        n = _merge_shards(shard_dir, OUT + "_dir", n_shards)
        print(f"[bc-zips] packing into .npz (caution: loads all into RAM) ...", flush=True)
        out = {}
        for f in os.listdir(OUT + "_dir"):
            if f.endswith(".npy"):
                out[f[:-4]] = np.load(os.path.join(OUT + "_dir", f))
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        np.savez(OUT, **out)
        shutil.rmtree(OUT + "_dir", ignore_errors=True)
    else:
        n = _merge_shards(shard_dir, OUT, n_shards)

    # ---- self-check: masked_rate MUST be 0.0 (memmap reads, no extra RAM) ----
    out_dir = OUT if not OUT.endswith(".npz") else OUT + "_dir"
    if not OUT.endswith(".npz"):
        am = np.load(os.path.join(out_dir, "action_mask.npy"), mmap_mode="r")
        lab = np.load(os.path.join(out_dir, "__labels__.npy"), mmap_mode="r")
        masked = float((am[np.arange(n), lab] < 0.5).mean())
        # DEDUP self-check
        dmasked = -1.0
        grp_path = os.path.join(out_dir, "__group__.npy")
        if os.path.exists(grp_path):
            grp = np.load(grp_path, mmap_mode="r")
            A = am.shape[1]
            chunk_sz = 10000
            dmask_hits = 0
            for s in range(0, n, chunk_sz):
                e = min(s + chunk_sz, n)
                am_c = np.array(am[s:e])
                grp_c = np.array(grp[s:e])
                lab_c = np.array(lab[s:e])
                dmask_c = am_c * (grp_c == np.arange(A)[None, :])
                dlab_c = grp_c[np.arange(e - s), lab_c]
                dmask_hits += int((dmask_c[np.arange(e - s), dlab_c] < 0.5).sum())
            dmasked = dmask_hits / n
            del grp
        ia = np.load(os.path.join(out_dir, "__is_attack__.npy"), mmap_mode="r")
        atk_sum = int(np.sum(ia))
        del am, lab, ia
        print(f"[bc-zips] wrote {OUT}: {n} rows, masked_rate={masked:.6f} dedup_masked_rate={dmasked:.6f} "
              f"(both MUST be 0.0); attack_rows={atk_sum} wouldko={'on' if B.WOULD_KO else 'off'}", flush=True)
    else:
        print(f"[bc-zips] wrote {OUT}: {n} rows (.npz self-check skipped)", flush=True)

    # ---- cleanup shard temp directory ----
    shutil.rmtree(shard_dir, ignore_errors=True)
    print("[bc-zips] shards cleaned up. DONE.", flush=True)


if __name__ == "__main__":
    main()
