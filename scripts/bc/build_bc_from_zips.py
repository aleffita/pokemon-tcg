"""Scalable BC-dataset build from ZIPPED Kaggle replay archives — STREAMING + RESUME edition.

Streams episode json straight out of .zip files (no full unzip), processes them in parallel, and
reuses the OFF-BY-ONE-FIXED rows_from_episode from build_bc_dataset.py (so the label fix + the
self-validating tripwire apply unchanged).

BATCH PROCESSING: episodes are processed in fixed-size batches (bc_flush, default 200). Each
batch submits only its episodes to the Pool, collects all results, flushes to a numbered shard
directory, and frees memory BEFORE the next batch starts. This bounds peak RAM to one batch of
results (~30k rows) regardless of total dataset size.

RESUME: each completed shard gets a `.done` marker file. On re-run, existing completed shards
are skipped automatically — only incomplete/missing shards are reprocessed. This means a crash
at episode 4000 (shard 20) can be resumed by re-running the same command; shards 0-19 are reused
and only the remaining episodes are processed.

  uv run tcg-build-bc OUT ZIP1 [ZIP2 ...]
  uv run tcg-build-bc --config configs/smoke.json OUT ZIP1 [ZIP2 ...]
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sys
import json
import shutil
import zipfile
from multiprocessing import Pool

import numpy as np

# import the FIXED rows_from_episode + shared encoder from the sibling module.
# build_bc_dataset reads sys.argv AT IMPORT (MAX_EPS=int(sys.argv[3])), so hide our zip args
# during import.  The project root must be on sys.path for rl.encoder.* and scripts.bc.*
# to resolve (multiprocessing workers inherit this path).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_argv = sys.argv
sys.argv = [_argv[0]]
from scripts.bc import build_bc_dataset as B   # B.rows_from_episode (off-by-one fixed), B.enc
from rl.train_config import load_config
sys.argv = _argv

DATASET_MANIFEST_VERSION = 1
WOULD_KO_HEURISTIC_VERSION = 2


def parse_args():
    p = argparse.ArgumentParser(
        description="Build BC dataset from zipped Kaggle replay archives"
    )
    p.add_argument("out", nargs="?", default=None, help="Output directory (default: data_dir from config + latest date)")
    p.add_argument("zips", nargs="*", default=None, help="Zip file(s) (default: all in replay_zip_dir)")
    p.add_argument("--config", default=None, help="Path to JSON config file")
    p.add_argument("--workers", type=int, default=None,
                   help="Number of worker processes (default: config or 8)")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Max episodes to process (0 = all, default: config or 0)")
    p.add_argument("--flush", type=int, default=None,
                   help="Episodes per batch/shard (default: config or 200)")
    p.add_argument("--ep-timeout", type=float, default=None,
                   help="Seconds per episode before timeout (default: config or 60)")
    return p.parse_args()


def _init_worker(build_config):
    """Apply the authoritative build config inside every spawned worker."""
    B.configure(**build_config)


def _worker_config_snapshot(_=None):
    """Small probe used by functional tests to verify spawn propagation."""
    return B.WOULD_KO, B.WK_NVAR, B.BOTH_SIDES, B.BUILD_SEED


def _job(arg):
    """Build one episode and return rows plus explicit audit metadata."""
    zp, name = arg
    try:
        with zipfile.ZipFile(zp) as z:
            ep = json.loads(z.read(name))
        ep_id = os.path.splitext(os.path.basename(name))[0]
        ep_meta = []
        wk_meta = []
        rows, labs, atk = [], [], []
        for r, l, ia in B.rows_from_episode(
            ep,
            episode_id=ep_id,
            ep_meta=ep_meta,
            would_ko_meta=wk_meta,
        ):
            rows.append(r); labs.append(l); atk.append(ia)
        return rows, labs, atk, ep_meta, wk_meta, {"episode_failures": 0}
    except Exception as exc:
        return [], [], [], [], [], {
            "episode_failures": 1,
            f"episode_failure_{type(exc).__name__}": 1,
        }


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
    if not os.path.exists(lab_path):
        return 0
    lab = np.load(lab_path)
    n = len(lab)
    del lab
    return n


def _flush_shard(
    shard_dir,
    shard_idx,
    rows,
    labels,
    attack,
    int_keys,
    meta=None,
    would_ko_meta=None,
    stats=None,
):
    """Write accumulated rows to a numbered shard directory on disk.

    Writes all .npy files first, then a .done marker LAST as an atomic completeness signal.
    If the process crashes mid-flush, the shard lacks .done and will be reprocessed on resume.
    meta: optional list of episode metadata dicts (D.3).
    """
    path = _shard_path(shard_dir, shard_idx)
    # clean up any partial shard from a previous crash
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)
    if rows:
        for k in rows[0].keys():
            dt = np.int32 if (k in int_keys or k == "__group__") else np.float32
            np.save(
                os.path.join(path, f"{k}.npy"),
                np.stack([r[k] for r in rows]).astype(dt),
            )
        np.save(
            os.path.join(path, "__labels__.npy"),
            np.array(labels, dtype=np.int64),
        )
        np.save(
            os.path.join(path, "__is_attack__.npy"),
            np.array(attack, dtype=np.int8),
        )
    # D.3: episode metadata sidecar
    if meta:
        np.save(
            os.path.join(path, "episode_meta.npy"),
            B.episode_meta_array(meta),
            allow_pickle=False,
        )
        np.save(
            os.path.join(path, "__would_ko_meta__.npy"),
            B.would_ko_meta_array(meta, would_ko_meta or []),
            allow_pickle=False,
        )
    with open(os.path.join(path, "shard_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats or {}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    # .done marker LAST — the shard is only valid if this file exists
    open(os.path.join(path, ".done"), "w").close()
    return len(rows)


def _shard_stats(shard_dir, idx):
    path = os.path.join(_shard_path(shard_dir, idx), "shard_stats.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _merge_counts(target, source):
    for key, value in source.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value
    return target


def _would_ko_stats(decisions):
    stats = Counter()
    stats["eligible_decisions"] = len(decisions)
    for decision in decisions:
        audit = decision["audit"]
        stats["eligible_options"] += int(audit.get("eligible_options", 0))
        stats["requested_trials"] += int(audit.get("requested_trials", 0))
        stats["valid_trials"] += int(audit.get("valid_trials", 0))
        stats["failed_trials"] += int(audit.get("failed_trials", 0))
        for option in audit.get("options", {}).values():
            stats["computed_options"] += int(bool(option.get("computed", False)))
            stats["zero_result_options"] += int(bool(option.get("zero_result", False)))
            stats["failed_options"] += int(not option.get("computed", False))
            stats["partial_options"] += int(
                bool(option.get("computed", False)) and int(option.get("failed_trials", 0)) > 0
            )
        for key in (
            "annotation_failures",
            "search_begin_failures",
            "simulation_failures",
            "search_release_failures",
            "search_end_failures",
            "resolution_limit_failures",
            "invalid_subselect_failures",
            "consistent_determinizations",
            "best_overlap_determinizations",
            "consistent_candidates_total",
            "opponent_synthetic_cards",
            "self_synthetic_cards",
            "resolved_subselects",
            "resolved_subselect_choices",
            "ambiguous_subselects",
        ):
            stats[key] += int(audit.get(key, 0))
    return dict(stats)


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
    episode_meta.npy (D.3 structured dtype) is handled separately via direct concatenation.
    """
    first_nonempty = next(
        (
            _shard_path(shard_dir, index)
            for index in range(n_shards)
            if os.path.exists(
                os.path.join(
                    _shard_path(shard_dir, index), "__labels__.npy"
                )
            )
        ),
        None,
    )
    if first_nonempty is None:
        raise RuntimeError("all completed dataset shards are empty")
    keys = sorted(f[:-4] for f in os.listdir(first_nonempty)
                  if f.endswith(".npy")
                  and f[:-4] not in ("episode_meta", "__would_ko_meta__"))

    shard_rows = []
    for i in range(n_shards):
        shard_rows.append(_shard_row_count(shard_dir, i))
    total = sum(shard_rows)

    os.makedirs(out_path, exist_ok=True)

    for ki, k in enumerate(keys):
        sample = np.load(os.path.join(first_nonempty, f"{k}.npy"))
        shape = (total,) + sample.shape[1:]
        dt = sample.dtype
        del sample

        out_mm = np.lib.format.open_memmap(
            os.path.join(out_path, f"{k}.npy"),
            mode="w+", dtype=dt, shape=shape,
        )
        offset = 0
        for i in range(n_shards):
            chunk_path = os.path.join(_shard_path(shard_dir, i), f"{k}.npy")
            if not os.path.exists(chunk_path):
                continue
            chunk = np.load(chunk_path)
            out_mm[offset:offset + len(chunk)] = chunk
            offset += len(chunk)
            del chunk
        out_mm.flush()
        del out_mm

        if (ki + 1) % 10 == 0:
            print(f"[bc-zips]   merged {ki + 1}/{len(keys)} keys ...", flush=True)

    # D.3: merge episode metadata (structured dtype, direct concat — no memmap needed)
    meta_parts = []
    for i in range(n_shards):
        p = os.path.join(_shard_path(shard_dir, i), "episode_meta.npy")
        if os.path.exists(p):
            meta_parts.append(np.load(p))
    if meta_parts:
        meta_merged = np.concatenate(meta_parts, axis=0)
        os.makedirs(out_path, exist_ok=True)
        np.save(os.path.join(out_path, "episode_meta.npy"), meta_merged, allow_pickle=False)
        print(f"[bc-zips]   merged episode_meta ({len(meta_merged)} rows)", flush=True)

    wk_parts = []
    row_offset = 0
    for i, shard_rows_count in enumerate(shard_rows):
        p = os.path.join(_shard_path(shard_dir, i), "__would_ko_meta__.npy")
        if os.path.exists(p):
            part = np.load(p)
            if len(part):
                part = part.copy()
                valid_row = part["row_start"] >= 0
                part["row_start"][valid_row] += row_offset
                wk_parts.append(part)
        row_offset += shard_rows_count
    wk_merged = (
        np.concatenate(wk_parts, axis=0)
        if wk_parts
        else np.empty(0, dtype=B.WOULD_KO_META_DTYPE)
    )
    np.save(
        os.path.join(out_path, "__would_ko_meta__.npy"),
        wk_merged,
        allow_pickle=False,
    )
    print(
        f"[bc-zips]   merged would_ko_meta ({len(wk_merged)} attack options)",
        flush=True,
    )

    return total


def _sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_tasks(zip_paths):
    """Collect replay members once, deduplicating identical episode IDs."""
    tasks = []
    seen = {}
    duplicate_count = 0
    sources = []
    for zip_path in zip_paths:
        zip_path = Path(zip_path)
        source = {
            "path": str(zip_path),
            "size_bytes": zip_path.stat().st_size,
            "sha256": _sha256(zip_path),
            "json_members": 0,
        }
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if not info.filename.endswith(".json"):
                    continue
                source["json_members"] += 1
                episode_id = os.path.splitext(os.path.basename(info.filename))[0]
                identity = (int(info.CRC), int(info.file_size))
                previous = seen.get(episode_id)
                if previous is None:
                    seen[episode_id] = identity
                    tasks.append((zip_path, info.filename))
                elif previous == identity:
                    duplicate_count += 1
                else:
                    raise ValueError(
                        f"episode id collision with different content: {episode_id}"
                    )
        sources.append(source)
    return tasks, sources, {
        "strategy": "episode_id_with_crc_and_size_collision_guard",
        "unique_episodes": len(tasks),
        "identical_duplicates_removed": duplicate_count,
    }


def _write_json(path, payload):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _build_contract(cfg, config_path, sources, dedup):
    relevant_config = {
        "bc_would_ko": bool(cfg.bc_would_ko),
        "bc_wk_nvar": int(cfg.bc_wk_nvar),
        "bc_both_sides": bool(cfg.bc_both_sides),
        "seed": int(cfg.seed),
        "max_episodes": int(cfg.max_episodes),
        "bc_flush": int(cfg.bc_flush),
        "self_aliases": sorted(cfg.bc_self_aliases),
    }
    payload = {
        "manifest_version": DATASET_MANIFEST_VERSION,
        "builder": "scripts/bc/build_bc_from_zips.py",
        "would_ko_heuristic_version": WOULD_KO_HEURISTIC_VERSION,
        "config_source": str(config_path),
        "config": relevant_config,
        "sources": sources,
        "deduplication": dedup,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload["build_fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _prepare_resume_manifest(shard_dir, contract):
    os.makedirs(shard_dir, exist_ok=True)
    path = os.path.join(shard_dir, "build_contract.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            old = json.load(fh)
        if old.get("build_fingerprint") != contract["build_fingerprint"]:
            raise RuntimeError(
                "existing shards were built with a different config/source contract; "
                "refusing an unsafe resume"
            )
    else:
        if _shard_complete(shard_dir, 0):
            raise RuntimeError(
                "existing shards have no build contract; refusing an unauditable resume"
            )
        _write_json(path, contract)


def _validate_would_ko_dataset(out_dir, enabled, stats):
    """Validate would-KO execution without treating a valid zero as a failure."""
    if not enabled:
        return "disabled"
    eligible = int(stats.get("eligible_options", 0))
    computed = int(stats.get("computed_options", 0))
    valid_trials = int(stats.get("valid_trials", 0))
    if eligible <= 0:
        raise RuntimeError(
            "bc_would_ko=true but the selected corpus produced no eligible attack options"
        )
    if computed <= 0 or valid_trials <= 0:
        raise RuntimeError(
            "bc_would_ko=true but no would-KO option was successfully computed"
        )

    wk_path = os.path.join(out_dir, "__would_ko_meta__.npy")
    opt_path = os.path.join(out_dir, "opt_attr.npy")
    if not os.path.exists(wk_path) or not os.path.exists(opt_path):
        raise RuntimeError("would-KO metadata or opt_attr is missing from the built dataset")
    wk = np.load(wk_path, mmap_mode="r")
    opt = np.load(opt_path, mmap_mode="r")
    if int(np.count_nonzero(wk["computed"])) != computed:
        raise RuntimeError("would-KO manifest counters disagree with __would_ko_meta__.npy")

    from rl.encoder.enc_constants import OPT_WK
    for record in wk:
        row_start = int(record["row_start"])
        row_count = int(record["row_count"])
        option_index = int(record["option_index"])
        if row_start < 0 or row_count <= 0:
            raise RuntimeError("would-KO record is not linked to dataset rows")
        if option_index < 0 or option_index >= opt.shape[1]:
            continue
        values = np.asarray(
            opt[row_start:row_start + row_count, option_index, OPT_WK:OPT_WK + 3],
            dtype=np.float32,
        )
        if not np.isfinite(values).all():
            raise RuntimeError("would-KO features contain NaN or infinity")
        if (values < 0).any() or (values > 1).any():
            raise RuntimeError("would-KO normalized features are outside [0, 1]")
        if not bool(record["computed"]) and np.any(values != 0):
            raise RuntimeError("failed would-KO computation contains non-zero features")
        if bool(record["zero_result"]) and np.any(values != 0):
            raise RuntimeError("would-KO zero-result audit disagrees with encoded features")
    del wk, opt
    return "computed"


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config: CLI > config file > defaults
    cli = {}
    if args.workers is not None:
        cli["bc_workers"] = args.workers
    if args.max_episodes is not None:
        cli["max_episodes"] = args.max_episodes
    if args.flush is not None:
        cli["bc_flush"] = args.flush
    if args.ep_timeout is not None:
        cli["bc_ep_timeout"] = args.ep_timeout
    cfg = load_config(cli_overrides=cli, config_path=args.config)

    # Resolve output dir and zips from config defaults
    OUT = args.out or os.path.join(cfg.data_dir, f"bc_{datetime.now().strftime('%Y_%m_%d')}")
    if args.zips:
        ZIPS = [Path(z) for z in args.zips]
    else:
        zip_dir = Path(cfg.replay_zip_dir)
        ZIPS = sorted(zip_dir.glob("*.zip"))
        if not ZIPS:
            print(f"No zip files found in {zip_dir}")
            return

    WORKERS = cfg.bc_workers
    CAP = cfg.max_episodes
    FLUSH = cfg.bc_flush
    TIMEOUT = float(cfg.bc_ep_timeout)
    # Configure the coordinator and pass the exact same authoritative values to
    # every spawned worker through Pool(initializer=...).
    build_config = {
        "bc_would_ko": cfg.bc_would_ko,
        "bc_wk_nvar": cfg.bc_wk_nvar,
        "bc_both_sides": cfg.bc_both_sides,
        "seed": cfg.seed,
        "self_aliases": cfg.bc_self_aliases,
    }
    B.configure(**build_config)

    tasks, source_manifest, dedup_manifest = _collect_tasks(ZIPS)
    if CAP:
        tasks = tasks[:CAP]
    dedup_manifest["selected_episodes"] = len(tasks)

    # Split tasks into fixed-size batches (one batch = one shard)
    n_batches = (len(tasks) + FLUSH - 1) // FLUSH
    print(f"[bc-zips] {len(tasks)} episodes from {len(ZIPS)} zip(s); "
          f"workers={WORKERS}, batch_size={FLUSH}, batches={n_batches}", flush=True)

    int_keys = set(B.enc.int_keys)
    shard_dir = (OUT.removesuffix(".npz") if OUT.endswith(".npz") else OUT) + "_shards"
    config_source = args.config or getattr(cfg, "_config_path", None) or "configs/train_config.json"
    build_contract = _build_contract(
        cfg,
        config_source,
        source_manifest,
        dedup_manifest,
    )
    _prepare_resume_manifest(shard_dir, build_contract)

    # ---- RESUME: count already-completed shards ----
    resumed = 0
    resumed_rows = 0
    aggregate_stats = {}
    while _shard_complete(shard_dir, resumed):
        resumed_rows += _shard_row_count(shard_dir, resumed)
        _merge_counts(aggregate_stats, _shard_stats(shard_dir, resumed))
        resumed += 1
    if resumed:
        print(f"[bc-zips] RESUME: found {resumed} completed shards ({resumed_rows} rows), "
              f"skipping first {resumed * FLUSH} episodes", flush=True)

    # ---- process remaining batches ----
    total_rows = resumed_rows
    skipped = 0

    # One Pool for all batches (workers stay warm — no re-import per batch)
    with Pool(WORKERS, initializer=_init_worker, initargs=(build_config,)) as pool:
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

            buf_rows, buf_labels, buf_attack, buf_meta, buf_wk_meta = [], [], [], [], []
            batch_stats = {}
            batch_skipped = 0
            for a in asyncs:
                try:
                    r, l, ia, m, wk, job_stats = a.get(timeout=TIMEOUT)
                except Exception:
                    r, l, ia, m, wk = [], [], [], [], []
                    job_stats = {"episode_failures": 1, "episode_timeout_or_worker_failures": 1}
                    batch_skipped += 1
                buf_rows.extend(r); buf_labels.extend(l); buf_attack.extend(ia); buf_meta.extend(m)
                buf_wk_meta.extend(wk)
                _merge_counts(batch_stats, job_stats)

            _merge_counts(batch_stats, _would_ko_stats(buf_wk_meta))

            # Flush this batch to disk and free memory
            n = _flush_shard(shard_dir, batch_idx, buf_rows, buf_labels, buf_attack, int_keys,
                             meta=buf_meta, would_ko_meta=buf_wk_meta, stats=batch_stats)
            total_rows += n
            skipped += batch_skipped
            _merge_counts(aggregate_stats, batch_stats)
            del buf_rows, buf_labels, buf_attack, buf_meta, buf_wk_meta

            print(f"[bc-zips]   batch {batch_idx + 1}/{n_batches} "
                  f"(eps {batch_start}-{batch_end - 1}) -> {n} rows "
                  f"[total: {total_rows}, skipped: {skipped}]", flush=True)

    if skipped:
        print(f"[bc-zips] skipped {skipped} hung/failed episodes (timeout {TIMEOUT}s each)", flush=True)

    n_shards = _discover_shards(shard_dir)
    print(f"[bc-zips] {total_rows} rows from {len(tasks)} episodes in {n_shards} shards", flush=True)
    if total_rows == 0:
        print("[bc-zips] NO ROWS"); return
    if cfg.bc_would_ko:
        if int(aggregate_stats.get("eligible_options", 0)) <= 0:
            raise RuntimeError(
                "bc_would_ko=true but no eligible attack option was observed; "
                "the dataset cannot prove the configured feature ran"
            )
        if int(aggregate_stats.get("computed_options", 0)) <= 0:
            raise RuntimeError(
                "bc_would_ko=true but every eligible option failed computation"
            )

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
        would_ko_status = _validate_would_ko_dataset(
            out_dir,
            bool(cfg.bc_would_ko),
            aggregate_stats,
        )
        print(f"[bc-zips] wrote {OUT}: {n} rows, masked_rate={masked:.6f} dedup_masked_rate={dmasked:.6f} "
              f"(both MUST be 0.0); attack_rows={atk_sum} "
              f"would_ko_status={would_ko_status} "
              f"computed_options={int(aggregate_stats.get('computed_options', 0))} "
              f"zero_result_options={int(aggregate_stats.get('zero_result_options', 0))} "
              f"failed_options={int(aggregate_stats.get('failed_options', 0))}", flush=True)
    else:
        would_ko_status = (
            "computed"
            if cfg.bc_would_ko and int(aggregate_stats.get("computed_options", 0)) > 0
            else "disabled"
        )
        print(f"[bc-zips] wrote {OUT}: {n} rows (.npz self-check skipped)", flush=True)

    completed_manifest = {
        **build_contract,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output": {
            "path": str(OUT),
            "rows": int(n),
            "shards": int(n_shards),
            "dtype_policy": "builder_native_arrays",
        },
        "would_ko": {
            "enabled": bool(cfg.bc_would_ko),
            "status": would_ko_status,
            "n_var": int(cfg.bc_wk_nvar),
            "heuristic_version": WOULD_KO_HEURISTIC_VERSION,
            "counters": aggregate_stats,
            "zero_semantics": (
                "computed=true and zero_result=true means a valid zero; "
                "computed=false with failed_trials>0 means computation failure"
            ),
        },
        "self_identity": {
            "aliases": sorted(B.SELF_ALIASES),
            "storage": "episode_meta.npy:is_self",
        },
        "hidden_replay_data": {
            "deck_storage": (
                "episode_meta.npy:player_deck_hash,opponent_deck_hash"
            ),
            "used_by_would_ko": False,
            "reason": (
                "live inference cannot observe the opponent's submitted deck"
            ),
        },
    }
    manifest_path = (
        os.path.join(OUT, "dataset_manifest.json")
        if not OUT.endswith(".npz")
        else f"{OUT}.manifest.json"
    )
    _write_json(manifest_path, completed_manifest)

    # ---- cleanup shard temp directory ----
    shutil.rmtree(shard_dir, ignore_errors=True)
    print("[bc-zips] shards cleaned up. DONE.", flush=True)


if __name__ == "__main__":
    main()
