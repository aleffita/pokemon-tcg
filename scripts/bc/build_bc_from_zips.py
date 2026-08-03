"""Build a BC dataset (one Parquet file per day) from zipped Kaggle replay archives.

Streams episode json straight out of .zip files (no full unzip), processes them in parallel
batches, and reuses the OFF-BY-ONE-FIXED rows_from_episode from build_bc_dataset.py (so the
label fix + the self-validating tripwire + the replay-log aux RL targets apply unchanged).

OUTPUT: one Parquet file per day (``data/bc_data/YYYY-MM-DD.parquet``, zstd-compressed) plus a
sidecar manifest (``YYYY-MM-DD.manifest.json``), and a row registered in the SQLite results
catalog (``rl.results_db.ResultsDB``, table ``datasets``). This REPLACES the previous multi-.npy
shard-directory format; the shard/merge/resume machinery from the .npy-era writer is gone --
resume is now day-level (skip a zip whose date is already registered), not row-level.

BATCH PROCESSING: episodes are processed in fixed-size batches (bc_flush, default 200). Each
batch is submitted to a worker Pool, collected, converted to one Arrow row-group, and written
straight to the day's ParquetWriter -- bounding peak RAM to one batch of decoded rows regardless
of the day's total size.

  uv run tcg-build-bc ZIP                       # -> data/bc_data/<date>.parquet
  uv run tcg-build-bc --output DIR ZIP           # override output directory
  uv run tcg-build-bc --all                      # every zip in replay_zip_dir
  uv run tcg-build-bc --all --resume             # skip zips already registered in the catalog
  uv run tcg-build-bc --latest                   # only the newest zip in replay_zip_dir
  uv run tcg-build-bc --config configs/smoke.json ZIP
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

# import the FIXED rows_from_episode + shared encoder from the sibling module.
# build_bc_dataset reads sys.argv AT IMPORT (MAX_EPS=int(sys.argv[3])), so hide our own args
# during import. The project root must be on sys.path for rl.encoder.* and scripts.bc.* to
# resolve (multiprocessing workers inherit this path).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_argv = sys.argv
sys.argv = [_argv[0]]
from scripts.bc import build_bc_dataset as B  # noqa: E402
from rl.train_config import load_config  # noqa: E402
from rl.results_db import ResultsDB  # noqa: E402
sys.argv = _argv

# Bumped from the old .npy-shard writer's DATASET_MANIFEST_VERSION=1: the Parquet layout, the
# sequential-metadata columns (decision_id/substep/terminal/reward), and the aux_* RL targets are
# a different on-disk contract, not a compatible successor.
DATASET_SCHEMA_VERSION = 3
AUX_TARGETS_VERSION = 1
WOULD_KO_HEURISTIC_VERSION = 2
# No canonical architecture/token-schema version registry exists yet in rl/encoder (CLAUDE.md
# Phase A -- "one canonical token schema" -- is not implemented). This string identifies the
# encoder generation the dataset was built with until that registry exists.
ENCODER_VERSION = "token_v2"


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a per-day Parquet BC dataset from zipped Kaggle replay archives"
    )
    p.add_argument(
        "paths", nargs="*", default=None,
        help="Zip file(s). DEPRECATED 2-arg form `OUT_DIR ZIP` is still accepted.",
    )
    p.add_argument("--output", "--out", dest="output", default=None,
                    help="Output directory (default: config data_dir, e.g. data/bc_data)")
    p.add_argument("--all", action="store_true",
                    help="Build every zip in replay_zip_dir")
    p.add_argument("--latest", action="store_true",
                    help="Build only the lexicographically latest zip in replay_zip_dir "
                         "(used by scripts/daily_pipeline.py)")
    p.add_argument("--resume", action="store_true",
                    help="Skip zips whose date is already registered in the results catalog")
    p.add_argument("--config", default=None, help="Path to JSON config file")
    p.add_argument("--db", default="model/results.db", help="Path to the SQLite results catalog")
    p.add_argument("--workers", type=int, default=None,
                    help="Number of worker processes (default: config or 8)")
    p.add_argument("--max-episodes", type=int, default=None,
                    help="Max episodes to process per zip (0 = all, default: config or 0)")
    p.add_argument("--flush", type=int, default=None,
                    help="Episodes per batch/row-group (default: config or 200)")
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
        for row, label, is_attack in B.rows_from_episode(
            ep,
            episode_id=ep_id,
            ep_meta=ep_meta,
            would_ko_meta=wk_meta,
        ):
            rows.append(row)
            labs.append(label)
            atk.append(is_attack)
        return rows, labs, atk, ep_meta, wk_meta, {"episode_failures": 0}
    except Exception as exc:
        import traceback
        print(f"[bc-zips][job-error {name}] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise


def _merge_counts(target, source):
    for key, value in source.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = target.get(key, 0) + value
    return target


def _would_ko_stats(decisions):
    """Aggregate would-KO audit counters across a batch of decisions (attack rows only)."""
    stats = {}

    def bump(key, amount=1):
        stats[key] = stats.get(key, 0) + amount

    bump("eligible_decisions", len(decisions))
    for decision in decisions:
        audit = decision["audit"]
        bump("eligible_options", int(audit.get("eligible_options", 0)))
        bump("requested_trials", int(audit.get("requested_trials", 0)))
        bump("valid_trials", int(audit.get("valid_trials", 0)))
        bump("failed_trials", int(audit.get("failed_trials", 0)))
        for option in audit.get("options", {}).values():
            bump("computed_options", int(bool(option.get("computed", False))))
            bump("zero_result_options", int(bool(option.get("zero_result", False))))
            bump("failed_options", int(not option.get("computed", False)))
    return stats


def _validate_would_ko_stats(enabled, stats):
    """Validate would-KO execution without treating a valid zero as a failure.

    Simplified from the .npy-era ``_validate_would_ko_dataset``: this checks the aggregate
    counters only (no per-row opt_attr scan over a merged array -- there is no such merged
    array in the Parquet writer's streaming design). The per-row would_ko feature values
    themselves are still produced by the encoder exactly as before; this is a build-time
    sanity gate, not a full re-audit.
    """
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
    return "computed"


def _discover_shards(shard_dir):  # kept only as a no-op alias for older callers/tests
    return 0


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


# ---------------------------------------------------------------------------
#  ROWS -> ARROW
# ---------------------------------------------------------------------------

_META_INT8 = (
    "side", "new_episode", "terminal", "outcome", "is_self",
    "aux_ko", "aux_terminal", "aux_valid",
)
_META_INT32 = ("step_id", "decision_id")
_META_FLOAT32 = ("reward", "aux_prize_delta", "aux_return")
_META_STR = ("player_name", "opponent_name", "player_deck_hash", "opponent_deck_hash")


def _rows_to_table(rows, labels, attack, meta, day_id, int_keys, enc_shapes):
    """Convert one batch's decoded rows + episode metadata into one pyarrow.Table.

    Scalar-shaped encoder fields (shape == (1,)) become plain scalar columns; every other
    fixed-shape field (e.g. opt_attr's (MAX_OPTIONS, OPT_STRUCT+N_ATTACK_FX)) is flattened
    row-major into a FixedSizeList column -- far cheaper to write/read than nested variable
    lists, since the encoder's shapes are fully deterministic. The flattened shape is returned
    alongside the table so the caller can persist it once in the day's manifest for a loader to
    reshape by.
    """
    n = len(rows)
    columns = {}
    shape_meta = {}
    keys = list(rows[0].keys())
    action_mask_shape = enc_shapes["action_mask"]
    for k in keys:
        if k == "__group__":
            shape, out_name = action_mask_shape, "opt_group"
        else:
            # Derive shape from data if the encoder didn't publish it (e.g. new
            # meta-feature columns added after the writer was frozen).
            shape = enc_shapes.get(k)
            if shape is None:
                sample = np.asarray(rows[0][k])
                shape = sample.shape if sample.ndim > 0 else (1,)
            out_name = k
        dt_np = np.int32 if (k in int_keys or k == "__group__") else np.float32
        pa_type = pa.int32() if dt_np == np.int32 else pa.float32()
        stacked = np.stack([r[k] for r in rows]).astype(dt_np)
        if shape == (1,):
            columns[out_name] = pa.array(stacked.reshape(n), type=pa_type)
            shape_meta[out_name] = "scalar"
        else:
            flat_len = int(np.prod(shape))
            flat = stacked.reshape(n, flat_len)
            values = pa.array(flat.reshape(-1), type=pa_type)
            columns[out_name] = pa.FixedSizeListArray.from_arrays(values, flat_len)
            shape_meta[out_name] = list(shape)

    columns["y"] = pa.array(np.asarray(labels, dtype=np.int32), type=pa.int32())
    columns["is_attack"] = pa.array(np.asarray(attack, dtype=np.int8), type=pa.int8())

    columns["episode_id"] = pa.array([int(m["episode_id"]) for m in meta], type=pa.int64())
    for k in _META_INT8:
        columns[k] = pa.array([int(bool(m[k]) if isinstance(m[k], bool) else m[k]) for m in meta], type=pa.int8())
    for k in _META_INT32:
        columns[k] = pa.array([int(m[k]) for m in meta], type=pa.int32())
    for k in _META_FLOAT32:
        columns[k] = pa.array([float(m[k]) for m in meta], type=pa.float32())
    for k in _META_STR:
        columns[k] = pa.array([str(m[k]) for m in meta], type=pa.string())
    columns["substep"] = pa.array([int(m["substep"]) for m in meta], type=pa.int8())
    columns["day_id"] = pa.array([int(day_id)] * n, type=pa.int32())

    table = pa.table(columns)
    return table, shape_meta


# ---------------------------------------------------------------------------
#  ONE DAY
# ---------------------------------------------------------------------------

def _build_one_day(zip_path, output_dir, cfg, db, *, workers, flush, timeout, max_episodes):
    zip_path = Path(zip_path)
    date = zip_path.stem
    os.makedirs(output_dir, exist_ok=True)
    out_path = Path(output_dir) / f"{date}.parquet"
    manifest_path = Path(output_dir) / f"{date}.manifest.json"
    tmp_path = Path(output_dir) / f"{date}.parquet.tmp"

    build_config = {
        "bc_would_ko": cfg.bc_would_ko,
        "bc_wk_nvar": cfg.bc_wk_nvar,
        "bc_both_sides": cfg.bc_both_sides,
        "seed": cfg.seed,
        "self_aliases": cfg.bc_self_aliases,
    }
    B.configure(**build_config)

    tasks, source_manifest, dedup_manifest = _collect_tasks([zip_path])
    if max_episodes:
        tasks = tasks[:max_episodes]
    if not tasks:
        print(f"[bc-zips] {date}: no episodes found in {zip_path}, skipping", flush=True)
        return None

    day_id = db.register_day(date=date, source_zip=str(zip_path))

    int_keys = set(B.enc.int_keys)
    enc_shapes = dict(B.enc.shapes)
    n_batches = (len(tasks) + flush - 1) // flush
    print(f"[bc-zips] {date}: {len(tasks)} episodes, workers={workers}, "
          f"batch_size={flush}, batches={n_batches}", flush=True)

    writer = None
    shape_meta = None
    total_rows = 0
    episodes_used = 0
    episode_failures = 0
    aggregate_wk = {}
    agents_observed = set()

    with Progress(
        TextColumn(f"[bold cyan]{date}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f} episodes"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress, Pool(workers, initializer=_init_worker, initargs=(build_config,)) as pool:
        bar = progress.add_task("build", total=len(tasks))
        try:
            for batch_idx in range(n_batches):
                batch = tasks[batch_idx * flush: min((batch_idx + 1) * flush, len(tasks))]
                asyncs = [pool.apply_async(_job, (t,)) for t in batch]

                buf_rows, buf_labels, buf_attack, buf_meta, buf_wk = [], [], [], [], []
                batch_stats = {}
                for a in asyncs:
                    try:
                        rows, labels, attacks, meta, wk, job_stats = a.get(timeout=timeout)
                    except Exception as exc:
                        import traceback
                        print(f"[bc-zips][worker-error] {type(exc).__name__}: {exc}", flush=True)
                        traceback.print_exc()
                        raise
                    if rows:
                        episodes_used += 1
                    episode_failures += int(job_stats.get("episode_failures", 0))
                    buf_rows.extend(rows)
                    buf_labels.extend(labels)
                    buf_attack.extend(attacks)
                    buf_meta.extend(meta)
                    buf_wk.extend(wk)
                    _merge_counts(batch_stats, job_stats)
                _merge_counts(batch_stats, _would_ko_stats(buf_wk))
                _merge_counts(aggregate_wk, batch_stats)

                for m in buf_meta:
                    agents_observed.add(str(m.get("player_name", "")))
                    agents_observed.add(str(m.get("opponent_name", "")))

                if buf_rows:
                    table, this_shape_meta = _rows_to_table(
                        buf_rows, buf_labels, buf_attack, buf_meta, day_id, int_keys, enc_shapes
                    )
                    if writer is None:
                        shape_meta = this_shape_meta
                        writer = pq.ParquetWriter(
                            str(tmp_path), table.schema,
                            compression="zstd", compression_level=3,
                        )
                    writer.write_table(table)
                    total_rows += len(buf_rows)

                progress.update(bar, advance=len(batch))
                del buf_rows, buf_labels, buf_attack, buf_meta, buf_wk
        finally:
            if writer is not None:
                writer.close()

    if total_rows == 0:
        tmp_path.unlink(missing_ok=True)
        print(f"[bc-zips] {date}: NO ROWS, skipping dataset registration", flush=True)
        return None

    would_ko_status = _validate_would_ko_stats(bool(cfg.bc_would_ko), aggregate_wk)

    os.replace(tmp_path, out_path)
    parquet_sha256 = _sha256(out_path)

    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_zip": str(zip_path),
        "source_sha256": source_manifest[0]["sha256"],
        "parquet_path": str(out_path),
        "parquet_sha256": parquet_sha256,
        "rows": int(total_rows),
        "episodes": int(episodes_used),
        "episode_failures": int(episode_failures),
        "date": date,
        "aux_targets": True,
        "aux_targets_version": AUX_TARGETS_VERSION,
        "encoder_version": ENCODER_VERSION,
        "agents_observed": sorted(a for a in agents_observed if a),
        "written_at": datetime.now(timezone.utc).isoformat(),
        "build_config": {
            "bc_would_ko": bool(cfg.bc_would_ko),
            "bc_wk_nvar": int(cfg.bc_wk_nvar),
            "bc_both_sides": bool(cfg.bc_both_sides),
            "seed": int(cfg.seed),
            "bc_self_aliases": sorted(cfg.bc_self_aliases),
        },
        "would_ko": {
            "enabled": bool(cfg.bc_would_ko),
            "status": would_ko_status,
            "heuristic_version": WOULD_KO_HEURISTIC_VERSION,
            "counters": aggregate_wk,
        },
        "deduplication": dedup_manifest,
        "column_shapes": shape_meta,
    }
    _write_json(manifest_path, manifest)

    dataset_id = db.register_dataset(
        day_id=day_id,
        path=str(out_path),
        schema_version=DATASET_SCHEMA_VERSION,
        rows=int(total_rows),
        sha256=parquet_sha256,
        aux_targets=True,
    )
    db.conn.commit()
    print(
        f"[bc-zips] {date}: wrote {out_path} ({total_rows} rows, {episodes_used} episodes, "
        f"{episode_failures} episode failures) dataset_id={dataset_id} "
        f"would_ko={would_ko_status}",
        flush=True,
    )
    return {"date": date, "rows": total_rows, "episodes": episodes_used, "path": str(out_path)}


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def _resolve_zip_paths(args, cfg):
    """Return (zip_paths, legacy_output_dir_or_None)."""
    if args.all or args.latest:
        if args.paths:
            raise ValueError("--all/--latest cannot be combined with explicit zip paths")
        zip_dir = Path(cfg.replay_zip_dir)
        zips = sorted(zip_dir.glob("*.zip"))
        if not zips:
            raise SystemExit(f"No zip files found in {zip_dir}")
        return ([zips[-1]] if args.latest else zips), None
    if not args.paths:
        raise SystemExit("Provide a zip path, or use --all")
    if (
        len(args.paths) == 2
        and not args.paths[0].endswith(".zip")
        and args.paths[1].endswith(".zip")
    ):
        print(
            "[bc-zips] WARNING: the `OUT_DIR ZIP` positional form is deprecated; "
            "use `tcg-build-bc [--output DIR] ZIP`",
            flush=True,
        )
        return [Path(args.paths[1])], args.paths[0]
    return [Path(p) for p in args.paths], None


def main():
    args = parse_args()

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

    zips, legacy_out = _resolve_zip_paths(args, cfg)
    output_dir = args.output or legacy_out or cfg.data_dir

    db = ResultsDB(args.db)
    try:
        results = []
        skipped_resume = 0
        for zip_path in zips:
            date = Path(zip_path).stem
            if args.resume:
                existing = db.conn.execute(
                    "SELECT ds.path FROM datasets ds JOIN days d ON d.id = ds.day_id "
                    "WHERE d.date = ?",
                    (date,),
                ).fetchone()
                if existing is not None:
                    print(f"[bc-zips] {date}: already registered ({existing['path']}), "
                          f"skipping (--resume)", flush=True)
                    skipped_resume += 1
                    continue
            result = _build_one_day(
                zip_path, output_dir, cfg, db,
                workers=cfg.bc_workers,
                flush=cfg.bc_flush,
                timeout=float(cfg.bc_ep_timeout),
                max_episodes=cfg.max_episodes,
            )
            if result:
                results.append(result)
        if len(zips) > 1 or args.all:
            total_rows = sum(r["rows"] for r in results)
            print(
                f"[bc-zips] DONE: {len(results)} day(s) built "
                f"({skipped_resume} skipped via --resume), {total_rows} total rows",
                flush=True,
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()
