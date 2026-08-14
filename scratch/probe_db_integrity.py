import sqlite3
import time
from pathlib import Path

db_path = Path("/Users/alefita/workdir/pokemon-tcg/model/results.db")
print(f"Opening DB: {db_path} ({db_path.stat().st_size / (1024*1024):.2f} MB)", flush=True)

conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
cursor = conn.cursor()

# 1. Table row counts
print("\n=== Table Row Counts ===", flush=True)
tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()]
table_counts = {}
for t in tables:
    cnt = cursor.execute(f"SELECT count(*) FROM {t};").fetchone()[0]
    table_counts[t] = cnt
    print(f"  {t:25s}: {cnt:>12,d}", flush=True)

# 2. Matches breakdown
print("\n=== Matches Breakdown ===", flush=True)
match_breakdown = cursor.execute("SELECT source, count(*) FROM matches GROUP BY source;").fetchall()
for src, cnt in match_breakdown:
    print(f"  Source '{src}': {cnt:>12,d}", flush=True)

# 3. Direct FK check per table
print("\n=== PRAGMA foreign_key_check per table ===", flush=True)
for t in tables:
    t0 = time.time()
    violations = cursor.execute(f"PRAGMA foreign_key_check('{t}');").fetchall()
    dt = time.time() - t0
    v_count = len(violations)
    if v_count > 0:
        # Sample first 3 violations
        sample = violations[:3]
        print(f"  [VIOLATION] Table '{t}': {v_count:,d} violations ({dt:.2f}s) -> Sample: {sample}", flush=True)
    else:
        print(f"  [OK] Table '{t}': 0 violations ({dt:.2f}s)", flush=True)

# 4. Detailed orphan counts
print("\n=== Detailed Orphan Breakdown ===", flush=True)

# match_steps
t0 = time.time()
orphan_steps = cursor.execute("""
    SELECT count(*) FROM match_steps WHERE match_id NOT IN (SELECT id FROM matches);
""").fetchone()[0]
print(f"  match_steps orphaned (match_id not in matches): {orphan_steps:,d} ({time.time()-t0:.2f}s)", flush=True)

# match_card_usage
t0 = time.time()
orphan_mcu_match = cursor.execute("""
    SELECT count(*) FROM match_card_usage WHERE match_id IS NOT NULL AND match_id NOT IN (SELECT id FROM matches);
""").fetchone()[0]
orphan_mcu_part = cursor.execute("""
    SELECT count(*) FROM match_card_usage WHERE participant_id IS NOT NULL AND participant_id NOT IN (SELECT id FROM match_participants);
""").fetchone()[0]
print(f"  match_card_usage orphaned match_id: {orphan_mcu_match:,d}", flush=True)
print(f"  match_card_usage orphaned participant_id: {orphan_mcu_part:,d} ({time.time()-t0:.2f}s)", flush=True)

# match_participants
t0 = time.time()
orphan_mp = cursor.execute("""
    SELECT count(*) FROM match_participants WHERE match_id NOT IN (SELECT id FROM matches);
""").fetchone()[0]
print(f"  match_participants orphaned match_id: {orphan_mp:,d} ({time.time()-t0:.2f}s)", flush=True)

# Child tables of match_steps
t0 = time.time()
orphan_so = cursor.execute("""
    SELECT count(*) FROM step_options WHERE step_id NOT IN (SELECT id FROM match_steps);
""").fetchone()[0]
print(f"  step_options orphaned step_id: {orphan_so:,d} ({time.time()-t0:.2f}s)", flush=True)

t0 = time.time()
orphan_se = cursor.execute("""
    SELECT count(*) FROM step_events WHERE step_id NOT IN (SELECT id FROM match_steps);
""").fetchone()[0]
print(f"  step_events orphaned step_id: {orphan_se:,d} ({time.time()-t0:.2f}s)", flush=True)

t0 = time.time()
orphan_bs = cursor.execute("""
    SELECT count(*) FROM board_snapshots WHERE step_id NOT IN (SELECT id FROM match_steps);
""").fetchone()[0]
print(f"  board_snapshots orphaned step_id: {orphan_bs:,d} ({time.time()-t0:.2f}s)", flush=True)

t0 = time.time()
orphan_pof = cursor.execute("""
    SELECT count(*) FROM pokemon_on_field WHERE snapshot_id NOT IN (SELECT id FROM board_snapshots);
""").fetchone()[0]
print(f"  pokemon_on_field orphaned snapshot_id: {orphan_pof:,d} ({time.time()-t0:.2f}s)", flush=True)

# Check child rows that belong to the orphaned match_steps
print("\n=== Child rows belonging to orphaned match_steps ===", flush=True)
t0 = time.time()
so_on_orphan_steps = cursor.execute("""
    SELECT count(*) FROM step_options WHERE step_id IN (
        SELECT id FROM match_steps WHERE match_id NOT IN (SELECT id FROM matches)
    );
""").fetchone()[0]
print(f"  step_options on orphaned match_steps: {so_on_orphan_steps:,d} ({time.time()-t0:.2f}s)", flush=True)

t0 = time.time()
se_on_orphan_steps = cursor.execute("""
    SELECT count(*) FROM step_events WHERE step_id IN (
        SELECT id FROM match_steps WHERE match_id NOT IN (SELECT id FROM matches)
    );
""").fetchone()[0]
print(f"  step_events on orphaned match_steps: {se_on_orphan_steps:,d} ({time.time()-t0:.2f}s)", flush=True)

t0 = time.time()
bs_on_orphan_steps = cursor.execute("""
    SELECT count(*) FROM board_snapshots WHERE step_id IN (
        SELECT id FROM match_steps WHERE match_id NOT IN (SELECT id FROM matches)
    );
""").fetchone()[0]
print(f"  board_snapshots on orphaned match_steps: {bs_on_orphan_steps:,d} ({time.time()-t0:.2f}s)", flush=True)

t0 = time.time()
pof_on_orphan_steps = cursor.execute("""
    SELECT count(*) FROM pokemon_on_field WHERE snapshot_id IN (
        SELECT id FROM board_snapshots WHERE step_id IN (
            SELECT id FROM match_steps WHERE match_id NOT IN (SELECT id FROM matches)
        )
    );
""").fetchone()[0]
print(f"  pokemon_on_field on orphaned match_steps: {pof_on_orphan_steps:,d} ({time.time()-t0:.2f}s)", flush=True)

conn.close()
print("\nProbe completed successfully.", flush=True)
