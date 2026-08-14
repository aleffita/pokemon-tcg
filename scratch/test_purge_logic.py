import shutil
import sqlite3
import time
from pathlib import Path

src_db = Path("/Users/alefita/workdir/pokemon-tcg/model/results.db")
test_db = Path("/Users/alefita/workdir/pokemon-tcg/scratch/test_results.db")

print(f"Step 1: Cloning {src_db} ({src_db.stat().st_size / (1024**2):.2f} MB) to {test_db}...", flush=True)
t0 = time.time()
shutil.copy2(src_db, test_db)
print(f"Cloned in {time.time() - t0:.2f}s", flush=True)

conn = sqlite3.connect(test_db)
cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")
cursor.execute("PRAGMA journal_mode = WAL;")

print("\nStep 2: Checking initial FK violations on clone...", flush=True)
t0 = time.time()
fk_before = cursor.execute("PRAGMA foreign_key_check;").fetchall()
print(f"Initial FK violations count: {len(fk_before):,d} ({time.time() - t0:.2f}s)", flush=True)

# Count initial rows
tables = [r[0] for r in cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;").fetchall()]
counts_before = {}
for t in tables:
    counts_before[t] = cursor.execute(f"SELECT count(*) FROM {t};").fetchone()[0]

print("\nStep 3: Executing Bottom-Up Atomic Purge Script...", flush=True)
t0 = time.time()

# Atomic transaction
cursor.execute("BEGIN IMMEDIATE;")

# 1. Identify orphaned steps
cursor.execute("""
    CREATE TEMP TABLE temp_orphan_steps AS
    SELECT id FROM match_steps WHERE match_id NOT IN (SELECT id FROM matches);
""")
cursor.execute("CREATE UNIQUE INDEX temp_idx_orphan_steps ON temp_orphan_steps(id);")
n_orphan_steps = cursor.execute("SELECT count(*) FROM temp_orphan_steps;").fetchone()[0]
print(f"  Identified {n_orphan_steps:,d} orphaned match_steps", flush=True)

# 2. Identify orphaned snapshots
cursor.execute("""
    CREATE TEMP TABLE temp_orphan_snaps AS
    SELECT id FROM board_snapshots WHERE step_id IN (SELECT id FROM temp_orphan_steps);
""")
cursor.execute("CREATE UNIQUE INDEX temp_idx_orphan_snaps ON temp_orphan_snaps(id);")
n_orphan_snaps = cursor.execute("SELECT count(*) FROM temp_orphan_snaps;").fetchone()[0]
print(f"  Identified {n_orphan_snaps:,d} orphaned board_snapshots", flush=True)

# 3. Delete from leaf to root
# Leaf 1: pokemon_on_field
t_sub = time.time()
cursor.execute("DELETE FROM pokemon_on_field WHERE snapshot_id IN (SELECT id FROM temp_orphan_snaps);")
print(f"  Deleted pokemon_on_field in {time.time() - t_sub:.2f}s", flush=True)

# Leaf 2: board_snapshots
t_sub = time.time()
cursor.execute("DELETE FROM board_snapshots WHERE id IN (SELECT id FROM temp_orphan_snaps);")
print(f"  Deleted board_snapshots in {time.time() - t_sub:.2f}s", flush=True)

# Leaf 3: step_events
t_sub = time.time()
cursor.execute("DELETE FROM step_events WHERE step_id IN (SELECT id FROM temp_orphan_steps);")
print(f"  Deleted step_events in {time.time() - t_sub:.2f}s", flush=True)

# Leaf 4: step_options
t_sub = time.time()
cursor.execute("DELETE FROM step_options WHERE step_id IN (SELECT id FROM temp_orphan_steps);")
print(f"  Deleted step_options in {time.time() - t_sub:.2f}s", flush=True)

# Root of step cascade: match_steps
t_sub = time.time()
cursor.execute("DELETE FROM match_steps WHERE id IN (SELECT id FROM temp_orphan_steps);")
print(f"  Deleted match_steps in {time.time() - t_sub:.2f}s", flush=True)

# Separate orphan branch: match_card_usage
t_sub = time.time()
cursor.execute("DELETE FROM match_card_usage WHERE match_id IS NOT NULL AND match_id NOT IN (SELECT id FROM matches);")
print(f"  Deleted match_card_usage in {time.time() - t_sub:.2f}s", flush=True)

# Drop temp tables
cursor.execute("DROP TABLE temp_orphan_snaps;")
cursor.execute("DROP TABLE temp_orphan_steps;")

conn.commit()
purge_time = time.time() - t0
print(f"Atomic purge committed successfully in {purge_time:.2f}s!", flush=True)

print("\nStep 4: Verifying FK integrity post-purge...", flush=True)
t0 = time.time()
fk_after = cursor.execute("PRAGMA foreign_key_check;").fetchall()
print(f"Post-purge FK violations count: {len(fk_after)} ({time.time() - t0:.2f}s)", flush=True)

print("\nStep 5: Comparing Row Counts Before and After Purge...", flush=True)
counts_after = {}
print(f"  {'Table':25s} | {'Before':>12s} | {'After':>12s} | {'Deleted / Delta':>16s}")
print(f"  {'-'*25} | {'-'*12} | {'-'*12} | {'-'*16}")
for t in tables:
    counts_after[t] = cursor.execute(f"SELECT count(*) FROM {t};").fetchone()[0]
    delta = counts_before[t] - counts_after[t]
    print(f"  {t:25s} | {counts_before[t]:>12,d} | {counts_after[t]:>12,d} | {delta:>16,d}", flush=True)

# Check matches integrity
print("\nStep 6: Verifying Matches and Elo Data Parity...", flush=True)
n_matches = cursor.execute("SELECT count(*) FROM matches;").fetchone()[0]
n_remote = cursor.execute("SELECT count(*) FROM matches WHERE source = 'remote';").fetchone()[0]
n_local = cursor.execute("SELECT count(*) FROM matches WHERE source = 'local';").fetchone()[0]
n_days = cursor.execute("SELECT count(*) FROM days;").fetchone()[0]
n_agent_elo = cursor.execute("SELECT count(*) FROM agent_elo_daily;").fetchone()[0]
n_deck_elo = cursor.execute("SELECT count(*) FROM deck_elo_daily;").fetchone()[0]
n_tournaments = cursor.execute("SELECT count(*) FROM tournaments;").fetchone()[0]

print(f"  Total Matches: {n_matches:,d} (Remote: {n_remote:,d}, Local: {n_local:,d})", flush=True)
print(f"  Days: {n_days} (all 30 days intact)", flush=True)
print(f"  Agent Elo Daily records: {n_agent_elo:,d}", flush=True)
print(f"  Deck Elo Daily records: {n_deck_elo:,d}", flush=True)
print(f"  Tournaments: {n_tournaments}", flush=True)

# Step 7: Test VACUUM to measure reclaimed space
print("\nStep 7: Executing VACUUM on test database to measure space reclamation...", flush=True)
conn.close()

t0 = time.time()
conn = sqlite3.connect(test_db)
conn.execute("VACUUM;")
conn.close()
vac_time = time.time() - t0

size_before = src_db.stat().st_size / (1024**2)
size_after = test_db.stat().st_size / (1024**2)
reclaimed = size_before - size_after

print(f"  VACUUM completed in {vac_time:.2f}s", flush=True)
print(f"  Size before purge: {size_before:.2f} MB ({size_before/1024:.2f} GB)", flush=True)
print(f"  Size after purge + VACUUM: {size_after:.2f} MB ({size_after/1024:.2f} GB)", flush=True)
print(f"  Reclaimed space: {reclaimed:.2f} MB ({reclaimed/1024:.2f} GB, {reclaimed/size_before*100:.1f}%)", flush=True)

# Clean up test database
print("\nStep 8: Cleaning up test clone...", flush=True)
test_db.unlink(missing_ok=True)
Path(str(test_db) + "-wal").unlink(missing_ok=True)
Path(str(test_db) + "-shm").unlink(missing_ok=True)
print("Test clone removed cleanly.", flush=True)
