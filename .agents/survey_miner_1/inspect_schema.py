import sqlite3

DB_PATH = "file:/Users/alefita/workdir/pokemon-tcg/model/results.db?mode=ro"

def main():
    conn = sqlite3.connect(DB_PATH, uri=True)
    cursor = conn.cursor()
    cursor.execute("SELECT type, name, tbl_name, sql FROM sqlite_master WHERE type='index'")
    indices = cursor.fetchall()
    print(f"Total indices: {len(indices)}", flush=True)
    for t, name, tbl, sql in indices:
        print(f"Index on {tbl}: {name} -> {sql}", flush=True)
    conn.close()

if __name__ == "__main__":
    main()
