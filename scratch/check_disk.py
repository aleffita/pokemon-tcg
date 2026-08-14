import shutil
import sqlite3
import time
from pathlib import Path

# Check disk space
db_path = Path("/Users/alefita/workdir/pokemon-tcg/model/results.db")
stat = shutil.disk_usage(db_path.parent)
print(f"Total disk space: {stat.total / (1024**3):.2f} GB", flush=True)
print(f"Used disk space:  {stat.used / (1024**3):.2f} GB", flush=True)
print(f"Free disk space:  {stat.free / (1024**3):.2f} GB", flush=True)
