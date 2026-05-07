"""Clean up records left over from the buggy ingest path.

Drops every row in the store whose `path` does NOT start with the
configured `memory_subdir` (i.e. anything ingested via `memo ingest`
under the buggy code that prefixed paths with `<vault>.name/`).

After this script runs, only curated memorias (created via `memo save`)
remain. Re-run `memo ingest <vault>` with the fixed code to repopulate
the ingested records with correct paths.

Read-then-write — uses `VecStore.delete()` to keep `meta`, `vec`, `fts`
in sync. Idempotent.
"""

from __future__ import annotations

import sqlite3
import sys

from memo.config import Config
from memo.store import VecStore


def main() -> int:
    cfg = Config.from_env()
    keep_prefix = cfg.memory_subdir.rstrip("/") + "/"

    con = sqlite3.connect(cfg.db_path)
    rows = con.execute("SELECT id, path FROM meta").fetchall()
    con.close()

    to_delete = [(rid, p) for (rid, p) in rows if not p.startswith(keep_prefix)]

    print(f"total records:        {len(rows)}")
    print(f"keep_prefix:          {keep_prefix}")
    print(f"records to delete:    {len(to_delete)}")
    print()
    if not to_delete:
        print("nothing to do")
        return 0

    print("Examples (first 5):")
    for rid, p in to_delete[:5]:
        print(f"  - {rid[:8]}  {p}")
    print()

    print("Deleting…")
    store = VecStore(cfg.db_path, dims=cfg.embedder_dims)
    deleted = 0
    for rid, _p in to_delete:
        if store.delete(rid):
            deleted += 1
    print(f"deleted: {deleted}")

    # Reality check
    con = sqlite3.connect(cfg.db_path)
    after = con.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    con.close()
    print(f"remaining records:    {after}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
