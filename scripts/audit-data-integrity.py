"""Audit data-integrity over the memo store.

For every record in `meta`, verifies:
  - the path resolves to an existing .md file under the vault,
  - the on-disk body parses with frontmatter,
  - body_hash in store matches sha256[:16] of the on-disk body.

Reports counts + the first N offenders per category so the user can
spot patterns (path-prefix bug, orphan rows, body drift).

Read-only — does NOT modify the store or the .md files.
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from collections import Counter

import frontmatter

from memo.config import Config


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def main() -> int:
    cfg = Config.from_env()
    vault = cfg.vault_path
    db = cfg.db_path

    if not db.is_file():
        print(f"DB missing: {db}")
        return 1

    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT id, path, title, body_hash FROM meta ORDER BY updated DESC"
    ).fetchall()

    total = len(rows)
    bad_path = []
    parse_error = []
    body_empty = []
    hash_mismatch = []
    ok = 0

    prefix_counter: Counter[str] = Counter()

    for r in rows:
        rel = r["path"]
        prefix_counter[rel.split("/", 1)[0]] += 1

        abs_path = vault / rel
        if not abs_path.is_file():
            bad_path.append((r["id"], rel))
            continue

        try:
            text = abs_path.read_text(encoding="utf-8")
            post = frontmatter.loads(text)
            body = post.content or ""
        except Exception as exc:
            parse_error.append((r["id"], rel, str(exc)[:60]))
            continue

        if not body.strip():
            body_empty.append((r["id"], rel))
            continue

        live_hash = _sha256_short(body)
        if live_hash != r["body_hash"]:
            hash_mismatch.append((r["id"], rel, r["body_hash"], live_hash))
            continue

        ok += 1

    con.close()

    # Report
    print("=== memo data-integrity audit ===")
    print(f"vault_path:    {vault}")
    print(f"db_path:       {db}")
    print(f"total records: {total}")
    print()
    print(f"  ✓ healthy:           {ok}")
    print(f"  ✗ bad_path:          {len(bad_path)}   (file does not exist at vault/path)")
    print(f"  ✗ parse_error:       {len(parse_error)}")
    print(f"  ✗ body_empty:        {len(body_empty)}   (file exists but body is blank)")
    print(
        f"  ✗ hash_mismatch:     {len(hash_mismatch)}   (on-disk body diverged from indexed hash)"
    )
    print()

    print("Path-prefix distribution (top-10) — useful to spot the 'Notes/Notes/...' bug:")
    for prefix, n in prefix_counter.most_common(10):
        marker = "  ← suspicious" if prefix == vault.name else ""
        print(f"  {n:5d}  {prefix}/{marker}")
    print()

    def _show(label, items, n=5):
        if not items:
            return
        print(f"--- {label} (showing {min(n, len(items))} of {len(items)}) ---")
        for entry in items[:n]:
            print("  ", entry)
        print()

    _show("bad_path", bad_path, n=10)
    _show("parse_error", parse_error, n=5)
    _show("body_empty", body_empty, n=5)
    _show("hash_mismatch", hash_mismatch, n=5)

    return 0


if __name__ == "__main__":
    sys.exit(main())
