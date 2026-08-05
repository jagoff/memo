"""Audit data-integrity over the memo store.

For every active record in `meta`, verifies:
  - the path resolves to an existing .md file under the configured memory
    directory or to the declared source of any `vault-ingest*` reference,
  - the on-disk body parses with frontmatter,
  - body_hash in store matches sha256[:16] of the on-disk body.

Reports counts + the first N offenders per category so the user can
spot patterns (path-prefix bug, orphan rows, body drift).

Read-only — does NOT modify the store or the .md files.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import frontmatter

from memo.chunker import DEFAULT_TARGET_CHARS, chunk_markdown
from memo.config import Config
from memo.flags import flag_bool
from memo.redact import sanitize_memory_input

_CHUNK_PATH_RE = re.compile(r"^(?P<path>.+)#chunk-(?P<seq>\d+)$")


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def main() -> int:
    cfg = Config.from_env()
    memory_dir = cfg.memory_dir
    db = cfg.db_path

    if not db.is_file():
        print(f"DB missing: {db}")
        return 1

    con = sqlite3.connect(db)
    try:
        con.row_factory = sqlite3.Row
        has_fts = (
            con.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'fts'"
            ).fetchone()
            is not None
        )
        meta_columns = {row["name"] for row in con.execute("PRAGMA table_info(meta)").fetchall()}
        if has_fts:
            if "deleted_at" in meta_columns:
                rows = con.execute(
                    "SELECT id, path, title, body_hash, extra_json, "
                    "(SELECT body FROM fts WHERE fts.id = meta.id LIMIT 1) AS indexed_body "
                    "FROM meta WHERE meta.deleted_at IS NULL ORDER BY updated DESC"
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT id, path, title, body_hash, extra_json, "
                    "(SELECT body FROM fts WHERE fts.id = meta.id LIMIT 1) AS indexed_body "
                    "FROM meta ORDER BY updated DESC"
                ).fetchall()
        else:
            if "deleted_at" in meta_columns:
                rows = con.execute(
                    "SELECT id, path, title, body_hash, extra_json, NULL AS indexed_body "
                    "FROM meta WHERE meta.deleted_at IS NULL ORDER BY updated DESC"
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT id, path, title, body_hash, extra_json, NULL AS indexed_body "
                    "FROM meta ORDER BY updated DESC"
                ).fetchall()
    finally:
        con.close()

    total = len(rows)
    bad_path = []
    parse_error = []
    body_empty = []
    hash_mismatch = []
    ok = 0

    prefix_counter: Counter[str] = Counter()
    memory_root = memory_dir.resolve()
    parsed_cache: dict[Path, tuple[str, list[dict[str, object]]]] = {}
    entropy = flag_bool("MEMO_REDACT_ENTROPY")

    for r in rows:
        rel = r["path"]
        prefix_counter[rel.split("/", 1)[0]] += 1

        chunk_match = _CHUNK_PATH_RE.fullmatch(rel)
        rel_path = Path(chunk_match.group("path") if chunk_match else rel)
        try:
            extra = json.loads(r["extra_json"] or "{}")
        except (TypeError, json.JSONDecodeError) as exc:
            parse_error.append((r["id"], rel, f"invalid extra_json: {exc}"[:60]))
            continue
        if not isinstance(extra, dict):
            parse_error.append((r["id"], rel, "extra_json is not an object"))
            continue

        source = str(extra.get("source") or "")
        is_vault_ingest = source == "vault-ingest" or source.startswith("vault-ingest-")
        if is_vault_ingest:
            source_value = extra.get("abs_path")
            parent_path = str(extra.get("parent_path") or rel_path)
            vault_label = str(extra.get("vault") or "").strip("/")
            source_path = Path(str(source_value or ""))
            logical_path = str(rel_path).replace("\\", "/")
            if (
                not source_value
                or not source_path.is_absolute()
                or parent_path != logical_path
                or (vault_label and not logical_path.startswith(f"{vault_label}/"))
            ):
                bad_path.append((r["id"], rel))
                continue
            source_relative = logical_path[len(vault_label) + 1 :] if vault_label else logical_path
            expected_suffix = Path(source_relative).parts
            actual_suffix = source_path.parts[-len(expected_suffix) :] if expected_suffix else ()
            if tuple(part.casefold() for part in actual_suffix) != tuple(
                part.casefold() for part in expected_suffix
            ):
                bad_path.append((r["id"], rel))
                continue
            abs_path = source_path.resolve()
            if not abs_path.is_file():
                bad_path.append((r["id"], rel))
                continue
        else:
            abs_path = (memory_root / rel_path).resolve()
            if rel_path.is_absolute() or not abs_path.is_relative_to(memory_root):
                bad_path.append((r["id"], rel))
                continue
            if not abs_path.is_file():
                bad_path.append((r["id"], rel))
                continue

        try:
            if is_vault_ingest:
                body = r["indexed_body"]
                if not isinstance(body, str):
                    raise ValueError("vault-ingest row is missing indexed FTS body")
            else:
                cached = parsed_cache.get(abs_path)
                if cached is None:
                    text = abs_path.read_text(encoding="utf-8")
                    post = frontmatter.loads(text)
                    raw_body = post.content or ""
                    body = sanitize_memory_input(
                        content=raw_body,
                        entropy=entropy,
                        allow_empty_content=True,
                    ).content
                    chunks = chunk_markdown(body, target_chars=DEFAULT_TARGET_CHARS)
                    cached = (body, chunks)
                    parsed_cache[abs_path] = cached
                body, chunks = cached
                if chunk_match:
                    chunk_seq = int(chunk_match.group("seq"))
                    body = next(str(chunk["body"]) for chunk in chunks if chunk["seq"] == chunk_seq)
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

    # Report
    print("=== memo data-integrity audit ===")
    print(f"memory_dir:    {memory_dir}")
    print(f"db_path:       {db}")
    print(f"total records: {total}")
    print()
    print(f"  ✓ healthy:           {ok}")
    print(f"  ✗ bad_path:          {len(bad_path)}   (file is missing or escapes memory_dir)")
    print(f"  ✗ parse_error:       {len(parse_error)}")
    print(f"  ✗ body_empty:        {len(body_empty)}   (file exists but body is blank)")
    print(
        f"  ✗ hash_mismatch:     {len(hash_mismatch)}   (on-disk body diverged from indexed hash)"
    )
    print()

    print("Path-prefix distribution (top-10) — useful to spot the 'Notes/Notes/...' bug:")
    for prefix, n in prefix_counter.most_common(10):
        marker = "  ← suspicious" if prefix == memory_dir.name else ""
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

    return 1 if bad_path or parse_error or body_empty or hash_mismatch else 0


if __name__ == "__main__":
    sys.exit(main())
