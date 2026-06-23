"""Cross-machine signal export/import (F3 of memo-sync).

memo's `.md` memories are the source of truth and sync between machines via git.
The user-signal tables (`access`, `memory_health`, `source_feedback`) are PRIMARY
data that live only in the local rebuildable `memvec.db` — a fresh clone + reindex
on another Mac restores every memory but zero signal. These helpers snapshot the
signal to `signal/*.json` next to the memories (so git carries it) and merge a
peer's snapshot back by stable memory id.

Merge is idempotent on re-pull (see `VecStore.merge_signal`):
  - access:          access_count = max(local, remote); last_accessed = max
  - memory_health:   the row with the newer `updated_at` wins
  - source_feedback: union by id

`source_feedback_vec` embeddings are NOT synced; they are re-derivable from
`query_text` and rebuilt by a future re-embed pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from memo.config import Config
    from memo.store import VecStore

SIGNAL_SCHEMA = "memo.sync.signal.v1"

# table name -> json filename
_FILES = {
    "access": "access.json",
    "memory_health": "memory_health.json",
    "source_feedback": "source_feedback.json",
}


def signal_dir_for(cfg: Config) -> Path:
    """The `signal/` directory — sibling of the memories dir under the repo root."""
    return cfg.memory_dir.parent / "signal"


def export_signal(store: VecStore, signal_dir: Path) -> dict[str, int]:
    """Dump every signal table to `signal_dir/<table>.json`. Returns row counts."""
    signal_dir.mkdir(parents=True, exist_ok=True)
    dump = store.dump_signal()
    counts: dict[str, int] = {}
    for table, filename in _FILES.items():
        rows = dump.get(table, [])
        payload = {"schema": SIGNAL_SCHEMA, "rows": rows}
        (signal_dir / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        counts[table] = len(rows)
    return counts


def import_signal(store: VecStore, signal_dir: Path) -> dict[str, int]:
    """Merge a peer's `signal/*.json` snapshot into the local store. Returns counts.

    Missing files are treated as empty (no-op for that table) so a partial or
    first-ever snapshot imports cleanly.
    """
    payload: dict[str, list[dict]] = {}
    for table, filename in _FILES.items():
        path = signal_dir / filename
        if not path.exists():
            payload[table] = []
            continue
        doc = json.loads(path.read_text())
        if doc.get("schema") != SIGNAL_SCHEMA:
            raise ValueError(f"{path}: unexpected schema {doc.get('schema')!r}, want {SIGNAL_SCHEMA!r}")
        payload[table] = doc.get("rows") or []
    return store.merge_signal(payload)
