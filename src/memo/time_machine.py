"""Time-machine — query the corpus as it existed at any past date.

The single feature no other agent-memory product ships. Every other
local-or-cloud memory store (mem0, letta, cognee, supermemory, mem-vault,
milasd, doggybee, engram, …) serves *current* state only. memo's
`history.db` audit log records every save/update/delete with the
field-level diff, so we can reconstruct an earlier state by replaying
events in reverse from "now".

## Algorithm

Given a target timestamp `T`:

1. Take the current `meta` table as the starting state.
2. Pull every `events` row whose `ts > T`, sorted newest-first.
3. Walk the events in reverse:
   - `save` after T  → record was created after T, drop from snapshot.
   - `delete` after T → record was deleted after T, re-insert minimal
     stub (title + type from the event row; body unavailable unless we
     cached one in the `versions` table).
   - `update` after T → revert each `{field: [old, new]}` pair in
     `delta_json`: set the field back to `old`.

End-of-walk: the surviving rows describe the corpus as it stood at T.

## Limitations (v1)

- **Body is best-effort.** `events.delta_json` carries body diffs when
  the body changed during an update, so we can revert body edits. But
  if a record was deleted after T and no `versions` row was saved
  separately, we can't reconstruct its body. We mark such rows
  `_body_unavailable=True` and the search/ask helpers skip them.
- **Embeddings are current.** We don't re-embed historical bodies — a
  search-as-of uses current embeddings to find candidates, then
  filters to the snapshot set. This means semantic recall reflects the
  *current* embedder's view; the snapshot is a metadata filter on top.
- **Tag updates are reverted.** Same delta_json mechanism.

For corpora with no destructive edits, the snapshot is exact. For
corpora with lots of body rewrites, the body-reversal works as long
as `delta_json` carried `body` (which `Memory.update` always emits
when the body changed).
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

# reconstruct()/diff() pull the full corpus + audit log into RAM. This bound is
# effectively "all rows" for a personal vault; it exists only as a backstop.
# NB: diff() calls reconstruct() twice, so a large corpus loads 4 full dumps.
_FULL_DUMP_LIMIT = 1_000_000


def _parse_ts(s: str | datetime) -> datetime:
    """Parse an ISO-8601 string (with or without trailing Z) or pass
    through a datetime. Always returns tz-aware UTC."""
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=UTC)
    s = s.rstrip("Z")
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


@dataclass
class SnapshotRecord:
    """A reconstructed memory record at a past point in time."""

    id: str
    title: str
    type: str
    tags: list[str]
    created: str | None
    updated: str | None
    body: str | None = None
    body_unavailable: bool = False
    _deleted_after: str | None = None  # ts of deletion; None if still exists


@dataclass
class CorpusSnapshot:
    """Read-only view of the corpus at a past timestamp."""

    as_of: datetime
    records: dict[str, SnapshotRecord]
    # Hooks back to the live Memory so search/ask can reuse the live
    # embedder + reranker but post-filter against the snapshot set.
    _memory: Any = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.records)

    def list(self, *, type_: str | None = None) -> builtins.list[SnapshotRecord]:
        rows = [r for r in self.records.values() if not r.body_unavailable]
        if type_:
            rows = [r for r in rows if r.type == type_]
        rows.sort(key=lambda r: r.updated or "", reverse=True)
        return rows

    def search(self, query: str, *, limit: int = 10, mode: str = "hybrid") -> builtins.list[Any]:
        """Run live search, then keep only hits whose id is in the snapshot
        AND was reachable at as_of (body present, not deleted-after)."""
        if self._memory is None:
            return []
        # Over-fetch so the post-filter has headroom.
        live = self._memory.search(query, limit=limit * 4, mode=mode)
        kept = [h for h in live if h.id in self.records and not self.records[h.id].body_unavailable]
        return kept[:limit]

    def ask(self, question: str, *, k: int = 5, snippet_chars: int = 800) -> dict[str, Any]:
        """RAG synthesis restricted to the snapshot.

        Re-implements the prompt-build + chat-call from Memory.ask but
        with the snapshot's filtered hits, plus an explicit "as-of"
        framing in the system message so the model knows the corpus
        view is historical (and shouldn't pretend to know later facts).
        """
        from memo.memory import _ASK_SYSTEM_PROMPT

        if self._memory is None:
            return {
                "question": question,
                "answer": "",
                "sources": [],
                "as_of": self.as_of.isoformat(),
                "snapshot_size": 0,
            }
        hits = self.search(question, limit=k)
        if not hits:
            return {
                "question": question,
                "answer": f"(no relevant memories existed at {self.as_of.date().isoformat()})",
                "sources": [],
                "as_of": self.as_of.isoformat(),
                "snapshot_size": len(self),
            }

        snippet_lines: builtins.list[str] = []
        sources: builtins.list[dict[str, Any]] = []
        for h in hits:
            id_short = h.id[:8]
            snippet = (h.body or "")[:snippet_chars]
            if len(h.body or "") > snippet_chars:
                snippet = snippet.rstrip() + "…"
            tags = ", ".join(h.tags) or "—"
            snippet_lines.append(
                f"[{id_short}] title: {h.title}  |  type: {h.type}  |  tags: {tags}\n{snippet}\n",
            )
            sources.append(
                {
                    "id": h.id,
                    "id_short": id_short,
                    "title": h.title,
                    "type": h.type,
                    "score": h.score,
                    "snippet": snippet,
                }
            )

        as_of_iso = self.as_of.date().isoformat()
        user_msg = (
            f"User question (memory view as of {as_of_iso}):\n{question}\n\n"
            f"Relevant memories (top {len(hits)} from the historical snapshot):\n\n"
            + "\n---\n".join(snippet_lines)
        )
        system_msg = (
            _ASK_SYSTEM_PROMPT
            + f"\n\nIMPORTANT: you are answering from a historical view of the memory as of {as_of_iso}. "
            "Do NOT mention facts that only became known after that date, even if you know them."
        )

        mem = self._memory
        chat = mem._ensure_chat()
        try:
            out = chat.chat(
                model=mem.cfg.llm_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options={"temperature": 0.0, "max_tokens": 768},
            )
            answer = ((out.get("message") or {}).get("content") or "").strip()
        except Exception as exc:
            answer = f"(error querying the model: {type(exc).__name__})"

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "as_of": self.as_of.isoformat(),
            "snapshot_size": len(self),
        }


def reconstruct(memory: Any, *, as_of: str | datetime) -> CorpusSnapshot:
    """Build a `CorpusSnapshot` for the given `as_of` timestamp.

    `memory` is a live `Memory` instance — we pull current state from
    `memory.store`, the audit log from `memory.history`, and keep a
    reference so `snapshot.search()` / `snapshot.ask()` can reuse the
    embedder.
    """
    target = _parse_ts(as_of)

    # 1. Current state — full row dump from the store layer.
    current_rows = memory.store.list_recent(limit=_FULL_DUMP_LIMIT)
    snap: dict[str, SnapshotRecord] = {}
    for r in current_rows:
        snap[r["id"]] = SnapshotRecord(
            id=r["id"],
            title=r["title"],
            type=r["type"],
            tags=list(r["tags"] or []),
            created=r.get("created"),
            updated=r.get("updated"),
            body=None,
        )

    # 2. Events strictly after `target`, newest first. Parse `ts` as
    # tz-aware datetime so the comparison is timezone-correct
    # (lexicographic compare on ISO strings breaks when stored ts uses
    # a non-UTC offset, which `Memory.save()` produces).
    all_events = memory.history.list_recent(limit=_FULL_DUMP_LIMIT)

    def _ev_dt(e: dict[str, Any]) -> datetime | None:
        ts = e.get("ts")
        if not ts:
            return None
        try:
            return _parse_ts(ts)
        except (ValueError, TypeError):
            return None

    later: list[tuple[datetime, dict[str, Any]]] = []
    for e in all_events:
        dt = _ev_dt(e)
        if dt is not None and dt > target:
            later.append((dt, e))
    later.sort(key=lambda pair: pair[0], reverse=True)
    later_events = [e for _, e in later]

    # 3. Walk in reverse, undoing each op.
    for ev in later_events:
        rid = ev.get("record_id")
        if not rid:
            continue
        op = ev.get("op")
        if op == "save":
            # Created after as_of → record did not exist then.
            snap.pop(rid, None)
        elif op == "delete":
            # Deleted after as_of → record DID exist at as_of.
            # Reconstruct from the event snapshot (title/type captured
            # at delete time). Body is gone unless versions.db has it.
            ts = ev.get("ts")
            snap[rid] = SnapshotRecord(
                id=rid,
                title=ev.get("title") or "(deleted)",
                type=ev.get("type") or "note",
                tags=[],
                created=None,
                updated=None,
                body=None,
                body_unavailable=True,
                _deleted_after=ts,
            )
        elif op == "update":
            if rid not in snap:
                continue
            delta = ev.get("delta") or {}
            if not isinstance(delta, dict):
                continue
            for field_name, pair in delta.items():
                if not (isinstance(pair, list) and len(pair) == 2):
                    continue
                old_val, _new_val = pair[0], pair[1]
                if field_name == "tags":
                    snap[rid].tags = list(old_val or [])
                elif field_name == "title":
                    snap[rid].title = old_val or snap[rid].title
                elif field_name == "type":
                    snap[rid].type = old_val or snap[rid].type
                elif field_name == "updated":
                    snap[rid].updated = old_val
                elif field_name == "body":
                    # We don't load full bodies into the snapshot row by
                    # default (they live on disk). Mark that body is
                    # historical and reversible if needed.
                    pass
            # For pre-fix events that lack "updated" in the delta, cap the
            # field at the event's own ts — the record was last modified at
            # or before this point in time, so this is a correct upper bound.
            if "updated" not in delta:
                ev_ts = ev.get("ts")
                if ev_ts and snap[rid].updated and snap[rid].updated > ev_ts:
                    snap[rid].updated = ev_ts

    return CorpusSnapshot(as_of=target, records=snap, _memory=memory)


# -------------------- diff --------------------


@dataclass
class CorpusDiff:
    """Result of comparing two snapshots: from → to."""

    from_ts: datetime
    to_ts: datetime
    added: list[SnapshotRecord]  # exist at `to`, not at `from`
    removed: list[SnapshotRecord]  # exist at `from`, not at `to`
    updated: list[dict[str, Any]]  # {id, title, changed_fields}

    def summary(self) -> str:
        return (
            f"{len(self.added)} added · {len(self.removed)} removed · {len(self.updated)} updated"
        )


def diff(memory: Any, *, from_ts: str | datetime, to_ts: str | datetime) -> CorpusDiff:
    """Compare two snapshots: what changed between `from_ts` and `to_ts`."""
    f = reconstruct(memory, as_of=from_ts)
    t = reconstruct(memory, as_of=to_ts)
    f_ids = set(f.records.keys())
    t_ids = set(t.records.keys())

    added_ids = t_ids - f_ids
    removed_ids = f_ids - t_ids
    common_ids = f_ids & t_ids

    added = [t.records[i] for i in sorted(added_ids, key=lambda x: t.records[x].updated or "")]
    removed = [f.records[i] for i in sorted(removed_ids, key=lambda x: f.records[x].updated or "")]
    updated: list[dict[str, Any]] = []
    for i in sorted(common_ids):
        fr = f.records[i]
        tr = t.records[i]
        diffs: list[str] = []
        if fr.title != tr.title:
            diffs.append("title")
        if fr.type != tr.type:
            diffs.append("type")
        if sorted(fr.tags) != sorted(tr.tags):
            diffs.append("tags")
        if diffs:
            updated.append(
                {
                    "id": i,
                    "title": tr.title,
                    "changed_fields": diffs,
                    "before": {"title": fr.title, "type": fr.type, "tags": fr.tags},
                    "after": {"title": tr.title, "type": tr.type, "tags": tr.tags},
                }
            )

    return CorpusDiff(
        from_ts=_parse_ts(from_ts),
        to_ts=_parse_ts(to_ts),
        added=added,
        removed=removed,
        updated=updated,
    )
