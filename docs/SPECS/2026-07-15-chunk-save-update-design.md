Status: shipped in #32 (feat: MCP client-sampling synthesis + save-time chunk emission + v3.6.0 metadata fix, merged 2026-07-16). `maybe_emit_chunks` is wired into `write_ops.save()` and `update_ops.update()` exactly as designed (commit 3e6f9c2c).

# Chunk emission at save/update time — design

**Date:** 2026-07-15
**Status:** approved (brainstorm), spec 2 of 2 (spec 1: MCP client-sampling synthesis)

## Problem — what is actually missing

The heading-aware chunker (`chunker.py`) is already substantially wired,
contrary to the stale note in `experimental_index.md`:

- `memo reindex` emits chunk records behind `MEMO_CHUNK_INGEST` (default
  off): long bodies split into reference-tier rows (`<parent>_chunk_<seq>`,
  `extra.parent_id`), incremental + rebuild paths, stale-chunk pruning
  (`_reindex_emit_chunks` / `_prune_chunks` in `maintain_ops.py`).
- Explicit search resolves chunk hits back to their parent memory
  (`search_ops.py`); `update()`/`delete()` refuse derived chunk ids;
  deleting a parent hard-deletes its chunks.
- Covered by `tests/test_chunk_ingest.py` and
  `tests/test_search_chunk_parent.py`.

The real gap: **`save()` and `update()` do not emit chunks.** A long
memory saved or edited through MCP (`memo_save`, `memo_update`) gets
section-level retrieval only after the next *manual* `memo reindex` —
nothing runs reindex automatically. Same-session recall of a long saved
document therefore misses section-level matches.

Secondary gap: stale documentation (`experimental_index.md` says "not yet
wired into the reindex pipeline"; the `chunker.py` header says "not covered
by the test suite" — both false today).

## Design

One new helper on the maintain mixin, called from both write paths, all
behind the existing `MEMO_CHUNK_INGEST` flag (no new flags):

```python
def maybe_emit_chunks(self, *, parent_id, parent_rel, title, body, tags, created, updated) -> int:
    """Best-effort chunk emission for one just-written memory."""
```

- Flag off → return 0 immediately (no queries, no imports beyond the flag).
- Flag on → delegate to the existing `_reindex_emit_chunks(...)`, which
  already handles every case: ≤1 chunk → prune stale rows (covers an
  update that shrank the body below the threshold), N chunks → upsert
  changed ones (chunk-level `body_hash` check keeps unchanged sections
  free), then prune removed seqs.
- Wrapped in try/except: **a save/update never fails because chunk
  emission failed** (`_log.warning`; the next reindex heals — chunks are
  derived data).

Call sites:
- `write_ops.save()` — after the successful `store.upsert` +
  graph-entity recording, inside the `not topic_write_superseded` branch,
  with the same values the upsert used (`record_id`, `rel_path`, `title`,
  `content`, `norm_tags`, `created_iso`, `now_iso`).
- `update_ops.update()` — after the `embedding_required` branch's
  `store.upsert`, with the new body/title/tags. Metadata-only updates
  (no body/title change) skip emission: chunks embed body content, and
  retag-only edits don't change it (parity with the skipped re-embed).
  Exception: the existing reindex path refreshes chunk titles/tags on the
  next run — acceptable staleness for derived display metadata.

Embedding goes through `_embed_cached` (inside `_reindex_emit_chunks`),
so re-saves of identical sections are cache hits.

## Docs refresh (same change)

- `experimental_index.md` chunker section: rewrite to reflect wired state
  (reindex + save/update, flag, parent resolution) and remove the "not yet
  wired" claim.
- `chunker.py` header: drop the "not covered by the test suite" line.
- `MEMO_CHUNK_INGEST` flag description: mention save/update emission.

## Non-goals

- No new flags, no sidecar vec table (the reference-tier row design is
  already shipped and tested; a chunk_vec sidecar would be a rewrite,
  not a completion).
- No automatic reindex scheduling.
- No recall-hook changes: chunks are reference-tier and stay excluded
  from ambient recall by design; explicit search benefits.

## Testing

Extend `tests/test_chunk_ingest.py`:
- save long multi-section body with flag on → chunk rows exist
  immediately (no reindex), parent row intact.
- save same body with flag off → no chunk rows.
- save short body with flag on → no chunk rows.
- update that changes a section → that chunk's row refreshed (body_hash
  changes), untouched sections keep their rows.
- update that shrinks the body below threshold → chunk rows pruned.
- chunk emission failure (monkeypatched `_reindex_emit_chunks` raising)
  → save still succeeds and returns the record.
- metadata-only update (tags) → no chunk re-emission (embedding skipped
  parity).

Gates: full pytest, mypy, ruff, `memo eval recall --gate` via pre-push
(flag defaults off — retrieval unchanged for existing corpora).

## Risks

- Save latency for long docs with flag on: one extra batched embed per
  save. Bounded: flag is opt-in, `_embed_cached` dedups, only >2000-char
  bodies chunk.
- Concurrent same-topic save races: emission runs outside the write lock,
  after upsert — a superseded topic write skips emission entirely
  (guarded branch), so no chunk rows for losing writers.
