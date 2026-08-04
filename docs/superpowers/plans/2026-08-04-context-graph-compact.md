# Graph-aware `memo_context_pack` source compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When `memo_context_pack` retrieves multiple distinct memories that share a rare entity, drop the redundant ones before they ever occupy a current/supporting/stale bucket slot — cutting pack size without changing which memories get retrieved or how `context_pack.py`'s bucketing/truncation works.

**Architecture:** One new wrapper function, `compact_hits_by_entity_overlap` (`src/memo/context_compact.py`), adapts `memory.search()` hit *objects* to the minimal dict shape already consumed by the existing, already-tested `compact_by_entity_overlap` (`src/memo/chat/graph_compact.py`, shipped in PR #186), delegates the IDF/overlap/grouping math to it unchanged, then maps surviving rows back to the original hit objects by a synthetic index (order-preserving, not `id`-based). It's wired into `server_context_pack.py`'s `memo_context_pack` tool right before the `build_context_pack(...)` call, gated by a new default-off `MEMO_CONTEXT_GRAPH_COMPACT` flag registered the standard way in `src/memo/flags_search.py`.

**Tech Stack:** Python 3.11+, pytest, existing `memo.*` package (no new dependencies; imports one function from `memo.chat.graph_compact`).

## Global Constraints

- Default OFF (`MEMO_CONTEXT_GRAPH_COMPACT` defaults to `False`) — ships inert.
- Config is a normal registered `FlagSpec` pair in `src/memo/flags_search.py` (typed accessors `flag_bool`/`flag_float`) — NOT the chat env-only pattern (`context_pack` isn't part of the `MEMO_CHAT_*` family).
- Fail-open: any exception during entity/IDF lookup or hit-to-dict adaptation returns the input `hits` list unchanged. Never raises, never blocks the tool response.
- No change to retrieval, ranking, or `context_pack.py`'s bucketing/truncation logic — only pre-bucket compaction of an already-ranked hit list.
- No new eval harness. Ships default-off with unit + integration test coverage; eval-gated graduation is an explicit follow-up (see spec's Out of scope).
- Spec: `docs/superpowers/specs/2026-08-04-context-graph-compact-design.md`.

---

### Task 1: `compact_hits_by_entity_overlap` wrapper function

**Files:**
- Create: `src/memo/context_compact.py`
- Test: `tests/test_context_compact.py`

**Interfaces:**
- Consumes: `memo.chat.graph_compact.compact_by_entity_overlap` (existing, shipped PR #186).
- Produces: `compact_hits_by_entity_overlap(hits: list[Any], memory: Any, *, min_idf_overlap: float, min_group_size: int = 2) -> list[Any]`. Later tasks call this exact signature. Returns hit objects unchanged (no field mutation) — collapsed hits are simply absent from the returned list, in original relative order.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_context_compact.py`:

```python
from types import SimpleNamespace

from memo.context_compact import compact_hits_by_entity_overlap


class _FakeGraph:
    def __init__(self, entities: dict[str, list[dict]], dfs: dict[str, float], total: int) -> None:
        self._entities = entities
        self._dfs = dfs
        self._total = total

    def memory_entities(self, memory_id: str) -> list[dict]:
        return self._entities.get(memory_id, [])

    def total_indexed_memories(self) -> int:
        return self._total

    def entity_doc_freqs(self, names) -> dict[str, float]:
        return {n: self._dfs[n] for n in names if n in self._dfs}


class _FakeMemory:
    def __init__(self, graph) -> None:
        self.graph = graph


def _hit(hid: str, score: float, title: str = "") -> SimpleNamespace:
    return SimpleNamespace(id=hid, title=title or f"T{hid}", score=score)


def test_ubiquitous_entity_overlap_does_not_collapse() -> None:
    entities = {
        "a": [{"name": "memo"}, {"name": "topic-x"}],
        "b": [{"name": "memo"}, {"name": "topic-y"}],
    }
    dfs = {"memo": 9.0, "topic-x": 1.0, "topic-y": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("a", 1.0), _hit("b", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert [h.id for h in out] == ["a", "b"]


def test_rare_shared_entity_collapses() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}]}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("c", 1.0), _hit("d", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert [h.id for h in out] == ["c"]


def test_no_entities_is_noop() -> None:
    mem = _FakeMemory(_FakeGraph({}, {}, total=10))
    hits = [_hit("e", 1.0), _hit("f", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert [h.id for h in out] == ["e", "f"]


def test_group_below_min_group_size_stays_uncollapsed() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}], "g": []}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    hits = [_hit("c", 1.0), _hit("d", 0.9), _hit("g", 0.5)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5, min_group_size=3)

    assert [h.id for h in out] == ["c", "d", "g"]


def test_lookup_failure_returns_hits_unchanged() -> None:
    class _BoomGraph:
        def memory_entities(self, memory_id):
            raise RuntimeError("graph db locked")

    mem = _FakeMemory(_BoomGraph())
    hits = [_hit("a", 1.0), _hit("b", 0.9)]

    out = compact_hits_by_entity_overlap(hits, mem, min_idf_overlap=0.5)

    assert out == hits
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_context_compact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.context_compact'`

- [ ] **Step 3: Write the implementation**

Create `src/memo/context_compact.py`:

```python
"""Graph-aware compaction of memo_context_pack hits: drop hits that share a
rare entity with a higher-ranked hit before they ever reach a pack bucket.
Thin adapter over the chat pipeline's already-tested, IDF-weighted
compact_by_entity_overlap — see
docs/superpowers/specs/2026-08-04-context-graph-compact-design.md for why
this wraps rather than duplicates that algorithm.
"""

from __future__ import annotations

from typing import Any

from memo.chat.graph_compact import compact_by_entity_overlap


def compact_hits_by_entity_overlap(
    hits: list[Any],
    memory: Any,
    *,
    min_idf_overlap: float,
    min_group_size: int = 2,
) -> list[Any]:
    """Drop `memory.search()` hit objects that are IDF-overlap duplicates of
    a higher-ranked hit, before they occupy a context-pack bucket slot.

    Adapts each hit to the dict shape `compact_by_entity_overlap` expects
    (carrying a synthetic `_hit_index` so surviving rows map back to the
    original hit objects without relying on `id` uniqueness), delegates the
    overlap/IDF grouping to it unchanged, then filters the original `hits`
    list — preserving its original rank order, not the delegate's internal
    resort.

    Fail-open: any exception (graph lookup, IDF lookup, or adaptation)
    returns `hits` unchanged — this must never block a tool response.
    """
    if len(hits) < min_group_size:
        return list(hits)
    try:
        dict_hits = []
        for i, h in enumerate(hits):
            score = getattr(h, "score", None)
            dict_hits.append(
                {
                    "id": str(getattr(h, "id", "") or ""),
                    "title": str(getattr(h, "title", "") or ""),
                    "score": 0.0 if score is None else score,
                    "_hit_index": i,
                }
            )
        compacted = compact_by_entity_overlap(
            dict_hits, memory, min_idf_overlap=min_idf_overlap, min_group_size=min_group_size
        )
        keep = {d["_hit_index"] for d in compacted}
    except Exception:
        return list(hits)
    return [h for i, h in enumerate(hits) if i in keep]


__all__ = ["compact_hits_by_entity_overlap"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_context_compact.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint and type-check**

Run: `uv run --no-sync ruff check src/memo/context_compact.py tests/test_context_compact.py && uv run --no-sync mypy src/memo/context_compact.py`
Expected: no errors (do NOT run `ruff format src/` — format only the touched files if needed, per the shared-worktree rule)

- [ ] **Step 6: Commit**

```bash
git add src/memo/context_compact.py tests/test_context_compact.py
git commit -m "feat(context-pack): add entity-overlap hit compaction wrapper"
```

---

### Task 2: Wire `MEMO_CONTEXT_GRAPH_COMPACT` flags

**Files:**
- Modify: `src/memo/flags_search.py`
- Test: `tests/test_flags.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `flag_bool("MEMO_CONTEXT_GRAPH_COMPACT")` and `flag_float("MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF")`, resolvable through the standard `flags.py` registry. Task 3 reads both.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_flags.py` (a new test function, near other `search` group flag tests):

```python
def test_context_graph_compact_flags_registered() -> None:
    assert flags.flag_bool("MEMO_CONTEXT_GRAPH_COMPACT") is False
    assert flags.flag_float("MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF") == 0.5
    env = {"MEMO_CONTEXT_GRAPH_COMPACT": "1", "MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF": "0.7"}
    assert flags.flag_bool("MEMO_CONTEXT_GRAPH_COMPACT", env=env) is True
    assert flags.flag_float("MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF", env=env) == 0.7
    assert flags.validate(env=env) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run --no-sync pytest tests/test_flags.py::test_context_graph_compact_flags_registered -v`
Expected: FAIL — flag not registered, defaults return `None`/raise, or `validate` reports an unknown var.

- [ ] **Step 3: Implement**

In `src/memo/flags_search.py`, add two `_spec(...)` entries to `SPECS` immediately after the existing `MEMO_CONTEXT_PACK` entry:

```python
    _spec(
        "MEMO_CONTEXT_GRAPH_COMPACT",
        "bool",
        False,
        "search",
        "Collapse memo_context_pack hits that share a rare (IDF-weighted) entity "
        "with a higher-ranked hit before they occupy a bucket slot. Mirrors "
        "MEMO_CHAT_GRAPH_COMPACT's algorithm via a thin adapter. Default off; "
        "no dedicated eval harness exists yet for memo_context_pack.",
    ),
    _spec(
        "MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF",
        "float",
        0.5,
        "search",
        "Minimum IDF-weighted entity overlap required to collapse two "
        "memo_context_pack hits (matches MEMO_CHAT_GRAPH_COMPACT_MIN_IDF's "
        "default). Conservative starting point.",
        min_val=0.0,
    ),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_flags.py -v`
Expected: PASS

- [ ] **Step 5: Lint and type-check**

Run: `uv run --no-sync ruff check src/memo/flags_search.py tests/test_flags.py && uv run --no-sync mypy src/memo/flags_search.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/memo/flags_search.py tests/test_flags.py
git commit -m "feat(context-pack): register MEMO_CONTEXT_GRAPH_COMPACT flags"
```

---

### Task 3: Wire compaction into `memo_context_pack`

**Files:**
- Modify: `src/memo/server_context_pack.py`
- Test: `tests/test_context_pack_surface.py`

**Interfaces:**
- Consumes: `compact_hits_by_entity_overlap` (Task 1), `MEMO_CONTEXT_GRAPH_COMPACT`/`MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF` flags (Task 2).
- Produces: no new public interface — this is the integration point; `memo_context_pack`'s response reflects compacted hits when the flag is on.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_context_pack_surface.py`:

```python
class _CompactFakeGraph:
    def memory_entities(self, memory_id):
        if memory_id in ("m1", "m2"):
            return [{"name": "proyecto-omega", "type": "topic", "mention_count": 1}]
        return []

    def total_indexed_memories(self):
        return 10

    def entity_doc_freqs(self, names):
        return {"proyecto-omega": 1.0} if "proyecto-omega" in names else {}


def _omega_hits():
    return [
        _hit(id="m1", title="Nota uno", score=0.9, body="cuerpo uno sobre proyecto omega"),
        _hit(id="m2", title="Nota dos", score=0.85, body="cuerpo dos sobre proyecto omega"),
    ]


def test_context_pack_graph_compact_collapses_when_enabled(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    monkeypatch.setenv("MEMO_CONTEXT_GRAPH_COMPACT", "1")
    mem_with_stub.search = lambda *a, **kw: _omega_hits()  # type: ignore[method-assign]
    monkeypatch.setattr(mem_with_stub, "graph", _CompactFakeGraph())
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))

    payload = tool.fn(question="qué sabés del proyecto omega?")

    rows = payload["current_facts"] + payload["supporting_context"] + payload["stale_or_conflicting"]
    ids = {row["id"] for row in rows}
    assert len(ids) == 1
    assert ids <= {"m1", "m2"}


def test_context_pack_graph_compact_noop_when_disabled(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    monkeypatch.setenv("MEMO_CONTEXT_GRAPH_COMPACT", "0")
    mem_with_stub.search = lambda *a, **kw: _omega_hits()  # type: ignore[method-assign]
    monkeypatch.setattr(mem_with_stub, "graph", _CompactFakeGraph())
    server = build_server(memory=mem_with_stub)
    tool = asyncio.run(server.get_tool("memo_context_pack"))

    payload = tool.fn(question="qué sabés del proyecto omega?")

    rows = payload["current_facts"] + payload["supporting_context"] + payload["stale_or_conflicting"]
    ids = {row["id"] for row in rows}
    assert ids == {"m1", "m2"}
```

Note: `_hit(...)` is the existing helper at the top of this file (`SimpleNamespace` builder) — extend its call here with `body=` since the existing default already supports `**overrides`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_context_pack_surface.py -v`
Expected: FAIL — both `m1` and `m2` survive regardless of the flag (compaction not wired in yet).

- [ ] **Step 3: Implement**

In `src/memo/server_context_pack.py`, insert before the `pack = build_context_pack(...)` line:

```python
        from memo.flags import flag_bool, flag_float

        if not flag_bool("MEMO_CONTEXT_PACK"):
            return {
                "status": "disabled",
                "reason": "MEMO_CONTEXT_PACK=0 disables explicit context-pack tools.",
                "question": question,
            }
        t0 = now_ms()
        hits = memory.search(
            question,
            limit=k,
            type_=type_,
            mode="hybrid",
            disable_reranker=True,
            read_through=False,
            quality_rerank=True,
        )
        if flag_bool("MEMO_CONTEXT_GRAPH_COMPACT"):
            from memo.context_compact import compact_hits_by_entity_overlap

            hits = compact_hits_by_entity_overlap(
                hits, memory, min_idf_overlap=flag_float("MEMO_CONTEXT_GRAPH_COMPACT_MIN_IDF") or 0.5
            )
        pack = build_context_pack(question, hits, snippet_chars=snippet_chars)
```

(The existing `from memo.flags import flag_bool` import line gains `flag_float`; everything else in the function body is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_context_pack_surface.py -v`
Expected: PASS

- [ ] **Step 5: Run the full context-pack test surface + lint/type-check**

Run: `uv run --no-sync pytest tests/test_context_pack.py tests/test_context_compact.py tests/test_context_pack_surface.py tests/test_context_pack_code.py tests/test_flags.py -v && uv run --no-sync ruff check src/memo/server_context_pack.py tests/test_context_pack_surface.py && uv run --no-sync mypy src/memo/server_context_pack.py`
Expected: PASS, no lint/type errors — confirms this change didn't regress the rest of the context-pack surface.

- [ ] **Step 6: Commit**

```bash
git add src/memo/server_context_pack.py tests/test_context_pack_surface.py
git commit -m "feat(context-pack): wire graph-aware hit compaction into memo_context_pack"
```

---

### Task 4: Full verification pass (no new code)

This task is a verification checkpoint, not new code.

**Files:** none.

- [ ] **Step 1: Full relevant test surface**

Run: `uv run --no-sync pytest tests/test_context_pack.py tests/test_context_compact.py tests/test_server_context_pack.py tests/test_context_pack_surface.py tests/test_flags.py -v` (drop any filename that doesn't exist — `test_server_context_pack.py` may not exist as a separate file; the actual `memo_context_pack` MCP-level tests live in `test_context_pack_surface.py`).
Expected: all PASS.

- [ ] **Step 2: `memo config validate` sanity**

Run: `uv run --no-sync pytest tests/test_flags.py -v` (already covers `validate`/`unknown_memo_vars` for the new flags via Task 2's test).
Expected: PASS — confirms the new flags don't trip typo detection.

- [ ] **Step 3: Record the outcome**

This flag ships default-off. No eval-harness gate exists to graduate it to default-on (see spec's Out of scope) — that is an explicit, separate follow-up requiring its own measurement story, not part of this plan.
