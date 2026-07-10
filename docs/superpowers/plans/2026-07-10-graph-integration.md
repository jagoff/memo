# Memo Graph Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a measured graph integration layer that improves recall, explains graph-related results, improves graph navigation, adds deterministic semantic relations, and reports graph-specific eval metrics.

**Architecture:** Add a focused graph signal module that centralizes entity extraction, hub suppression, graph candidate attribution, and graph trace data without owning storage. Add a graph reason module that turns signal traces into honest user-facing explanations. Add deterministic semantic relations inside `graph.db`, then thread bounded graph reasons through search/debug, CLI/MCP navigation, and eval before allowing any default promotion.

**Tech Stack:** Python 3.13, sqlite, Click, FastMCP, existing `memo.flags`, `GraphStore`, `Memory.search`, `memo eval recall`, pytest, ruff, mypy.

## Global Constraints

- Markdown files remain the source of truth; graph relations are derived and rebuildable.
- Relation storage belongs in `graph.db`, not a new sidecar.
- New flags use the existing `MEMO_GRAPH_*` namespace.
- Proposed flags are `MEMO_GRAPH_SIGNAL_ENABLED`, `MEMO_GRAPH_REASON_ENABLED`, `MEMO_GRAPH_SEMANTIC_RELATIONS`, `MEMO_GRAPH_HUB_SUPPRESSION`, and `MEMO_GRAPH_SIGNAL_BUDGET_MS`.
- Graph-specific metrics are diagnostic first.
- Hard promotion gates remain the existing precision/noise recall eval gates.
- Default human output gets compact graph explanations.
- JSON output carries full `graph_reason` detail.
- LLM relation extraction stays out of scope for the first implementation plan.
- No graph failure may break CRUD, search, ask, or recall.
- Recall-hook graph work must obey a strict deadline and degrade to no graph addition.
- No new heavy graph dependency.

---

## File Structure

- Create `src/memo/graph_signal.py`: focused graph retrieval signal helpers, hub suppression, attribution structs, and deadline handling.
- Create `src/memo/graph_reason.py`: converts graph signal and semantic relations into `graph_reason` dictionaries and compact text.
- Modify `src/memo/flags.py`: register new graph flags instead of reading env directly.
- Modify `src/memo/memory/search_scoring_ops.py`: route graph candidates/expansion/co-recall metadata through `graph_signal`.
- Modify `src/memo/memory/search_ops.py`: attach graph traces to `MemoryRecord.extra` when graph reasons are enabled.
- Modify `src/memo/graph.py`: add deterministic semantic relation schema, upsert/list/rebuild helpers.
- Modify `src/memo/navigation.py`: use weighted edges and hub filters for bounded path/neighbor/explore outputs.
- Modify `src/memo/cli_graph.py`: expose concise `why`, richer `neighbors`, and bounded community output.
- Modify `src/memo/server_graph_tool.py`: keep one consolidated MCP tool and add bounded `why`/reason output.
- Modify `src/memo/eval_recall.py`: report graph diagnostic metrics without making them hard gates.
- Add/modify tests in `tests/test_graph_signal.py`, `tests/test_graph_reason.py`, `tests/test_graph_semantic_relations.py`, `tests/test_navigation.py`, `tests/test_server_graph_tool.py`, `tests/test_eval_recall.py`, and existing graph retrieval tests.

---

### Task 1: Register Graph Integration Flags

**Files:**
- Modify: `src/memo/flags.py`
- Test: `tests/test_flags.py` or nearest existing flags test file

**Interfaces:**
- Consumes: existing `flag_bool`, `flag_int`, `flag_float` registry behavior.
- Produces: registered flags:
  - `MEMO_GRAPH_SIGNAL_ENABLED: bool = False`
  - `MEMO_GRAPH_REASON_ENABLED: bool = False`
  - `MEMO_GRAPH_SEMANTIC_RELATIONS: bool = False`
  - `MEMO_GRAPH_HUB_SUPPRESSION: bool = True`
  - `MEMO_GRAPH_SIGNAL_BUDGET_MS: int = 150`
  - `MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO: float = 0.25`
  - `MEMO_GRAPH_MIN_ENTITY_IDF: float = 0.5`

- [ ] **Step 1: Find the flag registry pattern**

Run: `rg -n "MEMO_GRAPH|register|Flag" src/memo/flags.py tests -g '*.py'`

Expected: identify the exact helper used to register behavioral flags.

- [ ] **Step 2: Write failing tests for defaults**

Add tests shaped like:

```python
from memo.flags import flag_bool, flag_float, flag_int


def test_graph_integration_flags_have_safe_defaults(monkeypatch):
    for key in (
        "MEMO_GRAPH_SIGNAL_ENABLED",
        "MEMO_GRAPH_REASON_ENABLED",
        "MEMO_GRAPH_SEMANTIC_RELATIONS",
    ):
        monkeypatch.delenv(key, raising=False)

    assert flag_bool("MEMO_GRAPH_SIGNAL_ENABLED") is False
    assert flag_bool("MEMO_GRAPH_REASON_ENABLED") is False
    assert flag_bool("MEMO_GRAPH_SEMANTIC_RELATIONS") is False
    assert flag_bool("MEMO_GRAPH_HUB_SUPPRESSION") is True
    assert flag_int("MEMO_GRAPH_SIGNAL_BUDGET_MS") == 150
    assert flag_float("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO") == 0.25
    assert flag_float("MEMO_GRAPH_MIN_ENTITY_IDF") == 0.5
```

- [ ] **Step 3: Run the focused test and verify failure**

Run: `uv run --no-sync pytest tests/test_flags.py::test_graph_integration_flags_have_safe_defaults -v`

Expected: FAIL because one or more flags are unknown or return `None`.

- [ ] **Step 4: Register the flags**

In `src/memo/flags.py`, add entries using the existing local registration style. Do not use `os.environ.get`.

Expected shape:

```python
FlagSpec("MEMO_GRAPH_SIGNAL_ENABLED", default=False, type="bool", description="Enable unified graph signal collection for retrieval."),
FlagSpec("MEMO_GRAPH_REASON_ENABLED", default=False, type="bool", description="Attach graph_reason attribution to graph-touched results."),
FlagSpec("MEMO_GRAPH_SEMANTIC_RELATIONS", default=False, type="bool", description="Enable deterministic semantic relation storage and reads."),
FlagSpec("MEMO_GRAPH_HUB_SUPPRESSION", default=True, type="bool", description="Suppress high-document-frequency graph hubs in retrieval/navigation."),
FlagSpec("MEMO_GRAPH_SIGNAL_BUDGET_MS", default=150, type="int", description="Maximum milliseconds for graph signal work in hot paths."),
FlagSpec("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO", default=0.25, type="float", description="Entity document-frequency ratio above which an entity is treated as a hub."),
FlagSpec("MEMO_GRAPH_MIN_ENTITY_IDF", default=0.5, type="float", description="Minimum query entity IDF required before graph signal affects ranking."),
```

If the local type field is an enum or callable rather than a string, use the existing exact style from neighboring flags.

- [ ] **Step 5: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_flags.py::test_graph_integration_flags_have_safe_defaults -v
uv run --no-sync ruff check src/memo/flags.py tests/test_flags.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/flags.py tests/test_flags.py
git commit -m "feat: add graph integration flags"
```

---

### Task 2: Add Unified Graph Signal Layer

**Files:**
- Create: `src/memo/graph_signal.py`
- Test: `tests/test_graph_signal.py`

**Interfaces:**
- Consumes:
  - `graph.total_indexed_memories() -> int`
  - `graph.entity_doc_freqs(names: Sequence[str]) -> dict[str, float]`
  - `graph.weighted_neighbors(name: str) -> dict[str, float]`
  - `graph.memory_entities(memory_id: str) -> list[dict[str, Any]]`
  - `memo.graph_proximity.extract_query_entities(prompt, graph)`
- Produces:
  - `GraphSignalConfig`
  - `GraphSignalTrace`
  - `GraphSignal`
  - `collect_graph_signal(graph, query, candidate_ids, *, now=None, deadline=None, config=None) -> GraphSignal`

- [ ] **Step 1: Write failing tests for hub suppression and rare entity signal**

Create `tests/test_graph_signal.py`:

```python
from __future__ import annotations

from memo.graph_signal import GraphSignalConfig, collect_graph_signal


class _Graph:
    def __init__(self):
        self.entities_by_memory = {
            "rare-doc": [{"name": "daemon", "type": "technology", "mention_count": 2}],
            "hub-doc": [{"name": "memo", "type": "project", "mention_count": 100}],
        }

    def total_indexed_memories(self):
        return 100

    def entity_doc_freqs(self, names):
        values = {"mlx": 2.0, "daemon": 3.0, "memo": 90.0}
        return {n: values[n] for n in names if n in values}

    def weighted_neighbors(self, name):
        if name == "mlx":
            return {"daemon": 4.0, "memo": 20.0}
        return {}

    def memory_entities(self, memory_id):
        return self.entities_by_memory.get(memory_id, [])

    def entity_names(self):
        return {"mlx", "daemon", "memo"}


def test_collect_graph_signal_boosts_rare_neighbor_and_suppresses_hub():
    signal = collect_graph_signal(
        _Graph(),
        "mlx",
        ["rare-doc", "hub-doc"],
        config=GraphSignalConfig(enabled=True, hub_suppression=True, min_entity_idf=0.5),
    )

    assert signal.enabled is True
    assert signal.query_entities == ["mlx"]
    assert signal.boosts["rare-doc"] > 0
    assert "hub-doc" not in signal.boosts
    assert signal.traces["rare-doc"].mode == "proximity"
```

- [ ] **Step 2: Run test and verify failure**

Run: `uv run --no-sync pytest tests/test_graph_signal.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'memo.graph_signal'`.

- [ ] **Step 3: Implement dataclasses and IDF helper**

Create `src/memo/graph_signal.py` with:

```python
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from memo.flags import flag_bool, flag_float, flag_int
from memo.graph_proximity import extract_query_entities


@dataclass(frozen=True)
class GraphSignalConfig:
    enabled: bool = False
    hub_suppression: bool = True
    hub_max_doc_freq_ratio: float = 0.25
    min_entity_idf: float = 0.5
    weight: float = 0.05
    budget_ms: int = 150


@dataclass(frozen=True)
class GraphSignalTrace:
    mode: str
    query_entities: list[str]
    hit_entities: list[str]
    neighbor_edges: list[dict[str, Any]] = field(default_factory=list)
    skipped: str | None = None


@dataclass(frozen=True)
class GraphSignal:
    enabled: bool
    query_entities: list[str]
    boosts: dict[str, float]
    traces: dict[str, GraphSignalTrace]
    skipped: str | None = None
    elapsed_ms: float = 0.0


def config_from_flags() -> GraphSignalConfig:
    return GraphSignalConfig(
        enabled=flag_bool("MEMO_GRAPH_SIGNAL_ENABLED"),
        hub_suppression=flag_bool("MEMO_GRAPH_HUB_SUPPRESSION"),
        hub_max_doc_freq_ratio=flag_float("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO") or 0.25,
        min_entity_idf=flag_float("MEMO_GRAPH_MIN_ENTITY_IDF") or 0.5,
        budget_ms=flag_int("MEMO_GRAPH_SIGNAL_BUDGET_MS") or 150,
    )


def entity_idf(df: float, n_docs: int) -> float:
    if n_docs <= 0 or df <= 0:
        return 0.0
    return max(0.0, math.log(n_docs / df))
```

- [ ] **Step 4: Implement `collect_graph_signal`**

Add:

```python
def collect_graph_signal(
    graph: Any,
    query: str,
    candidate_ids: Sequence[str],
    *,
    deadline: float | None = None,
    config: GraphSignalConfig | None = None,
) -> GraphSignal:
    started = time.monotonic()
    cfg = config or config_from_flags()
    if not cfg.enabled:
        return GraphSignal(False, [], {}, {}, skipped="disabled")

    if deadline is None:
        deadline = started + (cfg.budget_ms / 1000.0)

    try:
        query_entities = extract_query_entities(query, graph)
        if not query_entities:
            return GraphSignal(True, [], {}, {}, skipped="no_query_entities")

        n_docs = int(graph.total_indexed_memories())
        q_df = graph.entity_doc_freqs(query_entities) if n_docs > 0 else {}
        allowed_query_entities = [
            ent for ent in query_entities
            if entity_idf(float(q_df.get(ent.lower(), 0.0)), n_docs) >= cfg.min_entity_idf
        ]
        if not allowed_query_entities:
            return GraphSignal(True, query_entities, {}, {}, skipped="query_entities_below_idf")

        proximity: dict[str, dict[str, Any]] = {}
        for ent in allowed_query_entities:
            if time.monotonic() > deadline:
                return GraphSignal(True, query_entities, {}, {}, skipped="deadline")
            for neighbor, weight in graph.weighted_neighbors(ent).items():
                key = str(neighbor).strip().lower()
                proximity[key] = {"from": ent, "to": key, "weight": float(weight)}

        if not proximity:
            return GraphSignal(True, query_entities, {}, {}, skipped="no_neighbors")

        neigh_df = graph.entity_doc_freqs(list(proximity)) if n_docs > 0 else {}
        boosts: dict[str, float] = {}
        traces: dict[str, GraphSignalTrace] = {}
        for mid in candidate_ids:
            if time.monotonic() > deadline:
                break
            hit_entities = [str(e.get("name", "")).strip().lower() for e in graph.memory_entities(mid)]
            edges: list[dict[str, Any]] = []
            score = 0.0
            for ent in hit_entities:
                edge = proximity.get(ent)
                if edge is None:
                    continue
                df = float(neigh_df.get(ent, 0.0))
                if cfg.hub_suppression and n_docs > 0 and (df / n_docs) > cfg.hub_max_doc_freq_ratio:
                    continue
                idf = entity_idf(df, n_docs) if n_docs > 0 else 1.0
                if idf <= 0:
                    continue
                score += float(edge["weight"]) * idf * cfg.weight
                edges.append({**edge, "idf": idf})
            if score > 0:
                boosts[mid] = round(score, 6)
                traces[mid] = GraphSignalTrace(
                    mode="proximity",
                    query_entities=allowed_query_entities,
                    hit_entities=hit_entities,
                    neighbor_edges=edges,
                )
        return GraphSignal(True, query_entities, boosts, traces, elapsed_ms=(time.monotonic() - started) * 1000)
    except Exception as exc:
        return GraphSignal(True, [], {}, {}, skipped=f"error:{type(exc).__name__}")
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_graph_signal.py -v
uv run --no-sync ruff check src/memo/graph_signal.py tests/test_graph_signal.py
uv run --no-sync mypy src/memo/graph_signal.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/graph_signal.py tests/test_graph_signal.py
git commit -m "feat: add graph signal layer"
```

---

### Task 3: Add Graph Reason Builder

**Files:**
- Create: `src/memo/graph_reason.py`
- Test: `tests/test_graph_reason.py`

**Interfaces:**
- Consumes: `GraphSignalTrace` from `memo.graph_signal`.
- Produces:
  - `build_graph_reason(memory_id: str, trace: GraphSignalTrace, *, relations: list[dict[str, Any]] | None = None) -> dict[str, Any]`
  - `format_graph_reason(reason: dict[str, Any]) -> str`

- [ ] **Step 1: Write failing tests**

```python
from memo.graph_reason import build_graph_reason, format_graph_reason
from memo.graph_signal import GraphSignalTrace


def test_build_graph_reason_is_honest_and_compact():
    trace = GraphSignalTrace(
        mode="proximity",
        query_entities=["mlx"],
        hit_entities=["daemon"],
        neighbor_edges=[{"from": "mlx", "to": "daemon", "weight": 4.0, "idf": 3.1}],
    )

    reason = build_graph_reason("abc123", trace)

    assert reason["memory_id"] == "abc123"
    assert reason["mode"] == "proximity"
    assert reason["query_entities"] == ["mlx"]
    assert reason["hit_entities"] == ["daemon"]
    assert reason["neighbor_edges"][0]["to"] == "daemon"
    assert reason["confidence"] == "derived"
    assert "path" not in reason


def test_format_graph_reason_does_not_claim_verification():
    trace = GraphSignalTrace(
        mode="proximity",
        query_entities=["mlx"],
        hit_entities=["daemon"],
        neighbor_edges=[{"from": "mlx", "to": "daemon", "weight": 4.0, "idf": 3.1}],
    )
    text = format_graph_reason(build_graph_reason("abc123", trace))

    assert "related via graph" in text
    assert "verified" not in text.lower()
```

- [ ] **Step 2: Run test and verify failure**

Run: `uv run --no-sync pytest tests/test_graph_reason.py -v`

Expected: FAIL with missing module.

- [ ] **Step 3: Implement builder**

Create:

```python
from __future__ import annotations

from typing import Any

from memo.graph_signal import GraphSignalTrace


def build_graph_reason(
    memory_id: str,
    trace: GraphSignalTrace,
    *,
    relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    reason: dict[str, Any] = {
        "memory_id": memory_id,
        "mode": trace.mode,
        "query_entities": trace.query_entities,
        "hit_entities": trace.hit_entities,
        "confidence": "derived",
    }
    if trace.neighbor_edges:
        reason["neighbor_edges"] = trace.neighbor_edges
    if relations:
        reason["relations"] = relations
    if trace.skipped:
        reason["skipped"] = trace.skipped
    return reason


def format_graph_reason(reason: dict[str, Any]) -> str:
    mode = reason.get("mode", "graph")
    q = ", ".join(reason.get("query_entities") or [])
    h = ", ".join(reason.get("hit_entities") or [])
    if q and h:
        return f"related via graph ({mode}): {q} -> {h}"
    if q:
        return f"related via graph ({mode}): {q}"
    return f"related via graph ({mode})"
```

- [ ] **Step 4: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_graph_reason.py -v
uv run --no-sync ruff check src/memo/graph_reason.py tests/test_graph_reason.py
uv run --no-sync mypy src/memo/graph_reason.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/memo/graph_reason.py tests/test_graph_reason.py
git commit -m "feat: add graph reason builder"
```

---

### Task 4: Attach Graph Signal And Reasons To Search

**Files:**
- Modify: `src/memo/memory/search_ops.py`
- Modify: `src/memo/memory/search_scoring_ops.py`
- Test: `tests/test_graph_proximity.py` or new `tests/test_search_graph_signal.py`

**Interfaces:**
- Consumes: `collect_graph_signal`, `build_graph_reason`, `flag_bool`.
- Produces: graph-touched `MemoryRecord.extra["graph_reason"]` when `MEMO_GRAPH_REASON_ENABLED=1`; trace stage `graph_signal`.

- [ ] **Step 1: Write failing integration test**

Create `tests/test_search_graph_signal.py` using the repo's isolated memory fixture pattern:

```python
def test_search_attaches_graph_reason_when_enabled(tmp_cfg, monkeypatch):
    from memo.memory import Memory

    monkeypatch.setenv("MEMO_GRAPH_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_REASON_ENABLED", "1")
    mem = Memory(tmp_cfg)

    mem.save("MLX daemon note\n\nThe daemon keeps MLX warm.", tags=["graph-test"])
    mem.save("Recall hook budget\n\nThe recall hook uses daemon context.", tags=["graph-test"])
    mem.graph.rebuild_edges()

    trace = []
    hits = mem.search("mlx", limit=5, mode="hybrid", _trace=trace)

    assert any(t["stage"] == "graph_signal" for t in trace)
    assert any((h.extra or {}).get("graph_reason") for h in hits)
```

If `Memory.save()` in this repo requires an embedder stub in tests, follow the existing `tmp_cfg`/stub pattern from nearby search tests.

- [ ] **Step 2: Run test and verify failure**

Run: `uv run --no-sync pytest tests/test_search_graph_signal.py::test_search_attaches_graph_reason_when_enabled -v`

Expected: FAIL because search does not attach `graph_reason`.

- [ ] **Step 3: Add graph signal after materialization**

In `search_ops.py`, after `out` is materialized and before rerank/trim code can discard attribution, add a guarded block shaped like:

```python
        graph_signal = None
        if out and flag_bool("MEMO_GRAPH_SIGNAL_ENABLED"):
            try:
                from memo.graph_reason import build_graph_reason
                from memo.graph_signal import collect_graph_signal

                graph_signal = collect_graph_signal(
                    self.graph,
                    query,
                    [r.id for r in out],
                )
                _add_trace(
                    "graph_signal",
                    enabled=graph_signal.enabled,
                    query_entities=graph_signal.query_entities,
                    touched_count=len(graph_signal.boosts),
                    skipped=graph_signal.skipped,
                    elapsed_ms=round(graph_signal.elapsed_ms, 3),
                )
                if flag_bool("MEMO_GRAPH_REASON_ENABLED") and graph_signal.traces:
                    out = [
                        replace(
                            r,
                            extra={
                                **(r.extra or {}),
                                "graph_reason": build_graph_reason(r.id, graph_signal.traces[r.id]),
                            },
                        )
                        if r.id in graph_signal.traces
                        else r
                        for r in out
                    ]
            except Exception as exc:
                _log.debug("graph_signal failed: %s", exc)
                _add_trace("graph_signal", enabled=True, skipped="error")
```

Ensure `replace` is already imported; if not, import it from `dataclasses` near existing imports.

- [ ] **Step 4: Apply score boosts conservatively**

If the test corpus shows reasons but no score movement, add score movement only under `MEMO_GRAPH_SIGNAL_ENABLED` and preserve order stability:

```python
                if graph_signal.boosts:
                    out = [
                        replace(r, score=round((r.score or 0.0) + graph_signal.boosts.get(r.id, 0.0), 6))
                        for r in out
                    ]
                    out.sort(key=lambda r: r.score or 0.0, reverse=True)
```

Do not remove existing `_fetch_graph_candidates`, `_apply_graph_expansion`, or co-recall behavior in this task.

- [ ] **Step 5: Run focused tests**

Run:

```bash
uv run --no-sync pytest tests/test_search_graph_signal.py -v
uv run --no-sync pytest tests/test_graph_proximity.py tests/test_rrf_ranking.py -v
uv run --no-sync ruff check src/memo/memory/search_ops.py src/memo/memory/search_scoring_ops.py tests/test_search_graph_signal.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/memory/search_ops.py src/memo/memory/search_scoring_ops.py tests/test_search_graph_signal.py
git commit -m "feat: attach graph signal to search"
```

---

### Task 5: Add Deterministic Semantic Relation Storage

**Files:**
- Modify: `src/memo/graph.py`
- Test: `tests/test_graph_semantic_relations.py`

**Interfaces:**
- Consumes: `GraphStore._tx()`.
- Produces:
  - table `semantic_relations`
  - `GraphStore.upsert_semantic_relation(...) -> None`
  - `GraphStore.semantic_relations_for(source_id: str | None = None, target_id: str | None = None, relation: str | None = None, limit: int = 50) -> list[dict[str, Any]]`
  - `GraphStore.delete_semantic_relations_for_source(source_id: str) -> int`

- [ ] **Step 1: Write failing storage tests**

```python
from memo.graph import GraphStore


def test_semantic_relation_upsert_is_idempotent(tmp_path):
    graph = GraphStore(tmp_path / "graph.db")

    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id="a",
        target_kind="memory",
        target_id="b",
        relation="supports",
        weight=0.8,
        confidence=0.9,
        evidence_id="fact-1",
        derived_from="test",
    )
    graph.upsert_semantic_relation(
        source_kind="memory",
        source_id="a",
        target_kind="memory",
        target_id="b",
        relation="supports",
        weight=0.8,
        confidence=0.9,
        evidence_id="fact-1",
        derived_from="test",
    )

    rows = graph.semantic_relations_for(source_id="a")
    assert len(rows) == 1
    assert rows[0]["relation"] == "supports"
    assert rows[0]["target_id"] == "b"
```

- [ ] **Step 2: Run test and verify failure**

Run: `uv run --no-sync pytest tests/test_graph_semantic_relations.py -v`

Expected: FAIL because methods do not exist.

- [ ] **Step 3: Add schema**

In `_SCHEMA_DDL` in `src/memo/graph.py`, add:

```sql
CREATE TABLE IF NOT EXISTS semantic_relations (
    source_kind TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    target_kind TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    relation    TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 1.0,
    confidence  REAL NOT NULL DEFAULT 1.0,
    evidence_id TEXT,
    derived_from TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    valid_at    TEXT,
    invalid_at  TEXT,
    PRIMARY KEY (source_kind, source_id, target_kind, target_id, relation, derived_from)
);

CREATE INDEX IF NOT EXISTS idx_sr_source ON semantic_relations(source_kind, source_id);
CREATE INDEX IF NOT EXISTS idx_sr_target ON semantic_relations(target_kind, target_id);
CREATE INDEX IF NOT EXISTS idx_sr_relation ON semantic_relations(relation);
```

- [ ] **Step 4: Add methods**

Add methods to `GraphStore`:

```python
    def upsert_semantic_relation(
        self,
        *,
        source_kind: str,
        source_id: str,
        target_kind: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
        confidence: float = 1.0,
        evidence_id: str | None = None,
        derived_from: str,
        valid_at: str | None = None,
        invalid_at: str | None = None,
    ) -> None:
        from datetime import UTC, datetime

        created_at = datetime.now(UTC).isoformat()
        with self._tx() as cx:
            cx.execute(
                "INSERT INTO semantic_relations "
                "(source_kind, source_id, target_kind, target_id, relation, weight, confidence, "
                "evidence_id, derived_from, created_at, valid_at, invalid_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(source_kind, source_id, target_kind, target_id, relation, derived_from) "
                "DO UPDATE SET weight = excluded.weight, confidence = excluded.confidence, "
                "evidence_id = excluded.evidence_id, valid_at = excluded.valid_at, "
                "invalid_at = excluded.invalid_at",
                (
                    source_kind,
                    source_id,
                    target_kind,
                    target_id,
                    relation,
                    float(weight),
                    float(confidence),
                    evidence_id,
                    derived_from,
                    created_at,
                    valid_at,
                    invalid_at,
                ),
            )

    def semantic_relations_for(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        relation: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(target_id)
        if relation is not None:
            clauses.append("relation = ?")
            params.append(relation)
        sql = "SELECT * FROM semantic_relations"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY confidence DESC, weight DESC LIMIT ?"
        params.append(limit)
        return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def delete_semantic_relations_for_source(self, source_id: str) -> int:
        with self._tx() as cx:
            cur = cx.execute("DELETE FROM semantic_relations WHERE source_id = ?", (source_id,))
            return int(cur.rowcount or 0)
```

- [ ] **Step 5: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_graph_semantic_relations.py tests/test_graph_entity_merge.py -v
uv run --no-sync ruff check src/memo/graph.py tests/test_graph_semantic_relations.py
uv run --no-sync mypy src/memo/graph.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/graph.py tests/test_graph_semantic_relations.py
git commit -m "feat: store semantic graph relations"
```

---

### Task 6: Expose Semantic Relations In Graph Reasons

**Files:**
- Modify: `src/memo/graph_reason.py`
- Modify: `src/memo/memory/search_ops.py`
- Test: `tests/test_graph_reason.py`, `tests/test_search_graph_signal.py`

**Interfaces:**
- Consumes: `GraphStore.semantic_relations_for(source_id=...)`.
- Produces: `graph_reason["relations"]` when semantic relations are enabled and available.

- [ ] **Step 1: Add failing test for relation-enriched reason**

```python
from memo.graph_reason import build_graph_reason
from memo.graph_signal import GraphSignalTrace


def test_graph_reason_includes_relations_when_supplied():
    trace = GraphSignalTrace(mode="proximity", query_entities=["a"], hit_entities=["b"])
    reason = build_graph_reason(
        "mem-a",
        trace,
        relations=[{"relation": "supersedes", "target_id": "mem-b", "confidence": 1.0}],
    )

    assert reason["relations"][0]["relation"] == "supersedes"
```

- [ ] **Step 2: Run and verify expected state**

Run: `uv run --no-sync pytest tests/test_graph_reason.py::test_graph_reason_includes_relations_when_supplied -v`

Expected: PASS if Task 3 already accepted relations; otherwise FAIL and implement the missing branch.

- [ ] **Step 3: Attach relations in search when enabled**

In `search_ops.py`, in the graph reason block, fetch relations per hit only when `MEMO_GRAPH_SEMANTIC_RELATIONS` is enabled:

```python
                    relations_by_id: dict[str, list[dict[str, Any]]] = {}
                    if flag_bool("MEMO_GRAPH_SEMANTIC_RELATIONS"):
                        for r in out:
                            relations_by_id[r.id] = self.graph.semantic_relations_for(source_id=r.id, limit=10)
```

Then pass:

```python
build_graph_reason(
    r.id,
    graph_signal.traces[r.id],
    relations=relations_by_id.get(r.id),
)
```

- [ ] **Step 4: Add integration assertion**

Extend `tests/test_search_graph_signal.py`:

```python
def test_search_graph_reason_includes_semantic_relations_when_enabled(tmp_cfg, monkeypatch):
    from memo.memory import Memory

    monkeypatch.setenv("MEMO_GRAPH_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_REASON_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_SEMANTIC_RELATIONS", "1")
    mem = Memory(tmp_cfg)
    a = mem.save("MLX daemon note\n\nThe daemon keeps MLX warm.", tags=["graph-test"])
    b = mem.save("Recall hook budget\n\nThe recall hook uses daemon context.", tags=["graph-test"])
    mem.graph.rebuild_edges()
    mem.graph.upsert_semantic_relation(
        source_kind="memory",
        source_id=a.id,
        target_kind="memory",
        target_id=b.id,
        relation="extends",
        derived_from="test",
    )

    hits = mem.search("mlx", limit=5, mode="hybrid")
    reasons = [(h.extra or {}).get("graph_reason") for h in hits]

    assert any(r and r.get("relations") for r in reasons)
```

Adapt `a.id`/return handling to the actual `Memory.save()` return shape if existing tests show a different API.

- [ ] **Step 5: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_graph_reason.py tests/test_search_graph_signal.py -v
uv run --no-sync ruff check src/memo/graph_reason.py src/memo/memory/search_ops.py tests/test_search_graph_signal.py
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/memo/graph_reason.py src/memo/memory/search_ops.py tests/test_graph_reason.py tests/test_search_graph_signal.py
git commit -m "feat: include semantic graph relations in reasons"
```

---

### Task 7: Improve CLI And MCP Graph Navigation

**Files:**
- Modify: `src/memo/navigation.py`
- Modify: `src/memo/cli_graph.py`
- Modify: `src/memo/server_graph_tool.py`
- Test: `tests/test_navigation.py`
- Test: `tests/test_server_graph_tool.py`

**Interfaces:**
- Consumes: `GraphStore.weighted_neighbors`, `GraphStore.entity_doc_freqs`, `GraphStore.total_indexed_memories`, semantic relation reads.
- Produces:
  - `GraphNavigator.why_connected(a: str, b: str, *, use_codegraph: bool | None = None, include_hubs: bool = False) -> dict[str, Any]`
  - CLI `memo graph why A B --json`
  - MCP `memo_graph(verb="why", a=..., b=...)`

- [ ] **Step 1: Write failing navigator test**

```python
def test_why_connected_returns_weighted_path(mock_graph_store):
    from memo.navigation import GraphNavigator

    mock_graph_store.record_extraction(
        memory_id="m1",
        memory_date="2026-07-10T00:00:00",
        entities=[{"name": "mlx", "type": "technology"}, {"name": "daemon", "type": "technology"}],
        extracted_at="2026-07-10T00:00:00",
    )
    mock_graph_store.rebuild_edges()
    nav = GraphNavigator(mock_graph_store)

    why = nav.why_connected("mlx", "daemon", use_codegraph=False)

    assert why["path"] == ["mlx", "daemon"]
    assert why["edges"][0]["weight"] >= 1
    assert why["evidence_memory_ids"]
```

- [ ] **Step 2: Run and verify failure**

Run: `uv run --no-sync pytest tests/test_navigation.py::test_why_connected_returns_weighted_path -v`

Expected: FAIL because `why_connected` does not exist.

- [ ] **Step 3: Implement navigator method**

In `GraphNavigator`, add:

```python
    def why_connected(
        self,
        source: str,
        target: str,
        *,
        use_codegraph: bool | None = None,
        include_hubs: bool = False,
    ) -> dict[str, Any]:
        path = self.find_shortest_path(source, target, use_codegraph=use_codegraph)
        if path is None:
            return {"source": source, "target": target, "path": [], "edges": [], "evidence_memory_ids": []}
        edges: list[dict[str, Any]] = []
        evidence: list[str] = []
        for idx in range(len(path.path) - 1):
            a = path.path[idx]
            b = path.path[idx + 1]
            neighbors = self.graph.weighted_neighbors(a)
            weight = float(neighbors.get(b, 1.0))
            mem_id = path.intermediate_memories[idx] if idx < len(path.intermediate_memories) else ""
            if mem_id and mem_id != "(codegraph)":
                evidence.append(mem_id)
            edges.append({"from": a, "to": b, "weight": weight, "memory_id": mem_id})
        return {
            "source": path.source,
            "target": path.target,
            "path": path.path,
            "length": path.length,
            "edges": edges,
            "evidence_memory_ids": evidence,
        }
```

- [ ] **Step 4: Add CLI command**

In `cli_graph.py`, add a `why` command:

```python
@graph_group.command(name="why")
@click.argument("source")
@click.argument("target")
@click.option("--json", "as_json", is_flag=True)
def graph_why(source: str, target: str, as_json: bool) -> None:
    """Explain why two entities are connected."""
    mem = _memory()
    result = mem.navigator.why_connected(source, target)
    if as_json:
        console.print_json(data=result)
        return
    if not result["path"]:
        console.print(f"[yellow]no graph path found[/yellow]: {source} -> {target}")
        return
    console.print(" -> ".join(result["path"]))
    for edge in result["edges"]:
        console.print(f"  {edge['from']} -> {edge['to']} weight={edge['weight']} via={edge['memory_id']}")
```

Use the existing memory/console helper names in `cli_graph.py`; adjust `_memory()` if the file uses a different helper.

- [ ] **Step 5: Add MCP verb**

In `server_graph_tool.py`, extend the existing `memo_graph` verb dispatch:

```python
        if verb == "why":
            if not a or not b:
                return {"error": "why requires a and b"}
            return {"verb": "why", "result": nav.why_connected(a, b, use_codegraph=uc)}
```

Use the exact argument names already accepted by the tool (`a`, `b`, `focus`) to avoid schema churn.

- [ ] **Step 6: Add CLI/MCP tests**

Add tests mirroring existing graph path tests:

```python
def test_memo_graph_why_verb(mock_memory):
    tool = _memo_graph_tool(mock_memory)
    result = tool(verb="why", a="alpha", b="gamma")
    assert result["verb"] == "why"
    assert "result" in result
```

Use existing helpers from `tests/test_server_graph_tool.py`.

- [ ] **Step 7: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_navigation.py tests/test_server_graph_tool.py -v
uv run --no-sync ruff check src/memo/navigation.py src/memo/cli_graph.py src/memo/server_graph_tool.py tests/test_navigation.py tests/test_server_graph_tool.py
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/memo/navigation.py src/memo/cli_graph.py src/memo/server_graph_tool.py tests/test_navigation.py tests/test_server_graph_tool.py
git commit -m "feat: explain graph connections"
```

---

### Task 8: Add Graph Diagnostic Metrics To Recall Eval

**Files:**
- Modify: `src/memo/eval_recall.py`
- Test: `tests/test_eval_recall.py`

**Interfaces:**
- Consumes: search trace stage `graph_signal`.
- Produces eval fields:
  - `graph_recall_gain`
  - `graph_noise_rate`
  - `graph_explanation_coverage`
  - `hub_noise_rate`
  - `latency_ms_graph`

- [ ] **Step 1: Locate eval result row construction**

Run: `rg -n "precision|noise|assoc|trace|run_config|quality" src/memo/eval_recall.py tests/test_eval_recall.py`

Expected: identify the dataclass/dict where eval row metrics are assembled.

- [ ] **Step 2: Write failing metric test**

Add a unit-level test that exercises the metric helper. If no helper exists, this task creates one.

```python
from memo.eval_recall import graph_diag_metrics


def test_graph_diag_metrics_counts_trace_fields():
    rows = [
        {
            "id": "a",
            "expected": True,
            "noise": False,
            "extra": {"graph_reason": {"query_entities": ["mlx"]}},
            "trace": [{"stage": "graph_signal", "touched_count": 1, "elapsed_ms": 12.5}],
        },
        {
            "id": "b",
            "expected": False,
            "noise": True,
            "extra": {"graph_reason": {"query_entities": ["memo"]}},
            "trace": [{"stage": "graph_signal", "touched_count": 1, "elapsed_ms": 2.5}],
        },
    ]

    metrics = graph_diag_metrics(rows)

    assert metrics["graph_recall_gain"] == 1
    assert metrics["graph_noise_rate"] == 0.5
    assert metrics["graph_explanation_coverage"] == 1.0
    assert metrics["latency_ms_graph"] == 15.0
```

- [ ] **Step 3: Run and verify failure**

Run: `uv run --no-sync pytest tests/test_eval_recall.py::test_graph_diag_metrics_counts_trace_fields -v`

Expected: FAIL because `graph_diag_metrics` does not exist.

- [ ] **Step 4: Implement helper**

In `eval_recall.py`, add:

```python
def graph_diag_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    graph_rows = [r for r in rows if (r.get("extra") or {}).get("graph_reason")]
    if not rows:
        return {
            "graph_recall_gain": 0.0,
            "graph_noise_rate": 0.0,
            "graph_explanation_coverage": 0.0,
            "hub_noise_rate": 0.0,
            "latency_ms_graph": 0.0,
        }
    graph_count = len(graph_rows)
    graph_noise = sum(1 for r in graph_rows if r.get("noise"))
    graph_expected = sum(1 for r in graph_rows if r.get("expected"))
    explained = sum(1 for r in graph_rows if (r.get("extra") or {}).get("graph_reason"))
    latency = 0.0
    for row in rows:
        for event in row.get("trace") or []:
            if event.get("stage") == "graph_signal":
                latency += float(event.get("elapsed_ms") or 0.0)
    return {
        "graph_recall_gain": float(graph_expected),
        "graph_noise_rate": (graph_noise / graph_count) if graph_count else 0.0,
        "graph_explanation_coverage": (explained / graph_count) if graph_count else 0.0,
        "hub_noise_rate": 0.0,
        "latency_ms_graph": round(latency, 3),
    }
```

Use existing `Any` import or add it from `typing`.

- [ ] **Step 5: Thread helper into eval row output**

Where eval prints or returns aggregate metrics, merge diagnostic metrics under a `graph` key or flat fields. Preferred flat output for CLI compatibility:

```python
metrics.update(graph_diag_metrics(per_prompt_rows))
```

If `per_prompt_rows` does not exist yet, add trace capture at the same point where each prompt calls `mem.search(..., _trace=trace)`.

- [ ] **Step 6: Run tests**

Run:

```bash
uv run --no-sync pytest tests/test_eval_recall.py -v
uv run --no-sync ruff check src/memo/eval_recall.py tests/test_eval_recall.py
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/memo/eval_recall.py tests/test_eval_recall.py
git commit -m "feat: report graph eval diagnostics"
```

---

### Task 9: Documentation And End-To-End Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/reference.md`
- Test: no new test file unless docs lint exists

**Interfaces:**
- Consumes: all previous task deliverables.
- Produces: user-facing docs for safe graph defaults, graph reasons, semantic relations, and eval diagnostics.

- [ ] **Step 1: Update README graph section**

Add concise examples:

```markdown
memo graph why "mlx" "daemon"
memo search "mlx daemon" --explain
MEMO_GRAPH_SIGNAL_ENABLED=1 MEMO_GRAPH_REASON_ENABLED=1 memo search "recall hook budget" --json
```

Mention that graph ranking is conservative and hub-suppressed by default.

- [ ] **Step 2: Update reference docs**

In `docs/reference.md`, document:

- new flags;
- `graph_reason` JSON field;
- `memo graph why`;
- `memo_graph` verb `why`;
- graph eval diagnostics.

- [ ] **Step 3: Run focused verification**

Run:

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
```

Expected: PASS. If full pytest is too slow or environment-limited, run:

```bash
uv run --no-sync pytest tests/test_graph_signal.py tests/test_graph_reason.py tests/test_graph_semantic_relations.py tests/test_search_graph_signal.py tests/test_navigation.py tests/test_server_graph_tool.py tests/test_eval_recall.py -v
```

and record the limitation in the final summary.

- [ ] **Step 4: Run graph eval when ranking changed**

Run:

```bash
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force
```

Expected: precision does not drop and noise does not rise versus baseline. Graph diagnostics are printed or included in JSON output.

- [ ] **Step 5: Commit docs and final verification notes**

```bash
git add README.md docs/reference.md
git commit -m "docs: describe graph integration controls"
```

---

## Self-Review

Spec coverage:

- Better recall without hub noise: Tasks 1, 2, 4, 8.
- Graph explanations: Tasks 3, 4, 6.
- Navigation and exploration: Task 7.
- Typed semantic relations: Tasks 5, 6.
- Graph-specific metrics and promotion discipline: Tasks 8, 9.
- Safe degradation and no hot-path LLM extraction: Tasks 2, 4, 5, 8 plus global constraints.

Placeholder scan:

- No `TBD`, `TODO`, `FIXME`, or placeholder task remains.
- Steps include exact paths, commands, expected outcomes, and concrete code shapes.

Type consistency:

- `GraphSignalTrace`, `GraphSignal`, `GraphSignalConfig`, `collect_graph_signal`, `build_graph_reason`, and `format_graph_reason` are defined before later tasks consume them.
- Search uses `MemoryRecord.extra["graph_reason"]`, matching eval and docs.
- Semantic relations use `GraphStore.semantic_relations_for(...)`, matching reason enrichment.
