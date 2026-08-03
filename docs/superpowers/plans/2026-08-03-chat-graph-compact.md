# Graph-aware chat/ask source compaction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When chat/ask synthesis (`memo_ask`/`memo_chat_ask`/chat UI) retrieves multiple distinct memories that share a rare entity, collapse them into one representative source (with the rest cited by id) instead of paying their full token cost each — cutting synthesis-prompt tokens without changing which memories get retrieved.

**Architecture:** One new pure function, `compact_by_entity_overlap` (`src/memo/chat/graph_compact.py`), computes IDF-weighted entity overlap between already-ranked chat sources using two existing-but-unused `GraphStore` primitives (`entity_doc_freqs`, `total_indexed_memories`), and collapses qualifying groups. It's wired into `chat/pipeline.py` right before the `synth_head` slice, gated by a new default-off `MEMO_CHAT_GRAPH_COMPACT` env flag, and its output (`related_ids`) is rendered into the synthesis prompt by `chat/synthesis.py`.

**Tech Stack:** Python 3.11+, pytest, existing `memo.chat.*` package (no new dependencies).

## Global Constraints

- Default OFF (`MEMO_CHAT_GRAPH_COMPACT` defaults to `False`) — ships inert.
- Config is env-only, read directly in `src/memo/chat/config.py`, NOT registered in `flags.py`'s markdown-config/tuned-overlay chain — matches the existing 9 `MEMO_CHAT_*` knobs (`src/memo/flags.py:294-304`).
- Fail-open: any exception during graph/entity/IDF lookup returns the input `sources` list unchanged. Never raises, never blocks a chat response.
- No change to retrieval or ranking — only post-rank compaction of an already-decided source list.
- Spec: `docs/superpowers/specs/2026-08-03-chat-graph-compact-design.md`.

---

### Task 1: `compact_by_entity_overlap` pure function

**Files:**
- Create: `src/memo/chat/graph_compact.py`
- Test: `tests/test_chat_graph_compact.py`

**Interfaces:**
- Consumes: `memo.chat.dedup.score_of(s: dict[str, Any]) -> float` (existing, `src/memo/chat/dedup.py:27`).
- Produces: `compact_by_entity_overlap(sources: list[dict[str, Any]], memory: Any, *, min_idf_overlap: float, min_group_size: int = 2) -> list[dict[str, Any]]`. Later tasks call this exact signature. A collapsed representative dict gains a `related_ids: list[tuple[str, str]]` field (list of `(id, title)` pairs for the sources it absorbed); ungrouped/uncollapsed sources are unchanged dicts.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_chat_graph_compact.py`:

```python
from memo.chat.graph_compact import compact_by_entity_overlap


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


def _src(sid: str, score: float, title: str = "") -> dict:
    return {"id": sid, "title": title or f"T{sid}", "score": score}


def test_ubiquitous_entity_overlap_does_not_collapse() -> None:
    entities = {
        "a": [{"name": "memo"}, {"name": "topic-x"}],
        "b": [{"name": "memo"}, {"name": "topic-y"}],
    }
    dfs = {"memo": 9.0, "topic-x": 1.0, "topic-y": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    sources = [_src("a", 1.0), _src("b", 0.9)]

    out = compact_by_entity_overlap(sources, mem, min_idf_overlap=0.5)

    assert {s["id"] for s in out} == {"a", "b"}
    assert all("related_ids" not in s for s in out)


def test_rare_shared_entity_collapses() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}]}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    sources = [_src("c", 1.0), _src("d", 0.9)]

    out = compact_by_entity_overlap(sources, mem, min_idf_overlap=0.5)

    assert len(out) == 1
    assert out[0]["id"] == "c"
    assert out[0]["related_ids"] == [("d", "Td")]


def test_no_entities_is_noop() -> None:
    mem = _FakeMemory(_FakeGraph({}, {}, total=10))
    sources = [_src("e", 1.0), _src("f", 0.9)]

    out = compact_by_entity_overlap(sources, mem, min_idf_overlap=0.5)

    assert {s["id"] for s in out} == {"e", "f"}
    assert all("related_ids" not in s for s in out)


def test_group_below_min_group_size_stays_uncollapsed() -> None:
    entities = {"c": [{"name": "topic-z"}], "d": [{"name": "topic-z"}], "g": []}
    dfs = {"topic-z": 1.0}
    mem = _FakeMemory(_FakeGraph(entities, dfs, total=10))
    sources = [_src("c", 1.0), _src("d", 0.9), _src("g", 0.5)]

    out = compact_by_entity_overlap(sources, mem, min_idf_overlap=0.5, min_group_size=3)

    assert {s["id"] for s in out} == {"c", "d", "g"}
    assert all("related_ids" not in s for s in out)


def test_lookup_failure_returns_sources_unchanged() -> None:
    class _BoomGraph:
        def memory_entities(self, memory_id):
            raise RuntimeError("graph db locked")

    mem = _FakeMemory(_BoomGraph())
    sources = [_src("a", 1.0), _src("b", 0.9)]

    out = compact_by_entity_overlap(sources, mem, min_idf_overlap=0.5)

    assert out == sources
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_chat_graph_compact.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'memo.chat.graph_compact'`

- [ ] **Step 3: Write the implementation**

Create `src/memo/chat/graph_compact.py`:

```python
"""Graph-aware source compaction: collapse chat sources that share rare
entities into one representative, citing the rest by id instead of paying
their full token cost. IDF-weighted so ubiquitous entities never trigger a
collapse — see docs/superpowers/specs/2026-08-03-chat-graph-compact-design.md.
"""

from __future__ import annotations

import math
from typing import Any

from memo.chat.dedup import score_of


def _entity_set(memory: Any, source_id: str) -> set[str]:
    if not source_id:
        return set()
    ents = memory.graph.memory_entities(source_id)
    return {str(e.get("name") or "").strip().lower() for e in ents if e.get("name")}


def _idf_map(memory: Any, names: set[str]) -> dict[str, float]:
    if not names:
        return {}
    total = memory.graph.total_indexed_memories()
    if total <= 0:
        return {}
    dfs = memory.graph.entity_doc_freqs(sorted(names))
    return {name: max(0.0, math.log(total / df)) for name, df in dfs.items() if df > 0}


def _weighted_overlap(a: set[str], b: set[str], idf: dict[str, float]) -> float:
    union = a | b
    if not union:
        return 0.0
    union_weight = sum(idf.get(n, 0.0) for n in union)
    if union_weight <= 0:
        return 0.0
    shared_weight = sum(idf.get(n, 0.0) for n in (a & b))
    return shared_weight / union_weight


def compact_by_entity_overlap(
    sources: list[dict[str, Any]],
    memory: Any,
    *,
    min_idf_overlap: float,
    min_group_size: int = 2,
) -> list[dict[str, Any]]:
    """Collapse sources whose IDF-weighted entity overlap clears
    ``min_idf_overlap`` into one representative per group (the
    highest-``score_of`` member), attaching a ``related_ids`` pointer
    listing the absorbed sources' ``(id, title)``. Groups smaller than
    ``min_group_size`` are left as separate, unmodified sources.

    Fail-open: any exception during graph lookup returns ``sources``
    unchanged — this must never block a chat response.
    """
    if len(sources) < min_group_size:
        return list(sources)
    try:
        entity_sets = {
            i: _entity_set(memory, str(s.get("id") or "")) for i, s in enumerate(sources)
        }
        all_names: set[str] = set()
        for names in entity_sets.values():
            all_names.update(names)
        if not all_names:
            return list(sources)
        idf = _idf_map(memory, all_names)
    except Exception:
        return list(sources)

    ordered = sorted(range(len(sources)), key=lambda i: score_of(sources[i]), reverse=True)
    groups: list[list[int]] = []
    for idx in ordered:
        placed = False
        for group in groups:
            rep_idx = group[0]
            if _weighted_overlap(entity_sets[idx], entity_sets[rep_idx], idf) >= min_idf_overlap:
                group.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])

    out: list[dict[str, Any]] = []
    for group in groups:
        if len(group) >= min_group_size:
            rep = dict(sources[group[0]])
            rep["related_ids"] = [
                (str(sources[i].get("id") or ""), str(sources[i].get("title") or ""))
                for i in group[1:]
            ]
            out.append(rep)
        else:
            out.extend(sources[i] for i in group)
    out.sort(key=score_of, reverse=True)
    return out


__all__ = ["compact_by_entity_overlap"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_chat_graph_compact.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Lint and type-check**

Run: `uv run --no-sync ruff check src/memo/chat/graph_compact.py tests/test_chat_graph_compact.py && uv run --no-sync ruff format src/memo/chat/graph_compact.py tests/test_chat_graph_compact.py && uv run --no-sync mypy src/memo/chat/graph_compact.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/memo/chat/graph_compact.py tests/test_chat_graph_compact.py
git commit -m "feat(chat): add IDF-weighted entity-overlap source compaction"
```

---

### Task 2: Wire `MEMO_CHAT_GRAPH_COMPACT` config

**Files:**
- Modify: `src/memo/chat/config.py`
- Modify: `src/memo/flags.py:294-305` (the `owned` exclusion set)
- Test: `tests/test_chat_config.py`
- Test: `tests/test_flags.py:205-210`

**Interfaces:**
- Consumes: nothing new.
- Produces: `ChatConfig.graph_compact: bool` and `ChatConfig.graph_compact_min_idf: float`, populated by `ChatConfig.load()`. Task 3 reads both.

- [ ] **Step 1: Write the failing tests**

In `tests/test_chat_config.py`, extend `test_defaults_match_production` by adding these two lines after the `synth_head` assertion (line 16):

```python
    assert cfg.graph_compact is False
    assert cfg.graph_compact_min_idf == 0.5
```

Append a new test function at the end of the file:

```python
def test_graph_compact_env_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CHAT_GRAPH_COMPACT", "1")
    monkeypatch.setenv("MEMO_CHAT_GRAPH_COMPACT_MIN_IDF", "0.7")
    cfg = ChatConfig.load(tmp_path)
    assert cfg.graph_compact is True
    assert cfg.graph_compact_min_idf == 0.7
```

In `tests/test_flags.py`, replace the body of `test_chat_config_vars_not_flagged_unknown` (lines 205-210):

```python
def test_chat_config_vars_not_flagged_unknown() -> None:
    # chat/config.py's 11 MEMO_CHAT_* knobs are env-only (read directly, not
    # through this registry) but must not be reported as typos.
    env = {"MEMO_CHAT_BASE_K": "5", "MEMO_CHAT_GRAPH_COMPACT": "1"}
    assert flags.unknown_memo_vars(env=env) == []
    assert flags.validate(env=env) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_chat_config.py tests/test_flags.py::test_chat_config_vars_not_flagged_unknown -v`
Expected: FAIL — `AttributeError: 'ChatConfig' object has no attribute 'graph_compact'` (config tests), and `test_chat_config_vars_not_flagged_unknown` FAILs because `MEMO_CHAT_GRAPH_COMPACT` is not yet in the `owned` exclusion set (`unknown_memo_vars` reports it).

- [ ] **Step 3: Implement**

In `src/memo/chat/config.py`, add two fields to the `ChatConfig` dataclass, after `synth_head: int` (line 43):

```python
    synth_head: int
    graph_compact: bool
    graph_compact_min_idf: float
    feedback_dir: Path
```

And two lines in `ChatConfig.load()`, after the `synth_head=` line (line 59):

```python
            synth_head=_env_int("MEMO_CHAT_SYNTH_HEAD", 8),
            graph_compact=_env_bool("MEMO_CHAT_GRAPH_COMPACT", False),
            graph_compact_min_idf=_env_float("MEMO_CHAT_GRAPH_COMPACT_MIN_IDF", 0.5),
            feedback_dir=chat_root / "feedback",
```

In `src/memo/flags.py`, add two lines to the `owned` set, after `"MEMO_CHAT_SYNTH_HEAD",` (line 304):

```python
        "MEMO_CHAT_SYNTH_HEAD",
        "MEMO_CHAT_GRAPH_COMPACT",
        "MEMO_CHAT_GRAPH_COMPACT_MIN_IDF",
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_chat_config.py tests/test_flags.py -v`
Expected: PASS

- [ ] **Step 5: Full suite sanity check**

Run: `uv run --no-sync ruff check src/memo/chat/config.py src/memo/flags.py && uv run --no-sync mypy src/memo/chat/config.py src/memo/flags.py`
Expected: no errors

- [ ] **Step 6: Commit**

```bash
git add src/memo/chat/config.py src/memo/flags.py tests/test_chat_config.py tests/test_flags.py
git commit -m "feat(chat): add MEMO_CHAT_GRAPH_COMPACT config knobs"
```

---

### Task 3: Wire compaction into the pipeline and render `related_ids`

**Files:**
- Modify: `src/memo/chat/pipeline.py:163`
- Modify: `src/memo/chat/synthesis.py:23-43` (`build_messages`)
- Test: `tests/test_chat_pipeline.py`
- Test: `tests/test_chat_synthesis.py`

**Interfaces:**
- Consumes: `compact_by_entity_overlap` (Task 1), `ChatConfig.graph_compact`/`graph_compact_min_idf` (Task 2).
- Produces: no new public interface — this is the integration point; `chat_stream`'s `context`/`done` events now carry compacted sources when the flag is on.

- [ ] **Step 1: Write the failing tests**

In `tests/test_chat_synthesis.py`, append:

```python
def test_build_messages_includes_related_ids() -> None:
    src = _src("a", 1.0, related_ids=[("b", "Nota B")])
    messages = build_messages("¿pregunta?", [src], today="03/08/2026")
    assert "(+1 related: Nota B (b))" in messages[1]["content"]


def test_build_messages_omits_related_line_when_absent() -> None:
    messages = build_messages("¿pregunta?", [_src("a", 1.0)], today="03/08/2026")
    assert "related:" not in messages[1]["content"]
```

In `tests/test_chat_pipeline.py`, append:

```python
class _FakeGraph:
    def memory_entities(self, memory_id):
        if memory_id in ("m1", "m2"):
            return [{"name": "proyecto-omega", "type": "topic", "mention_count": 1}]
        return []

    def total_indexed_memories(self):
        return 10

    def entity_doc_freqs(self, names):
        return {"proyecto-omega": 1.0} if "proyecto-omega" in names else {}


class _TwoNoteMemory(_FakeMemory):
    def __init__(self, tmp_path):
        super().__init__(tmp_path)
        self.graph = _FakeGraph()

    def search(self, query, *, limit=None, mode="hybrid", **kw):
        return [
            _FakeRecord(
                id="m1", title="Nota uno", type="note", score=0.9,
                body="cuerpo uno sobre proyecto omega", path="notes/uno.md",
            ),
            _FakeRecord(
                id="m2", title="Nota dos", type="note", score=0.85,
                body="cuerpo dos sobre proyecto omega", path="notes/dos.md",
            ),
        ]


def test_graph_compact_collapses_related_sources_when_enabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_CHAT_GRAPH_COMPACT", "1")

    events = list(chat_stream(_TwoNoteMemory(tmp_path), "qué sabés del proyecto omega?"))
    context = next(e for e in events if e["type"] == "context")
    ids = [s["id"] for s in context["sources"]]

    assert ids.count("m1") + ids.count("m2") == 1  # one survivor, not both
    assert "r1" in ids
    survivor = next(s for s in context["sources"] if s["id"] in ("m1", "m2"))
    assert len(survivor.get("related_ids") or []) == 1


def test_graph_compact_noop_when_disabled(tmp_path) -> None:
    events = list(chat_stream(_TwoNoteMemory(tmp_path), "qué sabés del proyecto omega?"))
    context = next(e for e in events if e["type"] == "context")
    ids = {s["id"] for s in context["sources"]}
    assert {"m1", "m2", "r1"} <= ids
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest tests/test_chat_synthesis.py tests/test_chat_pipeline.py -v`
Expected: FAIL — synthesis tests fail on missing `related_ids` rendering (`AssertionError`); `test_graph_compact_collapses_related_sources_when_enabled` fails because both `m1` and `m2` are still present (compaction not wired in yet).

- [ ] **Step 3: Implement — synthesis rendering**

In `src/memo/chat/synthesis.py`, replace `build_messages` (lines 23-43):

```python
def _format_source(index: int, source: dict[str, Any]) -> str:
    block = f"[{index + 1}] {source.get('title', '')}\n{source.get('snippet', '')}"
    related = source.get("related_ids")
    if related:
        pointer = ", ".join(f"{title} ({sid})" for sid, title in related)
        block += f"\n(+{len(related)} related: {pointer})"
    return block


def build_messages(
    question: str, sources: list[dict[str, Any]], *, today: str
) -> list[dict[str, str]]:
    header = (
        "Sos un asistente RAG de precisión alta. Respondés EXCLUSIVAMENTE con "
        "información que aparece en los SNIPPETS del mensaje del usuario.\n\n"
        f"Fecha actual: {today}. Usá esta fecha para calcular edades y tiempos exactos."
    )
    rules = (
        "Reglas:\n"
        "- Prosa clara; un párrafo por aspecto; sin marcadores [n] ni citas numeradas.\n"
        "- No agregues conocimiento externo a los SNIPPETS.\n"
        f'- Si los SNIPPETS no responden la pregunta, respondé exactamente: "{REFUSAL}"'
    )
    snippets = "\n\n".join(_format_source(i, s) for i, s in enumerate(sources))
    return [
        {"role": "system", "content": f"{header}\n\n{rules}"},
        {"role": "user", "content": f"PREGUNTA: {question}\n\nSNIPPETS:\n{snippets}"},
    ]
```

- [ ] **Step 4: Implement — pipeline wiring**

In `src/memo/chat/pipeline.py`, insert before line 163 (`head = filter_by_relevance(sources, floor=cfg.relevance_floor)[: cfg.synth_head]`):

```python
    if cfg.graph_compact:
        from memo.chat.graph_compact import compact_by_entity_overlap

        sources = compact_by_entity_overlap(
            sources, memory, min_idf_overlap=cfg.graph_compact_min_idf
        )

    head = filter_by_relevance(sources, floor=cfg.relevance_floor)[: cfg.synth_head]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest tests/test_chat_synthesis.py tests/test_chat_pipeline.py -v`
Expected: PASS

- [ ] **Step 6: Run the full chat test surface + lint/type-check**

Run: `uv run --no-sync pytest tests/test_chat_*.py tests/test_cli_chat*.py -v && uv run --no-sync ruff check src/memo/chat/ && uv run --no-sync mypy src/memo/chat/`
Expected: PASS, no lint/type errors — confirms this change didn't regress the rest of the chat surface (fulldoc, dedup, feedback, fusion, sessions, HTTP).

- [ ] **Step 7: Commit**

```bash
git add src/memo/chat/pipeline.py src/memo/chat/synthesis.py tests/test_chat_pipeline.py tests/test_chat_synthesis.py
git commit -m "feat(chat): wire graph-aware source compaction into synthesis"
```

---

### Task 4: Measure against the eval harness (manual gate, not automated)

This task is a **measurement checkpoint**, not new code — per the spec's success criteria, `MEMO_CHAT_GRAPH_COMPACT` must not regress `memo eval chat` before anyone flips it on for real use.

**Files:** none (read-only verification against the live corpus).

- [ ] **Step 1: Baseline run (flag off)**

Run: `uv run --no-sync memo eval chat`
Record: pass-rate, p50/p95 latency.

- [ ] **Step 2: Run with the flag on**

Run: `MEMO_CHAT_GRAPH_COMPACT=1 uv run --no-sync memo eval chat`
Record: pass-rate, p50/p95 latency.

- [ ] **Step 3: Compare and decide**

If pass-rate is unchanged or improved and latency is not meaningfully worse: the flag is safe to enable manually (`MEMO_CHAT_GRAPH_COMPACT=1` in the environment, or wired into a LaunchAgent's `EnvironmentVariables` like other human-graduated flags). If pass-rate regresses: do not enable; revisit `MEMO_CHAT_GRAPH_COMPACT_MIN_IDF` (raise it — stricter overlap requirement) or treat the mechanism as another measured no-op, matching the project's existing precedent for graph experiments that didn't pay off. This is a human decision — no auto-tuner grid-search wiring is in scope for this plan (see spec's "Out of scope").

- [ ] **Step 4: If graduated on, monitor real token savings**

`memo eval chat` proves correctness didn't regress; it doesn't measure token savings directly (its corpus is synthetic Q&A, not live chat traffic). To close the loop on the spec's actual goal, after enabling the flag for real use, compare `memo tokens --json` output (backed by `token_meter`'s per-session ledger) from before enabling vs. a window after — look for a drop in average injected/answer tokens per chat turn. This is an after-the-fact monitoring step, not a gate — the flag stays enabled based on Step 3's `memo eval chat` comparison regardless of how long real usage takes to accumulate a measurable `memo tokens` delta.

- [ ] **Step 5: Record the outcome**

If graduated on: note it in a follow-up commit message or memo entry (not part of this codebase). If left off: no further action — the flag ships default-off regardless, so leaving it there is a valid, complete outcome.
