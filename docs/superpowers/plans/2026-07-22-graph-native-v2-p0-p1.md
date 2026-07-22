# Graph-Native v2 P0+P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a versioned curated graph projection and use it as one bounded, evidence-aware reranking signal across memo retrieval.

**Architecture:** Keep current entity tables as rebuildable raw evidence, add extractor provenance and a separate versioned serving projection in `graph.db`, and expose that projection through a narrow read model. Replace graph candidate injection, post-search expansion, and recall-only score mutation with deterministic weighted reciprocal-rank ordering over candidates that already passed normal eligibility gates.

**Tech Stack:** Python 3.11+, SQLite/WAL, Click, Pydantic-backed memo configuration, pytest, ruff, mypy.

## Global Constraints

- Markdown files remain the source of truth; SQLite remains rebuildable derived state.
- New behavioral flags must be registered in `src/memo/flags.py` domain modules and read only through `flag_bool`, `flag_int`, or `flag_float`.
- Persistent activation must use `memo config set` and `graph-config.md` / `recall-config.md`; do not use shell exports, raw environment reads, or source-default activation.
- Graph serving must fail open to the exact base ordering when disabled, missing, stale, over budget, or malformed.
- P1 may reorder only candidates already eligible through normal retrieval; it may not inject graph-only candidates.
- Keep MLX imports deferred inside functions.
- Tests must use isolated `Config` / `tmp_cfg` and never touch the real vault or state directory.
- CI verification order is ruff, mypy, pytest; retrieval changes also require the real recall eval.
- Commit each completed task independently and include only files listed by that task.

---

## File Map

- Create `src/memo/graph_projection.py`: projection schema, stable URIs, quality policy, atomic builder, read model, health report.
- Create `src/memo/memory/graph_ops.py`: `Memory`-level rebuild and freshness API that joins graph evidence with live store metadata.
- Modify `src/memo/graph.py`: provenance migration, provenance-aware writes, dirty-state marking, and projection composition.
- Modify `src/memo/graph_signal.py`: projection-only signal calculation, evidence traces, weighted reciprocal-rank ordering.
- Modify `src/memo/graph_reason.py`: structured projection/evidence explanation rendering.
- Modify `src/memo/memory/facade.py`: compose `_GraphOpsMixin` into the public `Memory` API.
- Modify `src/memo/memory/_base.py`: declare graph-operation seams used by sibling mixins.
- Modify `src/memo/memory/write_ops.py`: label regex and explicit extraction provenance.
- Modify `src/memo/memory/maintain_ops.py`: let typed LLM extraction upgrade regex rows and label LLM provenance.
- Modify `src/memo/memory/search_ops.py`: apply one final graph ordering pass and remove graph-only candidate/expansion serving.
- Modify `src/memo/memory/search_scoring_ops.py`: delete obsolete graph candidate and expansion helpers.
- Create `src/memo/flags_graph.py`: graph-domain flags and compatibility flags.
- Modify `src/memo/flags.py`, `src/memo/flags_search.py`, and `src/memo/flags_behavior.py`: register graph-domain specs once and remove duplicate ownership.
- Modify `src/memo/tui/config/catalog.py`: map graph flags to `graph-config.md` and accept legacy search paths while reading existing config.
- Modify `src/memo/config_md.py`: normalize legacy graph configuration keys without losing source attribution.
- Modify `src/memo/cli_graph.py`: full rebuild, projection-aware stats, and JSON diagnostics.
- Modify `src/memo/cli_dream.py` and `src/memo/cli_dream_passes.py`: rebuild dirty/stale projections during nightly maintenance.
- Modify `src/memo/recall_logic.py`: remove the second recall-only graph boost; recall receives the same ordering through `Memory.search`.
- Modify `src/memo/dream_tune.py` and `src/memo/dream_flags.py`: retire candidate-injection tuning and expose only curated signal/alpha tuning.
- Modify `eval/regression_labels.json`: add durable graph-decision queries that exercise rare-entity ordering and code-shaped noise suppression.
- Update focused tests under `tests/test_graph_*.py`, `tests/test_search_graph_signal.py`, `tests/test_cli_graph.py`, `tests/test_config_md.py`, `tests/test_cli_config.py`, `tests/test_recall_associative.py`, and `tests/test_dream_tune.py`.
- Update `docs/configuration.md` and `src/memo/experimental_index.md`: document projection lifecycle, activation keys, and retired experimental paths.

---

### Task 1: Register Graph-Domain Configuration With Legacy Read Compatibility

**Files:**
- Create: `src/memo/flags_graph.py`
- Modify: `src/memo/flags.py`
- Modify: `src/memo/flags_search.py`
- Modify: `src/memo/flags_behavior.py`
- Modify: `src/memo/tui/config/catalog.py`
- Modify: `src/memo/config_md.py`
- Test: `tests/test_flags.py`
- Test: `tests/test_config_catalog.py`
- Test: `tests/test_config_md.py`
- Test: `tests/test_cli_config.py`

**Interfaces:**
- Produces: registered keys `graph.projection_enabled`, `graph.projection_min_quality`, `graph.projection_max_age_hours`, `graph.signal_enabled`, `graph.reason_enabled`, `graph.semantic_relations`, `graph.hub_suppression`, `graph.hub_max_doc_freq_ratio`, `graph.min_entity_idf`, `graph.signal_budget_ms`, and `graph.signal_alpha`.
- Produces: `LEGACY_PATH_ALIASES: Mapping[str, str]`, used only while loading old Markdown keys such as `search.graph_signal_enabled`.
- Preserves: existing `MEMO_GRAPH_*` environment variable names and safe source defaults.

- [ ] **Step 1: Write failing registry and Markdown routing tests**

```python
def test_graph_flags_are_routed_to_graph_domain() -> None:
    from memo.flags import REGISTRY
    from memo.tui.config.catalog import domain_file_for_key, path_to_env

    assert REGISTRY["MEMO_GRAPH_PROJECTION_ENABLED"].group == "graph"
    assert path_to_env()["graph.signal_enabled"] == "MEMO_GRAPH_SIGNAL_ENABLED"
    assert domain_file_for_key("graph.signal_enabled") == "graph-config.md"


def test_legacy_search_graph_key_loads_as_canonical_graph_key(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    path = cfg / "search-config.md"
    path.write_text("```toml\n[search]\ngraph_signal_enabled = true\n```\n", encoding="utf-8")

    values = config_md.load_values({"MEMO_CONFIG_DIR": str(home)})

    assert values["graph.signal_enabled"].value is True
    assert values["graph.signal_enabled"].file == str(path)


def test_config_set_graph_signal_writes_graph_file(isolated_config_home, runner) -> None:
    result = runner.invoke(cli, ["config", "set", "graph.signal_enabled", "on"])
    assert result.exit_code == 0
    path = isolated_config_home / "config" / "graph-config.md"
    assert "signal_enabled = true" in path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the focused tests and confirm the missing keys fail**

Run: `uv run --no-sync pytest tests/test_flags.py tests/test_config_catalog.py tests/test_config_md.py tests/test_cli_config.py -q`

Expected: FAIL because projection/alpha specs do not exist and graph signal still belongs to the search group.

- [ ] **Step 3: Add one graph flag module and canonical legacy-key normalization**

```python
# src/memo/flags_graph.py
from __future__ import annotations

from memo.flags_base import FlagSpec, _spec

SPECS: tuple[FlagSpec, ...] = (
    _spec("MEMO_GRAPH_PROJECTION_ENABLED", "bool", False, "graph", "Serve retrieval graph reads from the active curated projection."),
    _spec("MEMO_GRAPH_PROJECTION_MIN_QUALITY", "float", 0.45, "graph", "Minimum deterministic node quality admitted to the serving projection.", min_val=0.0, max_val=1.0),
    _spec("MEMO_GRAPH_PROJECTION_MAX_AGE_HOURS", "int", 36, "graph", "Maximum active projection age before graph serving fails open.", min_val=1),
    _spec("MEMO_GRAPH_SIGNAL_ENABLED", "bool", False, "graph", "Reorder eligible retrieval hits with the curated graph signal."),
    _spec("MEMO_GRAPH_REASON_ENABLED", "bool", False, "graph", "Attach structured graph evidence to graph-touched hits."),
    _spec("MEMO_GRAPH_SEMANTIC_RELATIONS", "bool", False, "graph", "Include stored semantic relations in graph explanations."),
    _spec("MEMO_GRAPH_HUB_SUPPRESSION", "bool", True, "graph", "Prevent non-explicit high-document-frequency hubs from affecting ranking."),
    _spec("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO", "float", 0.25, "graph", "Document-frequency ratio that marks a projected entity as a hub.", min_val=0.0, max_val=1.0),
    _spec("MEMO_GRAPH_MIN_ENTITY_IDF", "float", 0.5, "graph", "Minimum query-entity IDF allowed to contribute graph signal.", min_val=0.0),
    _spec("MEMO_GRAPH_SIGNAL_BUDGET_MS", "int", 150, "graph", "Hard graph-signal deadline in milliseconds.", min_val=0),
    _spec("MEMO_GRAPH_SIGNAL_ALPHA", "float", 0.15, "graph", "Curated graph rank weight in weighted reciprocal-rank fusion.", min_val=0.0, max_val=0.5),
    _spec("MEMO_GRAPH_RETRIEVAL_ENABLED", "bool", False, "graph", "Deprecated compatibility switch; graph-only candidate injection is not served."),
    _spec("MEMO_GRAPH_FALLBACK_MIN_HITS", "int", 0, "graph", "Deprecated compatibility value; graph-only fallback is not served.", min_val=0),
    _spec("MEMO_GRAPH_DENSITY_BOOST", "float", 0.0, "graph", "Deprecated compatibility value; density score mutation is not served.", min_val=0.0),
    _spec("MEMO_GRAPH_EXPANSION_ENABLED", "bool", False, "graph", "Deprecated compatibility switch; graph-only expansion is not served."),
)
```

Register `_graph_specs` once in `_SPECS`, remove the same env names from search/behavior spec tuples, add `LEGACY_PATH_ALIASES` in `config_md.py`, and normalize a parsed path before validation/last-write-wins insertion:

```python
LEGACY_PATH_ALIASES = {
    "search.graph_signal_enabled": "graph.signal_enabled",
    "search.graph_reason_enabled": "graph.reason_enabled",
    "search.graph_semantic_relations": "graph.semantic_relations",
    "search.graph_hub_suppression": "graph.hub_suppression",
    "search.graph_hub_max_doc_freq_ratio": "graph.hub_max_doc_freq_ratio",
    "search.graph_min_entity_idf": "graph.min_entity_idf",
    "search.graph_signal_budget_ms": "graph.signal_budget_ms",
    "search.graph_expansion_enabled": "graph.expansion_enabled",
    "behavior.graph_retrieval_enabled": "graph.retrieval_enabled",
    "behavior.graph_fallback_min_hits": "graph.fallback_min_hits",
    "behavior.graph_density_boost": "graph.density_boost",
}


def _canonical_path_key(path_key: str) -> str:
    return LEGACY_PATH_ALIASES.get(path_key, path_key)
```

- [ ] **Step 4: Run focused tests and config validation**

Run: `uv run --no-sync pytest tests/test_flags.py tests/test_config_catalog.py tests/test_config_md.py tests/test_cli_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the graph configuration domain**

```bash
git add src/memo/flags_graph.py src/memo/flags.py src/memo/flags_search.py src/memo/flags_behavior.py src/memo/tui/config/catalog.py src/memo/config_md.py tests/test_flags.py tests/test_config_catalog.py tests/test_config_md.py tests/test_cli_config.py
git commit -m "feat: register curated graph configuration"
```

---

### Task 2: Add Extraction Provenance and Typed Upgrade Semantics

**Files:**
- Modify: `src/memo/graph.py`
- Modify: `src/memo/memory/write_ops.py`
- Modify: `src/memo/memory/maintain_ops.py`
- Test: `tests/test_graph_store.py`
- Test: `tests/test_save_search_roundtrip.py`

**Interfaces:**
- Produces: `GraphStore.record_extraction(..., extractor: str = "explicit", extractor_version: str = "1", confidence: float = 0.95) -> int`.
- Produces: `GraphStore.memory_extraction_provenance(memory_id: str) -> set[str]`.
- Produces: `GraphStore.mark_projection_dirty() -> None` and `GraphStore.projection_dirty() -> bool`.
- Consumes later: projection builder reads `entity_memory.extractor`, `extractor_version`, `confidence`, and `updated_at`.

- [ ] **Step 1: Write failing migration and upgrade tests**

```python
def test_legacy_entity_memory_rows_gain_conservative_provenance(tmp_path: Path) -> None:
    db = tmp_path / "graph.db"
    cx = sqlite3.connect(db)
    cx.executescript(
        "CREATE TABLE entities (id INTEGER PRIMARY KEY, name TEXT, type TEXT, mention_count INTEGER, first_seen TEXT, last_seen TEXT, UNIQUE(name,type));"
        "CREATE TABLE entity_memory (entity_id INTEGER, memory_id TEXT, occurrences INTEGER, extracted_at TEXT, UNIQUE(entity_id,memory_id));"
        "INSERT INTO entities VALUES (1,'memo','project',1,'2026-01-01','2026-01-01');"
        "INSERT INTO entity_memory VALUES (1,'m1',1,'2026-01-01T00:00:00Z');"
    )
    cx.commit()
    cx.close()

    graph = GraphStore(db)
    row = graph._conn.execute("SELECT extractor, confidence FROM entity_memory").fetchone()
    assert tuple(row) == ("legacy", 0.35)


def test_llm_extraction_replaces_regex_membership_and_marks_dirty(tmp_path: Path) -> None:
    graph = GraphStore(tmp_path / "graph.db")
    graph.record_extraction(memory_id="m1", memory_date="2026-01-01", entities=[{"name": "FastAPI", "type": "concept"}], extracted_at="2026-01-01T00:00:00Z", extractor="regex", confidence=0.45)
    graph.record_extraction(memory_id="m1", memory_date="2026-01-01", entities=[{"name": "FastAPI", "type": "technology"}], extracted_at="2026-01-02T00:00:00Z", extractor="llm", extractor_version="helper-v1", confidence=0.85)

    assert graph.memory_entities("m1") == [{"name": "fastapi", "type": "technology", "mention_count": 1}]
    assert graph.memory_extraction_provenance("m1") == {"llm"}
    assert graph.projection_dirty() is True


def test_entity_backfill_does_not_skip_regex_only_memory(mem_with_stub, monkeypatch) -> None:
    rec = mem_with_stub.save(content="FastAPI service architecture", title="Service", type_="note")
    assert mem_with_stub.graph.memory_extraction_provenance(rec.id) == {"regex"}
    monkeypatch.setattr(mem_with_stub, "_ensure_chat", lambda: FakeEntityChat("FastAPI", "technology"))

    counts = mem_with_stub.extract_entities(ids=[rec.id], skip_already_indexed=True)

    assert counts["processed"] == 1
    assert mem_with_stub.graph.memory_extraction_provenance(rec.id) == {"llm"}
```

- [ ] **Step 2: Run tests and confirm schema/provenance failures**

Run: `uv run --no-sync pytest tests/test_graph_store.py tests/test_save_search_roundtrip.py -q`

Expected: FAIL because the columns and provenance APIs are absent.

- [ ] **Step 3: Migrate columns idempotently and write provenance**

After `_SCHEMA_DDL`, inspect `PRAGMA table_info(entity_memory)` and add missing columns inside the existing initialization transaction:

```python
defaults = {
    "extractor": "TEXT NOT NULL DEFAULT 'legacy'",
    "extractor_version": "TEXT NOT NULL DEFAULT '0'",
    "confidence": "REAL NOT NULL DEFAULT 0.35",
    "updated_at": "TEXT",
}
columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(entity_memory)")}
for name, ddl in defaults.items():
    if name not in columns:
        self._conn.execute(f"ALTER TABLE entity_memory ADD COLUMN {name} {ddl}")
self._conn.execute("UPDATE entity_memory SET updated_at = extracted_at WHERE updated_at IS NULL")
```

Clamp confidence and persist all fields in `record_extraction`; every raw membership mutation writes `graph_projection_state.dirty = '1'`. Save-time auto extraction calls with `extractor="regex"`, while caller-provided `extra.entities` calls with `extractor="explicit"`. LLM maintenance calls with `extractor="llm"` and changes its skip filter to:

```python
if skip_already_indexed:
    target = [
        memory_id
        for memory_id in target
        if not (self.graph.memory_extraction_provenance(memory_id) & {"llm", "explicit"})
    ]
```

- [ ] **Step 4: Run provenance tests**

Run: `uv run --no-sync pytest tests/test_graph_store.py tests/test_save_search_roundtrip.py -q`

Expected: PASS.

- [ ] **Step 5: Commit provenance support**

```bash
git add src/memo/graph.py src/memo/memory/write_ops.py src/memo/memory/maintain_ops.py tests/test_graph_store.py tests/test_save_search_roundtrip.py
git commit -m "feat: track graph extraction provenance"
```

---

### Task 3: Implement Stable URIs and Deterministic Projection Quality

**Files:**
- Create: `src/memo/graph_projection.py`
- Test: `tests/test_graph_projection.py`

**Interfaces:**
- Produces: `entity_uri(entity_type: str, name: str) -> str`.
- Produces: `memory_uri(memory_id: str) -> str` and `fact_uri(fact_id: str) -> str`.
- Produces: immutable `ProjectionMemoryState`, `RawEntityEvidence`, `ProjectionDecision`, `ProjectionBuildConfig`.
- Produces: `evaluate_entity(evidence: RawEntityEvidence, config: ProjectionBuildConfig) -> ProjectionDecision`.

- [ ] **Step 1: Write URI and hard-rejection tests**

```python
@pytest.mark.parametrize(
    ("type_", "name", "expected"),
    [
        ("technology", "Fast API", "entity://technology/fastapi"),
        ("project", "Postgres", "entity://project/postgresql"),
        ("person", "José Núñez", "entity://person/josenunez"),
    ],
)
def test_entity_uri_is_stable(type_: str, name: str, expected: str) -> None:
    assert entity_uri(type_, name) == expected


@pytest.mark.parametrize("name", ["", "true", "42", "2026-07-22", "test_rank_hits", "assert foo == bar"])
def test_projection_rejects_non_knowledge_shapes(name: str) -> None:
    evidence = raw_evidence(name=name, extractor="regex", memories=(durable_memory("m1"),))
    decision = evaluate_entity(evidence, ProjectionBuildConfig())
    assert decision.eligible is False
    assert decision.reason


def test_explicit_non_code_evidence_can_keep_test_shaped_real_name() -> None:
    evidence = raw_evidence(name="Test Kitchen", extractor="explicit", confidence=0.95, memories=(durable_memory("m1"),))
    decision = evaluate_entity(evidence, ProjectionBuildConfig())
    assert decision.eligible is True


def test_reference_only_regex_concept_falls_below_quality_floor() -> None:
    evidence = raw_evidence(name="fixture helper", extractor="regex", memories=(reference_memory("r1"),))
    decision = evaluate_entity(evidence, ProjectionBuildConfig(min_quality=0.45))
    assert decision.eligible is False
    assert decision.reason == "quality_below_threshold"
```

- [ ] **Step 2: Run the new test file and confirm import failure**

Run: `uv run --no-sync pytest tests/test_graph_projection.py -q`

Expected: FAIL with `ModuleNotFoundError: memo.graph_projection`.

- [ ] **Step 3: Implement stable types and quality policy**

```python
@dataclass(frozen=True)
class ProjectionMemoryState:
    id: str
    type: str
    forgotten: bool = False


@dataclass(frozen=True)
class RawEntityEvidence:
    entity_id: int
    name: str
    entity_type: str
    extractors: tuple[str, ...]
    confidences: tuple[float, ...]
    memories: tuple[ProjectionMemoryState, ...]
    alias_count: int = 0


@dataclass(frozen=True)
class ProjectionBuildConfig:
    min_quality: float = 0.45
    hub_max_doc_freq_ratio: float = 0.25
    evidence_limit: int = 8


@dataclass(frozen=True)
class ProjectionDecision:
    eligible: bool
    quality: float
    reason: str | None
    uri: str


def entity_uri(entity_type: str, name: str) -> str:
    return f"entity://{entity_type.strip().lower()}/{quote(fold_key(name), safe='')}"


def evaluate_entity(evidence: RawEntityEvidence, config: ProjectionBuildConfig) -> ProjectionDecision:
    live = tuple(memory for memory in evidence.memories if not memory.forgotten)
    key = fold_key(evidence.name)
    uri = entity_uri(evidence.entity_type, evidence.name)
    explicit = "explicit" in evidence.extractors
    if not live:
        return ProjectionDecision(False, 0.0, "no_live_memory", uri)
    if not key:
        return ProjectionDecision(False, 0.0, "empty_key", uri)
    if _is_scalar_or_date(evidence.name):
        return ProjectionDecision(False, 0.0, "scalar_or_date", uri)
    if _is_code_shape(evidence.name) and not explicit:
        return ProjectionDecision(False, 0.0, "code_shape", uri)
    avg_confidence = sum(evidence.confidences) / max(1, len(evidence.confidences))
    durable_ratio = sum(memory.type not in {"reference"} for memory in live) / len(live)
    provenance = max((_EXTRACTOR_WEIGHT.get(value, 0.0) for value in evidence.extractors), default=0.0)
    quality = min(1.0, 0.35 * avg_confidence + 0.25 * provenance + 0.15 * durable_ratio + min(0.2, 0.05 * len(live)) + (0.05 if evidence.entity_type != "concept" else 0.0))
    if quality < config.min_quality:
        return ProjectionDecision(False, quality, "quality_below_threshold", uri)
    return ProjectionDecision(True, quality, None, uri)
```

Keep `_is_scalar_or_date`, `_is_code_shape`, and `_EXTRACTOR_WEIGHT` pure and dependency-free in the same module.

- [ ] **Step 4: Run quality tests and lint the new module**

Run: `uv run --no-sync pytest tests/test_graph_projection.py -q && uv run --no-sync ruff check src/memo/graph_projection.py tests/test_graph_projection.py`

Expected: PASS.

- [ ] **Step 5: Commit projection policy**

```bash
git add src/memo/graph_projection.py tests/test_graph_projection.py
git commit -m "feat: define curated graph projection policy"
```

---

### Task 4: Build and Atomically Activate Versioned Projections

**Files:**
- Modify: `src/memo/graph_projection.py`
- Modify: `src/memo/graph.py`
- Test: `tests/test_graph_projection.py`
- Test: `tests/test_graph_store.py`

**Interfaces:**
- Produces: `GraphProjectionStore(conn, tx_factory)` at `GraphStore.projection`.
- Produces: `GraphProjectionStore.rebuild(memories: Mapping[str, ProjectionMemoryState], config: ProjectionBuildConfig, now: datetime | None = None) -> ProjectionBuildResult`.
- Produces: `GraphProjectionStore.read_model(max_age_hours: int, now: datetime | None = None) -> GraphReadModel`.
- Produces: `GraphProjectionStore.health(now: datetime | None = None) -> dict[str, Any]`.
- Produces: immutable `ProjectedNode(uri, label, entity_type, doc_freq, degree, quality, is_hub, idf)` and `ProjectedEdge(source_uri, target_uri, relation, weight, confidence, evidence_ids, first_seen, last_seen)`.
- Produces: `GraphReadModel.resolve_query_entities(query: str)`, `node(uri: str)`, `memory_nodes(memory_id: str)`, `neighbors(uri: str)`, `total_memories`, `version`, and `built_at`.

- [ ] **Step 1: Write failing atomicity, evidence, and stale-read tests**

```python
def test_projection_rebuild_activates_complete_version(graph_with_raw_entities) -> None:
    result = graph_with_raw_entities.projection.rebuild(live_states("m1", "m2"), ProjectionBuildConfig())
    model = graph_with_raw_entities.projection.read_model(max_age_hours=36)

    assert result.activated is True
    assert model.version == result.version
    assert model.memory_nodes("m1")
    edge = next(iter(model.neighbors(entity_uri("technology", "mlx"))))
    assert edge.evidence_ids == ("memory://m1", "memory://m2")


def test_failed_projection_validation_preserves_previous_active_version(graph_with_raw_entities, monkeypatch) -> None:
    first = graph_with_raw_entities.projection.rebuild(live_states("m1", "m2"), ProjectionBuildConfig())
    monkeypatch.setattr(graph_with_raw_entities.projection, "_validate_version", lambda cx, version: (_ for _ in ()).throw(ProjectionBuildError("invalid")))

    with pytest.raises(ProjectionBuildError):
        graph_with_raw_entities.projection.rebuild(live_states("m1", "m2"), ProjectionBuildConfig())

    assert graph_with_raw_entities.projection.health()["active_version"] == first.version


def test_stale_projection_returns_unavailable_read_model(graph_with_raw_entities) -> None:
    built = datetime(2026, 7, 20, tzinfo=UTC)
    graph_with_raw_entities.projection.rebuild(live_states("m1", "m2"), ProjectionBuildConfig(), now=built)
    model = graph_with_raw_entities.projection.read_model(max_age_hours=24, now=built + timedelta(hours=25))
    assert model.available is False
    assert model.skip_reason == "projection_stale"


def test_rebuild_quarantines_rejections_without_deleting_raw_rows(graph_with_pollution) -> None:
    result = graph_with_pollution.projection.rebuild(live_states("m1"), ProjectionBuildConfig())
    assert result.rejected_count == 1
    assert graph_with_pollution.count_entities() == 2
    assert graph_with_pollution.projection.health()["rejection_reasons"]["code_shape"] == 1
```

- [ ] **Step 2: Run projection tests and confirm missing store failures**

Run: `uv run --no-sync pytest tests/test_graph_projection.py tests/test_graph_store.py -q`

Expected: FAIL because projection tables, builder, and read model are absent.

- [ ] **Step 3: Add projection DDL and a single-transaction cutover**

Add the four versioned tables and state table exactly as specified in the design. The rebuild implementation must:

```python
def rebuild(self, memories: Mapping[str, ProjectionMemoryState], config: ProjectionBuildConfig, now: datetime | None = None) -> ProjectionBuildResult:
    built_at = (now or datetime.now(UTC)).isoformat()
    version = uuid.uuid4().hex
    with self._tx_factory() as cx:
        raw = self._load_raw_evidence(cx, memories)
        nodes, rejections = self._decide_nodes(raw, config)
        edges = self._project_edges(cx, nodes, memories, config.evidence_limit)
        self._insert_version(cx, version, built_at, nodes, edges, rejections)
        self._validate_version(cx, version)
        prior = self._state(cx, "active_version")
        self._set_state(cx, "active_version", version)
        self._set_state(cx, "dirty", "0")
        self._set_state(cx, "last_success_at", built_at)
        self._set_state(cx, "last_error", "")
        self._retain_active_and_previous(cx, version, prior)
    return ProjectionBuildResult(version, True, len(nodes), len(edges), len(rejections), built_at)
```

Use sorted node/edge/evidence input before hashing/inserting, parameterized SQL everywhere, bounded JSON evidence lists, and a `ProjectionBuildError(MemoError)` for validation failures. Construct `GraphStore.projection` only after schema migration:

```python
from memo.graph_projection import GraphProjectionStore

self.projection = GraphProjectionStore(self._conn, self._tx)
```

- [ ] **Step 4: Run projection/store tests including legacy DB migration**

Run: `uv run --no-sync pytest tests/test_graph_projection.py tests/test_graph_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit atomic projection storage**

```bash
git add src/memo/graph_projection.py src/memo/graph.py tests/test_graph_projection.py tests/test_graph_store.py
git commit -m "feat: materialize versioned graph projections"
```

---

### Task 5: Add the Public Rebuild/Health API and CLI Surface

**Files:**
- Create: `src/memo/memory/graph_ops.py`
- Modify: `src/memo/memory/facade.py`
- Modify: `src/memo/memory/_base.py`
- Modify: `src/memo/cli_graph.py`
- Test: `tests/test_cli_graph.py`
- Test: `tests/test_graph_projection.py`

**Interfaces:**
- Produces: `Memory.rebuild_graph() -> GraphRebuildResult`.
- Produces: `Memory.graph_health() -> dict[str, Any]`.
- Produces: `Memory.rebuild_graph_if_due() -> GraphRebuildResult | None`.
- CLI: `memo graph rebuild [--json]` and `memo graph stats [--json]`.

- [ ] **Step 1: Write failing facade and CLI tests**

```python
def test_memory_rebuild_graph_prunes_orphans_and_activates_projection(mem_with_stub) -> None:
    rec = mem_with_stub.save(content="MLX daemon knowledge", title="MLX", type_="decision")
    mem_with_stub.graph.record_extraction(memory_id="gone", memory_date="2026-01-01", entities=[{"name": "orphan", "type": "concept"}], extracted_at="2026-01-01T00:00:00Z")

    result = mem_with_stub.rebuild_graph()

    assert result.orphan_links_pruned == 1
    assert result.projection.activated is True
    assert mem_with_stub.graph.entity_memories("orphan") == []
    assert mem_with_stub.graph.projection.read_model(36).memory_nodes(rec.id)


def test_graph_stats_json_reports_projection(runner, isolated_memory_env) -> None:
    rebuild = runner.invoke(cli, ["graph", "rebuild", "--json"], env=isolated_memory_env)
    stats = runner.invoke(cli, ["graph", "stats", "--json"], env=isolated_memory_env)
    assert rebuild.exit_code == 0
    payload = json.loads(stats.output)
    assert payload["projection"]["active_version"]
    assert "rejection_reasons" in payload["projection"]
```

- [ ] **Step 2: Run focused tests and confirm missing API failure**

Run: `uv run --no-sync pytest tests/test_cli_graph.py tests/test_graph_projection.py -q`

Expected: FAIL because the facade and JSON options do not exist.

- [ ] **Step 3: Implement graph operations over live store metadata**

```python
@dataclass(frozen=True)
class GraphRebuildResult:
    orphan_links_pruned: int
    entities_merged: int
    raw_edges: int
    projection: ProjectionBuildResult


class _GraphOpsMixin(_MemoryBase):
    def _projection_memory_states(self) -> dict[str, ProjectionMemoryState]:
        states: dict[str, ProjectionMemoryState] = {}
        for row in self.store.list_recent(limit=100_000):
            extra = row.get("extra") or {}
            states[row["id"]] = ProjectionMemoryState(
                id=row["id"],
                type=str(row.get("type") or "note"),
                forgotten=bool(extra.get(IS_FORGOTTEN_KEY)),
            )
        return states

    def rebuild_graph(self) -> GraphRebuildResult:
        states = self._projection_memory_states()
        pruned = self.graph.prune_memory_links(set(states))
        merged = self.graph.canonicalize_existing()
        edges = self.graph.rebuild_edges()
        projection = self.graph.projection.rebuild(
            states,
            ProjectionBuildConfig(
                min_quality=cast(float, flag_float("MEMO_GRAPH_PROJECTION_MIN_QUALITY")),
                hub_max_doc_freq_ratio=cast(float, flag_float("MEMO_GRAPH_HUB_MAX_DOC_FREQ_RATIO")),
            ),
        )
        return GraphRebuildResult(pruned, merged, edges, projection)
```

`graph_health()` combines raw `stats()` / `edge_stats()` with `projection.health()`. `rebuild_graph_if_due()` returns `None` unless projection is enabled and health reports dirty, missing, or older than the configured maximum age. Make CLI output the dataclass with `dataclasses.asdict` for JSON and a compact human summary otherwise.

- [ ] **Step 4: Run graph facade/CLI tests**

Run: `uv run --no-sync pytest tests/test_cli_graph.py tests/test_graph_projection.py -q`

Expected: PASS.

- [ ] **Step 5: Commit rebuild and diagnostics surfaces**

```bash
git add src/memo/memory/graph_ops.py src/memo/memory/facade.py src/memo/memory/_base.py src/memo/cli_graph.py tests/test_cli_graph.py tests/test_graph_projection.py
git commit -m "feat: expose curated graph rebuild and health"
```

---

### Task 6: Rebuild Dirty or Stale Projections During Dream Maintenance

**Files:**
- Modify: `src/memo/cli_dream_passes.py`
- Modify: `src/memo/cli_dream.py`
- Test: `tests/test_briefing_dream.py`
- Test: `tests/test_cli_dream_status.py`

**Interfaces:**
- Produces: `_run_graph_projection(mem: Memory, dry_run: bool = False) -> dict[str, Any]`.
- Adds: `graph_projection` receipt with status `disabled`, `fresh`, `would_rebuild`, `rebuilt`, or `error`.

- [ ] **Step 1: Write failing due/fresh/dry-run tests**

```python
def test_dream_graph_projection_rebuilds_dirty_projection(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")
    mem_with_stub.graph.mark_projection_dirty()
    result = _run_graph_projection(mem_with_stub)
    assert result["status"] == "rebuilt"
    assert mem_with_stub.graph.projection_dirty() is False


def test_dream_graph_projection_dry_run_never_mutates(mem_with_stub, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")
    mem_with_stub.graph.mark_projection_dirty()
    result = _run_graph_projection(mem_with_stub, dry_run=True)
    assert result["status"] == "would_rebuild"
    assert mem_with_stub.graph.projection_dirty() is True
```

- [ ] **Step 2: Run dream tests and confirm helper failure**

Run: `uv run --no-sync pytest tests/test_briefing_dream.py tests/test_cli_dream_status.py -q`

Expected: FAIL because `_run_graph_projection` and receipt wiring are absent.

- [ ] **Step 3: Implement and wire the maintenance pass after entity backfill**

```python
def _run_graph_projection(mem: Memory, dry_run: bool = False) -> dict[str, Any]:
    if not flag_bool("MEMO_GRAPH_PROJECTION_ENABLED"):
        return {"status": "disabled"}
    health = mem.graph_health()["projection"]
    due = bool(health["dirty"] or not health["active_version"] or health["stale"])
    if not due:
        return {"status": "fresh", "active_version": health["active_version"]}
    if dry_run:
        return {"status": "would_rebuild", "active_version": health["active_version"]}
    try:
        result = mem.rebuild_graph()
        return {"status": "rebuilt", **asdict(result)}
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
```

Run it after entity backfill so new typed evidence is included. Add the receipt to JSON and human status without adding graph work when the flag is off.

- [ ] **Step 4: Run dream tests**

Run: `uv run --no-sync pytest tests/test_briefing_dream.py tests/test_cli_dream_status.py -q`

Expected: PASS.

- [ ] **Step 5: Commit nightly projection maintenance**

```bash
git add src/memo/cli_dream_passes.py src/memo/cli_dream.py tests/test_briefing_dream.py tests/test_cli_dream_status.py
git commit -m "feat: refresh curated graph during dream"
```

---

### Task 7: Replace Additive Boosts With Projection-Only Rank Fusion

**Files:**
- Modify: `src/memo/graph_signal.py`
- Modify: `src/memo/graph_reason.py`
- Test: `tests/test_graph_signal.py`
- Test: `tests/test_graph_reason.py`

**Interfaces:**
- Produces: `GraphSignalConfig(enabled, alpha, rrf_k, budget_ms, min_entity_idf, hub_suppression, max_age_hours)`.
- Produces: `GraphEvidenceTrace(projection_version, mode, query_nodes, hit_nodes, edges, normalized_signal, hub_suppressed)`.
- Produces: `GraphSignalResult(signals, traces, ordered_ids, skipped, elapsed_ms)`.
- Produces: `collect_graph_signal(read_model: GraphReadModel, query: str, candidate_ids: Sequence[str], *, config: GraphSignalConfig | None = None, deadline: float | None = None) -> GraphSignalResult`.

- [ ] **Step 1: Replace old signal tests with identity, normalization, and evidence tests**

```python
def test_empty_signal_preserves_exact_candidate_order(projected_graph) -> None:
    result = collect_graph_signal(projected_graph, "unknown words", ["m2", "m1"], config=signal_config())
    assert result.ordered_ids == ["m2", "m1"]
    assert result.skipped == "no_query_entities"


def test_curated_signal_reorders_without_adding_candidates(projected_graph) -> None:
    result = collect_graph_signal(projected_graph, "mlx", ["generic", "daemon"], config=signal_config(alpha=0.15))
    assert result.ordered_ids == ["daemon", "generic"]
    assert set(result.ordered_ids) == {"generic", "daemon"}
    assert 0.0 < result.signals["daemon"] <= 1.0


def test_deadline_discards_partial_signal(projected_graph, monkeypatch) -> None:
    result = collect_graph_signal(projected_graph, "mlx", ["generic", "daemon"], config=signal_config(), deadline=time.monotonic() - 1)
    assert result.ordered_ids == ["generic", "daemon"]
    assert result.signals == {}
    assert result.skipped == "deadline"


def test_trace_contains_only_stored_evidence(projected_graph) -> None:
    result = collect_graph_signal(projected_graph, "mlx", ["daemon"], config=signal_config())
    trace = result.traces["daemon"]
    assert trace.projection_version == projected_graph.version
    assert trace.edges[0].evidence_ids == ("memory://m1", "memory://m2")
    assert trace.mode == "curated_proximity"
```

- [ ] **Step 2: Run signal/reason tests and observe old additive API failures**

Run: `uv run --no-sync pytest tests/test_graph_signal.py tests/test_graph_reason.py -q`

Expected: FAIL because the old API returns boosts and raw names.

- [ ] **Step 3: Implement bounded scoring and weighted reciprocal-rank ordering**

```python
def _edge_signal(query: ProjectedNode, hit: ProjectedNode, edge: ProjectedEdge) -> float:
    if query.degree <= 0 or hit.degree <= 0:
        return 0.0
    return (
        math.log1p(edge.weight)
        * query.idf
        * hit.idf
        * edge.confidence
        / math.sqrt(query.degree * hit.degree)
    )


def _fuse_order(candidate_ids: Sequence[str], signals: Mapping[str, float], *, alpha: float, rrf_k: int) -> list[str]:
    if not signals or alpha <= 0:
        return list(candidate_ids)
    base_rank = {memory_id: rank for rank, memory_id in enumerate(candidate_ids, start=1)}
    graph_rank = {
        memory_id: rank
        for rank, (memory_id, _) in enumerate(
            sorted(signals.items(), key=lambda item: (-item[1], base_rank[item[0]])),
            start=1,
        )
    }
    def fused(memory_id: str) -> float:
        base = 1.0 / (rrf_k + base_rank[memory_id])
        graph = alpha / (rrf_k + graph_rank[memory_id]) if memory_id in graph_rank else 0.0
        return base + graph
    return sorted(candidate_ids, key=lambda memory_id: (-fused(memory_id), base_rank[memory_id]))
```

Normalize each candidate from the bounded sum of its strongest distinct edge contributions by dividing by the maximum positive candidate signal. If any deadline check fails, return the original IDs and no traces. `build_graph_reason` serializes projection version, URI nodes, exact stored edge/evidence fields, signal, and `confidence="derived"`; it never reconstructs a path.

- [ ] **Step 4: Run signal and reason tests**

Run: `uv run --no-sync pytest tests/test_graph_signal.py tests/test_graph_reason.py -q`

Expected: PASS.

- [ ] **Step 5: Commit unified graph rank fusion**

```bash
git add src/memo/graph_signal.py src/memo/graph_reason.py tests/test_graph_signal.py tests/test_graph_reason.py
git commit -m "feat: rank eligible hits with curated graph evidence"
```

---

### Task 8: Make Search the Single Serving Integration Point

**Files:**
- Modify: `src/memo/memory/search_ops.py`
- Modify: `src/memo/memory/search_scoring_ops.py`
- Modify: `src/memo/memory/_base.py`
- Test: `tests/test_search_graph_signal.py`
- Test: `tests/test_memory_search.py`
- Test: `tests/test_search_explain.py`

**Interfaces:**
- Consumes: `GraphProjectionStore.read_model()` and `collect_graph_signal()`.
- Guarantees: graph ordering runs after forgotten/type/reference eligibility and after ordinary ranking, but before access/co-recall recording.
- Guarantees: graph pass preserves each `MemoryRecord.score`; only list order and evidence metadata change.
- Removes: graph-only RRF leg and `_apply_graph_expansion` serving behavior.

- [ ] **Step 1: Write failing serving-boundary tests**

```python
def test_search_graph_signal_never_injects_outside_candidate_set(mem_with_stub, monkeypatch) -> None:
    enable_projection_flags(monkeypatch)
    eligible = save_graph_fixture(mem_with_stub, "eligible")
    adjacent_but_not_retrieved = save_graph_fixture(mem_with_stub, "adjacent")
    mem_with_stub.rebuild_graph()
    monkeypatch.setattr(mem_with_stub.store, "search_bm25", lambda *args, **kwargs: [row_for(eligible)])

    hits = mem_with_stub.search("mlx", mode="bm25", limit=5)

    assert [hit.id for hit in hits] == [eligible.id]
    assert adjacent_but_not_retrieved.id not in {hit.id for hit in hits}


def test_graph_order_preserves_scores_used_by_recall_gates(mem_with_stub, monkeypatch) -> None:
    enable_projection_flags(monkeypatch)
    first, second = build_ordering_fixture(mem_with_stub)
    before = {hit.id: hit.score for hit in mem_with_stub.search("mlx", mode="bm25", limit=5)}
    mem_with_stub.rebuild_graph()
    after = mem_with_stub.search("mlx", mode="bm25", limit=5)
    assert {hit.id: hit.score for hit in after} == before
    assert [hit.id for hit in after][:2] == [second.id, first.id]


def test_missing_stale_and_disabled_projection_are_exact_identity(mem_with_stub, monkeypatch) -> None:
    base = mem_with_stub.search("mlx", mode="bm25", limit=5)
    monkeypatch.setenv("MEMO_GRAPH_SIGNAL_ENABLED", "1")
    monkeypatch.setenv("MEMO_GRAPH_PROJECTION_ENABLED", "1")
    enabled_without_projection = mem_with_stub.search("mlx", mode="bm25", limit=5)
    assert [hit.id for hit in enabled_without_projection] == [hit.id for hit in base]
    assert [hit.score for hit in enabled_without_projection] == [hit.score for hit in base]
```

- [ ] **Step 2: Run focused search tests and confirm old-path failures**

Run: `uv run --no-sync pytest tests/test_search_graph_signal.py tests/test_memory_search.py tests/test_search_explain.py -q`

Expected: FAIL because current search mutates scores and can inject graph candidates/expansion.

- [ ] **Step 3: Delete old serving paths and add one final-order helper**

Remove `graph_hits` from hybrid RRF lists, remove the `MEMO_GRAPH_EXPANSION_ENABLED` branch, and delete `_fetch_graph_candidates` / `_apply_graph_expansion` from `_SearchScoringMixin` and `_MemoryBase`. Add this helper near the end of `search()` after reference filtering:

```python
def _apply_curated_graph_order(self, query: str, out: list[MemoryRecord], trace: Callable[..., None]) -> list[MemoryRecord]:
    if not out or not flag_bool("MEMO_GRAPH_PROJECTION_ENABLED") or not flag_bool("MEMO_GRAPH_SIGNAL_ENABLED"):
        return out
    model = self.graph.projection.read_model(cast(int, flag_int("MEMO_GRAPH_PROJECTION_MAX_AGE_HOURS")))
    result = collect_graph_signal(model, query, [record.id for record in out])
    trace("graph_signal", projection_version=model.version, touched_count=len(result.signals), skipped=result.skipped, elapsed_ms=round(result.elapsed_ms, 3))
    if not result.signals:
        return out
    by_id = {record.id: record for record in out}
    ordered = [by_id[memory_id] for memory_id in result.ordered_ids]
    if flag_bool("MEMO_GRAPH_REASON_ENABLED"):
        ordered = [
            replace(record, extra={**(record.extra or {}), "graph_reason": build_graph_reason(record.id, result.traces[record.id])})
            if record.id in result.traces else record
            for record in ordered
        ]
    return ordered
```

If semantic-relation explanations are enabled, fetch relations only for traced IDs and pass those stored rows into `build_graph_reason`. Do not change `score`.

- [ ] **Step 4: Run search tests**

Run: `uv run --no-sync pytest tests/test_search_graph_signal.py tests/test_memory_search.py tests/test_search_explain.py -q`

Expected: PASS.

- [ ] **Step 5: Commit the single search integration**

```bash
git add src/memo/memory/search_ops.py src/memo/memory/search_scoring_ops.py src/memo/memory/_base.py tests/test_search_graph_signal.py tests/test_memory_search.py tests/test_search_explain.py
git commit -m "refactor: unify graph serving in search"
```

---

### Task 9: Remove the Duplicate Recall Proximity Reranker

**Files:**
- Modify: `src/memo/recall_logic.py`
- Modify: `src/memo/graph_proximity.py`
- Modify: `src/memo/flags_recall.py`
- Test: `tests/test_graph_proximity.py`
- Test: `tests/test_recall_logic_synthesis.py`
- Test: `tests/test_recall_hook_parity.py`
- Test: `tests/test_recall_associative.py`

**Interfaces:**
- Preserves: `extract_query_entities` only if raw navigation still imports it; serving signal uses `GraphReadModel.resolve_query_entities`.
- Removes: `graph_boost` argument from `rank_hits` and `MEMO_RECALL_GRAPH_PROXIMITY*` score mutation.
- Preserves: associative recall nudge, which remains independently budgeted and non-ranking.

- [ ] **Step 1: Write a parity test proving recall does not rerank twice**

```python
def test_recall_uses_search_graph_order_once(mem_stub, monkeypatch) -> None:
    ordered = [hit("graph-first", 0.61), hit("base-first", 0.72)]
    monkeypatch.setattr(mem_stub, "search", lambda *args, **kwargs: list(ordered))
    ranked = rank_hits(list(ordered), knobs(min_sim=0.5), query="mlx")
    assert [item.id for item in ranked] == ["graph-first", "base-first"]
    assert [item.score for item in ranked] == [0.61, 0.72]
```

- [ ] **Step 2: Run recall/proximity tests**

Run: `uv run --no-sync pytest tests/test_graph_proximity.py tests/test_recall_logic_synthesis.py tests/test_recall_hook_parity.py tests/test_recall_associative.py -q`

Expected: FAIL until graph boost construction and score sorting are removed.

- [ ] **Step 3: Remove the rank seam and retain associative behavior**

Delete `graph_boost` from `rank_hits`, its call stage, `_graph_boost` construction in `_recall_logic`, and every caller argument. Remove the two recall proximity flag specs after adding legacy config aliases to the graph signal equivalents only for validation:

```python
LEGACY_PATH_ALIASES.update({
    "recall.graph_proximity": "graph.signal_enabled",
    "recall.graph_proximity_weight": "graph.signal_alpha",
})
```

Keep `recall_assoc.build_nudge` unchanged except that codegraph/graph errors must continue returning `[]`.

- [ ] **Step 4: Run recall parity tests**

Run: `uv run --no-sync pytest tests/test_graph_proximity.py tests/test_recall_logic_synthesis.py tests/test_recall_hook_parity.py tests/test_recall_associative.py -q`

Expected: PASS.

- [ ] **Step 5: Commit recall consolidation**

```bash
git add src/memo/recall_logic.py src/memo/graph_proximity.py src/memo/flags_recall.py src/memo/config_md.py tests/test_graph_proximity.py tests/test_recall_logic_synthesis.py tests/test_recall_hook_parity.py tests/test_recall_associative.py
git commit -m "refactor: remove duplicate recall graph reranker"
```

---

### Task 10: Retire Candidate-Injection Tuning and Document the Projection

**Files:**
- Modify: `src/memo/dream_tune.py`
- Modify: `src/memo/dream_flags.py`
- Modify: `tests/test_dream_tune.py`
- Modify: `eval/regression_labels.json`
- Modify: `docs/configuration.md`
- Modify: `src/memo/experimental_index.md`

**Interfaces:**
- Replaces: retrieval graph tuner grid based on candidate injection/expansion.
- Produces: bounded candidates over `MEMO_GRAPH_SIGNAL_ALPHA` values `0.0`, `0.10`, `0.15`, `0.25` with `MEMO_GRAPH_SIGNAL_ENABLED` off/on.
- Documents: graph projection build, fail-open freshness, config keys, and deprecated compatibility flags.

- [ ] **Step 1: Rewrite tuner expectations before implementation**

```python
def test_graph_retrieval_tuner_grids_curated_signal_alpha() -> None:
    candidates = graph_signal_candidates()
    assert candidates == [
        {"MEMO_GRAPH_SIGNAL_ENABLED": False, "MEMO_GRAPH_SIGNAL_ALPHA": 0.0},
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.10},
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.15},
        {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": 0.25},
    ]
    assert all("MEMO_GRAPH_EXPANSION_ENABLED" not in candidate for candidate in candidates)
    assert all("MEMO_GRAPH_RETRIEVAL_ENABLED" not in candidate for candidate in candidates)
```

- [ ] **Step 2: Run tuner tests and confirm old candidate grid failure**

Run: `uv run --no-sync pytest tests/test_dream_tune.py -q`

Expected: FAIL because the tuner still flips retrieval/expansion.

- [ ] **Step 3: Replace the tuner grid and update ownership metadata**

Use a single managed tuple:

```python
_MANAGED_GRAPH_SIGNAL_FLAGS = ("MEMO_GRAPH_SIGNAL_ENABLED", "MEMO_GRAPH_SIGNAL_ALPHA")


def graph_signal_candidates() -> list[dict[str, bool | float]]:
    return [
        {"MEMO_GRAPH_SIGNAL_ENABLED": False, "MEMO_GRAPH_SIGNAL_ALPHA": 0.0},
        *[
            {"MEMO_GRAPH_SIGNAL_ENABLED": True, "MEMO_GRAPH_SIGNAL_ALPHA": alpha}
            for alpha in (0.10, 0.15, 0.25)
        ],
    ]
```

The tuner may write only its tuned overlay; final user-selected activation still uses Markdown config. Update docs with exact `memo config set graph.*` examples and label compatibility flags as accepted-but-inert. Add these durable real-corpus regression prompts, whose expected IDs were created during the graph design session:

```json
{
  "text": "what did the graph A/B conclude about retrieval coverage",
  "relevant": true,
  "expect_ids": ["4d53bc7e"],
  "_note": "Rare graph-evaluation decision must outrank generic graph implementation notes."
},
{
  "text": "what graph architecture and activation policy were selected for memo",
  "relevant": true,
  "expect_ids": ["10d5f527"],
  "_note": "Curated projection and Markdown activation decision."
},
{
  "text": "what namespace should memory to code graph links use",
  "relevant": true,
  "expect_ids": ["232028d1"],
  "noise_path_fragments": ["tests/", "test_"],
  "_note": "Stable codegraph URI decision; test/helper symbols must not crowd retrieval."
}
```

- [ ] **Step 4: Run tuner tests and docs checks**

Run: `uv run --no-sync pytest tests/test_dream_tune.py -q && uv run --no-sync ruff check src/memo/dream_tune.py src/memo/dream_flags.py tests/test_dream_tune.py`

Expected: PASS.

- [ ] **Step 5: Commit tuner cleanup and documentation**

```bash
git add src/memo/dream_tune.py src/memo/dream_flags.py tests/test_dream_tune.py eval/regression_labels.json docs/configuration.md src/memo/experimental_index.md
git commit -m "docs: graduate curated graph signal controls"
```

---

### Task 11: Run Focused, Static, Full, and Retrieval Verification

**Files:**
- Modify only files required by failures directly caused by Tasks 1-10.
- Record: command outputs in the implementation session notes, not generated repository artifacts.

**Interfaces:**
- Verifies: isolated graph behavior, CI order, retrieval regression, runtime isolation, and live graph diagnostics.

- [ ] **Step 1: Run all focused graph/config/search/recall tests**

Run:

```bash
uv run --no-sync pytest \
  tests/test_graph_store.py \
  tests/test_graph_projection.py \
  tests/test_graph_signal.py \
  tests/test_graph_reason.py \
  tests/test_search_graph_signal.py \
  tests/test_cli_graph.py \
  tests/test_config_md.py \
  tests/test_cli_config.py \
  tests/test_recall_hook_parity.py \
  tests/test_recall_associative.py \
  tests/test_dream_tune.py -q
```

Expected: PASS.

- [ ] **Step 2: Run repository static verification in CI order**

Run:

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
```

Expected: both commands exit 0.

- [ ] **Step 3: Run non-slow CI-parity tests**

Run: `uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing`

Expected: PASS with no failed tests.

- [ ] **Step 4: Run retrieval regression eval**

Run: `uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force`

Expected: completes and writes/prints metrics for all labeled prompts.

- [ ] **Step 5: Run macOS/runtime/config smoke tests**

Run:

```bash
uv run --no-sync pytest \
  tests/test_hook_contract.py \
  tests/test_recall_hooks.py \
  tests/test_recall_server.py \
  tests/test_runtime_isolation.py \
  tests/test_cli_migrate_vault.py -q
uv run --no-sync memo doctor --strict-runtime
uv run --no-sync memo config validate
```

Expected: tests and config validation pass; strict runtime reports the expected isolated runtime or a concrete pre-existing runtime warning that does not originate in these commits. Any directly caused failure returns to its owning task, repeats that task's test cycle, and is committed with that task's explicit file list.

---

### Task 12: Run the Real-Corpus A/B and Activate Through Markdown Config

**Files:**
- Modify: machine-local `~/.config/memo/config/graph-config.md` through CLI only.
- Modify: machine-local `~/.config/memo/config/recall-config.md` through CLI only if associative recall is not already enabled.
- Do not commit machine-local configuration.

**Interfaces:**
- Gate: no precision regression, no recall regression, no noise increase, positive nDCG or MRR movement, and p50 overhead no greater than 15 ms.
- Activates: projection, signal, reasons, semantic relations, hub suppression, associative recall, selected quality/age/budget/alpha.

- [ ] **Step 1: Rebuild the real graph with projection enabled only**

Run:

```bash
uv run --no-sync memo config set graph.projection_enabled on
uv run --no-sync memo config set graph.projection_min_quality 0.45
uv run --no-sync memo config set graph.projection_max_age_hours 36
uv run --no-sync memo config set graph.hub_max_doc_freq_ratio 0.25
uv run --no-sync memo graph rebuild --json
uv run --no-sync memo graph stats --json
```

Expected: config writes `graph-config.md`; rebuild returns an active version, positive eligible nodes/edges, and explicit rejection counts.

- [ ] **Step 2: Measure baseline A with signal disabled**

Run:

```bash
uv run --no-sync memo config set graph.signal_enabled off
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force
```

Expected: capture precision, recall, nDCG, MRR, noise, p50, and per-query rows.

- [ ] **Step 3: Measure B for bounded alpha values**

Run the same eval with `graph.signal_enabled on`, `graph.reason_enabled on`, `graph.semantic_relations on`, `graph.hub_suppression on`, `graph.signal_budget_ms 150`, and alpha `0.10`, `0.15`, then `0.25`. Retain the smallest alpha that passes the gate; if none passes, adjust only projection quality in `0.05` increments within `[0.45, 0.65]` and repeat.

Expected: at least one bounded configuration passes the selected early-activation gate without candidate injection.

- [ ] **Step 4: Persist the winning configuration through CLI**

Run the preferred pair first. The explicit alternatives below cover every bounded pair the measurement step is allowed to select:

```bash
uv run --no-sync memo config set graph.projection_enabled on
uv run --no-sync memo config set graph.projection_min_quality 0.45
uv run --no-sync memo config set graph.projection_max_age_hours 36
uv run --no-sync memo config set graph.signal_enabled on
uv run --no-sync memo config set graph.reason_enabled on
uv run --no-sync memo config set graph.semantic_relations on
uv run --no-sync memo config set graph.hub_suppression on
uv run --no-sync memo config set graph.hub_max_doc_freq_ratio 0.25
uv run --no-sync memo config set graph.min_entity_idf 0.5
uv run --no-sync memo config set graph.signal_budget_ms 150
uv run --no-sync memo config set graph.signal_alpha 0.15
uv run --no-sync memo config set recall.associative on
```

If the winning alpha is `0.10` or `0.25`, rerun only `uv run --no-sync memo config set graph.signal_alpha 0.10` or `uv run --no-sync memo config set graph.signal_alpha 0.25`. If tuning selected quality `0.50`, `0.55`, `0.60`, or `0.65`, rerun only the corresponding exact `uv run --no-sync memo config set graph.projection_min_quality 0.50`, `0.55`, `0.60`, or `0.65` command.

Expected: all writes succeed and graph flags are stored in `graph-config.md`.

- [ ] **Step 5: Validate effective sources and live behavior**

Run:

```bash
uv run --no-sync memo config validate
uv run --no-sync memo config show --effective
uv run --no-sync memo graph rebuild --json
uv run --no-sync memo graph stats --json
uv run --no-sync memo search "mlx daemon" --json
uv run --no-sync memo debug-recall "mlx daemon"
```

Expected: effective graph values cite the Markdown config source, projection is fresh/active, at least one applicable search hit contains `graph_reason`, and recall completes inside its budget. Restart the recall daemon with the repository-supported CLI only if effective output or daemon diagnostics prove it caches old config.

---

### Task 13: Integrate Cleanly and Push to `origin/master`

**Files:**
- No source edits unless resolving a direct upstream conflict.

**Interfaces:**
- Preserves: unrelated dirty files and divergent local history in `/Users/fer/repos/memo`.
- Produces: normal, non-force update of `origin/master` containing only graph design, plan, implementation, tests, and documentation.

- [ ] **Step 1: Confirm task branch cleanliness and commit list**

Run:

```bash
git status --short
git log --oneline origin/master..HEAD
git diff --check origin/master...HEAD
```

Expected: clean graph worktree, only scoped commits, no whitespace errors.

- [ ] **Step 2: Fetch and rebase on the current remote master**

Run:

```bash
git fetch origin --prune
git rebase origin/master
```

Expected: clean rebase. Resolve only direct conflicts in scoped files, rerun the tests touching each resolved file, and never force-push.

- [ ] **Step 3: Re-run fast final verification after rebase**

Run:

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120
uv run --no-sync memo config validate
```

Expected: all exit 0.

- [ ] **Step 4: Push the branch tip to remote master normally**

Run: `git push origin HEAD:master`

Expected: non-force push succeeds.

- [ ] **Step 5: Verify the remote contains the delivered tip**

Run:

```bash
git fetch origin
git merge-base --is-ancestor HEAD origin/master
git log -1 --oneline origin/master
```

Expected: `merge-base` exits 0 and remote master names the delivered graph commit chain.
