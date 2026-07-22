from __future__ import annotations

from contextlib import closing

from memo.config import Config
from memo.memory import Memory
from memo.tiers import VerificationState


def test_read_paths_preserve_verification_metadata(mock_memory) -> None:
    rec = mock_memory.save(content="verified probe alpha", title="Verified Probe")
    mock_memory.store._conn.execute(
        "UPDATE meta SET verification_state = ?, verified_at = ? WHERE id = ?",
        (VerificationState.VERIFIED.value, 123456, rec.id),
    )
    mock_memory.store._conn.commit()

    got = mock_memory.get(rec.id)
    listed = next(r for r in mock_memory.list(limit=10) if r.id == rec.id)
    searched = mock_memory.search("verified probe alpha", mode="bm25", limit=1)[0]

    assert got is not None
    assert got.verification_state == VerificationState.VERIFIED
    assert got.verified_at == 123456
    assert listed.verification_state == VerificationState.VERIFIED
    assert listed.verified_at == 123456
    assert searched.verification_state == VerificationState.VERIFIED
    assert searched.verified_at == 123456


def test_search_returns_matching(mem_with_stub: Memory):
    mem_with_stub.save(content="alpha", title="A")
    mem_with_stub.save(content="beta", title="B")
    hits = mem_with_stub.search("alpha", limit=2)
    assert any(h.title == "A" for h in hits)


def test_contextual_retrieval_prepends_context_only_when_enabled(
    mem_with_stub: Memory, monkeypatch
):
    seen_inputs: list[str] = []

    def _spy_embed(inputs):
        seen_inputs.extend(inputs)
        return [[1.0, 0.0, 0.0, 0.0] for _s in inputs]

    monkeypatch.setenv("MEMO_CONTEXTUAL_RETRIEVAL", "1")
    monkeypatch.setattr(mem_with_stub.embedder, "embed", _spy_embed)
    monkeypatch.setattr(
        mem_with_stub,
        "_generate_contextual_summary",
        lambda _prompt: "Esta memoria trata sobre el runbook de ingestion.",
    )
    body = "Runbook de ingestion.\n\n" + ("detalle operacional " * 40)

    rec = mem_with_stub.save(content=body, title="Ingestion Runbook")

    assert rec.body == body
    assert seen_inputs
    assert seen_inputs[0].startswith("[contexto: Esta memoria trata")
    assert "Ingestion Runbook" in seen_inputs[0]
    assert "Runbook de ingestion" in seen_inputs[0]


def test_contextual_retrieval_cache_reuses_generated_summary(mem_with_stub: Memory, monkeypatch):
    calls: list[str] = []

    def _generate(prompt: str) -> str:
        calls.append(prompt)
        return "Contexto cacheado para búsqueda semántica."

    monkeypatch.setenv("MEMO_CONTEXTUAL_RETRIEVAL", "1")
    monkeypatch.setattr(mem_with_stub, "_generate_contextual_summary", _generate)
    body = "Nota larga.\n\n" + ("contenido importante " * 40)

    first = mem_with_stub._compose_for_embed("Nota", body)
    second = mem_with_stub._compose_for_embed("Nota", body)

    assert first == second
    assert first.startswith("[contexto: Contexto cacheado")
    assert len(calls) == 1


def test_hybrid_search_fuses_vec_and_bm25(mem_with_stub: Memory):
    mem_with_stub.save(
        content="contenido sobre python testing y mocks",
        title="Python testing notes",
        tags=["python", "testing"],
    )
    mem_with_stub.save(
        content="receta de pizza casera con harina y queso",
        title="Pizza casera",
        tags=["receta", "cocina"],
    )
    bm = mem_with_stub.search("python testing", mode="bm25")
    assert bm and bm[0].title == "Python testing notes"
    v = mem_with_stub.search("python testing", mode="vec")
    assert v
    h = mem_with_stub.search("python testing", mode="hybrid", limit=2)
    assert any(r.title == "Python testing notes" for r in h)


def test_hybrid_search_uses_temporal_fact_edges(mem_with_stub: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")
    rec = mem_with_stub.save(
        content="short operational note",
        title="Capture graph design",
        type_="note",
        extra={
            "fact_edges": [
                {
                    "subject": "memo capture",
                    "predicate": "records",
                    "object": "graph facts",
                }
            ]
        },
    )

    out = mem_with_stub.search("graph facts", mode="hybrid", limit=3)

    hit = next(h for h in out if h.id == rec.id)
    assert hit.extra["fact_edge_matched"] is True
    assert hit.extra["related_fact_edges"][0]["subject"] == "memo capture"
    assert hit.extra["related_fact_edges"][0]["object"] == "graph facts"


def test_search_trace_reports_fact_candidate_count(mem_with_stub: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")
    mem_with_stub.save(
        content="note body",
        title="Temporal graph",
        extra={"fact_edges": [{"subject": "memo", "predicate": "uses", "object": "facts"}]},
    )

    envelope = mem_with_stub.search_with_trace("uses facts", mode="hybrid", limit=2)
    candidate_stage = next(s for s in envelope["trace"] if s["stage"] == "candidate_generation")

    assert candidate_stage["fact_count"] >= 1


def test_bm25_handles_empty_and_garbage_queries(mem_with_stub: Memory):
    mem_with_stub.save(content="x", title="X")
    assert mem_with_stub.search("", mode="bm25") == []
    out = mem_with_stub.search('weird " query', mode="bm25")
    assert isinstance(out, list)


def test_store_vec_search_excludes_invalidated(mem_with_stub: Memory):
    """Vec seam (queries.py inline SQL): default recall drops rows whose
    validity interval is closed as of now, and surfaced rows carry the real
    valid_at/invalid_at (not None)."""
    a = mem_with_stub.save(content="prod db is postgres", title="A", type_="fact")
    b = mem_with_stub.save(content="prod db is mysql", title="B", type_="fact")

    emb = [1.0, 0.0, 0.0, 0.0]
    before = {r["id"]: r for r in mem_with_stub.store.search(emb, limit=10)}
    assert a.id in before and b.id in before
    # columns now flow through the vec SELECT
    assert before[a.id]["valid_at"] == a.valid_at
    assert before[a.id]["invalid_at"] is None

    mem_with_stub.store.update_validity(
        id_=a.id, valid_at=a.valid_at, invalid_at="2000-01-01T00:00:00"
    )
    after = {r["id"] for r in mem_with_stub.store.search(emb, limit=10)}
    assert a.id not in after
    assert b.id in after


def test_store_bm25_search_excludes_invalidated(mem_with_stub: Memory):
    """BM25 seam (bm25_queries.py): default recall drops the closed interval;
    valid_at flows through the bm25 SELECT."""
    a = mem_with_stub.save(content="prod db is postgres", title="A", type_="fact")
    b = mem_with_stub.save(content="prod db is mysql", title="B", type_="fact")

    before = {r["id"]: r for r in mem_with_stub.store.search_bm25("prod db", limit=10)}
    assert a.id in before and b.id in before
    assert before[a.id]["valid_at"] == a.valid_at

    mem_with_stub.store.update_validity(
        id_=a.id, valid_at=a.valid_at, invalid_at="2000-01-01T00:00:00"
    )
    after = {r["id"] for r in mem_with_stub.store.search_bm25("prod db", limit=10)}
    assert a.id not in after
    assert b.id in after


def test_default_recall_excludes_invalidated(mem_with_stub: Memory):
    """End-to-end: default `mem.search(...)` (hybrid) and the bm25 mode both
    hide a record whose interval is closed, keeping the valid successor."""
    a = mem_with_stub.save(content="prod db is postgres", title="A", type_="fact")
    b = mem_with_stub.save(content="prod db is mysql", title="B", type_="fact")

    mem_with_stub.store.update_validity(
        id_=a.id, valid_at=a.valid_at, invalid_at="2000-01-01T00:00:00"
    )

    hybrid_ids = {r.id for r in mem_with_stub.search("prod db", limit=10)}
    assert a.id not in hybrid_ids
    assert b.id in hybrid_ids

    bm25_ids = {r.id for r in mem_with_stub.search("prod db", mode="bm25", limit=10)}
    assert a.id not in bm25_ids
    assert b.id in bm25_ids


def test_search_uses_query_prefix(tmp_cfg: Config, monkeypatch):
    seen_inputs: list[str] = []

    def _spy(self, inputs):
        seen_inputs.extend(inputs)
        return [[1.0, 0.0, 0.0, 0.0] for _ in inputs]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _spy)
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
    )
    with closing(Memory(cfg)) as mem:
        mem.save(content="cuerpo del doc", title="X")
        seen_inputs.clear()
        mem.search("buscame algo", limit=3)
    assert seen_inputs
    assert seen_inputs[0].startswith("Instruct:")
    assert "buscame algo" in seen_inputs[0]


def test_apply_decay_lets_fresher_memory_win_a_tie():
    from datetime import UTC, datetime, timedelta

    from memo.memory import MemoryRecord, _apply_decay

    now = datetime.now(tz=UTC)

    def _rec(id_: str, updated: datetime) -> MemoryRecord:
        return MemoryRecord(
            id=id_,
            path=f"{id_}.md",
            title=id_,
            type="note",
            tags=[],
            created=updated.isoformat(),
            updated=updated.isoformat(),
            body="b",
            extra={},
            score=0.70,
        )

    old = _rec("old", now - timedelta(days=400))
    fresh = _rec("fresh", now - timedelta(days=1))

    out = _apply_decay([old, fresh], halflife_days=180.0, alpha=0.15)
    assert [r.id for r in out] == ["fresh", "old"]
    assert out[0].score is not None and out[1].score is not None
    assert out[0].score > out[1].score


def test_search_with_trace_reports_retrieval_stages(mem_with_stub: Memory) -> None:
    mem_with_stub.save(content="alpha body", title="Alpha")
    mem_with_stub.save(content="beta body", title="Beta")

    envelope = mem_with_stub.search_with_trace("alpha", limit=2, mode="hybrid")

    assert envelope["hits"]
    stages = [item["stage"] for item in envelope["trace"]]
    assert stages[0] == "candidate_generation"
    assert "materialize" in stages
    assert stages[-1] == "final"
    assert envelope["trace"][-1]["output_count"] == len(envelope["hits"])


def test_hybrid_search_skips_rerank_when_rrf_has_confident_winner(
    mem_with_stub: Memory,
    monkeypatch,
) -> None:
    mem_with_stub.cfg = mem_with_stub.cfg.model_copy(update={"reranker_enabled": True})
    monkeypatch.setenv("MEMO_RERANK_SKIP_CONFIDENT_RRF", "1")
    monkeypatch.setenv("MEMO_RRF_K", "1")
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")

    def _row(rid: str, title: str) -> dict:
        return {
            "id": rid,
            "path": f"{rid}.md",
            "title": title,
            "type": "note",
            "tags": [],
            "created": "",
            "updated": "",
            "score": 1.0,
        }

    monkeypatch.setattr(mem_with_stub.embedder, "embed_query", lambda _q: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        mem_with_stub.store,
        "search",
        lambda *_a, **_k: [_row("clear", "Clear winner"), _row("vec-second", "Vec second")],
    )

    def _bm25(*_args, **kwargs):
        if kwargs.get("field_boost") == "exact":
            return []
        return [_row("clear", "Clear winner"), _row("bm-second", "BM second")]

    monkeypatch.setattr(mem_with_stub.store, "search_bm25", _bm25)
    monkeypatch.setattr(
        mem_with_stub,
        "_rerank",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("rerank should be skipped")),
    )

    envelope = mem_with_stub.search_with_trace(
        "clear winner",
        limit=3,
        mode="hybrid",
        load_bodies=False,
    )

    assert next(hit.id for hit in envelope["hits"]) == "clear"
    assert "rerank_skip" in [item["stage"] for item in envelope["trace"]]


def test_hybrid_search_still_reranks_ambiguous_rrf_results(
    mem_with_stub: Memory,
    monkeypatch,
) -> None:
    mem_with_stub.cfg = mem_with_stub.cfg.model_copy(update={"reranker_enabled": True})
    monkeypatch.setenv("MEMO_RERANK_SKIP_CONFIDENT_RRF", "1")
    monkeypatch.setenv("MEMO_RRF_K", "60")
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")
    calls = {"rerank": 0}

    def _row(rid: str, title: str) -> dict:
        return {
            "id": rid,
            "path": f"{rid}.md",
            "title": title,
            "type": "note",
            "tags": [],
            "created": "",
            "updated": "",
            "score": 1.0,
        }

    def _rerank(_query, hits, *, top_n):
        calls["rerank"] += 1
        return hits[:top_n]

    monkeypatch.setattr(mem_with_stub.embedder, "embed_query", lambda _q: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        mem_with_stub.store,
        "search",
        lambda *_a, **_k: [_row("top", "Top"), _row("vec-second", "Vec second")],
    )
    monkeypatch.setattr(
        mem_with_stub.store,
        "search_bm25",
        lambda *_a, **_k: [_row("top", "Top"), _row("bm-second", "BM second")],
    )
    monkeypatch.setattr(mem_with_stub, "_rerank", _rerank)

    envelope = mem_with_stub.search_with_trace(
        "ambiguous",
        limit=3,
        mode="hybrid",
        load_bodies=False,
    )

    assert calls["rerank"] == 1
    assert "rerank" in [item["stage"] for item in envelope["trace"]]


def test_hybrid_search_includes_exact_bm25_candidates_before_rerank(
    mem_with_stub: Memory,
    monkeypatch,
) -> None:
    mem_with_stub.cfg = mem_with_stub.cfg.model_copy(
        update={"reranker_enabled": True, "rerank_input_k": 3},
    )
    monkeypatch.setenv("MEMO_HEALTH_SCORES_DISABLED", "1")
    seen_field_boosts: list[str | None] = []

    def _row(rid: str, title: str) -> dict:
        return {
            "id": rid,
            "path": f"{rid}.md",
            "title": title,
            "type": "note",
            "tags": [],
            "created": "",
            "updated": "",
            "score": 1.0,
        }

    def _bm25(*_args, **kwargs):
        seen_field_boosts.append(kwargs.get("field_boost"))
        if kwargs.get("field_boost") == "exact":
            return [_row("exact", "Exact metadata hit")]
        return [_row("keyword", "Keyword hit")]

    def _rerank(_query, hits, *, top_n):
        return hits[:top_n]

    monkeypatch.setattr(mem_with_stub.embedder, "embed_query", lambda _q: [1.0, 0.0, 0.0, 0.0])
    monkeypatch.setattr(
        mem_with_stub.store,
        "search",
        lambda *_a, **_k: [_row("semantic", "Semantic hit")],
    )
    monkeypatch.setattr(mem_with_stub.store, "search_bm25", _bm25)
    monkeypatch.setattr(mem_with_stub, "_rerank", _rerank)

    envelope = mem_with_stub.search_with_trace(
        "exact metadata",
        limit=3,
        mode="hybrid",
        load_bodies=False,
    )

    assert "exact" in [hit.id for hit in envelope["hits"]]
    assert None in seen_field_boosts
    assert "exact" in seen_field_boosts
