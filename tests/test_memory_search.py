from __future__ import annotations

from memo.config import Config
from memo.memory import Memory


def test_search_returns_matching(mem_with_stub: Memory):
    mem_with_stub.save(content="alpha", title="A")
    mem_with_stub.save(content="beta", title="B")
    hits = mem_with_stub.search("alpha", limit=2)
    assert any(h.title == "A" for h in hits)


def test_contextual_retrieval_prepends_context_only_when_enabled(mem_with_stub: Memory, monkeypatch):
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


def test_bm25_handles_empty_and_garbage_queries(mem_with_stub: Memory):
    mem_with_stub.save(content="x", title="X")
    assert mem_with_stub.search("", mode="bm25") == []
    out = mem_with_stub.search('weird " query', mode="bm25")
    assert isinstance(out, list)


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
    mem = Memory(cfg)
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
            id=id_, path=f"{id_}.md", title=id_, type="note", tags=[],
            created=updated.isoformat(), updated=updated.isoformat(),
            body="b", extra={}, score=0.70,
        )

    old = _rec("old", now - timedelta(days=400))
    fresh = _rec("fresh", now - timedelta(days=1))

    out = _apply_decay([old, fresh], halflife_days=180.0, alpha=0.15)
    assert [r.id for r in out] == ["fresh", "old"]
    assert out[0].score is not None and out[1].score is not None
    assert out[0].score > out[1].score
