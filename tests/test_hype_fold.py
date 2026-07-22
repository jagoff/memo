"""Tests for the HyPE read-path max-fold (`memo.hype_fold`) + its wiring
into `Memory.search` (vec mode) behind `MEMO_HYPE_ENABLED`.

Pure-fold tests use fakes (no sqlite, no embedder). Wiring tests build a
real `Memory` on an isolated Config with a deterministic 4-dim stub
embedder (`MEMO_EMBEDDER_DIMS=4` pinned, per test-isolation rules).
"""

from __future__ import annotations

import pytest

from memo.config import Config
from memo.hype_fold import hype_fold

# ---------------------------------------------------------------------------
# Pure fold — fakes
# ---------------------------------------------------------------------------


class FakeHypeStore:
    """Duck-typed stand-in for `HypeStore`: `knn` returns fixed rows."""

    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = list(rows or [])
        self.knn_calls: list[tuple[list[float], int]] = []

    def knn(self, embedding: list[float], k: int) -> list[dict]:
        self.knn_calls.append((list(embedding), k))
        return [dict(r) for r in self.rows]


def _meta(mid: str) -> dict:
    return {
        "id": mid,
        "path": f"memories/{mid}.md",
        "title": f"title-{mid}",
        "type": "note",
        "tags": [],
        "created": "2026-07-13T00:00:00+00:00",
        "updated": "2026-07-13T00:00:00+00:00",
        "extra": {},
    }


def _fetch_from(metas: dict[str, dict]):
    calls: list[str] = []

    def fetch(mid: str) -> dict | None:
        calls.append(mid)
        row = metas.get(mid)
        return dict(row) if row is not None else None

    fetch.calls = calls  # type: ignore[attr-defined]
    return fetch


EMB = [1.0, 0.0, 0.0, 0.0]


def test_max_fold_raises_existing_hit_score_and_marks_hype() -> None:
    doc_hits = [{**_meta("a" * 32), "score": 0.5}]
    store = FakeHypeStore([{"memory_id": "a" * 32, "question": "q?", "score": 0.9}])
    out = hype_fold(doc_hits, EMB, store, _fetch_from({}), pool=30, limit=5)
    assert len(out) == 1
    assert out[0]["id"] == "a" * 32
    assert out[0]["score"] == pytest.approx(0.9)
    assert out[0]["hype"] is True
    # Purity: the caller's input dict is NOT mutated.
    assert doc_hits[0]["score"] == 0.5
    assert "hype" not in doc_hits[0]


def test_question_score_below_doc_score_leaves_hit_unchanged() -> None:
    doc_hits = [{**_meta("a" * 32), "score": 0.8}]
    store = FakeHypeStore([{"memory_id": "a" * 32, "question": "q?", "score": 0.3}])
    out = hype_fold(doc_hits, EMB, store, _fetch_from({}), pool=30, limit=5)
    assert out[0]["score"] == pytest.approx(0.8)
    assert "hype" not in out[0]


def test_new_candidate_appended_with_fetched_meta() -> None:
    a, b = "a" * 32, "b" * 32
    doc_hits = [{**_meta(a), "score": 0.6}]
    store = FakeHypeStore([{"memory_id": b, "question": "what is b?", "score": 0.9}])
    fetch = _fetch_from({b: _meta(b)})
    out = hype_fold(doc_hits, EMB, store, fetch, pool=30, limit=5)
    assert [r["id"] for r in out] == [b, a]  # sorted desc by score
    new = out[0]
    assert new["score"] == pytest.approx(0.9)
    assert new["hype"] is True
    assert new["title"] == f"title-{b}"  # meta fields carried over
    assert new["path"] == f"memories/{b}.md"


def test_fetch_meta_none_skips_candidate() -> None:
    a, gone = "a" * 32, "d" * 32
    doc_hits = [{**_meta(a), "score": 0.6}]
    store = FakeHypeStore([{"memory_id": gone, "question": "q?", "score": 0.99}])
    fetch = _fetch_from({})  # deleted / unknown → None
    out = hype_fold(doc_hits, EMB, store, fetch, pool=30, limit=5)
    assert [r["id"] for r in out] == [a]
    assert fetch.calls == [gone]  # type: ignore[attr-defined]


def test_limit_respected_and_sorted_desc() -> None:
    a, b, c, d, e = ("a" * 32, "b" * 32, "c" * 32, "d" * 32, "e" * 32)
    doc_hits = [
        {**_meta(a), "score": 0.9},
        {**_meta(b), "score": 0.5},
        {**_meta(c), "score": 0.4},
    ]
    store = FakeHypeStore(
        [
            {"memory_id": d, "question": "q1?", "score": 0.8},
            {"memory_id": e, "question": "q2?", "score": 0.7},
        ]
    )
    fetch = _fetch_from({d: _meta(d), e: _meta(e)})
    out = hype_fold(doc_hits, EMB, store, fetch, pool=30, limit=3)
    assert [r["id"] for r in out] == [a, d, e]
    assert [r["score"] for r in out] == sorted((r["score"] for r in out), reverse=True)


def test_empty_knn_returns_identical_output() -> None:
    a, b = "a" * 32, "b" * 32
    doc_hits = [{**_meta(a), "score": 0.9}, {**_meta(b), "score": 0.5}]
    store = FakeHypeStore([])
    fetch = _fetch_from({})
    out = hype_fold(doc_hits, EMB, store, fetch, pool=30, limit=5)
    assert out == doc_hits
    assert fetch.calls == []  # type: ignore[attr-defined]


def test_pool_passed_to_knn_as_k() -> None:
    store = FakeHypeStore([])
    hype_fold([], EMB, store, _fetch_from({}), pool=17, limit=5)
    assert store.knn_calls == [(EMB, 17)]


# ---------------------------------------------------------------------------
# Wiring — real Memory.search (vec mode), stub embedder dims=4
# ---------------------------------------------------------------------------

# Deterministic keyword→vector stub. Tokens are chosen so the asymmetric
# query-instruction prefix (which contains common words like "query") can
# never collide with them.
_QUERY_VEC = [0.8, 0.6, 0.0, 0.0]  # unit norm
_ALPHA_VEC = [1.0, 0.0, 0.0, 0.0]  # cos vs query = 0.8
_BETA_VEC = [0.0, 0.0, 1.0, 0.0]  # cos vs query = 0.0


def _stub_embed(self, inputs):  # mirrors MLXEmbedder.embed signature
    out = []
    for s in inputs:
        t = s.lower()
        if "zzquery" in t:
            out.append(list(_QUERY_VEC))
        elif "zzalpha" in t:
            out.append(list(_ALPHA_VEC))
        elif "zzbeta" in t:
            out.append(list(_BETA_VEC))
        else:
            out.append([0.0, 0.0, 0.0, 1.0])
    return out


@pytest.fixture
def hype_mem(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch):
    """Real `Memory` with a deterministic 4-dim embedder, HyPE-ready."""
    from memo.memory import Memory

    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")
    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    mem = Memory(cfg)
    yield mem
    mem.close()


def _populate_hype(cfg: Config, memory_id: str, embedding: list[float]) -> None:
    from memo.store.hype_store import HypeStore

    hs = HypeStore(cfg.db_path, 4)
    try:
        hs.replace_for_memory(
            memory_id, "bodyhash", "test-model", [("what is zzbeta about?", embedding)]
        )
    finally:
        hs.close()


def test_flag_off_results_identical_with_populated_hype_store(
    hype_mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MEMO_HYPE_ENABLED", raising=False)
    hype_mem.save(content="zzalpha note body", title="Alpha zzalpha")
    rec_b = hype_mem.save(content="zzbeta note body", title="Beta zzbeta")

    before = hype_mem.search("zzquery topic", mode="vec", limit=5)
    # Populate a question vector that WOULD promote B if the fold ran.
    _populate_hype(hype_mem.cfg, rec_b.id, list(_QUERY_VEC))
    after = hype_mem.search("zzquery topic", mode="vec", limit=5)

    assert [r.id for r in after] == [r.id for r in before]
    assert [r.score for r in after] == [r.score for r in before]


@pytest.mark.float32_precision  # asserts score == 0.8 within abs=1e-3; int8 cosine is ~1/127-quantized
def test_flag_on_question_space_candidate_surfaces(
    hype_mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec_a = hype_mem.save(content="zzalpha note body", title="Alpha zzalpha")
    rec_b = hype_mem.save(content="zzbeta note body", title="Beta zzbeta")
    _populate_hype(hype_mem.cfg, rec_b.id, list(_QUERY_VEC))
    monkeypatch.setenv("MEMO_HYPE_ENABLED", "1")

    # limit=1: the doc-space search returns only A (cos 0.8); B's doc vector
    # is orthogonal to the query. The question-space fold must surface B
    # (question cos ~1.0) as a NEW candidate that displaces A.
    top = hype_mem.search("zzquery topic", mode="vec", limit=1)
    assert [r.id for r in top] == [rec_b.id]
    assert top[0].score == pytest.approx(1.0, abs=1e-3)

    # limit=5: both present, B folded first, A keeps its doc score.
    both = hype_mem.search("zzquery topic", mode="vec", limit=5)
    assert [r.id for r in both][:2] == [rec_b.id, rec_a.id]
    assert both[1].score == pytest.approx(0.8, abs=1e-3)


def test_flag_on_fold_appended_candidate_respects_type_filter(
    hype_mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fold-appended candidate (question-space only, not a doc hit) must
    still honor the `type_` filter passed into `search()`. Without a filter
    closure around `fetch_meta`, `_hype_fold_candidates` bypasses the SQL-level
    type filtering entirely for these appended rows."""
    rec_a = hype_mem.save(content="zzalpha note body", title="Alpha zzalpha", type_="decision")
    rec_b = hype_mem.save(content="zzbeta note body", title="Beta zzbeta", type_="note")
    # B is only reachable via the question-space fold (doc vector orthogonal).
    _populate_hype(hype_mem.cfg, rec_b.id, list(_QUERY_VEC))
    monkeypatch.setenv("MEMO_HYPE_ENABLED", "1")

    # B is type="note"; filtering to type_="decision" must exclude it even
    # though the question-space match would otherwise surface it first.
    out = hype_mem.search("zzquery topic", mode="vec", type_="decision", limit=5)
    assert rec_b.id not in [r.id for r in out]
    assert [r.id for r in out] == [rec_a.id]


def test_fold_variant_mismatch_warns_in_trace_but_still_folds(
    hype_mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The HyPE store's stored questions were embedded 'raw' (document-side),
    but MEMO_HYPE_EMBED_RAW is OFF (active variant='query') — a scale
    mismatch. Search must still fold (never hard-fail) and record a warning
    trace note so the mismatch is diagnosable via `memo dream hype --reembed`."""
    from memo.store.hype_store import HypeStore

    monkeypatch.delenv("MEMO_HYPE_EMBED_RAW", raising=False)  # active variant = "query"
    rec_a = hype_mem.save(content="zzalpha note body", title="Alpha zzalpha")
    rec_b = hype_mem.save(content="zzbeta note body", title="Beta zzbeta")

    hs = HypeStore(hype_mem.cfg.db_path, 4)
    try:
        hs.replace_for_memory(
            rec_b.id,
            "bodyhash",
            "test-model",
            [("what is zzbeta about?", list(_QUERY_VEC))],
            variant="raw",
        )
    finally:
        hs.close()

    monkeypatch.setenv("MEMO_HYPE_ENABLED", "1")

    trace: list[dict] = []
    out = hype_mem.search("zzquery topic", mode="vec", limit=5, _trace=trace)

    # Fold still ran and still surfaced results — a mismatch never hard-fails.
    assert rec_a.id in [r.id for r in out]
    assert rec_b.id in [r.id for r in out]

    hype_stage = next(entry for entry in trace if entry["stage"] == "hype_fold")
    assert "warning" in hype_stage
    assert "raw" in hype_stage["warning"] and "query" in hype_stage["warning"]


def test_fold_mixed_variants_warns_even_when_active_variant_is_dominant(
    hype_mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Any stale minority variant makes cosine scores incomparable.

    A dominant-variant-only check misses a partially re-embedded index, so the
    trace must warn whenever *any* stored rows differ from the active variant.
    """
    from memo.store.hype_store import HypeStore

    monkeypatch.delenv("MEMO_HYPE_EMBED_RAW", raising=False)  # active = "query"
    rec_query_1 = hype_mem.save(content="zzalpha first", title="First zzalpha")
    rec_query_2 = hype_mem.save(content="zzalpha second", title="Second zzalpha")
    rec_raw = hype_mem.save(content="zzbeta stale", title="Stale zzbeta")
    _populate_hype(hype_mem.cfg, rec_query_1.id, list(_QUERY_VEC))
    _populate_hype(hype_mem.cfg, rec_query_2.id, list(_QUERY_VEC))

    hs = HypeStore(hype_mem.cfg.db_path, 4)
    try:
        hs.replace_for_memory(
            rec_raw.id,
            "bodyhash",
            "test-model",
            [("what is stale zzbeta about?", list(_QUERY_VEC))],
            variant="raw",
        )
    finally:
        hs.close()

    monkeypatch.setenv("MEMO_HYPE_ENABLED", "1")
    trace: list[dict] = []
    hype_mem.search("zzquery topic", mode="vec", limit=5, _trace=trace)

    hype_stage = next(entry for entry in trace if entry["stage"] == "hype_fold")
    assert "warning" in hype_stage
    assert "raw" in hype_stage["warning"] and "query" in hype_stage["warning"]


def test_fold_variant_match_has_no_warning_in_trace(
    hype_mem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the store's variant matches the active flag, no warning is added."""
    rec_b = hype_mem.save(content="zzbeta note body", title="Beta zzbeta")
    _populate_hype(hype_mem.cfg, rec_b.id, list(_QUERY_VEC))  # default variant="query"
    monkeypatch.delenv("MEMO_HYPE_EMBED_RAW", raising=False)  # active variant = "query"
    monkeypatch.setenv("MEMO_HYPE_ENABLED", "1")

    trace: list[dict] = []
    hype_mem.search("zzquery topic", mode="vec", limit=5, _trace=trace)

    hype_stage = next(entry for entry in trace if entry["stage"] == "hype_fold")
    assert "warning" not in hype_stage


def test_close_closes_hype_store(hype_mem, monkeypatch: pytest.MonkeyPatch) -> None:
    """Once a folded vec search materializes `_hype_store`, `Memory.close()`
    must close its underlying sqlite connection too — otherwise it leaks a
    file descriptor for the lifetime of the process. `HypeStore._conn` lazily
    reopens on next access (see `_ConnectionMixin`), so we can't assert a
    raise on subsequent use; instead assert the specific connection object
    held at fold-time was actually closed (`sqlite3.Connection.close()` sets
    a queryable closed state via a follow-up `execute` raising ProgrammingError
    directly on THAT connection object, bypassing the lazy-reopen property)."""
    import sqlite3

    monkeypatch.setenv("MEMO_HYPE_ENABLED", "1")
    hype_mem.save(content="zzalpha note body", title="Alpha zzalpha")
    hype_mem.search("zzquery topic", mode="vec", limit=5)  # materializes _hype_store
    assert hype_mem._hype_store is not None

    store = hype_mem._hype_store
    conn = store._conn  # the live thread-local sqlite3.Connection
    hype_mem.close()

    with pytest.raises(sqlite3.ProgrammingError):
        conn.execute("SELECT 1")
