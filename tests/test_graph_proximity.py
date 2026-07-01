"""Unit tests for graph_proximity.graph_boost_factory (pure, stub graph)."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from memo.graph_proximity import graph_boost_factory


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float | None


class _StubGraph:
    """Minimal GraphStore stand-in: query entity 'fastapi' neighbors 'pydantic'
    (edge weight 2.0). Memory 'a' mentions pydantic; memory 'b' mentions django."""

    def __init__(self, neighbors=None, mem_entities=None):
        self._neighbors = neighbors if neighbors is not None else {"fastapi": {"pydantic": 2.0}}
        self._mem_entities = (
            mem_entities
            if mem_entities is not None
            else {
                "a": [{"name": "pydantic", "type": "tech", "mention_count": 3}],
                "b": [{"name": "django", "type": "tech", "mention_count": 1}],
            }
        )

    def weighted_neighbors(self, name):
        return dict(self._neighbors.get(name.strip().lower(), {}))

    def memory_entities(self, memory_id):
        return list(self._mem_entities.get(memory_id, []))


def test_graph_proximal_hit_is_boosted_above_non_proximal():
    g = _StubGraph()
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.1)
    # b starts higher than a; a is graph-proximal so it should overtake b.
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    out = boost(hits)
    assert out[0].id == "a"
    assert out[0].score == 0.5 + 0.1 * 2.0  # weight * proximity (edge weight)
    assert out[1].id == "b"
    assert out[1].score == 0.6  # untouched


def test_weight_zero_is_noop():
    g = _StubGraph()
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.0)
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    out = boost(hits)
    assert out == hits  # identity, same order and scores


def test_no_query_entities_is_noop():
    g = _StubGraph()
    boost = graph_boost_factory(g, [], weight=0.1)
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    assert boost(hits) == hits


def test_empty_graph_is_noop():
    g = _StubGraph(neighbors={})  # no edges reachable from any query entity
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.1)
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    assert boost(hits) == hits


def test_none_score_hit_is_preserved():
    g = _StubGraph()
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.1)
    hits = [_Hit("a", None), _Hit("b", 0.6)]
    out = boost(hits)
    ids = {h.id: h.score for h in out}
    assert ids["a"] is None  # no boost applied to a scoreless hit


# --- IDF weighting (fix: raw entity-overlap promotes generic-entity junk) -----


class _IdfGraph(_StubGraph):
    """Stub that also reports corpus size + doc-frequencies so the IDF path
    engages. Query 'fastapi' neighbors both a RARE entity ('pydantic', df=1)
    and a UBIQUITOUS one ('common', df=N)."""

    def __init__(self, n_docs, doc_freqs, **kw):
        super().__init__(**kw)
        self._n = n_docs
        self._df = {k.lower(): float(v) for k, v in doc_freqs.items()}

    def total_indexed_memories(self):
        return self._n

    def entity_doc_freqs(self, names):
        return {n.strip().lower(): self._df[n.strip().lower()]
                for n in names if n.strip().lower() in self._df}


def test_idf_downweights_ubiquitous_neighbor():
    # 'a' reaches rare 'pydantic' (df=1 -> high idf); 'b' reaches ubiquitous
    # 'common' (df=N -> idf 0). Raw counting would tie/boost both; IDF must
    # boost only 'a'.
    g = _IdfGraph(
        n_docs=100,
        doc_freqs={"pydantic": 1, "common": 100},
        neighbors={"fastapi": {"pydantic": 1.0, "common": 1.0}},
        mem_entities={
            "a": [{"name": "pydantic", "type": "tech", "mention_count": 1}],
            "b": [{"name": "common", "type": "tech", "mention_count": 1}],
        },
    )
    boost = graph_boost_factory(g, ["FastAPI"], weight=0.1)
    out = boost([_Hit("a", 0.5), _Hit("b", 0.5)])
    scores = {h.id: h.score for h in out}
    assert scores["a"] > 0.5      # rare neighbor -> boosted
    assert scores["b"] == 0.5      # ubiquitous neighbor -> idf 0 -> untouched
    assert out[0].id == "a"


def test_min_idf_gate_suppresses_ubiquitous_only_query():
    # Query entity itself is ubiquitous (df=N -> idf 0); with a positive
    # min_idf the whole boost must be suppressed (identity).
    g = _IdfGraph(
        n_docs=100,
        doc_freqs={"fastapi": 100, "pydantic": 1},
        neighbors={"fastapi": {"pydantic": 1.0}},
    )
    hits = [_Hit("a", 0.5), _Hit("b", 0.6)]
    gated = graph_boost_factory(g, ["FastAPI"], weight=0.1, min_idf=1.0)
    assert gated(hits) == hits          # gate fails -> identity
    ungated = graph_boost_factory(g, ["FastAPI"], weight=0.1, min_idf=0.0)
    assert ungated(hits)[0].id == "a"   # no gate -> rare neighbor still boosts


# --- wiring into _recall_logic (flag-gated, default OFF) ---------------------


def _prox_mem(tmp_path, monkeypatch):
    """Real Memory with a 4-dim stub embedder. Query ('qzz') -> [1,0,0,0].
    Doc A ('DOCA') -> cosine 0.6; Doc B ('DOCB') -> cosine 0.8 (B beats A on
    vec alone). Graph: A's entities {MLX, Pydantic} co-occur (edge), B's entity
    {Django} is unconnected; an 'MLX' query is 1 hop from A only."""
    from memo.config import Config
    from memo.memory import Memory

    def _vec(text: str) -> list[float]:
        t = text or ""
        if "qzz" in t:  # query marker (lowercase so it isn't extracted as an entity)
            return [1.0, 0.0, 0.0, 0.0]
        if "DOCA" in t:
            return [0.6, 0.0, 0.8, 0.0]  # cosine 0.6 with the query
        if "DOCB" in t:
            return [0.8, 0.0, 0.0, 0.6]  # cosine 0.8 (beats A on vec alone)
        return [0.0, 1.0, 0.0, 0.0]

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed", lambda self, inputs: [_vec(t) for t in inputs]
    )
    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", lambda self, q: _vec(q))
    cfg = Config(
        data_dir=tmp_path / "data",
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / "state",
        embedder_dims=4,
    )
    mem = Memory(cfg)
    rec_a = mem.save(content="DOCA note about rate limiting and pagination details here", title="DOCA", type_="note")
    rec_b = mem.save(content="DOCB note about caching layers and retries described here", title="DOCB", type_="note")
    # A co-occurs MLX + Pydantic -> rebuild_edges yields an mlx<->pydantic edge, so
    # an 'MLX' query is 1 hop from A's 'pydantic' entity.
    mem.graph.record_extraction(
        memory_id=rec_a.id,
        memory_date="2026-01-01",
        entities=[{"name": "MLX", "type": "technology"}, {"name": "Pydantic", "type": "technology"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    mem.graph.record_extraction(
        memory_id=rec_b.id,
        memory_date="2026-01-01",
        entities=[{"name": "Django", "type": "technology"}],
        extracted_at="2026-01-01T00:00:00Z",
    )
    mem.graph.rebuild_edges()
    return mem, cfg, rec_a.id, rec_b.id


def _common_env(monkeypatch):
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.setenv("MEMO_RECALL_FORCE_MODE", "1")  # stub embedder reports cold
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "0")
    monkeypatch.setenv("MEMO_EMBEDDER_DIMS", "4")


def test_recall_flag_off_leaves_ranking_unchanged(tmp_path, monkeypatch):
    from memo.recall_logic import _recall_logic

    _common_env(monkeypatch)
    monkeypatch.delenv("MEMO_RECALL_GRAPH_PROXIMITY", raising=False)
    mem, cfg, a_id, b_id = _prox_mem(tmp_path, monkeypatch)
    context, _cb = _recall_logic("how do I configure MLX here for the qzz thing", cwd=None, mem=mem, cfg=cfg)
    # Default OFF: vec ranking stands, so B (cosine 0.8) renders before A (0.6).
    assert context.index(b_id[:8]) < context.index(a_id[:8])
    mem.close()


@pytest.mark.no_stub_embedder
def test_recall_flag_on_reorders_graph_proximal_up(tmp_path, monkeypatch):
    from memo.recall_logic import _recall_logic

    _common_env(monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_GRAPH_PROXIMITY", "1")
    monkeypatch.setenv("MEMO_RECALL_GRAPH_PROXIMITY_WEIGHT", "0.5")
    mem, cfg, a_id, b_id = _prox_mem(tmp_path, monkeypatch)
    context, _cb = _recall_logic("how do I configure MLX here for the qzz thing", cwd=None, mem=mem, cfg=cfg)
    # Flag ON: A is 1 hop from the 'MLX' query entity -> boosted above B.
    assert context.index(a_id[:8]) < context.index(b_id[:8])
    mem.close()


# --- extract_query_entities (fix #2: fire on natural lowercase prompts) -------


class _VocabGraph:
    def __init__(self, names: set[str]) -> None:
        self._names = names

    def entity_names(self) -> set[str]:
        return self._names


def test_extract_query_entities_matches_graph_vocabulary_lowercase() -> None:
    from memo.graph_proximity import extract_query_entities

    g = _VocabGraph({"recall hook", "synapse", "memflow", "budget"})
    # Natural lowercase prompt: the proper-noun regex extracts nothing useful.
    q = [e.lower() for e in extract_query_entities("how does the recall hook budget work", g)]
    assert "recall hook" in q  # bigram matched from graph vocabulary
    assert "budget" in q       # unigram matched from graph vocabulary


def test_extract_query_entities_keeps_regex_proper_nouns() -> None:
    from memo.graph_proximity import extract_query_entities

    g = _VocabGraph(set())  # empty vocab -> only the regex contributes
    q = {e.lower() for e in extract_query_entities("How does Synapse connect to MLX?", g)}
    assert "mlx" in q or "synapse" in q


def test_extract_query_entities_no_stopword_noise() -> None:
    from memo.graph_proximity import extract_query_entities

    g = _VocabGraph({"synapse"})  # 'synapse' not in the prompt
    q = extract_query_entities("the quick brown fox jumps over", g)
    assert q == []  # no real entity present -> nothing leaks in as noise
