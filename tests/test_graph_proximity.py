"""Unit tests for graph_proximity.graph_boost_factory (pure, stub graph)."""

from __future__ import annotations

from dataclasses import dataclass

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
