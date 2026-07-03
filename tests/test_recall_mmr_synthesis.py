"""Tests for the M3 recall-ranking knobs: MMR diversity + synthesis boost.

Both knobs default 0.0 = OFF and must leave ranking byte-identical
(same objects, same order, same scores) when off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from memo.recall_logic import RankKnobs, rank_hits


@dataclass
class _Hit:
    id: str
    score: float | None
    title: str = ""
    body: str = ""
    type: str = "note"
    extra: dict[str, Any] = field(default_factory=dict)


def _mk(id: str, score: float | None, **kw: Any) -> _Hit:
    kw.setdefault("title", f"title {id}")
    kw.setdefault("body", f"distinct body for memory {id}, long enough to pass the gate")
    return _Hit(id=id, score=score, **kw)


# ── MMR diversity ────────────────────────────────────────────────────────────


def _redundant_pool() -> list[_Hit]:
    """a and b are near-identical (high Jaccard); c is diverse."""
    shared = "sqlite vec store thread local connections wal busy timeout float32 blobs"
    return [
        _mk("a", 0.90, title="vec store notes", body=f"{shared} variant alpha"),
        _mk("b", 0.85, title="vec store notes bis", body=f"{shared} variant beta"),
        _mk(
            "c",
            0.80,
            title="release workflow",
            body="bump tag push five manifests changelog keepachangelog",
        ),
    ]


def test_mmr_demotes_redundant_near_duplicate() -> None:
    # b duplicates a's content; MMR should promote the diverse c above b.
    out = rank_hits(
        _redundant_pool(),
        RankKnobs(min_sim=0.0, min_body_chars=0, mmr_lambda=0.5),
    )
    assert [h.id for h in out] == ["a", "c", "b"]


def test_mmr_first_pick_is_max_relevance() -> None:
    out = rank_hits(
        _redundant_pool(),
        RankKnobs(min_sim=0.0, min_body_chars=0, mmr_lambda=0.5),
    )
    assert out[0].id == "a"  # skip-below floor on the top hit is unaffected


def test_mmr_does_not_mutate_scores() -> None:
    out = rank_hits(
        _redundant_pool(),
        RankKnobs(min_sim=0.0, min_body_chars=0, mmr_lambda=0.5),
    )
    assert {h.id: h.score for h in out} == {"a": 0.90, "b": 0.85, "c": 0.80}


def test_mmr_lambda_zero_byte_identical() -> None:
    hits = _redundant_pool()
    baseline = rank_hits(hits, RankKnobs(min_sim=0.0, min_body_chars=0))
    off = rank_hits(hits, RankKnobs(min_sim=0.0, min_body_chars=0, mmr_lambda=0.0))
    assert len(off) == len(baseline)
    assert all(x is y for x, y in zip(off, baseline, strict=True))  # same objects


# ── synthesis boost ──────────────────────────────────────────────────────────


def test_synthesis_boost_lifts_synthesis_hit() -> None:
    hits = [_mk("raw", 0.90), _mk("syn", 0.85, type="synthesis")]
    out = rank_hits(
        hits,
        RankKnobs(min_sim=0.0, min_body_chars=0, synthesis_boost=0.10),
    )
    assert [h.id for h in out] == ["syn", "raw"]
    assert out[0].score is not None and abs(out[0].score - 0.95) < 1e-9
    assert out[1].score == 0.90  # non-synthesis hit untouched


def test_synthesis_boost_zero_byte_identical() -> None:
    hits = [_mk("raw", 0.90), _mk("syn", 0.85, type="synthesis")]
    baseline = rank_hits(hits, RankKnobs(min_sim=0.0, min_body_chars=0))
    off = rank_hits(hits, RankKnobs(min_sim=0.0, min_body_chars=0, synthesis_boost=0.0))
    assert len(off) == len(baseline)
    assert all(x is y for x, y in zip(off, baseline, strict=True))  # same objects


def test_synthesis_boost_skips_none_score() -> None:
    hits = [_mk("raw", 0.90), _mk("syn", None, type="synthesis")]
    out = rank_hits(
        hits,
        RankKnobs(min_sim=0.0, min_body_chars=0, synthesis_boost=0.10),
    )
    assert {h.id: h.score for h in out} == {"raw": 0.90, "syn": None}


# ── explain collector ────────────────────────────────────────────────────────


def test_explain_records_synthesis_boost_stage() -> None:
    hits = [_mk("raw", 0.90), _mk("syn", 0.85, type="synthesis")]
    explain: dict[str, dict[str, Any]] = {}
    rank_hits(
        hits,
        RankKnobs(min_sim=0.0, min_body_chars=0, synthesis_boost=0.10),
        explain=explain,
    )
    assert explain["syn"]["synthesis_boost"] == 0.10
    assert "synthesis_boost" not in explain["raw"]  # no delta for non-synthesis


def test_explain_records_mmr_stage_and_final_ranks() -> None:
    explain: dict[str, dict[str, Any]] = {}
    out = rank_hits(
        _redundant_pool(),
        RankKnobs(min_sim=0.0, min_body_chars=0, mmr_lambda=0.5),
        explain=explain,
    )
    assert [h.id for h in out] == ["a", "c", "b"]
    for hid in ("a", "b", "c"):
        assert "mmr" in explain[hid]
        assert "mmr_score" in explain[hid]["mmr"]
        assert "max_sim_to_selected" in explain[hid]["mmr"]
    # b was demoted because of its similarity to the already-selected a.
    assert explain["b"]["mmr"]["max_sim_to_selected"] > explain["c"]["mmr"]["max_sim_to_selected"]
    # rank reflects the MMR order.
    assert [explain[hid]["rank"] for hid in ("a", "c", "b")] == [1, 2, 3]


def test_explain_no_stages_when_both_off() -> None:
    hits = [_mk("raw", 0.90), _mk("syn", 0.85, type="synthesis")]
    explain: dict[str, dict[str, Any]] = {}
    rank_hits(hits, RankKnobs(min_sim=0.0, min_body_chars=0), explain=explain)
    for entry in explain.values():
        assert "mmr" not in entry
        assert "synthesis_boost" not in entry


# ── flag registry + _recall_logic wiring ────────────────────────────────────


def test_flags_registered_default_off() -> None:
    from memo.flags import REGISTRY, flag_float

    for name in ("MEMO_RECALL_MMR_LAMBDA", "MEMO_RECALL_SYNTHESIS_BOOST"):
        spec = REGISTRY[name]
        assert spec.kind == "float"
        assert spec.default == 0.0
        assert flag_float(name, env={}) == 0.0


def test_recall_logic_wires_flags_into_knobs(monkeypatch, tmp_path) -> None:
    import memo.recall_logic as rl

    monkeypatch.setenv("MEMO_RECALL_MMR_LAMBDA", "0.35")
    monkeypatch.setenv("MEMO_RECALL_SYNTHESIS_BOOST", "0.2")

    captured: dict[str, Any] = {}

    def fake_rank_hits(hits: list[Any], knobs: RankKnobs, **kw: Any) -> list[Any]:
        captured["knobs"] = knobs
        return []

    monkeypatch.setattr(rl, "rank_hits", fake_rank_hits)
    mem = SimpleNamespace(
        search=lambda *a, **k: [],
        embedder=SimpleNamespace(is_warm=True),
    )
    cfg = SimpleNamespace(state_dir=tmp_path)
    out, _cb = rl._recall_logic("what did we decide about the store", cwd=None, mem=mem, cfg=cfg)
    assert out == "{}"
    knobs = captured["knobs"]
    assert knobs.mmr_lambda == 0.35
    assert knobs.synthesis_boost == 0.2
