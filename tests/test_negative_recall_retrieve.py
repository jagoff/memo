"""RETRIEVE + TRIGGER slice tests for the Negative Recall ⛔ AVOID channel.

Covers the flag-gated wiring added to ``memo.recall_logic`` — the ⛔ retrieval
pass, its budget/trigger gating, the normal-recall exclusion, and the end-to-end
splice into the recall string. Everything uses a STUBBED embedder (no real MLX):
the pure helpers run against a fake ``mem`` object, and the integration tests
mirror the ``test_recall_logic_synthesis`` pattern (real ``Memory`` + a 4-dim
marker embedder) so a query mentioning the marker has cosine 1.0 with the marked
docs and 0.0 with the rest — deterministic, no MLX cold-load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from memo.negative_recall import AVOID_BLOCK_HEADER, FAILURE_PATTERN_TYPE
from memo.recall_logic import (
    _NEGATIVE_RECALL_K_CAP,
    RECALL_HEADER,
    _avoid_only_output,
    _negative_budget_ok,
    _negative_recall_block,
    _negative_recall_hits,
    _recall_excluded_types,
    _recall_logic,
    _widen_negative_params,
)

_STRUCTURED_BODY = (
    "Pattern: reverting embeddings to Ollama\n"
    "Context: choosing the embedder backend\n"
    "Wrong: switched embeddings back to Ollama for speed\n"
    "Right: keep MLX embeddings; Ollama regressed retrieval"
)


@dataclass(frozen=True)
class _Hit:
    """Structural stand-in for a MemoryRecord returned by ``mem.search``."""

    id: str
    title: str
    body: str
    score: float
    extra: dict[str, Any] = field(default_factory=dict)


class _StubMem:
    """Fake ``Memory`` whose ``search`` returns canned hits and records kwargs."""

    def __init__(self, hits: list[_Hit], *, raise_exc: bool = False) -> None:
        self._hits = hits
        self._raise = raise_exc
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def search(self, query: str, **kwargs: Any) -> list[_Hit]:
        self.calls.append((query, kwargs))
        if self._raise:
            raise RuntimeError("boom")
        return list(self._hits)


# ── _recall_excluded_types (normal-recall exclusion / no-dup mechanism) ───────


def test_excluded_types_omits_failure_pattern_by_default(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_ENABLED", raising=False)
    excluded = _recall_excluded_types()
    assert "secret" in excluded
    assert FAILURE_PATTERN_TYPE not in excluded  # flows into normal recall as today


def test_excluded_types_adds_failure_pattern_when_enabled(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")
    excluded = _recall_excluded_types()
    assert FAILURE_PATTERN_TYPE in excluded  # dropped from normal → own ⛔ block
    assert "secret" in excluded  # unconditional exclusion preserved


# ── _widen_negative_params (trigger widening — pure) ──────────────────────────


def test_widen_is_exact_noop_without_risk() -> None:
    assert _widen_negative_params(2, 0.6, 0.0) == (2, 0.6)


def test_widen_raises_k_and_lowers_floor_with_risk() -> None:
    widened_k, widened_floor = _widen_negative_params(2, 0.6, 1.0)
    assert widened_k == 4  # 2 + round(1.0 * 2)
    assert widened_floor == pytest.approx(0.45)  # 0.6 - 1.0 * 0.15


def test_widen_caps_k_and_floors_at_zero() -> None:
    # K widening never exceeds the hard cap, even from an already-high base.
    assert _widen_negative_params(5, 0.6, 1.0)[0] == _NEGATIVE_RECALL_K_CAP
    assert _widen_negative_params(_NEGATIVE_RECALL_K_CAP, 0.6, 1.0)[0] == _NEGATIVE_RECALL_K_CAP
    # The loosened floor never goes negative.
    assert _widen_negative_params(2, 0.1, 1.0)[1] == 0.0


# ── _negative_budget_ok (⛔ yields FIRST under token pressure) ─────────────────


def test_budget_ok_when_unlimited_or_ample() -> None:
    assert _negative_budget_ok(0, risk=0.0) is True  # 0 == unlimited
    assert _negative_budget_ok(500, risk=0.0) is True


def test_budget_starved_skips_unless_risky() -> None:
    assert _negative_budget_ok(50, risk=0.0) is False  # below floor, no risk → yield
    assert _negative_budget_ok(50, risk=0.5) is True  # high-risk context overrides yield


# ── _negative_recall_hits (the ⛔ retrieval pass) ─────────────────────────────


def test_negative_hits_apply_floor_cap_and_cheap_search_kwargs() -> None:
    mem = _StubMem(
        [
            _Hit("a" * 8, "t1", _STRUCTURED_BODY, 0.9),
            _Hit("b" * 8, "t2", _STRUCTURED_BODY, 0.7),
            _Hit("c" * 8, "t3", _STRUCTURED_BODY, 0.5),  # below floor
            _Hit("d" * 8, "t4", _STRUCTURED_BODY, 0.3),  # below floor
        ]
    )
    hits = _negative_recall_hits(mem, "q", neg_k=2, neg_min_sim=0.6, exclude_tags=None)
    assert [h.id for h in hits] == ["a" * 8, "b" * 8]  # floor drops 0.5/0.3, cap keeps 2

    _query, kwargs = mem.calls[0]
    # Single-type vec kNN reusing the cached query embedding (mode="vec" ⇒ the
    # embed_query LRU hit), reranker disabled, usage tracking off — hook-cheap.
    assert kwargs["type_"] == FAILURE_PATTERN_TYPE
    assert kwargs["mode"] == "vec"
    assert kwargs["disable_reranker"] is True
    assert kwargs["_track_usage"] is False
    assert kwargs["limit"] == 6  # over-fetch: max(neg_k * 3, neg_k)


def test_negative_hits_zero_k_short_circuits_without_searching() -> None:
    mem = _StubMem([_Hit("a" * 8, "t", _STRUCTURED_BODY, 0.9)])
    assert _negative_recall_hits(mem, "q", neg_k=0, neg_min_sim=0.6, exclude_tags=None) == []
    assert mem.calls == []  # no store round-trip


def test_negative_hits_swallow_search_failure() -> None:
    mem = _StubMem([], raise_exc=True)
    assert _negative_recall_hits(mem, "q", neg_k=2, neg_min_sim=0.6, exclude_tags=None) == []


# ── _negative_recall_block (gating + render) ──────────────────────────────────


def _enabled(monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")


def test_block_default_off_is_noop(monkeypatch) -> None:
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_ENABLED", raising=False)
    mem = _StubMem([_Hit("a" * 8, "t", _STRUCTURED_BODY, 0.9)])
    block = _negative_recall_block(mem, "q", exclude_tags=None, token_budget=1000, can_embed=True)
    assert block == ""
    assert mem.calls == []  # feature off ⇒ never searches


def test_block_renders_avoid_block_when_relevant(monkeypatch) -> None:
    _enabled(monkeypatch)
    mem = _StubMem([_Hit("abcd1234", "reverting to ollama", _STRUCTURED_BODY, 0.9)])
    block = _negative_recall_block(mem, "q", exclude_tags=None, token_budget=1000, can_embed=True)
    assert block.startswith(AVOID_BLOCK_HEADER)
    assert "✗" in block and "✓" in block
    assert "[abcd1234]" in block


def test_block_skipped_when_cannot_embed(monkeypatch) -> None:
    _enabled(monkeypatch)
    mem = _StubMem([_Hit("a" * 8, "t", _STRUCTURED_BODY, 0.9)])
    block = _negative_recall_block(mem, "q", exclude_tags=None, token_budget=1000, can_embed=False)
    assert block == ""
    assert mem.calls == []


def test_block_yields_first_under_token_pressure(monkeypatch) -> None:
    _enabled(monkeypatch)  # no trigger ⇒ risk 0 ⇒ starved budget yields
    mem = _StubMem([_Hit("a" * 8, "t", _STRUCTURED_BODY, 0.9)])
    block = _negative_recall_block(mem, "q", exclude_tags=None, token_budget=50, can_embed=True)
    assert block == ""
    assert mem.calls == []  # bailed before the store round-trip


def test_trigger_widens_floor_on_risky_prompt(monkeypatch) -> None:
    _enabled(monkeypatch)
    # Hit sits at 0.56 — below the 0.6 default floor, above the trigger-loosened
    # floor (2 risk signals ⇒ risk 2/3 ⇒ floor 0.6 - 0.10 = 0.50).
    hit = _Hit("abcd1234", "release trap", _STRUCTURED_BODY, 0.56)
    risky_prompt = "cut a release and delete the old tag"

    # Trigger OFF: the base floor drops the sub-floor hit ⇒ nothing surfaces.
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_TRIGGER_ENABLED", raising=False)
    off = _negative_recall_block(
        _StubMem([hit]), risky_prompt, exclude_tags=None, token_budget=1000, can_embed=True
    )
    assert off == ""

    # Trigger ON: the loosened floor lets the risk-context anti-memory through.
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_TRIGGER_ENABLED", "1")
    on = _negative_recall_block(
        _StubMem([hit]), risky_prompt, exclude_tags=None, token_budget=1000, can_embed=True
    )
    assert on.startswith(AVOID_BLOCK_HEADER)


# ── _avoid_only_output (⛔ can fire alone) ─────────────────────────────────────


def test_avoid_only_output_wraps_block_in_hook_envelope() -> None:
    out = json.loads(_avoid_only_output("⛔ AVOID — thing"))
    hook = out["hookSpecificOutput"]
    assert hook["hookEventName"] == "UserPromptSubmit"
    assert hook["additionalContext"] == "⛔ AVOID — thing"


# ── Integration: real Memory + stubbed marker embedder ───────────────────────


def _marker_vec(text: str) -> list[float]:
    """4-dim marker embedder: any text mentioning AVOIDME → [1,0,0,0], else
    [0,1,0,0]. A query mentioning AVOIDME has cosine 1.0 with marked docs."""
    return [1.0, 0.0, 0.0, 0.0] if "AVOIDME" in (text or "") else [0.0, 1.0, 0.0, 0.0]


def _make_mem(tmp_path: Path, monkeypatch, *, stub_query: bool = True):
    from memo.config import Config
    from memo.memory import Memory

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed", lambda self, inputs: [_marker_vec(t) for t in inputs]
    )
    if stub_query:
        monkeypatch.setattr("memo.embedder.MLXEmbedder.embed_query", lambda self, q: _marker_vec(q))
    # Deterministic vec-only recall over the cold stub embedder.
    monkeypatch.setenv("MEMO_RECALL_MODE", "vec")
    monkeypatch.setenv("MEMO_RECALL_FORCE_MODE", "1")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.5")
    monkeypatch.setenv("MEMO_RECALL_FORMAT", "full")
    # Isolate the ⛔ channel: silence the orthogonal graph-associative nudge that
    # otherwise references a connected memory by id in the normal-section tail.
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "0")
    cfg = Config(
        data_dir=tmp_path / "data",
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / "state",
        embedder_dims=4,
        reranker_enabled=False,
    )
    return Memory(cfg), cfg


_FP_CONTENT = "AVOIDME\n" + _STRUCTURED_BODY
_NOTE_CONTENT = (
    "AVOIDME the warm recall daemon keeps MLX resident so the recall hook "
    "stays under its latency budget across sessions"
)
_QUERY = "AVOIDME which embedder backend should we use"


def test_enabled_surfaces_avoid_block_and_excludes_from_normal(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")
    mem, cfg = _make_mem(tmp_path, monkeypatch)
    mem.save(content=_FP_CONTENT, title="reverting to ollama", type_=FAILURE_PATTERN_TYPE)
    mem.save(content=_NOTE_CONTENT, title="warm daemon note", type_="note")

    raw, _cb = _recall_logic(_QUERY, cwd=None, mem=mem, cfg=cfg)
    ctx = json.loads(raw)["hookSpecificOutput"]["additionalContext"]

    assert AVOID_BLOCK_HEADER in ctx  # distinct ⛔ block present
    split = ctx.index(RECALL_HEADER)  # ⛔ sits above the normal "## Memory" section
    avoid_part, normal_part = ctx[:split].lower(), ctx[split:].lower()

    assert "reverting to ollama" in avoid_part  # failure_pattern is in the ⛔ block
    assert "reverting to ollama" not in normal_part  # excluded from normal (no dup)
    assert "warm daemon note" in normal_part  # a normal hit still renders normally
    mem.close()


def test_disabled_is_noop_failure_pattern_flows_into_normal(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_ENABLED", raising=False)
    mem, cfg = _make_mem(tmp_path, monkeypatch)
    mem.save(content=_FP_CONTENT, title="reverting to ollama", type_=FAILURE_PATTERN_TYPE)
    mem.save(content=_NOTE_CONTENT, title="warm daemon note", type_="note")

    raw, _cb = _recall_logic(_QUERY, cwd=None, mem=mem, cfg=cfg)
    ctx = json.loads(raw)["hookSpecificOutput"]["additionalContext"]

    assert AVOID_BLOCK_HEADER not in ctx  # no ⛔ block when the feature is off
    assert "reverting to ollama" in ctx.lower()  # fp is a plain hit, exactly as today
    mem.close()


def test_avoid_fires_alone_when_normal_recall_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")
    mem, cfg = _make_mem(tmp_path, monkeypatch)
    # Only a failure_pattern exists — normal recall excludes it and finds nothing.
    mem.save(content=_FP_CONTENT, title="reverting to ollama", type_=FAILURE_PATTERN_TYPE)

    raw, _cb = _recall_logic(_QUERY, cwd=None, mem=mem, cfg=cfg)
    ctx = json.loads(raw)["hookSpecificOutput"]["additionalContext"]

    assert AVOID_BLOCK_HEADER in ctx  # the anti-memory surfaces on its own
    assert RECALL_HEADER not in ctx  # no normal "## Memory" section rode along
    mem.close()


# ── Fix: associative recall must not resurface an excluded failure_pattern ────
#
# The ⛔ pass excludes failure_patterns from the normal ## Memory section, but
# the graph associate() walk excludes only seed_ids — so a failure_pattern
# connected to a normal seed could reappear in the "🔗 Also connected" tail, a
# duplicate of the ⛔ block. build_nudge must drop it when the feature is on.


class _AssocRec:
    def __init__(self, id_: str, title: str, type_: str) -> None:
        self.id, self.title, self.type = id_, title, type_
        self.extra: dict[str, Any] = {}
        self.updated = "2026-01-01T00:00:00+00:00"


class _AssocMem:
    graph = object()  # _verified_pair_ids degrades to set() (no _conn)

    def __init__(self, recs: dict[str, _AssocRec]) -> None:
        self._recs = recs

    def get(self, id_: str) -> Any:
        return self._recs.get(id_)


_FP_ID, _NOTE_ID = "f" * 32, "a" * 32


def _assoc_setup(monkeypatch) -> None:
    from memo import recall_assoc
    from memo.associative import AssociativeHit

    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "1")
    monkeypatch.setattr(recall_assoc, "_codegraph_adj", lambda: None)
    monkeypatch.setattr(
        recall_assoc,
        "associate",
        lambda *a, **k: [
            AssociativeHit(_FP_ID, "ent", 0.9),
            AssociativeHit(_NOTE_ID, "ent", 0.8),
        ],
    )


def _run_build_nudge() -> list[str]:
    from memo.recall_assoc import build_nudge

    recs = {
        _FP_ID: _AssocRec(_FP_ID, "the trap", FAILURE_PATTERN_TYPE),
        _NOTE_ID: _AssocRec(_NOTE_ID, "a note", "note"),
    }
    seeds = [type("S", (), {"id": "seed0000"})()]
    return [item.id for item in build_nudge(_AssocMem(recs), seeds)]


def test_build_nudge_excludes_failure_pattern_when_negative_recall_on(monkeypatch) -> None:
    _assoc_setup(monkeypatch)
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")

    ids = _run_build_nudge()

    assert _FP_ID not in ids  # surfaced only in the ⛔ block — never the assoc tail
    assert _NOTE_ID in ids  # a normal graph-neighbour still surfaces


def test_build_nudge_keeps_failure_pattern_when_feature_off(monkeypatch) -> None:
    _assoc_setup(monkeypatch)
    monkeypatch.delenv("MEMO_NEGATIVE_RECALL_ENABLED", raising=False)

    ids = _run_build_nudge()

    assert _FP_ID in ids  # feature off ⇒ today's behavior unchanged
    assert _NOTE_ID in ids


def test_negative_pass_reuses_query_embedding_no_second_forward(tmp_path, monkeypatch) -> None:
    # Query cache ON (default) so the ⛔ pass's mem.search(mode="vec") is a cache
    # hit on the embedding the main search already computed.
    monkeypatch.setenv("MEMO_QUERY_CACHE_SIZE", "256")
    monkeypatch.setenv("MEMO_NEGATIVE_RECALL_ENABLED", "1")

    # Leave the REAL cache-backed embed_query in place (stub_query=False), then
    # override embed AFTER _make_mem (which sets its own marker embed) so the
    # counter is the one that runs.
    mem, cfg = _make_mem(tmp_path, monkeypatch, stub_query=False)

    embed_inputs: list[list[str]] = []

    def _counting_embed(self, inputs):
        embed_inputs.append(list(inputs))
        return [_marker_vec(t) for t in inputs]

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _counting_embed)
    mem.save(content=_FP_CONTENT, title="reverting to ollama", type_=FAILURE_PATTERN_TYPE)
    mem.save(content=_NOTE_CONTENT, title="warm daemon note", type_="note")

    embed_inputs.clear()  # discount document embeds done at save time
    raw, _cb = _recall_logic(_QUERY, cwd=None, mem=mem, cfg=cfg)
    ctx = json.loads(raw)["hookSpecificOutput"]["additionalContext"]

    # The query is embedded exactly ONCE for the whole recall — the ⛔ pass reused
    # the cached embedding rather than issuing a second MLX forward.
    assert len(embed_inputs) == 1
    assert AVOID_BLOCK_HEADER in ctx  # and the ⛔ pass really ran
    mem.close()
