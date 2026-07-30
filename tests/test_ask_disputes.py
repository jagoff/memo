"""Dispute helpers for ask: batched lookup, annotation, deterministic gate."""

from __future__ import annotations

from memo.memory import ask_ops
from memo.memory.ask_disputes import (
    DISPUTE_PROMPT_SUFFIX,
    append_dispute_caveat,
    contested_or_none,
    dispute_header_segment,
    dispute_map,
)

ID_A = "a1b2c3d4" + "0" * 24
ID_B = "e5f6a7b8" + "0" * 24
ID_C = "99887766" + "0" * 24


class _Pair:
    def __init__(self, a, b):
        self.memory_id_a = a
        self.memory_id_b = b


class _Store:
    def __init__(self, pairs_by_status):
        self._p = pairs_by_status

    def pairs_for_ids(self, ids, *, status="open"):
        return self._p.get(status, [])


class _Mem:
    def __init__(self, store):
        self.contradict_store = store


class _BrokenMem:
    @property
    def contradict_store(self):
        raise RuntimeError("no contradictions backend")


def _src(mem_id, *, disputed_by=None, source="memory"):
    d = {
        "source": source,
        "id": mem_id,
        "id_short": mem_id[:8],
        "title": "t",
        "type": "note",
        "score": 0.9,
        "snippet": "body",
    }
    if disputed_by:
        d["disputed_by"] = list(disputed_by)
    return d


def test_dispute_map_merges_open_and_competing_both_directions():
    store = _Store({"open": [_Pair(ID_A, ID_B)], "competing": [_Pair(ID_C, ID_A)]})
    out = dispute_map(_Mem(store), [ID_A])
    assert out[ID_A] == [ID_B, ID_C]
    assert out[ID_B] == [ID_A]
    assert out[ID_C] == [ID_A]


def test_dispute_map_empty_ids_no_store_call():
    assert dispute_map(_BrokenMem(), []) == {}


def test_dispute_map_fail_open_on_store_error():
    assert dispute_map(_BrokenMem(), [ID_A]) == {}


def test_header_segment_lists_disputing_short_ids():
    seg = dispute_header_segment(ID_A, {ID_A: [ID_B]})
    assert seg == f"  |  ⚔ disputed-by: [{ID_B[:8]}]"
    assert dispute_header_segment(ID_B, {ID_A: [ID_B]}) == ""


def test_contested_when_all_cited_sources_disputed():
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    answer = f"The port is 9999 [{ID_A[:8]}]."
    msg = contested_or_none(answer, sources)
    assert msg is not None
    assert "couldn't find" in msg  # journey_check abstain marker — MUST hold
    assert ID_A[:8] in msg and ID_B[:8] in msg


def test_not_contested_when_a_clean_source_is_cited():
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    answer = f"The port is 8765 [{ID_A[:8]}] [{ID_C[:8]}]."
    assert contested_or_none(answer, sources) is None


def test_contested_without_citations_only_if_all_memory_sources_disputed():
    all_disputed = [_src(ID_A, disputed_by=[ID_B])]
    assert contested_or_none("no citations here", all_disputed) is not None
    mixed = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    assert contested_or_none("no citations here", mixed) is None


def test_no_disputes_never_contested():
    assert contested_or_none("answer [a1b2c3d4]", [_src(ID_A)]) is None


def test_caveat_appended_for_cited_disputed_source():
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    answer = f"The port is 8765 [{ID_A[:8]}] [{ID_C[:8]}]."
    out = append_dispute_caveat(answer, sources)
    assert out.startswith(answer)
    assert f"[{ID_A[:8]}] is contested by [{ID_B[:8]}]" in out


def test_caveat_skipped_when_llm_already_cited_the_disputing_id():
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    answer = f"[{ID_A[:8]}] says 8765 but [{ID_B[:8]}] disagrees; also [{ID_C[:8]}]."
    assert append_dispute_caveat(answer, sources) == answer


def test_caveat_skipped_for_uncited_disputed_source():
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    answer = f"The port is 8765 [{ID_C[:8]}]."
    assert append_dispute_caveat(answer, sources) == answer


def test_prompt_suffix_mentions_marker():
    assert "disputed-by" in DISPUTE_PROMPT_SUFFIX


# ---- integration: _build_ask_context annotation (mock_memory fixture) ----


def _seed_two(mock_memory):
    a = mock_memory.save(content="port is 8765", title="port fact A", type_="fact")
    b = mock_memory.save(content="port is 9999", title="port fact B", type_="fact")
    return a.id, b.id


def _fake_pairs(monkeypatch, a_id, b_id):
    from memo.memory import ask_disputes

    monkeypatch.setattr(
        ask_disputes,
        "dispute_map",
        lambda mem, ids: {a_id: [b_id], b_id: [a_id]} if a_id in ids else {},
    )


def test_build_ask_context_annotates_disputed_sources(mock_memory, monkeypatch):
    a_id, b_id = _seed_two(mock_memory)
    _fake_pairs(monkeypatch, a_id, b_id)
    _, sources, user_msg, _ = mock_memory._build_ask_context(
        "what port?", k=5, type_=None, snippet_chars=200, include_repos=False
    )
    by_id = {s["id"]: s for s in sources}
    assert by_id[a_id]["disputed_by"] == [b_id]
    assert f"⚔ disputed-by: [{b_id[:8]}]" in user_msg


def test_build_ask_context_flag_off_no_annotation(mock_memory, monkeypatch):
    a_id, b_id = _seed_two(mock_memory)
    _fake_pairs(monkeypatch, a_id, b_id)
    monkeypatch.setenv("MEMO_ASK_DISPUTES", "0")
    _, sources, user_msg, _ = mock_memory._build_ask_context(
        "what port?", k=5, type_=None, snippet_chars=200, include_repos=False
    )
    assert all("disputed_by" not in s for s in sources)
    assert "disputed-by" not in user_msg


# ---- gate in ask(): stubbed-context pattern from tests/test_ask_strict.py ----


class _Chat:
    def __init__(self, answer):
        self._a = answer

    def chat(self, **k):
        return {"message": {"content": self._a}}

    def chat_stream(self, **k):
        yield self._a


def _prep_gate(mock_memory, monkeypatch, *, answer, sources):
    monkeypatch.setattr(
        ask_ops._AskOpsMixin,
        "_build_ask_context",
        lambda self, q, **k: (q, list(sources), "ctx", []),
    )
    monkeypatch.setattr(ask_ops._AskOpsMixin, "_verbatim_short_circuit", lambda self, q, h: None)
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: _Chat(answer))


def test_ask_contested_abstention_when_only_disputed_cited(mock_memory, monkeypatch):
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    _prep_gate(mock_memory, monkeypatch, answer=f"Port is 9999 [{ID_A[:8]}].", sources=sources)
    out = mock_memory.ask("what port?")
    assert out["abstained"] == "disputed"
    assert "couldn't find" in out["answer"]
    assert out["disputed"] == {ID_A: [ID_B]}
    assert out["sources"]  # sources still returned


def test_ask_caveat_on_partially_disputed_answer(mock_memory, monkeypatch):
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    _prep_gate(
        mock_memory,
        monkeypatch,
        answer=f"Port is 8765 [{ID_A[:8]}] [{ID_C[:8]}].",
        sources=sources,
    )
    out = mock_memory.ask("what port?")
    assert "abstained" not in out
    assert f"[{ID_A[:8]}] is contested by [{ID_B[:8]}]" in out["answer"]
    assert out["disputed"] == {ID_A: [ID_B]}


def test_ask_clean_sources_unchanged_shape(mock_memory, monkeypatch):
    sources = [_src(ID_A), _src(ID_C)]
    _prep_gate(mock_memory, monkeypatch, answer=f"Port is 8765 [{ID_A[:8]}].", sources=sources)
    out = mock_memory.ask("what port?")
    assert "disputed" not in out and "abstained" not in out
    assert out["answer"] == f"Port is 8765 [{ID_A[:8]}]."


def test_ask_contested_skips_grounding_judge(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    calls = []
    monkeypatch.setattr(ask_ops, "score_grounding", lambda *a, **k: calls.append(1) or 1.0)
    sources = [_src(ID_A, disputed_by=[ID_B])]
    _prep_gate(mock_memory, monkeypatch, answer=f"Port is 9999 [{ID_A[:8]}].", sources=sources)
    out = mock_memory.ask("what port?")
    assert out["abstained"] == "disputed"
    assert calls == []  # contested abstention short-circuits the judge LLM call


def test_ask_judge_abstention_gets_no_caveat(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    monkeypatch.setenv("MEMO_ASK_FALLBACK_MSG", "I couldn't find that.")
    monkeypatch.setattr(ask_ops, "score_grounding", lambda *a, **k: 0.1)
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    _prep_gate(
        mock_memory,
        monkeypatch,
        answer=f"Port is 8765 [{ID_A[:8]}] [{ID_C[:8]}].",
        sources=sources,
    )
    out = mock_memory.ask("what port?")
    assert out["answer"] == "I couldn't find that."  # fallback, no caveat appended


def test_ask_stream_done_event_contested(mock_memory, monkeypatch):
    sources = [_src(ID_A, disputed_by=[ID_B])]
    _prep_gate(mock_memory, monkeypatch, answer=f"Port is 9999 [{ID_A[:8]}].", sources=sources)
    events = list(mock_memory.ask_stream("what port?"))
    done = next(e for e in events if e.get("event") == "done")
    assert done["abstained"] == "disputed"
    assert "couldn't find" in done["answer"]
    assert done["disputed"] == {ID_A: [ID_B]}


def test_verbatim_short_circuit_skipped_for_disputed_top_hit(mock_memory, monkeypatch):
    # A disputed hit must NOT be dumped verbatim (it would bypass the gate);
    # it falls through to the LLM path, where the gate abstains.
    sources = [_src(ID_A, disputed_by=[ID_B])]

    class _H:
        id = ID_A
        body = "port is 9999"

    monkeypatch.setattr(
        ask_ops._AskOpsMixin,
        "_build_ask_context",
        lambda self, q, **k: (q, list(sources), "ctx", [_H()]),
    )
    calls = []

    def _fake_verbatim(self, q, hits):
        calls.append([h.id for h in hits])
        return None

    monkeypatch.setattr(ask_ops._AskOpsMixin, "_verbatim_short_circuit", _fake_verbatim)
    monkeypatch.setattr(
        ask_ops, "_filter_verbatim_hits", lambda hits, sources, use_context_pack=False: hits
    )
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: _Chat("nope"))
    mock_memory.ask("port is 9999")
    assert calls and ID_A not in calls[0]  # disputed hit filtered out


# ---- final-review fixes: LLM-error masking + disputed map on verbatim ----


class _ChatRaises:
    def chat(self, **k):
        raise RuntimeError("boom")


def test_ask_llm_error_not_masked_as_disputed_abstention(mock_memory, monkeypatch):
    # All sources disputed + the LLM call raises: the error sentinel has no
    # citations, so contested_or_none would (incorrectly) fire on it — the
    # gate must be skipped entirely and the error returned untouched.
    sources = [_src(ID_A, disputed_by=[ID_B])]
    _prep_gate(mock_memory, monkeypatch, answer="unused", sources=sources)
    monkeypatch.setattr(mock_memory, "_ensure_chat", lambda: _ChatRaises())
    out = mock_memory.ask("what port?")
    assert out["answer"].startswith("(error querying the model:")
    assert "abstained" not in out


def test_ask_verbatim_return_includes_disputed_map(mock_memory, monkeypatch):
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    monkeypatch.setattr(
        ask_ops._AskOpsMixin,
        "_build_ask_context",
        lambda self, q, **k: (q, list(sources), "ctx", []),
    )
    monkeypatch.setattr(
        ask_ops._AskOpsMixin,
        "_verbatim_short_circuit",
        lambda self, q, h: "verbatim body [a1b2c3d4]",
    )
    out = mock_memory.ask("literal phrase")
    assert out["answer"] == "verbatim body [a1b2c3d4]"
    assert out["disputed"] == {ID_A: [ID_B]}


def test_ask_stream_verbatim_done_includes_disputed_map(mock_memory, monkeypatch):
    sources = [_src(ID_A, disputed_by=[ID_B]), _src(ID_C)]
    monkeypatch.setattr(
        ask_ops._AskOpsMixin,
        "_build_ask_context",
        lambda self, q, **k: (q, list(sources), "ctx", []),
    )
    monkeypatch.setattr(
        ask_ops._AskOpsMixin,
        "_verbatim_short_circuit",
        lambda self, q, h: "verbatim body [a1b2c3d4]",
    )
    events = list(mock_memory.ask_stream("literal phrase"))
    done = next(e for e in events if e.get("event") == "done")
    assert done["answer"] == "verbatim body [a1b2c3d4]"
    assert done["disputed"] == {ID_A: [ID_B]}


# ---- unified abstention vocabulary: disputed / low_confidence / no_evidence ----


def test_ask_judge_abstention_marks_low_confidence(mock_memory, monkeypatch):
    # The grounding-judge abstention must be machine-readable like the
    # disputed one — consumers should never have to sniff the fallback text.
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    monkeypatch.setenv("MEMO_ASK_FALLBACK_MSG", "I couldn't find that.")
    monkeypatch.setattr(ask_ops, "score_grounding", lambda *a, **k: 0.1)
    sources = [_src(ID_A), _src(ID_C)]
    _prep_gate(mock_memory, monkeypatch, answer="Port is 1234.", sources=sources)
    out = mock_memory.ask("what port?")
    assert out["answer"] == "I couldn't find that."
    assert out["abstained"] == "low_confidence"


def test_ask_stream_judge_abstention_marks_low_confidence(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    monkeypatch.setenv("MEMO_ASK_FALLBACK_MSG", "I couldn't find that.")
    monkeypatch.setattr(ask_ops, "score_grounding", lambda *a, **k: 0.1)
    sources = [_src(ID_A), _src(ID_C)]
    _prep_gate(mock_memory, monkeypatch, answer="Port is 1234.", sources=sources)
    events = list(mock_memory.ask_stream("what port?"))
    done = next(e for e in events if e.get("event") == "done")
    assert done["answer"] == "I couldn't find that."
    assert done["abstained"] == "low_confidence"


def test_ask_no_sources_marks_no_evidence(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_ASK_FALLBACK_MSG", "I couldn't find that.")
    _prep_gate(mock_memory, monkeypatch, answer="unused", sources=[])
    out = mock_memory.ask("what port?")
    assert out["answer"] == "I couldn't find that."
    assert out["sources"] == []
    assert out["abstained"] == "no_evidence"


def test_ask_stream_no_sources_marks_no_evidence(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_ASK_FALLBACK_MSG", "I couldn't find that.")
    _prep_gate(mock_memory, monkeypatch, answer="unused", sources=[])
    events = list(mock_memory.ask_stream("what port?"))
    done = next(e for e in events if e.get("event") == "done")
    assert done["answer"] == "I couldn't find that."
    assert done["abstained"] == "no_evidence"


def test_ask_grounded_answer_not_marked_abstained(mock_memory, monkeypatch):
    # Judge passes: no abstained field at all (same shape as before).
    monkeypatch.setenv("MEMO_GROUNDING_ASK_MIN", "0.85")
    monkeypatch.setattr(ask_ops, "score_grounding", lambda *a, **k: 0.95)
    sources = [_src(ID_A), _src(ID_C)]
    _prep_gate(mock_memory, monkeypatch, answer=f"Port is 8765 [{ID_A[:8]}].", sources=sources)
    out = mock_memory.ask("what port?")
    assert "abstained" not in out
