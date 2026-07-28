"""Dispute helpers for ask: batched lookup, annotation, deterministic gate."""

from __future__ import annotations

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
