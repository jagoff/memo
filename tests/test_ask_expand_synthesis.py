"""Lazy synthesis_sources expansion at ask time (MEMO_ASK_EXPAND_SYNTHESIS)."""

from __future__ import annotations


def _seed(mem):
    s1 = mem.save(content="El daemon usa un socket unix en recall.sock", title="daemon socket")
    s2 = mem.save(content="El budget del hook es 5 segundos", title="hook budget")
    synth = mem.save(
        content="El recall path está optimizado end-to-end.",
        title="Recall overview",
        type_="synthesis",
        extra={"synthesis_sources": [s1.id, s2.id]},
    )
    return s1, s2, synth


def _stub_search_to(mem, rec):
    mem.search = lambda *a, **k: [mem.get(rec.id)]  # type: ignore[assignment]


def _legacy_secret(mem, *, content: str, title: str):
    record = mem.save(content=content, title=title)
    mem.store.bulk_update_type([record.id], "secret")
    return mem.get(record.id)


def test_flag_off_no_expansion(mock_memory, monkeypatch):
    monkeypatch.delenv("MEMO_ASK_EXPAND_SYNTHESIS", raising=False)
    s1, _s2, synth = _seed(mock_memory)
    _stub_search_to(mock_memory, synth)
    _, sources, user_msg, _ = mock_memory._build_ask_context(
        "como funciona el recall?", k=3, type_=None, snippet_chars=200, include_repos=False
    )
    assert all("expanded_from" not in s for s in sources)
    assert s1.id[:8] not in user_msg


def test_flag_on_appends_bounded_sources(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_ASK_EXPAND_SYNTHESIS", "1")
    s1, s2, synth = _seed(mock_memory)
    _stub_search_to(mock_memory, synth)
    _, sources, user_msg, _ = mock_memory._build_ask_context(
        "como funciona el recall?", k=3, type_=None, snippet_chars=200, include_repos=False
    )
    expanded = [s for s in sources if s.get("expanded_from") == synth.id]
    assert {s["id"] for s in expanded} == {s1.id, s2.id}
    assert s1.id[:8] in user_msg and s2.id[:8] in user_msg
    assert f"source-of [{synth.id[:8]}]" in user_msg


def test_flag_on_skips_non_id_sources(mock_memory, monkeypatch):
    # community-kind syntheses store ENTITY NAMES in synthesis_sources —
    # non-resolvable strings must skip silently.
    monkeypatch.setenv("MEMO_ASK_EXPAND_SYNTHESIS", "1")
    synth = mock_memory.save(
        content="tema",
        title="community synth",
        type_="synthesis",
        extra={"synthesis_sources": ["not-an-id", "either"]},
    )
    _stub_search_to(mock_memory, synth)
    _, sources, _, _ = mock_memory._build_ask_context(
        "que hay?", k=3, type_=None, snippet_chars=200, include_repos=False
    )
    assert all("expanded_from" not in s for s in sources)


def test_flag_on_prefers_source_memories_key(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_ASK_EXPAND_SYNTHESIS", "1")
    real = mock_memory.save(content="dato concreto", title="fuente real")
    synth = mock_memory.save(
        content="tema",
        title="community synth",
        type_="synthesis",
        extra={"synthesis_sources": ["entity-name"], "synthesis_source_memories": [real.id]},
    )
    _stub_search_to(mock_memory, synth)
    _, sources, _, _ = mock_memory._build_ask_context(
        "que hay?", k=3, type_=None, snippet_chars=200, include_repos=False
    )
    assert any(s.get("expanded_from") == synth.id and s["id"] == real.id for s in sources)


def test_flag_on_skips_sensitive_sources_without_context_pack(mock_memory, monkeypatch):
    monkeypatch.setenv("MEMO_ASK_EXPAND_SYNTHESIS", "1")
    secret = _legacy_secret(
        mock_memory,
        content="SECRET-LEAK-REVIEW",
        title="Secret source",
    )
    synth = mock_memory.save(
        content="tema",
        title="safe synthesis",
        type_="synthesis",
        extra={"synthesis_sources": [secret.id]},
    )
    _stub_search_to(mock_memory, synth)

    _, sources, user_msg, _ = mock_memory._build_ask_context(
        "que hay?",
        k=3,
        type_=None,
        snippet_chars=200,
        include_repos=False,
        use_context_pack=False,
    )

    assert "SECRET-LEAK-REVIEW" not in user_msg
    assert all(s.get("id") != secret.id for s in sources)
