from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from memo.context_pack import build_context_pack
from memo.memory import Memory
from memo.memory.ask_ops import _fit_context_pack_prompt
from memo.repo_index import RepoSearchHit


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float
    title: str
    body: str
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _repo_hit(*, text: str) -> RepoSearchHit:
    return RepoSearchHit(
        id="repo-hit",
        repo_id="r1",
        repo_name="code-repo",
        url="x",
        ref="HEAD",
        commit_sha="deadbeef",
        file_id="f1",
        path="src/foo.py",
        language="python",
        line_start=10,
        line_end=12,
        text=text,
        score=0.6,
        match_type="hybrid",
    )


def _capturing_stream(
    prompts: list[str],
    deltas: list[str],
):
    def _stream(
        self, model: str, messages: list[dict[str, str]], options: dict[str, Any] | None = None
    ) -> Iterator[str]:
        prompts.append(messages[-1]["content"])
        yield from deltas

    return _stream


def test_build_context_pack_separates_current_and_stale() -> None:
    pack = build_context_pack(
        "what is current?",
        [
            _Hit("new-current-id", 0.8, "Current", "Use the new plan."),
            _Hit("old-stale-id", 0.9, "Old", "Use the old plan.", extra={"superseded_by": "new"}),
        ],
        snippet_chars=80,
    )
    assert [s["id"] for s in pack.current_facts] == ["new-current-id"]
    assert [s["id"] for s in pack.stale_or_conflicting] == ["old-stale-id"]
    assert "current" in pack.summary.lower()
    assert "stale/conflicting" in pack.to_prompt()


def test_build_context_pack_budget_trims_supporting_before_current() -> None:
    hits = [
        _Hit("current", 0.8, "Current", "A" * 200),
        _Hit("support", 0.7, "Support", "B" * 200),
        _Hit("stale", 0.9, "Stale", "C" * 200, extra={"invalidated": True}),
    ]
    pack = build_context_pack("q", hits, snippet_chars=200, budget_chars=500)
    prompt = pack.to_prompt()
    assert "[current]" in prompt
    assert len(prompt) <= 500


def test_build_context_pack_trims_single_oversized_current_fact_to_budget() -> None:
    hit = _Hit("current-id", 0.8, "Current", "A" * 10000)
    pack = build_context_pack("q", [hit], snippet_chars=10000, budget_chars=4000)
    prompt = pack.to_prompt()

    assert len(prompt) <= 4000
    assert "[current-]" in prompt
    assert "title: Current | type: note | quality: current" in prompt
    assert pack.current_facts[0]["id"] == "current-id"


def test_build_context_pack_ignores_malformed_optional_quality_metadata() -> None:
    pack = build_context_pack(
        "q",
        [_Hit("current-id", 0.8, "Current", "Body", extra={"support_count": "many", "roi_score": "high"})],
        snippet_chars=100,
        budget_chars=4000,
    )

    assert pack.current_facts[0]["id"] == "current-id"
    assert pack.current_facts[0]["quality_bucket"] == "current"


def test_ask_uses_context_pack_only_when_enabled(mem_with_stub, monkeypatch) -> None:
    rec = mem_with_stub.save(content="alpha body", title="Alpha")
    captured: dict[str, str] = {}

    def _stub_chat(self, model, messages, options=None):
        captured["user"] = messages[-1]["content"]
        return {"message": {"content": f"Answer [{rec.id[:8]}]."}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)

    monkeypatch.setenv("MEMO_CONTEXT_PACK", "0")
    off = mem_with_stub.ask("what about alpha?", k=1)
    assert "Relevant context pack" not in captured["user"]
    assert "quality_bucket" not in off["sources"][0]

    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    on = mem_with_stub.ask("what about alpha?", k=1)
    assert "Relevant context pack" in captured["user"]
    assert "Context summary:" in captured["user"]
    assert on["sources"][0]["quality_bucket"] == "current"


def test_context_pack_prompt_keeps_repo_snippets(mem_with_stub, monkeypatch) -> None:
    mem_with_stub.save(content="alpha body", title="Alpha")

    monkeypatch.setattr(
        type(mem_with_stub.store), "list_repo_sources", lambda self, **kw: [{"name": "code-repo"}]
    )
    monkeypatch.setattr(
        Memory,
        "repo_search",
        lambda self, q, **kw: [_repo_hit(text="def alpha(): pass")],
    )

    _, sources, user_msg, _ = mem_with_stub._build_ask_context(
        "alpha?",
        k=3,
        type_=None,
        snippet_chars=120,
        include_repos=True,
        use_context_pack=True,
    )

    assert "Relevant context pack" in user_msg
    assert "Repository snippets:" in user_msg
    assert any(source["source"] == "repo" for source in sources)


def test_context_pack_final_budget_enforces_budget_with_expanded_and_repo(
    mem_with_stub, monkeypatch
) -> None:
    current = mem_with_stub.save(content="current " + ("A" * 900), title="Current")
    support = mem_with_stub.save(content="support " + ("B" * 900), title="Support")
    stale = mem_with_stub.save(
        content="stale " + ("C" * 900),
        title="Stale",
        extra={"invalidated": True},
    )
    expanded = mem_with_stub.save(content="expanded " + ("D" * 900), title="Expanded")
    synth = mem_with_stub.save(
        content="synthesis " + ("E" * 900),
        title="Synthesis",
        type_="synthesis",
        extra={"synthesis_sources": [expanded.id]},
    )

    monkeypatch.setenv("MEMO_ASK_EXPAND_SYNTHESIS", "1")
    monkeypatch.setattr(
        mem_with_stub,
        "search",
        lambda *args, **kwargs: [
            mem_with_stub.get(current.id),
            mem_with_stub.get(support.id),
            mem_with_stub.get(stale.id),
            mem_with_stub.get(synth.id),
        ],
    )
    monkeypatch.setattr(
        type(mem_with_stub.store), "list_repo_sources", lambda self, **kw: [{"name": "code-repo"}]
    )
    monkeypatch.setattr(Memory, "repo_search", lambda self, q, **kw: [_repo_hit(text="R" * 900)])

    _, sources, user_msg, _ = mem_with_stub._build_ask_context(
        "alpha?",
        k=1,
        type_=None,
        snippet_chars=250,
        include_repos=True,
        use_context_pack=True,
    )

    context_text = user_msg.split("Relevant context pack", 1)[1]
    assert len(context_text) <= 2000
    assert "Expanded source memories:" in context_text
    assert "Repository snippets:" in context_text
    assert any(source.get("expanded_from") == synth.id for source in sources)
    assert any(source["source"] == "repo" for source in sources)


def test_context_pack_budget_trims_supporting_and_stale_before_expanded_and_repo() -> None:
    pack = build_context_pack(
        "q",
        [
            _Hit("current", 0.8, "Current", "A" * 200),
            _Hit("support", 0.7, "Support", "B" * 200),
            _Hit("stale", 0.6, "Stale", "C" * 200, extra={"invalidated": True}),
        ],
        snippet_chars=200,
        budget_chars=0,
    )
    expanded_rows = [
        {
            "id": "expanded",
            "id_short": "expanded",
            "title": "Expanded",
            "type": "note",
            "snippet": "D" * 200,
            "quality_bucket": "current",
            "quality_reasons": [],
            "context_note": "source-of [synth]",
        }
    ]
    repo_rows = [
        {
            "id": "repo",
            "id_short": "repo-hit",
            "path": "src/foo.py",
            "line_start": 10,
            "line_end": 12,
            "match_type": "hybrid",
            "snippet": "R" * 200,
        }
    ]

    prompt, current_rows, supporting_rows, stale_rows, kept_expanded_rows, kept_repo_rows = (
        _fit_context_pack_prompt(
            pack,
            expanded_rows=expanded_rows,
            repo_rows=repo_rows,
            budget_chars=1100,
        )
    )

    assert len(prompt) <= 1100
    assert len(current_rows) == 1
    assert supporting_rows == []
    assert stale_rows == []
    assert len(kept_expanded_rows) == 1
    assert len(kept_repo_rows) == 1


def test_context_pack_omits_sensitive_expanded_memory(mem_with_stub, monkeypatch) -> None:
    sensitive = mem_with_stub.save(
        content="top secret",
        title="Sensitive source",
        type_="secret",
    )
    synth = mem_with_stub.save(
        content="synthesis body",
        title="Synthesis",
        type_="synthesis",
        extra={"synthesis_sources": [sensitive.id]},
    )

    monkeypatch.setenv("MEMO_ASK_EXPAND_SYNTHESIS", "1")
    monkeypatch.setattr(mem_with_stub, "search", lambda *args, **kwargs: [mem_with_stub.get(synth.id)])

    _, sources, user_msg, _ = mem_with_stub._build_ask_context(
        "alpha?",
        k=2,
        type_=None,
        snippet_chars=120,
        include_repos=False,
        use_context_pack=True,
    )

    assert sensitive.id[:8] not in user_msg
    assert "sensitive expanded source memory omitted" in user_msg
    assert all(source.get("expanded_from") != synth.id for source in sources)


def test_context_pack_expanded_memory_gets_quality_metadata(mem_with_stub, monkeypatch) -> None:
    expanded = mem_with_stub.save(content="expanded body", title="Expanded source")
    synth = mem_with_stub.save(
        content="synthesis body",
        title="Synthesis",
        type_="synthesis",
        extra={"synthesis_sources": [expanded.id]},
    )

    monkeypatch.setenv("MEMO_ASK_EXPAND_SYNTHESIS", "1")
    monkeypatch.setattr(mem_with_stub, "search", lambda *args, **kwargs: [mem_with_stub.get(synth.id)])

    _, sources, user_msg, _ = mem_with_stub._build_ask_context(
        "alpha?",
        k=2,
        type_=None,
        snippet_chars=120,
        include_repos=False,
        use_context_pack=True,
    )

    expanded_source = next(source for source in sources if source.get("expanded_from") == synth.id)
    assert expanded.id[:8] in user_msg
    assert expanded_source["quality_bucket"] == "current"
    assert isinstance(expanded_source["quality_reasons"], list)


def test_context_pack_prevents_sensitive_top_hit_verbatim_bypass(
    mem_with_stub, monkeypatch
) -> None:
    sensitive = mem_with_stub.save(content="secret literal leak", title="Secret", type_="secret")
    safe = mem_with_stub.save(content="safe fallback context", title="Safe")
    captured: dict[str, str] = {}

    def _stub_chat(self, model, messages, options=None):
        captured["user"] = messages[-1]["content"]
        return {"message": {"content": "Filtered answer."}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    monkeypatch.setattr(
        mem_with_stub,
        "search",
        lambda *args, **kwargs: [mem_with_stub.get(sensitive.id), mem_with_stub.get(safe.id)],
    )

    out = mem_with_stub.ask("literal leak", k=2, include_repos=False)

    assert out["answer"] == "Filtered answer."
    assert sensitive.id not in {source["id"] for source in out["sources"]}
    assert sensitive.id[:8] not in captured["user"]
    assert "secret literal leak" not in captured["user"]
    assert safe.id in {source["id"] for source in out["sources"]}


def test_sensitive_top_hit_verbatim_short_circuit_is_preserved_without_context_pack(
    mem_with_stub, monkeypatch
) -> None:
    sensitive = mem_with_stub.save(content="secret literal leak", title="Secret", type_="secret")
    safe = mem_with_stub.save(content="safe fallback context", title="Safe")

    monkeypatch.setenv("MEMO_CONTEXT_PACK", "0")
    monkeypatch.setattr(
        mem_with_stub,
        "search",
        lambda *args, **kwargs: [mem_with_stub.get(sensitive.id), mem_with_stub.get(safe.id)],
    )

    out = mem_with_stub.ask("literal leak", k=2, include_repos=False)

    assert out["answer"] == f"secret literal leak\n\n[{sensitive.id[:8]}]"


def test_chat_ask_keeps_standard_prompt_when_context_pack_flag_is_on(
    mem_with_stub, monkeypatch
) -> None:
    rec = mem_with_stub.save(content="alpha body", title="Alpha")
    captured: dict[str, str] = {}

    def _stub_chat(self, model, messages, options=None):
        captured["user"] = messages[-1]["content"]
        return {"message": {"content": f"Chat answer [{rec.id[:8]}]."}}

    monkeypatch.setattr("memo.llm.MLXChat.chat", _stub_chat)
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")

    out = mem_with_stub.chat_ask("what about alpha?", k=1)

    assert out["answer"].startswith("Chat answer")
    assert "Relevant context pack" not in captured["user"]


def test_ask_stream_uses_context_pack_only_when_enabled(mem_with_stub, monkeypatch) -> None:
    rec = mem_with_stub.save(content="alpha body", title="Alpha")
    prompts: list[str] = []

    monkeypatch.setattr(
        "memo.llm.MLXChat.chat_stream",
        _capturing_stream(prompts, ["streamed answer"]),
    )

    monkeypatch.setenv("MEMO_CONTEXT_PACK", "0")
    off_events = list(mem_with_stub.ask_stream("what about alpha?", k=1, include_repos=False))
    off_sources = off_events[0]["sources"]
    assert "Relevant context pack" not in prompts[-1]
    assert "quality_bucket" not in off_sources[0]

    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")
    on_events = list(mem_with_stub.ask_stream("what about alpha?", k=1, include_repos=False))
    on_sources = on_events[0]["sources"]
    assert "Relevant context pack" in prompts[-1]
    assert "Context summary:" in prompts[-1]
    assert on_sources[0]["quality_bucket"] == "current"
    assert on_events[-1]["answer"] == "streamed answer"
    assert any(rec.id == source["id"] for source in on_sources)


def test_chat_ask_stream_keeps_standard_prompt_when_context_pack_flag_is_on(
    mem_with_stub, monkeypatch
) -> None:
    mem_with_stub.save(content="alpha body", title="Alpha")
    prompts: list[str] = []

    monkeypatch.setattr(
        "memo.llm.MLXChat.chat_stream",
        _capturing_stream(prompts, ["chat stream answer"]),
    )
    monkeypatch.setenv("MEMO_CONTEXT_PACK", "1")

    events = list(mem_with_stub.chat_ask_stream("what about alpha?", k=1))

    assert events[-1]["answer"] == "chat stream answer"
    assert "Relevant context pack" not in prompts[-1]
