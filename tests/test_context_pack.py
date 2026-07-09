from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memo.context_pack import build_context_pack
from memo.memory import Memory


@dataclass(frozen=True)
class _Hit:
    id: str
    score: float
    title: str
    body: str
    type: str = "note"
    tags: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


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
    from memo.repo_index import RepoSearchHit

    mem_with_stub.save(content="alpha body", title="Alpha")

    monkeypatch.setattr(
        type(mem_with_stub.store), "list_repo_sources", lambda self, **kw: [{"name": "code-repo"}]
    )
    monkeypatch.setattr(
        Memory,
        "repo_search",
        lambda self, q, **kw: [
            RepoSearchHit(
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
                text="def alpha(): pass",
                score=0.6,
                match_type="hybrid",
            )
        ],
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
