from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import memo.dashboard_logs as dashboard_logs
from memo.memory import MemoryRecord
from memo.recall_server import (
    RECALL_DIRECTIVE,
    _apply_preference_boost,
    _apply_project_boost,
    _recall_logic,
    dedup_hits,
)


def _rec(id_: str, title: str, score: float, tags: list[str] | None = None) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"notes/{id_}.md",
        title=title,
        type="note",
        tags=tags or [],
        created="2026-05-21T00:00:00+00:00",
        updated="2026-05-21T00:00:00+00:00",
        body="body " * 20,
        extra={},
        score=score,
    )


def test_apply_project_boost_copies_frozen_records_and_resorts() -> None:
    global_hit = _rec("global01", "Global", 0.70)
    project_hit = _rec("project1", "Project", 0.60, ["project:memo"])

    boosted = _apply_project_boost([global_hit, project_hit], "project:memo", 0.15)

    assert [h.id for h in boosted] == ["project1", "global01"]
    assert boosted[0].score == pytest.approx(0.75)
    assert project_hit.score == pytest.approx(0.60)


def test_recall_logic_project_boost_handles_frozen_records(monkeypatch, tmp_path) -> None:
    global_hit = _rec("global01", "Global", 0.70)
    project_hit = _rec("project1", "Project", 0.60, ["project:memo"])

    class StubMemory:
        def search(
            self,
            query: str,
            limit: int,
            mode: str,
            recency: bool = False,
            exclude_types=None,
            exclude_tags=None,
        ) -> list[MemoryRecord]:
            return [global_hit, project_hit]

    monkeypatch.setenv("MEMO_PROJECT_TAG", "memo")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    result, _log = _recall_logic(
        "project-specific query",
        cwd=str(tmp_path),
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )

    payload = json.loads(result)
    context = payload["hookSpecificOutput"]["additionalContext"]

    assert context.index("Project") < context.index("Global")
    # 3-tier ranking: project hit 0.60+0.25 (tier-1), global hit 0.70+0.10 (tier-2)
    assert "score 0.85" in context


def test_dedup_hits_drops_duplicate_id_and_near_identical_content() -> None:
    a = _rec("id000001", "Decisión MLX", 0.80)
    a_dup_id = _rec("id000001", "Decisión MLX", 0.50)  # same id, lower score
    a_near = _rec("id000002", "Decisión MLX", 0.70)  # different id, same title+body
    b = _rec("id000003", "Otra cosa distinta", 0.65)

    out = dedup_hits([a, a_dup_id, a_near, b])

    ids = [h.id for h in out]
    assert ids == ["id000001", "id000003"]  # dup id + near-dup content collapsed


def test_recall_logic_emits_authority_directive(monkeypatch, tmp_path) -> None:
    hit = _rec("auth0001", "Some fact", 0.80)

    class StubMemory:
        def search(
            self,
            query: str,
            limit: int,
            mode: str,
            recency: bool = False,
            exclude_types=None,
            exclude_tags=None,
        ) -> list[MemoryRecord]:
            return [hit]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    result, _log = _recall_logic(
        "anything",
        cwd=None,
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]
    assert RECALL_DIRECTIVE in context
    assert "authoritative" in context.lower()


def test_recall_logic_emits_directive_only_on_first_turn(monkeypatch, tmp_path) -> None:
    hit = _rec("once0001", "One fact", 0.90)

    class StubMemory:
        def search(self, query, limit, mode, recency=False, exclude_types=None, exclude_tags=None):
            return [hit]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_DIRECTIVE_ONCE", "1")

    first, _ = _recall_logic(
        "anything",
        cwd=None,
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
        turn=1,
    )
    later, _ = _recall_logic(
        "anything",
        cwd=None,
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
        turn=2,
    )

    assert RECALL_DIRECTIVE in json.loads(first)["hookSpecificOutput"]["additionalContext"]
    assert RECALL_DIRECTIVE not in json.loads(later)["hookSpecificOutput"]["additionalContext"]


def test_recall_logic_caps_total_context_and_logs_exact_cost(monkeypatch, tmp_path) -> None:
    hit = _rec("cap00001", "Long fact", 0.90)
    object.__setattr__(hit, "body", "substantial context " * 200)

    class StubMemory:
        def search(self, query, limit, mode, recency=False, exclude_types=None, exclude_tags=None):
            return [hit]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_TOKEN_BUDGET", "160")
    monkeypatch.setenv("MEMO_RECALL_TOP_K", "1")
    monkeypatch.setenv("MEMO_RECALL_FEEDBACK_HINT", "0")

    result, log_result = _recall_logic(
        "anything",
        cwd=None,
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
        session_id="sid-cap",
        turn=1,
        client="claude-code",
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]

    assert "Long fact" in context
    # CITE_INSTRUCTION is budget-exempt (appended after capping); strip it before
    # asserting the budget so the cap is measured on the core content only.
    from memo.recall_logic import CITE_INSTRUCTION

    cite_suffix = f"\n{CITE_INSTRUCTION}"
    core_context = context.removesuffix(cite_suffix)
    assert len(core_context) <= 160 * 4
    assert log_result is not None
    log_result()
    costs = dashboard_logs.read_context_cost_log(tmp_path)
    assert costs[-1]["chars"] == len(context)
    assert costs[-1]["kind"] == "recall"


def test_recall_logic_passes_recency_to_search(monkeypatch, tmp_path) -> None:
    seen = {}

    class StubMemory:
        def search(
            self,
            query: str,
            limit: int,
            mode: str,
            recency: bool = False,
            exclude_types=None,
            exclude_tags=None,
        ) -> list[MemoryRecord]:
            seen["recency"] = recency
            return [_rec("r0000001", "Fresh", 0.9)]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    _recall_logic(
        "q", cwd=None, mem=StubMemory(), cfg=SimpleNamespace(state_dir=tmp_path), debug=False
    )
    assert seen["recency"] is True


def test_apply_preference_boost_reorders_by_learned_type() -> None:
    note = _rec("n0000001", "a note", 0.70)
    decision = _rec("d0000001", "a decision", 0.68)
    object.__setattr__(decision, "type", "decision")  # frozen record

    prefs = SimpleNamespace(preferred_types={"decision": 0.9})
    out = _apply_preference_boost([note, decision], prefs)

    # decision was behind on raw score but the learned type pref lifts it
    assert next(h.id for h in out) == "d0000001"
    # empty prefs → unchanged order
    same = _apply_preference_boost([note, decision], SimpleNamespace(preferred_types={}))
    assert [h.id for h in same] == ["n0000001", "d0000001"]


def test_recall_logic_records_what_surfaced(monkeypatch, tmp_path) -> None:
    recorded = {}

    class FakeContextual:
        class context:
            @staticmethod
            def get_preferences():
                return SimpleNamespace(preferred_types={})

        @staticmethod
        def record_search(prompt, ids):
            recorded["prompt"] = prompt
            recorded["ids"] = ids

    class StubMemory:
        contextual = FakeContextual()

        def search(
            self,
            query: str,
            limit: int,
            mode: str,
            recency: bool = False,
            exclude_types=None,
            exclude_tags=None,
        ) -> list[MemoryRecord]:
            return [_rec("surf0001", "surfaced", 0.9)]

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    _recall_logic(
        "mi pregunta",
        cwd=None,
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
    )
    assert recorded["prompt"] == "mi pregunta"
    assert recorded["ids"] == ["surf0001"]


def test_recall_logic_adds_related_nudge_below_the_cut(monkeypatch, tmp_path) -> None:
    hits = [_rec(f"id{i:07d}", f"hit {i}", 0.9 - i * 0.05) for i in range(5)]

    class StubMemory:
        def search(
            self,
            query: str,
            limit: int,
            mode: str,
            recency: bool = False,
            exclude_types=None,
            exclude_tags=None,
        ) -> list[MemoryRecord]:
            return hits

    monkeypatch.setenv("MEMO_RECALL_TOP_K", "3")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")  # isolate from prefs
    monkeypatch.setenv("MEMO_RECALL_TOKEN_BUDGET", "0")
    # These 5 synthetic "hit N" records tokenize near-identically; disable the
    # pre-top-K paraphrase collapse (default ON since v3.0.0) so the nudge-cut
    # under test sees all 5 rather than one collapsed survivor.
    monkeypatch.setenv("MEMO_RECALL_DEDUP_COLLAPSE", "0")

    result, _log = _recall_logic(
        "q", cwd=None, mem=StubMemory(), cfg=SimpleNamespace(state_dir=tmp_path), debug=False
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]
    # top-3 in the main block, next 2 in the related nudge
    nudge_line = context.split("related):", 1)[1]
    assert "hit 3" in nudge_line and "hit 4" in nudge_line
    assert "hit 0" not in nudge_line  # top hits stay in the main block


class _StubMicroEmbedder:
    """Deterministic 2-dim embedder for the cold-embedder fallback path."""

    def __init__(self, query_vec: list[float], doc_vecs: list[list[float]]) -> None:
        self._query_vec = query_vec
        self._doc_vecs = doc_vecs

    def embed_query(self, text: str) -> list[float]:
        return self._query_vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._doc_vecs[: len(texts)]


def test_fallback_scoring_does_not_mutate_shared_hits(monkeypatch, tmp_path) -> None:
    """The micro-embedder fallback must score into NEW records, never mutate
    the shared frozen hits returned by `search`."""

    def _body_rec(id_: str, title: str) -> MemoryRecord:
        # distinct bodies so dedup doesn't collapse them as near-identical
        return MemoryRecord(
            id=id_,
            path=f"notes/{id_}.md",
            title=title,
            type="note",
            tags=[],
            created="2026-05-21T00:00:00+00:00",
            updated="2026-05-21T00:00:00+00:00",
            body=f"unique body for {title} " * 8,
            extra={},
            score=0.0,
        )

    a = _body_rec("aaaa0001", "Alpha")
    b = _body_rec("bbbb0001", "Beta")
    candidates = [a, b]

    class StubMemory:
        embedder = SimpleNamespace(is_warm=False)
        # embedder_dims ≤ 10 makes recall skip its dim-validation guard for the
        # 2-dim stub vectors (the guard reads mem.cfg.embedder_dims).
        cfg = SimpleNamespace(embedder_dims=2)

        def search(self, query, limit, mode, recency=False, exclude_types=None, exclude_tags=None):
            return candidates

        def _read_body(self, path):
            return "body"

    # query closer to Beta (doc index 1) than Alpha → Beta must rank first
    micro = _StubMicroEmbedder(query_vec=[0.0, 1.0], doc_vecs=[[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "0")
    # neutralise dev-env trimming so both hits surface deterministically
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")

    result, _log = _recall_logic(
        "q",
        cwd=None,
        mem=StubMemory(),
        cfg=SimpleNamespace(state_dir=tmp_path),
        debug=False,
        micro_embedder=micro,
    )
    context = json.loads(result)["hookSpecificOutput"]["additionalContext"]
    assert context.index("Beta") < context.index("Alpha")  # rescored + resorted
    # the shared frozen hits keep their original (None-equivalent 0.0) score
    assert a.score == 0.0 and b.score == 0.0
    # original list order is untouched (we sorted a NEW list)
    assert [h.id for h in candidates] == ["aaaa0001", "bbbb0001"]


def test_project_tag_failure_is_logged_not_silent(monkeypatch, tmp_path, caplog) -> None:
    """A failing project_tag resolution must be swallowed but observable."""
    import memo.recall_logic as rl

    def _boom(cwd):
        raise RuntimeError("project resolution blew up")

    monkeypatch.setattr("memo.project.current_project_tag", _boom)
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.0")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")

    class StubMemory:
        def search(self, query, limit, mode, recency=False, exclude_types=None, exclude_tags=None):
            return [_rec("ok000001", "Surfaced", 0.9)]

    with caplog.at_level("DEBUG", logger=rl._logger.name):
        result, _log = _recall_logic(
            "q",
            cwd=str(tmp_path),
            mem=StubMemory(),
            cfg=SimpleNamespace(state_dir=tmp_path),
            debug=False,
        )
    # control flow preserved: recall still succeeds
    assert "Surfaced" in json.loads(result)["hookSpecificOutput"]["additionalContext"]
    # but the failure is now observable
    assert any("project_tag resolution failed" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Compact format tests
# ---------------------------------------------------------------------------

from memo.recall_logic import render_recall_compact  # noqa: E402


def _rich_rec(id_: str, title: str, score: float, body: str = "") -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"notes/{id_}.md",
        title=title,
        type="note",
        tags=["tag1"],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body=body or ("substantial body text for " * 5),
        extra={},
        score=score,
    )


def test_compact_format_one_line_per_hit_no_headers_tags_scores() -> None:
    hits = [
        _rich_rec("aabbccdd11223344", "First fact", 0.90),
        _rich_rec("eeff001122334455", "Second fact", 0.80),
        _rich_rec("ffee998877665544", "Third fact", 0.70),
    ]
    block = render_recall_compact(hits, token_budget=0)

    assert block.startswith("<memo-recall readonly>\n")
    assert block.endswith("\n</memo-recall>")

    inner_lines = block.split("\n")[1:-1]
    assert len(inner_lines) == 3

    for hit, line in zip(hits, inner_lines, strict=True):
        assert line.startswith(f"[{hit.id[:8]}]"), f"line {line!r} should start with id8"
        assert "score" not in line.lower()
        assert "tag1" not in line
        assert "##" not in line
        assert "_Saved" not in line


def test_compact_format_respects_token_budget() -> None:
    # Each line is ~40 chars; token_budget=1 means max_chars=4 → only fits wrapper
    # token_budget=4 means max_chars=16 → still too small for any hit line
    # Use a budget large enough for 2 but not 3 hits
    hits = [
        _rich_rec("aaaa000011112222", "Hit alpha", 0.90, "body alpha"),
        _rich_rec("bbbb000022223333", "Hit beta", 0.80, "body beta"),
        _rich_rec("cccc000033334444", "Hit gamma", 0.70, "body gamma"),
    ]
    # Build a budget that accommodates the first 2 but not the 3rd
    two_hit_block = render_recall_compact(hits[:2], token_budget=0)
    # Ceiling division so max_chars = budget_tokens * 4 >= len(two_hit_block)
    budget_tokens = (len(two_hit_block) + 3) // 4

    block = render_recall_compact(hits, token_budget=budget_tokens)
    inner_lines = block.split("\n")[1:-1]

    assert len(inner_lines) == 2
    assert "[aaaa0000]" in block
    assert "[bbbb0000]" in block
    assert "[cccc0000]" not in block


def test_compact_format_full_unchanged_regression(monkeypatch, tmp_path) -> None:
    """render_recall_context (full format) still emits ## Memory header and directive."""
    from memo.recall_logic import RECALL_DIRECTIVE, RECALL_HEADER, render_recall_context

    hits = [_rich_rec("reg00001aabbccdd", "Regression fact", 0.85)]
    monkeypatch.setenv("MEMO_RECALL_FEEDBACK_HINT", "0")
    monkeypatch.setenv("MEMO_RECALL_DIRECTIVE_ONCE", "0")

    block = render_recall_context(hits, [], turn=1, body_chars=400, token_budget=0)

    assert RECALL_HEADER in block
    assert RECALL_DIRECTIVE in block
    assert "## Memory" in block
    assert "score" in block.lower()


def test_compact_format_is_much_smaller_than_full() -> None:
    # Use long bodies so the full block is large; compact truncates to 60 chars.
    long_body = "This is a very detailed explanation of the fact in question. " * 5
    hits = [
        _rich_rec("c1111111aaaabbbb", "Compact fact one", 0.90, long_body),
        _rich_rec("c2222222ccccdddd", "Compact fact two", 0.85, long_body),
        _rich_rec("c3333333eeeeffff", "Compact fact three", 0.80, long_body),
    ]
    from memo.recall_logic import render_recall_context

    full_block = render_recall_context(hits, [], turn=None, body_chars=400, token_budget=0)
    compact_block = render_recall_compact(hits, token_budget=0)

    ratio = len(compact_block) / len(full_block)
    assert ratio <= 0.30, (
        f"compact ({len(compact_block)} chars) should be ≤30% of full ({len(full_block)} chars), "
        f"got {ratio:.0%}"
    )


# ---------------------------------------------------------------------------
# Trivial prompt gate (MEMO_RECALL_TRIVIAL_BAIL)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402

from click.testing import CliRunner  # noqa: E402


def _trivial_env(tmp_path: Path) -> dict[str, str]:
    """Env that bypasses the char-length gate and enables debug bail output."""
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_RECALL_DEBUG": "1",
        "MEMO_RECALL_MIN_PROMPT_CHARS": "1",
    }


def _invoke_hook(prompt: str, env: dict) -> CliRunner:  # type: ignore[type-arg]
    from memo.cli import cli

    runner = CliRunner()
    payload = json.dumps({"prompt": prompt})
    return runner.invoke(cli, ["recall-hook"], input=payload, env=env)


def test_trivial_bail_fires_on_si(tmp_path: Path) -> None:
    result = _invoke_hook("sí", _trivial_env(tmp_path))
    assert result.exit_code == 0
    assert "trivial prompt" in result.output


def test_trivial_bail_fires_on_yes_please(tmp_path: Path) -> None:
    result = _invoke_hook("yes please", _trivial_env(tmp_path))
    assert "trivial prompt" in result.output


def test_trivial_bail_skips_longer_prompt(tmp_path: Path) -> None:
    """A prompt >3 words is not trivial even when it starts with 'yes'."""
    result = _invoke_hook("yes, please implement the auth module", _trivial_env(tmp_path))
    assert "trivial prompt" not in result.output


def test_trivial_bail_fires_on_ok(tmp_path: Path) -> None:
    result = _invoke_hook("ok", _trivial_env(tmp_path))
    assert "trivial prompt" in result.output


def test_trivial_bail_disabled(tmp_path: Path) -> None:
    """MEMO_RECALL_TRIVIAL_BAIL=0 → 'ok' does not bail as trivial."""
    env = {**_trivial_env(tmp_path), "MEMO_RECALL_TRIVIAL_BAIL": "0"}
    result = _invoke_hook("ok", env)
    assert "trivial prompt" not in result.output
