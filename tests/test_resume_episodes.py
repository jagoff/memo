"""Tests for memo's episodic-memory layer (Phase 1) — the semantic index
behind `memo resume`. Covers the EpisodeStore, the indexer (prompt-arc + skip),
semantic search (+ cold degrade), the picker's semantic ordering, and the
`memo episodes index` CLI. No real MLX: embeddings are stubbed (dims pinned).
"""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.resume._types import ResumeCandidate

_DIMS = 4


def _unit(i: int) -> list[float]:
    """A 4-dim one-hot unit vector (L2 norm 1), for deterministic cosine ranking."""
    v = [0.0] * _DIMS
    v[i] = 1.0
    return v


def _cand(
    sid: str, *, agent: str = "claude", cwd: str = "/repo", summary: str = ""
) -> ResumeCandidate:
    return ResumeCandidate(
        agent=agent,
        provider=f"{agent}-native",
        uri=f"{agent}://session/{sid}",
        session_id=sid,
        title=summary or sid,
        updated_at="2026-05-23T10:00:00Z",
        cwd=cwd,
        summary=summary,
        resume_command=["claude", "--resume", sid] if agent == "claude" else [agent, "resume", sid],
    )


# ── EpisodeStore ──────────────────────────────────────────────────────────────


def _store(tmp_path: Path):
    from memo.store.episode_store import EpisodeStore

    return EpisodeStore(tmp_path / "episodes.db", _DIMS)


def test_episode_store_upsert_search_and_skip(tmp_path: Path) -> None:
    with closing(_store(tmp_path)) as store:
        for i, sid in enumerate(("a", "b", "c")):
            store.upsert(
                agent="claude",
                session_id=sid,
                content_hash=f"h{i}",
                embedding=_unit(i),
                cwd="/repo",
                updated_at="2026-05-23T10:00:00Z",
                summary=f"session {sid}",
                resume_command=["claude", "--resume", sid],
                turn_count=i,
            )
        assert store.count() == 3
        assert store.content_hash_for("claude", "b") == "h1"
        assert store.content_hash_for("claude", "zzz") is None

        # Query closest to "b" (one-hot dim 1) ranks b first.
        rows = store.search(_unit(1), k=3)
        assert rows[0]["session_id"] == "b"
        assert rows[0]["resume_command"] == ["claude", "--resume", "b"]
        assert rows[0]["score"] > rows[1]["score"]

        store.clear()
        assert store.count() == 0


def test_episode_store_rejects_wrong_dims(tmp_path: Path) -> None:
    with closing(_store(tmp_path)) as store:
        with pytest.raises(ValueError, match="dim mismatch"):
            store.upsert(
                agent="claude",
                session_id="x",
                content_hash="h",
                embedding=[1.0, 0.0],
                cwd="/repo",
                updated_at="",
                summary="",
                resume_command=[],
                turn_count=0,
            )


# ── indexer: prompt_arc + index_candidate skip ────────────────────────────────


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_prompt_arc_gathers_summary_and_prompts(tmp_path: Path) -> None:
    from memo.resume._index import prompt_arc

    tp = tmp_path / "t.jsonl"
    _write_jsonl(
        tp,
        [
            {
                "type": "user",
                "message": {"content": [{"type": "text", "text": "first prompt about auth"}]},
            },
            {"type": "assistant", "message": {"content": [{"type": "text", "text": "ok"}]}},
            {
                "type": "user",
                "message": {
                    "content": [{"type": "text", "text": "second prompt about vec0 timeout"}]
                },
            },
        ],
    )
    cand = _cand("s1", summary="running summary text")
    cand.metadata["path"] = str(tp)
    arc = prompt_arc(cand)
    assert "running summary text" in arc
    assert "auth" in arc and "vec0 timeout" in arc


def test_index_candidate_skips_unchanged(tmp_path: Path) -> None:
    from memo.resume._index import index_candidate

    cand = _cand("s1", summary="some work on the resume picker")
    calls = {"n": 0}

    def fake_embed(texts):
        calls["n"] += 1
        return [_unit(0)]

    with closing(_store(tmp_path)) as store:
        assert index_candidate(store, cand, embed_fn=fake_embed) is True
        assert calls["n"] == 1
        # Same content → hash match → skip, no second embed.
        assert index_candidate(store, cand, embed_fn=fake_embed) is False
        assert calls["n"] == 1
        assert store.count() == 1


# ── semantic_search: warm path + cold degrade ────────────────────────────────


def _episodic_cfg(tmp_path: Path):
    from memo.config import Config

    # Override dims via from_env kwargs — never mutate the global env (would leak
    # MEMO_EMBEDDER_DIMS into sibling tests and corrupt their real-dims stores).
    return Config.from_env(
        state_dir=tmp_path / "state", data_dir=tmp_path / "data", embedder_dims=_DIMS
    )


def test_semantic_search_ranks_by_meaning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import memo.embedder_client as ec
    from memo.resume._index import open_store, semantic_search

    cfg = _episodic_cfg(tmp_path)
    store = open_store(cfg)
    assert store is not None and store.dims == _DIMS
    with closing(store):
        for i, sid in enumerate(("alpha", "beta", "gamma")):
            store.upsert(
                agent="claude",
                session_id=sid,
                content_hash=f"h{i}",
                embedding=_unit(i),
                cwd="/repo",
                updated_at="2026-05-23T10:00:00Z",
                summary=f"work {sid}",
                resume_command=["claude", "--resume", sid],
                turn_count=1,
            )
    # Warm daemon; query embeds to the "beta" one-hot.
    monkeypatch.setattr(ec, "ping", lambda **_kw: {"ok": True})
    monkeypatch.setattr(ec, "embed_query", lambda _q, **_kw: _unit(1))

    hits = semantic_search(cfg, "anything about beta")
    assert hits[0].session_id == "beta"
    assert hits[0].provider == "episode"
    assert hits[0].resume_command == ["claude", "--resume", "beta"]


def test_semantic_search_degrades_when_cold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.embedder_client as ec
    from memo.resume._index import open_store, semantic_search

    cfg = _episodic_cfg(tmp_path)
    store = open_store(cfg)
    assert store is not None
    with closing(store):
        store.upsert(
            agent="claude",
            session_id="x",
            content_hash="h",
            embedding=_unit(0),
            cwd="/repo",
            updated_at="",
            summary="x",
            resume_command=[],
            turn_count=0,
        )
    # Cold embedder → no semantic, caller stays on substring.
    monkeypatch.setattr(ec, "ping", lambda **_kw: None)
    assert semantic_search(cfg, "beta") == []


def test_semantic_search_empty_query_returns_nothing(tmp_path: Path) -> None:
    from memo.resume._index import semantic_search

    assert semantic_search(_episodic_cfg(tmp_path), "   ") == []


# ── picker semantic ordering (pure, no TTY) ──────────────────────────────────


def test_tui_semantic_ordering_and_hydration() -> None:
    from memo.resume._tui import _apply_semantic, _resume_tui_visible, _ResumeTuiState

    a, b = _cand("aaaa", cwd="/repo"), _cand("bbbb", cwd="/repo")
    state = _ResumeTuiState(candidates=[a, b], current_cwd="/repo", query="x", filter_mode="all")
    # Semantic hits rank b first, and surface an episode-only session "cccc".
    c = _cand("cccc", cwd="/repo", summary="old session beyond recency")
    _apply_semantic(state, "x", [b, c, a])

    visible = _resume_tui_visible(state)
    assert [v.session_id for v in visible] == ["bbbb", "cccc", "aaaa"]
    # The episode-only hit was merged into the candidate pool.
    assert any(v.session_id == "cccc" for v in state.candidates)


def test_tui_semantic_inactive_when_query_changes() -> None:
    from memo.resume._tui import _apply_semantic, _resume_tui_visible, _ResumeTuiState

    a, b = _cand("aaaa", cwd="/repo", summary="alpha"), _cand("bbbb", cwd="/repo", summary="beta")
    state = _ResumeTuiState(candidates=[a, b], current_cwd="/repo", query="beta", filter_mode="all")
    _apply_semantic(state, "old-query", [b, a])  # stale: searched a different query
    # Query != semantic_query ⇒ fall back to substring (matches "beta" → b only).
    visible = _resume_tui_visible(state)
    assert [v.session_id for v in visible] == ["bbbb"]


# ── CLI: memo episodes index ─────────────────────────────────────────────────


def test_cli_episodes_index_backfills(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import memo.embedder_client as ec
    from memo.cli import cli

    # A memo snapshot to index, no other agents.
    state_dir = tmp_path / "state"
    (state_dir / "sessions").mkdir(parents=True)
    (state_dir / "sessions" / "sess1.json").write_text(
        json.dumps(
            {
                "session_id": "sess1",
                "cwd": str(tmp_path / "repo"),
                "project": "repo",
                "running_summary": "worked on the episode index",
                "updated": "2026-05-23T10:00:00Z",
                "turn_count": 4,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ec, "embed", lambda _texts, **_kw: [_unit(0)])

    empty = tmp_path / "homes"
    empty.mkdir()
    env = {
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_EMBEDDER_DIMS": str(_DIMS),
        "MEMO_NONINTERACTIVE": "1",
        "CLAUDE_HOME": str(empty / "c"),
        "CODEX_HOME": str(empty / "co"),
        "DEVIN_HOME": str(empty / "d"),
        "GEMINI_HOME": str(empty / "g"),
        "OPENCODE_DATA": str(empty / "o"),
    }
    result = CliRunner().invoke(cli, ["episodes", "index", "--json"], env=env)
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["enabled"] is True
    assert payload["indexed"] == 1
    assert payload["total"] == 1


# ── Phase 2: episodes queryable (CLI search + MCP tool) ──────────────────────


def test_cli_episodes_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import memo.embedder_client as ec
    from memo.cli import cli
    from memo.store.episode_store import EpisodeStore

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    with closing(EpisodeStore(state_dir / "episodes.db", _DIMS)) as store:
        for i, sid in enumerate(("alpha", "beta")):
            store.upsert(
                agent="claude",
                session_id=sid,
                content_hash=f"h{i}",
                embedding=_unit(i),
                cwd="/repo",
                updated_at="2026-05-23T10:00:00Z",
                summary=f"work {sid}",
                resume_command=["claude", "--resume", sid],
                turn_count=1,
            )
    # allow_cold path skips ping; query embeds to the "beta" one-hot.
    monkeypatch.setattr(ec, "embed_query", lambda _q, **_kw: _unit(1))
    env = {
        "MEMO_STATE_DIR": str(state_dir),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_EMBEDDER_DIMS": str(_DIMS),
        "MEMO_NONINTERACTIVE": "1",
    }
    result = CliRunner().invoke(cli, ["episodes", "search", "beta", "--json"], env=env)
    assert result.exit_code == 0, result.output
    hits = json.loads(result.output)
    assert hits[0]["session_id"] == "beta"
    assert hits[0]["provider"] == "episode"


def test_mcp_episodes_search_tool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import memo.resume._index as idx
    from memo.config import Config
    from memo.memory import Memory
    from memo.server import build_server

    cfg = Config.from_env(
        state_dir=tmp_path / "state", data_dir=tmp_path / "data", embedder_dims=_DIMS
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed", lambda self, inputs: [_unit(0) for _ in inputs]
    )
    mem = Memory(cfg)
    try:
        hit = _cand("ep1", summary="worked on episodic memory")
        hit.metadata["score"] = 0.9
        # Tool delegates to semantic_search; stub it so the MCP surface is tested
        # without a live index/embedder.
        monkeypatch.setattr(idx, "semantic_search", lambda *_a, **_k: [hit])
        server = build_server(memory=mem)
        tool = asyncio.run(server.get_tool("memo_episodes_search")).fn
        out = tool(query="episodic memory", limit=5)
        assert out["query"] == "episodic memory"
        assert out["results"][0]["session_id"] == "ep1"
        assert out["results"][0]["score"] == 0.9
        assert out["results"][0]["resume_command"] == ["claude", "--resume", "ep1"]
    finally:
        mem.close()


# ── Phase 2: repo-delta + open-loops preview ─────────────────────────────────


def test_session_preview_repo_delta_and_open_loops(tmp_path: Path) -> None:
    import subprocess

    from memo.config import Config
    from memo.resume._preview import session_preview

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git(
        "-c",
        "user.email=t@t",
        "-c",
        "user.name=t",
        "commit",
        "--allow-empty",
        "-q",
        "-m",
        "first work",
    )
    (repo / "dirty.txt").write_text("wip", encoding="utf-8")  # uncommitted

    state_dir = tmp_path / "state"
    (state_dir / "sessions").mkdir(parents=True)
    # get_session needs an id ≥4 chars to resolve the snapshot.
    (state_dir / "sessions" / "sess-0001.json").write_text(
        json.dumps(
            {
                "session_id": "sess-0001",
                "cwd": str(repo),
                "prompt_trail": ["fix the bug", "add tests"],
            }
        ),
        encoding="utf-8",
    )
    cfg = Config.from_env(state_dir=state_dir, data_dir=tmp_path / "data", embedder_dims=_DIMS)
    cand = ResumeCandidate(
        agent="claude",
        provider="memo",
        uri="memo://session/sess-0001",
        session_id="sess-0001",
        title="work",
        updated_at="2020-01-01T00:00:00Z",  # before the commit → it shows under "since"
        cwd=str(repo),
        summary="work",
        resume_command=["claude", "--resume", "sess-0001"],
    )
    text = "\n".join(session_preview(cfg, cand))
    assert "commit(s) here since" in text and "first work" in text
    assert "uncommitted file" in text
    assert "Open loops" in text and "fix the bug" in text


def test_session_preview_non_git_cwd_is_empty(tmp_path: Path) -> None:
    from memo.config import Config
    from memo.resume._preview import session_preview

    cfg = Config.from_env(
        state_dir=tmp_path / "state", data_dir=tmp_path / "data", embedder_dims=_DIMS
    )
    cand = _cand("s1", cwd=str(tmp_path / "not-a-repo"))
    assert session_preview(cfg, cand) == []
