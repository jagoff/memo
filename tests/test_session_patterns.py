"""Tests for session-pattern implementations: project detection, memory relations, MCP tools."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from memo.config import Config
from memo.server_session_patterns import (
    _normalize_hash,
    _project_from_cwd,
    _session_directory,
    register,
)


@pytest.fixture
def session_mem(tmp_path: Path):
    from memo.memory import Memory

    data = tmp_path / "data"
    vault = tmp_path / "vault"
    state = tmp_path / "state"
    data.mkdir()
    vault.mkdir()
    state.mkdir()

    import hashlib
    os.environ["MEMO_EMBEDDER_VIA_DAEMON"] = "0"

    cfg = Config(data_dir=data, vault_path=vault, state_dir=state, embedder_dims=4, reranker_enabled=False)
    mem = Memory(cfg)

    def stub(inputs):
        out = []
        for s in inputs:
            d = hashlib.sha256(s.encode()).digest()
            v = [((d[i] / 255.0) * 2 - 1) for i in range(4)]
            norm = (sum(x * x for x in v) ** 0.5) or 1
            v = [x / norm for x in v]
            out.append(v)
        return out

    mem.embedder.embed = stub
    mem.embedder.embed_query = lambda q: stub([q])[0]
    return mem


class _MockServer:
    """Captures tool registrations without a real MCP server."""
    def __init__(self):
        self._tools = {}

    def tool(self):
        def dec(f):
            self._tools[f.__name__] = f
            return f
        return dec


# ── 1. Project detection ─────────────────────────────────────────

class TestProjectDetection:
    def test_project_from_cwd_returns_string(self):
        p = _project_from_cwd()
        assert isinstance(p, str)
        assert len(p) > 0

    def test_session_directory_returns_cwd(self):
        d = _session_directory()
        assert d == os.getcwd()

    def test_memory_project_property(self, session_mem):
        p = session_mem.project
        assert isinstance(p, str)
        assert len(p) > 0


# ── 2. Tables exist in schema ────────────────────────────────────

class TestTablesCreated:
    def test_sessions_table(self, session_mem):
        r = session_mem.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='sessions'"
        ).fetchone()
        assert r is not None

    def test_memory_relations_table(self, session_mem):
        r = session_mem.store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memory_relations'"
        ).fetchone()
        assert r is not None


# ── 3. memory_relations ──────────────────────────────────────────

class TestMemoryRelations:
    def test_insert_and_query_relation(self, session_mem):
        r1 = session_mem.save(content="Use Postgres", title="DB choice", type_="decision")
        r2 = session_mem.save(content="Use SQLite", title="DB choice", type_="decision")

        cx = session_mem.store._conn
        cx.execute(
            "INSERT INTO memory_relations (id, source_id, target_id, relation, judgment_status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', datetime('now'))",
            ("rel-test-1", r1.id, r2.id, "conflicts_with"),
        )

        row = cx.execute("SELECT * FROM memory_relations WHERE id='rel-test-1'").fetchone()
        assert row is not None
        assert row["relation"] == "conflicts_with"
        assert row["source_id"] == r1.id
        assert row["target_id"] == r2.id

    def test_judge_relation(self, session_mem):
        """Mark a pending relation as judged via direct SQL."""
        r1 = session_mem.save(content="A", title="A", type_="decision")
        r2 = session_mem.save(content="B", title="B", type_="decision")
        cx = session_mem.store._conn
        cx.execute(
            "INSERT INTO memory_relations (id, source_id, target_id, relation, judgment_status, created_at) "
            "VALUES ('rel-judge-1', ?, ?, 'conflicts_with', 'pending', datetime('now'))",
            (r1.id, r2.id),
        )
        cx.execute(
            "UPDATE memory_relations SET judgment_status='judged', relation='supersedes', confidence=1.0 WHERE id='rel-judge-1'"
        )
        row = cx.execute("SELECT * FROM memory_relations WHERE id='rel-judge-1'").fetchone()
        assert row["judgment_status"] == "judged"
        assert row["relation"] == "supersedes"


# ── 4. MCP tools ────────────────────────────────────────────────

class TestMCPSessionTools:
    def test_all_tools_registered(self):
        mock = _MockServer()
        register(mock, None)  # none because tools don't inspect memory during registration
        expected = {
            "mem_session_start", "mem_session_end", "mem_context", "mem_timeline",
            "mem_judge", "mem_compare", "mem_suggest_topic_key", "mem_current_project",
            "mem_review", "mem_doctor", "mem_stats",
        }
        registered = set(mock._tools.keys())
        missing = expected - registered
        assert not missing, f"Missing tools: {missing}"

    def test_mem_current_project_run(self, session_mem):
        mock = _MockServer()
        register(mock, session_mem)
        res = mock._tools["mem_current_project"]()
        assert "project" in res
        assert "project_source" in res
        assert "project_path" in res
        assert res["project_source"] in {"config", "git_remote", "git_root", "git_child", "dir_basename"}


# ── 5. Helper functions ──────────────────────────────────────────

class TestHelperFunctions:
    def test_normalize_hash_deterministic(self):
        h1 = _normalize_hash("Test Title", "architecture", "project")
        h2 = _normalize_hash("Test Title", "architecture", "project")
        h3 = _normalize_hash("Other", "bugfix", "personal")
        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 16
