"""Tests for `memo code-facts` — architectural fact mining from codegraph.

All tests run against a synthetic codegraph.db under tmp_path (never the
real `.codegraph/codegraph.db`) and an isolated `Config` via `tmp_cfg`.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.code_traceability import _explicit_references
from memo.config import Config
from memo.memory import Memory

_STUB_DIMS = 8

# The seed below yields 5 facts: 2 call hubs (store_write ×2 calls,
# save_memory ×1), 1 decorated API surface entry (save_cmd), and 2
# cross-directory package-dependency pairs (memory→store ×2, cli→memory ×1).
_EXPECTED_FACTS = 5


def _seed_codegraph(db_path: Path) -> None:
    """Synthetic codegraph.db: call hubs, one decorated command, cross-dir deps."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            decorators TEXT
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes
            (id, kind, name, qualified_name, file_path, start_line, end_line, decorators)
        VALUES
            ('fn:store_write', 'function', 'store_write', 'store.writer.store_write',
             'src/memo/store/writer.py', 10, 40, NULL),
            ('fn:save_memory', 'function', 'save_memory', 'memory.save.save_memory',
             'src/memo/memory/save.py', 5, 60, NULL),
            ('fn:update_memory', 'function', 'update_memory', 'memory.update.update_memory',
             'src/memo/memory/update.py', 5, 50, NULL),
            ('fn:save_cmd', 'function', 'save_cmd', 'cli_save.save_cmd',
             'src/memo/cli_save.py', 8, 40, '["click.command"]');
        INSERT INTO edges (source, target, kind) VALUES
            ('fn:save_memory', 'fn:store_write', 'calls'),
            ('fn:update_memory', 'fn:store_write', 'calls'),
            ('fn:save_cmd', 'fn:save_memory', 'calls');
        """
    )
    conn.commit()
    conn.close()


def _seed_codegraph_no_decorators(db_path: Path) -> None:
    """Synthetic codegraph.db with no decorated functions (fallback branch).

    `cli_save.py` has two top-level symbols plus one nested helper that the
    top-level count must exclude.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            decorators TEXT
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes
            (id, kind, name, qualified_name, file_path, start_line, end_line, decorators)
        VALUES
            ('fn:save_cmd', 'function', 'save_cmd', 'cli_save.save_cmd',
             'src/memo/cli_save.py', 8, 40, NULL),
            ('fn:nested', 'function', 'nested', 'cli_save.save_cmd.nested',
             'src/memo/cli_save.py', 12, 20, NULL),
            ('fn:other_cmd', 'function', 'other_cmd', 'cli_save.other_cmd',
             'src/memo/cli_save.py', 50, 70, NULL);
        """
    )
    conn.commit()
    conn.close()


def _seed_codegraph_with_test_noise(db_path: Path) -> None:
    """Synthetic codegraph.db where tests/ nodes would out-rank src/ ones.

    `tests/test_cli_x.py` matches the `%cli_%.py` fallback pattern and has
    more top-level symbols than `src/memo/cli_save.py`; its `_helper` hub
    receives more call edges than the src hub. None of it may become a fact.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            start_line INTEGER,
            end_line INTEGER,
            decorators TEXT
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes
            (id, kind, name, qualified_name, file_path, start_line, end_line, decorators)
        VALUES
            ('fn:store_write', 'function', 'store_write', 'store.writer.store_write',
             'src/memo/store/writer.py', 10, 40, NULL),
            ('fn:save_cmd', 'function', 'save_cmd', 'cli_save.save_cmd',
             'src/memo/cli_save.py', 8, 40, NULL),
            ('fn:test_helper', 'function', '_helper', 'test_cli_x._helper',
             'tests/test_cli_x.py', 5, 20, '["click.command"]'),
            ('fn:test_a', 'function', 'test_a', 'test_cli_x.test_a',
             'tests/test_cli_x.py', 30, 40, NULL),
            ('fn:test_b', 'function', 'test_b', 'test_cli_x.test_b',
             'tests/test_cli_x.py', 50, 60, NULL);
        INSERT INTO edges (source, target, kind) VALUES
            ('fn:save_cmd', 'fn:store_write', 'calls'),
            ('fn:test_a', 'fn:test_helper', 'calls'),
            ('fn:test_b', 'fn:test_helper', 'calls');
        """
    )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def _stub_embedder(monkeypatch):
    """Deterministic per-text embeddings so `--apply` saves never load MLX."""

    def _embed(self, inputs):
        out = []
        for text in inputs:
            digest = hashlib.sha256((text or "").encode("utf-8")).digest()
            values = [((digest[i] / 255.0) * 2.0) - 1.0 for i in range(_STUB_DIMS)]
            norm = sum(v * v for v in values) ** 0.5
            out.append([v / norm for v in values])
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _embed)


def _env(cfg: Config) -> dict[str, str]:
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        # Match the stub embedder's output dims (Config's dim assertion is
        # env-driven even when MLXEmbedder.embed is monkeypatched).
        "MEMO_EMBEDDER_DIMS": str(_STUB_DIMS),
        "MEMO_EMBEDDER_MODEL": "stub",
        # Vec-similarity save dedup would misfire on hash-stub embeddings.
        "MEMO_SAVE_DEDUP_CHECK": "0",
        # Keep tags deterministic regardless of the cwd pytest runs from.
        "MEMO_AUTO_PROJECT_TAG": "0",
    }


def _read_memory(cfg: Config) -> Memory:
    return Memory(
        Config(
            data_dir=cfg.data_dir,
            state_dir=cfg.state_dir,
            embedder_dims=_STUB_DIMS,
            reranker_enabled=False,
        )
    )


def test_dry_run_is_default_and_writes_nothing(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_codegraph(db)

    res = CliRunner().invoke(cli, ["code-facts", "--db", str(db)], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    assert "dry-run" in res.output
    # Nothing persisted: no memory .md files anywhere under the data dir.
    assert list(Path(tmp_cfg.data_dir).rglob("*.md")) == []


def test_apply_saves_fact_memories_with_tags_and_code_refs(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_codegraph(db)

    res = CliRunner().invoke(cli, ["code-facts", "--apply", "--db", str(db)], env=_env(tmp_cfg))
    assert res.exit_code == 0, res.output

    mem = _read_memory(tmp_cfg)
    try:
        records = mem.list(type_="fact", limit=100)
    finally:
        mem.close()

    assert len(records) == _EXPECTED_FACTS
    for rec in records:
        assert "codegraph-derived" in rec.tags
        assert "project:memo" in rec.tags
        phash = rec.extra.get("provenance_hash")
        assert isinstance(phash, str) and len(phash) == 16
        # code_refs must be in the exact shape code_traceability parses.
        refs = _explicit_references(rec.extra)
        assert refs, rec.extra
        for ref in refs:
            assert ref.uri.startswith("codegraph://")
            assert ref.file_path
            assert ref.start_line is not None

    bodies = [rec.body.strip() for rec in records]
    assert any(
        b.startswith("Code hub: store.writer.store_write") and "receives 2 call edges" in b
        for b in bodies
    ), bodies
    assert any("cli_save.save_cmd" in b for b in bodies), bodies
    assert any("src/memo/memory -> src/memo/store" in b and "(2 edges)" in b for b in bodies), (
        bodies
    )


def test_second_apply_run_dedups_on_provenance_hash(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_codegraph(db)
    env = _env(tmp_cfg)
    runner = CliRunner()

    first = runner.invoke(cli, ["code-facts", "--apply", "--db", str(db)], env=env)
    assert first.exit_code == 0, first.output

    second = runner.invoke(cli, ["code-facts", "--apply", "--db", str(db), "--json"], env=env)
    assert second.exit_code == 0, second.output
    data = json.loads(second.output)
    assert data["saved"] == 0
    assert data["skipped"] == _EXPECTED_FACTS
    assert all(fact["status"] == "skipped" for fact in data["facts"])

    mem = _read_memory(tmp_cfg)
    try:
        assert len(mem.list(type_="fact", limit=100)) == _EXPECTED_FACTS
    finally:
        mem.close()


def test_json_output_shape_is_stable(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_codegraph(db)

    res = CliRunner().invoke(cli, ["code-facts", "--db", str(db), "--json"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    assert set(data) == {"dry_run", "db", "facts", "saved", "skipped"}
    assert data["dry_run"] is True
    assert data["saved"] == 0
    assert data["skipped"] == 0
    assert len(data["facts"]) == _EXPECTED_FACTS
    for fact in data["facts"]:
        assert set(fact) == {"category", "text", "provenance_hash", "status", "code_refs"}
        assert fact["status"] == "new"
        assert fact["code_refs"], fact
    assert {fact["category"] for fact in data["facts"]} == {
        "call-hub",
        "api-surface",
        "package-dependency",
    }


def test_top_flag_caps_call_hubs(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_codegraph(db)

    res = CliRunner().invoke(
        cli, ["code-facts", "--db", str(db), "--top", "1", "--json"], env=_env(tmp_cfg)
    )

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    hubs = [fact for fact in data["facts"] if fact["category"] == "call-hub"]
    assert len(hubs) == 1
    # Highest in-degree wins the single slot.
    assert "store.writer.store_write" in hubs[0]["text"]


def test_api_surface_falls_back_to_cli_module_top_level_counts(
    tmp_path: Path, tmp_cfg: Config
) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_codegraph_no_decorators(db)

    res = CliRunner().invoke(cli, ["code-facts", "--db", str(db), "--json"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    surface = [fact for fact in data["facts"] if fact["category"] == "api-surface"]
    assert len(surface) == 1
    # 2 top-level symbols; the nested helper is excluded.
    assert "src/memo/cli_save.py" in surface[0]["text"]
    assert "2 top-level symbols" in surface[0]["text"]


def test_tests_files_never_rank_in_surface_or_hubs(tmp_path: Path, tmp_cfg: Config) -> None:
    db = tmp_path / "cg" / ".codegraph" / "codegraph.db"
    _seed_codegraph_with_test_noise(db)

    res = CliRunner().invoke(cli, ["code-facts", "--db", str(db), "--json"], env=_env(tmp_cfg))

    assert res.exit_code == 0, res.output
    data = json.loads(res.output)
    # No fact of any category may be mined from tests/ files.
    for fact in data["facts"]:
        assert "tests/" not in fact["text"], fact
        for ref in fact["code_refs"]:
            assert not ref["file_path"].startswith("tests/"), fact
    # The decorated tests/ node is excluded, so surface falls back to the
    # src cli_*.py count — despite tests/test_cli_x.py having more symbols.
    surface = [fact for fact in data["facts"] if fact["category"] == "api-surface"]
    assert len(surface) == 1
    assert "src/memo/cli_save.py" in surface[0]["text"]
    assert "1 top-level symbols" in surface[0]["text"]
    # The tests/ helper (2 call edges) must not out-rank the src hub (1 edge).
    hubs = [fact for fact in data["facts"] if fact["category"] == "call-hub"]
    assert len(hubs) == 1
    assert "store.writer.store_write" in hubs[0]["text"]


def test_missing_db_errors_cleanly(tmp_path: Path, tmp_cfg: Config) -> None:
    res = CliRunner().invoke(
        cli, ["code-facts", "--db", str(tmp_path / "nope.db")], env=_env(tmp_cfg)
    )
    assert res.exit_code != 0
    assert "codegraph index not found" in res.output
