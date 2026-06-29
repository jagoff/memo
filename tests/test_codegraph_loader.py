"""Tests for the codegraph code-graph fallback loader."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memo import codegraph_loader


def _seed_db(db_path: Path) -> None:
    """Create a minimal codegraph.db with three symbols A→B→C (calls)."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT);
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes (id, kind, name) VALUES
            ('function:a', 'function', 'Alpha'),
            ('function:b', 'function', 'Beta'),
            ('function:c', 'function', 'Gamma');
        INSERT INTO edges (source, target, kind) VALUES
            ('function:a', 'function:b', 'calls'),
            ('function:b', 'function:c', 'calls'),
            ('function:a', 'function:c', 'contains');
        """
    )
    conn.commit()
    conn.close()


def test_load_builds_symbol_adjacency(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    codegraph_loader.reset()

    adjacency, edge_weights = codegraph_loader.load()

    # Names are lowercased; only symbol→symbol kinds are traversed (no 'contains').
    assert adjacency["alpha"] == {"beta"}
    assert adjacency["beta"] == {"alpha", "gamma"}
    assert ("alpha", "gamma") not in edge_weights
    assert codegraph_loader.node_count() == 3


def test_find_path_uses_codegraph(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    codegraph_loader.reset()

    assert codegraph_loader.find_path("Alpha", "Gamma") == ["alpha", "beta", "gamma"]
    assert codegraph_loader.find_path("Alpha", "nope") is None


def test_refresh_is_noop_and_is_stale_only_when_missing(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    codegraph_loader.reset()

    # codegraph self-maintains: refresh never rebuilds, index is stale only if absent.
    assert codegraph_loader.refresh(force=True) is False
    assert codegraph_loader.is_stale() is False
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing.db")
    assert codegraph_loader.is_stale() is True
