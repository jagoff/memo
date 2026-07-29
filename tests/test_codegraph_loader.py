"""Tests for the codegraph code-graph fallback loader."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from memo import codegraph_loader


def _seed_db(db_path: Path, names: tuple[str, str, str] = ("Alpha", "Beta", "Gamma")) -> None:
    """Create a minimal codegraph.db with three symbols A→B→C (calls)."""
    a, b, c = names
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT, name TEXT);
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO edges (source, target, kind) VALUES
            ('function:a', 'function:b', 'calls'),
            ('function:b', 'function:c', 'calls'),
            ('function:a', 'function:c', 'contains');
        """
    )
    conn.executemany(
        "INSERT INTO nodes (id, kind, name) VALUES (?, ?, ?)",
        [
            ("function:a", "function", a),
            ("function:b", "function", b),
            ("function:c", "function", c),
        ],
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
    assert len(adjacency) == 3


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


def test_discovery_finds_nearest_db_walking_up(monkeypatch, tmp_path: Path) -> None:
    project = tmp_path / "project"
    db = project / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    subdir = project / "src" / "pkg"
    subdir.mkdir(parents=True)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "1")
    monkeypatch.chdir(subdir)
    # The module-level fallback points nowhere: only cwd discovery can find the DB.
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing.db")
    codegraph_loader.reset()

    adjacency, _ = codegraph_loader.load()

    assert adjacency["alpha"] == {"beta"}
    # is_stale() reflects the same discovered DB, not the missing module fallback.
    assert codegraph_loader.is_stale() is False


def test_discovery_off_falls_back_to_module_db(monkeypatch, tmp_path: Path) -> None:
    cwd_db = tmp_path / "project" / ".codegraph" / "codegraph.db"
    cwd_db.parent.mkdir(parents=True)
    _seed_db(cwd_db)
    module_db = tmp_path / "module" / ".codegraph" / "codegraph.db"
    module_db.parent.mkdir(parents=True)
    _seed_db(module_db, names=("Delta", "Epsilon", "Zeta"))
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.chdir(cwd_db.parent.parent)  # cwd has a discoverable DB — must be ignored
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", module_db)
    codegraph_loader.reset()

    adjacency, _ = codegraph_loader.load()

    assert "delta" in adjacency
    assert "alpha" not in adjacency


def test_explicit_db_path_wins_over_fallback(monkeypatch, tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.db"
    _seed_db(explicit, names=("Delta", "Epsilon", "Zeta"))
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing.db")
    codegraph_loader.reset()

    adjacency, _ = codegraph_loader.load(db_path=explicit)

    assert "delta" in adjacency


def test_db_override_rescues_missing_fallback(monkeypatch, tmp_path: Path) -> None:
    """MEMO_CODEGRAPH_DB pins an index when cwd discovery finds nothing and the
    module default is dead (launchd daemon at $HOME, pipx install)."""
    override = tmp_path / "pinned" / ".codegraph" / "codegraph.db"
    override.parent.mkdir(parents=True)
    _seed_db(override, names=("Delta", "Epsilon", "Zeta"))
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)  # discovery on, but nothing to discover here
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "1")
    monkeypatch.setenv("MEMO_CODEGRAPH_DB", str(override))
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing.db")
    codegraph_loader.reset()

    adjacency, _ = codegraph_loader.load()

    assert "delta" in adjacency
    assert codegraph_loader.is_stale() is False


def test_discovery_wins_over_db_override(monkeypatch, tmp_path: Path) -> None:
    """A discoverable project index beats the MEMO_CODEGRAPH_DB pin — the
    override rescues cwd-less processes, it never breaks project-awareness."""
    project_db = tmp_path / "project" / ".codegraph" / "codegraph.db"
    project_db.parent.mkdir(parents=True)
    _seed_db(project_db)
    override = tmp_path / "override.db"
    _seed_db(override, names=("Delta", "Epsilon", "Zeta"))
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "1")
    monkeypatch.setenv("MEMO_CODEGRAPH_DB", str(override))
    monkeypatch.chdir(project_db.parent.parent)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing.db")
    codegraph_loader.reset()

    adjacency, _ = codegraph_loader.load()

    assert "alpha" in adjacency
    assert "delta" not in adjacency


def test_db_override_reads_markdown_config_when_env_unset(monkeypatch, tmp_path: Path) -> None:
    """With no MEMO_CODEGRAPH_DB export, the flags registry (Markdown config
    layer) supplies the pin — this is how launchd daemons, which inherit no
    shell env, get one."""
    override = tmp_path / "pinned.db"
    _seed_db(override, names=("Delta", "Epsilon", "Zeta"))
    config_home = tmp_path / "memo-config"
    (config_home / "config").mkdir(parents=True)
    (config_home / "config" / "advanced-config.md").write_text(
        '# Advanced config\n\n```toml\n[misc]\ncodegraph_db = "' + str(override) + '"\n```\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(config_home))
    monkeypatch.delenv("MEMO_CODEGRAPH_DB", raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing.db")
    codegraph_loader.reset()

    adjacency, _ = codegraph_loader.load()

    assert "delta" in adjacency


def _add_delta_edge(db: Path) -> None:
    """Append a Delta node called by Alpha, so a reload is observable."""
    conn = sqlite3.connect(db)
    conn.execute("INSERT INTO nodes (id, kind, name) VALUES ('function:d', 'function', 'Delta')")
    conn.execute(
        "INSERT INTO edges (source, target, kind) VALUES ('function:a', 'function:d', 'calls')"
    )
    conn.commit()
    conn.close()


def test_reload_when_mtime_advances(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    codegraph_loader.reset()
    adjacency, _ = codegraph_loader.load()
    assert "delta" not in adjacency

    stat = db.stat()
    _add_delta_edge(db)
    os.utime(db, (stat.st_atime, stat.st_mtime + 10))

    adjacency, _ = codegraph_loader.load()  # no force — the mtime advance triggers reload

    assert adjacency["delta"] == {"alpha"}


def test_cache_hit_when_mtime_unchanged(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    codegraph_loader.reset()
    first = codegraph_loader.load()

    stat = db.stat()
    _add_delta_edge(db)
    os.utime(db, (stat.st_atime, stat.st_mtime))  # pin mtime: cache must NOT reload

    assert codegraph_loader.load() is first


def test_edge_cap_exceeded_without_cache_raises(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)  # 2 traversable ('calls') edges
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_MAX_EDGES", "1")
    codegraph_loader.reset()

    with pytest.raises(RuntimeError, match="MEMO_CODEGRAPH_MAX_EDGES"):
        codegraph_loader.load()


def test_edge_cap_exceeded_serves_stale_cache(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    codegraph_loader.reset()
    first = codegraph_loader.load()

    stat = db.stat()
    _add_delta_edge(db)  # now 3 traversable edges
    os.utime(db, (stat.st_atime, stat.st_mtime + 10))  # mtime advanced: cache is stale
    monkeypatch.setenv("MEMO_CODEGRAPH_MAX_EDGES", "1")

    # Over the cap, the stale cached graph is served instead of failing.
    assert codegraph_loader.load() is first


def test_edge_cap_high_loads_normally(monkeypatch, tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_MAX_EDGES", "1000")
    codegraph_loader.reset()

    adjacency, _ = codegraph_loader.load()

    assert adjacency["alpha"] == {"beta"}
