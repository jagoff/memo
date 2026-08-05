from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from pathlib import Path

from memo import code_evidence
from memo.code_evidence import codegraph_evidence, normalize_code_path
from memo.code_impact import code_change_impact
from memo.code_traceability import CodeReferenceResolver, codegraph_uri


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _seed_codegraph(repo: Path) -> Path:
    db = repo / ".codegraph" / "codegraph.db"
    db.parent.mkdir()
    source_a = (repo / "src" / "a.py").read_bytes()
    source_b = (repo / "src" / "b.py").read_bytes()
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE project_metadata (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE schema_versions (version INTEGER);
        CREATE TABLE files (
            path TEXT PRIMARY KEY,
            content_hash TEXT,
            errors TEXT
        );
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT,
            name TEXT,
            qualified_name TEXT,
            file_path TEXT,
            start_line INTEGER,
            end_line INTEGER
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO project_metadata VALUES
            ('index_state', 'complete'),
            ('files_discovered', '2'),
            ('files_accounted', '2'),
            ('indexed_with_version', 'test-1'),
            ('updated_at', '1785283200000');
        INSERT INTO schema_versions VALUES (8);
        INSERT INTO nodes VALUES
            ('function:a', 'function', 'alpha', 'alpha', 'src/a.py', 1, 2),
            ('function:b', 'function', 'beta', 'beta', 'src/b.py', 1, 2);
        INSERT INTO edges VALUES ('function:a', 'function:b', 'calls');
        """
    )
    conn.executemany(
        "INSERT INTO files (path, content_hash, errors) VALUES (?, ?, '[]')",
        [
            ("src/a.py", hashlib.sha256(source_a).hexdigest()),
            ("src/b.py", hashlib.sha256(source_b).hexdigest()),
        ],
    )
    conn.commit()
    conn.close()
    return db


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    (repo / "src" / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", "src")
    _git(repo, "commit", "-m", "initial")
    return repo, _seed_codegraph(repo)


def test_codegraph_evidence_detects_fresh_and_stale_source(tmp_path: Path) -> None:
    repo, db = _repo(tmp_path)

    fresh = codegraph_evidence(db_path=db, repo_root=repo, paths=["src/a.py"])
    assert fresh.coverage_status == "complete"
    assert fresh.recording_status == "complete"
    assert fresh.freshness == "current"
    assert fresh.index_generation == "codegraph:8:1785283200000"

    (repo / "src" / "a.py").write_text("def alpha():\n    return 99\n", encoding="utf-8")
    stale = codegraph_evidence(db_path=db, repo_root=repo, paths=["src/a.py"])
    assert stale.coverage_status == "known_gaps"
    assert stale.freshness == "stale"
    assert stale.gaps[0].reason == "stale_source"


def test_normalize_code_path_preserves_dot_directories() -> None:
    assert normalize_code_path("./.github/workflows/ci.yml") == ".github/workflows/ci.yml"


def test_code_reference_resolver_reuses_snapshot_and_path_evidence(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo, db = _repo(tmp_path)
    git_head_calls = 0
    real_git_head = code_evidence._git_head

    def counted_git_head(repo_root: Path) -> str | None:
        nonlocal git_head_calls
        git_head_calls += 1
        return real_git_head(repo_root)

    monkeypatch.setattr(code_evidence, "_git_head", counted_git_head)
    resolver = CodeReferenceResolver(db_path=db, repo_root=repo, repo_id="test-repo")
    payload = {
        "code_refs": [
            {
                "uri": codegraph_uri("test-repo", "function:a"),
                "file_path": "src/a.py",
            },
            {
                "uri": codegraph_uri("test-repo", "function:b"),
                "file_path": "src/b.py",
            },
        ]
    }

    first = resolver.resolve(payload)
    second = resolver.resolve(payload)

    assert git_head_calls == 1
    assert [ref.code_evidence for ref in first] == [ref.code_evidence for ref in second]
    assert all(ref.code_evidence["freshness"] == "current" for ref in first)


def test_code_change_impact_traverses_one_hop(tmp_path: Path) -> None:
    repo, _db = _repo(tmp_path)

    result = code_change_impact(repo, ["src/a.py"], depth=1)

    assert result["available"] is True
    symbols = {item["stable_symbol_id"]: item for item in result["symbols"]}
    assert symbols["function:a"]["distance"] == 0
    assert symbols["function:b"]["distance"] == 1
    assert result["code_evidence"]["schema"] == "memo.code_evidence.v1"
