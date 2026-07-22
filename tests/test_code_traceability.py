from __future__ import annotations

import sqlite3
from pathlib import Path

from memo import code_traceability, codegraph_loader
from memo.code_traceability import (
    codegraph_repo_id,
    codegraph_uri,
    parse_codegraph_uri,
    resolve_code_references,
)


def _seed_codegraph(path: Path) -> None:
    path.parent.mkdir(parents=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL,
            file_path TEXT NOT NULL,
            language TEXT NOT NULL,
            start_line INTEGER NOT NULL,
            end_line INTEGER NOT NULL
        );
        INSERT INTO nodes VALUES
          ('file:src/memo/graph.py', 'file', 'graph.py', 'src/memo/graph.py',
           'src/memo/graph.py', 'python', 1, 900),
          ('function:abc123', 'function', 'rebuild_edges', 'GraphStore.rebuild_edges',
           'src/memo/graph.py', 'python', 400, 450);
        """
    )
    conn.commit()
    conn.close()


def test_codegraph_uri_round_trips_stable_symbol_id() -> None:
    uri = codegraph_uri("repo-1", "function:abc/123")
    assert uri == "codegraph://repo-1/function%3Aabc%2F123"
    assert parse_codegraph_uri(uri) == ("repo-1", "function:abc/123")


def test_repo_id_is_stable_for_equivalent_remote_spellings(tmp_path: Path) -> None:
    assert codegraph_repo_id(tmp_path, remote="git@github.com:Org/Memo.git") == codegraph_repo_id(
        tmp_path, remote="git@github.com:Org/Memo"
    )


def test_resolve_file_capture_to_codegraph_file_node(tmp_path: Path) -> None:
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph(db)

    refs = resolve_code_references(
        {
            "files_read": [str(tmp_path / "src/memo/graph.py")],
            "files_modified": ["src/memo/graph.py"],
        },
        db_path=db,
        repo_root=tmp_path,
        repo_id="memo-repo",
    )

    assert {(ref.stable_symbol_id, ref.relation) for ref in refs} == {
        ("file:src/memo/graph.py", "read"),
        ("file:src/memo/graph.py", "modified"),
    }
    assert all(ref.uri.startswith("codegraph://memo-repo/") for ref in refs)


def test_explicit_uri_survives_without_codegraph_database(tmp_path: Path) -> None:
    uri = codegraph_uri("memo-repo", "function:abc123")

    refs = resolve_code_references(
        {"code_refs": [uri]},
        db_path=tmp_path / "missing.db",
        repo_root=tmp_path,
    )

    assert len(refs) == 1
    assert refs[0].uri == uri
    assert refs[0].relation == "explicit"


def test_default_resolver_uses_shared_codegraph_from_git_worktree(
    tmp_path: Path, monkeypatch
) -> None:
    main_repo = tmp_path / "main"
    worktree = tmp_path / "worktree"
    db = main_repo / ".codegraph" / "codegraph.db"
    _seed_codegraph(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", worktree / ".codegraph/codegraph.db")
    monkeypatch.setattr(code_traceability, "_git_common_repo_root", lambda _root: main_repo)

    refs = resolve_code_references({"files_modified": ["src/memo/graph.py"]})

    assert [(ref.stable_symbol_id, ref.relation) for ref in refs] == [
        ("file:src/memo/graph.py", "modified")
    ]
