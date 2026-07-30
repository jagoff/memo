from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path

from memo.config import Config
from memo.memory import Memory
from memo.repo_index_search import classify_repo_path
from memo.repo_signals import (
    collect_git_change_signals,
    expand_cochange_paths,
    service_for_path,
)
from memo.repo_structural import search_codegraph_paths


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(root: Path) -> Path:
    repo = root / "intelligence-repo"
    for path, text in {
        "src/alpha.py": "def alphaunique():\n    return 'alphaunique'\n",
        "services/beta/worker.py": "def worker():\n    return 'worker'\n",
        "tests/test_alpha.py": "def test_alphaunique():\n    assert True\n",
        "vendor/lib/copied.py": "alphaunique = 'vendored'\n",
    }.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    (repo / "src/alpha.py").write_text(
        "def alphaunique():\n    return 'alphaunique-v2'\n",
        encoding="utf-8",
    )
    (repo / "services/beta/worker.py").write_text(
        "def worker():\n    return 'worker-v2'\n",
        encoding="utf-8",
    )
    _git(repo, "add", "src/alpha.py", "services/beta/worker.py")
    _git(repo, "commit", "-m", "change alpha and worker")
    return repo


def _cfg(tmp_path: Path) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        embedder_dims=4,
        reranker_enabled=False,
    )


def test_git_change_signals_capture_cross_service_and_expand(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    signals = collect_git_change_signals(repo, max_commits=20)

    assert signals["analyzed_commits"] == 2
    expanded = expand_cochange_paths(signals, ["src/alpha.py"])
    assert any(row["path"] == "services/beta/worker.py" for row in expanded)
    assert service_for_path("services/beta/api.py") == "services/beta"


def test_scope_classification() -> None:
    assert classify_repo_path("src/app.py") == "production"
    assert classify_repo_path("tests/test_app.py") == "tests"
    assert classify_repo_path("packages/a/__tests__/x.ts") == "tests"
    assert classify_repo_path("third_party/lib/x.cc") == "vendor"


def test_structural_provider_reads_codegraph_and_one_hop(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    db_path = root / ".codegraph" / "codegraph.db"
    db_path.parent.mkdir(parents=True)
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE nodes (
          id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
          file_path TEXT, start_line INTEGER, end_line INTEGER,
          signature TEXT, is_exported INTEGER
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes VALUES
          ('a', 'function', 'alphaunique', 'src.alphaunique', 'src/alpha.py', 1, 2,
           'def alphaunique()', 1),
          ('b', 'function', 'worker', 'services.beta.worker', 'services/beta/worker.py', 1, 2,
           'def worker()', 1);
        INSERT INTO edges VALUES ('a', 'b', 'calls');
        """
    )
    connection.commit()
    connection.close()

    result = search_codegraph_paths(root, "where is alphaunique", limit=10)

    assert result["status"] == "available"
    paths = {row["path"]: row for row in result["paths"]}
    assert "src/alpha.py" in paths
    assert "services/beta/worker.py" in paths
    assert paths["src/alpha.py"]["evidence"][0]["kind"] == "symbol_match"


def test_unified_search_scopes_artifacts_cochange_and_atomic_visibility(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    memory = Memory(_cfg(tmp_path))
    try:
        indexed = memory.repo_index(str(repo), name="intelligence", with_embeddings=False)
        status = memory.repo_status("intelligence")
        assert status is not None
        assert status["artifact_verification"]["generation"]["ok"] is True
        assert status["artifact_verification"]["change_signals"]["ok"] is True

        production = memory.repo_search(
            "alphaunique",
            repo="intelligence",
            mode="lexical",
            scope="production",
        )
        assert production
        assert all(hit.scope == "production" for hit in production)
        tests = memory.repo_search(
            "alphaunique",
            repo="intelligence",
            mode="lexical",
            scope="tests",
        )
        assert tests and all(hit.scope == "tests" for hit in tests)
        vendor = memory.repo_search(
            "alphaunique",
            repo="intelligence",
            mode="lexical",
            scope="vendor",
        )
        assert vendor and all(hit.scope == "vendor" for hit in vendor)

        unified = memory.repo_search(
            "alphaunique",
            repo="intelligence",
            mode="unified",
            scope="production",
        )
        assert any(hit.path == "services/beta/worker.py" for hit in unified)
        assert any("cochange" in hit.channel_scores for hit in unified)
        assert all(hit.index_generation for hit in unified)

        memory.store.update_repo_status(indexed["repo_id"], "indexing")
        assert (
            memory.repo_search(
                "alphaunique",
                repo="intelligence",
                mode="lexical",
            )
            == []
        )
        memory.store.update_repo_status(indexed["repo_id"], "semantic_pending")
        assert memory.repo_search("alphaunique", repo="intelligence", mode="lexical")
    finally:
        memory.close()
