from __future__ import annotations

import hashlib
import sqlite3
import subprocess
from pathlib import Path

from memo.code_context import (
    CODE_CONTEXT_SCHEMA,
    CodeContextFinding,
    CodeContextProviderResult,
    CodeContextRequest,
    build_code_context_pack,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


def _seed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    sources = {
        "src/api/routes.py": "def route():\n    return controller()\n",
        "src/core/service.py": "def service():\n    return 1\n",
        "src/data/store.py": "def store():\n    return 1\n",
    }
    for relative, content in sources.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    subprocess.run(["git", "init", str(repo)], check=True, capture_output=True, text=True)
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "add", "src")
    _git(repo, "commit", "-m", "initial")

    db = repo / ".codegraph" / "codegraph.db"
    db.parent.mkdir()
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE project_metadata (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE schema_versions (version INTEGER);
        CREATE TABLE files (
            path TEXT PRIMARY KEY,
            content_hash TEXT,
            language TEXT,
            errors TEXT
        );
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY,
            kind TEXT,
            name TEXT,
            qualified_name TEXT,
            file_path TEXT,
            language TEXT,
            start_line INTEGER,
            end_line INTEGER,
            signature TEXT,
            docstring TEXT
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO project_metadata VALUES
            ('index_state', 'complete'),
            ('files_discovered', '3'),
            ('files_accounted', '3'),
            ('indexed_with_version', 'test-architecture-1'),
            ('updated_at', '1785283200000');
        INSERT INTO schema_versions VALUES (8);
        INSERT INTO nodes VALUES
            ('route:http', 'route', 'GET /items', 'GET /items',
             'src/api/routes.py', 'python', 1, 2, 'GET /items', ''),
            ('function:controller', 'function', 'controller', 'api.controller',
             'src/api/routes.py', 'python', 1, 2, 'controller()', ''),
            ('function:service', 'function', 'service', 'core.service',
             'src/core/service.py', 'python', 1, 2, 'service()', ''),
            ('function:store', 'function', 'store', 'data.store',
             'src/data/store.py', 'python', 1, 2, 'store()', '');
        INSERT INTO edges VALUES
            ('route:http', 'function:controller', 'calls'),
            ('function:controller', 'function:service', 'calls'),
            ('function:service', 'function:controller', 'references'),
            ('function:service', 'function:store', 'calls');
        """
    )
    conn.executemany(
        "INSERT INTO files (path, content_hash, language, errors) VALUES (?, ?, 'python', '[]')",
        [
            (relative, hashlib.sha256((repo / relative).read_bytes()).hexdigest())
            for relative in sources
        ],
    )
    conn.commit()
    conn.close()
    return repo


def test_scout_pack_exposes_architecture_without_absence_claims(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)

    pack = build_code_context_pack(repo, mode="scout", limit=50)

    assert pack["schema"] == CODE_CONTEXT_SCHEMA
    assert pack["available"] is True
    assert pack["architecture"]["hotspots"]
    assert pack["architecture"]["boundaries"]
    assert pack["architecture"]["cycles"]
    assert pack["architecture"]["routes"]
    assert pack["architecture"]["packages"]
    assert pack["architecture"]["layers"]
    assert pack["claims"]["provider_records_complete"] is True
    assert pack["claims"]["positive_findings_only"] is True
    assert pack["claims"]["absence_claim_allowed"] is False
    assert pack["code_evidence"]["schema"] == "memo.code_evidence.v1"


def test_verify_exact_path_can_establish_source_completeness(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)

    pack = build_code_context_pack(
        repo,
        focus="src/api/routes.py",
        mode="verify",
        limit=50,
    )

    assert pack["page"]["exhausted"] is True
    assert pack["request"]["scope"] == "src"
    assert pack["request"]["scope_inferred"] is True
    assert pack["architecture"]["blast_radius"]
    assert pack["code_evidence"]["requested_paths"] == ["src/api/routes.py"]
    assert pack["code_evidence"]["freshness"] == "current"
    assert pack["claims"]["source_universe_complete"] is True
    assert pack["claims"]["absence_claim_allowed"] is True
    assert pack["claims"]["absence_claim_scope"] == {
        "kind": "path",
        "paths": ["src/api/routes.py"],
    }
    assert pack["claims"]["architecture_absence_claim_allowed"] is False
    assert any(item["kind"] == "cross_scope_edges" for item in pack["omissions"])


def test_audit_cursor_paginates_one_stable_result_set(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)

    first = build_code_context_pack(repo, mode="audit", limit=2)

    assert first["page"]["returned"] == 2
    assert first["page"]["exhausted"] is False
    assert first["claims"]["absence_claim_allowed"] is False
    assert first["omissions"][0]["kind"] == "pagination"
    cursor = first["page"]["next_cursor"]
    assert cursor

    seen = list(first["findings"])
    page = first
    while page["page"]["next_cursor"]:
        page = build_code_context_pack(
            repo,
            mode="audit",
            limit=2,
            cursor=page["page"]["next_cursor"],
        )
        seen.extend(page["findings"])

    assert page["page"]["exhausted"] is True
    assert len(seen) == first["page"]["total_findings"]
    assert len({item["id"] for item in seen}) == len(seen)
    assert page["claims"]["source_universe_complete"] is False


def test_cursor_is_bound_to_focus_and_generation(tmp_path: Path) -> None:
    repo = _seed_repo(tmp_path)
    first = build_code_context_pack(repo, mode="audit", limit=1)

    invalid = build_code_context_pack(
        repo,
        focus="service",
        mode="audit",
        limit=1,
        cursor=first["page"]["next_cursor"],
    )

    assert invalid["available"] is False
    assert invalid["reason"] == "invalid_cursor"


class _FixtureProvider:
    name = "fixture"

    def collect(self, request: CodeContextRequest) -> CodeContextProviderResult:
        return CodeContextProviderResult(
            provider=self.name,
            provider_version="1",
            index_generation="fixture:g1",
            findings=(
                CodeContextFinding(
                    kind="package",
                    id="package:src",
                    label="src",
                    score=1.0,
                    data={"package": "src"},
                ),
            ),
            code_evidence={
                "schema": "memo.code_evidence.v1",
                "recording_status": "complete",
                "coverage_status": "complete",
                "freshness": "current",
                "requested_paths": ["src/example.py"],
                "requested_scopes": [],
                "limitations": [],
            },
            records_complete=True,
        )


def test_contract_accepts_a_provider_other_than_codegraph(tmp_path: Path) -> None:
    pack = build_code_context_pack(
        tmp_path,
        focus="src/example.py",
        mode="verify",
        provider=_FixtureProvider(),
    )

    assert pack["provider"]["name"] == "fixture"
    assert pack["findings"][0]["id"] == "package:src"
    assert pack["claims"]["absence_claim_allowed"] is True
