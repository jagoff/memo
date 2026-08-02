from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import frontmatter

from memo.chunker import DEFAULT_TARGET_CHARS, chunk_markdown
from memo.redact import sanitize_memory_input


def _run_audit(tmp_path: Path, data_dir: Path, state_dir: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "MEMO_DATA_DIR": str(data_dir),
            "MEMO_STATE_DIR": str(state_dir),
            "MEMO_CONFIG_DIR": str(tmp_path / "no-config"),
            "MEMO_NONINTERACTIVE": "1",
        }
    )
    env.pop("MEMO_VAULT_PATH", None)
    return subprocess.run(
        [sys.executable, "scripts/audit-data-integrity.py"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )


def test_audit_data_integrity_supports_data_dir_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()

    body = "canonical production fact"
    memory_path = data_dir / "fact.md"
    memory_path.write_text(
        frontmatter.dumps(frontmatter.Post(body, id="a" * 32, title="Fact")),
        encoding="utf-8",
    )
    parsed_body = frontmatter.loads(memory_path.read_text(encoding="utf-8")).content
    body_hash = hashlib.sha256(parsed_body.encode("utf-8")).hexdigest()[:16]

    connection = sqlite3.connect(state_dir / "memvec.db")
    try:
        connection.execute(
            "CREATE TABLE meta ("
            "id TEXT PRIMARY KEY, path TEXT, title TEXT, body_hash TEXT, "
            "updated TEXT, extra_json TEXT"
            ")"
        )
        connection.execute(
            "INSERT INTO meta (id, path, title, body_hash, updated) VALUES (?, ?, ?, ?, ?)",
            ("a" * 32, memory_path.name, "Fact", body_hash, "2026-07-27T00:00:00Z"),
        )
        connection.commit()
    finally:
        connection.close()

    result = _run_audit(tmp_path, data_dir, state_dir)

    assert result.returncode == 0, result.stderr
    assert f"memory_dir:    {data_dir}" in result.stdout
    assert "✓ healthy:           1" in result.stdout
    assert "✗ bad_path:          0" in result.stdout


def test_audit_data_integrity_ignores_soft_deleted_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()

    body = "active canonical fact"
    memory_path = data_dir / "active.md"
    memory_path.write_text(
        frontmatter.dumps(frontmatter.Post(body, id="a" * 32, title="Active")),
        encoding="utf-8",
    )
    parsed_body = frontmatter.loads(memory_path.read_text(encoding="utf-8")).content
    body_hash = hashlib.sha256(parsed_body.encode("utf-8")).hexdigest()[:16]

    connection = sqlite3.connect(state_dir / "memvec.db")
    try:
        connection.execute(
            "CREATE TABLE meta ("
            "id TEXT PRIMARY KEY, path TEXT, title TEXT, body_hash TEXT, "
            "updated TEXT, extra_json TEXT, deleted_at TEXT"
            ")"
        )
        connection.executemany(
            "INSERT INTO meta "
            "(id, path, title, body_hash, updated, deleted_at) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (
                    "a" * 32,
                    memory_path.name,
                    "Active",
                    body_hash,
                    "2026-07-27T00:00:00Z",
                    None,
                ),
                (
                    "b" * 32,
                    "already-archived.md",
                    "Archived",
                    "0" * 16,
                    "2026-07-27T00:00:00Z",
                    "2026-07-27T01:00:00Z",
                ),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    result = _run_audit(tmp_path, data_dir, state_dir)

    assert result.returncode == 0, result.stdout
    assert "total records: 1" in result.stdout
    assert "✓ healthy:           1" in result.stdout
    assert "✗ bad_path:          0" in result.stdout


def test_audit_data_integrity_fails_for_missing_or_escaping_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()

    connection = sqlite3.connect(state_dir / "memvec.db")
    try:
        connection.execute(
            "CREATE TABLE meta ("
            "id TEXT PRIMARY KEY, path TEXT, title TEXT, body_hash TEXT, "
            "updated TEXT, extra_json TEXT"
            ")"
        )
        connection.executemany(
            "INSERT INTO meta (id, path, title, body_hash, updated) VALUES (?, ?, ?, ?, ?)",
            [
                ("b" * 32, "missing.md", "Missing", "0" * 16, "2026-07-27T00:00:00Z"),
                ("c" * 32, "../escape.md", "Escape", "0" * 16, "2026-07-27T00:00:00Z"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    result = _run_audit(tmp_path, data_dir, state_dir)

    assert result.returncode == 1
    assert "✗ bad_path:          2" in result.stdout


def test_audit_data_integrity_validates_synthetic_chunks(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()

    body = "\n\n".join(
        f"## Section {index}\n" + ("production chunk content " * 120) for index in range(4)
    )
    memory_path = data_dir / "chunked.md"
    memory_path.write_text(
        frontmatter.dumps(frontmatter.Post(body, id="d" * 32, title="Chunked")),
        encoding="utf-8",
    )
    parsed_body = frontmatter.loads(memory_path.read_text(encoding="utf-8")).content
    chunks = chunk_markdown(parsed_body, target_chars=DEFAULT_TARGET_CHARS)
    assert len(chunks) > 1

    connection = sqlite3.connect(state_dir / "memvec.db")
    try:
        connection.execute(
            "CREATE TABLE meta ("
            "id TEXT PRIMARY KEY, path TEXT, title TEXT, body_hash TEXT, "
            "updated TEXT, extra_json TEXT"
            ")"
        )
        connection.executemany(
            "INSERT INTO meta (id, path, title, body_hash, updated) VALUES (?, ?, ?, ?, ?)",
            [
                (
                    f"{'d' * 32}_chunk_{chunk['seq']}",
                    f"{memory_path.name}#chunk-{chunk['seq']}",
                    "Chunk",
                    hashlib.sha256(str(chunk["body"]).encode("utf-8")).hexdigest()[:16],
                    "2026-07-27T00:00:00Z",
                )
                for chunk in chunks
            ],
        )
        connection.commit()
    finally:
        connection.close()

    result = _run_audit(tmp_path, data_dir, state_dir)

    assert result.returncode == 0, result.stdout
    assert f"✓ healthy:           {len(chunks)}" in result.stdout
    assert "✗ bad_path:          0" in result.stdout


def test_audit_data_integrity_hashes_the_sanitized_persisted_body(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()

    raw_body = "Credential-shaped fixture AKIAAAAAAAAAAAAAAAAA must be redacted."
    sanitized = sanitize_memory_input(content=raw_body).content
    assert sanitized != raw_body
    memory_path = data_dir / "redacted.md"
    memory_path.write_text(
        frontmatter.dumps(frontmatter.Post(raw_body, id="e" * 32, title="Redacted")),
        encoding="utf-8",
    )

    connection = sqlite3.connect(state_dir / "memvec.db")
    try:
        connection.execute(
            "CREATE TABLE meta ("
            "id TEXT PRIMARY KEY, path TEXT, title TEXT, body_hash TEXT, "
            "updated TEXT, extra_json TEXT"
            ")"
        )
        connection.execute(
            "INSERT INTO meta (id, path, title, body_hash, updated) VALUES (?, ?, ?, ?, ?)",
            (
                "e" * 32,
                memory_path.name,
                "Redacted",
                hashlib.sha256(sanitized.encode("utf-8")).hexdigest()[:16],
                "2026-07-27T00:00:00Z",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    result = _run_audit(tmp_path, data_dir, state_dir)

    assert result.returncode == 0, result.stdout
    assert "✓ healthy:           1" in result.stdout
    assert "✗ hash_mismatch:     0" in result.stdout


def test_audit_data_integrity_validates_external_vault_ingest_rows(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    source_dir = tmp_path / "external" / "Notes"
    data_dir.mkdir()
    state_dir.mkdir()
    source_dir.mkdir(parents=True)

    indexed_body = "external indexed chunk"
    source_path = source_dir / "reference.md"
    source_path.write_text("# External source\n\nCurrent body", encoding="utf-8")
    extra = {
        "source": "vault-ingest",
        "vault": "notes",
        "abs_path": str(source_path),
        "parent_path": "notes/reference.md",
        "chunk_seq": 0,
    }

    connection = sqlite3.connect(state_dir / "memvec.db")
    try:
        connection.execute(
            "CREATE TABLE meta ("
            "id TEXT PRIMARY KEY, path TEXT, title TEXT, body_hash TEXT, "
            "updated TEXT, extra_json TEXT"
            ")"
        )
        connection.execute("CREATE VIRTUAL TABLE fts USING fts5(id UNINDEXED, title, tags, body)")
        connection.execute(
            "INSERT INTO meta "
            "(id, path, title, body_hash, updated, extra_json) VALUES (?, ?, ?, ?, ?, ?)",
            (
                "f" * 32,
                "notes/reference.md#chunk-0",
                "External",
                hashlib.sha256(indexed_body.encode("utf-8")).hexdigest()[:16],
                "2026-07-27T00:00:00Z",
                json.dumps(extra),
            ),
        )
        connection.execute(
            "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
            ("f" * 32, "External", "notes", indexed_body),
        )
        connection.commit()
    finally:
        connection.close()

    result = _run_audit(tmp_path, data_dir, state_dir)

    assert result.returncode == 0, result.stdout
    assert "✓ healthy:           1" in result.stdout
    assert "✗ bad_path:          0" in result.stdout


def test_audit_data_integrity_validates_external_vault_ingest_media_rows(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    state_dir = tmp_path / "state"
    source_dir = tmp_path / "external" / "Work" / "attachments"
    data_dir.mkdir()
    state_dir.mkdir()
    source_dir.mkdir(parents=True)

    fixtures = [
        ("vault-ingest-pdf", "report.pdf", "indexed PDF text"),
        ("vault-ingest-image", "diagram.webp", "indexed OCR text"),
        ("vault-ingest-audio", "meeting.m4a", "indexed transcript"),
    ]
    connection = sqlite3.connect(state_dir / "memvec.db")
    try:
        connection.execute(
            "CREATE TABLE meta ("
            "id TEXT PRIMARY KEY, path TEXT, title TEXT, body_hash TEXT, "
            "updated TEXT, extra_json TEXT"
            ")"
        )
        connection.execute("CREATE VIRTUAL TABLE fts USING fts5(id UNINDEXED, title, tags, body)")
        for index, (source, name, indexed_body) in enumerate(fixtures):
            source_path = source_dir / name
            source_path.write_bytes(b"external source fixture")
            logical_path = f"work/attachments/{name}"
            extra = {
                "source": source,
                "vault": "work",
                "abs_path": str(source_path),
                "parent_path": logical_path,
            }
            memory_id = str(index + 1) * 32
            connection.execute(
                "INSERT INTO meta "
                "(id, path, title, body_hash, updated, extra_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    logical_path,
                    name,
                    hashlib.sha256(indexed_body.encode("utf-8")).hexdigest()[:16],
                    "2026-08-02T00:00:00Z",
                    json.dumps(extra),
                ),
            )
            connection.execute(
                "INSERT INTO fts (id, title, tags, body) VALUES (?, ?, ?, ?)",
                (memory_id, name, source, indexed_body),
            )
        connection.commit()
    finally:
        connection.close()

    result = _run_audit(tmp_path, data_dir, state_dir)

    assert result.returncode == 0, result.stdout
    assert "✓ healthy:           3" in result.stdout
    assert "✗ bad_path:          0" in result.stdout
