from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

from memo.cli_diag import _db_health_report
from memo.config import Config


def _cfg(tmp_path: Path, *, dims: int = 4) -> Config:
    return Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        embedder_dims=dims,
        reranker_enabled=False,
    )


def test_db_health_checks_integrity_and_vector_dims(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True)
    with closing(sqlite3.connect(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE meta (id TEXT PRIMARY KEY, updated TEXT)")
        conn.execute("CREATE TABLE repo_sources (id TEXT PRIMARY KEY, indexed_at TEXT)")
        conn.execute("CREATE TABLE repo_chunks (id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE vec (id TEXT PRIMARY KEY, embedding FLOAT[4])")
        conn.execute("CREATE TABLE repo_vec (id TEXT PRIMARY KEY, embedding FLOAT[4])")
        conn.execute("INSERT INTO meta VALUES ('m1', '2026-05-24T00:00:00Z')")
        conn.execute("INSERT INTO repo_sources VALUES ('r1', '2026-05-24T01:00:00Z')")
        conn.commit()

    report = {item["label"]: item for item in _db_health_report(cfg)}

    assert report["memvec"]["ok"] is True
    assert report["memvec"]["integrity_check"] == "ok"
    assert report["memvec"]["vec_dims"] == 4
    assert report["memvec"]["repo_vec_dims"] == 4
    assert report["history"]["status"] == "missing"


def test_db_health_flags_embedding_dimension_mismatch(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.state_dir.mkdir(parents=True)
    with closing(sqlite3.connect(cfg.db_path)) as conn:
        conn.execute("CREATE TABLE vec (id TEXT PRIMARY KEY, embedding FLOAT[4])")
        conn.execute("CREATE TABLE repo_vec (id TEXT PRIMARY KEY, embedding FLOAT[4])")
        conn.commit()

    report = {item["label"]: item for item in _db_health_report(_cfg(tmp_path, dims=8))}

    assert report["memvec"]["ok"] is False
    assert report["memvec"]["status"] == "dimension_mismatch"
    assert report["memvec"]["expected_dims"] == 8
