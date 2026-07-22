"""`memo migrate-vault` — copy memorias to a new data_dir, rebuild index."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.config import Config
from memo.memory import Memory
from memo.store import VecStore


@pytest.fixture
def seeded_old_layout(tmp_path: Path, monkeypatch):
    """Seed an 'old' data_dir with three memorias, indexed.

    Returns (cfg, memo_files) where cfg points at the seeded layout.
    """
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    old_data = tmp_path / "old"
    old_data.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    cfg = Config(data_dir=old_data, state_dir=state, embedder_dims=4)
    mem = Memory(cfg)
    files = []
    for i, title in enumerate(["A primero", "B segundo", "C tercero"]):
        rec = mem.save(content=f"contenido del memo {i}", title=title)
        files.append(rec.path)
    mem.store.close() if hasattr(mem.store, "close") else None
    return cfg, files


def test_migrate_copies_files_and_reindexes(
    tmp_path: Path,
    seeded_old_layout,
    monkeypatch,
):
    cfg, _files = seeded_old_layout
    new_data = tmp_path / "new"
    cfg_file = tmp_path / "memo-config.toml"

    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    runner = CliRunner()
    env = {
        "MEMO_CONFIG_FILE": str(cfg_file),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),  # source
        "MEMO_STATE_DIR": str(cfg.state_dir),
        # Match the 4-dim stub embedder. Tests in this repo override
        # MLXEmbedder.embed via monkeypatch but the dim assertion in
        # `Config` is driven by env, so we have to pin it here.
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    result = runner.invoke(
        cli,
        ["migrate-vault", str(new_data), "--yes"],
        env=env,
    )
    assert result.exit_code == 0, result.output

    # All 3 .md files copied to new location.
    new_files = sorted(new_data.rglob("*.md"))
    assert len(new_files) == 3

    # Old files preserved (migration is non-destructive).
    old_files = sorted(cfg.data_dir.rglob("*.md"))
    assert len(old_files) == 3

    # Config file updated to point at new dir.
    body = cfg_file.read_text(encoding="utf-8")
    assert f'data_dir = "{new_data.resolve()}"' in body

    # memvec.db rebuilt from new location.
    assert (cfg.state_dir / "memvec.db").is_file()


def test_migrate_refuses_non_empty_destination(tmp_path: Path, seeded_old_layout):
    cfg, _ = seeded_old_layout
    new_data = tmp_path / "non-empty"
    new_data.mkdir()
    (new_data / "stranger.md").write_text("not yours", encoding="utf-8")

    runner = CliRunner()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    result = runner.invoke(
        cli,
        ["migrate-vault", str(new_data), "--yes"],
        env=env,
    )
    assert result.exit_code == 1
    assert "non-empty" in result.output


def test_migrate_refuses_same_src_and_dst(tmp_path: Path, seeded_old_layout):
    cfg, _ = seeded_old_layout
    runner = CliRunner()
    env = {
        "MEMO_CONFIG_FILE": str(tmp_path / "config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }
    result = runner.invoke(
        cli,
        ["migrate-vault", str(cfg.data_dir), "--yes"],
        env=env,
    )
    assert result.exit_code == 1
    assert "same" in result.output.lower()


def _base_env(tmp_path: Path, cfg: Config, cfg_file: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(cfg_file),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(cfg.data_dir),
        "MEMO_STATE_DIR": str(cfg.state_dir),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_EMBEDDER_MODEL": "stub",
    }


def test_migrate_preserves_db_and_access_signal(tmp_path: Path, seeded_old_layout, monkeypatch):
    """The data-loss bug fix: migrate must NOT drop memvec.db, so user-signal
    data (access counts) keyed on the stable id survives the migration."""
    cfg, _ = seeded_old_layout
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    # Bump access telemetry on the first memoria.
    store = VecStore(cfg.state_dir / "memvec.db", dims=4)
    ids = [r["id"] for r in store._conn.execute("SELECT id FROM meta ORDER BY path").fetchall()]
    target = ids[0]
    store.touch([target])
    store.touch([target])
    assert store.get_access(target)["access_count"] == 2

    new_data = tmp_path / "new"
    cfg_file = tmp_path / "memo-config.toml"
    result = CliRunner().invoke(
        cli,
        ["migrate-vault", str(new_data), "--yes"],
        env=_base_env(tmp_path, cfg, cfg_file),
    )
    assert result.exit_code == 0, result.output
    assert "removed stale memvec.db" not in result.output

    # Same DB file, same ids, signal intact.
    store2 = VecStore(cfg.state_dir / "memvec.db", dims=4)
    ids2 = [r["id"] for r in store2._conn.execute("SELECT id FROM meta").fetchall()]
    assert set(ids2) == set(ids)
    assert store2.get_access(target)["access_count"] == 2


def test_migrate_into_vault_sets_layout(tmp_path: Path, seeded_old_layout, monkeypatch):
    """--into-vault moves memorias under <vault>/Obsidian/AI/memory and writes
    memories_in_vault=1 so the vault is the source of truth."""
    cfg, _ = seeded_old_layout
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    vault = tmp_path / "vault"
    vault.mkdir()
    cfg_file = tmp_path / "memo-config.toml"
    env = _base_env(tmp_path, cfg, cfg_file)
    env["MEMO_VAULT_PATH"] = str(vault)

    result = CliRunner().invoke(cli, ["migrate", "--into-vault", "--yes"], env=env)
    assert result.exit_code == 0, result.output

    dst = vault / "Obsidian" / "AI" / "memory"
    assert len(sorted(dst.rglob("*.md"))) == 3
    body = cfg_file.read_text(encoding="utf-8")
    assert "memories_in_vault = true" in body
    assert f'vault_path = "{vault.resolve()}"' in body


def test_migrate_rollback_restores_config(tmp_path: Path, seeded_old_layout, monkeypatch):
    """--rollback restores the pre-migration config snapshot."""
    cfg, _ = seeded_old_layout
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    cfg_file = tmp_path / "memo-config.toml"
    # Seed an initial config so there's something to snapshot + restore.
    from memo.setup.config_io import write_config_file

    write_config_file(data_dir=cfg.data_dir, path=cfg_file)
    before = cfg_file.read_text(encoding="utf-8")

    env = _base_env(tmp_path, cfg, cfg_file)
    new_data = tmp_path / "new"
    r1 = CliRunner().invoke(cli, ["migrate-vault", str(new_data), "--yes"], env=env)
    assert r1.exit_code == 0, r1.output
    assert cfg_file.read_text(encoding="utf-8") != before  # config changed

    r2 = CliRunner().invoke(cli, ["migrate-vault", "--rollback"], env=env)
    assert r2.exit_code == 0, r2.output
    assert cfg_file.read_text(encoding="utf-8") == before  # restored


def test_consolidate_db_merges_sidecars_and_is_idempotent(
    tmp_path: Path, seeded_old_layout, monkeypatch
):
    """--consolidate-db merges the sidecar DBs into memvec.db, renames legacy
    files to *.bak, flips single_db=1, and is a no-op on re-run."""
    cfg, _ = seeded_old_layout
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    # Seeding three memorias wrote history events (+ created graph.db).
    assert (cfg.state_dir / "history.db").is_file()
    with closing(sqlite3.connect(cfg.state_dir / "history.db")) as conn:
        n_events = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    assert n_events >= 3

    cfg_file = tmp_path / "memo-config.toml"
    env = _base_env(tmp_path, cfg, cfg_file)
    result = CliRunner().invoke(cli, ["migrate", "--consolidate-db"], env=env)
    assert result.exit_code == 0, result.output

    # Legacy renamed aside; events now live inside memvec.db.
    assert (cfg.state_dir / "history.db.bak").is_file()
    assert not (cfg.state_dir / "history.db").is_file()
    with closing(sqlite3.connect(cfg.state_dir / "memvec.db")) as conn:
        merged = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    assert merged == n_events
    # Config flipped on.
    assert "single_db = true" in cfg_file.read_text(encoding="utf-8")

    # Idempotent: re-running doesn't error or duplicate.
    r2 = CliRunner().invoke(cli, ["migrate", "--consolidate-db"], env=env)
    assert r2.exit_code == 0, r2.output
    with closing(sqlite3.connect(cfg.state_dir / "memvec.db")) as conn:
        merged2 = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    assert merged2 == n_events


def test_consolidate_db_merges_episode_fact_edge_and_verbatim_sidecars(
    tmp_path: Path, seeded_old_layout, monkeypatch
):
    """episodes.db / fact_edges.db / verbatim.db must be merged too — they
    collapse onto db_path under single_db=1 and were silently orphaned."""
    from memo.store.episode_store import EpisodeStore
    from memo.store.fact_edge_store import FactEdgeStore
    from memo.store.turn_store import TurnStore

    cfg, _ = seeded_old_layout
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    ep = EpisodeStore(cfg.state_dir / "episodes.db", 4, embedder_model="stub")
    ep.upsert(
        agent="claude",
        session_id="s1",
        content_hash="h1",
        embedding=[1.0, 0.0, 0.0, 0.0],
        cwd="/tmp",
        updated_at="2026-01-01T00:00:00+00:00",
        summary="arreglamos el bug",
        resume_command=["claude", "--resume", "s1"],
        turn_count=3,
    )
    ep.close()
    fe = FactEdgeStore(cfg.state_dir / "fact_edges.db")
    fe.upsert_fact(subject="memo", predicate="usa", object="sqlite-vec")
    fe.close()
    ts = TurnStore(cfg.state_dir / "verbatim.db")
    ts.replace_session(
        "s1", "claude", [{"idx": 0, "role": "user", "ts": "2026-01-01", "text": "hola decisión"}]
    )
    ts.close()

    cfg_file = tmp_path / "memo-config.toml"
    env = _base_env(tmp_path, cfg, cfg_file)
    result = CliRunner().invoke(cli, ["migrate", "--consolidate-db"], env=env)
    assert result.exit_code == 0, result.output

    for name in ("episodes.db", "fact_edges.db", "verbatim.db"):
        assert (cfg.state_dir / f"{name}.bak").is_file(), result.output
        assert not (cfg.state_dir / name).is_file()

    with closing(sqlite3.connect(cfg.state_dir / "memvec.db")) as conn:
        assert conn.execute("SELECT count(*) FROM episode_meta").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM fact_edges").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM turns").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM turns_fts").fetchone()[0] == 1

    # The episode VECTOR made it across too (vec0 needs the extension loaded,
    # so verify through the store instead of a raw connection).
    merged_ep = EpisodeStore(cfg.state_dir / "memvec.db", 4, embedder_model="stub")
    try:
        assert merged_ep.count() == 1
        hits = merged_ep.search([1.0, 0.0, 0.0, 0.0], k=1)
        assert hits and hits[0]["session_id"] == "s1"
    finally:
        merged_ep.close()


def test_consolidate_db_merges_all_six_graph_tables(
    tmp_path: Path, seeded_old_layout, monkeypatch
):
    """graph.db has SIX tables; the migration used to merge only entities +
    entity_memory, silently orphaning co_recall / entity_edges / entity_aliases
    / semantic_relations into graph.db.bak. All six must cross over."""
    from memo.graph import GraphStore

    cfg, _ = seeded_old_layout
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )

    # Opening GraphStore materializes the full 6-table schema on graph.db.
    GraphStore(cfg.state_dir / "graph.db").close()
    with closing(sqlite3.connect(cfg.state_dir / "graph.db")) as conn:
        conn.execute("INSERT INTO co_recall(id_a, id_b, count) VALUES ('a', 'b', 2)")
        conn.execute(
            "INSERT INTO entity_edges(a_id, b_id, weight, first_seen, last_seen) "
            "VALUES (1, 2, 3, '2026-01-01', '2026-01-02')"
        )
        conn.execute(
            "INSERT INTO entity_aliases(alias_key, canonical_id, alias_name) "
            "VALUES ('k1', 1, 'Memo')"
        )
        conn.execute(
            "INSERT INTO semantic_relations("
            "source_kind, source_id, target_kind, target_id, relation, "
            "derived_from, created_at) "
            "VALUES ('memory', 'm1', 'memory', 'm2', 'supports', 'test', '2026-01-01')"
        )
        conn.commit()

    cfg_file = tmp_path / "memo-config.toml"
    env = _base_env(tmp_path, cfg, cfg_file)
    result = CliRunner().invoke(cli, ["migrate", "--consolidate-db"], env=env)
    assert result.exit_code == 0, result.output
    assert (cfg.state_dir / "graph.db.bak").is_file()
    assert not (cfg.state_dir / "graph.db").is_file()

    with closing(sqlite3.connect(cfg.state_dir / "memvec.db")) as conn:
        for tbl in ("co_recall", "entity_edges", "entity_aliases", "semantic_relations"):
            assert conn.execute(f"SELECT count(*) FROM {tbl}").fetchone()[0] == 1, (  # noqa: S608
                f"{tbl} orphaned by consolidate"
            )


def test_links_reindex_safe_under_single_db(tmp_path: Path, monkeypatch):
    """`memo links reindex` must truncate the crossref table, not unlink the DB
    file — under single_db that file IS memvec.db."""
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    data = tmp_path / "data"
    state = tmp_path / "state"
    data.mkdir()
    state.mkdir()
    cfg = Config(data_dir=data, state_dir=state, single_db=True, embedder_dims=4)
    mem = Memory(cfg)
    mem.save(content="links a [[Otra]] cosa", title="Con Links")
    assert cfg.db_path.is_file()

    cfg_file = tmp_path / "memo-config.toml"
    env = _base_env(tmp_path, cfg, cfg_file)
    env["MEMO_SINGLE_DB"] = "1"
    result = CliRunner().invoke(cli, ["links", "reindex", "--yes"], env=env)
    assert result.exit_code == 0, result.output
    # The DB file (== memvec.db) must still exist with memorias intact.
    assert cfg.db_path.is_file()
    assert (
        Memory(
            Config.from_env(
                **{
                    "data_dir": data,
                    "state_dir": state,
                    "single_db": True,
                    "embedder_dims": 4,
                    "embedder_model": cfg.embedder_model,
                }
            )
        ).store.count()
        == 1
    )
