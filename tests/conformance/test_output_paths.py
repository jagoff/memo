"""Every `-o/--out` surface: a clean error or a written file. Never a traceback.

/tmp is a symlink on macOS, which is why the symlink case is not exotic.

Two prior defects motivate this gate:
  - `atomic_write_text` rejected any destination whose parent is a symlink
    (fixed by follow-the-symlink commit `94f5faa6`) -- `memo graph mindmap`
    and `memo federation export` share that primitive.
  - `memo backup --out` and `memo export <fmt>` raw-tracebacked on a missing
    parent directory (fixed by the destination pre-check commit `3a30e660`).

A Click *usage* error (wrong flag, missing required option) also exits
non-zero, and it is easy to mistake for "the command failed cleanly" if you
only check `result.exception`. Click gives `UsageError` its own exit code
(2), distinct from the exit code (1) a plain `ClickException` raises -- a
non-zero exit_code == 2 means the harness called the command wrong, not that
the command's own destination-handling logic ever ran. That is the signal
this module uses to keep the vacuous-pass trap out of these assertions.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.graph import GraphStore

from .conftest import _env, seeded_id

pytestmark = pytest.mark.conformance

_CREATED = "2026-01-01T00:00:00+00:00"
_ENTITY_A = "conformance-mindmap-alpha"
_ENTITY_B = "conformance-mindmap-beta"


def _key_file(tmp_path: Path) -> Path:
    """A federation signing key valid enough that `_read_key` accepts it, so
    the only thing left that can fail in the export is the destination path."""
    path = tmp_path / "federation.key"
    path.write_bytes(b"conformance-signing-key-material")
    path.chmod(0o600)
    return path


def _seed_graph_entities(graph_db: Path, *, memory_id: str) -> None:
    store = GraphStore(graph_db)
    try:
        store.record_extraction(
            memory_id=memory_id,
            memory_date=_CREATED,
            entities=[
                {"name": _ENTITY_A, "type": "concept"},
                {"name": _ENTITY_B, "type": "concept"},
            ],
            extracted_at=_CREATED,
        )
    finally:
        store.close()


def _delete_seeded_graph_rows(graph_db: Path, *, memory_id: str) -> None:
    """Undo exactly what `_seed_graph_entities` wrote.

    Not `GraphStore.drop_for_memoria` -- it zeroes `mention_count` on the
    touched entities but leaves their rows in `entities`, which is still a
    residue a payload-size assertion over `export_json()`/`top_entities()`
    would see. Delete by the exact memory_id/name+type this fixture wrote,
    so a shared `graph.db` comes back byte-for-byte to how it looked before
    this fixture ran. `record_extraction` also calls `_mark_projection_dirty`
    (graph.py:355), which inserts a `('dirty', '1')` row into
    `graph_projection_state` (graph.py:126) -- `graph_projection.py:545`
    reads that key, and a leftover row here would perturb the MCP
    graph-export payload-size baseline this same `graph.db` is shared with,
    so clear it too.
    """
    if not graph_db.exists():
        return
    conn = sqlite3.connect(str(graph_db))
    try:
        with conn:
            conn.execute("DELETE FROM entity_memory WHERE memory_id = ?", (memory_id,))
            conn.execute(
                "DELETE FROM entities WHERE name IN (?, ?) AND type = 'concept'",
                (_ENTITY_A, _ENTITY_B),
            )
            conn.execute("DELETE FROM graph_projection_state WHERE key = 'dirty'")
    finally:
        conn.close()


def _projection_dirty_row(graph_db: Path) -> str | None:
    """The raw `graph_projection_state` 'dirty' value, or None if absent --
    used to prove `_delete_seeded_graph_rows` actually clears the row
    `_mark_projection_dirty` (graph.py:355) inserts, not just the derived
    `GraphStore.projection_dirty()` reading (which returns True on a bare
    missing row too, so it can't tell "never written" from "written then
    left behind")."""
    conn = sqlite3.connect(str(graph_db))
    try:
        row = conn.execute(
            "SELECT value FROM graph_projection_state WHERE key = 'dirty'"
        ).fetchone()
        return None if row is None else str(row[0])
    finally:
        conn.close()


@pytest.fixture(scope="module")
def graph_seeded(big_corpus) -> Iterator[None]:
    """`graph mindmap` renders `mem.navigator.export_json()`, which is built
    from the entity graph -- populated by `Memory.save()`'s extraction pass, a
    path `big_corpus` never runs (it upserts straight into `VecStore`). Left
    empty, the graph is empty and `graph mindmap` prints "Graph is empty" and
    returns *before* ever reaching the output-path logic under test, which
    would make that row of the table pass vacuously. `record_extraction` is
    the same storage primitive the extraction pass writes through, and it
    needs no MLX.

    `big_corpus.graph_db` is session-scoped and shared with modules this plan
    hasn't written yet (the MCP response-budget plan asserts `memo_graph` /
    `memo_graph_export` payload sizes against this same corpus) -- a residual
    entity/edge here would perturb that baseline in an unrelated task, days
    from this file. The `finally` runs the delete on every exit path,
    including a setup failure, even though `record_extraction` is already
    atomic (one transaction, rolled back whole on error) and so leaves
    nothing to undo in that case -- belt and suspenders, same reasoning as
    `big_corpus`'s own `mp.undo()` on its setup-failure path.
    """
    memory_id = seeded_id(0)
    try:
        _seed_graph_entities(big_corpus.graph_db, memory_id=memory_id)
        yield
    finally:
        _delete_seeded_graph_rows(big_corpus.graph_db, memory_id=memory_id)


def _mindmap_argv(dest: Path, tmp_path: Path) -> list[str]:
    return ["graph", "mindmap", _ENTITY_A, "--no-open", "-o", str(dest)]


def _federation_export_argv(dest: Path, tmp_path: Path) -> list[str]:
    return [
        "federation",
        "export",
        str(dest),
        "--principal",
        "conformance-principal",
        "--key-file",
        str(_key_file(tmp_path)),
    ]


def _backup_argv(dest: Path, tmp_path: Path) -> list[str]:
    return ["backup", "--out", str(dest)]


def _export_json_argv(dest: Path, tmp_path: Path) -> list[str]:
    return ["export", "json", str(dest)]


# label, argv-builder. Each builder takes (dest, tmp_path) and returns full argv
# built from each command's REAL signature -- verified by reading the source,
# not guessed from a flag-name pattern:
#   memo graph mindmap [ENTITY] [--depth N] [--node-cap N] [-o/--out PATH] [--open/--no-open]
#   memo federation export OUTPUT_PATH --principal P [--owner O] --key-file PATH
#   memo backup [--out PATH]                      (option on the group itself)
#   memo export json OUTPUT_PATH                   (positional, no flag)
OUTPUT_SURFACES: list[tuple[str, Callable[[Path, Path], list[str]]]] = [
    ("graph mindmap", _mindmap_argv),
    ("federation export", _federation_export_argv),
    ("backup", _backup_argv),
    ("export json", _export_json_argv),
]


def _assert_clean_and_on_topic(result, label: str) -> None:
    """A raw exception (anything Click's own `main()` did not convert into a
    `SystemExit` via `ClickException.show()`) is a traceback bug -- fail on
    it. A `UsageError` exit (code 2) means argv was wrong, not that the
    command's own destination handling ran -- fail on that too, rather than
    silently accepting it as "the surface passed."
    """
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"{label} raised {result.exception!r} instead of a clean error:\n{result.output}"
    )
    assert result.exit_code != 2, (
        f"{label} exited with a Click usage error (wrong flag/argument), which "
        f"never reached the destination-path handling this test checks:\n{result.output}"
    )


@pytest.mark.parametrize("label,argv_for", OUTPUT_SURFACES)
def test_symlinked_parent_is_accepted(big_corpus, graph_seeded, tmp_path, label, argv_for) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    dest = link / "out.dat"

    result = CliRunner().invoke(cli, argv_for(dest, tmp_path), env=_env(big_corpus))

    _assert_clean_and_on_topic(result, label)
    assert result.exit_code == 0, (
        f"{label} did not accept a symlinked parent directory cleanly:\n{result.output}"
    )
    # The write must actually land -- through the symlink and into the real
    # directory it points at -- not merely "not crash".
    assert (real / "out.dat").is_file(), (
        f"{label} exited 0 but never wrote through the symlinked parent:\n{result.output}"
    )


@pytest.mark.parametrize("label,argv_for", OUTPUT_SURFACES)
def test_missing_parent_gives_a_clean_error(
    big_corpus, graph_seeded, tmp_path, label, argv_for
) -> None:
    dest = tmp_path / "does" / "not" / "exist" / "out.dat"

    result = CliRunner().invoke(cli, argv_for(dest, tmp_path), env=_env(big_corpus))

    _assert_clean_and_on_topic(result, label)
    if result.exit_code == 0:
        # Some surfaces self-heal (mkdir -p the tree); if they claim success
        # the file must actually be there.
        assert dest.is_file(), (
            f"{label} exited 0 without writing the destination it claimed to "
            f"succeed on:\n{result.output}"
        )
    else:
        assert "Traceback" not in result.output


def test_graph_seeded_reverts_its_write_on_teardown(tmp_path) -> None:
    """`graph_seeded` mutates a session-scoped `graph.db` shared with tests
    this plan hasn't written yet -- a happy-path run of the parametrized
    tests above never proves the mutation is undone, since they only run
    *during* the fixture's lifetime. Drive the fixture generator directly
    (the same technique `test_fixture_cleanup.py` uses for `big_corpus`)
    against a throwaway `Config`, so this proves teardown without touching --
    or depending on -- the real session-scoped `big_corpus`.
    """
    from memo.config import Config

    cfg = Config(
        data_dir=tmp_path / "data",
        vault_path=tmp_path / "vault",
        state_dir=tmp_path / "state",
        reranker_enabled=False,
    )
    memory_id = seeded_id(0)

    gen = graph_seeded.__wrapped__(cfg)
    next(gen)  # run setup through the yield

    store = GraphStore(cfg.graph_db)
    try:
        assert store.top_entities(limit=10), "fixture setup did not write any entities"
        assert store.entity_memories(_ENTITY_A) == [memory_id]
    finally:
        store.close()
    assert _projection_dirty_row(cfg.graph_db) == "1", (
        "fixture setup did not mark the projection dirty -- the row this "
        "test proves teardown clears was never written in the first place"
    )

    with pytest.raises(StopIteration):
        next(gen)  # advance past the yield -- runs the fixture's `finally`

    store = GraphStore(cfg.graph_db)
    try:
        assert store.top_entities(limit=10) == [], "seeded entities survived teardown"
        assert store.entity_memories(_ENTITY_A) == [], "seeded edge survived teardown"
    finally:
        store.close()
    assert _projection_dirty_row(cfg.graph_db) is None, (
        "graph_seeded's dirty-projection row survived teardown -- it would "
        "perturb the MCP graph-export payload-size baseline this graph.db "
        "is shared with"
    )
