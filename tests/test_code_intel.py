"""code_intel engine — shared read-only joins between codegraph and memories.

Covers:
- open_graph: explicit path → (read-only connection, db repo_id); missing DB →
  None; the connection refuses writes (mode=ro).
- ref_status: the ONE verification semantics shared by recall and dream —
  'vigente' / 'desaparecido' on file/symbol match, None (unverifiable) for
  non-dict refs, refs without a file_path, refs minted against another repo
  (explicit repo_id field, or codegraph:// uri host when the field is absent),
  and sqlite errors. A present-but-empty repo_id field means "no repo claim"
  and stays verifiable even when the uri names another repo (parity with the
  dream fixtures).
- memories_citing: JSON1 join over meta.extra_json code_refs by exact
  file_path or symbol (label / qualified_name); reference tier excluded;
  corrupt extra_json ignored; [] on error.
- symbols_for_files / neighbors: undirected expansion over traversable edge
  kinds only ('contains' never traverses), seed included, hops clamped to 2.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from memo.code_intel import (
    memories_citing,
    neighbors,
    open_graph,
    ref_repo_claim,
    ref_status,
    symbols_for_files,
)
from memo.code_traceability import codegraph_repo_id

# --- synthetic codegraph.db (shape copied from test_dream_code_drift) -----------

_NODES = [
    # (id, kind, name, qualified_name, file_path, start_line, end_line)
    ("function:save", "function", "save", "memo.store.save", "src/memo/store.py", 10, 42),
    ("file:src/memo/store.py", "file", "store.py", None, "src/memo/store.py", None, None),
    ("function:alpha", "function", "alpha", "memo.a.alpha", "src/memo/a.py", 1, 5),
    ("function:beta", "function", "beta", "memo.b.beta", "src/memo/b.py", 1, 5),
    ("function:gamma", "function", "gamma", "memo.c.gamma", "src/memo/c.py", 1, 5),
    ("function:delta", "function", "delta", "memo.d.delta", "src/memo/d.py", 1, 5),
    ("file:src/memo/a.py", "file", "a.py", None, "src/memo/a.py", None, None),
]

_EDGES = [
    # a → b → c → d over 'calls'; the 'contains' edge must never traverse.
    ("function:alpha", "function:beta", "calls"),
    ("function:beta", "function:gamma", "calls"),
    ("function:gamma", "function:delta", "calls"),
    ("file:src/memo/a.py", "function:alpha", "contains"),
]


def _seed_graph(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
            file_path TEXT, start_line INTEGER, end_line INTEGER
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        """
    )
    conn.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", _NODES)
    conn.executemany("INSERT INTO edges VALUES (?, ?, ?)", _EDGES)
    conn.commit()
    conn.close()


@pytest.fixture
def graph_db(tmp_path: Path) -> Path:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_graph(db)
    return db


@pytest.fixture
def engine(graph_db: Path):
    opened = open_graph(graph_db)
    assert opened is not None
    conn, db_repo_id = opened
    yield conn, db_repo_id
    conn.close()


def _ref(
    file_path: str,
    label: str = "",
    qualified: str = "",
    kind: str = "function",
    repo_id: str = "",
) -> dict:
    # repo_id defaults to "" (explicit no-repo-claim → judged on file/symbol);
    # a non-empty repo_id must match the DB's repo or the ref is unverifiable.
    return {
        "uri": f"codegraph://{repo_id or 'testrepo'}/{label or file_path}",
        "repo_id": repo_id,
        "stable_symbol_id": label or file_path,
        "kind": kind,
        "label": label,
        "qualified_name": qualified,
        "file_path": file_path,
        "relation": "modified",
        "confidence": 0.95,
    }


# --- open_graph ------------------------------------------------------------------


def test_open_graph_returns_ro_connection_and_repo_id(graph_db):
    opened = open_graph(graph_db)

    assert opened is not None
    conn, db_repo_id = opened
    try:
        assert db_repo_id == codegraph_repo_id(graph_db.parent.parent)
        row = conn.execute("SELECT name FROM nodes WHERE id = 'function:save'").fetchone()
        assert row[0] == "save"
        # mode=ro: the engine can never write the codegraph index.
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM nodes")
    finally:
        conn.close()


def test_open_graph_missing_db_returns_none(tmp_path):
    assert open_graph(tmp_path / "nowhere" / "codegraph.db") is None


# --- ref_status ------------------------------------------------------------------


def test_live_symbol_ref_is_vigente(engine):
    conn, db_repo_id = engine
    ref = _ref("src/memo/store.py", label="save", qualified="memo.store.save")

    assert ref_status(conn, ref, db_repo_id) == "vigente"


def test_renamed_symbol_is_desaparecido(engine):
    conn, db_repo_id = engine
    ref = _ref("src/memo/store.py", label="old_save", qualified="memo.store.old_save")

    assert ref_status(conn, ref, db_repo_id) == "desaparecido"


def test_missing_file_is_desaparecido(engine):
    conn, db_repo_id = engine
    ref = _ref("src/memo/gone.py", label="gone", qualified="memo.gone.gone")

    assert ref_status(conn, ref, db_repo_id) == "desaparecido"


def test_file_kind_judged_on_path_alone(engine):
    conn, db_repo_id = engine
    # label deliberately diverges from the node name: file refs carry no
    # symbol, so existence is judged on file_path alone.
    ref = _ref("src/memo/store.py", label="renamed.py", kind="file")

    assert ref_status(conn, ref, db_repo_id) == "vigente"


def test_foreign_repo_id_field_is_unverifiable(engine):
    conn, db_repo_id = engine
    ref = _ref(
        "src/synapse/router.py",
        label="route",
        qualified="synapse.router.route",
        repo_id="feedfacefeedface",
    )

    assert ref_status(conn, ref, db_repo_id) is None


def test_matching_repo_id_is_verified(engine):
    conn, db_repo_id = engine
    ref = _ref("src/memo/store.py", label="save", repo_id=db_repo_id)

    assert ref_status(conn, ref, db_repo_id) == "vigente"


def test_ref_without_file_path_is_unverifiable(engine):
    conn, db_repo_id = engine

    assert ref_status(conn, _ref("", label="mystery"), db_repo_id) is None


def test_non_dict_ref_is_unverifiable(engine):
    conn, db_repo_id = engine

    assert ref_status(conn, "codegraph://testrepo/save", db_repo_id) is None


def test_foreign_uri_without_repo_id_field_is_unverifiable(engine):
    conn, db_repo_id = engine
    # No repo_id field at all → the codegraph:// uri host is the repo claim.
    ref = _ref("src/memo/store.py", label="save")
    del ref["repo_id"]

    assert ref_status(conn, ref, db_repo_id) is None


def test_empty_repo_id_field_wins_over_foreign_uri(engine):
    conn, db_repo_id = engine
    # Parity pin for the dream fixtures: an explicit repo_id="" means "no repo
    # claim" and stays verifiable even though the uri names another repo.
    ref = _ref("src/memo/store.py", label="save", qualified="memo.store.save")
    assert ref["repo_id"] == "" and ref["uri"].startswith("codegraph://testrepo/")

    assert ref_status(conn, ref, db_repo_id) == "vigente"


def test_ref_repo_claim_field_wins_then_uri_then_empty():
    # The ONE extraction every consumer shares: explicit repo_id field first
    # (empty string = "no claim"), else the codegraph:// uri host, else ''.
    assert ref_repo_claim({"repo_id": "aaa", "uri": "codegraph://bbb/x"}) == "aaa"
    assert ref_repo_claim({"repo_id": "", "uri": "codegraph://bbb/x"}) == ""
    assert ref_repo_claim({"uri": "codegraph://bbb/x"}) == "bbb"
    assert ref_repo_claim({"uri": "https://example.com/x"}) == ""
    assert ref_repo_claim({}) == ""
    assert ref_repo_claim("not-a-dict") == ""


def test_sqlite_error_degrades_to_unverifiable(engine):
    _conn, db_repo_id = engine
    empty = sqlite3.connect(":memory:")  # no nodes table → OperationalError
    try:
        assert ref_status(empty, _ref("src/memo/store.py", label="save"), db_repo_id) is None
    finally:
        empty.close()


# --- memories_citing --------------------------------------------------------------

_SAVE_REFS = [_ref("src/memo/store.py", label="save", qualified="memo.store.save")]
_OTHER_REFS = [_ref("src/other/module.py", label="other", qualified="pkg.other")]


def _meta_conn(rows: list[tuple[str, str, str, str]]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # mirrors mem.store._conn
    conn.execute("CREATE TABLE meta (id TEXT PRIMARY KEY, type TEXT, title TEXT, extra_json TEXT)")
    conn.executemany("INSERT INTO meta VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    return conn


@pytest.fixture
def store_conn():
    conn = _meta_conn(
        [
            ("m1", "fact", "save writes md first", json.dumps({"code_refs": _SAVE_REFS})),
            ("m2", "reference", "vault chunk", json.dumps({"code_refs": _SAVE_REFS})),
            ("m3", "fact", "corrupt extra", "{not json"),
            ("m4", "note", "unrelated", json.dumps({"code_refs": _OTHER_REFS})),
            ("m5", "fact", "no refs", json.dumps({})),
            ("m6", "fact", "scalar refs", json.dumps({"code_refs": "oops"})),
            ("m7", "fact", "non-dict entries", json.dumps({"code_refs": ["just-a-string"]})),
        ]
    )
    yield conn
    conn.close()


def test_citing_by_path(store_conn):
    out = memories_citing(store_conn, paths={"src/memo/store.py"})

    assert [m["id"] for m in out] == ["m1"]
    assert out[0]["title"] == "save writes md first"
    assert out[0]["refs"] == _SAVE_REFS


def test_citing_by_label_symbol(store_conn):
    out = memories_citing(store_conn, symbols={"save"})

    assert [m["id"] for m in out] == ["m1"]


def test_citing_by_qualified_name(store_conn):
    out = memories_citing(store_conn, symbols={"memo.store.save"})

    assert [m["id"] for m in out] == ["m1"]


def test_reference_tier_is_excluded(store_conn):
    out = memories_citing(store_conn, paths={"src/memo/store.py"}, symbols={"save"})

    assert "m2" not in [m["id"] for m in out]


def test_no_criteria_returns_empty(store_conn):
    assert memories_citing(store_conn) == []


def test_limit_caps_results(store_conn):
    out = memories_citing(store_conn, paths={"src/memo/store.py", "src/other/module.py"}, limit=1)

    assert len(out) == 1


def test_error_returns_empty():
    conn = sqlite3.connect(":memory:")  # no meta table
    try:
        assert memories_citing(conn, paths={"src/memo/store.py"}) == []
    finally:
        conn.close()


# --- symbols_for_files ------------------------------------------------------------


def test_symbols_for_files_excludes_file_nodes(engine):
    conn, _ = engine

    assert symbols_for_files(conn, ["src/memo/store.py"]) == {"save"}
    assert symbols_for_files(conn, ["src/memo/a.py", "src/memo/b.py"]) == {"alpha", "beta"}


def test_symbols_for_files_empty_input(engine):
    conn, _ = engine

    assert symbols_for_files(conn, []) == set()


# --- neighbors ----------------------------------------------------------------------


def test_neighbors_one_hop_includes_seed(engine):
    conn, _ = engine

    assert neighbors(conn, {"alpha"}, hops=1) == {"alpha", "beta"}


def test_neighbors_two_hops(engine):
    conn, _ = engine

    assert neighbors(conn, {"alpha"}, hops=2) == {"alpha", "beta", "gamma"}


def test_neighbors_hops_clamped_to_two(engine):
    conn, _ = engine

    assert neighbors(conn, {"alpha"}, hops=5) == {"alpha", "beta", "gamma"}


def test_neighbors_expansion_is_undirected(engine):
    conn, _ = engine

    assert neighbors(conn, {"gamma"}, hops=1) == {"beta", "gamma", "delta"}


def test_neighbors_contains_edge_never_traverses(engine):
    conn, _ = engine

    assert neighbors(conn, {"a.py"}, hops=2) == {"a.py"}


def test_neighbors_empty_seed(engine):
    conn, _ = engine

    assert neighbors(conn, set(), hops=2) == set()
