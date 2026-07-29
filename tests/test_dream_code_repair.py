"""Dream code-drift auto-repair — re-point dead refs with a UNIQUE candidate.

Covers the repair branch inside `_run_code_drift` (flag
MEMO_DREAM_CODE_REPAIR_ENABLED, default OFF):
- flag off → fully-drifted memories archive exactly as today (no repair,
  receipt `repaired` stays empty);
- unique rename candidate (same file, same kind, namespace preserved) → the
  ref is re-pointed in place (uri regenerated), the old ref is preserved in
  extra.code_refs_history, the memory is NOT archived, and the receipt
  records {id, from, to};
- unique move candidate (same name + kind in another file) → file_path
  repaired;
- 0 candidates or >1 candidates → archive as today (repair never guesses);
- a same-file sibling whose name bears NO similarity to the dead symbol is
  never a rename candidate (a deleted symbol must not be re-pointed at an
  unrelated neighbor), with or without a namespace on the dead qualified;
- partial drift (a live ref remains) → repair is never attempted;
- dry-run reports would-repair entries without writing;
- a failed persist lands in result["errors"] and neither repairs nor
  archives (retries next night).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from memo.cli_dream_passes import _run_code_drift
from memo.code_traceability import codegraph_repo_id, codegraph_uri
from memo.dream_flags import CODE_DRIFT_FLAG, CODE_REPAIR_FLAG
from memo.errors import StorageError

# --- synthetic codegraph.db (shape copied from test_dream_code_drift) -----------

SAVE_NODE = ("function:save", "function", "save", "memo.store.save", "src/memo/store.py", 10, 42)
LOAD_NODE = ("function:load", "function", "load", "memo.store.load", "src/memo/store.py", 50, 80)
MOVED_SAVE_NODE = (
    "function:queries.save",
    "function",
    "save",
    "memo.store.queries.save",
    "src/memo/store/queries.py",
    5,
    40,
)
FILE_NODE = ("file:src/memo/store.py", "file", "store.py", None, "src/memo/store.py", None, None)


def _graph_db(tmp_path: Path, nodes: list[tuple]) -> Path:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
            file_path TEXT, start_line INTEGER, end_line INTEGER
        );
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        """
    )
    conn.executemany("INSERT INTO nodes VALUES (?, ?, ?, ?, ?, ?, ?)", nodes)
    conn.commit()
    conn.close()
    return db


def _ref(
    file_path: str,
    label: str = "",
    qualified: str = "",
    kind: str = "function",
    repo_id: str = "",
) -> dict:
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


# The rename scenario: `old_save` no longer exists in store.py, but `save`
# (same file, same kind, same `memo.store.` namespace) does.
OLD_SAVE_REF = _ref("src/memo/store.py", label="old_save", qualified="memo.store.old_save")


def _save_with_refs(mock_memory, refs: list[dict], content: str):
    return mock_memory.save(content=content, type_="fact", extra={"code_refs": refs})


def _enable(monkeypatch) -> None:
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    monkeypatch.setenv(CODE_REPAIR_FLAG, "1")


# --- flag off → current archive behavior intact ----------------------------------


def test_flag_off_archives_as_today(mock_memory, tmp_path, monkeypatch):
    monkeypatch.setenv(CODE_DRIFT_FLAG, "1")
    monkeypatch.delenv(CODE_REPAIR_FLAG, raising=False)
    db = _graph_db(tmp_path, [SAVE_NODE, FILE_NODE])
    rec = _save_with_refs(mock_memory, [OLD_SAVE_REF], "old_save() handles retries")

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["repaired"] == []
    assert [e["id"] for e in res["outdated"]] == [rec.id]
    assert mock_memory.get(rec.id) is None


# --- unique candidate → repaired in place, never archived ------------------------


def test_unique_rename_candidate_repairs_in_place(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    db = _graph_db(tmp_path, [SAVE_NODE, FILE_NODE])
    rec = _save_with_refs(mock_memory, [OLD_SAVE_REF], "old_save() writes md first")
    repo = codegraph_repo_id(db.parent.parent)

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["status"] == "ok"
    assert res["scanned"] == 1
    assert res["outdated"] == []
    assert res["partial"] == []
    assert res["repaired"] == [
        {"id": rec.id, "from": OLD_SAVE_REF["uri"], "to": codegraph_uri(repo, "function:save")}
    ]
    got = mock_memory.get(rec.id)
    assert got is not None
    ref = got.extra["code_refs"][0]
    assert ref["uri"] == codegraph_uri(repo, "function:save")
    assert ref["repo_id"] == repo
    assert ref["stable_symbol_id"] == "function:save"
    assert ref["label"] == "save"
    assert ref["qualified_name"] == "memo.store.save"
    assert ref["file_path"] == "src/memo/store.py"
    assert ref["start_line"] == 10
    assert ref["end_line"] == 42
    assert ref["relation"] == "modified"  # untouched fields preserved
    history = got.extra["code_refs_history"]
    assert len(history) == 1
    assert history[0]["label"] == "old_save"
    assert history[0]["uri"] == OLD_SAVE_REF["uri"]


def test_unique_move_candidate_repairs_file_path(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    db = _graph_db(tmp_path, [MOVED_SAVE_NODE])
    ref = _ref("src/memo/store.py", label="save", qualified="memo.store.save")
    rec = _save_with_refs(mock_memory, [ref], "save() moved into the store subpackage")
    repo = codegraph_repo_id(db.parent.parent)

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["outdated"] == []
    assert res["repaired"] == [
        {"id": rec.id, "from": ref["uri"], "to": codegraph_uri(repo, "function:queries.save")}
    ]
    got = mock_memory.get(rec.id)
    assert got is not None
    repaired = got.extra["code_refs"][0]
    assert repaired["file_path"] == "src/memo/store/queries.py"
    assert repaired["qualified_name"] == "memo.store.queries.save"
    assert repaired["label"] == "save"
    assert repaired["start_line"] == 5


# --- 0 or >1 candidates → today's archive flow ------------------------------------


def test_two_candidates_archive_as_today(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    # Both `save` and `old_save2` are name-plausible renames of `old_save` in
    # the same `memo.store.` namespace: two candidates -> repair must not guess.
    old_save2 = (
        "function:old_save2",
        "function",
        "old_save2",
        "memo.store.old_save2",
        "src/memo/store.py",
        90,
        99,
    )
    db = _graph_db(tmp_path, [SAVE_NODE, old_save2, FILE_NODE])
    rec = _save_with_refs(mock_memory, [OLD_SAVE_REF], "old_save() handles retries")

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["repaired"] == []
    assert [e["id"] for e in res["outdated"]] == [rec.id]
    assert mock_memory.get(rec.id) is None


def test_deleted_symbol_with_single_unrelated_sibling_archives(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    # `old_save` was DELETED, not renamed: the only remaining function in the
    # file (`load`, same `memo.store.` namespace) bears no name similarity, so
    # it must never become the "rename" — re-pointing would silently corrupt
    # the citation toward an unrelated symbol.
    db = _graph_db(tmp_path, [LOAD_NODE, FILE_NODE])
    rec = _save_with_refs(mock_memory, [OLD_SAVE_REF], "old_save() handles retries")

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["repaired"] == []
    assert [e["id"] for e in res["outdated"]] == [rec.id]
    assert mock_memory.get(rec.id) is None


def test_deleted_symbol_without_namespace_never_repairs_to_any_sibling(
    mock_memory, tmp_path, monkeypatch
):
    _enable(monkeypatch)
    # qualified_name without a dot (top-level symbol): no namespace filter is
    # available, so the ONLY guard against "any sibling of the same kind" is
    # name similarity. `write_log` must not repair a dead `parse_config` ref.
    write_log = ("function:write_log", "function", "write_log", "write_log", "src/utils.py", 1, 10)
    db = _graph_db(tmp_path, [write_log])
    ref = _ref("src/utils.py", label="parse_config", qualified="parse_config")
    rec = _save_with_refs(mock_memory, [ref], "parse_config() reads the ini")

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["repaired"] == []
    assert [e["id"] for e in res["outdated"]] == [rec.id]
    assert mock_memory.get(rec.id) is None


def test_zero_candidates_archive_as_today(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    db = _graph_db(tmp_path, [SAVE_NODE, FILE_NODE])
    ref = _ref("src/memo/gone.py", label="gone", qualified="memo.gone.gone")
    rec = _save_with_refs(mock_memory, [ref], "notes about a deleted module")

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["repaired"] == []
    assert [e["id"] for e in res["outdated"]] == [rec.id]
    assert mock_memory.get(rec.id) is None


# --- partial drift → repair never attempted ---------------------------------------


def test_partial_drift_never_repairs(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    db = _graph_db(tmp_path, [SAVE_NODE, FILE_NODE])
    live = _ref("src/memo/store.py", label="save", qualified="memo.store.save")
    # The dead ref would have `save` as a unique rename candidate — but the
    # memory still has a live ref, so it is partial, and partial never repairs.
    dead = _ref("src/memo/store.py", label="old_load", qualified="memo.store.old_load")
    rec = _save_with_refs(mock_memory, [live, dead], "save() plus a renamed helper")

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["repaired"] == []
    assert [e["id"] for e in res["partial"]] == [rec.id]
    got = mock_memory.get(rec.id)
    assert got is not None
    assert got.extra["code_refs"][1]["label"] == "old_load"
    assert "code_refs_history" not in got.extra


# --- dry-run → would-repair reported, nothing written ------------------------------


def test_dry_run_reports_would_repair_without_writing(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    db = _graph_db(tmp_path, [SAVE_NODE, FILE_NODE])
    rec = _save_with_refs(mock_memory, [OLD_SAVE_REF], "old_save() writes md first")
    repo = codegraph_repo_id(db.parent.parent)

    res = _run_code_drift(mock_memory, db_path=db, dry_run=True)

    assert res["repaired"] == [
        {"id": rec.id, "from": OLD_SAVE_REF["uri"], "to": codegraph_uri(repo, "function:save")}
    ]
    assert res["outdated"] == []
    got = mock_memory.get(rec.id)
    assert got is not None
    assert got.extra["code_refs"][0]["label"] == "old_save"
    assert "code_refs_history" not in got.extra


# --- failed persist → error recorded, neither repaired nor archived ----------------


def test_failed_repair_persist_is_reported_not_archived(mock_memory, tmp_path, monkeypatch):
    _enable(monkeypatch)
    db = _graph_db(tmp_path, [SAVE_NODE, FILE_NODE])
    rec = _save_with_refs(mock_memory, [OLD_SAVE_REF], "old_save() writes md first")

    def _boom(*_args, **_kwargs):
        raise StorageError("disk full")

    monkeypatch.setattr(mock_memory, "update", _boom)

    res = _run_code_drift(mock_memory, db_path=db)

    assert res["repaired"] == []
    assert res["outdated"] == []
    assert any("repair failed" in err and rec.id in err for err in res.get("errors", []))
    got = mock_memory.get(rec.id)
    assert got is not None
    assert got.extra["code_refs"][0]["label"] == "old_save"
