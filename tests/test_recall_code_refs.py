"""MEMO_RECALL_CODE_REFS_ENABLED: verified '↳ code:' citation lines per hit."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

from memo import codegraph_loader


def _seed_codegraph_db(db_path: Path) -> None:
    """Minimal codegraph.db mirroring the real nodes/edges shape (file_path cols)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE nodes (
            id TEXT PRIMARY KEY, kind TEXT, name TEXT, qualified_name TEXT,
            file_path TEXT, start_line INTEGER, end_line INTEGER
        );
        CREATE INDEX idx_nodes_file_path ON nodes (file_path);
        CREATE TABLE edges (source TEXT, target TEXT, kind TEXT);
        INSERT INTO nodes VALUES
            ('function:render', 'function', 'render_recall_context',
             'memo.recall_logic.render_recall_context',
             'src/memo/recall_logic.py', 215, 345),
            ('file:recall', 'file', 'recall_logic.py',
             'src/memo/recall_logic.py', 'src/memo/recall_logic.py', NULL, NULL);
        """
    )
    conn.commit()
    conn.close()


def _hit(**kw):
    base = dict(
        id="a1b2c3d4" * 4,
        title="Título",
        type="decision",
        tags=[],
        created="2026-03-05T10:00:00",
        updated="2026-03-07T10:00:00",
        body="cuerpo suficientemente largo para renderizar " * 3,
        score=0.8,
        extra={},
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _ref(path: str = "src/memo/recall_logic.py", line: int | None = 215, **kw):
    base: dict = dict(file_path=path, start_line=line)
    base.update(kw)
    return base


def _render(hits, **kw):
    from memo.recall_logic import render_recall_context

    return render_recall_context(hits, [], turn=1, body_chars=200, token_budget=0, **kw)


def test_flag_off_output_identical_and_db_never_opened(monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_CODE_REFS_ENABLED", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("sqlite must not be touched with the flag off")

    monkeypatch.setattr(sqlite3, "connect", _boom)
    out = _render([_hit(extra={"code_refs": [_ref()]})])
    bare = _render([_hit()])
    assert "↳ code" not in out
    assert out == bare


def test_flag_on_existing_ref_renders_vigente(monkeypatch, tmp_path: Path):
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = _render([_hit(extra={"code_refs": [_ref()]})])
    assert "  ↳ code: src/memo/recall_logic.py:215 (vigente)" in out


def test_flag_on_label_verifies_against_node_name(monkeypatch, tmp_path: Path):
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = _render([_hit(extra={"code_refs": [_ref(label="render_recall_context")]})])
    assert "(vigente)" in out
    # A renamed symbol must not verify against a surviving file_path alone.
    out = _render([_hit(extra={"code_refs": [_ref(label="renamed_function")]})])
    assert "(vigente)" not in out
    assert "(desaparecido)" in out


def test_flag_on_db_missing_renders_no_verificado(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing" / "codegraph.db")
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = _render([_hit(extra={"code_refs": [_ref()]})])
    assert "  ↳ code: src/memo/recall_logic.py:215 (no verificado)" in out
    assert "(vigente)" not in out


def test_flag_on_ref_gone_renders_desaparecido(monkeypatch, tmp_path: Path):
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = _render([_hit(extra={"code_refs": [_ref(path="src/memo/deleted_module.py", line=10)]})])
    assert "  ↳ code: src/memo/deleted_module.py:10 (desaparecido)" in out
    assert "(vigente)" not in out


def test_sqlite_error_degrades_to_no_verificado(monkeypatch, tmp_path: Path):
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    db.write_text("this is definitely not a sqlite database " * 10)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = _render([_hit(extra={"code_refs": [_ref()]})])  # must not raise
    assert "  ↳ code: src/memo/recall_logic.py:215 (no verificado)" in out
    assert "(vigente)" not in out


def test_caps_two_refs_per_memory_four_per_render(monkeypatch, tmp_path: Path):
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    hit1 = _hit(
        id="1" * 32,
        extra={"code_refs": [_ref(line=11), _ref(line=12), _ref(line=13)]},
    )
    hit2 = _hit(id="2" * 32, title="Otro", extra={"code_refs": [_ref(line=21), _ref(line=22)]})
    hit3 = _hit(id="3" * 32, title="Tercero", extra={"code_refs": [_ref(line=31)]})
    out = _render([hit1, hit2, hit3])
    assert out.count("↳ code:") == 4  # 2 (hit1 capped) + 2 (hit2) + 0 (hit3, render cap)
    assert ":13 " not in out  # hit1's third ref dropped by the per-memory cap
    assert ":31 " not in out  # hit3's ref dropped by the per-render cap


def test_non_dict_and_pathless_refs_are_skipped(monkeypatch, tmp_path: Path):
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    refs = ["codegraph://abc/function:render", {"uri": "codegraph://abc/x"}, _ref()]
    out = _render([_hit(extra={"code_refs": refs})])
    assert out.count("↳ code:") == 1  # only the entry with a file_path renders


def test_pipx_like_missing_checkout_db_discovers_project_db(monkeypatch, tmp_path: Path):
    """pipx/uv-tool runtime: CODEGRAPH_DB falls inside site-packages and does
    not exist — cwd discovery (same resolution as codegraph_loader.load())
    must still find the project's own index."""
    monkeypatch.setattr(
        codegraph_loader,
        "CODEGRAPH_DB",
        tmp_path / "site-packages" / ".codegraph" / "codegraph.db",
    )
    project = tmp_path / "project"
    _seed_codegraph_db(project / ".codegraph" / "codegraph.db")
    monkeypatch.chdir(project)
    monkeypatch.delenv("MEMO_CODEGRAPH_DISCOVERY", raising=False)  # default on
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = _render([_hit(extra={"code_refs": [_ref()]})])
    assert "  ↳ code: src/memo/recall_logic.py:215 (vigente)" in out


def test_ref_from_foreign_repo_degrades_to_no_verificado(monkeypatch, tmp_path: Path):
    """A ref whose codegraph:// uri names another repo's graph must never be
    verified against the locally resolved DB."""
    from memo.code_traceability import codegraph_repo_id, codegraph_uri

    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    local_id = codegraph_repo_id(tmp_path)  # db.parent.parent, no git → path hash
    foreign_id = "f" * 16
    assert foreign_id != local_id

    ref = _ref(uri=codegraph_uri(foreign_id, "function:render"))
    out = _render([_hit(extra={"code_refs": [ref]})])
    assert "  ↳ code: src/memo/recall_logic.py:215 (no verificado)" in out
    assert "(vigente)" not in out

    # The same ref stamped with the resolved DB's own repo_id verifies normally.
    ref = _ref(uri=codegraph_uri(local_id, "function:render"))
    out = _render([_hit(extra={"code_refs": [ref]})])
    assert "(vigente)" in out


def test_file_kind_ref_with_divergent_label_verifies_by_path(monkeypatch, tmp_path: Path):
    """kind='file' refs are judged on file_path alone — a label that differs
    from nodes.name must not read as drift (mirrors the dream pass's
    _code_ref_exists semantics)."""
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    ref = _ref(kind="file", label="algún label que no es el node name")
    out = _render([_hit(extra={"code_refs": [ref]})])
    assert "  ↳ code: src/memo/recall_logic.py:215 (vigente)" in out
    assert "(desaparecido)" not in out


def test_symbol_ref_matches_qualified_name(monkeypatch, tmp_path: Path):
    """Symbol refs match name OR qualified_name, like the dream pass."""
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    # qualified_name-only ref: node name differs, qualified_name matches.
    ref = _ref(qualified_name="memo.recall_logic.render_recall_context")
    out = _render([_hit(extra={"code_refs": [ref]})])
    assert "(vigente)" in out

    # label mismatch + matching qualified_name still verifies (OR semantics).
    ref = _ref(
        label="renamed_function",
        qualified_name="memo.recall_logic.render_recall_context",
    )
    out = _render([_hit(extra={"code_refs": [ref]})])
    assert "(vigente)" in out


# --- format wiring: full AND balanced render the line, compact never does -------


def _render_balanced(hits):
    from memo.recall_logic import render_recall_balanced

    return render_recall_balanced(hits, token_budget=0, turn=1)


def test_balanced_format_renders_code_refs(monkeypatch, tmp_path: Path):
    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = _render_balanced([_hit(extra={"code_refs": [_ref()]})])
    assert "  ↳ code: src/memo/recall_logic.py:215 (vigente)" in out


def test_balanced_format_flag_off_unchanged_and_db_never_opened(monkeypatch):
    monkeypatch.delenv("MEMO_RECALL_CODE_REFS_ENABLED", raising=False)

    def _boom(*args, **kwargs):
        raise AssertionError("sqlite must not be touched with the flag off")

    monkeypatch.setattr(sqlite3, "connect", _boom)
    out = _render_balanced([_hit(extra={"code_refs": [_ref()]})])
    bare = _render_balanced([_hit()])
    assert "↳ code" not in out
    assert out == bare


def test_compact_format_never_renders_code_refs(monkeypatch, tmp_path: Path):
    """Compact stays one-line-per-hit by design, even with the flag ON."""
    from memo.recall_logic import render_recall_compact

    db = tmp_path / ".codegraph" / "codegraph.db"
    _seed_codegraph_db(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.setenv("MEMO_RECALL_CODE_REFS_ENABLED", "1")

    out = render_recall_compact([_hit(extra={"code_refs": [_ref()]})], token_budget=0)
    assert "↳ code" not in out
