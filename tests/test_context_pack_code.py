"""context-pack --code — graph-neighborhood section in the context pack.

Covers:
- --code <symbol>: adds a '## Código relacionado' section with the 1-hop
  neighbors (name — file_path:start_line) and the memories citing them.
- --code <path containing '/'>: the file's defined symbols seed the walk.
- without --code: payload identical to today (no section, no extra key).
- missing codegraph DB / unknown anchor: section silently omitted, exit 0.
- the section is budgeted with the pack (_trim_to_budget makes room for it).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.context_pack import DEFAULT_BUDGET_CHARS, ContextPack

# --- synthetic codegraph.db (shape copied from test_dream_code_drift) -----------

_NODES = [
    # (id, kind, name, qualified_name, file_path, start_line, end_line)
    ("function:alpha", "function", "alpha", "memo.a.alpha", "src/memo/a.py", 1, 5),
    ("function:beta", "function", "beta", "memo.b.beta", "src/memo/b.py", 10, 15),
    ("function:gamma", "function", "gamma", "memo.c.gamma", "src/memo/c.py", 20, 25),
    ("file:src/memo/a.py", "file", "a.py", None, "src/memo/a.py", None, None),
]

_EDGES = [
    # alpha → beta → gamma over 'calls'; the 'contains' edge never traverses.
    ("function:alpha", "function:beta", "calls"),
    ("function:beta", "function:gamma", "calls"),
    ("file:src/memo/a.py", "function:alpha", "contains"),
]

_CITING_ID = "a1b2c3d4e5f60718"


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
    db = tmp_path / "proj" / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_graph(db)
    return db


def _meta_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row  # mirrors mem.store._conn
    conn.execute("CREATE TABLE meta (id TEXT PRIMARY KEY, type TEXT, title TEXT, extra_json TEXT)")
    refs = [
        {
            "file_path": "src/memo/b.py",
            "kind": "function",
            "label": "beta",
            "qualified_name": "memo.b.beta",
            "repo_id": "",
        }
    ]
    conn.execute(
        "INSERT INTO meta VALUES (?, ?, ?, ?)",
        (_CITING_ID, "fact", "beta returns copies", json.dumps({"code_refs": refs})),
    )
    conn.commit()
    return conn


def _hit(body: str = "The current state is documented here.") -> SimpleNamespace:
    return SimpleNamespace(
        id="abc12345deadbeef",
        title="Current status",
        type="note",
        body=body,
        score=0.91,
        tags=[],
        extra={},
    )


class _FakeMemory:
    def __init__(self, hits: list[SimpleNamespace]) -> None:
        self._hits = hits
        self.store = SimpleNamespace(_conn=_meta_conn())

    def search(self, *args, **kwargs):
        return list(self._hits)


def _env(tmp_path: Path, codegraph_db: Path | None) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_CONTEXT_PACK": "1",
        "MEMO_CODEGRAPH_DISCOVERY": "0",
        "MEMO_CODEGRAPH_DB": str(codegraph_db or tmp_path / "missing" / "codegraph.db"),
        "MEMO_EMBEDDER_DIMS": "4",
        "MEMO_RERANKER_ENABLED": "0",
    }


def _invoke(
    tmp_path: Path,
    monkeypatch,
    args: list[str],
    *,
    codegraph_db: Path | None = None,
    hits: list[SimpleNamespace] | None = None,
):
    fake = _FakeMemory(hits if hits is not None else [_hit()])
    monkeypatch.setattr("memo.cli_search._get_memory", lambda cfg: fake)
    try:
        return CliRunner().invoke(cli, args, env=_env(tmp_path, codegraph_db))
    finally:
        fake.store._conn.close()


def test_code_symbol_section_lists_neighbors_and_citing_memory(
    tmp_path, graph_db, monkeypatch
) -> None:
    result = _invoke(
        tmp_path,
        monkeypatch,
        ["context-pack", "how does alpha work?", "--code", "alpha", "--json"],
        codegraph_db=graph_db,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    section = payload["code_context"]
    assert section.startswith("## Código relacionado")
    assert "alpha — src/memo/a.py:1" in section
    assert "beta — src/memo/b.py:10" in section
    assert "gamma" not in section  # two hops away — the walk is 1 hop
    assert f"[{_CITING_ID[:8]}] beta returns copies" in section


def test_code_path_anchor_seeds_from_file_symbols(tmp_path, graph_db, monkeypatch) -> None:
    result = _invoke(
        tmp_path,
        monkeypatch,
        ["context-pack", "q", "--code", "src/memo/b.py", "--json"],
        codegraph_db=graph_db,
    )

    assert result.exit_code == 0, result.output
    section = json.loads(result.output)["code_context"]
    # beta is defined in the file; alpha and gamma are its 1-hop neighbors.
    assert "beta — src/memo/b.py:10" in section
    assert "alpha — src/memo/a.py:1" in section
    assert "gamma — src/memo/c.py:20" in section


def test_without_code_pack_is_unchanged(tmp_path, graph_db, monkeypatch) -> None:
    result = _invoke(
        tmp_path,
        monkeypatch,
        ["context-pack", "q", "--json"],
        codegraph_db=graph_db,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {
        "question",
        "summary",
        "current_facts",
        "supporting_context",
        "stale_or_conflicting",
        "omissions",
    }
    assert "Código relacionado" not in result.output


def test_missing_db_omits_section_silently(tmp_path, monkeypatch) -> None:
    result = _invoke(
        tmp_path,
        monkeypatch,
        ["context-pack", "q", "--code", "alpha", "--json"],
        codegraph_db=None,
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert "code_context" not in payload
    assert payload["question"] == "q"


def test_unknown_symbol_omits_section(tmp_path, graph_db, monkeypatch) -> None:
    result = _invoke(
        tmp_path,
        monkeypatch,
        ["context-pack", "q", "--code", "no_such_symbol", "--json"],
        codegraph_db=graph_db,
    )

    assert result.exit_code == 0, result.output
    assert "code_context" not in json.loads(result.output)


def test_panel_output_renders_section(tmp_path, graph_db, monkeypatch) -> None:
    result = _invoke(
        tmp_path,
        monkeypatch,
        ["context-pack", "q", "--code", "alpha"],
        codegraph_db=graph_db,
    )

    assert result.exit_code == 0, result.output
    assert "Código relacionado" in result.output


def test_section_is_budgeted_with_the_pack(tmp_path, graph_db, monkeypatch) -> None:
    # A single oversized current fact makes _trim_to_budget fill the budget
    # tightly, so pack + section only fit if the section reserved its room.
    result = _invoke(
        tmp_path,
        monkeypatch,
        ["context-pack", "q", "--code", "alpha", "--snippet-chars", "10000", "--json"],
        codegraph_db=graph_db,
        hits=[_hit(body="A" * 10000)],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    section = payload["code_context"]
    pack_prompt = ContextPack(
        payload["question"],
        payload["summary"],
        payload["current_facts"],
        payload["supporting_context"],
        payload["stale_or_conflicting"],
        payload["omissions"],
    ).to_prompt()
    assert len(pack_prompt) + 2 + len(section) <= DEFAULT_BUDGET_CHARS
    # The pack itself was trimmed near the reserved cap, not far below it.
    assert len(pack_prompt) >= DEFAULT_BUDGET_CHARS - len(section) - 200
