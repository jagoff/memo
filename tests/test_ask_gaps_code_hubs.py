"""ask-gaps code-hub gaps — top call-magnets in the codegraph nobody documented.

Covers:
- code_hub_gaps: hub without a citing memory → "hub sin memoria" line, ordered
  by incoming call-edges, src/ nodes only; hub WITH a citing memory → absent;
  ``top`` caps the candidate pool BEFORE the citation filter; flag off → []
  with zero graph work; missing index → [].
- briefing integration: at most ONE hub line rides ``briefing_lines``, read
  from the nightly dream receipt (``code_drift.hub_gaps``) — SessionStart does
  ZERO graph work; flag off / no receipt → no line.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from memo import ask_gaps as ag

# --- synthetic codegraph.db (shape copied from test_dream_code_drift) -----------

_NODES = [
    # (id, kind, name, qualified_name, file_path, start_line, end_line)
    (
        "function:flag_bool",
        "function",
        "flag_bool",
        "memo.flags.flag_bool",
        "src/memo/flags.py",
        10,
        20,
    ),
    ("function:save", "function", "save", "memo.store.save", "src/memo/store.py", 30, 60),
    # Non-src hub: receives calls but must never surface (src/% anchor).
    ("function:helper", "function", "helper", "tools.helper.helper", "tools/helper.py", 1, 5),
    ("function:c1", "function", "c1", "memo.x.c1", "src/memo/x.py", 1, 2),
    ("function:c2", "function", "c2", "memo.x.c2", "src/memo/x.py", 3, 4),
    ("function:c3", "function", "c3", "memo.x.c3", "src/memo/x.py", 5, 6),
]

_EDGES = [
    # flag_bool: 3 incoming calls; save: 1; helper (non-src): 2.
    ("function:c1", "function:flag_bool", "calls"),
    ("function:c2", "function:flag_bool", "calls"),
    ("function:c3", "function:flag_bool", "calls"),
    ("function:c1", "function:save", "calls"),
    ("function:c1", "function:helper", "calls"),
    ("function:c2", "function:helper", "calls"),
]

_TOP_HUB_LINE = "hub sin memoria: flag_bool (3 callers) — src/memo/flags.py"


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
def graph_db(tmp_path: Path, monkeypatch) -> Path:
    """Seeded index, pinned via MEMO_CODEGRAPH_DB (cwd discovery is off in tests)."""
    db = tmp_path / "proj" / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_graph(db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DB", str(db))
    return db


def _cite(mem, label: str, qualified: str, file_path: str):
    return mem.save(
        content=f"{label} is documented",
        type_="fact",
        extra={
            "code_refs": [
                {
                    "kind": "function",
                    "label": label,
                    "qualified_name": qualified,
                    "file_path": file_path,
                }
            ]
        },
    )


# --- code_hub_gaps ---------------------------------------------------------------


def test_uncited_hubs_are_gaps_ordered_by_callers(mock_memory, graph_db):
    gaps = ag.code_hub_gaps(mock_memory)

    assert gaps == [
        _TOP_HUB_LINE,
        "hub sin memoria: save (1 callers) — src/memo/store.py",
    ]


def test_non_src_hub_never_surfaces(mock_memory, graph_db):
    assert not any("helper" in g for g in ag.code_hub_gaps(mock_memory))


def test_cited_hub_is_not_a_gap(mock_memory, graph_db):
    _cite(mock_memory, "flag_bool", "memo.flags.flag_bool", "src/memo/flags.py")

    gaps = ag.code_hub_gaps(mock_memory)

    assert not any("flag_bool" in g for g in gaps)
    assert gaps == ["hub sin memoria: save (1 callers) — src/memo/store.py"]


def test_top_caps_candidates_before_citation_filter(mock_memory, graph_db):
    assert ag.code_hub_gaps(mock_memory, top=1) == [_TOP_HUB_LINE]
    # The single candidate is cited -> nothing left, even with `save` uncited.
    _cite(mock_memory, "flag_bool", "memo.flags.flag_bool", "src/memo/flags.py")
    assert ag.code_hub_gaps(mock_memory, top=1) == []


def test_flag_off_returns_empty_with_zero_graph_work(mock_memory, graph_db, monkeypatch):
    from memo import code_intel

    monkeypatch.setenv("MEMO_GAPS_CODE_HUBS", "0")

    def _boom(*args, **kwargs):
        raise AssertionError("flag off must never open the graph")

    monkeypatch.setattr(code_intel, "open_graph", _boom)

    assert ag.code_hub_gaps(mock_memory) == []


def test_missing_index_returns_empty(mock_memory, tmp_path, monkeypatch):
    monkeypatch.setenv("MEMO_CODEGRAPH_DB", str(tmp_path / "nope" / "codegraph.db"))

    assert ag.code_hub_gaps(mock_memory) == []


# --- briefing integration: receipt only, zero graph work at SessionStart ------------


def _seed_receipt(cfg, hub_gaps: list) -> None:
    path = Path(cfg.state_dir) / "dream" / "last.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"code_drift": {"status": "ok", "hub_gaps": hub_gaps}}), encoding="utf-8"
    )


def test_briefing_reads_at_most_one_hub_line_from_receipt(tmp_cfg, monkeypatch):
    from memo import code_intel

    _seed_receipt(tmp_cfg, [_TOP_HUB_LINE, "hub sin memoria: save (1 callers) — src/memo/store.py"])

    def _boom(*args, **kwargs):
        raise AssertionError("SessionStart must never open the graph")

    monkeypatch.setattr(code_intel, "open_graph", _boom)
    monkeypatch.setattr(ag, "_hub_gap_lines", _boom)

    assert ag.briefing_lines(tmp_cfg, session_id="s-hub") == [_TOP_HUB_LINE]


def test_briefing_without_receipt_is_silent_even_with_live_hubs(mock_memory, tmp_cfg, graph_db):
    # Uncited hubs exist in the live index, but no nightly receipt was written:
    # the briefing budget is zero graph queries, so nothing is computed live.
    assert ag.briefing_lines(tmp_cfg, session_id="s-hub2") == []


def test_briefing_corrupt_receipt_hub_section_is_silent(tmp_cfg):
    _seed_receipt(tmp_cfg, "not-a-list")  # type: ignore[arg-type]

    assert ag.briefing_lines(tmp_cfg, session_id="s-bad") == []


def test_briefing_flag_off_renders_nothing_and_does_no_work(tmp_cfg, monkeypatch):
    monkeypatch.setenv("MEMO_GAPS_CODE_HUBS", "0")
    _seed_receipt(tmp_cfg, [_TOP_HUB_LINE])

    def _boom(*args, **kwargs):
        raise AssertionError("flag off must never touch the store or the graph")

    monkeypatch.setattr(ag, "_hub_gap_lines", _boom)

    assert ag.briefing_lines(tmp_cfg, session_id="s-off") == []
