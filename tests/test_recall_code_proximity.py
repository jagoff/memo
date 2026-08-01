"""Recall code-proximity boost (MEMO_RECALL_CODE_PROXIMITY_BOOST, default 0.0 = OFF).

Covers:
- flag off (unset or explicit 0.0): rank_hits output identical AND zero extra
  work — subprocess.run is patched to explode, proving no git call happens.
- flag on: a hit whose code_ref cites a symbol in the 2-hop codegraph
  neighborhood of the uncommitted changes (or a changed file directly) gains
  +flag and reorders; refs outside the neighborhood stay untouched.
- fail-open: git failure / non-zero exit / missing graph DB → no boost, no
  exception.
"""

from __future__ import annotations

import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from memo import codegraph_loader
from memo import recall_logic as rl
from memo.recall_logic import RankKnobs

# --- synthetic codegraph.db (shape copied from test_dream_code_drift) -----------

_NODES = [
    # (id, kind, name, qualified_name, file_path, start_line, end_line)
    ("function:alpha", "function", "alpha", "memo.a.alpha", "src/memo/a.py", 1, 5),
    ("function:beta", "function", "beta", "memo.b.beta", "src/memo/b.py", 1, 5),
    ("function:gamma", "function", "gamma", "memo.c.gamma", "src/memo/c.py", 1, 5),
    ("function:delta", "function", "delta", "memo.d.delta", "src/memo/d.py", 1, 5),
    ("file:src/memo/a.py", "file", "a.py", None, "src/memo/a.py", None, None),
]

_EDGES = [
    # a → b → c → d over 'calls': changing a.py reaches gamma at 2 hops, never delta.
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
def graph_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db = tmp_path / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_graph(db)
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", db)
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.delenv("MEMO_CODEGRAPH_DB", raising=False)
    return db


# --- hits --------------------------------------------------------------------------


@dataclass
class _Hit:
    id: str
    score: float | None
    title: str = ""
    body: str = ""
    type: str = "note"
    extra: dict[str, Any] = field(default_factory=dict)


def _mk(id: str, score: float | None, **kw: Any) -> _Hit:
    """Hit with content unique per id so dedup_hits never collapses distinct hits."""
    kw.setdefault("title", f"title {id}")
    kw.setdefault("body", f"distinct body for memory {id}, long enough to pass the gate")
    return _Hit(id=id, score=score, **kw)


def _citing(id: str, score: float | None, *refs: dict[str, Any]) -> _Hit:
    return _mk(id, score, extra={"code_refs": list(refs)})


_KNOBS = RankKnobs(top_k=5, min_sim=0.0, min_body_chars=0)


def _fake_git(stdout: str, returncode: int = 0) -> Any:
    def _run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr="")

    return _run


# --- flag off = ZERO extra work ------------------------------------------------------


def _boom(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("flag 0.0 must never spawn a subprocess")


def test_flag_off_is_zero_work_and_ranking_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", _boom)
    hits = [
        _mk("far", 0.8),
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]

    monkeypatch.delenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", raising=False)
    baseline = rl.rank_hits(hits, _KNOBS)
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.0")
    out = rl.rank_hits(hits, _KNOBS)

    assert [h.id for h in baseline] == [h.id for h in out] == ["far", "near"]
    assert [h.score for h in baseline] == [h.score for h in out] == [0.8, 0.7]


def test_eval_knob_disables_repo_io_even_when_live_flag_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _boom)
    hits = [
        _mk("far", 0.8),
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]
    knobs = RankKnobs(
        top_k=5,
        min_sim=0.0,
        min_body_chars=0,
        code_proximity=False,
    )

    out = rl.rank_hits(hits, knobs)

    assert [h.id for h in out] == ["far", "near"]
    assert [h.score for h in out] == [0.8, 0.7]


# --- flag on: boost + reorder ---------------------------------------------------------


def test_symbol_in_neighborhood_boosts_and_reorders(
    monkeypatch: pytest.MonkeyPatch, graph_db: Path
) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    # changed a.py → symbols {alpha} → 2 hops = {alpha, beta, gamma}: 'gamma' is in.
    hits = [
        _mk("far", 0.8),
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert [h.id for h in out] == ["near", "far"]
    assert out[0].score == pytest.approx(1.0)
    assert out[1].score == pytest.approx(0.8)


def test_qualified_name_in_neighborhood_boosts(
    monkeypatch: pytest.MonkeyPatch, graph_db: Path
) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    # The neighborhood is node names; a ref carrying only a qualified_name that
    # EQUALS a node name (label empty) still matches via the qualified branch.
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    hits = [
        _citing(
            "q",
            0.7,
            {
                "file_path": "src/memo/c.py",
                "kind": "function",
                "label": "",
                "qualified_name": "gamma",
            },
        ),
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert out[0].score == pytest.approx(1.0)


def test_ref_citing_changed_file_path_boosts(
    monkeypatch: pytest.MonkeyPatch, graph_db: Path
) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.2")
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    # file-kind ref, no symbol: matches because the file itself changed.
    hits = [_citing("filehit", 0.5, {"file_path": "src/memo/a.py", "kind": "file", "label": ""})]

    out = rl.rank_hits(hits, _KNOBS)

    assert out[0].score == pytest.approx(0.7)


def test_ref_outside_neighborhood_not_boosted(
    monkeypatch: pytest.MonkeyPatch, graph_db: Path
) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    # delta is 3 hops from alpha → outside the clamped 2-hop neighborhood.
    hits = [
        _mk("far", 0.8),
        _citing("delta", 0.7, {"file_path": "src/memo/d.py", "kind": "function", "label": "delta"}),
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert [h.id for h in out] == ["far", "delta"]
    assert [h.score for h in out] == [0.8, 0.7]


def test_boost_applies_once_per_hit(monkeypatch: pytest.MonkeyPatch, graph_db: Path) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    # Two matching refs on one hit still add +flag exactly once.
    hits = [
        _citing(
            "multi",
            0.5,
            {"file_path": "src/memo/b.py", "kind": "function", "label": "beta"},
            {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"},
        )
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert out[0].score == pytest.approx(0.8)


# --- repo gating: refs claiming another repo never match the local neighborhood ---------


def test_foreign_repo_refs_are_never_boosted(
    monkeypatch: pytest.MonkeyPatch, graph_db: Path
) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    # Both refs name-match the LOCAL diff/neighborhood, but claim another repo
    # (explicit field / codegraph:// uri host): the neighborhood was computed
    # against THIS repo's graph, so neither may gain the boost.
    hits = [
        _citing(
            "foreign-field",
            0.7,
            {
                "file_path": "src/memo/a.py",
                "kind": "file",
                "label": "",
                "repo_id": "feedfacefeedface",
            },
        ),
        _citing(
            "foreign-uri",
            0.6,
            {
                "file_path": "src/memo/c.py",
                "kind": "function",
                "label": "gamma",
                "uri": "codegraph://feedfacefeedface/function:gamma",
            },
        ),
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert [h.id for h in out] == ["foreign-field", "foreign-uri"]
    assert [h.score for h in out] == [0.7, 0.6]


# --- render cwd: git diff + DB discovery follow the render, not the process -------------


def test_git_diff_and_db_discovery_run_in_render_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    proj = tmp_path / "proj"
    db = proj / ".codegraph" / "codegraph.db"
    db.parent.mkdir(parents=True)
    _seed_graph(db)
    monkeypatch.delenv("MEMO_CODEGRAPH_DB", raising=False)
    monkeypatch.delenv("MEMO_CODEGRAPH_DISCOVERY", raising=False)
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    seen: dict[str, Any] = {}

    def _run(args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if isinstance(args, list) and args[:2] == ["git", "diff"]:
            seen["cwd"] = kwargs.get("cwd")
            return subprocess.CompletedProcess(args, 0, stdout="src/memo/a.py\n", stderr="")
        # Anything else (the repo_id `git config` probe) fails → path fallback.
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", _run)
    knobs = RankKnobs(top_k=5, min_sim=0.0, min_body_chars=0, cwd=str(proj))
    hits = [
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]

    out = rl.rank_hits(hits, knobs)

    # The daemon's process cwd is / or $HOME — the render cwd must drive both
    # the git diff and the .codegraph discovery.
    assert seen["cwd"] == str(proj)
    assert out[0].score == pytest.approx(1.0)


def test_render_cwd_without_index_never_boosts_from_another_db(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Another repo's index exists and is env-pinned, but the render repo has
    # no .codegraph anywhere up its own tree: the pin must NOT answer for it.
    other = tmp_path / "other" / ".codegraph" / "codegraph.db"
    other.parent.mkdir(parents=True)
    _seed_graph(other)
    monkeypatch.setenv("MEMO_CODEGRAPH_DB", str(other))
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "1")
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    bare = tmp_path / "bare-repo"
    bare.mkdir()
    knobs = RankKnobs(top_k=5, min_sim=0.0, min_body_chars=0, cwd=str(bare))
    hits = [
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]

    out = rl.rank_hits(hits, knobs)

    assert out[0].score == pytest.approx(0.7)


# --- fail-open -------------------------------------------------------------------------


def test_git_failure_no_boost_no_exception(monkeypatch: pytest.MonkeyPatch, graph_db: Path) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")

    def _raise(*args: Any, **kwargs: Any) -> Any:
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", _raise)
    hits = [
        _mk("far", 0.8),
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert [h.id for h in out] == ["far", "near"]
    assert [h.score for h in out] == [0.8, 0.7]


def test_git_nonzero_exit_no_boost(monkeypatch: pytest.MonkeyPatch, graph_db: Path) -> None:
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _fake_git("fatal: not a git repository\n", 128))
    hits = [
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert out[0].score == pytest.approx(0.7)


def test_missing_graph_db_no_boost(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(codegraph_loader, "CODEGRAPH_DB", tmp_path / "missing" / "codegraph.db")
    monkeypatch.setenv("MEMO_CODEGRAPH_DISCOVERY", "0")
    monkeypatch.delenv("MEMO_CODEGRAPH_DB", raising=False)
    monkeypatch.setenv("MEMO_RECALL_CODE_PROXIMITY_BOOST", "0.3")
    monkeypatch.setattr(subprocess, "run", _fake_git("src/memo/a.py\n"))
    hits = [
        _citing("near", 0.7, {"file_path": "src/memo/c.py", "kind": "function", "label": "gamma"}),
    ]

    out = rl.rank_hits(hits, _KNOBS)

    assert out[0].score == pytest.approx(0.7)
