"""'Recall Quality' TUI panel — prec@5 trend + citation stats from grounding.log.

The panel reads two files under state_dir:
  - eval/history.jsonl   (eval pass receipts: prec_at_k / noise_at_k / k / labels)
  - grounding.log        (per-turn grounding rows: recall_id / method / used_score)

Every read path must degrade gracefully: missing/empty/corrupt files render a
quiet "sin datos aún" state — never crash the TUI.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from rich.console import Console
from rich.panel import Panel

import memo.tui.dashboard.panels as tui_panels
from memo.dashboard_panels import (
    _grounding_citation_stats,
    _panel_corpus,
    _panel_recall_quality,
    _panel_recall_trend,
    _panel_utility,
    _read_eval_history,
)


def _render(panel: Panel, *, width: int = 100) -> str:
    console = Console(record=True, width=width, force_terminal=False)
    console.print(panel)
    return console.export_text()


def _write_history(state_dir: Path, precs: list[float]) -> None:
    eval_dir = state_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            {
                "ts": f"2026-07-{i + 1:02d}T03:00:00+00:00",
                "prec_at_k": p,
                "noise_at_k": 0.0,
                "k": 5,
                "labels": 12,
                "source": "dream",
            }
        )
        for i, p in enumerate(precs)
    ]
    (eval_dir / "history.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _grounding_row(recall_id: str, method: str, turn: int = 1, project: str | None = None) -> str:
    row = {
        "ts": "2026-07-02T10:00:00+00:00",
        "session_id": "sess-1",
        "turn": turn,
        "recall_id": recall_id,
        "used_score": 1.0 if method == "cited" else 0.3,
        "method": method,
    }
    if project is not None:
        row["project"] = project
    return json.dumps(row)


def _write_grounding(state_dir: Path, rows: list[str]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "grounding.log").write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_panel_renders_trend_and_citations(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_history(state_dir, [0.1, 0.15, 0.2])
    _write_grounding(
        state_dir,
        [
            _grounding_row("aaaaaaaa", "cited"),
            _grounding_row("aaaaaaaa", "cited", turn=2),
            _grounding_row("aaaaaaaa", "cited", turn=3),
            _grounding_row("bbbbbbbb", "cited"),
            _grounding_row("cccccccc", "lexical"),
            _grounding_row("dddddddd", "embed"),
        ],
    )

    out = _render(_panel_recall_trend(state_dir))

    assert "recall quality" in out
    assert "prec@5" in out
    assert "0.20" in out  # latest value
    assert "(3 runs)" in out
    # Top cited: aaaaaaaa ×3 first, bbbbbbbb ×1 second.
    assert "[aaaaaaaa]" in out
    assert "×3" in out  # noqa: RUF001
    assert "[bbbbbbbb]" in out
    # Recalled-never-cited: cccccccc + dddddddd → 2 of 4.
    assert "2 of 4 recalled" in out
    assert "sin datos aún" not in out


def test_live_recall_panels_name_composite_ranking_score(tmp_path: Path) -> None:
    from memo.dashboard import append_recall_log

    append_recall_log(
        tmp_path,
        prompt="composite panel one",
        hits=[{"id": "a" * 8, "score": 1.2, "title": "A"}],
        via="daemon",
    )
    append_recall_log(
        tmp_path,
        prompt="composite panel two",
        hits=[{"id": "b" * 8, "score": 0.7, "title": "B"}],
        via="daemon",
    )

    out = _render(_panel_recall_quality(tmp_path))
    utility_out = _render(_panel_utility(tmp_path))  # type: ignore[arg-type]

    assert "hit / comp>0.85" in out
    assert "composite p50" in out
    assert "top composite" in utility_out
    assert "final score >0.85" in utility_out
    assert "strong" not in out + utility_out


def test_utility_panel_names_observed_context_activity(tmp_path: Path) -> None:
    out = _render(_panel_utility(tmp_path))  # type: ignore[arg-type]

    assert "context activity" in out
    assert "tokens injected" in out
    assert "memories surfaced" in out
    assert "tokens saved" not in out
    assert "cost saved" not in out
    assert "worth it?" not in out


def test_history_caps_at_last_seven_entries(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_history(state_dir, [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.4])

    rows = _read_eval_history(state_dir, limit=7)
    assert len(rows) == 7
    assert rows[-1]["prec_at_k"] == 0.4

    out = _render(_panel_recall_trend(state_dir))
    assert "0.40" in out
    assert "(7 runs)" in out


def test_missing_files_render_quiet_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()

    out = _render(_panel_recall_trend(state_dir))
    assert "recall quality" in out
    assert "sin datos aún" in out


def test_empty_files_render_quiet_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    (state_dir / "eval").mkdir(parents=True)
    (state_dir / "eval" / "history.jsonl").write_text("", encoding="utf-8")
    (state_dir / "grounding.log").write_text("", encoding="utf-8")

    out = _render(_panel_recall_trend(state_dir))
    assert "sin datos aún" in out


def test_corrupt_history_lines_are_skipped(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    eval_dir = state_dir / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "history.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"ts": "2026-07-01T03:00:00+00:00", "prec_at_k": 0.1, "k": 5}),
                "{not json at all",
                json.dumps({"ts": "2026-07-02T03:00:00+00:00", "prec_at_k": 0.3, "k": 5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = _read_eval_history(state_dir)
    assert [r["prec_at_k"] for r in rows] == [0.1, 0.3]

    out = _render(_panel_recall_trend(state_dir))
    assert "0.30" in out
    assert "(2 runs)" in out


def test_corrupt_grounding_lines_are_skipped(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_grounding(
        state_dir,
        [
            _grounding_row("aaaaaaaa", "cited"),
            "garbage-not-json",
            _grounding_row("bbbbbbbb", "lexical"),
        ],
    )

    stats = _grounding_citation_stats(state_dir)
    assert stats["top_cited"] == [("aaaaaaaa", 1)]
    assert stats["never_cited"] == 1
    assert stats["seen"] == 2

    out = _render(_panel_recall_trend(state_dir))
    assert "[aaaaaaaa]" in out
    assert "1 of 2 recalled" in out


def test_project_bearing_rows_are_tolerated(tmp_path: Path) -> None:
    # Fase 2 writer stamps a `project` field on grounding rows — the citation
    # stats reader (dict-based) must count them exactly like project-less rows.
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_grounding(
        state_dir,
        [
            _grounding_row("aaaaaaaa", "cited", project="project:memo"),
            _grounding_row("aaaaaaaa", "cited", turn=2, project="project:memo"),
            _grounding_row("bbbbbbbb", "lexical", project="project:synapse"),
            _grounding_row("cccccccc", "embed"),  # legacy row, no project
        ],
    )

    stats = _grounding_citation_stats(state_dir)
    assert stats["top_cited"] == [("aaaaaaaa", 2)]
    assert stats["never_cited"] == 2
    assert stats["seen"] == 3

    out = _render(_panel_recall_trend(state_dir))
    assert "[aaaaaaaa]" in out


def test_grounding_only_shows_partial_data(tmp_path: Path) -> None:
    # No eval history yet — citations still render, prec row shows quiet state.
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_grounding(state_dir, [_grounding_row("aaaaaaaa", "cited")])

    out = _render(_panel_recall_trend(state_dir))
    assert "[aaaaaaaa]" in out
    assert "prec@5" in out
    assert "sin datos aún" in out  # the prec row's quiet state


def test_history_only_shows_partial_data(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_history(state_dir, [0.2])

    out = _render(_panel_recall_trend(state_dir))
    assert "0.20" in out
    assert "sin datos aún" in out  # citation rows' quiet state


def test_non_numeric_prec_entries_are_ignored(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    eval_dir = state_dir / "eval"
    eval_dir.mkdir(parents=True)
    (eval_dir / "history.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"ts": "t", "prec_at_k": "oops", "k": 5}),
                json.dumps({"ts": "t", "prec_at_k": 0.25, "k": 5}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = _render(_panel_recall_trend(state_dir))
    assert "0.25" in out
    assert "(1 runs)" in out


def _corpus_memory(rows: list[dict], total: int) -> SimpleNamespace:
    # The row cache is module-level and keyed on id(memory), so a recycled id
    # would serve a previous fake's rows within the 10s TTL.
    tui_panels._corpus_cache.set(None)
    store = SimpleNamespace(list_recent=lambda limit: rows[:limit], count=lambda: total)
    return SimpleNamespace(store=store)


def test_corpus_panel_reports_store_total_not_page_size() -> None:
    """The counters are built from a bounded page of the newest records;
    rendering their sum showed the page size (10000) as the corpus total."""
    rows = [{"type": "note", "tags": ["project:memo"]} for _ in range(10_000)]

    out = _render(_panel_corpus(_corpus_memory(rows, 12_619)))

    assert "12619 memories" in out
    assert "10000 memories" not in out
    # The window the type breakdown was computed from has to be visible, or
    # the breakdown reads as if it described all 12619.
    assert "newest 10000" in out


def test_corpus_panel_omits_window_note_when_page_covers_corpus() -> None:
    rows = [{"type": "note", "tags": ["project:memo"]} for _ in range(3)]

    out = _render(_panel_corpus(_corpus_memory(rows, 3)))

    assert "3 memories" in out
    assert "newest" not in out


def test_render_layout_includes_recall_quality_panel(tmp_path: Path) -> None:
    """The panel is wired into the live `memo tui` layout."""
    from memo.dashboard_tui import render

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    _write_history(state_dir, [0.2])

    class _StubStore:
        def list_recent(self, limit: int = 0) -> list[dict]:
            return []

        def count(self) -> int:
            return 0

    class _StubCfg:
        memory_dir = tmp_path / "memories"

    class _StubMemory:
        store = _StubStore()
        cfg = _StubCfg()
        embedder = object()

    console = Console(record=True, width=140, height=40, force_terminal=False)
    console.print(render(_StubMemory(), state_dir))
    out = console.export_text()
    assert "recall quality" in out
    assert "prec@5" in out
