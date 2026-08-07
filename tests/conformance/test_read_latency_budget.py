"""Under a budget it cannot meet, search returns inside the budget AND says what
it dropped. The assertion is on the decision and the report, not on a race.

`Memory.search`'s shed ladder itself is covered exhaustively by
`tests/test_search_degradation.py` (each rung, in isolation, against a
3-memory store). This module proves the CLI surface built on top of it --
`memo search` -- actually forwards `_degraded` to a human instead of quietly
absorbing it, against the ~10k-memory corpus these defects were found at, not
a toy fixture. The MCP surface (`memo_search`) gets its own coverage in
`tests/test_server_core_search.py` since it is an independently-written merge
expression in a different file.

`--json` stays a bare hit array UNCONDITIONALLY, whether or not anything was
shed -- the top-level JSON type must never depend on a runtime condition an
automated caller cannot predict. Degradation is reported on stderr only
(`degraded: <stages> (search budget)`), never folded into stdout.

`Memory.search(..., _track_usage=True)` (the default) writes to the shared
`access`/`meta.roi_score` rows of whatever store it runs against -- `touch()`
and `boost_roi_batch()` in `memory/search_scoring_ops.py`. The two direct
`Memory(big_corpus)` calls below pass `_track_usage=False` to avoid that
mutation outright (there is no result-shape reason to track usage for a
budget probe). The CLI test goes through `memo search`'s real write path
(no `--no-track-usage` flag exists), so it snapshots/restores
`big_corpus.db_path` (+ WAL/SHM) around the call -- the same technique
`links_reindex_result` (test_index_rebuild_preserves.py) uses for
`crossref.db`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.memory.facade import Memory

from .conftest import DIMS, _env

pytestmark = pytest.mark.conformance


def _db_sidecars(db_path: Path) -> tuple[Path, Path, Path]:
    return (
        db_path,
        db_path.with_name(db_path.name + "-wal"),
        db_path.with_name(db_path.name + "-shm"),
    )


def _snapshot_db(db_path: Path) -> dict[Path, bytes]:
    return {p: p.read_bytes() for p in _db_sidecars(db_path) if p.exists()}


def _restore_db(db_path: Path, snapshot: dict[Path, bytes]) -> None:
    for p in _db_sidecars(db_path):
        p.unlink(missing_ok=True)
    for p, data in snapshot.items():
        p.write_bytes(data)


def _direct_cfg(big_corpus):
    """`big_corpus` itself carries the pydantic-default `embedder_dims=1024` --
    seeding writes 64-dim vectors straight into `VecStore` (bypassing
    `Config.from_env()`, the only path that actually reads
    `MEMO_EMBEDDER_DIMS`), so a bare `Memory(big_corpus)` trips the store's
    dims guard (`store/schema.py`) instead of running a search. The CLI test
    below sidesteps this correctly by going through `_env(big_corpus)` +
    `CliRunner`, which DOES route through `Config.from_env()`; a direct
    `Memory()` construction needs the same correction by hand.
    """
    return big_corpus.model_copy(update={"embedder_dims": DIMS})


def test_search_returns_inside_a_generous_budget(big_corpus) -> None:
    mem = Memory(_direct_cfg(big_corpus))
    try:
        started = time.monotonic()
        degraded: list[str] = []
        mem.search(
            "topic00",
            mode="bm25",
            limit=10,
            _budget_ms=30000,
            _degraded=degraded,
            _track_usage=False,
        )
        assert (time.monotonic() - started) < 30.0
        assert degraded == [], "a generous budget must never shed a stage"
    finally:
        mem.close()


def test_a_tight_budget_degrades_and_reports(big_corpus) -> None:
    mem = Memory(_direct_cfg(big_corpus))
    try:
        degraded: list[str] = []
        hits = mem.search(
            "topic00",
            mode="hybrid",
            limit=10,
            _budget_ms=1,
            _degraded=degraded,
            _track_usage=False,
        )
        # A deterministic decision, not a timing race: 1ms cannot afford the
        # 2000ms embed estimate (COST_EMBED_MS) regardless of machine speed,
        # so rung four always fires. HyDE/graph-signal/rerank are all
        # flag-off in this corpus's config, so this is the ONLY rung that can
        # fire -- an exact-list assertion, same style as
        # tests/test_search_degradation.py.
        assert degraded == ["embed_skipped_bm25_only"], (
            "search silently ran every stage under a 1ms budget"
        )
        assert hits, "the shed must still fall back to BM25, not to nothing"
    finally:
        mem.close()


def test_cli_reports_degradation_on_stderr_only(big_corpus) -> None:
    """`--json` stays a bare array UNCONDITIONALLY -- the top-level JSON type
    must never depend on a runtime condition the caller cannot predict (a
    script doing `json.loads(out)[0]` today must keep working under
    contention, not break exactly when it is least watched). Degradation is
    reported on stderr only; a `--json` consumer is no worse off than before
    this feature existed (no signal), and stderr carries the signal for
    anyone who wants it."""
    snapshot = _snapshot_db(big_corpus.db_path)
    try:
        result = CliRunner().invoke(
            cli,
            ["search", "topic00", "--json", "--limit", "5"],
            env={**_env(big_corpus), "MEMO_SEARCH_BUDGET_MS": "1"},
        )
        assert result.exit_code == 0, result.output
        # `result.output` mixes stdout+stderr in call order (Click >= 8.2) --
        # parse `result.stdout` alone, never `.output`, so a stderr note can
        # never be mistaken for stdout contamination.
        payload = json.loads(result.stdout)
        assert isinstance(payload, list), (
            f"--json must stay a bare hit array even when a stage was shed, got: {payload!r}"
        )
        assert "degraded: embed_skipped_bm25_only" in result.stderr, (
            f"a 1ms budget shed a stage but the CLI never reported it on stderr: {result.stderr!r}"
        )
    finally:
        _restore_db(big_corpus.db_path, snapshot)


def test_cli_reports_no_degraded_note_on_a_healthy_search(big_corpus) -> None:
    """Converse of the above: an unaffected search's --json output must stay
    byte-identical to today's bare hit array, and stderr must stay silent."""
    snapshot = _snapshot_db(big_corpus.db_path)
    try:
        result = CliRunner().invoke(
            cli,
            ["search", "topic00", "--json", "--limit", "5"],
            env=_env(big_corpus),
        )
        assert result.exit_code == 0, result.output
        payload = json.loads(result.stdout)
        assert isinstance(payload, list), (
            f"an unaffected search must return the bare hit array unchanged, got: {payload!r}"
        )
        assert result.stderr == "", (
            f"a healthy search must not write a degraded note: {result.stderr!r}"
        )
    finally:
        _restore_db(big_corpus.db_path, snapshot)
