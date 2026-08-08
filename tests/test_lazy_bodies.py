"""`Memory.list(with_bodies=False)` — corpus sweeps stop materializing bodies.

`Memory.list` used to read every row's markdown off disk and YAML-parse it,
even for callers that never touch `record.body`. On the live ~10k-row corpus
that is ~44s per call, which blew the 120s MCP budget for the analytics,
temporal, federation-preview and procedure-candidate sweeps.

Each converted caller is pinned twice:

1. it triggers ZERO `_read_body` reads (the work is provably gone), and
2. its output is identical to the eager run (`with_bodies` forced True),
   so the conversion is behaviour-preserving.

Two anti-vacuity controls keep (2) honest:
`test_force_with_bodies_harness_overrides_both_directions` proves the eager/lazy
switch actually changes what `Memory.list` returns, and
`test_body_blanking_is_detectable` proves a caller that DOES read bodies
produces different output when they are suppressed. Without both, `lazy ==
eager` could be green simply because nothing ever differed.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from click.testing import CliRunner

from memo.memory import Memory

_SEED_NOTES = 6
_SEED_FACTS = 2
_SEED_ROWS = _SEED_NOTES + _SEED_FACTS


def _seed(mem: Memory) -> None:
    """A corpus with real, distinct, non-empty bodies on disk.

    The `fact` rows exist so `cli_code_facts._existing_hashes` (which filters
    `type_="fact"`) sweeps a non-empty result set — otherwise its "no body
    reads" assertion would be trivially green.
    """
    for i in range(_SEED_NOTES):
        mem.save(
            content=f"body number {i}\n\n" + f"distinctive detail {i} " * 20,
            title=f"note {i}",
            type_="note",
            tags=["alpha", "beta", f"idx-{i}"],
            auto_project=False,
            extra={"outcome_stats": {"successes": 3, "total": 4, "utility": 0.9}},
        )
    for i in range(_SEED_FACTS):
        mem.save(
            content=f"fact body {i}\n\n" + f"mined detail {i} " * 20,
            title=f"fact {i}",
            type_="fact",
            tags=["codegraph-derived", "alpha", f"fact-{i}"],
            auto_project=False,
            extra={"provenance_hash": f"hash-{i}"},
        )


@pytest.fixture
def body_reads(mock_memory: Memory, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records every `_read_body` call made on the fixture's Memory."""
    seen: list[str] = []
    original = mock_memory._read_body

    def spy(rel_path: str) -> str:
        seen.append(rel_path)
        return original(rel_path)

    monkeypatch.setattr(mock_memory, "_read_body", spy)
    return seen


# --- Memory.list itself ---------------------------------------------------


def test_list_default_still_materializes_bodies(mock_memory: Memory, body_reads: list[str]) -> None:
    _seed(mock_memory)
    body_reads.clear()

    records = mock_memory.list(limit=50)

    assert len(records) == _SEED_ROWS
    assert len(body_reads) == _SEED_ROWS, "default must read one body per row"
    assert sorted(r.body.splitlines()[0] for r in records) == sorted(
        [f"body number {i}" for i in range(_SEED_NOTES)]
        + [f"fact body {i}" for i in range(_SEED_FACTS)]
    )


def test_list_with_bodies_false_skips_read_body(mock_memory: Memory, body_reads: list[str]) -> None:
    _seed(mock_memory)
    body_reads.clear()

    records = mock_memory.list(limit=50, with_bodies=False)

    assert body_reads == [], "with_bodies=False must not touch _read_body at all"
    assert [r.id for r in records] == [r.id for r in mock_memory.list(limit=50)]
    assert all(r.body == "" for r in records)
    assert all(r.title and r.tags and r.updated for r in records), "metadata still populated"


# --- converted callers ----------------------------------------------------


def _analytics_summary(mem: Memory) -> Any:
    from memo.analytics import AnalyticsEngine

    return AnalyticsEngine(mem).compute_corpus_metrics().__dict__


def _analytics_growth(mem: Memory) -> Any:
    from memo.analytics import AnalyticsEngine

    return AnalyticsEngine(mem).compute_growth_data().__dict__


def _temporal_stale(mem: Memory) -> Any:
    from memo.temporal import TemporalAnalyzer

    return TemporalAnalyzer(mem).detect_stale_memories(days_threshold=0)


def _temporal_patterns(mem: Memory) -> Any:
    from memo.temporal import TemporalAnalyzer

    return TemporalAnalyzer(mem).detect_temporal_patterns()


def _federation_preview(mem: Memory) -> Any:
    from memo.federation import FederationManager

    return FederationManager(mem).preview(principal="owner-a", owner_principal="owner-a")


def _procedure_candidates(mem: Memory) -> Any:
    return mem.procedure_candidates(min_successes=2, min_utility=0.5)


def _code_facts_existing_hashes(mem: Memory) -> Any:
    from memo.cli_code_facts import _existing_hashes

    return sorted(_existing_hashes(mem))


def _gc_vault_orphans(mem: Memory) -> Any:
    import memo.cli_ops as cli_ops
    from memo.cli_ops import ops_group

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(cli_ops, "_get_memory", lambda cfg: mem)
        result = CliRunner().invoke(ops_group, ["gc-vault-orphans", "--dry-run", "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


_CONVERTED = [
    pytest.param(_analytics_summary, id="analytics.compute_corpus_metrics"),
    pytest.param(_analytics_growth, id="analytics.compute_growth_data"),
    pytest.param(_temporal_stale, id="temporal.detect_stale_memories"),
    pytest.param(_temporal_patterns, id="temporal.detect_temporal_patterns"),
    pytest.param(_federation_preview, id="federation.preview"),
    pytest.param(_procedure_candidates, id="memory.procedure_candidates"),
    pytest.param(_code_facts_existing_hashes, id="cli_code_facts._existing_hashes"),
    pytest.param(_gc_vault_orphans, id="cli_ops.gc-vault-orphans"),
]


def _force_with_bodies(mp: pytest.MonkeyPatch, value: bool) -> None:
    """Override every `Memory.list` call's `with_bodies` decision."""
    original = Memory.list

    def patched(self: Memory, **kwargs: Any) -> Any:
        kwargs["with_bodies"] = value
        return original(self, **kwargs)

    mp.setattr(Memory, "list", patched)


def test_force_with_bodies_harness_overrides_both_directions(mock_memory: Memory) -> None:
    """The eager-vs-lazy comparison below is only meaningful if this switch works.

    A `_force_with_bodies` that silently did nothing would make every
    `lazy == eager` assertion compare two identical lazy runs.
    """
    _seed(mock_memory)

    with pytest.MonkeyPatch.context() as mp:
        _force_with_bodies(mp, True)
        forced_on = mock_memory.list(limit=50, with_bodies=False)
    with pytest.MonkeyPatch.context() as mp:
        _force_with_bodies(mp, False)
        forced_off = mock_memory.list(limit=50, with_bodies=True)

    assert forced_on and forced_off
    assert all(r.body for r in forced_on), "forced True must beat a caller's with_bodies=False"
    assert all(r.body == "" for r in forced_off), "forced False must beat a caller's True"


@pytest.mark.parametrize("caller", _CONVERTED)
def test_converted_caller_reads_no_bodies(
    mock_memory: Memory, body_reads: list[str], caller: Any
) -> None:
    _seed(mock_memory)
    body_reads.clear()

    caller(mock_memory)

    assert body_reads == [], f"{caller.__name__} still materializes bodies"


@pytest.mark.parametrize("caller", _CONVERTED)
def test_converted_caller_output_matches_eager_run(mock_memory: Memory, caller: Any) -> None:
    _seed(mock_memory)

    lazy = caller(mock_memory)
    with pytest.MonkeyPatch.context() as mp:
        _force_with_bodies(mp, True)
        eager = caller(mock_memory)

    assert lazy == eager


# --- anti-vacuity control -------------------------------------------------


def test_body_blanking_is_detectable(mock_memory: Memory) -> None:
    """A caller that DOES read bodies must notice when they are blanked.

    Without this, every equality assertion above could be green simply because
    the fixture corpus has empty bodies, or because the harness never actually
    changes what `Memory.list` returns.
    """
    from memo.memory_profile import build_memory_profile

    _seed(mock_memory)

    with_bodies = build_memory_profile(mock_memory)
    with pytest.MonkeyPatch.context() as mp:
        _force_with_bodies(mp, False)
        without_bodies = build_memory_profile(mock_memory)

    assert with_bodies != without_bodies
    assert [row["text"] for row in with_bodies["active"]] != [
        row["text"] for row in without_bodies["active"]
    ]
    assert all(row["text"] for row in with_bodies["active"])
    assert all(row["text"] == "" for row in without_bodies["active"])
