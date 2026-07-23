"""Path-parity tests: the recall-hook SUBPROCESS fallback must rank exactly
like the daemon path.

Both paths now share one pipeline — ``knobs_from_flags`` -> ``rank_hits`` ->
``apply_injection_filters`` -> top_k/nudge split — so for identical inputs the
injected ids (and their order) must be byte-identical between:

* the subprocess path (``memo recall-hook`` with no daemon socket), and
* a direct ``rank_hits`` call with the same knobs.

The expected side is computed with the same functions the daemon path
(_recall_logic) uses, so any drift in the hook's inline wiring fails here.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.memory import MemoryRecord
from memo.recall_logic import apply_injection_filters, knobs_from_flags, rank_hits

if TYPE_CHECKING:
    from memo.config import Config

# ---------------------------------------------------------------------------
# Fixture pool — crafted so the M3 knobs actually reorder it:
#   * A and B share almost all body tokens (high Jaccard) -> MMR demotes B;
#   * C is type=decision -> preference boost target;
#   * D is type=synthesis -> synthesis-boost target;
#   * E carries the project tag -> tier-1 project boost target.
# ---------------------------------------------------------------------------

_SHARED_TOKENS = "the quick brown fox jumps over the lazy dog near the river bank at dawn"


def _rec(
    id_: str,
    title: str,
    type_: str,
    score: float,
    body: str,
    tags: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=id_,
        path=f"notes/{id_}.md",
        title=title,
        type=type_,
        tags=tags or [],
        created="2026-01-01T00:00:00+00:00",
        updated="2026-01-01T00:00:00+00:00",
        body=body,
        extra={},
        score=score,
    )


def _make_pool() -> list[MemoryRecord]:
    return [
        _rec("aaaa1111aaaa1111", "Alpha shared context", "note", 0.82, _SHARED_TOKENS),
        _rec(
            "bbbb2222bbbb2222",
            "Beta shared context",
            "note",
            0.80,
            _SHARED_TOKENS + " twilight",
        ),
        _rec(
            "cccc3333cccc3333",
            "Gamma release decision",
            "decision",
            0.62,
            "release gating decided via worktree cherry-pick and tag push",
        ),
        _rec(
            "dddd4444dddd4444",
            "Delta cross-cluster insight",
            "synthesis",
            0.55,
            "distilled recurring pattern across consolidation sessions",
        ),
        _rec(
            "eeee5555eeee5555",
            "Epsilon project note",
            "note",
            0.50,
            "project scoped observation about the local build cache",
            tags=["project:parityproj"],
        ),
    ]


_PREFS = SimpleNamespace(preferred_types={"decision": 1.0})


class _StubMemory:
    """Memory double for the hook's in-process invocation."""

    contextual = SimpleNamespace(
        context=SimpleNamespace(get_preferences=staticmethod(lambda: _PREFS)),
        record_search=staticmethod(lambda prompt, ids: None),
    )

    def __init__(self, cfg: object) -> None:
        self.cfg = cfg

    def search(
        self,
        query: str,
        limit: int = 5,
        mode: str = "bm25",
        recency: bool = False,
        exclude_types: object = None,
        exclude_tags: object = None,
    ) -> list[MemoryRecord]:
        return _make_pool()

    def close(self) -> None:
        pass


def _base_env(tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin flags via the process env so BOTH sides (CliRunner in-process hook
    and the direct knobs_from_flags call) resolve the identical knobs."""
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_cfg.data_dir))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_cfg.state_dir))
    monkeypatch.setenv("MEMO_VAULT_PATH", str(tmp_cfg.vault_path))
    monkeypatch.setenv("MEMO_RECALL_MODE", "bm25")  # no warm-signal dependence
    monkeypatch.setenv("MEMO_RECALL_TOP_K", "3")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.1")
    monkeypatch.setenv("MEMO_RECALL_MIN_BODY_CHARS", "0")
    monkeypatch.setenv("MEMO_RECALL_TOKEN_BUDGET", "0")
    monkeypatch.setenv("MEMO_RECALL_FORMAT", "full")
    monkeypatch.setenv("MEMO_RECALL_EXPAND_CONTEXT", "0")
    monkeypatch.setenv("MEMO_RECALL_ASSOCIATIVE", "0")
    monkeypatch.setenv("MEMO_RECALL_CITE_INSTRUCTION", "0")
    monkeypatch.setenv("MEMO_RECALL_FEEDBACK_HINT", "0")
    monkeypatch.delenv("MEMO_RECALL_SESSION_MODE", raising=False)


def _run_hook(prompt: str, cwd: str) -> dict:
    runner = CliRunner()
    payload = json.dumps({"prompt": prompt, "cwd": cwd})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return json.loads(result.output.strip())


def _injected_ids(context: str) -> tuple[list[str], list[str]]:
    """(main-block id8s in order, rank-overflow nudge id8s in order)."""
    main = re.findall(r"\*\*\[([0-9a-f]{8})\]", context)
    nudge_ids: list[str] = []
    for line in context.splitlines():
        if "Also in your memory (related):" in line:
            nudge_ids = re.findall(r"\[([0-9a-f]{8})\]", line)
    return main, nudge_ids


def _expected_split(cwd: str, prefs: object | None) -> tuple[list[str], list[str]]:
    """Rank the same pool exactly like the daemon path does — including the
    pre-top-K dedup-collapse (MEMO_RECALL_DEDUP_COLLAPSE, default ON) that both
    the daemon and (now) the subprocess apply between apply_injection_filters
    and the top_k/nudge split."""
    from memo.flags import flag_bool, flag_float
    from memo.recall_logic import collapse_near_dups

    knobs = knobs_from_flags(cwd=cwd)
    qualifying = rank_hits(_make_pool(), knobs, preferences=prefs)
    qualifying = apply_injection_filters(qualifying)
    if flag_bool("MEMO_RECALL_DEDUP_COLLAPSE") and len(qualifying) > 1:
        qualifying = collapse_near_dups(
            qualifying, threshold=flag_float("MEMO_RECALL_INTRA_DEDUP_THRESHOLD") or 0.8
        )
    main = [h.id[:8] for h in qualifying[: knobs.top_k]]
    nudge = [h.id[:8] for h in qualifying[knobs.top_k : knobs.top_k + 2]]
    return main, nudge


# ---------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------


def test_subprocess_matches_rank_hits_default_knobs(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Baseline parity: default boost knobs (no MMR/synth/project/pref env)."""
    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setattr("memo.memory.Memory", _StubMemory)

    out = _run_hook("how did we decide the parity ranking approach", str(tmp_path))
    got_main, got_nudge = _injected_ids(out["hookSpecificOutput"]["additionalContext"])
    # CONTEXTUAL defaults on -> the hook reads the stub's preferences; mirror it.
    exp_main, exp_nudge = _expected_split(str(tmp_path), _PREFS)

    assert got_main == exp_main
    assert got_nudge == exp_nudge


def test_subprocess_matches_rank_hits_with_mmr_synth_pref_project_on(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Full-knob parity: MMR + synthesis boost + preference boost + project
    tiers all ON — the knobs the old inline ranking silently skipped."""
    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_MMR_LAMBDA", "0.7")
    monkeypatch.setenv("MEMO_RECALL_SYNTHESIS_BOOST", "0.15")
    monkeypatch.setenv("MEMO_RECALL_CONTEXTUAL", "1")
    monkeypatch.setenv("MEMO_PROJECT_TAG", "parityproj")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    monkeypatch.setattr("memo.memory.Memory", _StubMemory)

    out = _run_hook("how did we decide the parity ranking approach", str(tmp_path))
    got_main, got_nudge = _injected_ids(out["hookSpecificOutput"]["additionalContext"])
    exp_main, exp_nudge = _expected_split(str(tmp_path), _PREFS)

    assert got_main == exp_main
    assert got_nudge == exp_nudge
    # Self-check: the knobs actually reordered the pool (otherwise this test
    # would pass even if the hook ignored them, as the old inline path did).
    raw_order = [h.id[:8] for h in _make_pool()]
    assert got_main + got_nudge != raw_order[:5]


# ---------------------------------------------------------------------------
# Daemon-semantics post-rank filters now apply on the subprocess path
# ---------------------------------------------------------------------------


def test_subprocess_applies_skip_below_floor(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A weak top hit under MEMO_RECALL_SKIP_BELOW injects nothing — same as
    the daemon's apply_injection_filters (the old inline path ignored it)."""
    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0.95")
    monkeypatch.setattr("memo.memory.Memory", _StubMemory)

    out = _run_hook("how did we decide the parity ranking approach", str(tmp_path))
    assert out == {}


def test_subprocess_applies_gap_trim_to_top1(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A large rank-1 -> rank-2 score gap trims injection to the single top hit."""
    strong = _rec("f0f0f0f0f0f0f0f0", "Strong isolated hit", "note", 0.95, "very strong body")
    weak = _rec("0e0e0e0e0e0e0e0e", "Weak tail hit", "note", 0.55, "a much weaker body")

    class _GapStub(_StubMemory):
        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            return [strong, weak]

    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0.2")
    monkeypatch.setattr("memo.memory.Memory", _GapStub)

    out = _run_hook("how did we decide the parity ranking approach", str(tmp_path))
    got_main, got_nudge = _injected_ids(out["hookSpecificOutput"]["additionalContext"])
    assert got_main == ["f0f0f0f0"]
    assert got_nudge == []  # trimmed away, not demoted into the nudge


def test_subprocess_backfills_below_old_topk_slice(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Daemon semantics: gate the FULL pool, then slice top_k. The old inline
    path sliced first, so a gated-out hit inside the slice shrank the
    injection instead of backfilling from below the cut."""
    high = _rec("1a1a1a1a1a1a1a1a", "High passer", "note", 0.90, "high body one two three")
    gated = _rec("2b2b2b2b2b2b2b2b", "Gated mid", "note", 0.55, "mid body four five six")
    backfill = _rec("3c3c3c3c3c3c3c3c", "Backfill hit", "note", 0.65, "low body seven eight")

    class _BackfillStub(_StubMemory):
        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            return [high, gated, backfill]

    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_TOP_K", "2")
    monkeypatch.setenv("MEMO_RECALL_MIN_SIM", "0.6")
    monkeypatch.setenv("MEMO_RECALL_SKIP_BELOW", "0")
    monkeypatch.setenv("MEMO_RECALL_GAP_THRESHOLD", "0")
    monkeypatch.setattr("memo.memory.Memory", _BackfillStub)

    out = _run_hook("how did we decide the parity ranking approach", str(tmp_path))
    got_main, _ = _injected_ids(out["hookSpecificOutput"]["additionalContext"])
    assert got_main == ["1a1a1a1a", "3c3c3c3c"]  # 2b… gated by min_sim, 3c… backfills


# ---------------------------------------------------------------------------
# Bails stay intact around the new pipeline
# ---------------------------------------------------------------------------


def test_no_hits_bail_reports_knobs_min_sim(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty rank result still bails with the resolved min_sim in the reason."""

    class _EmptyStub(_StubMemory):
        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            return []

    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_DEBUG", "1")
    monkeypatch.setattr("memo.memory.Memory", _EmptyStub)

    runner = CliRunner()
    payload = json.dumps({"prompt": "how did we decide the parity ranking approach"})
    result = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert result.exit_code == 0
    # CliRunner mixes the stderr debug line into output; the JSON bail is its own line.
    assert "{}" in [ln.strip() for ln in result.output.splitlines()]
    assert "no hits above min_sim=0.1" in result.output


def test_session_dedup_survives_pipeline_swap(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Second identical turn injects nothing (all hits already recalled)."""
    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setattr("memo.memory.Memory", _StubMemory)

    runner = CliRunner()
    payload = json.dumps(
        {
            "prompt": "how did we decide the parity ranking approach",
            "cwd": str(tmp_path),
            "session_id": "parity-dedup-001",
        }
    )
    first = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert first.exit_code == 0
    assert "additionalContext" in first.output

    second = runner.invoke(cli, ["recall-hook"], input=payload, catch_exceptions=False)
    assert second.exit_code == 0
    assert json.loads(second.output.strip()) == {}


# ---------------------------------------------------------------------------
# '_uncertain' quarantine parity (daemon excludes it via uncertain_exclusion)
# ---------------------------------------------------------------------------


def test_subprocess_excludes_uncertain_quarantine_by_default(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The subprocess fallback must pass the default-on '_uncertain' exclusion
    to search, exactly like the daemon path (recall_logic uncertain_exclusion)."""
    seen: dict[str, object] = {}

    class _RecordingStub(_StubMemory):
        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            seen["exclude_tags"] = exclude_tags
            return _make_pool()

    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.delenv("MEMO_RECALL_EXCLUDE_UNCERTAIN", raising=False)  # default on
    monkeypatch.setattr("memo.memory.Memory", _RecordingStub)

    _run_hook("how did we decide the parity ranking approach", str(tmp_path))
    assert seen["exclude_tags"] == {"_uncertain"}


def test_subprocess_uncertain_exclusion_flag_off(
    tmp_cfg: Config, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    class _RecordingStub(_StubMemory):
        def search(
            self, query, limit=5, mode="bm25", recency=False, exclude_types=None, exclude_tags=None
        ):
            seen["exclude_tags"] = exclude_tags
            return _make_pool()

    _base_env(tmp_cfg, monkeypatch)
    monkeypatch.setenv("MEMO_RECALL_EXCLUDE_UNCERTAIN", "0")
    monkeypatch.setattr("memo.memory.Memory", _RecordingStub)

    _run_hook("how did we decide the parity ranking approach", str(tmp_path))
    assert seen["exclude_tags"] is None
