"""Wiring/integration tests for the verification-state decay feature.

Unit behavior lives in test_verification_state.py (transition) and
test_rerank_verification.py (penalty math). These tests prove the two halves are
actually WIRED into the product:

* Part A — `memo maintain` runs `_transition_stale_memories` iff
  MEMO_VERIFICATION_STATE_TRACKING is on.
* Part B — the live search-scoring pipeline runs `_apply_verification_decay`
  iff the flag is on.
* Lifecycle — a memory marked VERIFIED without a `verified_at` gets one stamped
  on reindex, so it enters the decay clock.
"""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli
from memo.memory import Memory
from memo.memory.maintain_ops import _MaintainOpsMixin
from memo.memory.search_scoring_ops import _SearchScoringMixin
from memo.tiers import VerificationState


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        **extra,
    }


# -- Part A: maintain wiring -------------------------------------------------


def test_maintain_runs_transition_when_flag_on(tmp_path: Path, monkeypatch):
    calls = {"n": 0}
    orig = _MaintainOpsMixin._transition_stale_memories

    def spy(self, *, dry_run=False):
        calls["n"] += 1
        return orig(self, dry_run=dry_run)

    monkeypatch.setattr(_MaintainOpsMixin, "_transition_stale_memories", spy)
    result = CliRunner().invoke(
        cli,
        ["maintain", "--json", "--skip-contradict", "--skip-consolidate", "--skip-stale"],
        env=_env(tmp_path, MEMO_VERIFICATION_STATE_TRACKING="1"),
    )
    assert result.exit_code == 0, result.output
    assert calls["n"] == 1
    receipt = json.loads(result.output)
    assert receipt["verification_transitioned"] == 0  # empty corpus


def test_maintain_skips_transition_when_flag_off(tmp_path: Path, monkeypatch):
    calls = {"n": 0}
    orig = _MaintainOpsMixin._transition_stale_memories

    def spy(self, *, dry_run=False):
        calls["n"] += 1
        return orig(self, dry_run=dry_run)

    monkeypatch.setattr(_MaintainOpsMixin, "_transition_stale_memories", spy)
    result = CliRunner().invoke(
        cli,
        ["maintain", "--json", "--skip-contradict", "--skip-consolidate", "--skip-stale"],
        env=_env(tmp_path),  # flag OFF
    )
    assert result.exit_code == 0, result.output
    assert calls["n"] == 0


# -- Part B: search wiring ---------------------------------------------------


def test_search_applies_decay_when_flag_on(mock_memory: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_VERIFICATION_STATE_TRACKING", "1")
    mock_memory.save(
        content="verification decay wiring probe token alpha", title="probe", type_="fact"
    )

    calls = {"n": 0}
    orig = _SearchScoringMixin._apply_verification_decay

    def spy(self, results):
        calls["n"] += 1
        return orig(self, results)

    monkeypatch.setattr(_SearchScoringMixin, "_apply_verification_decay", spy)
    res = mock_memory.search("verification decay wiring probe token alpha", limit=5, mode="bm25")
    assert res  # the fact matched
    assert calls["n"] >= 1


def test_search_skips_decay_when_flag_off(mock_memory: Memory, monkeypatch):
    monkeypatch.setenv("MEMO_VERIFICATION_STATE_TRACKING", "0")
    mock_memory.save(
        content="verification decay wiring probe token beta", title="probe2", type_="fact"
    )

    calls = {"n": 0}
    orig = _SearchScoringMixin._apply_verification_decay

    def spy(self, results):
        calls["n"] += 1
        return orig(self, results)

    monkeypatch.setattr(_SearchScoringMixin, "_apply_verification_decay", spy)
    mock_memory.search("verification decay wiring probe token beta", limit=5, mode="bm25")
    assert calls["n"] == 0


# -- Lifecycle: verified_at auto-stamp on reindex ----------------------------


def test_verified_without_timestamp_gets_stamped_on_reindex(mock_memory: Memory):
    rec = mock_memory.save(content="a fact that will be marked verified", title="vt", type_="fact")
    md_path = mock_memory.cfg.memory_dir / rec.path
    text = md_path.read_text(encoding="utf-8")
    # Insert `verification_state: verified` into the frontmatter, no verified_at.
    assert text.startswith("---")
    end = text.index("---", 3)
    md_path.write_text(text[:end] + "verification_state: verified\n" + text[end:], encoding="utf-8")

    mock_memory.reindex()

    updated = mock_memory.get(rec.id)
    assert updated.verification_state == VerificationState.VERIFIED
    assert updated.verified_at is not None  # decay clock started
