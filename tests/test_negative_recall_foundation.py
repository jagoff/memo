"""Foundation tests for the Negative Recall ⛔ AVOID channel.

Covers only the pure helpers in ``memo.negative_recall`` (parse / render /
derive / risky-context), the flag registry defaults, and the graduation-gate
completeness for the new ``*_ENABLED`` flags. No MLX, no store, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memo import dream_flags as df
from memo import negative_recall as nr
from memo.flags import flag_bool, flag_float, flag_int
from memo.tiers import DURABLE_TYPES


@dataclass(frozen=True)
class _Hit:
    """Structural stand-in for a MemoryRecord (matches nr.AvoidHit)."""

    id: str
    title: str
    body: str
    extra: dict[str, Any] = field(default_factory=dict)


def _isolated_env(tmp_path: Path) -> dict[str, str]:
    """Env with no markdown config / no tuned overlay, so registered defaults win."""
    return {
        "MEMO_CONFIG_DIR": str(tmp_path / "no-md-config"),
        "MEMO_STATE_DIR": str(tmp_path / "no-overlay-state"),
    }


# ── parse_failure_pattern ────────────────────────────────────────────────────


def test_parse_well_formed_body_extracts_all_four_fields() -> None:
    body = (
        "Pattern: bump does not regen uv.lock\n"
        "Context: cutting a memo release\n"
        "Wrong: ran memo release bump and pushed\n"
        "Right: regenerate uv.lock and .mcpb before tagging"
    )
    parsed = nr.parse_failure_pattern(body)
    assert parsed is not None
    assert parsed.pattern == "bump does not regen uv.lock"
    assert parsed.context == "cutting a memo release"
    assert parsed.wrong == "ran memo release bump and pushed"
    assert parsed.right == "regenerate uv.lock and .mcpb before tagging"
    assert parsed.is_actionable is True


def test_parse_folds_provenance_from_extra() -> None:
    body = "Pattern: p\nWrong: w\nRight: r"
    extra = {nr.FP_SOURCE_KEY: nr.FP_SOURCE_SUPERSEDE, nr.FP_LINKS_KEY: ["aaaa1111", "bbbb2222"]}
    parsed = nr.parse_failure_pattern(body, extra)
    assert parsed is not None
    assert parsed.source == nr.FP_SOURCE_SUPERSEDE
    assert parsed.links == ("aaaa1111", "bbbb2222")


def test_parse_joins_continuation_lines_until_blank() -> None:
    body = "Pattern: first part\ncontinued on the next line\n\nRight: the fix"
    parsed = nr.parse_failure_pattern(body)
    assert parsed is not None
    assert parsed.pattern == "first part continued on the next line"
    assert parsed.right == "the fix"
    assert parsed.context == ""  # absent label -> empty


def test_parse_returns_none_when_no_labels_present() -> None:
    assert nr.parse_failure_pattern("just some prose with no labels") is None
    assert nr.parse_failure_pattern("") is None


def test_parse_git_miner_shaped_body_keeps_only_labelled_lines() -> None:
    # git_miner emits Pattern/Context then a blank line + free body + Files line.
    body = (
        "Pattern: race in write coordinator\n"
        "Context: repo memo, commit deadbeef (2026-07-01)\n"
        "\n"
        "some free-form explanation that is not a labelled field\n"
        "Files: a.py, b.py"
    )
    parsed = nr.parse_failure_pattern(body)
    assert parsed is not None
    assert parsed.pattern == "race in write coordinator"
    assert parsed.context == "repo memo, commit deadbeef (2026-07-01)"
    assert parsed.wrong == ""
    assert parsed.right == ""


# ── format_avoid_block ───────────────────────────────────────────────────────


def test_format_avoid_block_empty_hits_returns_empty_string() -> None:
    assert nr.format_avoid_block([]) == ""


def test_format_avoid_block_renders_header_wrong_and_right() -> None:
    hit = _Hit(
        id="abcd1234ef",
        title="release bump trap",
        body="Pattern: p\nContext: c\nWrong: pushed without regen\nRight: regen first",
    )
    block = nr.format_avoid_block([hit])
    assert block.startswith(nr.AVOID_BLOCK_HEADER)
    assert "1. [abcd1234] release bump trap" in block
    assert "✗ pushed without regen" in block
    assert "✓ regen first" in block


def test_format_avoid_block_stays_off_cognition() -> None:
    hit = _Hit(id="a" * 32, title="t", body="Pattern: p\nWrong: w\nRight: r")
    block = nr.format_avoid_block([hit]).lower()
    for verb in ("suggest", "you should", "agent", "recommend", "must "):
        assert verb not in block


def test_format_avoid_block_falls_back_to_body_when_unstructured() -> None:
    hit = _Hit(id="deadbeef", title="loose note", body="no labelled structure here")
    block = nr.format_avoid_block([hit])
    assert "1. [deadbeef] loose note" in block
    assert "no labelled structure here" in block
    assert "✗" not in block and "✓" not in block


def test_format_avoid_block_truncates_long_fields() -> None:
    long_wrong = "x" * 500
    hit = _Hit(id="abcd1234", title="t", body=f"Pattern: p\nWrong: {long_wrong}\nRight: r")
    block = nr.format_avoid_block([hit], max_field_chars=50)
    wrong_line = next(ln for ln in block.splitlines() if ln.strip().startswith("✗"))
    assert wrong_line.strip().endswith("…")
    assert len(wrong_line.strip()) <= 55  # "✗ " prefix + 50 chars + ellipsis margin


# ── derive_failure_pattern_from_supersede ────────────────────────────────────


def test_derive_from_supersede_shape_and_provenance() -> None:
    superseded = _Hit(id="1111aaaa", title="use Ollama", body="we embed via Ollama")
    superseding = _Hit(id="2222bbbb", title="use MLX", body="we embed via MLX now")
    payload = nr.derive_failure_pattern_from_supersede(superseded, superseding)

    assert payload["type"] == nr.FAILURE_PATTERN_TYPE
    assert payload["extra"][nr.FP_SOURCE_KEY] == nr.FP_SOURCE_SUPERSEDE
    assert payload["extra"]["wrong_id"] == "1111aaaa"
    assert payload["extra"]["right_id"] == "2222bbbb"
    assert payload["extra"][nr.FP_LINKS_KEY] == ["1111aaaa", "2222bbbb"]
    assert "negative-recall" in payload["tags"]
    assert len(payload["title"]) <= 80

    # Round-trips: the derived body parses back with the two approaches.
    parsed = nr.parse_failure_pattern(payload["body"], payload["extra"])
    assert parsed is not None
    assert parsed.wrong == "we embed via Ollama"
    assert parsed.right == "we embed via MLX now"
    assert parsed.source == nr.FP_SOURCE_SUPERSEDE


def test_derive_from_supersede_collapses_multiline_bodies() -> None:
    superseded = _Hit(id="1111aaaa", title="old", body="line one\nline two\n\nline three")
    superseding = _Hit(id="2222bbbb", title="new", body="do this")
    payload = nr.derive_failure_pattern_from_supersede(superseded, superseding)
    parsed = nr.parse_failure_pattern(payload["body"])
    assert parsed is not None
    assert "\n" not in parsed.wrong
    assert parsed.wrong == "line one line two line three"


# ── derive_failure_pattern_from_avoid_verdict ────────────────────────────────


def test_derive_from_avoid_verdict_shape_and_provenance() -> None:
    memory = _Hit(id="cafe1234", title="prefer git rebase", body="always rebase before push")
    verdict = {
        "verdict": "correction",
        "prompt": "why did the push fail?",
        "reaction": "no — force-push was wrong here, use --force-with-lease",
        "turn": 7,
        "session_id": "sess-9",
    }
    payload = nr.derive_failure_pattern_from_avoid_verdict(memory, verdict)

    assert payload["type"] == nr.FAILURE_PATTERN_TYPE
    assert payload["extra"][nr.FP_SOURCE_KEY] == nr.FP_SOURCE_AVOID_VERDICT
    assert payload["extra"]["origin_id"] == "cafe1234"
    assert payload["extra"][nr.FP_LINKS_KEY] == ["cafe1234"]
    assert payload["extra"]["verdict"] == "correction"
    assert payload["extra"]["verdict_turn"] == 7
    assert payload["extra"]["verdict_session"] == "sess-9"

    parsed = nr.parse_failure_pattern(payload["body"], payload["extra"])
    assert parsed is not None
    assert "always rebase before push" in parsed.wrong
    assert "force-with-lease" in parsed.right


def test_derive_from_avoid_verdict_falls_back_without_reaction() -> None:
    memory = _Hit(id="cafe1234", title="t", body="b")
    payload = nr.derive_failure_pattern_from_avoid_verdict(memory, {"verdict": "negative"})
    parsed = nr.parse_failure_pattern(payload["body"])
    assert parsed is not None
    assert "verify before relying on it" in parsed.right
    # Optional turn/session keys omitted when absent.
    assert "verdict_turn" not in payload["extra"]
    assert "verdict_session" not in payload["extra"]


# ── risky_context ────────────────────────────────────────────────────────────


def test_risky_context_scores_release_and_delete_prompts() -> None:
    assert nr.risky_context("please cut a release and delete the old tag") > 0.0
    assert nr.risky_context("rm -rf the build dir") > 0.0
    assert nr.risky_context("reset --hard to origin/master") > 0.0


def test_risky_context_benign_prompt_is_zero() -> None:
    assert nr.risky_context("what does this function return?") == 0.0
    assert nr.risky_context("") == 0.0


def test_risky_context_is_graded_and_saturates() -> None:
    one = nr.risky_context("let's do a release")
    many = nr.risky_context("delete the prod database, force-push, and migrate everything")
    assert 0.0 < one < many
    assert many == 1.0
    assert one <= 1.0


# ── constants + flags + gates ────────────────────────────────────────────────


def test_failure_pattern_type_matches_durable_tier() -> None:
    assert nr.FAILURE_PATTERN_TYPE in DURABLE_TYPES


def test_new_flags_default_off(tmp_path: Path) -> None:
    env = _isolated_env(tmp_path)
    assert flag_bool("MEMO_NEGATIVE_RECALL_ENABLED", env=env) is False
    assert flag_bool("MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED", env=env) is False
    assert flag_bool("MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED", env=env) is False
    assert flag_bool("MEMO_NEGATIVE_RECALL_TRIGGER_ENABLED", env=env) is False
    assert flag_int("MEMO_NEGATIVE_RECALL_K", env=env) == 2
    assert flag_float("MEMO_NEGATIVE_RECALL_MIN_SIM", env=env) == 0.6


def test_new_enabled_flags_declare_manual_gates() -> None:
    for name in (
        "MEMO_NEGATIVE_RECALL_ENABLED",
        "MEMO_NEGATIVE_RECALL_CAPTURE_ENABLED",
        "MEMO_NEGATIVE_RECALL_REINFORCE_ENABLED",
        "MEMO_NEGATIVE_RECALL_TRIGGER_ENABLED",
    ):
        assert name in df.GATES, f"{name} missing a graduation gate"
        assert df.GATES[name].kind == "manual"
