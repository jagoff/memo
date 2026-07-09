from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli
from memo.quality_compact import preview_quality_compaction


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_QUALITY_COMPACT": "1",
        "MEMO_RERANKER_ENABLED": "0",
    }


def test_quality_compact_preview_empty_corpus_is_read_only(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--preview", "--json"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "preview"
    assert payload["proposals"] == []
    assert payload["applied"] == []


def test_quality_compact_preview_skips_cross_scope_and_sensitive_records(mock_memory) -> None:
    canonical = mock_memory.save(
        content="Stable canonical memory.",
        title="Canonical",
        tags=["project:memo"],
    )
    included_a = mock_memory.save(
        content="First duplicate.",
        title="First duplicate",
        tags=["project:memo"],
        extra={"canonical_id": canonical.id},
    )
    included_b = mock_memory.save(
        content="Second duplicate.",
        title="Second duplicate",
        tags=["project:memo"],
        extra={"superseded_by": canonical.id},
    )
    mock_memory.save(
        content="Other project duplicate.",
        title="Other project duplicate",
        tags=["project:other"],
        extra={"canonical_id": canonical.id},
    )
    mock_memory.save(
        content="Top secret duplicate.",
        title="Secret duplicate",
        type_="secret",
        tags=["project:memo"],
        extra={"canonical_id": canonical.id},
    )

    payload = preview_quality_compaction(mock_memory, limit=20)

    assert payload["mode"] == "preview"
    assert payload["applied"] == []
    assert payload["errors"] == []
    assert len(payload["proposals"]) == 1
    proposal = payload["proposals"][0]
    assert proposal["canonical_title"] == "Canonical"
    assert set(proposal["source_ids"]) == {included_a.id, included_b.id}
    assert proposal["scope"] == "project:memo"
    assert proposal["reasons"] == ["explicit_canonical_or_superseded_by"]


def test_quality_compact_preview_skips_ambiguous_multi_project_scope(mock_memory) -> None:
    canonical = mock_memory.save(
        content="Stable canonical memory.",
        title="Canonical",
        tags=["project:memo"],
    )
    ambiguous = mock_memory.save(
        content="Ambiguous duplicate.",
        title="Ambiguous duplicate",
        tags=["project:memo", "project:other"],
        extra={"canonical_id": canonical.id},
    )

    payload = preview_quality_compaction(mock_memory, limit=20)

    assert payload["proposals"] == []
    assert payload["errors"] == [f"ambiguous_scope:{ambiguous.id}"]


def test_quality_compact_preview_skips_unresolved_canonical_targets(mock_memory) -> None:
    missing_canonical_id = "f" * 32
    mock_memory.save(
        content="Duplicate with missing canonical.",
        title="Missing canonical duplicate",
        tags=["project:memo"],
        extra={"canonical_id": missing_canonical_id},
    )

    payload = preview_quality_compaction(mock_memory, limit=20)

    assert payload["proposals"] == []
    assert payload["errors"] == [f"unresolved_canonical:{missing_canonical_id}"]


def test_quality_compact_command_requires_flag(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.pop("MEMO_QUALITY_COMPACT")
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--preview", "--json"],
        env=env,
    )
    assert result.exit_code != 0
    assert "MEMO_QUALITY_COMPACT=1 is required" in result.output


def test_quality_compact_apply_is_rejected_in_task_5(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--apply"],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "quality compaction apply is not implemented yet; use --preview" in result.output
