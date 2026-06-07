"""`memo maintain` — corpus-freshness orchestrator.

Unit-covers the older-side picker and the daily `--if-due` guard, plus a
dry-run on an empty isolated corpus (no MLX needed — nothing to embed).
"""

from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner

from memo.cli import cli
from memo.cli_maintain import _older_id


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def test_older_id_picks_earlier_updated():
    recs = {
        "a": SimpleNamespace(updated="2026-01-01T00:00:00+00:00"),
        "b": SimpleNamespace(updated="2026-05-01T00:00:00+00:00"),
    }
    mem = SimpleNamespace(get=lambda i: recs.get(i))
    assert _older_id(mem, "a", "b") == ("a", "b")
    assert _older_id(mem, "b", "a") == ("a", "b")  # order-independent


def test_older_id_falls_back_to_pair_order_when_missing():
    mem = SimpleNamespace(get=lambda i: None)
    assert _older_id(mem, "x", "y") == ("x", "y")


def test_if_due_is_noop_when_recently_run(tmp_path: Path):
    state = tmp_path / "state"
    (state / "maintain").mkdir(parents=True)
    (state / "maintain" / ".last_run_ts").write_text(str(time.time()), encoding="utf-8")

    result = CliRunner().invoke(cli, ["maintain", "--if-due"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    # not due → no work, no output
    assert result.output.strip() == ""


def test_if_due_disabled_by_env(tmp_path: Path):
    result = CliRunner().invoke(
        cli, ["maintain", "--if-due"], env={**_env(tmp_path), "MEMO_MAINTAIN_DISABLE": "1"})
    assert result.exit_code == 0, result.output
    assert result.output.strip() == ""


def test_dry_run_on_empty_corpus_is_safe_noop(tmp_path: Path):
    result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    import json
    receipt = json.loads(result.output)
    assert receipt["dry_run"] is True
    assert receipt["superseded"] == []
    assert receipt["merged"] == []
    assert receipt["archived_stale"] == []


# -- Memory.lint() (was untested) ------------------------------------------


def test_lint_empty_corpus_returns_empty_categories(mock_memory):
    out = mock_memory.lint()
    assert out == {
        "legacy_extra": [],
        "few_tags": [],
        "body_skinny": [],
        "untitled": [],
    }


def test_lint_flags_few_tags_and_skinny_body(mock_memory):
    rec = mock_memory.save(content="short", title="Tiny", tags=["only-one"])
    out = mock_memory.lint()
    assert rec.id in {e["id"] for e in out["few_tags"]}
    assert rec.id in {e["id"] for e in out["body_skinny"]}


def test_lint_flags_legacy_extra_fields(mock_memory):
    rec = mock_memory.save(
        content="x" * 200,
        title="Has legacy fields",
        tags=["a", "b", "c"],
        extra={"usage_count": 5, "agent_id": "old"},
    )
    legacy = {e["id"]: e for e in mock_memory.lint()["legacy_extra"]}
    assert rec.id in legacy
    assert "usage_count" in legacy[rec.id]["reason"]
    assert "agent_id" in legacy[rec.id]["reason"]


def test_lint_clean_memoria_has_no_issues(mock_memory):
    rec = mock_memory.save(
        content="x" * 200,
        title="Well formed memoria",
        tags=["project:memo", "domain:test", "technique:unit"],
    )
    out = mock_memory.lint()
    flagged = {e["id"] for cat in out.values() for e in cat}
    assert rec.id not in flagged
