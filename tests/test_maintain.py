"""`memo maintain` — corpus-freshness orchestrator.

Unit-covers the older-side picker and the daily `--if-due` guard, plus a
dry-run on an empty isolated corpus (no MLX needed — nothing to embed).
Includes tests for the proactive synthesis pass (MEMO_MAINT_SYNTHESIZE=1).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from memo.cli import cli
from memo.cli_maintain import (
    _older_id,
    _read_synthesis_last_run,
    _synthesis_state_path,
    _write_synthesis_last_run,
)
from memo.config import Config


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
        cli, ["maintain", "--if-due"], env={**_env(tmp_path), "MEMO_MAINTAIN_DISABLE": "1"}
    )
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


# -- Proactive synthesis (MEMO_MAINT_SYNTHESIZE) --------------------------------


def test_synthesis_state_helpers_round_trip(tmp_path: Path):
    """_write / _read synthesis_state.json persist and return the ISO timestamp."""
    state = tmp_path / "state"
    state.mkdir()
    cfg = Config(data_dir=tmp_path / "data", state_dir=state, reranker_enabled=False)

    assert _read_synthesis_last_run(cfg) is None  # no file yet

    ts = "2026-06-13T12:00:00+00:00"
    _write_synthesis_last_run(cfg, ts)

    assert _synthesis_state_path(cfg).is_file()
    stored = json.loads(_synthesis_state_path(cfg).read_text(encoding="utf-8"))
    assert stored["last_run"] == ts
    assert _read_synthesis_last_run(cfg) == ts


def test_maint_synthesize_flag_off_no_state_file(tmp_path: Path):
    """When MEMO_MAINT_SYNTHESIZE is not set, synthesis_state.json is never created."""
    env = _env(tmp_path)
    # Do NOT set MEMO_MAINT_SYNTHESIZE
    result = CliRunner().invoke(cli, ["maintain", "--json"], env=env)
    assert result.exit_code == 0, result.output
    state_file = tmp_path / "state" / "synthesis_state.json"
    assert not state_file.exists()


def test_maint_synthesize_flag_on_creates_state_file(tmp_path: Path):
    """With MEMO_MAINT_SYNTHESIZE=1, maintain creates synthesis_state.json and
    includes synthesis_count in the JSON receipt.

    The corpus is empty (no memorias to cluster), so synthesis_count=0,
    but the timestamp file must still be written after a real (non-dry-run) run.
    """
    env = {
        **_env(tmp_path),
        "MEMO_MAINT_SYNTHESIZE": "1",
        "MEMO_SYNTHESIS_ENABLED": "0",  # disable step-4 synthesis to avoid double-run
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }
    result = CliRunner().invoke(cli, ["maintain", "--json"], env=env)
    assert result.exit_code == 0, result.output

    receipt = json.loads(result.output)
    assert "synthesis_count" in receipt
    assert receipt["synthesis_count"] == 0  # empty corpus → no clusters

    state_file = tmp_path / "state" / "synthesis_state.json"
    assert state_file.is_file(), "synthesis_state.json must be written after the maintain run"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    assert "last_run" in data
    assert data["last_run"]  # non-empty ISO timestamp


def test_maint_synthesize_dry_run_does_not_write_state(tmp_path: Path):
    """A --dry-run maintain with MEMO_MAINT_SYNTHESIZE=1 must NOT persist the state file."""
    env = {
        **_env(tmp_path),
        "MEMO_MAINT_SYNTHESIZE": "1",
        "MEMO_SYNTHESIS_ENABLED": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }
    result = CliRunner().invoke(cli, ["maintain", "--dry-run", "--json"], env=env)
    assert result.exit_code == 0, result.output

    state_file = tmp_path / "state" / "synthesis_state.json"
    assert not state_file.exists(), "dry-run must not write synthesis_state.json"


def test_maint_synthesize_non_fatal_on_error(tmp_path: Path):
    """If synthesize_cross_cluster raises, maintain logs a warning and continues."""
    from memo.cli_maintain import _synthesis_state_path

    env = {
        **_env(tmp_path),
        "MEMO_MAINT_SYNTHESIZE": "1",
        "MEMO_SYNTHESIS_ENABLED": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }

    # Patch Memory.synthesize_cross_cluster to raise.
    with patch("memo.memory.Memory.synthesize_cross_cluster", side_effect=RuntimeError("boom")):
        result = CliRunner().invoke(cli, ["maintain", "--json"], env=env)

    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    # Error should be recorded but not fatal.
    assert any("maint_synthesize" in e for e in receipt.get("errors", []))
    # State file should NOT be written when synthesis crashed.
    assert not _synthesis_state_path(
        Config(
            data_dir=tmp_path / "data",
            state_dir=tmp_path / "state",
            reranker_enabled=False,
        )
    ).exists()


def test_maintain_writes_timestamped_run_receipt(tmp_cfg):
    import json as _json

    from click.testing import CliRunner

    from memo.cli_maintain import maintain_cmd

    runner = CliRunner()
    res = runner.invoke(
        maintain_cmd,
        ["--skip-contradict", "--skip-consolidate", "--skip-stale", "--skip-synthesize", "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
            "MEMO_OUTCOME_RANKING_ENABLED": "0",
        },
    )
    assert res.exit_code == 0, res.output
    runs = list((tmp_cfg.state_dir / "maintain" / "runs").glob("*.json"))
    assert len(runs) == 1
    receipt = _json.loads(runs[0].read_text(encoding="utf-8"))
    assert receipt["run"] == runs[0].stem
    # last.json still written (daily guard + undo default read it)
    assert (tmp_cfg.state_dir / "maintain" / "last.json").is_file()


def test_undo_targets_and_restore_from_inactive(mock_memory):
    from memo.cli_maintain import _restore_archived, _undo_targets

    a = mock_memory.save(content="stale one", title="Stale")
    qc = mock_memory.save(content="compact me", title="Compact")
    f = mock_memory.save(content="dead weight", title="Dead")
    assert mock_memory.lifecycle.archive_memory(a.id) is True
    assert mock_memory.lifecycle.archive_memory(qc.id, superseded_by="canonical") is True
    assert mock_memory.get(a.id) is None
    assert mock_memory.get(qc.id) is None
    mock_memory.forget(f.id, reason="test")

    receipt = {
        "superseded": [],
        "merged": [],
        "archived_stale": [{"id": a.id, "days": 400}],
        "quality_compacted": [{"proposal_id": "quality-compact-demo", "archived_ids": [qc.id]}],
        "dead_archived": [f.id],
        "forgotten": [],
    }
    archived, forgotten = _undo_targets(receipt)
    assert archived == [a.id, qc.id] and forgotten == [f.id]

    restored, missing = _restore_archived(mock_memory, archived, dry_run=False)
    assert set(restored) == {a.id, qc.id} and missing == []
    mock_memory.reindex()
    assert mock_memory.get(a.id) is not None
    assert mock_memory.get(qc.id) is not None
    assert mock_memory.unforget(f.id) is not None


def test_maintain_undo_cli_dry_run_reads_receipt(tmp_cfg):
    import json as _json

    from click.testing import CliRunner

    from memo.cli_maintain import maintain_cmd

    d = tmp_cfg.state_dir / "maintain"
    d.mkdir(parents=True, exist_ok=True)
    (d / "last.json").write_text(
        _json.dumps(
            {
                "ts": 1.0,
                "superseded": [{"older": "a" * 32, "action": "archive"}],
                "merged": [],
                "archived_stale": [],
                "dead_archived": [],
                "forgotten": [],
            }
        ),
        encoding="utf-8",
    )
    runner = CliRunner()
    res = runner.invoke(
        maintain_cmd,
        ["undo", "--dry-run", "--json"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
            "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        },
    )
    assert res.exit_code == 0, res.output
    out = _json.loads(res.output)
    assert out["dry_run"] is True
    assert out["missing"] == ["a" * 32]  # nothing in inactive/ to restore
