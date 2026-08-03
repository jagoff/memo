"""`memo dream ledger|index-health|staging|shadow` — standalone CLI commands.

The pass logic itself is unit-tested elsewhere (test_dream_ledger.py,
test_dream_staging.py, test_dream_shadow.py, test_store_index_health.py);
these tests exercise the CLI rendering (plain-text and --json) and flag
plumbing (--open, --drop, --resume, --promote, --reject).
"""

from __future__ import annotations

import json
from unittest.mock import patch

from click.testing import CliRunner

from memo import dream_ledger, dream_staging
from memo.cli_dream import dream_cmd


def _env(tmp_path) -> dict[str, str]:
    return {
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_NONINTERACTIVE": "1",
    }


# -- dream ledger ------------------------------------------------------------


def test_ledger_empty_state_plain_text(tmp_path):
    result = CliRunner().invoke(dream_cmd, ["ledger"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "dream ledger:" in result.output
    assert "enable MEMO_DREAM_LEDGER_ENABLED" in result.output


def test_ledger_renders_action_and_outcome_entries(tmp_path):
    state = tmp_path / "state"
    dream_ledger.record_action(
        state,
        action="merge",
        pass_name="consolidate",
        affected_ids=["abc12345"],
    )

    result = CliRunner().invoke(dream_cmd, ["ledger"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "merge" in result.output
    assert "consolidate" in result.output
    assert "abc12345"[:8] in result.output


def test_ledger_json_output_has_summary_and_entries(tmp_path):
    state = tmp_path / "state"
    dream_ledger.record_action(state, action="archive", pass_name="prune_floor")

    result = CliRunner().invoke(dream_cmd, ["ledger", "--json"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert set(payload) == {"summary", "entries"}
    assert payload["summary"]["actions"] == 1


def test_ledger_open_filters_to_unresolved_actions(tmp_path):
    state = tmp_path / "state"
    dream_ledger.record_action(state, action="merge", pass_name="consolidate")

    result = CliRunner().invoke(dream_cmd, ["ledger", "--open", "--json"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert len(payload["entries"]) == 1


# -- dream index-health -------------------------------------------------------


def test_index_health_plain_text_renders_checks(mock_memory, tmp_path):
    with (
        patch("memo.cli_dream._get_memory", return_value=mock_memory),
        patch(
            "memo.store.index_health.check_index_health",
            return_value={
                "status": "ok",
                "checks": {"orphan_chunks": {"count": 0}, "wrong_dims": {"count": 2}},
                "errors": [],
            },
        ),
    ):
        result = CliRunner().invoke(dream_cmd, ["index-health"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "index health:" in result.output
    assert "orphan_chunks" in result.output
    assert "wrong_dims" in result.output


def test_index_health_json_and_repair_flag(mock_memory, tmp_path):
    captured = {}

    def fake_check(cfg, mem, *, repair=False):
        captured["repair"] = repair
        return {"status": "ok", "checks": {}, "errors": [], "repaired": 3}

    with (
        patch("memo.cli_dream._get_memory", return_value=mock_memory),
        patch("memo.store.index_health.check_index_health", fake_check),
    ):
        result = CliRunner().invoke(
            dream_cmd, ["index-health", "--repair", "--json"], env=_env(tmp_path)
        )

    assert result.exit_code == 0, result.output
    assert captured["repair"] is True
    assert json.loads(result.output)["repaired"] == 3


def test_index_health_plain_text_shows_errors(mock_memory, tmp_path):
    with (
        patch("memo.cli_dream._get_memory", return_value=mock_memory),
        patch(
            "memo.store.index_health.check_index_health",
            return_value={"status": "issues", "checks": {}, "errors": ["orphan_chunks: boom"]},
        ),
    ):
        result = CliRunner().invoke(dream_cmd, ["index-health"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "orphan_chunks: boom" in result.output


# -- dream staging -------------------------------------------------------------


def test_staging_empty_state_plain_text(tmp_path):
    result = CliRunner().invoke(dream_cmd, ["staging"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "no staged proposals" in result.output


def _seed_staged_proposal(cfg) -> str:
    proposal = dream_staging.StagedProposal(
        proposal_id="prop-1",
        kind="synthesis",
        source_ids=("s1",),
        save_kwargs={"content": "body", "title": "t"},
        conflict_ids=("conflict-abc",),
        conflict_summary="semantic_contradiction",
        evidence_uris=(),
        staged_at="2026-01-01T00:00:00",
        state="staged",
        attempts=1,
    )
    dream_staging._save(cfg, [proposal])
    return proposal.proposal_id


def test_staging_lists_parked_proposals(tmp_path):
    from memo.config import Config

    with patch.dict("os.environ", _env(tmp_path)):
        cfg = Config.from_env()
    _seed_staged_proposal(cfg)

    result = CliRunner().invoke(dream_cmd, ["staging"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "prop-1" in result.output
    assert "synthesis" in result.output
    assert "semantic_contradiction" in result.output


def test_staging_json_output(tmp_path):
    from memo.config import Config

    with patch.dict("os.environ", _env(tmp_path)):
        cfg = Config.from_env()
    _seed_staged_proposal(cfg)

    result = CliRunner().invoke(dream_cmd, ["staging", "--json"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload[0]["proposal_id"] == "prop-1"


def test_staging_drop_removes_proposal(tmp_path):
    from memo.config import Config

    with patch.dict("os.environ", _env(tmp_path)):
        cfg = Config.from_env()
    _seed_staged_proposal(cfg)

    result = CliRunner().invoke(dream_cmd, ["staging", "--drop", "prop-1"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "dropped prop-1" in result.output
    assert dream_staging.list_staged(cfg) == []


def test_staging_drop_nonexistent_reports_not_found(tmp_path):
    result = CliRunner().invoke(dream_cmd, ["staging", "--drop", "missing"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "not found" in result.output


def test_staging_resume_invokes_pass_and_reports_result(mock_memory, tmp_path):
    with (
        patch("memo.cli_dream._get_memory", return_value=mock_memory),
        patch(
            "memo.dream_staging.resume_staged_proposals",
            return_value={"reapplied": 2, "dropped": 0},
        ),
    ):
        result = CliRunner().invoke(dream_cmd, ["staging", "--resume"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "resumed" in result.output
    assert "reapplied" in result.output


# -- dream shadow ---------------------------------------------------------------


def test_shadow_no_flags_declared_plain_text(tmp_path):
    with patch("memo.dream_shadow.review_rows", return_value=[]):
        result = CliRunner().invoke(dream_cmd, ["shadow"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "no shadow-kind flags declared" in result.output


def test_shadow_status_renders_review_rows(tmp_path):
    rows = [
        {
            "flag": "MEMO_SOME_SHADOW_FLAG",
            "review_ready": True,
            "streak": 5,
            "review_nights": 5,
            "mean_delta": 0.02,
            "cost_p50": 120,
            "last_verdict": "clean",
        }
    ]
    with patch("memo.dream_shadow.review_rows", return_value=rows):
        result = CliRunner().invoke(dream_cmd, ["shadow", "--status"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert "MEMO_SOME_SHADOW_FLAG" in result.output
    assert "review-ready" in result.output


def test_shadow_json_output(tmp_path):
    rows = [{"flag": "F", "review_ready": False}]
    with patch("memo.dream_shadow.review_rows", return_value=rows):
        result = CliRunner().invoke(dream_cmd, ["shadow", "--json"], env=_env(tmp_path))

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == rows


def test_shadow_reject_records_and_reports(tmp_path):
    result = CliRunner().invoke(
        dream_cmd,
        ["shadow", "--reject", "MEMO_SOME_FLAG", "--reason", "too risky"],
        env=_env(tmp_path),
    )

    assert result.exit_code == 0, result.output
    assert "rejected MEMO_SOME_FLAG" in result.output


def test_shadow_promote_invokes_pass_with_flags(tmp_path):
    captured = {}

    def fake_promote(cfg, flag, *, force_latency=False, apply=False):
        captured.update(flag=flag, force_latency=force_latency, apply=apply)
        return {"flag": flag, "ok": True, "applied": apply}

    with patch("memo.dream_shadow.promote", fake_promote):
        result = CliRunner().invoke(
            dream_cmd,
            ["shadow", "--promote", "MEMO_SOME_FLAG", "--apply", "--force-latency"],
            env=_env(tmp_path),
        )

    assert result.exit_code == 0, result.output
    assert captured == {"flag": "MEMO_SOME_FLAG", "force_latency": True, "apply": True}
    assert "'ok': True" in result.output
