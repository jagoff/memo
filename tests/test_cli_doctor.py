import json
import sqlite3
from contextlib import closing
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.memory import Memory
from memo.trust_preflight import trust_preflight


def _agent_report(*, ok: bool) -> dict:
    return {
        "ok": ok,
        "checks": {
            "detected": True,
            "mcp_configured": ok,
            "mcp_runtime_current": ok,
            "runtime_isolated": True,
            "runtime_pair": True,
            "runtime_version": "3.12.1",
            "runtime_version_match": ok,
            "runtime_smoke": ok,
            "storage_writable": True,
            "profile": "core",
            "profile_current": ok,
            "protocol_mode": "compact",
            "protocol_current": ok,
            "instruction_marker": True,
            "instruction_writable": True,
        },
    }


def test_doctor_never_prints_a_fabricated_tokens_saved_line(tmp_path):
    """Round-2 regression: `compute_roi()` no longer returns `tokens_saved_human`
    (round-1 removed the field), but doctor read it via `.get(...) or "0"`,
    printing a literal fabricated "✓ tokens saved: ~0" precisely when there
    WAS grounded data (the `_grounded > 0` gate)."""
    from memo.cli import cli
    from memo.dashboard import append_grounding_log, append_recall_log

    state = tmp_path / "state"
    state.mkdir(parents=True)
    append_recall_log(
        state,
        prompt="how do I configure the deploy pipeline",
        via="subprocess",
        session_id="s",
        turn=1,
        client="claude-code",
        hits=[{"id": "a" * 8, "score": 0.8, "title": "t", "snippet": "x"}],
    )
    append_grounding_log(
        state,
        session_id="s",
        turn=1,
        recall_id="a" * 8,
        used_score=0.9,
        method="lexical",
        client="claude-code",
    )

    (tmp_path / "memorias").mkdir()
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "memorias"),
        "MEMO_STATE_DIR": str(state),
    }
    r = CliRunner().invoke(cli, ["doctor"], env=env)
    assert "tokens saved" not in r.output.lower()


def test_doctor_off_hint_points_at_sync_setup(tmp_path):
    from memo.cli import cli

    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "memorias"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    (tmp_path / "memorias").mkdir()
    r = CliRunner().invoke(cli, ["doctor"], env=env)
    # data_dir is not a git clone here → the OFF branch fires
    assert "memo sync setup" in r.output


def test_doctor_json_combines_agent_health(monkeypatch):
    from memo.cli import cli

    monkeypatch.setattr("memo.cli_doctor.Config.from_env", classmethod(lambda _cls: MagicMock()))
    monkeypatch.setattr(
        "memo.cli_doctor._doctor_report",
        lambda *_args, **_kwargs: {"ok": True, "runtime": {}},
    )
    monkeypatch.setattr(
        "memo.runtime.agent_registry.verify_agent",
        lambda _agent: _agent_report(ok=False),
    )

    result = CliRunner().invoke(cli, ["doctor", "--agent", "codex", "--json"])
    payload = json.loads(result.output)

    assert result.exit_code == 1
    assert payload["ok"] is False
    assert payload["agent_setup"][0]["checks"]["profile"] == "core"


def test_doctor_text_agent_failure_shows_repair(monkeypatch, tmp_path):
    from memo.cli import cli

    data_dir = tmp_path / "memorias"
    state_dir = tmp_path / "state"
    data_dir.mkdir()
    state_dir.mkdir()
    monkeypatch.setattr(
        "memo.runtime.agent_registry.verify_agent",
        lambda _agent: _agent_report(ok=False),
    )

    result = CliRunner().invoke(
        cli,
        ["doctor", "--agent", "codex"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(data_dir),
            "MEMO_STATE_DIR": str(state_dir),
        },
    )

    assert result.exit_code == 1
    assert "agent:codex" in result.output
    assert "version-match=" in result.output
    assert "repair with `memo setup codex`" in result.output


def test_gc_report_closes_memory_after_scan():
    from memo.cli_doctor import _gc_report

    cfg = MagicMock()
    memory = MagicMock()
    expected = {"orphan_store": [], "orphan_disk": [], "stale_synthesis": []}
    memory.gc.return_value = expected

    with patch("memo.memory.Memory", return_value=memory):
        report = _gc_report(cfg, fix=False)

    assert report == expected
    memory.gc.assert_called_once_with(fix=False)
    memory.close.assert_called_once_with()


def test_trust_preflight_detects_privacy_findings_without_disclosure(
    mem_with_stub: Memory, monkeypatch
):
    from memo.cli import cli

    record = mem_with_stub.save(
        content="safe indexed body",
        title="Trust preflight",
        auto_project=False,
    )
    assert trust_preflight(mem_with_stub.cfg)["ok"] is True

    token = "sk-" + "DoctorCanary123456789012345"
    private_canary = "DOCTOR_PRIVATE_CANARY"
    path = mem_with_stub.cfg.memory_dir / record.path
    path.write_text(
        path.read_text(encoding="utf-8")
        + f"\nmanual edit {token} <private>{private_canary}</private>\n",
        encoding="utf-8",
    )
    report = trust_preflight(mem_with_stub.cfg)
    assert report["ok"] is False
    assert report["secret_pattern_files"] == 1
    assert report["private_marker_files"] == 1

    monkeypatch.setattr(
        "memo.cli_doctor.Config.from_env", classmethod(lambda _cls: mem_with_stub.cfg)
    )
    result = CliRunner().invoke(cli, ["doctor", "--db", "--json"])
    serialized = result.output
    payload = json.loads(serialized)
    assert payload["trust"]["secret_pattern_files"] == 1
    assert token not in serialized
    assert private_canary not in serialized

    text_result = CliRunner().invoke(cli, ["doctor", "--db"])
    assert text_result.exit_code == 1
    assert "trust: identity=" in text_result.output
    assert "review the vault, then run `memo reindex --rebuild`" in text_result.output
    assert token not in text_result.output
    assert private_canary not in text_result.output


def test_trust_preflight_reports_blocked_ambiguous_and_exact_groups(mem_with_stub: Memory):
    first = mem_with_stub.save(
        content="first body",
        title="First",
        topic_key="collision",
        tags=["project:first"],
        defer_embed=True,
    )
    second = mem_with_stub.save(
        content="second body",
        title="Second",
        topic_key="collision",
        tags=["project:second"],
        defer_embed=True,
    )
    third = mem_with_stub.save(
        content="third body",
        title="Third",
        tags=["project:third"],
        defer_embed=True,
    )
    first_identity = mem_with_stub.store.get_identity_keys(first.id)
    with mem_with_stub.store._tx() as connection:
        connection.execute("DROP INDEX IF EXISTS idx_meta_active_topic_unique")
        connection.execute(
            "UPDATE meta SET namespace = ? WHERE id = ?",
            (first_identity["namespace"], second.id),
        )
        connection.execute(
            "UPDATE meta SET tags = ?, namespace = NULL WHERE id = ?",
            (json.dumps(["project:one", "project:two"]), third.id),
        )
        connection.execute(
            "UPDATE meta SET normalized_title = ?, normalized_content_hash = ?, "
            "type = ? WHERE id = ?",
            (
                first_identity["normalized_title"],
                first_identity["normalized_content_hash"],
                "note",
                second.id,
            ),
        )
        connection.execute(
            "INSERT INTO schema_meta(key, value) VALUES "
            "('identity_topic_unique', 'blocked') "
            "ON CONFLICT(key) DO UPDATE SET value='blocked'"
        )

    report = trust_preflight(mem_with_stub.cfg)
    assert report["ok"] is False
    assert report["identity_constraint"] == "blocked"
    assert report["topic_collision_groups"] == 1
    assert report["exact_duplicate_groups"] == 1
    assert report["multiple_project_tag_rows"] == 1
    assert report["legacy_identity_rows"] == 1


def test_trust_preflight_reports_legacy_schema_unavailable(tmp_cfg):
    with closing(sqlite3.connect(tmp_cfg.db_path)) as connection, connection:
        connection.execute("CREATE TABLE meta (id TEXT PRIMARY KEY, path TEXT, tags TEXT)")
        connection.execute("INSERT INTO meta(id, path, tags) VALUES ('legacy', 'legacy.md', '[]')")
        connection.execute("PRAGMA user_version=4")

    report = trust_preflight(tmp_cfg)
    assert report["ok"] is False
    assert report["identity_constraint"] == "unavailable"
    assert report["legacy_identity_rows"] == 1
