import json
import sqlite3
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from memo.memory import Memory
from memo.trust_preflight import trust_preflight


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
    with sqlite3.connect(tmp_cfg.db_path) as connection:
        connection.execute("CREATE TABLE meta (id TEXT PRIMARY KEY, path TEXT, tags TEXT)")
        connection.execute(
            "INSERT INTO meta(id, path, tags) VALUES ('legacy', 'legacy.md', '[]')"
        )
        connection.execute("PRAGMA user_version=4")

    report = trust_preflight(tmp_cfg)
    assert report["ok"] is False
    assert report["identity_constraint"] == "unavailable"
    assert report["legacy_identity_rows"] == 1
