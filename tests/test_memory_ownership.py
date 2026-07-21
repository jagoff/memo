from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from memo.cli import cli


def test_confidence_explain_closes_owned_memory(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    import memo.cli_confidence as confidence

    memory = MagicMock()
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)
    monkeypatch.setattr(confidence, "build_calibration", lambda *_args: {})

    result = CliRunner().invoke(cli, ["confidence", "explain"])

    assert result.exit_code == 0, result.output
    memory.close.assert_called_once_with()


def test_install_seed_closes_owned_memory(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    from memo.cli_install_mcp import _seed_install_memory

    tmp_cfg.state_dir.mkdir(parents=True, exist_ok=True)
    memory = MagicMock()
    memory.save.return_value = SimpleNamespace(id="a" * 32)
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)

    _seed_install_memory()

    memory.close.assert_called_once_with()


def test_file_import_closes_owned_memory_on_failure(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.history_importers as importers

    memory = MagicMock()
    memory._ensure_chat.return_value = object()
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)
    monkeypatch.setattr(
        "memo.transcript_miner.mine_exchange_stream",
        MagicMock(side_effect=RuntimeError("extract failed")),
    )

    with pytest.raises(RuntimeError, match="extract failed"):
        importers.run_file_import(iter([("user", "assistant")]))

    memory.close.assert_called_once_with()


def test_codex_import_closes_owned_memory_on_failure(
    tmp_cfg, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.history_importers as importers
    import memo.transcript_miner as miner

    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    memory = MagicMock()
    memory._ensure_chat.return_value = object()
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)
    monkeypatch.setattr(miner, "find_transcripts", lambda *_args, **_kwargs: [transcript])
    monkeypatch.setattr(
        miner,
        "mine_exchange_stream",
        MagicMock(side_effect=RuntimeError("extract failed")),
    )

    with pytest.raises(RuntimeError, match="extract failed"):
        importers.run_codex_import(root=tmp_path)

    memory.close.assert_called_once_with()


def test_transcript_miner_closes_owned_memory_on_failure(
    tmp_cfg, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.transcript_miner as miner

    transcript = tmp_path / "session.jsonl"
    transcript.write_text("{}\n", encoding="utf-8")
    memory = MagicMock()
    memory._ensure_chat.return_value = object()
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)
    monkeypatch.setattr(miner, "find_transcripts", lambda *_args, **_kwargs: [transcript])
    monkeypatch.setattr(
        miner,
        "mine_exchange_stream",
        MagicMock(side_effect=RuntimeError("extract failed")),
    )

    with pytest.raises(RuntimeError, match="extract failed"):
        miner.mine_transcripts(root=tmp_path)

    memory.close.assert_called_once_with()


def test_verdict_closes_only_internally_owned_memory(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.verdict as verdict

    memory = MagicMock()
    monkeypatch.setattr("memo.flags.flag_bool", lambda _name: False)
    monkeypatch.setattr(
        verdict,
        "score_next_turn",
        lambda *_args, **_kwargs: {
            "prompt": "a sufficiently long prior query",
            "verdict": "positive",
            "recall_ids": ["abcd1234"],
        },
    )
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)

    assert verdict.record_verdicts(tmp_cfg, {"session_id": "s1"}) is not None

    memory.feedback_record.assert_called_once()
    memory.close.assert_called_once_with()


def test_dashboard_one_shot_closes_owned_memory(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    import memo.tui.dashboard.app as dashboard

    memory = MagicMock()
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)
    monkeypatch.setattr(dashboard, "render", lambda *_args: "dashboard")

    dashboard.run_tui(once=True)

    memory.close.assert_called_once_with()


def test_bucket_migration_closes_owned_memory(tmp_cfg, monkeypatch: pytest.MonkeyPatch) -> None:
    import memo.runtime.migrate as migrate

    memory = MagicMock()
    memory.reindex.return_value = {"checked": 0, "added": 0, "reindexed": 0, "skipped": 0}
    monkeypatch.setattr(migrate.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setattr(migrate, "_bucket_by_project", lambda _cfg: 0)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)
    monkeypatch.setattr(
        "memo.setup.config_io._resolve_config_path",
        lambda: tmp_cfg.state_dir / "config.toml",
    )

    result = CliRunner().invoke(cli, ["migrate-vault", "--bucket-by-project"])

    assert result.exit_code == 0, result.output
    memory.close.assert_called_once_with()


def test_mcp_build_failure_closes_only_owned_memory(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.server as server_module

    owned = MagicMock()
    injected = MagicMock()
    monkeypatch.setattr(server_module.Config, "from_env", lambda: tmp_cfg)
    monkeypatch.setattr(server_module, "Memory", lambda _cfg: owned)
    monkeypatch.setattr(
        server_module,
        "FastMCP",
        MagicMock(side_effect=RuntimeError("server construction failed")),
    )

    with pytest.raises(RuntimeError, match="server construction failed"):
        server_module.build_server()
    owned.close.assert_called_once_with()

    with pytest.raises(RuntimeError, match="server construction failed"):
        server_module.build_server(memory=injected)
    injected.close.assert_not_called()


def test_maint_constructor_failure_closes_runner_and_lock(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.maint_server as maint

    runner = MagicMock()
    closed = MagicMock()
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr(maint, "_default_runner", lambda _cfg: runner)
    monkeypatch.setattr(maint, "_read_pid", lambda _state_dir: None)
    monkeypatch.setattr(maint.os, "open", lambda *_args, **_kwargs: 77)
    monkeypatch.setattr(maint.os, "close", closed)
    monkeypatch.setattr(maint.fcntl, "flock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        maint,
        "_MaintServer",
        MagicMock(side_effect=RuntimeError("constructor failed")),
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        maint.run_server(tmp_cfg.state_dir)

    runner.close.assert_called_once_with()
    closed.assert_called_once_with(77)


def test_recall_constructor_failure_closes_memory_and_lock(
    tmp_cfg, monkeypatch: pytest.MonkeyPatch
) -> None:
    import memo.recall_socket as recall

    memory = MagicMock()
    closed = MagicMock()
    monkeypatch.setattr("memo.config.Config.from_env", lambda: tmp_cfg)
    monkeypatch.setattr("memo.memory.Memory", lambda _cfg: memory)
    monkeypatch.setattr(recall, "_read_pid", lambda _state_dir: None)
    monkeypatch.setattr(recall, "set_process_gpu_priority", lambda _enabled: None)
    monkeypatch.setattr(recall.os, "open", lambda *_args, **_kwargs: 78)
    monkeypatch.setattr(recall.os, "close", closed)
    monkeypatch.setattr(recall.fcntl, "flock", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        recall,
        "_RecallServer",
        MagicMock(side_effect=RuntimeError("constructor failed")),
    )

    with pytest.raises(RuntimeError, match="constructor failed"):
        recall.run_server(tmp_cfg.state_dir)

    memory.close.assert_called_once_with()
    closed.assert_called_once_with(78)
