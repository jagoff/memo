from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "missing.toml"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
    }


def test_verbatim_search_and_status_json(tmp_path: Path):
    from memo.cli import cli
    from memo.config import Config
    from memo.store.turn_store import TurnStore

    env = _env(tmp_path)
    cfg = Config(data_dir=Path(env["MEMO_DATA_DIR"]), state_dir=Path(env["MEMO_STATE_DIR"]))
    store = TurnStore(cfg.verbatim_db)
    try:
        store.replace_session(
            "session-1",
            "claude-code",
            [
                {
                    "idx": 0,
                    "role": "user",
                    "ts": "2026-07-13T10:00:00+00:00",
                    "text": "the exact deployment decision lived here",
                }
            ],
        )
    finally:
        store.close()

    runner = CliRunner()
    search = runner.invoke(cli, ["verbatim", "search", "deployment", "--json"], env=env)
    assert search.exit_code == 0, search.output
    hits = json.loads(search.output)
    assert hits[0]["session_id"] == "session-1"

    status = runner.invoke(cli, ["verbatim", "status", "--json"], env=env)
    assert status.exit_code == 0, status.output
    report = json.loads(status.output)
    assert report["sessions"] == 1
    assert report["turns"] == 1


def test_verbatim_search_rejects_unbounded_cli_limit(tmp_path: Path):
    from memo.cli import cli

    result = CliRunner().invoke(
        cli,
        ["verbatim", "search", "deployment", "--limit", "10000", "--json"],
        env=_env(tmp_path),
    )

    assert result.exit_code != 0
    assert "100" in result.output
