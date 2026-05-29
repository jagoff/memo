from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from click.testing import CliRunner

from memo.cli import cli
from memo.cli_diag import _profile_repair_plan
from memo.config import Config


def _env(tmp_path: Path, **extra: str) -> dict[str, str]:
    env = {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }
    env.update(extra)
    return env


def test_profile_status_json_reports_active_profile(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["profile", "status", "--json", "--no-db"],
        env=_env(tmp_path, MEMO_MODEL_PROFILE="quality"),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["schema"] == "memo.profile_status.v1"
    assert payload["profile"] == "quality"
    assert payload["active"]["embedder_dims"] == 2560
    assert payload["db"]["status"] == "not_checked"


def test_profile_status_json_surfaces_model_env_overrides(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["profile", "status", "--json", "--no-db"],
        env=_env(
            tmp_path,
            MEMO_MODEL_PROFILE="quality",
            MEMO_EMBEDDER_DIMS="1024",
        ),
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["profile"] == "quality"
    assert {
        "field": "embedder_dims",
        "expected": 2560,
        "actual": 1024,
    } in payload["overrides"]


def test_profile_repair_plan_flags_db_dimension_mismatch(tmp_path: Path) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        model_profile="quality",
        embedder_dims=2560,
    )
    cfg.state_dir.mkdir(parents=True)
    with sqlite3.connect(cfg.db_path) as conn:
        conn.execute("CREATE TABLE vec (id TEXT PRIMARY KEY, embedding FLOAT[1024])")
        conn.execute("CREATE TABLE repo_vec (id TEXT PRIMARY KEY, embedding FLOAT[1024])")

    plan = _profile_repair_plan(cfg)

    assert plan["ok"] is False
    assert plan["status"] == "dimension_mismatch"
    assert any(action["kind"] == "memory_index_rebuild" for action in plan["actions"])
