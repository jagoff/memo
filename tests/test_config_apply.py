"""Preserving and transactional application of TUI configuration drafts."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from memo import config_md
from memo.errors import ConfigConflictError, ConfigTransactionError
from memo.tui.config.apply import (
    ConfigTransaction,
    recover_interrupted_transaction,
    render_draft,
)
from memo.tui.config.session import ConfigSession


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_DIR": str(tmp_path / "memo-home"),
        "MEMO_CONFIG_FILE": str(tmp_path / "legacy.toml"),
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }


def _write_config(
    tmp_path: Path, table: str, key: str, value: object, *, intro: str = ""
) -> Path:
    path = tmp_path / "memo-home" / "config" / f"{table}-config.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = f'"{value}"' if isinstance(value, str) else str(value).lower()
    path.write_text(
        f"{intro}```toml\n[{table}]\n{key} = {rendered}\n```\n",
        encoding="utf-8",
    )
    return path


def _plan_set(tmp_path: Path, key: str, value: object):
    session = ConfigSession.open(_env(tmp_path))
    session.set_value(key, value)
    return session.review()


def test_render_preserves_markdown_outside_toml(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        "recall",
        "top_k",
        3,
        intro="Keep this prose exactly.\n",
    )
    plan = _plan_set(tmp_path, "recall.top_k", 5)

    rendered = render_draft(plan, _env(tmp_path))[path]

    assert rendered.startswith("Keep this prose exactly.\n")
    assert "top_k = 5" in rendered


def test_render_retains_independent_external_edit(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "recall", "top_k", 3, intro="Original.\n")
    plan = _plan_set(tmp_path, "recall.top_k", 5)
    path.write_text(path.read_text(encoding="utf-8") + "External note.\n", encoding="utf-8")

    rendered = render_draft(plan, _env(tmp_path))[path]

    assert rendered.endswith("External note.\n")
    assert "top_k = 5" in rendered


def test_commit_rolls_back_every_file_on_second_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recall = _write_config(tmp_path, "recall", "top_k", 3)
    search = _write_config(tmp_path, "search", "default_limit", 10)
    session = ConfigSession.open(_env(tmp_path))
    session.set_value("recall.top_k", 5)
    session.set_value("search.default_limit", 20)
    plan = session.review()
    rendered = render_draft(plan, _env(tmp_path))
    original = {path: path.read_text(encoding="utf-8") for path in (recall, search)}
    real_replace = os.replace
    calls = 0

    def fail_second(src: str | bytes, dst: str | bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", fail_second)

    with pytest.raises(ConfigTransactionError, match="injected replace failure"):
        ConfigTransaction(_env(tmp_path)).commit(rendered, plan.snapshot)

    assert {path: path.read_text(encoding="utf-8") for path in original} == original


def test_same_key_external_edit_raises_conflict(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "recall", "top_k", 3)
    plan = _plan_set(tmp_path, "recall.top_k", 5)
    path.write_text(path.read_text(encoding="utf-8").replace("top_k = 3", "top_k = 4"))

    with pytest.raises(ConfigConflictError) as exc:
        render_draft(plan, _env(tmp_path))

    assert exc.value.keys == ("recall.top_k",)


def test_successful_commit_updates_all_files(tmp_path: Path) -> None:
    recall = _write_config(tmp_path, "recall", "top_k", 3)
    search = _write_config(tmp_path, "search", "default_limit", 10)
    session = ConfigSession.open(_env(tmp_path))
    session.set_value("recall.top_k", 5)
    session.set_value("search.default_limit", 20)
    plan = session.review()
    assert config_md.flag_values(_env(tmp_path))["MEMO_RECALL_TOP_K"] == "3"

    receipt = ConfigTransaction(_env(tmp_path)).commit(
        render_draft(plan, _env(tmp_path)), plan.snapshot
    )

    assert receipt.state == "complete"
    assert "top_k = 5" in recall.read_text(encoding="utf-8")
    assert "default_limit = 20" in search.read_text(encoding="utf-8")
    assert config_md.flag_values(_env(tmp_path))["MEMO_RECALL_TOP_K"] == "5"


def test_external_edit_between_render_and_commit_is_rejected(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "recall", "top_k", 3)
    plan = _plan_set(tmp_path, "recall.top_k", 5)
    rendered = render_draft(plan, _env(tmp_path))
    path.write_text(path.read_text(encoding="utf-8") + "Late edit.\n", encoding="utf-8")

    with pytest.raises(ConfigConflictError) as exc:
        ConfigTransaction(_env(tmp_path)).commit(rendered, plan.snapshot)

    assert exc.value.keys == ("recall.top_k",)


def test_unset_preserves_other_values_and_prose(tmp_path: Path) -> None:
    path = tmp_path / "memo-home" / "config" / "recall-config.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        "Notes stay.\n```toml\n[recall]\ntop_k = 3\nmin_sim = 0.5\n```\n",
        encoding="utf-8",
    )
    session = ConfigSession.open(_env(tmp_path))
    session.unset_value("recall.top_k")
    plan = session.review()

    rendered = render_draft(plan, _env(tmp_path))[path]

    assert rendered.startswith("Notes stay.\n")
    assert "top_k" not in rendered
    assert "min_sim = 0.5" in rendered


def test_recovery_restores_backups_from_committing_manifest(tmp_path: Path) -> None:
    path = _write_config(tmp_path, "recall", "top_k", 3)
    plan = _plan_set(tmp_path, "recall.top_k", 5)
    receipt = ConfigTransaction(_env(tmp_path)).commit(
        render_draft(plan, _env(tmp_path)), plan.snapshot
    )
    assert receipt.manifest is not None
    manifest = json.loads(receipt.manifest.read_text(encoding="utf-8"))
    manifest["state"] = "committing"
    receipt.manifest.write_text(json.dumps(manifest), encoding="utf-8")

    recovered = recover_interrupted_transaction(tmp_path / "memo-home")

    assert recovered is not None
    assert recovered.state == "recovered"
    assert "top_k = 3" in path.read_text(encoding="utf-8")
