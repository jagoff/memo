"""Tests for the `memo onboard` Day-0 wizard."""
from __future__ import annotations

import os
import time

from click.testing import CliRunner


def _env(tmp_path):
    return {
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_VAULT_PATH": str(tmp_path / "vault"),
        "MEMO_EMBEDDER_VIA_DAEMON": "0",
        "MEMO_SKIP_MODEL_VERSION_CHECK": "1",
    }


def test_onboard_backfill_days_flag_registered():
    from memo.flags import REGISTRY

    spec = REGISTRY["MEMO_ONBOARD_BACKFILL_DAYS"]
    assert spec.default == 90


def test_recent_memories_orders_by_mtime_and_skips_buckets(tmp_path):
    from memo.cli_onboard import _recent_memories

    root = tmp_path / "mem"
    root.mkdir()
    for i, name in enumerate(["old", "mid", "new"]):
        p = root / f"{name}.md"
        p.write_text(f"---\nid: {'a' * 32}\n---\n# titulo {name}\n", encoding="utf-8")
        os.utime(p, (time.time() - 100 + i, time.time() - 100 + i))
    bucket = root / "_profile"
    bucket.mkdir()
    (bucket / "profile.md").write_text("# not a memory\n", encoding="utf-8")

    out = _recent_memories(root, n=2)
    assert [m["title"] for m in out] == ["titulo new", "titulo mid"]


def test_recent_memories_empty_dir(tmp_path):
    from memo.cli_onboard import _recent_memories

    assert _recent_memories(tmp_path / "nope") == []


def test_recent_memories_prefers_yaml_frontmatter_title(tmp_path):
    from memo.cli_onboard import _recent_memories

    root = tmp_path / "mem"
    root.mkdir()
    p = root / "2026-07-13-kebab-stem.md"
    p.write_text(
        "---\n"
        f"id: {'a' * 32}\n"
        "title: 'Titulo legible desde yaml'\n"
        "---\n"
        "cuerpo sin heading\n",
        encoding="utf-8",
    )
    out = _recent_memories(root, n=1)
    assert out[0]["title"] == "Titulo legible desde yaml"


def _stub_shims(monkeypatch):
    import click as _click

    @_click.command(name="install-shims-stub")
    def _noop() -> None:
        pass

    monkeypatch.setattr("memo.cli_onboard.install_shims_cmd", _noop)


def _fake_memories(tmp_path, n=3):
    data = tmp_path / "data"
    data.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        (data / f"m{i}.md").write_text(
            f"---\nid: {'a' * 32}\n---\n# aprendizaje {i}\n", encoding="utf-8"
        )


def test_onboard_noninteractive_without_yes_prints_guidance(tmp_path, monkeypatch):
    from memo.cli import cli

    calls = []
    monkeypatch.setattr("memo.cli_hooks.wire_recall_hook", lambda *a, **k: calls.append(1) or {})
    result = CliRunner().invoke(cli, ["onboard"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert "--yes" in result.output
    assert calls == []  # ningún paso corrió


def test_onboard_yes_runs_all_steps(tmp_path, monkeypatch):
    from memo.cli import cli

    _stub_shims(monkeypatch)
    _fake_memories(tmp_path)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook",
        lambda *a, **k: {"action": "added", "command": "memo recall-hook"},
    )
    mined = []

    def _fake_mine(root=None, **kw):
        mined.append(kw)
        return {"status": "ok", "files_total": 5, "candidates": 12, "saved": 7, "skipped_dup": 2}

    monkeypatch.setattr("memo.transcript_miner.mine_transcripts", _fake_mine)

    result = CliRunner().invoke(cli, ["onboard", "--yes"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert len(mined) == 1 and mined[0]["dry_run"] is False and mined[0]["since_days"] == 90
    assert "7 memorias" in result.output
    assert "aprendizaje 2" in result.output  # las 3 cosas que ya sé de vos
    assert "memo import whatsapp" in result.output


def test_onboard_days_override_and_dry_run(tmp_path, monkeypatch):
    from memo.cli import cli

    _stub_shims(monkeypatch)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: {"action": "already", "command": "x"}
    )
    mined = []

    def _fake_mine(root=None, **kw):
        mined.append(kw)
        return {"status": "ok", "files_total": 2, "candidates": 3, "saved": 0,
                "skipped_dup": 0, "dry_run": True}

    monkeypatch.setattr("memo.transcript_miner.mine_transcripts", _fake_mine)

    result = CliRunner().invoke(
        cli, ["onboard", "--yes", "--days", "7", "--dry-run"], env=_env(tmp_path)
    )
    assert result.exit_code == 0, result.output
    assert len(mined) == 1 and mined[0]["dry_run"] is True and mined[0]["since_days"] == 7


def test_onboard_json_summary(tmp_path, monkeypatch):
    import json as _json

    from memo.cli import cli

    _stub_shims(monkeypatch)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: {"action": "added", "command": "x"}
    )
    monkeypatch.setattr(
        "memo.transcript_miner.mine_transcripts",
        lambda root=None, **kw: {"status": "ok", "files_total": 0, "candidates": 0,
                                 "saved": 0, "skipped_dup": 0},
    )
    result = CliRunner().invoke(cli, ["onboard", "--yes", "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output[result.output.index("{"):])
    assert payload["hook"]["action"] == "added"
    assert payload["backfill"]["status"] == "ok"
    assert isinstance(payload["memories"], list)
