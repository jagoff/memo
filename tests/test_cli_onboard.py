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


def test_recent_memories_includes_global_bucket(tmp_path):
    from memo.cli_onboard import _recent_memories

    root = tmp_path / "mem"
    bucket = root / "_global"
    bucket.mkdir(parents=True)
    (bucket / "x.md").write_text(
        f"---\nid: {'a' * 32}\ntitle: global memory\n---\nbody\n", encoding="utf-8"
    )

    out = _recent_memories(root, n=3)
    assert [m["title"] for m in out] == ["global memory"]


def test_recent_memories_skips_files_without_id_frontmatter(tmp_path):
    from memo.cli_onboard import _recent_memories

    root = tmp_path / "mem"
    root.mkdir()
    (root / "stray.md").write_text("# just a stray file\nno frontmatter here\n", encoding="utf-8")

    assert _recent_memories(root, n=3) == []


def test_recent_memories_prefers_yaml_frontmatter_title(tmp_path):
    from memo.cli_onboard import _recent_memories

    root = tmp_path / "mem"
    root.mkdir()
    p = root / "2026-07-13-kebab-stem.md"
    p.write_text(
        f"---\nid: {'a' * 32}\ntitle: 'Titulo legible desde yaml'\n---\ncuerpo sin heading\n",
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


def _stub_prewarm(monkeypatch):
    """Neutralize the real embedder warm (loads MLX) — return the recorded calls."""
    calls = []
    monkeypatch.setattr("memo.cli_onboard._warm_embedder", lambda cfg, **kw: calls.append(cfg))
    return calls


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
    _stub_prewarm(monkeypatch)
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
    assert "memo import json" in result.output


def test_onboard_dry_run_does_not_wire_hook_or_shims(tmp_path, monkeypatch):
    from memo.cli import cli

    hook_calls = []
    shim_calls = []
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: hook_calls.append(1) or {}
    )

    import click as _click

    @_click.command(name="install-shims-stub")
    def _spy() -> None:
        shim_calls.append(1)

    monkeypatch.setattr("memo.cli_onboard.install_shims_cmd", _spy)
    prewarm_calls = _stub_prewarm(monkeypatch)
    monkeypatch.setattr(
        "memo.transcript_miner.mine_transcripts",
        lambda root=None, **kw: {"status": "ok", "files_total": 0, "candidates": 0},
    )

    result = CliRunner().invoke(cli, ["onboard", "--yes", "--dry-run"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert hook_calls == []
    assert shim_calls == []
    assert prewarm_calls == []  # dry-run never warms the embedder
    assert "dry-run, salteado" in result.output


def test_onboard_days_override_and_dry_run(tmp_path, monkeypatch):
    from memo.cli import cli

    _stub_shims(monkeypatch)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: {"action": "already", "command": "x"}
    )
    mined = []

    def _fake_mine(root=None, **kw):
        mined.append(kw)
        return {
            "status": "ok",
            "files_total": 2,
            "candidates": 3,
            "saved": 0,
            "skipped_dup": 0,
            "dry_run": True,
        }

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
    _stub_prewarm(monkeypatch)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: {"action": "added", "command": "x"}
    )
    monkeypatch.setattr(
        "memo.transcript_miner.mine_transcripts",
        lambda root=None, **kw: {
            "status": "ok",
            "files_total": 0,
            "candidates": 0,
            "saved": 0,
            "skipped_dup": 0,
        },
    )
    result = CliRunner().invoke(cli, ["onboard", "--yes", "--json"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    payload = _json.loads(result.output[result.output.index("{") :])
    assert payload["hook"]["action"] == "added"
    assert payload["backfill"]["status"] == "ok"
    assert payload["prewarm"]["action"] == "warmed"
    assert isinstance(payload["memories"], list)


def test_onboard_yes_warms_embedder(tmp_path, monkeypatch):
    """The Day-0 wizard warms the embedder so the first recall runs vec."""
    from memo.cli import cli

    _stub_shims(monkeypatch)
    prewarm_calls = _stub_prewarm(monkeypatch)
    monkeypatch.setattr(
        "memo.cli_hooks.wire_recall_hook", lambda *a, **k: {"action": "added", "command": "x"}
    )
    monkeypatch.setattr(
        "memo.transcript_miner.mine_transcripts",
        lambda root=None, **kw: {"status": "ok", "files_total": 0, "candidates": 0},
    )
    result = CliRunner().invoke(cli, ["onboard", "--yes"], env=_env(tmp_path))
    assert result.exit_code == 0, result.output
    assert len(prewarm_calls) == 1  # warmed exactly once
    assert "prewarm" in result.output


def test_warm_embedder_stamps_prewarm_ts(tmp_path, monkeypatch):
    """_warm_embedder writes the .prewarm_ts warm signal the recall hook reads."""
    from memo.config import Config
    from memo.runtime.daemon import _warm_embedder

    class _FakeEmb:
        def embed(self, inputs):
            return [[0.0] * 4 for _ in inputs]

    monkeypatch.setattr("memo.embedder_select.make_embedder", lambda cfg, **kw: _FakeEmb())
    state = tmp_path / "state"
    state.mkdir()
    cfg = Config(
        data_dir=tmp_path / "data",
        vault_path=tmp_path / "vault",
        state_dir=state,
        reranker_enabled=False,
    )
    _warm_embedder(cfg)
    assert (state / ".prewarm_ts").is_file()
    assert float((state / ".prewarm_ts").read_text().strip()) > 0
