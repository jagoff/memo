from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from click.testing import CliRunner

import memo.cli_maintain as cli_maintain
from memo.cli import cli
from memo.config import Config
from memo.lifecycle import LifecycleManager
from memo.memory import Memory
from memo.quality_compact import preview_quality_compaction


def _env(tmp_path: Path) -> dict[str, str]:
    return {
        "MEMO_CONFIG_FILE": str(tmp_path / "memo-config.toml"),
        "MEMO_NONINTERACTIVE": "1",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
        "MEMO_QUALITY_COMPACT": "1",
        "MEMO_EMBEDDER_MODEL": "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        "MEMO_EMBEDDER_DIMS": "1024",
        "MEMO_MODEL_PROFILE": "balanced",
        "MEMO_RERANKER_ENABLED": "0",
    }


def _legacy_secret(mem: Memory, *, content: str, title: str, tags: list[str], extra: dict):
    record = mem.save(content=content, title=title, tags=tags, extra=extra)
    mem.store.bulk_update_type([record.id], "secret")
    return mem.get(record.id)


def _seed_quality_compact_records(tmp_path: Path) -> tuple[dict[str, str], str]:
    env = _env(tmp_path)
    cfg = Config(
        data_dir=Path(env["MEMO_DATA_DIR"]),
        state_dir=Path(env["MEMO_STATE_DIR"]),
        reranker_enabled=False,
    )
    mem = Memory(cfg)
    canonical = mem.save(
        content="Stable canonical memory.",
        title="Canonical",
        tags=["project:memo"],
        defer_embed=True,
    )
    source = mem.save(
        content="Duplicate memory.",
        title="Duplicate",
        tags=["project:memo"],
        extra={"canonical_id": canonical.id},
        defer_embed=True,
    )
    mem.close()
    return env, source.id


def test_quality_compact_preview_empty_corpus_is_read_only(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--preview", "--json"],
        env=_env(tmp_path),
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "preview"
    assert payload["proposals"] == []
    assert payload["applied"] == []


def test_quality_compact_preview_skips_cross_scope_and_sensitive_records(mock_memory) -> None:
    canonical = mock_memory.save(
        content="Stable canonical memory.",
        title="Canonical",
        tags=["project:memo"],
    )
    included_a = mock_memory.save(
        content="First duplicate.",
        title="First duplicate",
        tags=["project:memo"],
        extra={"canonical_id": canonical.id},
    )
    included_b = mock_memory.save(
        content="Second duplicate.",
        title="Second duplicate",
        tags=["project:memo"],
        extra={"superseded_by": canonical.id},
    )
    mock_memory.save(
        content="Other project duplicate.",
        title="Other project duplicate",
        tags=["project:other"],
        extra={"canonical_id": canonical.id},
    )
    _legacy_secret(
        mock_memory,
        content="Top secret duplicate.",
        title="Secret duplicate",
        tags=["project:memo"],
        extra={"canonical_id": canonical.id},
    )

    payload = preview_quality_compaction(mock_memory, limit=20)

    assert payload["mode"] == "preview"
    assert payload["applied"] == []
    assert payload["errors"] == []
    assert len(payload["proposals"]) == 1
    proposal = payload["proposals"][0]
    assert proposal["canonical_title"] == "Canonical"
    assert set(proposal["source_ids"]) == {included_a.id, included_b.id}
    assert proposal["scope"] == "project:memo"
    assert proposal["reasons"] == ["explicit_canonical_or_superseded_by"]


def test_quality_compact_preview_skips_ambiguous_multi_project_scope(mock_memory) -> None:
    canonical = mock_memory.save(
        content="Stable canonical memory.",
        title="Canonical",
        tags=["project:memo"],
    )
    ambiguous = mock_memory.save(
        content="Ambiguous duplicate.",
        title="Ambiguous duplicate",
        tags=["project:memo"],
        extra={"canonical_id": canonical.id},
    )
    # New writes reject multiple project namespaces. Seed the shape through
    # the rebuildable index to model a hand-edited/legacy Markdown record and
    # keep quality-compaction's defensive behavior covered.
    assert mock_memory.store.update_meta(
        id_=ambiguous.id,
        title=ambiguous.title,
        type_=ambiguous.type,
        tags=["project:memo", "project:other"],
        updated=ambiguous.updated,
        extra=ambiguous.extra,
        namespace=None,
    )

    payload = preview_quality_compaction(mock_memory, limit=20)

    assert payload["proposals"] == []
    assert payload["errors"] == [f"ambiguous_scope:{ambiguous.id}"]


def test_quality_compact_preview_skips_unresolved_canonical_targets(mock_memory) -> None:
    missing_canonical_id = "f" * 32
    mock_memory.save(
        content="Duplicate with missing canonical.",
        title="Missing canonical duplicate",
        tags=["project:memo"],
        extra={"canonical_id": missing_canonical_id},
    )

    payload = preview_quality_compaction(mock_memory, limit=20)

    assert payload["proposals"] == []
    assert payload["errors"] == [f"unresolved_canonical:{missing_canonical_id}"]


def test_quality_compact_command_requires_flag(tmp_path: Path) -> None:
    env = _env(tmp_path)
    env.pop("MEMO_QUALITY_COMPACT")
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--preview", "--json"],
        env=env,
    )
    assert result.exit_code != 0
    assert "MEMO_QUALITY_COMPACT=1 is required" in result.output


def test_quality_compact_apply_rejects_preview_combination(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--preview", "--apply"],
        env=_env(tmp_path),
    )
    assert result.exit_code != 0
    assert "choose either --preview or --apply" in result.output


def test_quality_compact_apply_writes_receipt_shape(tmp_path: Path) -> None:
    env = _env(tmp_path)
    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--apply", "--json"],
        env=env,
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["mode"] == "apply"
    assert "quality_compacted" in payload
    assert payload["errors"] == []
    receipt_path = Path(env["MEMO_STATE_DIR"]) / "maintain" / "last.json"
    assert receipt_path.is_file()
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted["mode"] == "apply"
    assert "quality_compacted" in persisted


def test_quality_compact_receipt_paths_are_unique_within_same_second(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = Config(
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        reranker_enabled=False,
    )
    ns_values = iter([111, 222])

    monkeypatch.setattr(cli_maintain.time, "time", lambda: 123.0)
    monkeypatch.setattr(cli_maintain.time, "time_ns", lambda: next(ns_values))

    first = cli_maintain._prepare_quality_compact_receipt_paths(cfg)[2]
    second = cli_maintain._prepare_quality_compact_receipt_paths(cfg)[2]

    assert first != second


def test_quality_compact_apply_receipt_publish_is_atomic(tmp_path: Path, monkeypatch) -> None:
    env, source_id = _seed_quality_compact_records(tmp_path)
    maintain_dir = Path(env["MEMO_STATE_DIR"]) / "maintain"
    maintain_dir.mkdir(parents=True, exist_ok=True)
    last_path = maintain_dir / "last.json"
    previous = {"mode": "previous", "ts": 1.0}
    last_path.write_text(json.dumps(previous), encoding="utf-8")

    real_replace = os.replace

    def boom(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        if Path(dst).name == "last.json":
            raise OSError("simulated last replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(cli_maintain.os, "replace", boom)

    def fail_if_embedded(*_args, **_kwargs):
        raise AssertionError("rollback restoration must not require an embedding model")

    monkeypatch.setattr(
        "memo.memory.maintain_ops._MaintainOpsMixin._embed_cached",
        fail_if_embedded,
    )

    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--apply", "--json"],
        env=env,
    )

    assert result.exit_code != 0
    assert "quality compaction receipt persistence failed" in result.output
    assert json.loads(last_path.read_text(encoding="utf-8")) == previous
    assert list((maintain_dir / "runs").glob("*.json")) == []

    cfg = Config(
        data_dir=Path(env["MEMO_DATA_DIR"]),
        state_dir=Path(env["MEMO_STATE_DIR"]),
        reranker_enabled=False,
    )
    mem = Memory(cfg)
    try:
        restored = mem.get(source_id)
        assert restored is not None
        assert restored.extra.get("_memo_embed_pending") is True
        assert "superseded_by" not in restored.extra
        assert "superseded_at" not in restored.extra
    finally:
        mem.close()


def test_quality_compact_apply_rolls_back_attempted_archive_ids(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env, source_id = _seed_quality_compact_records(tmp_path)
    cfg = Config(
        data_dir=Path(env["MEMO_DATA_DIR"]),
        state_dir=Path(env["MEMO_STATE_DIR"]),
        reranker_enabled=False,
    )

    def partial_move_then_raise(
        self: LifecycleManager,
        memory_id: str,
        *,
        superseded_by: str | None = None,
    ) -> bool:
        del superseded_by
        rec = self.memory.get(memory_id)
        assert rec is not None
        inactive_dir = self.memory.cfg.memory_dir / "inactive"
        inactive_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.memory._resolve_existing(rec.path)
        target_path = inactive_dir / f"{memory_id}.md"
        shutil.move(str(source_path), str(target_path))
        raise RuntimeError("simulated move-then-raise")

    real_replace = os.replace

    def boom(src: os.PathLike[str] | str, dst: os.PathLike[str] | str) -> None:
        if Path(dst).name == "last.json":
            raise OSError("simulated last replace failure")
        real_replace(src, dst)

    monkeypatch.setattr(LifecycleManager, "archive_memory", partial_move_then_raise)
    monkeypatch.setattr(cli_maintain.os, "replace", boom)

    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--apply", "--json"],
        env=env,
    )

    assert result.exit_code != 0
    assert "quality compaction apply failed" in result.output
    assert not (cfg.memory_dir / "inactive" / f"{source_id}.md").exists()
    assert not (Path(env["MEMO_STATE_DIR"]) / "maintain" / "last.json").exists()
    assert list((Path(env["MEMO_STATE_DIR"]) / "maintain" / "runs").glob("*.json")) == []

    mem = Memory(cfg)
    try:
        assert mem.get(source_id) is not None
    finally:
        mem.close()


def test_quality_compact_apply_error_rolls_back_before_persisting_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    env, source_id = _seed_quality_compact_records(tmp_path)
    cfg = Config(
        data_dir=Path(env["MEMO_DATA_DIR"]),
        state_dir=Path(env["MEMO_STATE_DIR"]),
        reranker_enabled=False,
    )

    def partial_move_then_raise(
        self: LifecycleManager,
        memory_id: str,
        *,
        superseded_by: str | None = None,
    ) -> bool:
        del superseded_by
        rec = self.memory.get(memory_id)
        assert rec is not None
        inactive_dir = self.memory.cfg.memory_dir / "inactive"
        inactive_dir.mkdir(parents=True, exist_ok=True)
        source_path = self.memory._resolve_existing(rec.path)
        target_path = inactive_dir / f"{memory_id}.md"
        shutil.move(str(source_path), str(target_path))
        raise RuntimeError("simulated move-then-raise")

    monkeypatch.setattr(LifecycleManager, "archive_memory", partial_move_then_raise)

    result = CliRunner().invoke(
        cli,
        ["maintain", "quality-compact", "--apply", "--json"],
        env=env,
    )

    assert result.exit_code != 0
    assert "quality compaction apply failed" in result.output
    assert not (Path(env["MEMO_STATE_DIR"]) / "maintain" / "last.json").exists()
    assert list((Path(env["MEMO_STATE_DIR"]) / "maintain" / "runs").glob("*.json")) == []
    assert not (cfg.memory_dir / "inactive" / f"{source_id}.md").exists()

    mem = Memory(cfg)
    try:
        assert mem.get(source_id) is not None
    finally:
        mem.close()
