"""Obsidian vault detector — reads `obsidian.json`, sorts by recency, skips dead paths."""

from __future__ import annotations

import json
from pathlib import Path

from memo.setup.vaults import VaultInfo, detect_obsidian_vaults


def _write_registry(path: Path, vaults: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"vaults": vaults}), encoding="utf-8")


def test_returns_empty_when_registry_missing(tmp_path: Path):
    assert detect_obsidian_vaults(registry_path=tmp_path / "missing.json") == []


def test_parses_registry_and_sorts_by_ts_desc(tmp_path: Path):
    older = tmp_path / "Older"
    newer = tmp_path / "Newer"
    older.mkdir()
    newer.mkdir()
    registry = tmp_path / "obsidian.json"
    _write_registry(
        registry,
        {
            "abc123": {"path": str(older), "ts": 1000},
            "def456": {"path": str(newer), "ts": 2000, "open": True},
        },
    )
    vaults = detect_obsidian_vaults(registry_path=registry)
    assert len(vaults) == 2
    # Newer first (sorted by ts desc).
    assert vaults[0].name == "Newer"
    assert vaults[0].last_opened_ms == 2000
    assert vaults[1].name == "Older"


def test_skips_paths_that_no_longer_exist(tmp_path: Path):
    alive = tmp_path / "Alive"
    alive.mkdir()
    registry = tmp_path / "obsidian.json"
    _write_registry(
        registry,
        {
            "abc": {"path": str(alive), "ts": 1000},
            "def": {"path": str(tmp_path / "ghost-vault"), "ts": 2000},
        },
    )
    vaults = detect_obsidian_vaults(registry_path=registry)
    assert [v.name for v in vaults] == ["Alive"]


def test_handles_corrupt_registry_gracefully(tmp_path: Path):
    registry = tmp_path / "obsidian.json"
    registry.write_text("{not valid json", encoding="utf-8")
    assert detect_obsidian_vaults(registry_path=registry) == []


def test_handles_missing_ts(tmp_path: Path):
    """Some entries omit `ts`; treat as 0 instead of crashing."""
    vault = tmp_path / "NoTs"
    vault.mkdir()
    registry = tmp_path / "obsidian.json"
    _write_registry(registry, {"abc": {"path": str(vault)}})
    vaults = detect_obsidian_vaults(registry_path=registry)
    assert len(vaults) == 1
    assert vaults[0].last_opened_ms == 0


def test_vault_info_is_frozen():
    """`VaultInfo` is immutable so it can be safely compared / hashed in tests."""
    info = VaultInfo(name="X", path=Path("/tmp/X"), last_opened_ms=0)
    try:
        info.name = "Y"  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("VaultInfo should be frozen")
