"""Shared pytest fixtures.

Tests must NEVER touch the user's real vault or the production state
dir. The `tmp_cfg` fixture builds an isolated `Config` rooted at
`tmp_path` so every test gets fresh storage. Tests that exercise real
MLX inference are gated by `@pytest.mark.requires_mlx` — auto-skipped
if `mlx_lm` isn't importable.

CliRunner-based tests must invoke commands with `env={"MEMO_NONINTERACTIVE":
"1"}` so the first-run picker doesn't fire mid-test. The `tmp_cfg`
fixture only protects the storage layer — the CLI's first-run gate
checks env vars / TTY independently.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from memo.config import Config


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    """Isolated `Config` with data dir + vault + state dir under `tmp_path`.

    `data_dir` is set explicitly (the new layout). `vault_path` is also
    set to a separate directory so legacy-path tests (which exercise the
    `_resolve_existing` fallback) keep working.
    """
    data = tmp_path / "data"
    vault = tmp_path / "vault"
    state = tmp_path / "state"
    data.mkdir()
    (vault / "99-obsidian" / "99-AI" / "memory").mkdir(parents=True)
    state.mkdir()
    # Point the TOML config-file lookup at a non-existent path so
    # `Config.from_env()` doesn't accidentally pick up the developer's
    # real `~/.config/memo/config.toml` while running tests.
    os.environ.setdefault("MEMO_CONFIG_FILE", str(tmp_path / "memo-config.toml"))
    # Disable auto-project tagging in the default test fixture so tests
    # that assert exact tag sets aren't polluted by the cwd-derived
    # `project:<repo>` tag. Tests that exercise the auto-tag flow opt
    # back in explicitly via monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1").
    os.environ.setdefault("MEMO_AUTO_PROJECT_TAG", "0")
    return Config(data_dir=data, vault_path=vault, state_dir=state)


@pytest.fixture(autouse=True)
def _skip_if_no_mlx(request) -> None:
    """Auto-skip `requires_mlx` tests when `mlx_lm` isn't importable
    (Linux CI, x86_64 dev boxes)."""
    if request.node.get_closest_marker("requires_mlx"):
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            pytest.skip("mlx_lm not importable — Apple Silicon only")
