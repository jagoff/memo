"""Shared pytest fixtures.

Tests must NEVER touch the user's real vault or the production state
dir. The `tmp_cfg` fixture builds an isolated `Config` rooted at
`tmp_path` so every test gets fresh storage. Tests that exercise real
MLX inference are gated by `@pytest.mark.requires_mlx` — auto-skipped
if `mlx_lm` isn't importable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from memo.config import Config


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    """Isolated `Config` with vault + state dir under `tmp_path`."""
    vault = tmp_path / "vault"
    state = tmp_path / "state"
    (vault / "04-Archive" / "99-obsidian-system" / "99-AI" / "memory").mkdir(parents=True)
    state.mkdir()
    return Config(vault_path=vault, state_dir=state)


@pytest.fixture(autouse=True)
def _skip_if_no_mlx(request) -> None:
    """Auto-skip `requires_mlx` tests when `mlx_lm` isn't importable
    (Linux CI, x86_64 dev boxes)."""
    if request.node.get_closest_marker("requires_mlx"):
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            pytest.skip("mlx_lm not importable — Apple Silicon only")
