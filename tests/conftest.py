"""Shared pytest fixtures.

Tests must NEVER touch the user's real vault or the production state
dir. The `tmp_cfg` fixture builds an isolated `Config` rooted at
`tmp_path` so every test gets fresh storage. Tests that exercise real
MLX inference are gated by `@pytest.mark.requires_mlx` — auto-skipped
if `mlx_lm` isn't importable.

CliRunner-based tests inherit `MEMO_NONINTERACTIVE=1` and a non-existent
`MEMO_CONFIG_FILE` from the module-level `os.environ.setdefault` calls
below, so the first-run picker never fires mid-test and the developer's
real `~/.config/memo/config.toml` never leaks in. Individual tests can
still override via `monkeypatch.setenv` or `CliRunner().invoke(env=...)`
when verifying first-run/interactive behavior. The `tmp_cfg` fixture
adds isolated storage on top.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

import pytest

# Test-wide defaults applied at conftest import. These are `setdefault` so
# individual tests can still override via `monkeypatch.setenv` /
# `CliRunner().invoke(env=...)`. Goal: even tests that forget to pin these
# explicitly never read the developer's real `~/.config/memo/config.toml`
# and never trigger the first-run picker mid-test.
os.environ.setdefault("MEMO_NONINTERACTIVE", "1")
os.environ.setdefault(
    "MEMO_CONFIG_FILE",
    str(Path(tempfile.gettempdir()) / "memo-test-nonexistent-config.toml"),
)

from memo.config import Config


@pytest.fixture
def mock_memory(tmp_cfg):
    """Real `Memory` instance isolated under `tmp_cfg`. Shared across all test modules."""
    from memo.memory import Memory

    mem = Memory(tmp_cfg)

    def _fake_embedding(text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        values = [
            ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
            for i in range(tmp_cfg.embedder_dims)
        ]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    mem.embedder.embed = lambda inputs: [_fake_embedding(text) for text in inputs]
    mem.embedder.embed_query = lambda query: _fake_embedding(query)
    mem._chat = _FakeChat()
    return mem


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
    (vault / "Obsidian" / "AI" / "memory").mkdir(parents=True)
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
    return Config(data_dir=data, vault_path=vault, state_dir=state, reranker_enabled=False)


class _FakeChat:
    """Deterministic local chat double for non-MLX unit tests."""

    def complete(self, prompt: str, temperature: float = 0.0, **_: object) -> str:
        if "estimated_complexity" in prompt:
            goal = "Test goal"
            for line in prompt.splitlines():
                if line.startswith("Goal:"):
                    goal = line.removeprefix("Goal:").strip() or goal
                    break
            return json.dumps(
                {
                    "goal": goal,
                    "steps": [f"search for {goal}", "analyze relationships"],
                    "estimated_complexity": 3,
                    "estimated_insight_value": 7,
                }
            )
        if "new_insight" in prompt:
            return json.dumps(
                {
                    "new_insight": "Related memorias suggest a practical pattern.",
                    "supporting_memorias": [],
                    "confidence": 0.75,
                    "novelty_score": 0.7,
                }
            )
        return "The notes are causally related through shared constraints and outcomes."

    def chat(self, model: str, messages: list[dict[str, str]], options: dict | None = None) -> dict:
        system = messages[0].get("content", "") if messages else ""
        user = messages[-1].get("content", "") if messages else ""

        if "extract entities" in system.lower():
            entities = []
            lowered = user.lower()
            if "mlx" in lowered:
                entities.append({"name": "mlx", "type": "technology"})
            if "memo" in lowered:
                entities.append({"name": "memo", "type": "project"})
            if "apple" in lowered:
                entities.append({"name": "apple silicon", "type": "technology"})
            return {"message": {"content": json.dumps({"entities": entities})}}

        if "temporal relationship" in system.lower():
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "relationship": "consistent",
                            "rationale": "The notes do not contradict each other.",
                            "confidence": 0.8,
                        }
                    )
                }
            }

        if "suggest" in system.lower():
            return {"message": {"content": json.dumps({"suggestions": []})}}

        if "cluster" in user.lower() or "merge" in system.lower():
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "summary": "Related test memorias.",
                            "relationship": "facets",
                            "rationale": "They cover compatible aspects.",
                            "merged_title": "Merged",
                            "merged_body": "Merged content",
                            "merge_strategy": "synthesis",
                        }
                    )
                }
            }

        return {"message": {"content": "{}"}}


@pytest.fixture(autouse=True)
def _skip_if_no_mlx(request) -> None:
    """Auto-skip `requires_mlx` tests when `mlx_lm` isn't importable
    (Linux CI, x86_64 dev boxes)."""
    if request.node.get_closest_marker("requires_mlx"):
        try:
            import mlx_lm  # noqa: F401
        except ImportError:
            pytest.skip("mlx_lm not importable — Apple Silicon only")
