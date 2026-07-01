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
# Most server tests exercise the complete administrative contract. Production
# defaults to the five-tool agent profile; the dedicated surface-profile tests
# delete/override this value when asserting that default.
os.environ.setdefault("MEMO_MCP_PROFILE", "full")

# Hermetic recall-daemon isolation. A developer machine usually has memo's warm
# recall daemon listening at the *default* state dir
# (`~/.local/share/memo/recall.sock`). Two code paths would otherwise consume it
# mid-test and silently break hermeticity:
#   1. `MEMO_EMBEDDER_VIA_DAEMON=1` makes `Memory` route embeds over the socket;
#   2. `embedder_client`'s socket-first helpers resolve a `None` state_dir to
#      `Config.from_env().state_dir` (and cache it) — without a pinned
#      `MEMO_STATE_DIR` that is exactly the real default dir.
# Either way the live daemon returns real nearest-neighbour vectors at the
# production model's dims, corrupting assertions that assume the stub embedder
# (semantic search always returns *some* neighbour, so "unrelated → empty"
# negative tests start failing). Force the flag off (hard-set: overrides a
# developer's exported `=1`) and point the default state dir away from the real
# socket. Tests that exercise daemon routing opt back in via `monkeypatch.setenv`
# / `CliRunner(...).invoke(env=...)`.
os.environ["MEMO_EMBEDDER_VIA_DAEMON"] = "0"
os.environ.setdefault(
    "MEMO_STATE_DIR",
    str(Path(tempfile.gettempdir()) / "memo-test-nonexistent-state"),
)

from memo.config import Config
from memo.embed_base import EmbedderBase


@pytest.fixture
def mem_with_stub(tmp_cfg: Config, monkeypatch):
    """`Memory` with a deterministic 4-dim embedder for high-level tests."""
    from memo.memory import Memory

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )

    def _stub_embed(self, inputs):
        out = []
        for s in inputs:
            h = sum(ord(c) for c in s) % 4
            v = [0.0] * 4
            v[h] = 1.0
            out.append(v)
        return out

    monkeypatch.setattr("memo.embedder.MLXEmbedder.embed", _stub_embed)
    mem = Memory(cfg)
    yield mem
    mem.close()


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
    mem._chat = _FakeChat()  # type: ignore[assignment]
    yield mem
    mem.close()


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


class _StubEmbedder(EmbedderBase):
    """Deterministic, dependency-free embedder for backend-free unit runs.

    Produces stable `sha256`-derived unit vectors of the config's dim, so the
    whole non-`requires_mlx` suite runs with neither MLX nor sentence-transformers
    installed (the Linux CI condition) instead of erroring at the embed step.
    """

    def __init__(self, dims: int) -> None:
        self._dims = max(1, int(dims))

    @property
    def dims(self) -> int:
        return self._dims

    def _vec(self, text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8")).digest()
        vals = [((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(self._dims)]
        norm = sum(v * v for v in vals) ** 0.5 or 1.0
        return [v / norm for v in vals]

    def embed(self, inputs):
        return [self._vec(t) for t in inputs]

    def embed_text(self, text: str) -> list[float]:
        return self._vec(text)

    def embed_image(self, path) -> list[float]:
        return self._vec(f"image::{path}")

    def embed_audio(self, path) -> list[float]:
        return self._vec(f"audio::{path}")

    def unload(self) -> None:  # lifecycle no-op
        return None

    @property
    def model_name(self) -> str:
        return "stub-embedder"

    @property
    def is_warm(self) -> bool:
        return True


@pytest.fixture(autouse=True)
def _stub_embedder_backend_free(request, monkeypatch) -> None:
    """Make `Memory` use a deterministic stub embedder whenever no real backend
    is available (Linux CI: no MLX, no sentence-transformers), so embedder-dependent
    unit tests run instead of erroring at the embed step.

    Skipped for `requires_mlx` tests (they exercise the real embedder). On a dev
    box where MLX *is* importable the real embedder is kept — set
    `MEMO_TEST_FORCE_STUB=1` to force the stub locally and reproduce the CI path.
    """
    if request.node.get_closest_marker("requires_mlx"):
        return
    # Tests that drive the embedder themselves (dim/norm validation, batching,
    # cache, the real asymmetric-prefix path) opt out and keep full control.
    if request.node.get_closest_marker("no_stub_embedder"):
        return
    from memo.platform_detect import mlx_available

    if mlx_available() and not os.environ.get("MEMO_TEST_FORCE_STUB"):
        return  # real MLX embedder on Apple-Silicon dev boxes (no behaviour change)

    def _make_stub(cfg, *, cache_size=None):
        return _StubEmbedder(cfg.embedder_dims)

    monkeypatch.setattr("memo.embedder_select.make_embedder", _make_stub)
