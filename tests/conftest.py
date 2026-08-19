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

import atexit
import hashlib
import inspect
import json
import os
import shutil
import tempfile
from pathlib import Path

# Sanitize process-wide configuration before importing memo.config. A developer's
# live model/vault settings must never influence collection or touch real state.
os.environ["MEMO_NONINTERACTIVE"] = "1"
os.environ["MEMO_CONFIG_FILE"] = str(
    Path(tempfile.gettempdir()) / "memo-test-nonexistent-config.toml"
)
# Rich snapshots and CliRunner assertions must not inherit an outer shell's
# request to force ANSI escapes into captured, non-terminal output.
os.environ.pop("FORCE_COLOR", None)
# Tests run with cwd at the repo root (real .codegraph there) — keep cwd-discovery off for hermeticity.
os.environ["MEMO_CODEGRAPH_DISCOVERY"] = "0"
# A developer machine may pin MEMO_CODEGRAPH_DB (shell export) to a real index;
# tests must resolve only the paths they seed.
os.environ.pop("MEMO_CODEGRAPH_DB", None)
# Unique per test process (mkdtemp), NOT a fixed shared path: a fixed fallback
# accumulates a live sqlite index across runs, leaking one run's state (and a
# possibly stale schema) into every later run on the machine.
_TEST_PROCESS_STATE_DIR = Path(tempfile.mkdtemp(prefix="memo-test-state-"))
os.environ["MEMO_STATE_DIR"] = str(_TEST_PROCESS_STATE_DIR)
atexit.register(shutil.rmtree, _TEST_PROCESS_STATE_DIR, ignore_errors=True)
for _model_flag in (
    "MEMO_MODEL_PROFILE",
    "MEMO_LLM_MODEL",
    "MEMO_LLM_REVISION",
    "MEMO_HELPER_MODEL",
    "MEMO_HELPER_REVISION",
    "MEMO_EMBEDDER_MODEL",
    "MEMO_EMBEDDER_REVISION",
    "MEMO_EMBEDDER_DIMS",
    "MEMO_EMBEDDER_BACKEND",
    "MEMO_ST_EMBEDDER_MODEL",
    "MEMO_ST_EMBEDDER_REVISION",
    "MEMO_RERANKER_MODEL",
    "MEMO_RERANKER_REVISION",
    "MEMO_RERANKER_ENABLED",
):
    os.environ.pop(_model_flag, None)

import pytest  # noqa: E402
from freezegun import configure as configure_freezegun  # noqa: E402

from memo.config import Config  # noqa: E402

pytest_plugins = ["tests.resource_hygiene_plugin"]

configure_freezegun(extend_ignore_list=["transformers"])

# Test-wide defaults were hard-set before importing memo.config above. Individual
# tests can still override them with monkeypatch or CliRunner env mappings.
# `memo.config_md.config_dir()` defaults to `~/.config/memo/config` when
# `MEMO_CONFIG_DIR` is unset, so tests that build a `Config` via `from_env()`
# without pinning it read the developer's REAL markdown config (data_dir,
# model_profile, dream flags, etc. — whatever is actually configured on this
# machine) and silently override the test's own env/TOML-derived values.
# Point it at an empty dir by default so `field_values()` returns empty unless a
# test opts in with its own `MEMO_CONFIG_DIR` (monkeypatch.setenv or an explicit
# `env=`).
#
# The dir must be UNIQUE PER RUN. It used to be the fixed
# `$TMPDIR/memo-test-nonexistent-config-dir`, and "nonexistent" was an
# assumption, not an invariant: the first test that wrote a Markdown config
# without pinning its own `MEMO_CONFIG_DIR` created it — after which every
# later pytest run ON THAT MACHINE read those files as real config. Observed
# 2026-08-09: a stray `models-config.md` carrying `embedder_dims = 1024` and
# `model_profile = "balanced"` made `_embedder_was_pinned()` true, so
# `Config.from_env()` stopped adopting an index's embedder profile and 27 tests
# failed — reproducibly, forever, until the directory was deleted by hand. CI
# never saw it (fresh runner, empty TMPDIR), so it read as flakiness local to
# one machine. A per-run directory dies with the process that polluted it.
os.environ.setdefault(
    "MEMO_CONFIG_DIR",
    tempfile.mkdtemp(prefix="memo-test-config-"),
)
# Most server tests exercise the complete administrative contract. Production
# defaults to the 30-tool agent profile; the dedicated surface-profile tests
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
os.environ["MEMO_EMBEDDER_CLIENT_REQUIRE_DAEMON"] = "0"

# Neutralize production trinity flags that a developer's shell exports for live
# behavior but that break tests asserting a flag's *default* (support-lift 0.0,
# supersede gate off). Hard-set off (overrides an exported `=0.1` / `=3`) so
# `pytest` is hermetic on a dev machine. Tests that need them on opt back in via
# `monkeypatch.setenv` / `CliRunner(...).invoke(env=...)`.
os.environ["MEMO_SUPPORT_CONFIDENCE_LIFT"] = "0"
os.environ["MEMO_SUPERSEDE_SUPPORT_GATE"] = "0"

# Hard-set auto-update off so a developer's explicit opt-in cannot leak into
# pytest. Tests that exercise the updater opt back in explicitly.
os.environ["MEMO_AUTO_UPDATE"] = "0"

# Disable auto-project tagging so tests asserting exact tag sets aren't polluted
# by the cwd-derived `project:<repo>` tag. Hard-set at MODULE scope, like every
# other neutralisation here: the old `setdefault` inside the `tmp_cfg` fixture
# body lost to a developer's exported `=1` (4 confirmed failures) and, being
# applied lazily on first `tmp_cfg` use, made the value order-dependent for
# tests that don't use that fixture. Tests exercising the auto-tag flow opt back
# in via monkeypatch.setenv("MEMO_AUTO_PROJECT_TAG", "1").
os.environ["MEMO_AUTO_PROJECT_TAG"] = "0"

# Same class: `MEMO_SAVE_ABSORB` defaults ON, and the stub embedders used across
# the suite make every pair a cosine-1.0 near-duplicate — so on Apple Silicon a
# second save fires a REAL MLXChat generation (~6s) that folds the two records
# into one. Linux CI never sees it (MLX raises, absorb falls back to
# warn-and-create). Hard-set off; tests/test_save_absorb.py opts back in.
os.environ["MEMO_SAVE_ABSORB"] = "0"

# Trust & belief-revision program flags (memo v3.0.0+). A machine running the
# *activated* trust program exports these via ~/.claude/settings.json `env` (and
# the launchd fleet), and Claude Code passes them down to a `pytest` subprocess —
# so the render tests that assert a flag's DEFAULT (no `_trust_`/`⚔` line when
# MEMO_HIT_DOSSIER is off, etc.) fail as a false positive that never reproduces in
# CI. Drop them so the suite is hermetic on an activated dev machine; tests that
# exercise a flag opt back in via `monkeypatch.setenv` / `CliRunner(env=...)`.
for _trust_flag in (
    "MEMO_HIT_DOSSIER",
    "MEMO_RECALL_EPISTEMIC_LABELS",
    "MEMO_DECLARE_DISPUTES",
    "MEMO_CONTRADICT_PENALTY_ENABLED",
    "MEMO_GROUNDING_JUDGE",
    "MEMO_GROUNDING_ASK_MIN",
    "MEMO_CLAIM_SUPPORT",
    "MEMO_BELIEF_COMPETING",
    "MEMO_BELIEF_NWAY",
    "MEMO_FLOOR_CALIBRATION",
):
    os.environ.pop(_trust_flag, None)

# int8 vec quantization (memo v3.9.0+) graduated to the default. Existing
# fixtures/tests that build a `Memory`/`VecStore` and assert exact float32
# cosine scores (e.g. `pytest.approx(1.0, abs=1e-6)`) predate quantization and
# would flake against int8's ~1/127 precision. Soft-set off (`setdefault`, like
# the other pins above) so the suite is hermetic by default, but the CI int8
# lane can still export `MEMO_VEC_QUANTIZE=int8` and have it take effect; the
# dedicated `tests/test_vec_quantize.py` opts back in via `monkeypatch.delenv`/
# `monkeypatch.setenv`, and direct `VecStore(...)` construction without
# `vec_quant=` is unaffected (its own default is independent of this env var).
os.environ.setdefault("MEMO_VEC_QUANTIZE", "off")

# memo v4.3.0 graduated eight recommended flags from default-OFF to default-ON:
# the recall honest-empty gate + intra-injection dedup, verification-state decay,
# the crossref backlink index, save-time date normalization, and three nightly
# dream passes (validity extraction, quarantine graduation, flag graduation).
# Existing tests assert each flag's PRIOR default (an unset recall gate, no
# crossref rows, un-normalized bodies, a skipped dream pass), so hard-set them off
# — overriding both the new built-in default and an activated dev machine's
# markdown config — to keep `pytest` hermetic. Tests that exercise a flag opt back
# in via `monkeypatch.setenv(<flag>, "1")`; off-path tests must set "0" explicitly
# (a bare `delenv` now exposes the ON default rather than the OFF one).
for _default_on_flag in (
    "MEMO_RECALL_UNMATCHED_TERM_GATE",
    "MEMO_RECALL_INTRA_DEDUP",
    "MEMO_VERIFICATION_STATE_TRACKING",
    "MEMO_CROSSREF_INDEX",
    "MEMO_SAVE_NORMALIZE_DATES",
    "MEMO_DREAM_VALIDITY_EXTRACT_ENABLED",
    "MEMO_DREAM_GRADUATION_ENABLED",
    "MEMO_DREAM_FLAG_GRADUATION_ENABLED",
):
    os.environ[_default_on_flag] = "0"


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
    # Same trap as `memory_with_memories` below: the 4-bucket stub makes many
    # pairs cosine-1.0 near-duplicates, so default-ON MEMO_SAVE_ABSORB fires a
    # REAL 30B LLM merge on Apple Silicon (minutes per test) and folds the
    # second save INTO the first — tests asserting two distinct records then
    # fail here while staying green on a CI box with no MLX models.
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "0")
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
            ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0 for i in range(tmp_cfg.embedder_dims)
        ]
        norm = sum(v * v for v in values) ** 0.5
        return [v / norm for v in values]

    mem.embedder.embed = lambda inputs: [_fake_embedding(text) for text in inputs]
    mem.embedder.embed_query = lambda query: _fake_embedding(query)
    mem._chat = _FakeChat()  # type: ignore[assignment]
    yield mem
    mem.close()


@pytest.fixture
def memory_with_memories(tmp_cfg: Config, monkeypatch):
    """`Memory` with a constant-vector stub embedder (like `mem_with_stub`),
    seeded with two memories that share the keyword "chat" so
    `memo_search(query="chat")` reliably returns both regardless of any
    single memory's later content. Used by MCP tool-surface integration
    tests (e.g. the emission-ledger wiring) that need real, findable hits
    rather than a bare `Memory` instance.
    """
    from memo.memory import Memory

    cfg = Config(
        data_dir=tmp_cfg.data_dir,
        vault_path=tmp_cfg.vault_path,
        state_dir=tmp_cfg.state_dir,
        embedder_dims=4,
        reranker_enabled=False,
    )
    monkeypatch.setattr(
        "memo.embedder.MLXEmbedder.embed",
        lambda self, inputs: [[1.0, 0.0, 0.0, 0.0] for _ in inputs],
    )
    # The constant stub makes the two seeds cosine-1.0 near-duplicates, so
    # default-ON MEMO_SAVE_ABSORB would fire a REAL 30B LLM chat per setup on
    # Apple Silicon (~15s each, 180s timeout under GPU contention) and, when
    # that chat succeeds, absorb seed #2 INTO seed #1 — leaving one seeded
    # memory instead of the two the "both hits for query=chat" premise needs.
    monkeypatch.setenv("MEMO_SAVE_ABSORB", "0")
    mem = Memory(cfg)
    mem.save(content="Chat UI feedback loop and streaming design notes", title="Chat design notes")
    mem.save(content="Second chat memory used to exercise ledger dedup", title="Chat dedup memory")
    yield mem
    mem.close()


@pytest.fixture
def call_tool(memory_with_memories):
    """Invoke a registered MCP tool directly, bypassing the JSON-RPC
    transport -- the pattern `tests/test_server.py`'s `_tool()` helper
    already establishes (`FastMCP.get_tool` is async; the plain callable
    lives on the returned `FunctionTool`'s `.fn`). Builds the server once
    against `memory_with_memories` and returns a `(name, **kwargs) -> dict`
    callable. Some tools (e.g. `memo_ask`) are declared `async def`, so
    `.fn(**kwargs)` returns a coroutine rather than the result -- awaited
    here via a second one-shot `asyncio.run` so callers get a plain dict
    regardless of the tool's sync/async shape.
    """
    import asyncio

    from memo.server import build_server

    server = build_server(memory=memory_with_memories)

    def _call(name: str, **kwargs: object):
        tool = asyncio.run(server.get_tool(name))
        if tool is None:
            raise RuntimeError(f"tool {name!r} not registered")
        result = tool.fn(**kwargs)
        return asyncio.run(result) if inspect.isawaitable(result) else result

    return _call


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


@pytest.fixture(autouse=True)
def _default_unit_runtime_to_mlx(monkeypatch) -> None:
    """Make non-smoke LLM tests use the lightweight MLXChat double path.

    Production still rejects helper LLM calls off Apple Silicon. Unit tests that
    exercise helper-backed code patch `MLXChat.chat` or patch the extraction
    function itself, so constructing the wrapper is enough and must not depend
    on the host OS. Linux compatibility tests override this back to False.
    """
    import memo.embedder_select as embedder_select
    import memo.platform_detect as platform_detect

    monkeypatch.setattr(platform_detect, "mlx_available", lambda: True)
    monkeypatch.setattr(embedder_select, "mlx_available", lambda: True)
