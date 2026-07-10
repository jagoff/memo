"""Linux/Ubuntu compatibility guards (CPU sentence-transformers backend).

These pin the platform-aware behavior so a healthy Linux `[cpu]` install is not
misreported: `memo doctor` must probe sentence-transformers (not MLX) and not
false-fail; LLM features degrade with a clean MemoError; the Obsidian registry
path is per-OS; and RepoCorpus routes through the embedder factory rather than
hard-constructing MLXEmbedder.
"""

from __future__ import annotations

import sys
import types
from contextlib import closing

import pytest
from click.testing import CliRunner

from memo.cli import cli
from memo.errors import MemoError


def _install_fake_st(monkeypatch) -> None:
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)


# ── doctor: backend-aware import probe ──────────────────────────────────────


def test_doctor_json_does_not_false_fail_on_cpu_backend(monkeypatch, tmp_cfg):
    _install_fake_st(monkeypatch)
    from memo.cli_diag import _doctor_report

    cfg = tmp_cfg.model_copy(update={"embedder_backend": "st"})
    report = _doctor_report(cfg, check_db=False, strict_runtime=False, do_gc=False, fix=False)

    labels = {item["label"] for item in report["imports"]}
    assert "sentence_transformers" in labels
    assert "mlx" not in labels  # MLX is not probed on the CPU backend
    st = next(i for i in report["imports"] if i["label"] == "sentence_transformers")
    assert st["ok"] is True


def test_doctor_json_probes_mlx_on_mlx_backend(tmp_cfg):
    from memo.cli_diag import _doctor_report

    cfg = tmp_cfg.model_copy(update={"embedder_backend": "mlx"})
    report = _doctor_report(cfg, check_db=False, strict_runtime=False, do_gc=False, fix=False)
    labels = {item["label"] for item in report["imports"]}
    assert "mlx" in labels
    assert "sentence_transformers" not in labels


def test_doctor_human_uses_st_probe_on_cpu_backend(tmp_path):
    result = CliRunner().invoke(
        cli,
        ["doctor"],
        env={
            "MEMO_NONINTERACTIVE": "1",
            "MEMO_DATA_DIR": str(tmp_path / "data"),
            "MEMO_STATE_DIR": str(tmp_path / "state"),
            "MEMO_EMBEDDER_BACKEND": "st",
        },
        catch_exceptions=False,
    )
    # Backend-aware: the CPU backend reports sentence-transformers, never the
    # "mlx + mlx_lm importable" line. (Exit code not asserted — depends on
    # whether [cpu] is installed on the host.)
    assert "sentence-transformers" in result.output
    assert "mlx + mlx_lm importable" not in result.output


# ── LLM features degrade with a clean error off MLX ─────────────────────────


def test_ensure_chat_raises_memoerror_without_mlx(monkeypatch, tmp_cfg):
    import memo.platform_detect as pd

    monkeypatch.setattr(pd, "mlx_available", lambda: False)  # simulate Linux
    monkeypatch.setenv("MEMO_EMBEDDER_VIA_DAEMON", "0")  # force in-process embedder
    from memo.memory import Memory

    cfg = tmp_cfg.model_copy(update={"embedder_backend": "st"})
    with closing(Memory(cfg)) as mem:
        with pytest.raises(MemoError, match="MLX"):
            mem._ensure_chat()


# ── per-OS Obsidian registry path ───────────────────────────────────────────


@pytest.mark.parametrize(
    ("platform", "fragment"),
    [
        ("darwin", "Library/Application Support/obsidian"),
        ("linux", ".config/obsidian"),
    ],
)
def test_obsidian_registry_path_per_os(monkeypatch, platform, fragment):
    monkeypatch.setattr(sys, "platform", platform)
    from memo.setup.vaults import _default_obsidian_registry

    assert fragment in str(_default_obsidian_registry())


# ── RepoCorpus routes through the factory ───────────────────────────────────


def test_repocorpus_uses_factory_not_hardcoded_mlx(tmp_cfg):
    from memo.repo_index import RepoCorpus

    cfg = tmp_cfg.model_copy(update={"embedder_backend": "st"})
    corpus = RepoCorpus(cfg)
    try:
        assert type(corpus.embedder).__name__ == "STEmbedder"
    finally:
        corpus.store.close()


# ── per-OS MCP-client config paths ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("platform", "fragment"),
    [
        ("darwin", "Library/Application Support/Claude"),
        ("linux", ".config/Claude"),
    ],
)
def test_claude_desktop_config_path_per_os(monkeypatch, platform, fragment):
    monkeypatch.setattr(sys, "platform", platform)
    from memo.cli_install_mcp import _claude_desktop_config_path

    assert fragment in str(_claude_desktop_config_path())


def test_devin_desktop_scan_includes_mcp_path():
    from memo.runtime.mcp_config import KNOWN_MCP_CONFIGS

    assert "~/.devin/mcp.json" in KNOWN_MCP_CONFIGS


# ── reranker (MLX-only) forced off on non-Apple-Silicon ─────────────────────


def test_reranker_forced_off_on_non_apple_silicon(monkeypatch, tmp_path):
    # The default MODEL_PROFILES set reranker_enabled=True; on a non-Apple-Silicon
    # host that must be overridden (the cross-encoder is MLX-only) so hybrid search
    # never tries to load it.
    monkeypatch.setattr("memo.config.is_apple_silicon", lambda: False)
    monkeypatch.delenv("MEMO_RERANKER_ENABLED", raising=False)
    monkeypatch.delenv("MEMO_MODEL_PROFILE", raising=False)
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    from memo.config import Config

    assert Config.from_env().reranker_enabled is False


def test_reranker_explicit_env_overrides_platform_gate(monkeypatch, tmp_path):
    monkeypatch.setattr("memo.config.is_apple_silicon", lambda: False)
    monkeypatch.setenv("MEMO_RERANKER_ENABLED", "1")
    monkeypatch.setenv("MEMO_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MEMO_STATE_DIR", str(tmp_path / "s"))
    from memo.config import Config

    assert Config.from_env().reranker_enabled is True
