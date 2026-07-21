from __future__ import annotations

import re
import sys
from pathlib import Path
from types import ModuleType

import pytest

from memo.config import MODEL_PROFILES, Config
from memo.errors import MemoError
from memo.model_pins import (
    PINNED_MODEL_REVISIONS,
    hf_hub_cache_dir,
    model_identity,
    model_spec,
    resolve_model_snapshot,
)

_SHA40 = re.compile(r"[0-9a-f]{40}")


def test_hf_cache_dir_follows_hugging_face_environment_precedence(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-hub"
    hf_home = tmp_path / "hf-home"
    xdg = tmp_path / "xdg"

    assert hf_hub_cache_dir({"HF_HUB_CACHE": str(explicit), "HF_HOME": str(hf_home)}) == explicit
    assert hf_hub_cache_dir({"HF_HOME": str(hf_home), "XDG_CACHE_HOME": str(xdg)}) == (
        hf_home / "hub"
    )
    assert hf_hub_cache_dir({"XDG_CACHE_HOME": str(xdg)}) == xdg / "huggingface" / "hub"


def test_every_shipped_model_has_an_immutable_revision() -> None:
    cfg = Config()
    configured = {
        cfg.llm_model: cfg.llm_revision,
        cfg.helper_model: cfg.helper_revision,
        cfg.embedder_model: cfg.embedder_revision,
        cfg.st_embedder_model: cfg.st_embedder_revision,
        cfg.reranker_model: cfg.reranker_revision,
    }

    assert configured.items() <= PINNED_MODEL_REVISIONS.items()
    assert all(_SHA40.fullmatch(revision or "") for revision in configured.values())


def test_profiles_keep_model_and_revision_together() -> None:
    for profile in MODEL_PROFILES.values():
        for role in ("llm", "helper", "embedder", "reranker"):
            model_key = f"{role}_model"
            if model_key not in profile:
                continue
            model = str(profile[model_key])
            revision = str(profile[f"{role}_revision"])
            assert PINNED_MODEL_REVISIONS[model] == revision


def test_custom_remote_requires_inline_or_explicit_commit() -> None:
    with pytest.raises(ValueError, match="40-character commit SHA"):
        model_spec("someone/custom-model")

    sha = "a" * 40
    assert model_spec(f"someone/custom-model@{sha}").revision == sha
    assert model_spec("someone/custom-model", revision=sha).revision == sha


def test_branch_and_short_hash_are_rejected() -> None:
    with pytest.raises(ValueError, match="40-character commit SHA"):
        model_spec("someone/custom-model@main")
    with pytest.raises(ValueError, match="40-character commit SHA"):
        model_spec("someone/custom-model", revision="deadbeef")


def test_local_path_is_allowed_without_a_revision(tmp_path: Path) -> None:
    local = tmp_path / "model"
    spec = model_spec(str(local))

    assert spec.is_local is True
    assert spec.revision is None
    assert resolve_model_snapshot(str(local)) == str(local)
    assert model_identity(str(local)) == str(local)


def test_remote_snapshot_is_resolved_before_loader_use(monkeypatch) -> None:
    calls: dict[str, str] = {}
    hf = ModuleType("huggingface_hub")

    def snapshot_download(*, repo_id: str, revision: str) -> str:
        calls.update(repo_id=repo_id, revision=revision)
        return "/cache/exact-snapshot"

    hf.snapshot_download = snapshot_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf)
    sha = "b" * 40

    assert resolve_model_snapshot("someone/custom-model", revision=sha) == ("/cache/exact-snapshot")
    assert calls == {"repo_id": "someone/custom-model", "revision": sha}


def test_env_remote_override_without_revision_fails_clearly(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("MEMO_LLM_MODEL", "someone/custom-model")
    monkeypatch.delenv("MEMO_LLM_REVISION", raising=False)

    with pytest.raises(MemoError, match=r"MEMO_LLM_MODEL.*40-character commit SHA"):
        Config.from_env()


def test_env_remote_override_accepts_exact_explicit_revision(monkeypatch, tmp_path) -> None:
    sha = "c" * 40
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("MEMO_LLM_MODEL", "someone/custom-model")
    monkeypatch.setenv("MEMO_LLM_REVISION", sha)

    cfg = Config.from_env()

    assert cfg.llm_model == "someone/custom-model"
    assert cfg.llm_revision == sha


def test_audited_preferred_model_override_inherits_its_known_pin(monkeypatch, tmp_path) -> None:
    model = "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit-DWQ"
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("MEMO_LLM_MODEL", model)
    monkeypatch.delenv("MEMO_LLM_REVISION", raising=False)

    cfg = Config.from_env()

    assert cfg.llm_revision == PINNED_MODEL_REVISIONS[model]


def test_env_local_override_does_not_inherit_profile_revision(monkeypatch, tmp_path) -> None:
    local = tmp_path / "local-model"
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("MEMO_LLM_MODEL", str(local))
    monkeypatch.delenv("MEMO_LLM_REVISION", raising=False)

    cfg = Config.from_env()

    assert cfg.llm_model == str(local)
    assert cfg.llm_revision is None


def test_markdown_remote_override_does_not_inherit_profile_revision(monkeypatch, tmp_path) -> None:
    config_home = tmp_path / "config-home"
    config_dir = config_home / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "models-config.md").write_text(
        '```toml\n[models]\nllm_model = "someone/custom-model"\n```\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(config_home))
    monkeypatch.setenv("MEMO_CONFIG_FILE", str(tmp_path / "missing.toml"))
    monkeypatch.delenv("MEMO_LLM_MODEL", raising=False)
    monkeypatch.delenv("MEMO_LLM_REVISION", raising=False)

    with pytest.raises(MemoError, match=r"MEMO_LLM_MODEL.*40-character commit SHA"):
        Config.from_env()
