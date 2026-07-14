"""Source-aware draft behavior for the configuration TUI."""

from __future__ import annotations

from pathlib import Path

import pytest

from memo.tui.config import session as session_module
from memo.tui.config.session import ConfigSession, ValueSource


def test_session_distinguishes_markdown_and_effective_env(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text("```toml\n[recall]\ntop_k = 7\n```\n", encoding="utf-8")
    env = {
        "MEMO_CONFIG_DIR": str(home),
        "MEMO_RECALL_TOP_K": "2",
        "MEMO_DATA_DIR": str(tmp_path / "data"),
        "MEMO_STATE_DIR": str(tmp_path / "state"),
    }

    state = ConfigSession.open(env).state("recall.top_k")

    assert state.configured_value == 7
    assert state.effective_value == 2
    assert state.source is ValueSource.ENV
    assert state.env_override == "2"


def test_process_session_uses_runtime_config_as_effective_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from memo.config import Config

    runtime_data = tmp_path / "runtime-data"
    runtime = Config(data_dir=runtime_data, state_dir=tmp_path / "runtime-state")
    monkeypatch.setattr(Config, "from_env", classmethod(lambda cls: runtime))
    monkeypatch.setattr(
        session_module.os,
        "environ",
        {"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")},
    )

    state = ConfigSession.open().state("storage.data_dir")

    assert state.effective_value == runtime_data.resolve()


def test_behavior_flag_effective_value_uses_runtime_accessor(tmp_path: Path) -> None:
    session = ConfigSession.open(
        {
            "MEMO_CONFIG_DIR": str(tmp_path / "memo-home"),
            "MEMO_RECALL_TOP_K": "-1",
        }
    )

    state = session.state("recall.top_k")

    assert state.effective_value == 3
    assert state.source is ValueSource.ENV
    assert state.issues


def test_runtime_accessor_can_resolve_invalid_flag_to_none(tmp_path: Path) -> None:
    session = ConfigSession.open(
        {
            "MEMO_CONFIG_DIR": str(tmp_path / "memo-home"),
            "MEMO_DECAY_HALFLIFE_REFERENCE": "invalid",
        }
    )

    state = session.state("search.decay_halflife_reference")

    assert state.effective_value is None
    assert state.issues


def test_draft_set_and_unset_never_write(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    env = {"MEMO_CONFIG_DIR": str(home), "MEMO_DATA_DIR": str(tmp_path / "data")}
    session = ConfigSession.open(env)

    session.set_value("recall.top_k", 9)

    assert session.state("recall.top_k").pending_value == 9
    assert not home.exists()

    session.unset_value("recall.top_k")

    assert session.state("recall.top_k").pending_unset is True
    assert not home.exists()


def test_invalid_numeric_value_blocks_review(tmp_path: Path) -> None:
    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")})

    session.set_value("recall.top_k", -1)

    assert any(issue.blocking and issue.key == "recall.top_k" for issue in session.issues())
    assert session.review().blocked is True


def test_runtime_only_setting_rejects_persistent_draft(tmp_path: Path) -> None:
    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")})

    with pytest.raises(ValueError, match="runtime-only"):
        session.set_value("misc.noninteractive", True)


def test_discard_clears_all_pending_operations(tmp_path: Path) -> None:
    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")})
    session.set_value("recall.top_k", 9)
    session.set_value("recall.min_sim", 0.7)

    session.discard()

    assert session.review().changes == ()
    assert all(state.pending_value is None for state in session.states())


def test_memories_in_vault_requires_vault_path(tmp_path: Path) -> None:
    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")})

    session.set_value("storage.memories_in_vault", True)

    assert any(
        issue.key == "storage.memories_in_vault" and "vault_path" in issue.message
        for issue in session.issues()
    )


def test_known_embedder_model_requires_matching_dimensions(tmp_path: Path) -> None:
    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")})

    session.set_value("models.embedder_dims", 2560)

    assert any(
        issue.key == "models.embedder_dims" and "1024" in issue.message
        for issue in session.issues()
    )


def test_platform_gated_draft_is_blocked_when_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(session_module, "is_apple_silicon", lambda: False)
    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(tmp_path / "memo-home")})

    session.set_value("models.reranker_enabled", True)

    state = session.state("models.reranker_enabled")
    assert state.available is False
    assert any(issue.blocking and "Apple Silicon" in issue.message for issue in state.issues)


def test_session_surfaces_malformed_markdown(tmp_path: Path) -> None:
    home = tmp_path / "memo-home"
    cfg = home / "config"
    cfg.mkdir(parents=True)
    (cfg / "recall-config.md").write_text("```toml\n[recall\ntop_k = 5\n```\n", encoding="utf-8")

    session = ConfigSession.open({"MEMO_CONFIG_DIR": str(home)})

    assert any(issue.blocking and "TOML parse error" in issue.message for issue in session.issues())
