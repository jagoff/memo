"""First-run opt-in consent for the anonymous usage heartbeat."""

from __future__ import annotations

import pytest


@pytest.fixture
def _consent_env(tmp_path, monkeypatch):
    """Isolated config dir + interactive terminal, check flag unset."""
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("MEMO_UPDATE_CHECK_ENABLED", raising=False)
    import memo.cli as cli

    monkeypatch.setattr("memo.runtime.install._init_is_interactive", lambda: True)
    return cli


def test_consent_yes_enables_usage_sharing(_consent_env, monkeypatch):
    from memo.flags import flag_bool

    monkeypatch.setattr("click.confirm", lambda *a, **k: True)
    _consent_env._prompt_usage_sharing()

    assert flag_bool("MEMO_UPDATE_CHECK_ENABLED") is True


def test_consent_no_leaves_usage_sharing_off(_consent_env, monkeypatch):
    from memo.flags import flag_bool

    monkeypatch.setattr("click.confirm", lambda *a, **k: False)
    _consent_env._prompt_usage_sharing()

    assert flag_bool("MEMO_UPDATE_CHECK_ENABLED") is False


def test_consent_skipped_when_non_interactive(tmp_path, monkeypatch):
    """No TTY / non-interactive → never prompts, never writes config."""
    monkeypatch.setenv("MEMO_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.delenv("MEMO_UPDATE_CHECK_ENABLED", raising=False)
    monkeypatch.setattr("memo.runtime.install._init_is_interactive", lambda: False)
    monkeypatch.setattr(
        "click.confirm",
        lambda *a, **k: pytest.fail("must not prompt when non-interactive"),
    )
    import memo.cli as cli

    cli._prompt_usage_sharing()

    from memo.flags import flag_bool

    assert flag_bool("MEMO_UPDATE_CHECK_ENABLED") is False
