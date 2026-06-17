"""Identity layer — stable machine id + session/terminal provenance.

`current(cfg)` resolves a frozen `Identity` snapshot: `machine_id` from the
persisted `cfg.device_id`, `hostname` from the socket, `session_id` from env.
No MLX, no network — pure resolution against an isolated `Config` (the `tmp_cfg`
fixture, per tests/conftest.py isolation rules).
"""

from __future__ import annotations

from memo.config import Config
from memo.identity import Identity, current

# Every var `_session_id()` consults — cleared so the dev machine's own
# CLAUDE_SESSION_ID can't leak into the deterministic assertions below.
_SESSION_ENV_VARS = ("MEMO_SESSION_ID", "CLAUDE_SESSION_ID", "CLAUDE_CODE_SESSION_ID")


def _clear_session_env(monkeypatch) -> None:
    for k in _SESSION_ENV_VARS:
        monkeypatch.delenv(k, raising=False)


def test_machine_id_is_device_id_and_stable(tmp_cfg: Config):
    ident = current(tmp_cfg)
    assert isinstance(ident, Identity)
    assert ident.machine_id == tmp_cfg.device_id
    # device_id is persisted (state_dir/.device_id) → identical on a second call
    assert current(tmp_cfg).machine_id == ident.machine_id


def test_hostname_nonempty_and_label_includes_it(tmp_cfg: Config, monkeypatch):
    _clear_session_env(monkeypatch)
    ident = current(tmp_cfg)
    assert ident.hostname  # non-empty
    assert ident.hostname in ident.label
    # with no session id supplied, the label is exactly the hostname
    assert ident.session_id is None
    assert ident.label == ident.hostname


def test_session_id_from_env_in_label(tmp_cfg: Config, monkeypatch):
    _clear_session_env(monkeypatch)
    monkeypatch.setenv("MEMO_SESSION_ID", "abcdef0123456789")
    ident = current(tmp_cfg)
    assert ident.session_id == "abcdef0123456789"
    # label carries only the 8-char prefix (commit-attribution friendly)
    assert ident.label == f"{ident.hostname}·abcdef01"
    assert "abcdef01" in ident.label


def test_as_dict_has_expected_keys(tmp_cfg: Config):
    d = current(tmp_cfg).as_dict()
    assert set(d) == {"machine_id", "hostname", "session_id", "terminal", "label"}
