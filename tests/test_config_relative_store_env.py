"""A relative ``MEMO_DATA_DIR``/``MEMO_STATE_DIR`` from the environment is refused.

Two incidents on 2026-08-09, one root cause. A QA sweep left
``MEMO_DATA_DIR="sweep/store_cli_2/data"`` — a RELATIVE path — in two places:

* ``~/.claude.json`` (the memo MCP server's ``env``). The client's cwd was the
  home directory, so it resolved to a writable ``~/sweep/...``, memo created a
  brand-new empty store there, and every one of the 41 MCP tools answered from
  an empty corpus for days. ``memo_search`` returned zero hits in all three
  modes, ``memo_profile`` reported ``available: false``. The CLI, reading the
  real store, was fine — so nothing looked broken.
* ``com.memo.watch.plist``. launchd runs agents with cwd ``/``, so the same
  string resolved to ``/sweep``, a read-only filesystem, and the agent
  crash-looped under ``KeepAlive`` with a clear ``RuntimeError``.

The difference between "silently answers from an empty corpus" and "fails
immediately with a readable error" was nothing but the cwd it happened to
inherit. An environment-supplied store path is always meant to name one exact
directory; when it is relative, the caller cannot know which one it got.

Only the ENVIRONMENT is constrained. ``_apply_repo_and_legacy_paths`` still
sets relative ``data_dir="memorias"`` / ``state_dir=".memo-state"`` for the
zero-config in-repo case, where the cwd is the thing being described.
"""

from __future__ import annotations

import pytest

from memo.config import Config
from memo.errors import MemoError


@pytest.mark.parametrize("var", ["MEMO_DATA_DIR", "MEMO_STATE_DIR"])
def test_relative_store_env_is_refused(var: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(var, "sweep/store_cli_2/data")

    with pytest.raises(MemoError) as excinfo:
        Config.from_env()

    message = str(excinfo.value)
    assert var in message
    assert "sweep/store_cli_2/data" in message
    assert "absolute" in message.lower()


@pytest.mark.parametrize("var", ["MEMO_DATA_DIR", "MEMO_STATE_DIR"])
def test_absolute_store_env_is_accepted(
    var: str, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(var, str(tmp_path))

    cfg = Config.from_env()

    assert str(getattr(cfg, {"MEMO_DATA_DIR": "data_dir", "MEMO_STATE_DIR": "state_dir"}[var]))


@pytest.mark.parametrize("var", ["MEMO_DATA_DIR", "MEMO_STATE_DIR"])
def test_tilde_store_env_is_accepted(var: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """`~/...` names one exact directory regardless of cwd."""
    monkeypatch.setenv(var, "~/memo-store")

    Config.from_env()


def test_zero_config_repo_relative_defaults_still_work(monkeypatch: pytest.MonkeyPatch) -> None:
    """The in-repo default is relative BY DESIGN and must keep working."""
    monkeypatch.delenv("MEMO_DATA_DIR", raising=False)
    monkeypatch.delenv("MEMO_STATE_DIR", raising=False)

    Config.from_env()
