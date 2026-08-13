"""`memo reindex --rebuild` must not leave a global flag behind.

The command needs ``MEMO_SKIP_MODEL_VERSION_CHECK=1`` while it runs: a rebuild
re-embeds from the markdown source of truth, so the stamped-model guard has
nothing to compare against yet. It set the flag with
``os.environ.setdefault(...)`` and never removed it — harmless in a CLI process
that exits a second later, permanent in any process that keeps running.

The pytest process is exactly that. Once
``tests/conformance/test_index_rebuild_preserves.py`` invoked the command
through ``CliRunner``, the flag stayed set for every test after it, and
``pytest tests/`` on master reported **20 failures** that all pass in
isolation — the embedder-stamping and vec-dims guards in
``tests/test_store.py`` silently disarmed, plus an int8/float32 column clash in
``tests/test_vec_quantize.py``. ``tests/conformance/test_mcp_response_budget.py``
had already had to ``monkeypatch.delenv`` it defensively, noting the fix
belonged upstream of that workaround.

The MCP server is the other long-lived host: it exposes ``memo_reindex``, so a
single rebuild there would have disarmed the guard for the rest of the process.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from memo.cli import cli

FLAG = "MEMO_SKIP_MODEL_VERSION_CHECK"


def _env(data: Path, state: Path) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(data),
        "MEMO_STATE_DIR": str(state),
        "MEMO_NONINTERACTIVE": "1",
    }


@pytest.fixture
def store_dirs(tmp_path: Path) -> tuple[Path, Path]:
    data, state = tmp_path / "data", tmp_path / "state"
    data.mkdir()
    state.mkdir()
    return data, state


def test_rebuild_does_not_leak_the_flag(
    store_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, state = store_dirs
    monkeypatch.delenv(FLAG, raising=False)

    result = CliRunner().invoke(cli, ["reindex", "--rebuild"], env=_env(data, state))

    assert result.exit_code == 0, result.output
    assert FLAG not in os.environ, "reindex --rebuild left the guard disarmed process-wide"


def test_rebuild_restores_a_preexisting_value(
    store_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator who set it deliberately keeps their value."""
    data, state = store_dirs
    monkeypatch.setenv(FLAG, "0")

    CliRunner().invoke(cli, ["reindex", "--rebuild"], env=_env(data, state))

    assert os.environ[FLAG] == "0"


def test_plain_reindex_never_sets_it(
    store_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    data, state = store_dirs
    monkeypatch.delenv(FLAG, raising=False)

    CliRunner().invoke(cli, ["reindex"], env=_env(data, state))

    assert FLAG not in os.environ
