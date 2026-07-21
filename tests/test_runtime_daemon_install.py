from __future__ import annotations

from pathlib import Path

import pytest
from click import ClickException

from memo.runtime.daemon import _resolve_watcher_binary


def test_resolve_watcher_binary_preserves_explicit_path(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)

    assert _resolve_watcher_binary("/stable/bin/memo") == "/stable/bin/memo"


def test_resolve_watcher_binary_requires_discoverable_memo(monkeypatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: None)

    with pytest.raises(ClickException, match="Could not locate"):
        _resolve_watcher_binary(None)


def test_resolve_watcher_binary_rejects_project_venv(monkeypatch) -> None:
    candidate = "/workspace/memo/.venv/bin/memo"
    monkeypatch.setattr("shutil.which", lambda _: candidate)
    monkeypatch.setattr(
        "memo.runtime.install._is_project_venv_path",
        lambda path: path == Path(candidate),
    )

    with pytest.raises(ClickException, match="project venv"):
        _resolve_watcher_binary(None)
