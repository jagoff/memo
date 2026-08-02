from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.resource_hygiene_plugin import resource_warning_messages

pytest_plugins = ["pytester"]


def _expose_repo_to_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = str(Path(__file__).parents[1])
    current = os.environ.get("PYTHONPATH")
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join((repo_root, current)) if current else repo_root
    )
    # Keep pytester's subprocess focused on the explicitly loaded hygiene plugin.
    # Third-party auto-loaded plugins can emit unrelated startup warnings before
    # pytest produces a terminal summary, especially under PYTHONWARNINGS=error.
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")


def test_resource_warning_filter_rejects_only_unclosed_resources() -> None:
    caught = [
        SimpleNamespace(message=ResourceWarning("unclosed database in <sqlite3.Connection>")),
        SimpleNamespace(message=ResourceWarning("unclosed <MemoryObjectReceiveStream>")),
        SimpleNamespace(message=ResourceWarning("harmless advisory")),
        SimpleNamespace(message=DeprecationWarning("old API")),
    ]
    assert resource_warning_messages(caught) == [
        "unclosed database in <sqlite3.Connection>",
        "unclosed <MemoryObjectReceiveStream>",
    ]


def test_resource_warning_filter_rejects_unclosed_cpython_socket() -> None:
    message = "unclosed <socket.socket fd=3, family=2, type=1, proto=0, laddr=('0.0.0.0', 0)>"
    caught = [SimpleNamespace(message=ResourceWarning(message))]

    assert resource_warning_messages(caught) == [message]


def test_resource_hygiene_flag_fails_an_unclosed_sqlite_test(
    pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _expose_repo_to_subprocess(monkeypatch)
    pytester.makepyfile(
        """
        import sqlite3

        def test_leak():
            sqlite3.connect(\":memory:\")
        """
    )
    result = pytester.runpytest_subprocess(
        "-p", "tests.resource_hygiene_plugin", "--resource-hygiene", "-q"
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*unclosed database*"])


def test_resource_hygiene_flag_catches_a_leak_pending_before_test_setup(
    pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _expose_repo_to_subprocess(monkeypatch)
    monkeypatch.delenv("PYTHONWARNINGS", raising=False)
    pytester.makepyfile(
        """
        import gc
        import sqlite3

        gc.disable()
        cycle = []
        connection = sqlite3.connect(\":memory:\")
        cycle.append((cycle, connection))
        del connection, cycle

        def test_noop():
            pass
        """
    )
    result = pytester.runpytest_subprocess(
        "-p", "tests.resource_hygiene_plugin", "--resource-hygiene", "-q"
    )
    result.assert_outcomes(passed=1, errors=1)
    result.stdout.fnmatch_lines(["*unclosed database*"])


def test_resource_hygiene_plugin_is_inert_without_flag(
    pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    _expose_repo_to_subprocess(monkeypatch)
    pytester.makepyfile(
        """
        import sqlite3
        import warnings

        def test_leak_is_not_gated_by_default():
            with warnings.catch_warnings():
                warnings.simplefilter(\"ignore\", ResourceWarning)
                sqlite3.connect(\":memory:\")
        """
    )
    result = pytester.runpytest_subprocess("-p", "tests.resource_hygiene_plugin", "-q")
    result.assert_outcomes(passed=1)
