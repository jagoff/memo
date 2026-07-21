from __future__ import annotations

import pytest
from click import ClickException

from memo.runtime.codex import _codex_match_app_server_response


def test_match_app_server_response_ignores_non_object_json() -> None:
    seen: list[str] = []

    assert _codex_match_app_server_response(b"[]", 7, seen) is None
    assert _codex_match_app_server_response(b'"noise"', 7, seen) is None
    assert seen == ["[]", '"noise"']


def test_match_app_server_response_ignores_unrelated_and_returns_result() -> None:
    seen: list[str] = []

    assert _codex_match_app_server_response(b'{"id":6,"result":{}}', 7, seen) is None
    assert _codex_match_app_server_response(b'{"id":7,"result":{"ok":true}}', 7, seen) == {
        "ok": True
    }


def test_match_app_server_response_raises_remote_error() -> None:
    with pytest.raises(ClickException, match="request failed"):
        _codex_match_app_server_response(
            b'{"id":7,"error":{"message":"request failed"}}',
            7,
            [],
        )
