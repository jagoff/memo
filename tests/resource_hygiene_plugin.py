from __future__ import annotations

import gc
import warnings
from collections.abc import Iterable
from typing import Any

import pytest

_UNCLOSED_RESOURCE_MARKERS = (
    "unclosed database",
    "unclosed <memoryobject",
    "unclosed <socket.socket",
    "unclosed socket",
    "unclosed file",
)


def resource_warning_messages(caught: Iterable[Any]) -> list[str]:
    messages: list[str] = []
    for warning in caught:
        if not isinstance(warning.message, ResourceWarning):
            continue
        text = str(warning.message)
        if any(marker in text.lower() for marker in _UNCLOSED_RESOURCE_MARKERS):
            messages.append(text)
    return messages


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("memo testing").addoption(
        "--resource-hygiene",
        action="store_true",
        default=False,
        help="fail selected tests that leak SQLite, stream, socket, or file resources",
    )


@pytest.fixture(autouse=True)
def _enforce_resource_hygiene(request: pytest.FixtureRequest):
    if not request.config.getoption("--resource-hygiene"):
        yield
        return

    gc.collect()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        yield
        gc.collect()

    leaked = resource_warning_messages(caught)
    if leaked:
        pytest.fail("unclosed resources:\n" + "\n".join(f"- {item}" for item in leaked))
