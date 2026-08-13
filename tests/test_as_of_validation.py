"""A malformed ``as_of`` must be refused at the boundary, not silently dropped.

The store is deliberately lenient: ``_normalize_as_of`` falls back to the raw
string and ``_passes_validity_gate`` returns True on a bound it cannot parse,
so a garbled boundary never crashes a query mid-flight. That leniency is right
where it lives and is unchanged here.

What it cost at the EDGE was a lie. ``memo search "wal" --as-of not-a-date``
exited 0 and printed today's records, and ``memo_search_valid_as_of`` echoed
``{"as_of": "not-a-date"}`` beside present-day hits — a time-machine query
answered from the present, presented as the past. The sibling surfaces already
refuse the same input loudly (``memo_search_as_of``, ``memo_ask_as_of`` and
``memo diff`` all raise ``Invalid isoformat string``), so the leniency was not
even consistent.

Boundary parsing accepts exactly what ``_normalize_as_of`` understands: a bare
``YYYY-MM-DD`` date, a naive datetime, an offset-aware datetime, and ``Z`` as
an offset alias.
"""

from __future__ import annotations

import pytest

from memo.asof import validate_as_of
from memo.config import Config
from memo.errors import ValidationError

ACCEPTED = [
    "2026-01-01",
    "2026-01-01T14:00:00",
    "2026-01-01T14:00:00+00:00",
    "2026-01-01T14:00:00Z",
    "2026-01-01T14:00:00.123456-03:00",
]

REJECTED = [
    "not-a-date",
    "qa-as_of",
    "2026-13-01",
    "01/01/2026",
    "yesterday",
    "",
    "   ",
]


@pytest.mark.parametrize("value", ACCEPTED)
def test_validate_as_of_accepts_iso_bounds(value: str) -> None:
    assert validate_as_of(value) == value


def test_validate_as_of_passes_none_through() -> None:
    """`None` means "no valid-time filter" and is not an error."""
    assert validate_as_of(None) is None


@pytest.mark.parametrize("value", REJECTED)
def test_validate_as_of_rejects_garbage(value: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        validate_as_of(value)
    message = str(excinfo.value)
    assert "as-of" in message.lower()
    assert "YYYY-MM-DD" in message


def test_validate_as_of_names_the_offending_value() -> None:
    """The caller has to see WHICH string was refused to fix the typo."""
    with pytest.raises(ValidationError, match="not-a-date"):
        validate_as_of("not-a-date")


def _env(tmp_cfg: Config) -> dict[str, str]:
    return {
        "MEMO_DATA_DIR": str(tmp_cfg.data_dir),
        "MEMO_STATE_DIR": str(tmp_cfg.state_dir),
        "MEMO_NONINTERACTIVE": "1",
    }


def test_cli_search_refuses_a_malformed_as_of(tmp_cfg: Config) -> None:
    """`memo search --as-of garbage` must fail, not answer from the present."""
    from click.testing import CliRunner

    from memo.cli import cli

    result = CliRunner().invoke(cli, ["search", "wal", "--as-of", "not-a-date"], env=_env(tmp_cfg))

    assert result.exit_code != 0, result.output
    assert "not-a-date" in result.output


@pytest.mark.asyncio
async def test_mcp_search_valid_as_of_refuses_a_malformed_bound(tmp_cfg: Config) -> None:
    """The MCP twin must refuse it too instead of echoing it back as honoured."""
    from memo.memory import Memory
    from memo.server import build_server

    memory = Memory(tmp_cfg)
    try:
        server = build_server(memory=memory)
        with pytest.raises(Exception, match="not-a-date"):
            await server.call_tool(
                "memo_search_valid_as_of", {"query": "wal", "as_of": "not-a-date"}
            )
    finally:
        memory.close()
