"""Fixture builder shared by proxy transform tests.

Test-only: fixture builders like `make_zones` have no reason to ship inside
the installed wheel, so this lives under `tests/`, not `src/memo/proxy/`.
"""

from __future__ import annotations

from memo.proxy.zones import Zones


def make_zones(tool_names: list[str]) -> Zones:
    return Zones(
        tools=[
            {
                "name": name,
                "description": f"description of {name} " * 10,
                "input_schema": {"type": "object", "properties": {}},
            }
            for name in tool_names
        ],
        live_messages=[{"role": "user", "content": "hi"}],
    )
