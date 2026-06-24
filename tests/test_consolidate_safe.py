"""consolidate-safe default: dry-run unless explicitly forced.

Consolidation merges memorias and archives originals — a data-loss
operation. The safe contract:

- MCP `memo_consolidate_apply` defaults to `dry_run=True` (no mutation
  unless the caller opts in).
- CLI `memo consolidate apply` previews by default; `--force` is required
  to mutate, gated by an interactive confirmation unless `--yes` is given.
"""

from __future__ import annotations

import asyncio

from click.testing import CliRunner

from memo.cli import cli


class _SpyConsolidator:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def consolidate_all(self, **kwargs):
        self.calls.append(kwargs)
        return {"clusters": [], "proposals": [], "results": []}


class _SpyMem:
    def __init__(self) -> None:
        self.consolidator = _SpyConsolidator()


def _patch_cli_memory(monkeypatch) -> _SpyMem:
    spy = _SpyMem()
    monkeypatch.setattr("memo.cli_consolidate._get_memory", lambda cfg: spy)
    monkeypatch.setattr("memo.cli_consolidate.Config.from_env", staticmethod(lambda: object()))
    return spy


# ── MCP server default ────────────────────────────────────────────────────


def test_server_consolidate_defaults_to_read_only():
    """memo_consolidate (core) must default to dry_run=True — no mutation without opt-in."""
    from fastmcp import FastMCP

    from memo import server_core_records

    spy = _SpyMem()
    server = FastMCP("t")
    server_core_records.register(server, spy)
    tool = asyncio.run(server.get_tool("memo_consolidate")).fn
    tool()  # no dry_run arg → must default to a non-destructive preview
    assert spy.consolidator.calls[0]["dry_run"] is True


# ── CLI safe default ──────────────────────────────────────────────────────


def test_cli_consolidate_apply_defaults_to_dry_run(monkeypatch):
    spy = _patch_cli_memory(monkeypatch)
    result = CliRunner().invoke(cli, ["consolidate", "apply"])
    assert result.exit_code == 0, result.output
    assert spy.consolidator.calls, "consolidate_all was not called"
    assert spy.consolidator.calls[0]["dry_run"] is True


def test_cli_consolidate_apply_force_aborts_on_declined_confirm(monkeypatch):
    spy = _patch_cli_memory(monkeypatch)
    result = CliRunner().invoke(cli, ["consolidate", "apply", "--force"], input="n\n")
    assert result.exit_code != 0
    assert spy.consolidator.calls == [], "declined confirm must not mutate"


def test_cli_consolidate_apply_force_applies_on_confirm(monkeypatch):
    spy = _patch_cli_memory(monkeypatch)
    result = CliRunner().invoke(cli, ["consolidate", "apply", "--force"], input="y\n")
    assert result.exit_code == 0, result.output
    assert spy.consolidator.calls[0]["dry_run"] is False


def test_cli_consolidate_apply_force_yes_skips_confirm(monkeypatch):
    spy = _patch_cli_memory(monkeypatch)
    result = CliRunner().invoke(cli, ["consolidate", "apply", "--force", "--yes"])
    assert result.exit_code == 0, result.output
    assert spy.consolidator.calls[0]["dry_run"] is False
