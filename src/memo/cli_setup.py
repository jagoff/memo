from __future__ import annotations

import json
from pathlib import Path

import click

from memo.errors import SetupError
from memo.runtime.agent_registry import apply_setup_plan, build_setup_plan
from memo.runtime.mcp import _format_command


@click.command(name="setup")
@click.argument(
    "agent",
    required=False,
    default="all",
    type=click.Choice(["codex", "claude-code", "all"]),
)
@click.option(
    "--detect",
    is_flag=True,
    help="Configure only selected agents whose command is present on PATH.",
)
@click.option("--dry-run", is_flag=True, help="Print the complete plan without changing state.")
@click.option("--json", "as_json", is_flag=True, help="Emit the plan and receipt as JSON.")
def setup_cmd(agent: str, detect: bool, dry_run: bool, as_json: bool) -> None:
    """Configure memo MCP and memory-first instructions for an agent."""
    try:
        plan = build_setup_plan([agent], cwd=Path.cwd(), detect=detect)
        receipt = apply_setup_plan(plan, dry_run=dry_run)
    except SetupError as exc:
        raise click.ClickException(str(exc)) from exc

    payload = {"plan": plan.to_dict(), "receipt": receipt}
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        mode = "plan" if dry_run else "setup"
        click.echo(f"memo {mode} → {plan.memo_mcp}")
        if not plan.actions:
            click.echo("  no detected agents selected")
        for action, result in zip(plan.actions, receipt["results"], strict=True):
            marker = "✓" if result["ok"] else "✗"
            click.echo(f"  {marker} {action.agent}: {result['status']}")
            click.echo(f"    MCP: {_format_command(action.mcp_command)}")
            click.echo(f"    instructions: {action.instruction_path} ({result.get('instruction', '-')})")
            if result.get("backup"):
                click.echo(f"    backup: {result['backup']}")
            if result.get("error"):
                click.echo(f"    error: {result['error']}")
                click.echo(f"    remediation: {result.get('remediation', '-')}")
            click.echo(f"    {action.restart_guidance}")
    if not receipt["ok"]:
        raise click.exceptions.Exit(1)
