"""`memo profile` command group — model-profile status + repair plan.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(profile_group)`.
"""

from __future__ import annotations

import json

import click

from memo.cli_common import console
from memo.cli_diag import _profile_repair_plan, _profile_status_report
from memo.config import Config


@click.group(name="profile")
def profile_group() -> None:
    """Inspect active model profile, embedding dims, and repair guidance."""


@profile_group.command(name="status")
@click.option("--json", "as_json", is_flag=True)
@click.option("--no-db", "no_db", is_flag=True, help="Skip read-only DB dimension checks.")
def profile_status(as_json: bool, no_db: bool) -> None:
    """Show the active model profile and whether DB dimensions align."""
    cfg = Config.from_env()
    report = _profile_status_report(cfg, include_db=not no_db)
    if as_json:
        click.echo(json.dumps(report, ensure_ascii=False, indent=2))
        return

    marker = "[green]✓[/green]" if report["ok"] else "[red]✗[/red]"
    console.print(f"{marker} profile: {report['profile']} ({report['status']})")
    active = report["active"]
    console.print(f"  embedder: {active['embedder_model']} [{active['embedder_dims']}]")
    console.print(f"  llm:      {active['llm_model']}")
    console.print(f"  helper:   {active['helper_model']}")
    revision = active.get("reranker_revision") or "unpinned"
    console.print(
        f"  reranker: {active['reranker_model']} "
        f"revision={revision} enabled={active['reranker_enabled']}"
    )
    db = report["db"]
    if db["status"] != "not_checked":
        console.print(
            "  db dims:  "
            f"vec={db.get('vec_dims')} repo_vec={db.get('repo_vec_dims')} "
            f"expected={db.get('expected_dims')}"
        )
    for override in report["overrides"]:
        console.print(
            "[yellow]![/yellow] override "
            f"{override['field']}: expected={override['expected']} actual={override['actual']}"
        )
    for model in report["models"]:
        marker = "[green]✓[/green]" if model["cached"] else "[yellow]![/yellow]"
        console.print(f"  {marker} cached {model['role']}: {model['model']}")


@profile_group.command(name="repair-plan")
@click.option("--json", "as_json", is_flag=True)
@click.option("--no-db", "no_db", is_flag=True, help="Skip read-only DB dimension checks.")
def profile_repair_plan(as_json: bool, no_db: bool) -> None:
    """Print a non-executing repair plan for profile or dimension drift."""
    cfg = Config.from_env()
    plan = _profile_repair_plan(cfg, include_db=not no_db)
    if as_json:
        click.echo(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    console.print(f"profile repair plan: {plan['status']}")
    for action in plan["actions"]:
        console.print(f"- {action['severity']} {action['kind']}: {action['reason']}")
        for command in action["commands"]:
            console.print(f"  {command}")
