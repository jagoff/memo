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


@profile_group.command(name="memory")
@click.option(
    "--scope",
    type=click.Choice(["current", "user", "project", "agent"]),
    default="current",
    show_default=True,
)
@click.option("--limit", type=click.IntRange(min=0, max=50), default=8, show_default=True)
@click.option("--budget-chars", type=click.IntRange(min=256, max=12000), default=4000)
@click.option("--json", "as_json", is_flag=True)
def memory_profile(scope: str, limit: int, budget_chars: int, as_json: bool) -> None:
    """Show the stable/active memory profile used by agents."""
    from memo.cli_common import get_memory
    from memo.memory_profile import build_memory_profile

    mem = get_memory(Config.from_env())
    try:
        payload = build_memory_profile(
            mem, scope=scope, limit=limit, budget_chars=budget_chars
        )
    finally:
        mem.close()
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    marker = "✓" if payload["available"] else "-"
    console.print(f"{marker} memory profile ({scope})")
    console.print(f"  stable: {len(payload['stable'])} · active: {len(payload['active'])}")
    for item in payload["active"]:
        console.print(f"  - [{item['id_short']}] {item['type']}: {item['title']}")


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
    console.print(
        f"  embedder: {active['embedder_model']}@{active['embedder_revision']} "
        f"[{active['embedder_dims']}]"
    )
    console.print(f"  llm:      {active['llm_model']}@{active['llm_revision']}")
    console.print(f"  helper:   {active['helper_model']}@{active['helper_revision']}")
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
