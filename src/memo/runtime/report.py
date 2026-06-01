"""Presentation of the runtime install report (extracted from cli_runtime)."""

from __future__ import annotations

from typing import Any

from memo.cli_common import console


def _print_runtime_install_report(report: dict[str, Any]) -> None:
    mode = report["mode"]
    root = report.get("root") or "(unknown)"
    if report["warnings"]:
        console.print(f"[yellow]![/yellow] install mode: {mode}  [dim]{root}[/dim]")
    else:
        console.print(f"[green]✓[/green] install mode: {mode}  [dim]{root}[/dim]")

    for key, label in (
        ("memo_cmd", "memo"),
        ("mcp_cmd", "memo-mcp"),
    ):
        raw = report.get(key)
        resolved = report.get(key.replace("_cmd", "_resolved"))
        if raw and resolved and raw != resolved:
            console.print(f"[dim]{label:14s}[/dim] {raw} -> {resolved}")
        elif raw:
            console.print(f"[dim]{label:14s}[/dim] {raw}")
        else:
            console.print(f"[dim]{label:14s}[/dim] (not found)")
    console.print(f"[dim]{'python':14s}[/dim] {report['python']}")
    for warning in report["warnings"]:
        console.print(f"[yellow]![/yellow] {warning}")
