"""`memo share` command group — memoria sharing + comments.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(share_group)`.
"""

from __future__ import annotations

import json

import click
from rich.table import Table

from memo.cli_common import console
from memo.cli_common import get_memory as _get_memory
from memo.config import Config

# -- sharing commands -----------------------------------------------------------


@click.group(name="share")
def share_group() -> None:
    """Memory sharing — share memorias with others."""
    pass


@share_group.command(name="with-user")
@click.argument("memoria_id")
@click.argument("shared_with")
@click.option("--permission", type=click.Choice(["read", "comment", "edit", "admin"]), default="read",
              help="Permission level")
@click.option("--expires-days", type=int, help="Days until expiration")
def share_with_user(memoria_id: str, shared_with: str, permission: str, expires_days: int | None) -> None:
    """Share a memoria with a user.

    Example: memo share with-user abc123 user@example.com --permission comment
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    share = mem.sharing.share_with_user(
        memoria_id=memoria_id,
        shared_with=shared_with,
        permission=permission,
        expires_days=expires_days,
    )

    console.print(f"[green]Shared {memoria_id[:8]} with {shared_with}[/green]")
    console.print(f"Permission: {permission}")
    if share.expires_at:
        console.print(f"Expires: {share.expires_at}")


@share_group.command(name="unshare")
@click.argument("memoria_id")
@click.argument("shared_with")
def share_unshare(memoria_id: str, shared_with: str) -> None:
    """Unshare a memoria from a user.

    Example: memo share unshare abc123 user@example.com
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    success = mem.sharing.unshare_with_user(memoria_id, shared_with)

    if success:
        console.print(f"[green]Unshared {memoria_id[:8]} from {shared_with}[/green]")
    else:
        console.print("[yellow]Share not found[/yellow]")


@share_group.command(name="create-link")
@click.argument("memoria_id")
@click.option("--permission", type=click.Choice(["read", "comment", "edit"]), default="read",
              help="Permission level")
@click.option("--expires-hours", type=int, default=24, help="Hours until expiration")
@click.option("--password", help="Optional password protection")
def share_create_link(memoria_id: str, permission: str, expires_hours: int, password: str | None) -> None:
    """Create a temporary sharing link.

    Example: memo share create-link abc123 --permission read --expires-hours 48
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    link = mem.sharing.create_link(
        memoria_id=memoria_id,
        permission=permission,
        expires_hours=expires_hours,
        password=password,
    )

    console.print("[green]Share link created[/green]")
    console.print(f"Link: {link}")
    console.print(f"Expires in {expires_hours} hours")


@share_group.command(name="list")
@click.argument("memoria_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def share_list(memoria_id: str, as_json: bool) -> None:
    """List all shares for a memoria.

    Example: memo share list abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    shares = mem.sharing.share_store.get_shares(memoria_id)

    if as_json:
        click.echo(json.dumps([s.__dict__ for s in shares], indent=2))
        return

    if not shares:
        console.print("[dim]No shares found[/dim]")
        return

    table = Table(title=f"Shares for {memoria_id[:8]}")
    table.add_column("Shared With", style="cyan")
    table.add_column("Permission", style="yellow")
    table.add_column("Shared At", style="green")
    table.add_column("Expires", style="magenta")

    for s in shares[:20]:
        table.add_row(
            s.shared_with,
            s.permission,
            s.shared_at[:19],
            s.expires_at[:19] if s.expires_at else "Never",
        )

    console.print(table)
    if len(shares) > 20:
        console.print(f"[dim]...and {len(shares) - 20} more[/dim]")


@share_group.command(name="comment")
@click.argument("memoria_id")
@click.argument("content")
@click.option("--author", default="user", help="Comment author")
@click.option("--parent", help="Parent comment ID for replies")
def share_comment(memoria_id: str, content: str, author: str, parent: str | None) -> None:
    """Add a comment to a memoria.

    Example: memo share comment abc123 "This is a comment" --author "John Doe"
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    comment = mem.sharing.add_comment(
        memoria_id=memoria_id,
        author=author,
        content=content,
        parent_id=parent,
    )

    console.print("[green]Comment added[/green]")
    console.print(f"Author: {comment.author}")
    console.print(f"Content: {comment.content}")


@share_group.command(name="comments")
@click.argument("memoria_id")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def share_comments(memoria_id: str, as_json: bool) -> None:
    """List all comments for a memoria.

    Example: memo share comments abc123
    """
    cfg = Config.from_env()
    mem = _get_memory(cfg)

    comments = mem.sharing.get_comments(memoria_id)

    if as_json:
        click.echo(json.dumps([c.__dict__ for c in comments], indent=2))
        return

    if not comments:
        console.print("[dim]No comments found[/dim]")
        return

    table = Table(title=f"Comments for {memoria_id[:8]}")
    table.add_column("Author", style="cyan")
    table.add_column("Content", style="yellow")
    table.add_column("Created", style="green")

    for c in comments[:20]:
        table.add_row(
            c.author,
            c.content[:50],
            c.created_at[:19],
        )

    console.print(table)
    if len(comments) > 20:
        console.print(f"[dim]...and {len(comments) - 20} more[/dim]")
