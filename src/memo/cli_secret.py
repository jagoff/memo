"""CLI commands for secret storage management."""

from __future__ import annotations

import json
import shlex

import click

from memo.config import Config
from memo.memory import Memory


@click.group()
def secret() -> None:
    """Manage encrypted secrets (passwords, tokens, SSH keys)."""
    pass


@secret.command()
@click.option("--name", required=True, help="Secret name (e.g., 'openai-api-key')")
@click.option(
    "--kind",
    type=click.Choice(
        ["api_token", "password", "ssh_key", "db_credential", "certificate", "generic"]
    ),
    help="Secret kind (auto-detected if not provided)",
)
@click.option("--value", required=False, help="Secret value (if not provided, read from stdin)")
@click.option("--no-confirm", is_flag=True, help="Skip confirmation prompt")
def save(name: str, kind: str | None, value: str | None, no_confirm: bool) -> None:
    """Save a secret (encrypted)."""
    if not value:
        value = click.get_text_stream("stdin").read().strip()

    if not value:
        raise click.ClickException("No value provided")

    cfg = Config.from_env()
    mem = Memory(cfg)

    try:
        rec = mem.save_secret(
            value=value,
            name=name,
            kind=kind,
            interactive=not no_confirm,
        )
    finally:
        mem.close()
    click.echo(f"✓ Saved secret '{name}' (id={rec.id[:8]})")


@secret.command()
@click.option("--name", required=True, help="Secret name")
def get(name: str) -> None:
    """Retrieve a secret (decrypted)."""
    cfg = Config.from_env()
    mem = Memory(cfg)

    try:
        value = mem.get_secret(name)
        click.echo(value)
    except Exception as exc:
        raise click.ClickException(f"Failed to retrieve secret: {exc}") from exc
    finally:
        mem.close()


@secret.command()
@click.option(
    "--kind",
    type=click.Choice(
        ["api_token", "password", "ssh_key", "db_credential", "certificate", "generic"]
    ),
    help="Filter by kind",
)
def list(kind: str | None) -> None:
    """List saved secrets (names only, no values)."""
    cfg = Config.from_env()
    mem = Memory(cfg)

    try:
        secrets_list = mem.list_secrets(kind=kind)
    finally:
        mem.close()
    if not secrets_list:
        click.echo("No secrets found")
        return

    click.echo(f"{'Name':<30} {'Kind':<15} {'Accesses':<10}")
    click.echo("-" * 55)
    for s in secrets_list:
        click.echo(f"{s['name']:<30} {s['kind']:<15} {s['accessed_count']:<10}")


@secret.command()
@click.option("--name", required=True, help="Secret name")
@click.confirmation_option(prompt="Are you sure you want to delete this secret?")
def forget(name: str) -> None:
    """Delete a secret."""
    cfg = Config.from_env()
    mem = Memory(cfg)

    try:
        mem.forget_secret(name)
        click.echo(f"✓ Deleted secret '{name}'")
    except Exception as exc:
        raise click.ClickException(f"Failed to delete secret: {exc}") from exc
    finally:
        mem.close()


@secret.command()
@click.option("--format", type=click.Choice(["env", "json"]), default="env", help="Export format")
def export(format: str) -> None:
    """Export all secrets (interactive confirm required)."""
    if not click.confirm("Export all secrets? This is a security risk. Continue?"):
        raise click.Abort()

    cfg = Config.from_env()
    mem = Memory(cfg)

    try:
        secrets_list = mem.list_secrets()
        if not secrets_list:
            click.echo("No secrets to export")
            return

        if format == "env":
            for s in secrets_list:
                value = mem.get_secret(s["name"])
                click.echo(f"export MEMO_SECRET_{_slugify(s['name']).upper()}={shlex.quote(value)}")
        elif format == "json":
            payload = {}
            for s in secrets_list:
                value = mem.get_secret(s["name"])
                payload[s["name"]] = {"kind": s["kind"], "value": value}
            click.echo(json.dumps(payload, indent=2))
    finally:
        mem.close()


def _slugify(text: str) -> str:
    """Convert name to env var safe slug."""
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").upper()
