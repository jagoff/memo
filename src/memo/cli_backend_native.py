"""`memo backend-native` command group — Synapse backend-native capabilities.

Extracted from cli.py (3a god-module decomposition). Registered onto the
root group in cli.py via `cli.add_command(backend_native_group)`.
"""

from __future__ import annotations

import json

import click

from memo.cli_common import _backend_native_trace_id, _memo_backend_version, console
from memo.cli_diag import _profile_status_report
from memo.config import Config


@click.group(name="backend-native")
def backend_native_group() -> None:
    """Expose Synapse backend_native.v1 endpoints."""


@backend_native_group.command(name="capabilities")
@click.option("--json", "as_json", is_flag=True)
@click.option("--trace-id", default="")
def backend_native_capabilities(as_json: bool, trace_id: str) -> None:
    from memo.memory import NATIVE_BACKEND_PROTOCOL_VERSION, SYNAPSE_BACKEND_NATIVE_SCHEMA

    cfg = Config.from_env()
    payload = {
        "schema": SYNAPSE_BACKEND_NATIVE_SCHEMA,
        "protocol_version": NATIVE_BACKEND_PROTOCOL_VERSION,
        "backend": "memo",
        "supported": True,
        "backend_version": _memo_backend_version(),
        "capabilities": {
            "health": True,
            "capabilities": True,
            "replay_resolve": True,
            "memory_replay": True,
            "operational_replay": True,
            "trace_id": True,
        },
        "endpoints": {
            "capabilities": "backend-native capabilities --json",
            "replay_resolve": "backend-native replay-resolve <uri> --json",
        },
        "provenance_prefixes": ["memo://"],
        "trace_id": _backend_native_trace_id(trace_id),
        "model_profile": _profile_status_report(cfg, include_db=False, include_env=False),
    }
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print("[green]memo backend_native.v1: supported[/green]")


@backend_native_group.command(name="replay-resolve")
@click.argument("uri")
@click.option("--json", "as_json", is_flag=True)
@click.option("--trace-id", default="")
def backend_native_replay_resolve(uri: str, as_json: bool, trace_id: str) -> None:
    from memo.memory import Memory

    mem = Memory(Config.from_env())
    payload = mem.backend_native_replay_resolve(
        uri,
        trace_id=_backend_native_trace_id(trace_id),
        backend_version=_memo_backend_version(),
    )
    if as_json:
        click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    console.print(f"{payload['status']}: {payload['detail']}")
