"""Retrieve crushed JSON from crush cache via CLI.

Wave 1 token economy: L1 SmartCrusher offloads low-relevance JSON rows to cache.
This command retrieves the original (full) content using the marker hash —
either the SmartCrusher's own <<memo-crush:HASH>> form or the context-
compression proxy's bare hex key (memo.proxy.ccr); both read the one cache.

Usage: memo retrieve <<memo-crush:HASH>>  (or: memo retrieve HASH)
Output: JSON {"original": <original_string>, "hash": <hash>}
"""

from __future__ import annotations

import json

import click

from memo.config import Config
from memo.store.crush_cache import CrushCache


@click.command()
@click.argument("hash_marker", type=str)
def retrieve_cmd(hash_marker: str) -> None:
    """Retrieve original content from crush cache.

    Takes either an ingest-time crush marker (e.g.,
    <<memo-crush:abc123def456>>) or the proxy's bare hex key and returns the
    original that was offloaded — both name the same cache.

    Returns JSON with "original" (the full original string) and "hash" fields.
    """
    # Parse marker format: <<memo-crush:HASH>> -> extract HASH; anything else
    # is taken as a bare hex key (CrushCache validates the shape).
    if hash_marker.startswith("<<memo-crush:") and hash_marker.endswith(">>"):
        hash_val = hash_marker[13:-2]  # Strip <<memo-crush: and >>
    else:
        hash_val = hash_marker

    config = Config.from_env()
    cache = CrushCache(config.state_dir)

    original = cache.retrieve(hash_val)
    if original is None:
        click.echo(f"Error: cache entry not found or expired: {hash_val}", err=True)
        return

    # Output as JSON for easy shell piping
    click.echo(json.dumps({"original": original, "hash": hash_val}, ensure_ascii=False))
