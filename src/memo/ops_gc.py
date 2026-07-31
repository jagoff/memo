"""Pure GC logic over memo record dicts (`Record.to_dict()` shape).

Ported from synapse `ops.gc_vault_orphans` / `ops.gc_memo_duplicates` when the
synapse control plane was deprecated (2026-07-30). Pure functions — callers
(`cli_ops`) list records and perform the deletions.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable
from typing import Any


def find_vault_orphans(
    records: list[dict[str, Any]],
    *,
    path_exists: Callable[[str], bool] = os.path.exists,
) -> list[dict[str, Any]]:
    """Vault-ingested records whose source file no longer exists on disk.

    Only records where ``extra.source`` contains ``'vault-ingest'`` and
    ``extra.abs_path`` is set are considered; everything else is untouchable.
    """
    orphans: list[dict[str, Any]] = []
    for r in records:
        extra = r.get("extra") or {}
        abs_path = extra.get("abs_path")
        source = str(extra.get("source") or "")
        if "vault-ingest" in source and abs_path and not path_exists(abs_path):
            orphans.append(r)
    return orphans


def find_exact_duplicates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records whose body is an exact duplicate of a newer record.

    Groups by SHA-256 of the body; in each group the record with the newest
    ``updated`` (fallback ``created``) survives and the rest are returned as
    stale. Blank bodies are ignored entirely.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        body = str(r.get("body") or "")
        if not body.strip():
            continue
        groups.setdefault(hashlib.sha256(body.encode()).hexdigest(), []).append(r)
    stale: list[dict[str, Any]] = []
    for members in groups.values():
        if len(members) < 2:
            continue
        members.sort(key=lambda r: str(r.get("updated") or r.get("created") or ""), reverse=True)
        stale.extend(members[1:])
    return stale
