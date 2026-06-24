"""Backend-native replay resolution for `Memory`.

`_ReplayOpsMixin` resolves `memo://` / `memflow://` replay URIs back to their
source payloads (memory, repo-index chunk, repo). Extracted from
maintain_ops.py (god-module decomposition); composed into `Memory` in facade.py.
"""

from __future__ import annotations

from typing import Any

try:
    from consciousness_contracts.uri import is_memo_uri, parse_uri

    _HAS_URI_HELPERS = True
except ImportError:
    _HAS_URI_HELPERS = False

from memo.memory._base import _MemoryBase
from memo.memory.record import (
    MEMO_BACKEND_NAME,
    NATIVE_BACKEND_PROTOCOL_VERSION,
    SYNAPSE_BACKEND_NATIVE_SCHEMA,
    AmbiguousIdError,
)
from memo.util import stable_hash as _stable_content_hash
from memo.util import utc_now_iso as _utc_now_iso


class _ReplayOpsMixin(_MemoryBase):
    def backend_native_replay_resolve(
        self,
        uri: str,
        *,
        trace_id: str = "",
        backend_version: str = "",
    ) -> dict[str, Any]:
        """Resolve Synapse backend_native.v1 evidence without mutating Memo."""

        def payload(
            status: str,
            detail: str,
            *,
            content_hash: str = "",
            target: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            out: dict[str, Any] = {
                "schema": SYNAPSE_BACKEND_NATIVE_SCHEMA,
                "protocol_version": NATIVE_BACKEND_PROTOCOL_VERSION,
                "backend": MEMO_BACKEND_NAME,
                "uri": uri,
                "status": status,
                "detail": detail,
                "content_hash": content_hash,
                "observed_at": _utc_now_iso(),
                "backend_version": backend_version,
                "trace_id": trace_id,
                "resolution_mode": "backend_native",
            }
            if target is not None:
                out["target"] = target
            return out

        # Use shared URI helpers if available
        if _HAS_URI_HELPERS:
            if not is_memo_uri(uri):
                return payload(
                    "unsupported",
                    "Memo backend-native only replays memo://memoria/<id>, "
                    "memo://repo/<id|name|url>, and memo://repo-index/<name>/<commit> evidence.",
                )
            parts = parse_uri(uri)
            if parts is None:
                return payload("missing", "Invalid URI format.")

            resource_type = parts.resource_type
            resource_id = parts.resource_id or ""
            subpath = parts.subpath or ""

            if resource_type == "memoria":
                if not resource_id:
                    return payload("missing", "memo://memoria URI did not include an id.")
                try:
                    rec = self.get(resource_id)
                except AmbiguousIdError as exc:
                    return payload(
                        "error",
                        f"ambiguous memory id prefix {exc.prefix!r}: {len(exc.matches)} matches",
                    )
                if rec is None:
                    return payload("missing", "Memo memory was not found.")
                return payload(
                    "found",
                    f"resolved memory: {rec.id}",
                    content_hash=_stable_content_hash(rec.to_dict()),
                    target={"kind": "memoria", "id": rec.id, "path": rec.path},
                )

            elif resource_type == "repo-index":
                # parse_uri splits memo://repo-index/<name>/<commit> as:
                # resource_id=<name>, subpath=<commit>
                repo_name = resource_id or (
                    subpath.split("/", 1)[0] if subpath and "/" in subpath else ""
                )
                commit_prefix = (
                    subpath
                    if resource_id
                    else (subpath.split("/", 1)[1] if subpath and "/" in subpath else "")
                )
                if not repo_name:
                    return payload(
                        "missing", "memo://repo-index URI must include <repo-name>/<commit-prefix>."
                    )
                source = self.store.get_repo_source(repo_name)
                if source is None:
                    return payload("missing", "Memo repo source was not found.")
                commit = str(source.get("commit_sha") or "")
                if (
                    commit_prefix
                    and commit_prefix != "unknown"
                    and not commit.startswith(commit_prefix)
                ):
                    return payload(
                        "missing",
                        "Memo repo source exists but commit did not match the receipt URI.",
                        target={
                            "kind": "repo_index",
                            "repo_id": source.get("id") or "",
                            "name": source.get("name") or repo_name,
                            "commit_sha": commit,
                        },
                    )
                resolved = self._repo_replay_payload(source)
                return payload(
                    "found",
                    f"resolved repo index: {source.get('name')}@{commit[:12]}",
                    content_hash=_stable_content_hash(resolved),
                    target=resolved,
                )

            elif resource_type == "repo":
                if not resource_id:
                    return payload("missing", "memo://repo URI did not include a repo id/name/url.")
                source = self.store.get_repo_source(resource_id)
                if source is None:
                    return payload("missing", "Memo repo source was not found.")
                resolved = self._repo_replay_payload(source)
                return payload(
                    "found",
                    f"resolved repo: {source.get('name')}",
                    content_hash=_stable_content_hash(resolved),
                    target=resolved,
                )

            else:
                return payload(
                    "unsupported",
                    f"Unsupported memo:// resource type: {resource_type}",
                )
        else:
            # Fallback to manual parsing
            memoria_prefix = "memo://memoria/"
            repo_index_prefix = "memo://repo-index/"
            repo_prefix = "memo://repo/"

            if uri.startswith(memoria_prefix):
                memoria_id = uri[len(memoria_prefix) :].strip()
                if not memoria_id:
                    return payload("missing", "memo://memoria URI did not include an id.")
                try:
                    rec = self.get(memoria_id)
                except AmbiguousIdError as exc:
                    return payload(
                        "error",
                        f"ambiguous memory id prefix {exc.prefix!r}: {len(exc.matches)} matches",
                    )
                if rec is None:
                    return payload("missing", "Memo memory was not found.")
                return payload(
                    "found",
                    f"resolved memory: {rec.id}",
                    content_hash=_stable_content_hash(rec.to_dict()),
                    target={"kind": "memoria", "id": rec.id, "path": rec.path},
                )

            if uri.startswith(repo_index_prefix):
                rest = uri[len(repo_index_prefix) :].strip("/")
                if not rest or "/" not in rest:
                    return payload(
                        "missing",
                        "memo://repo-index URI must include <repo-name>/<commit-prefix>.",
                    )
                repo_name, commit_prefix = rest.split("/", 1)
                source = self.store.get_repo_source(repo_name)
                if source is None:
                    return payload("missing", "Memo repo source was not found.")
                commit = str(source.get("commit_sha") or "")
                if (
                    commit_prefix
                    and commit_prefix != "unknown"
                    and not commit.startswith(commit_prefix)
                ):
                    return payload(
                        "missing",
                        "Memo repo source exists but commit did not match the receipt URI.",
                        target={
                            "kind": "repo_index",
                            "repo_id": source.get("id") or "",
                            "name": source.get("name") or repo_name,
                            "commit_sha": commit,
                        },
                    )
                resolved = self._repo_replay_payload(source)
                return payload(
                    "found",
                    f"resolved repo index: {source.get('name')}@{commit[:12]}",
                    content_hash=_stable_content_hash(resolved),
                    target=resolved,
                )

            if uri.startswith(repo_prefix):
                repo_key = uri[len(repo_prefix) :].strip()
                if not repo_key:
                    return payload("missing", "memo://repo URI did not include a repo id/name/url.")
                source = self.store.get_repo_source(repo_key)
                if source is None:
                    return payload("missing", "Memo repo source was not found.")
                resolved = self._repo_replay_payload(source)
                return payload(
                    "found",
                    f"resolved repo: {source.get('name')}",
                    content_hash=_stable_content_hash(resolved),
                    target=resolved,
                )

            return payload(
                "unsupported",
                "Memo backend-native only replays memo://memoria/<id>, "
                "memo://repo/<id|name|url>, and memo://repo-index/<name>/<commit> evidence.",
            )

    def _repo_replay_payload(self, source: dict[str, Any]) -> dict[str, Any]:
        repo_id = str(source.get("id") or "")
        counts = (
            self.store.repo_counts(repo_id)
            if repo_id
            else {
                "files": 0,
                "lines": 0,
                "chunks": 0,
                "embedded_chunks": 0,
            }
        )
        pending_chunks = counts["chunks"] - counts["embedded_chunks"]
        return {
            "kind": "repo",
            "id": repo_id,
            "name": source.get("name") or "",
            "url": source.get("url") or "",
            "ref": source.get("ref") or "",
            "commit_sha": source.get("commit_sha") or "",
            "indexed_at": source.get("indexed_at") or "",
            "status": source.get("status") or "",
            "semantic_status": (
                "semantic_ready"
                if counts["chunks"] and pending_chunks == 0
                else "semantic_pending"
                if pending_chunks > 0
                else str(source.get("status") or "")
            ),
            "counts": {
                **counts,
                "pending_chunks": pending_chunks,
            },
        }
