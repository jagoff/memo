"""Memo-native replay resolution for `Memory`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memo.memory._base import _MemoryBase
from memo.memory.record import (
    MEMO_BACKEND_NAME,
    MEMO_BACKEND_NATIVE_SCHEMA,
    NATIVE_BACKEND_PROTOCOL_VERSION,
    AmbiguousIdError,
)
from memo.util import stable_hash as _stable_content_hash
from memo.util import utc_now_iso as _utc_now_iso


@dataclass(frozen=True)
class _ReplayUri:
    resource_type: str
    resource_id: str
    subpath: str


def _parse_replay_uri(uri: str) -> _ReplayUri | None:
    if not isinstance(uri, str) or not uri.startswith("memo://"):
        return None
    rest = uri[len("memo://") :].strip("/")
    if not rest:
        return _ReplayUri("", "", "")
    resource_type, separator, path = rest.partition("/")
    if not separator:
        return _ReplayUri(resource_type, "", "")
    resource_id, _, subpath = path.partition("/")
    return _ReplayUri(resource_type, resource_id, subpath)


class _ReplayOpsMixin(_MemoryBase):
    def backend_native_replay_resolve(
        self,
        uri: str,
        *,
        trace_id: str = "",
        backend_version: str = "",
    ) -> dict[str, Any]:
        """Resolve a Memo evidence URI without mutating storage."""

        def payload(
            status: str,
            detail: str,
            *,
            content_hash: str = "",
            target: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            out: dict[str, Any] = {
                "schema": MEMO_BACKEND_NATIVE_SCHEMA,
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

        parts = _parse_replay_uri(uri)
        if parts is None:
            return payload(
                "unsupported",
                "Memo backend-native only replays memo://memoria/<id>, "
                "memo://repo/<id|name|url>, and memo://repo-index/<name>/<commit> evidence.",
            )

        resource_type = parts.resource_type
        resource_id = parts.resource_id
        subpath = parts.subpath

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

        if resource_type == "repo-index":
            if not resource_id or not subpath:
                return payload(
                    "missing", "memo://repo-index URI must include <repo-name>/<commit-prefix>."
                )
            source = self.store.get_repo_source(resource_id)
            if source is None:
                return payload("missing", "Memo repo source was not found.")
            commit = str(source.get("commit_sha") or "")
            if subpath != "unknown" and not commit.startswith(subpath):
                return payload(
                    "missing",
                    "Memo repo source exists but commit did not match the replay URI.",
                    target={
                        "kind": "repo_index",
                        "repo_id": source.get("id") or "",
                        "name": source.get("name") or resource_id,
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

        if resource_type == "repo":
            repo_key = resource_id + (f"/{subpath}" if subpath else "")
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
            f"Unsupported memo:// resource type: {resource_type or '<empty>'}",
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
        # The two counts can be observed between index updates or while an old
        # derived row is being reconciled. Replay payloads must never expose an
        # impossible negative pending count to downstream contract consumers.
        pending_chunks = max(0, counts["chunks"] - counts["embedded_chunks"])
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
