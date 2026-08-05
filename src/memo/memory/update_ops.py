"""Update operations for `Memory` — patch title/type/tags/body on existing records.

Extracted from `_WriteOpsMixin` to keep each file under 800 lines.
`Memory` facade inherits this mixin alongside the others.
"""

from __future__ import annotations

import builtins
import logging
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

import frontmatter

from memo.contracts import (
    LEGACY_PROVENANCE_KEYS,
    PROVENANCE_KEYS,
    ActorIdentity,
    TrustTier,
    normalize_provenance,
)
from memo.errors import IdentityConflictError, StorageError
from memo.identity import (
    canonical_topic_key,
    namespace_for_index,
    normalized_content_hash,
    normalized_title,
)
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _VALID_TYPES,
    MemoryRecord,
    _extract_provenance,
    _normalise_tags,
    _now_iso,
    is_derived_chunk_id,
)
from memo.tiers import VerificationState
from memo.util import sha256_short as _sha256_short
from memo.write_policy import actor_for_existing_record

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PreparedUpdateEmbedding:
    """Model result computed before entering the authority write lock."""

    text: str
    vector: list[float]


class _RetryPreparedUpdate(RuntimeError):
    """The optimistic update view changed before the commit lock was acquired."""


@dataclass(frozen=True)
class _UpdateIdentity:
    """Canonical identity fields validated before an update is persisted."""

    namespace: str
    normalized_title: str
    normalized_content_hash: str


def _resolve_updated_body(
    old_body: str,
    id_: str,
    *,
    content: str | None,
    replace: tuple[str, str] | None,
    append: str | None,
) -> str:
    if replace is not None:
        old, new = replace
        if not old:
            raise ValueError("replace: old string must be non-empty")
        occurrences = old_body.count(old)
        if occurrences == 0:
            raise ValueError(f"replace: old string not found in body of {id_[:8]}")
        if occurrences > 1:
            raise ValueError(
                f"replace: old string occurs {occurrences} times in {id_[:8]}; must be unique"
            )
        return old_body.replace(old, new, 1)
    if append is not None:
        return (
            old_body.rstrip("\n") + "\n\n" + append.strip() if old_body.strip() else append.strip()
        )
    return content if content is not None else old_body


def _normalized_update_extra(
    extra: dict[str, Any] | None,
    existing_extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Collapse legacy provenance keys into the native nested contract."""
    normalized = dict(extra) if extra is not None else dict(existing_extra or {})
    provenance = normalize_provenance(normalized)
    for key in PROVENANCE_KEYS | LEGACY_PROVENANCE_KEYS:
        normalized.pop(key, None)
    if provenance:
        normalized["provenance"] = provenance
    return normalized


def _policy_actor_for_update(
    actor: ActorIdentity | None,
    *,
    semantic_change: bool,
    existing_extra: dict[str, Any] | None,
    new_extra: dict[str, Any],
) -> ActorIdentity:
    """Select update authority and enforce the default agent trust ceiling."""
    if actor is not None:
        return actor
    if not semantic_change:
        return actor_for_existing_record(existing_extra)

    if new_extra.get("trust_tier") in {
        TrustTier.HUMAN.value,
        TrustTier.TOOL_OBSERVED.value,
        TrustTier.AGENT_VERIFIED.value,
    }:
        new_extra["trust_tier"] = TrustTier.AGENT_INFERRED.value
    return ActorIdentity(actor_id="memo-update", actor_kind="agent")


class _UpdateOpsMixin(_MemoryBase):
    # -- update -------------------------------------------------------------

    def update(
        self,
        id_: str,
        *,
        title: str | None = None,
        type_: str | None = None,
        tags: builtins.list[str] | None = None,
        content: str | None = None,
        replace: tuple[str, str] | None = None,
        append: str | None = None,
        extra: dict[str, Any] | None = None,
        actor: ActorIdentity | None = None,
    ) -> MemoryRecord | None:
        """Patch a record with model work outside the serialized commit section."""
        if is_derived_chunk_id(id_):
            raise ValueError("derived chunk records are read-only; update the parent memory")

        # Optimistically derive and embed the prospective record without the
        # cross-process lock. The commit path rebuilds the prospective view
        # from current state and accepts the vector only when its exact input
        # still matches. A concurrent edit therefore causes a bounded retry,
        # never a stale vector or a model call inside the authority lock.
        for _attempt in range(4):
            # Captured before the commit: the assertion edge a renamed fact
            # leaves behind still names the old title, and only the pre-update
            # record knows it.
            prepared_title = self._current_title(id_) if title is not None else None
            prepared = self._prepare_update_embedding(
                id_,
                title=title,
                type_=type_,
                tags=tags,
                content=content,
                replace=replace,
                append=append,
                extra=extra,
            )
            try:
                with self._data_dir_write_lock():
                    updated = self._update_locked(
                        id_,
                        title=title,
                        type_=type_,
                        tags=tags,
                        content=content,
                        replace=replace,
                        append=append,
                        extra=extra,
                        actor=actor,
                        _prepared_embedding=prepared,
                    )
                if updated is not None:
                    self._refresh_assertion_edge(updated, previous_title=prepared_title)
                if updated is not None and prepared is not None:
                    # Chunk vectors are derived, potentially expensive model
                    # work. Keep them outside the authority lock; failures are
                    # already best-effort and the next reindex heals them.
                    self.maybe_emit_chunks(
                        parent_id=updated.id,
                        parent_rel=updated.path,
                        title=updated.title,
                        body=updated.body,
                        tags=updated.tags,
                        created=updated.created,
                        updated=updated.updated,
                        valid_at=updated.valid_at,
                        invalid_at=updated.invalid_at,
                    )
                return updated
            except _RetryPreparedUpdate:
                continue
        raise StorageError("update could not stabilize after concurrent edits")

    def _current_title(self, id_: str) -> str | None:
        with suppress(Exception):
            record = self.get(id_)
            if record is not None:
                return str(record.title)
        return None

    def _refresh_assertion_edge(self, updated: MemoryRecord, *, previous_title: str | None) -> None:
        """Close a renamed fact's stale ``memory asserts <title>`` edge.

        A `fact` with no declared edges gets that coarse assertion at save
        time. Only the save paths ever wrote it, so a rename left the graph and
        the briefing asserting the old title forever. The old edge is
        invalidated rather than deleted — the edges are bi-temporal, so "this
        memory asserted X until now" stays queryable — and the current title is
        upserted as the open assertion. Best-effort: a derived-graph failure
        must never fail a committed update.
        """
        if previous_title is None or previous_title == updated.title:
            return
        if updated.type != "fact":
            return
        from memo.fact_extraction import FACT_ASSERTION_PREDICATE, upsert_declared_fact_edges

        with suppress(Exception):
            for edge in self.fact_edges.query(source_record_id=updated.id):
                if (
                    edge.get("predicate") == FACT_ASSERTION_PREDICATE
                    and edge.get("object") == previous_title
                    and not edge.get("invalid_at")
                ):
                    self.fact_edges.invalidate(str(edge["id"]))
            upsert_declared_fact_edges(
                self.fact_edges,
                record_id=updated.id,
                title=updated.title,
                type_=updated.type,
                created=updated.created,
                updated=updated.updated,
                extra=updated.extra,
            )

    def _prepare_update_embedding(
        self,
        id_: str,
        *,
        title: str | None = None,
        type_: str | None = None,
        tags: builtins.list[str] | None = None,
        content: str | None = None,
        replace: tuple[str, str] | None = None,
        append: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> _PreparedUpdateEmbedding | None:
        """Compute a prospective update vector without acquiring the write lock.

        The commit path repeats validation, sanitation, and identity checks.
        This helper is only an optimization/scheduling step; it is never the
        authority for deciding what gets persisted.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        row = self.store.get(resolved)
        if row is None:
            return None
        topic_key, normalized_hash = self.store.get_dedup_keys(resolved)
        if type_ is not None and type_ not in _VALID_TYPES:
            raise ValueError(f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}")
        if sum(value is not None for value in (content, replace, append)) > 1:
            raise ValueError("update: pass at most one of content=, replace=, append=")

        new_title = (title.strip() if title else row["title"]) or row["title"]
        new_tags = _normalise_tags(tags) if tags is not None else row["tags"]
        new_extra = extra if extra is not None else dict(row.get("extra") or {})
        old_body = self._read_body(row["path"])
        new_body = _resolve_updated_body(
            old_body,
            resolved,
            content=content,
            replace=replace,
            append=append,
        )[: self.cfg.max_content_chars]

        from memo.flags import flag_bool
        from memo.redact import sanitize_memory_input

        sanitized = sanitize_memory_input(
            content=new_body,
            title=new_title,
            tags=new_tags,
            topic_key=topic_key,
            normalized_hash=normalized_hash,
            extra=new_extra,
            entropy=flag_bool("MEMO_REDACT_ENTROPY"),
        )
        new_body = sanitized.content
        new_title = sanitized.title or "untitled"
        new_body_hash = _sha256_short(new_body)
        embed_pending = bool(sanitized.extra.get("_memo_embed_pending"))
        embedding_required = new_body_hash != row["body_hash"] or (
            new_title != row["title"] and not embed_pending
        )
        if not embedding_required:
            return None

        text = self._compose_for_embed(new_title, new_body)
        vector = self._embed_cached(text, ctx=f"update id={resolved[:8]}")
        return _PreparedUpdateEmbedding(text=text, vector=vector)

    def _update_locked(
        self,
        id_: str,
        *,
        title: str | None = None,
        type_: str | None = None,
        tags: builtins.list[str] | None = None,
        content: str | None = None,
        replace: tuple[str, str] | None = None,
        append: str | None = None,
        extra: dict[str, Any] | None = None,
        actor: ActorIdentity | None = None,
        _prepared_embedding: _PreparedUpdateEmbedding | None = None,
        _defer_embed: bool = False,
    ) -> MemoryRecord | None:
        """Patch one or more fields on an existing record.

        Only the kwargs you pass are touched; everything else stays as-is.
        Re-embed only if `content` changed (body_hash check). The file
        path stays stable — renaming the slug after the fact would break
        wikilinks the user may have created in their vault.

        `replace=(old, new)` is an exact-string surgical edit — old must occur
        exactly once (ValueError otherwise) so unchanged content stays
        byte-identical; `append` adds a paragraph. Both route through the same
        versioned path as `content`.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        id_ = resolved
        r = self.store.get(id_)
        if r is None:
            return None
        topic_key, normalized_hash = self.store.get_dedup_keys(id_)
        if type_ is not None and type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )
        if sum(x is not None for x in (content, replace, append)) > 1:
            raise ValueError("update: pass at most one of content=, replace=, append=")

        new_title = (title.strip() if title else r["title"]) or r["title"]
        new_type = type_ or r["type"]
        new_tags = _normalise_tags(tags) if tags is not None else r["tags"]
        new_extra = _normalized_update_extra(extra, r.get("extra"))
        now_iso = _now_iso()

        # Body resolution: provided > on-disk > empty.
        old_body = self._read_body(r["path"])
        new_body = _resolve_updated_body(
            old_body,
            id_,
            content=content,
            replace=replace,
            append=append,
        )
        new_body = new_body[: self.cfg.max_content_chars]

        # Sanitize the COMPLETE prospective record, not just explicitly passed
        # fields. That also scrubs a legacy secret when a metadata-only edit
        # rewrites old Markdown. Pattern/private protection is always on;
        # entropy remains the opt-in tier.
        from memo.flags import flag_bool
        from memo.redact import sanitize_memory_input, sanitize_persisted_text

        sanitized = sanitize_memory_input(
            content=new_body,
            title=new_title,
            tags=new_tags,
            topic_key=topic_key,
            normalized_hash=normalized_hash,
            extra=new_extra,
            entropy=flag_bool("MEMO_REDACT_ENTROPY"),
        )
        new_body = sanitized.content
        new_title = sanitized.title or "untitled"
        new_tags = _normalise_tags(sanitized.tags)
        new_extra = sanitized.extra
        topic_key = sanitized.topic_key
        normalized_hash = sanitized.normalized_hash
        semantic_change = bool(
            new_body != old_body
            or new_title != r["title"]
            or new_type != r["type"]
            or new_tags != r["tags"]
        )
        policy_actor = _policy_actor_for_update(
            actor,
            semantic_change=semantic_change,
            existing_extra=r.get("extra"),
            new_extra=new_extra,
        )
        decision = self.write_policy.preflight(
            title=new_title,
            content=new_body,
            tags=new_tags,
            extra=new_extra,
            actor=policy_actor,
        )
        self.write_policy.enforce(decision)
        new_extra["write_policy"] = decision.to_dict()
        new_extra["visibility"] = decision.visibility.value
        new_extra["trust_tier"] = decision.trust_tier.value
        new_extra["owner_principal"] = self.cfg.device_id
        new_body_hash = _sha256_short(new_body)
        identity = self._validate_update_identity(
            id_=id_,
            path=str(r["path"]),
            title=new_title,
            type_=new_type,
            tags=new_tags,
            body=new_body,
            topic_key=topic_key,
        )
        new_namespace = identity.namespace
        new_normalized_title = identity.normalized_title
        new_normalized_content_hash = identity.normalized_content_hash

        body_changed = new_body_hash != r["body_hash"]
        title_changed = new_title != r["title"]
        embed_pending = bool(new_extra.get("_memo_embed_pending"))
        # A record saved with ``defer_embed`` has no vector yet.  Metadata-only
        # corrections must remain available in no-model environments instead
        # of trying to load an embedder merely to update the title.  The
        # pending marker keeps the eventual reindex responsible for composing
        # the vector from the corrected title and body.
        embedding_required = body_changed or (title_changed and not embed_pending)
        vector_write_required = embedding_required and not _defer_embed
        if embedding_required and _defer_embed:
            new_extra["_memo_embed_pending"] = True

        semantic_edit = bool(
            body_changed or title_changed or new_type != r["type"] or new_tags != r["tags"]
        )
        prior_snapshot = (
            sanitize_persisted_text(str(r["title"]), entropy=flag_bool("MEMO_REDACT_ENTROPY")).text
            or "untitled",
            str(r["type"]),
            [
                sanitize_persisted_text(str(tag), entropy=flag_bool("MEMO_REDACT_ENTROPY")).text
                for tag in r["tags"]
            ],
            sanitize_persisted_text(old_body, entropy=flag_bool("MEMO_REDACT_ENTROPY")).text,
        )

        source_path = self._resolve_existing(r["path"])
        source_markdown = source_path.read_bytes() if source_path.exists() else None
        target_path = self.cfg.memory_dir / r["path"]
        previous_target_markdown = (
            source_markdown
            if source_path == target_path
            else target_path.read_bytes()
            if target_path.exists()
            else None
        )
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Re-embed when the body OR title changed — both are part of the
        # embed input now (see `_compose_for_embed`). Pure retag/type changes
        # still skip it. The vector MUST have been computed before entering
        # the authority lock. Revalidate its exact input here; a mismatch means
        # the optimistic view raced with another writer and the caller retries.
        if vector_write_required:
            embed_text = self._compose_for_embed(new_title, new_body)
            if _prepared_embedding is None or _prepared_embedding.text != embed_text:
                raise _RetryPreparedUpdate
            embedding = _prepared_embedding.vector
            # A successful embedding discharges a prior defer/failure marker.
            # Keeping it would make later metadata edits believe the vector is
            # still absent and could suppress a required title re-embed.
            new_extra.pop("_memo_embed_pending", None)

        post = frontmatter.Post(
            new_body,
            id=id_,
            title=new_title,
            type=new_type,
            tags=new_tags,
            created=r["created"],
            updated=now_iso,
        )
        post["extra"] = new_extra or {}
        post["verification_state"] = r.get("verification_state", "unverified")
        if r.get("verified_at") is not None:
            post["verified_at"] = r["verified_at"]
        if r.get("review_after") is not None:
            post["review_after"] = r["review_after"]
        if topic_key is not None:
            post["topic_key"] = topic_key
        if normalized_hash is not None:
            post["normalized_hash"] = normalized_hash
        if r.get("valid_at") is not None:
            post["valid_at"] = r["valid_at"]
        if r.get("invalid_at") is not None:
            post["invalid_at"] = r["invalid_at"]
        self._atomic_write_text(str(r["path"]), frontmatter.dumps(post))

        try:
            if vector_write_required:
                self.store.upsert(
                    id_=id_,
                    path=r["path"],
                    title=new_title,
                    type_=new_type,
                    tags=new_tags,
                    created=r["created"],
                    updated=now_iso,
                    body_hash=new_body_hash,
                    embedding=embedding,
                    extra=new_extra,
                    body_text=new_body,
                    topic_key=topic_key,
                    normalized_hash=normalized_hash,
                    namespace=new_namespace,
                    normalized_title=new_normalized_title,
                    normalized_content_hash=new_normalized_content_hash,
                    valid_at=r.get("valid_at"),
                    invalid_at=r.get("invalid_at"),
                )
            elif embedding_required:
                self.store.upsert_text_only(
                    id_=id_,
                    path=r["path"],
                    title=new_title,
                    type_=new_type,
                    tags=new_tags,
                    created=r["created"],
                    updated=now_iso,
                    body_hash=new_body_hash,
                    extra=new_extra,
                    body_text=new_body,
                    topic_key=topic_key,
                    normalized_hash=normalized_hash,
                    namespace=new_namespace,
                    normalized_title=new_normalized_title,
                    normalized_content_hash=new_normalized_content_hash,
                    valid_at=r.get("valid_at"),
                    invalid_at=r.get("invalid_at"),
                )
            else:
                self.store.update_meta(
                    id_=id_,
                    title=new_title,
                    type_=new_type,
                    tags=new_tags,
                    updated=now_iso,
                    extra=new_extra,
                    namespace=new_namespace,
                    normalized_title=new_normalized_title,
                    normalized_content_hash=new_normalized_content_hash,
                )
        except Exception:
            if previous_target_markdown is None:
                target_path.unlink(missing_ok=True)
            else:
                self._atomic_write_text(
                    str(r["path"]),
                    previous_target_markdown.decode("utf-8"),
                )
            raise

        # Version history is derivative state. Record the sanitized prior view
        # only after Markdown and SQLite both committed, so rejected/rolled-back
        # updates leave no phantom version behind.
        if semantic_edit:
            with suppress(Exception):
                self.versioning.track_update(
                    id_,
                    prior_snapshot[0],
                    prior_snapshot[1],
                    prior_snapshot[2],
                    prior_snapshot[3],
                    reason="pre-update snapshot",
                )

        # Legacy vault-layout migration: `_resolve_existing` may have read the
        # body from `vault_path / rel` (pre-migrate row) while the rewrite
        # above landed at `memory_dir / rel`. Remove the stale vault copy so
        # exactly one canonical .md exists — otherwise a later reindex can
        # resurrect the pre-update body from the leftover duplicate.
        # Best-effort: the new-layout file + index row are already canonical,
        # so a failed unlink degrades to the old (duplicate) state, no worse.
        if source_path != target_path and source_markdown is not None:
            try:
                source_path.unlink(missing_ok=True)
            except OSError as exc:
                _log.warning(
                    "update(%s): stale legacy copy %s left in place — %s",
                    id_[:8],
                    source_path,
                    exc,
                )

        # Crossref edges (flag-gated): body edits can add/remove typed links.
        if body_changed:
            from memo.flags import flag_bool as _flag_bool

            if _flag_bool("MEMO_CROSSREF_INDEX"):
                try:
                    self.crossref.index_source(id_, new_body)
                except Exception as exc:
                    _log.debug("update(%s): crossref index skipped — %s", id_[:8], exc)

        # Audit log: build a delta of just the fields that changed.
        delta: dict[str, tuple[Any, Any]] = {}
        if title_changed:
            delta["title"] = (r["title"], new_title)
        if new_type != r["type"]:
            delta["type"] = (r["type"], new_type)
        if new_tags != r["tags"]:
            delta["tags"] = (r["tags"], new_tags)
        if body_changed:
            delta["body_hash"] = (r["body_hash"], new_body_hash)
        # Record the old `updated` so time-machine can restore the exact
        # timestamp when replaying history in reverse (reconstruct() uses
        # old_val). Only add when something else changed — a pure
        # timestamp-bump carries no semantic change worth logging.
        if delta:
            delta["updated"] = (r.get("updated"), now_iso)
        # Track provenance churn so a new trace/actor on the same memory shows
        # up in history.
        old_prov = _extract_provenance(r.get("extra") or {})
        new_prov = _extract_provenance(new_extra)
        if old_prov != new_prov:
            delta["_provenance"] = (old_prov, new_prov)
        if delta:
            with suppress(Exception):
                self.history.log_update(
                    ts=now_iso,
                    record_id=id_,
                    title=new_title,
                    type_=new_type,
                    delta=delta,
                )

        updated_rec = MemoryRecord(
            id=id_,
            path=r["path"],
            title=new_title,
            type=new_type,
            tags=new_tags,
            created=r["created"],
            updated=now_iso,
            body=new_body,
            extra=new_extra,
            verification_state=VerificationState(
                str(r.get("verification_state", VerificationState.UNVERIFIED.value))
            ),
            verified_at=r.get("verified_at"),
            review_after=r.get("review_after"),
            valid_at=r.get("valid_at"),
            invalid_at=r.get("invalid_at"),
        )
        if delta:
            with suppress(Exception):
                self.operational.receipt(
                    "update",
                    subject_uri=f"memo://memoria/{id_}",
                    trace_id=str(new_prov.get("trace_id") or ""),
                    actor_id=str(new_prov.get("actor_id") or "memo"),
                    metadata={
                        "id": id_,
                        "type": new_type,
                        "title": new_title,
                        "delta_keys": sorted(delta),
                    },
                )
        self._write_gen += 1
        return updated_rec

    def _validate_update_identity(
        self,
        *,
        id_: str,
        path: str,
        title: str,
        type_: str,
        tags: builtins.list[str],
        body: str,
        topic_key: str | None,
    ) -> _UpdateIdentity:
        """Reject topic/exact collisions and return canonical index fields."""
        namespace = namespace_for_index(tags, path=path)
        if namespace is None:
            raise IdentityConflictError(
                kind="ambiguous_namespace",
                incoming={"record_id": id_, "namespace": None},
            )

        canonical_topic = canonical_topic_key(topic_key)
        if canonical_topic is not None:
            topic_conflicts = [
                row
                for row in self.store.find_active_by_topic_identity(namespace, canonical_topic)
                if str(row.get("id")) != id_
            ]
            if topic_conflicts:
                raise IdentityConflictError(
                    kind="update_topic_identity_conflict",
                    incoming={
                        "record_id": id_,
                        "namespace": namespace,
                        "topic_key": canonical_topic,
                    },
                    conflicts=topic_conflicts,
                )

        canonical_title = normalized_title(title)
        content_hash = normalized_content_hash(body)
        exact_conflicts = [
            row
            for row in self.store.find_active_by_exact_identity(
                namespace,
                type_,
                canonical_title,
                content_hash,
            )
            if str(row.get("id")) != id_
        ]
        if exact_conflicts:
            raise IdentityConflictError(
                kind="update_exact_identity_conflict",
                incoming={
                    "record_id": id_,
                    "namespace": namespace,
                    "type": type_,
                    "normalized_title": canonical_title,
                },
                conflicts=exact_conflicts,
            )
        return _UpdateIdentity(
            namespace=namespace,
            normalized_title=canonical_title,
            normalized_content_hash=content_hash,
        )

    # -- last saved ----------------------------------------------------------

    def last_saved_id(self, *, limit: int = 20) -> str | None:
        """Id of the most recent memory saved on this device that still exists.

        Scans the audit log's `save` events (newest first, this device only —
        synced events from other machines don't count as "just saved here")
        and returns the first record_id not deleted since. Backs the
        "rename what I just saved" flows (`memo rename`, MCP `memo_rename`).
        """
        events = self.history.list_recent(
            limit=limit,
            op="save",
            device_id=self.history.device_id,
        )
        for ev in events:
            rid = ev.get("record_id")
            if rid and self.get(rid) is not None:
                return str(rid)
        return None
