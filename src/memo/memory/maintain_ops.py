"""Maintenance + provenance operations for `Memory`.

`_MaintainOpsMixin` holds the corpus-wide maintenance surface (reindex,
lint, gc, entity extraction, consolidation), provenance lookups, the synapse
freeze-write guard, and the backend-native replay resolver, moved verbatim
from the former `memory.py` god-file.
"""

from __future__ import annotations

import builtins
import json
from typing import Any

import frontmatter

try:
    from consciousness_contracts.uri import is_memo_uri, parse_uri
    _HAS_URI_HELPERS = True
except ImportError:
    _HAS_URI_HELPERS = False

from memo.embedder import assert_valid_embedding
from memo.lifecycle import FORGET_AFTER_KEY, FORGET_REASON_KEY
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _CONSOLIDATE_SYSTEM_PROMPT,
    _EXTRACT_ENTITIES_SYSTEM_PROMPT,
    _SYNTHESIS_SYSTEM_PROMPT,
    _VALID_TYPES,
    MEMO_BACKEND_NAME,
    NATIVE_BACKEND_PROTOCOL_VERSION,
    SYNAPSE_BACKEND_NATIVE_SCHEMA,
    AmbiguousIdError,
    WriteRefused,
    _build_freeze_query,
    _derive_title,
    _extract_provenance,
    _log,
    _normalise_tags,
    _now_iso,
    chat_with_timeout,
    strip_llm_output,
)
from memo.util import sha256_short as _sha256_short
from memo.util import stable_hash as _stable_content_hash
from memo.util import utc_now_iso as _utc_now_iso


class _MaintainOpsMixin(_MemoryBase):
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
                        f"ambiguous memoria id prefix {exc.prefix!r}: {len(exc.matches)} matches",
                    )
                if rec is None:
                    return payload("missing", "Memo memoria was not found.")
                return payload(
                    "found",
                    f"resolved memoria: {rec.id}",
                    content_hash=_stable_content_hash(rec.to_dict()),
                    target={"kind": "memoria", "id": rec.id, "path": rec.path},
                )
            
            elif resource_type == "repo-index":
                # parse_uri splits memo://repo-index/<name>/<commit> as:
                # resource_id=<name>, subpath=<commit>
                repo_name = resource_id or (subpath.split("/", 1)[0] if subpath and "/" in subpath else "")
                commit_prefix = subpath if resource_id else (subpath.split("/", 1)[1] if subpath and "/" in subpath else "")
                if not repo_name:
                    return payload("missing", "memo://repo-index URI must include <repo-name>/<commit-prefix>.")
                source = self.store.get_repo_source(repo_name)
                if source is None:
                    return payload("missing", "Memo repo source was not found.")
                commit = str(source.get("commit_sha") or "")
                if commit_prefix and commit_prefix != "unknown" and not commit.startswith(commit_prefix):
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
                memoria_id = uri[len(memoria_prefix):].strip()
                if not memoria_id:
                    return payload("missing", "memo://memoria URI did not include an id.")
                try:
                    rec = self.get(memoria_id)
                except AmbiguousIdError as exc:
                    return payload(
                        "error",
                        f"ambiguous memoria id prefix {exc.prefix!r}: {len(exc.matches)} matches",
                    )
                if rec is None:
                    return payload("missing", "Memo memoria was not found.")
                return payload(
                    "found",
                    f"resolved memoria: {rec.id}",
                    content_hash=_stable_content_hash(rec.to_dict()),
                    target={"kind": "memoria", "id": rec.id, "path": rec.path},
                )

            if uri.startswith(repo_index_prefix):
                rest = uri[len(repo_index_prefix):].strip("/")
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
                if commit_prefix and commit_prefix != "unknown" and not commit.startswith(commit_prefix):
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
                repo_key = uri[len(repo_prefix):].strip()
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
        counts = self.store.repo_counts(repo_id) if repo_id else {
            "files": 0,
            "lines": 0,
            "chunks": 0,
            "embedded_chunks": 0,
        }
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
                "semantic_ready" if counts["chunks"] and pending_chunks == 0
                else "semantic_pending" if pending_chunks > 0
                else str(source.get("status") or "")
            ),
            "counts": {
                **counts,
                "pending_chunks": pending_chunks,
            },
        }

    # -- synapse freeze-write protocol -------------------------------------

    def _enforce_synapse_freeze(
        self,
        *,
        title: str | None,
        content: str,
        tags: builtins.list[str] | None,
        trace_id: str,
    ) -> None:
        """Query synapse for blocking RealityConflicts; raise on hit.

        Derives a query from the most signal-dense fields available
        (title, first non-empty tags, first content line). Best-effort:
        if synapse is not on PATH, returns without raising — the
        opt-in nature already implies "best information available".
        """
        # Deferred import: keeps memo's hard deps free of synapse.
        from memo import synapse_client

        if not synapse_client.is_available():
            return
        query = _build_freeze_query(title=title, content=content, tags=tags)
        if not query:
            return
        try:
            conflicts = synapse_client.list_conflicts(
                query, trace_id=trace_id,
            )
        except Exception as exc:  # pragma: no cover - subprocess noise
            _log.debug("synapse freeze-check failed: %s", exc)
            return
        blocked, conflict = synapse_client.has_blocking_freeze(conflicts)
        if blocked and conflict is not None:
            raise WriteRefused(conflict)

    # -- provenance ---------------------------------------------------------

    def provenance(self, id_: str) -> dict[str, Any] | None:
        """Return the full provenance trail for a memoria.

        Combines the current state (provenance subset of `meta.extra_json`)
        with the per-op history (each save/update event carrying its own
        provenance snapshot in `delta_json`). Returns `None` if the id is
        unknown.

        Shape:

            {
              "id": "<full id>",
              "current": {synapse_trace_id, synapse_route_reason, ...},
              "events": [
                {"ts", "op", "title", "type", "provenance": {...}},
                ...
              ]
            }
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        rec = self.store.get(resolved)
        if rec is None:
            return None
        current = _extract_provenance(rec.get("extra") or {})
        events: list[dict[str, Any]] = []
        for raw in self.history.list_recent(limit=10_000, record_id=resolved):
            entry: dict[str, Any] = {
                "ts": raw.get("ts"),
                "op": raw.get("op"),
                "title": raw.get("title"),
                "type": raw.get("type"),
            }
            delta = raw.get("delta") or {}
            if isinstance(delta, dict) and "_provenance" in delta:
                prov = delta["_provenance"]
                # save op stores `{...keys...}`; update op stores
                # `[old_dict, new_dict]` (delta-pair convention). Surface
                # the post-state in both cases.
                if isinstance(prov, list) and len(prov) == 2:
                    entry["provenance"] = prov[1] or {}
                elif isinstance(prov, dict):
                    entry["provenance"] = prov
            events.append(entry)
        events.reverse()  # oldest first
        return {"id": resolved, "current": current, "events": events}

    # -- reindex / gc -------------------------------------------------------

    def reindex(self, *, force: bool = False) -> dict[str, int]:
        """Scan the memory dir, re-embed entries whose on-disk body
        diverged from `body_hash`. Picks up edits the user made in
        Obsidian directly. Also indexes any `.md` with a valid `id` in
        frontmatter that the store doesn't know about (e.g. restored
        from a backup or copied from another machine).

        With `force=True`, re-embeds EVERY indexed entry regardless of
        body_hash match. Use after an embedder model swap, after a
        change to `_compose_for_embed`, or to refresh the index after
        a corruption/incident.

        Returns counts: `{"checked", "reindexed", "added", "skipped"}`.
        """
        memory_root = self.cfg.memory_dir
        checked = reindexed = added = skipped = 0
        if not memory_root.is_dir():
            return {"checked": 0, "reindexed": 0, "added": 0, "skipped": 0}

        for md_path in sorted(memory_root.rglob("*.md")):
            checked += 1
            try:
                post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
            except Exception as exc:
                _log.warning("reindex: skipping %s (parse error): %s", md_path.name, exc)
                skipped += 1
                continue
            meta: dict[str, Any] = post.metadata
            md_id = meta.get("id")
            if not md_id or not isinstance(md_id, str):
                skipped += 1
                continue
            body = post.content or ""
            new_hash = _sha256_short(body)
            existing = self.store.get(md_id)
            # Path relative to memory_dir — paths in the store no longer
            # carry the legacy `<vault>/<memory_subdir>/...` prefix.
            rel = str(md_path.relative_to(self.cfg.memory_dir))

            title = (meta.get("title") or _derive_title(body) or "untitled").strip()
            type_ = meta.get("type") or "note"
            if type_ not in _VALID_TYPES:
                _log.warning("reindex: invalid type %r in %s, coercing to 'note'", type_, md_path.name)
                type_ = "note"
            tags = _normalise_tags(list(meta.get("tags") or []))
            created = meta.get("created") or _now_iso()
            updated = meta.get("updated") or created
            extra = meta.get("extra") or {}
            # Obsidian-friendly: accept `forget_after` / `forget_reason` as
            # TOP-LEVEL frontmatter keys (what a user naturally types in their
            # editor), folding them into the extra bag the lifecycle layer
            # reads. The nested `extra:` form still works and takes precedence.
            for _fk in (FORGET_AFTER_KEY, FORGET_REASON_KEY):
                if _fk in meta and _fk not in extra:
                    extra = {**extra, _fk: meta[_fk]}

            if existing is None:
                # Path-collision guard: an .md may have its frontmatter id
                # regenerated (manual edit, restore-from-backup, or a stale
                # row pointing at a file whose id was rewritten) while the
                # vault-relative path stays the same. The store's
                # UNIQUE(meta.path) constraint blocks a plain INSERT, so we
                # drop the orphan row before re-adding under the new id.
                stale = self.store.get_by_path(rel)
                if stale is not None:
                    _log.warning(
                        "reindex: path %r reused with new id (%s → %s); "
                        "replacing stale row",
                        rel, stale["id"][:8], md_id[:8],
                    )
                    self.store.delete(stale["id"])
                emb = self.embedder.embed([self._compose_for_embed(title, body)])[0]
                assert_valid_embedding(emb, self.cfg.embedder_dims, context=f"reindex add {md_id[:8]}")
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=created, updated=updated, body_hash=new_hash,
                    embedding=emb, extra=extra if extra else None,
                    body_text=body,
                )
                added += 1
                continue
            missing_vector = not self.store.has_vector(md_id)
            if force or existing["body_hash"] != new_hash or missing_vector:
                if isinstance(extra, dict):
                    extra = dict(extra)
                    extra.pop("_memo_embed_pending", None)
                emb = self.embedder.embed([self._compose_for_embed(title, body)])[0]
                assert_valid_embedding(emb, self.cfg.embedder_dims, context=f"reindex update {md_id[:8]}")
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=existing["created"], updated=_now_iso(),
                    body_hash=new_hash, embedding=emb,
                    extra=extra if extra else None,
                    body_text=body,
                )
                reindexed += 1
        # Successful reindex: every meta.path now uses the current
        # memory_dir-relative layout, so future startups can skip the
        # legacy-path probe in `_maybe_warn_legacy_paths`.
        self.store.set_user_version(1)
        counts = {"checked": checked, "reindexed": reindexed, "added": added, "skipped": skipped}
        if reindexed or added:
            from memo.receipts import emit_receipt

            emit_receipt(
                "reindex",
                text=(
                    f"Memo reindex: checked={checked} reindexed={reindexed} "
                    f"added={added} skipped={skipped} force={force}"
                ),
                meta={
                    "checked": checked,
                    "reindexed": reindexed,
                    "added": added,
                    "skipped": skipped,
                    "force": force,
                },
            )
        return counts

    def lint(self) -> dict[str, builtins.list[dict[str, Any]]]:
        """Surface memorias with quality issues.

        Categories:
        - `legacy_extra`: has `extra` keys from mem-vault migration
          (`agent_id`, `last_used`, `usage_count`, `user_id`, `description`).
          These don't affect retrieval but bloat the frontmatter — worth
          a manual cleanup pass.
        - `few_tags`: <3 tags. The CLAUDE.md convention is ≥3 (project +
          domain + technique). Few tags hurt discovery via `memo top <tag>`.
        - `body_skinny`: body shorter than 100 chars. May still be useful
          for one-liner facts but worth checking if the user meant to
          write more.
        - `untitled`: title is literally "untitled" or matches the slug.

        Returns a dict of category → list of {id, title, reason} dicts.
        Pure read; never modifies the store.
        """
        legacy_keys = frozenset({
            "agent_id", "last_used", "usage_count", "user_id", "description",
        })
        out: dict[str, list[dict[str, Any]]] = {
            "legacy_extra": [],
            "few_tags": [],
            "body_skinny": [],
            "untitled": [],
        }
        for r in self.store.list_recent(limit=100_000):
            entry = {"id": r["id"], "title": r["title"]}
            extra = r.get("extra") or {}
            if any(k in extra for k in legacy_keys):
                out["legacy_extra"].append(
                    {**entry, "reason": "mem-vault legacy fields in extra: "
                                        + ", ".join(sorted(set(extra) & legacy_keys))},
                )
            if len(r.get("tags") or []) < 3:
                out["few_tags"].append(
                    {**entry, "reason": f"only {len(r.get('tags') or [])} tag(s)"},
                )
            body = self._read_body(r["path"]) or ""
            if len(body.strip()) < 100:
                out["body_skinny"].append(
                    {**entry, "reason": f"body {len(body.strip())} chars"},
                )
            t = (r["title"] or "").strip().lower()
            if t == "untitled" or not t:
                out["untitled"].append({**entry, "reason": "title missing or 'untitled'"})
        return out

    # -- knowledge graph ----------------------------------------------------

    def extract_entities(
        self, *, ids: builtins.list[str] | None = None, all_: bool = False,
        skip_already_indexed: bool = True,
        max_batch: int | None = None,
    ) -> dict[str, int]:
        """Extract named entities from memorias and write to the graph.

        Modes:
        - `ids=[...]`: process exactly the listed memoria ids.
        - `all_=True`: process every memoria in the store.

        With `skip_already_indexed=True` (default), memorias that
        already have entries in `entity_memoria` are skipped — useful
        for incremental runs after adding new memorias. Pass False to
        force re-extraction (e.g. after improving the prompt).

        Returns counts: `{processed, entities_extracted, links_written, skipped, errors}`.
        Cost: ~0.5-1s per memoria with Qwen2.5-3B. 223 memorias ≈ 2-4 min.
        """
        if not all_ and not ids:
            raise ValueError("pass either ids=[...] or all_=True")

        if all_:
            target = [r["id"] for r in self.store.list_recent(limit=100_000)]
        else:
            target = list(ids or [])

        # Pre-filter already-indexed unless --force.
        if skip_already_indexed:
            target = [
                tid for tid in target
                if not self.graph.memoria_entities(tid)
            ]

        if max_batch is not None:
            target = target[:max_batch]

        counts = {"processed": 0, "entities_extracted": 0,
                  "links_written": 0, "skipped": 0, "errors": 0}

        if not target:
            return counts

        chat = self._ensure_chat()

        for tid in target:
            r = self.store.get(tid)
            if r is None:
                counts["skipped"] += 1
                continue
            body = self._read_body(r["path"])
            if not body.strip():
                counts["skipped"] += 1
                continue
            # Build prompt: title + body excerpt. Cap to ~3000 chars to
            # keep the helper LLM cheap; entities tend to live in the
            # opening paragraphs.
            user_msg = (
                f"Title: {r['title']}\n"
                f"Tags: {', '.join(r['tags']) if r['tags'] else '—'}\n\n"
                f"{body[:3000]}"
            )
            try:
                out = chat_with_timeout(
                    chat, timeout=30,
                    model=self.cfg.helper_model,
                    messages=[
                        {"role": "system", "content": _EXTRACT_ENTITIES_SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    options={"temperature": 0.0, "max_tokens": 384, "thinking": False},
                )
                if out is None:
                    _log.warning("extract_entities: LLM timeout for %s", tid[:8])
                    counts["errors"] += 1
                    continue
                text = ((out.get("message") or {}).get("content") or "").strip()
            except Exception as exc:
                _log.warning("extract_entities: LLM call failed for %s: %s", tid[:8], exc)
                counts["errors"] += 1
                continue
            text = strip_llm_output(text)
            try:
                data = json.loads(text) if text else {}
            except (ValueError, TypeError):
                counts["errors"] += 1
                continue
            ents = data.get("entities") if isinstance(data, dict) else None
            if not isinstance(ents, list):
                ents = []
            # Filter to dicts with both name + type fields.
            ents = [
                {"name": e.get("name"), "type": e.get("type")}
                for e in ents
                if isinstance(e, dict) and e.get("name") and e.get("type")
            ]
            n = self.graph.record_extraction(
                memoria_id=tid,
                memoria_date=r["created"][:10] if r.get("created") else _now_iso()[:10],
                entities=ents,
                extracted_at=_now_iso(),
            )
            counts["processed"] += 1
            counts["entities_extracted"] += len(ents)
            counts["links_written"] += n
        return counts

    def consolidate(
        self, *, threshold: float = 0.85, max_clusters: int = 50,
        type_: str | None = None, skip_llm: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        """Propose near-duplicate merges (LLM synthesis step).

        Optional off-resident-set path: when ``MEMO_MAINT_VIA_DAEMON=1`` and the
        maintenance daemon is reachable, the heavy synthesis LLM runs in that
        daemon's process (keeping it out of memo-mcp's resident set) and returns
        the proposals here. Any miss (flag off, daemon down) runs in-process
        exactly as before — see :meth:`_consolidate_in_process`. The daemon
        itself calls ``_consolidate_in_process`` directly, so it never re-routes
        to itself.

        When ``skip_llm=True`` the LLM classification step is skipped and all
        clusters are returned with ``relationship="duplicate"`` assumed. Used by
        the high-confidence fast lane in ``AdvancedConsolidator`` where cosine
        ≥ auto_threshold makes LLM classification unnecessary. Always runs
        in-process (daemon path is bypassed) since the O(N²) clustering is cheap
        without the LLM.
        """
        from memo.flags import flag_bool

        if not skip_llm and flag_bool("MEMO_MAINT_VIA_DAEMON"):
            from memo import maint_client

            proposals = maint_client.consolidate(
                threshold=threshold, max_clusters=max_clusters, type_=type_,
            )
            if proposals is not None:
                return proposals
            # daemon unreachable → fall through to in-process (graceful)
        return self._consolidate_in_process(
            threshold=threshold, max_clusters=max_clusters, type_=type_,
            skip_llm=skip_llm,
        )

    def _consolidate_in_process(
        self, *, threshold: float = 0.85, max_clusters: int = 50,
        type_: str | None = None, skip_llm: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        """Find clusters of near-duplicate memorias and propose actions.

        Algorithm:
        1. Pull all stored embeddings (we have them already; no re-embed).
        2. Greedy single-link clustering by cosine ≥ `threshold`.
           Each memoria joins the first existing cluster it's
           ≥-similar to, or starts a new one.
        3. Drop singletons. The remaining clusters are candidates.
        4. For each cluster, MLXChat 7B reads the bodies and emits a
           JSON `{summary, relationship, rationale}` per
           `_CONSOLIDATE_SYSTEM_PROMPT`.
        5. Return ranked clusters (largest first), capped at
           `max_clusters` to keep the LLM cost finite on big corpora.

        DOES NOT modify anything. The user reviews the output and
        decides via `memo update` / `memo delete`.

        Threshold tuning: 0.85 catches obvious dupes, 0.92+ only catches
        near-identical text. The default 0.85 is conservative for the
        Qwen3-Embedding-0.6B vector space.
        """
        # 1) Pull all (id, embedding, title, type, tags) tuples.
        #    Direct SQL is cheaper than going through the public store
        #    API per-row; we only need 1024 floats x N to fit in RAM,
        #    fine for thousands of entries.
        import struct

        store_conn = self.store._conn
        rows = store_conn.execute(
            "SELECT vec.id AS id, vec.embedding AS emb, "
            "       meta.title, meta.type, meta.tags, meta.path, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id "
            + ("WHERE meta.type = ? " if type_ else "")
            + "ORDER BY meta.updated DESC",
            (type_,) if type_ else (),
        ).fetchall()

        items: list[dict[str, Any]] = []
        for r in rows:
            blob = r["emb"]
            v = list(struct.unpack(f"<{len(blob)//4}f", blob))
            items.append({
                "id": r["id"],
                "title": r["title"],
                "type": r["type"],
                "tags": json.loads(r["tags"]) if r["tags"] else [],
                "path": r["path"],
                "updated": r["updated"],
                "emb": v,
            })

        if not items:
            return []

        # 2) Greedy single-link clustering. Use numpy for O(N²) dot products
        #    when available (1024-dim × 2000 items in Python = ~400s; numpy = <1s).
        try:
            import numpy as _np
            _mat = _np.array([it["emb"] for it in items], dtype=_np.float32)  # (N, D)
            # L2-normalise rows so dot == cosine.
            _norms = _np.linalg.norm(_mat, axis=1, keepdims=True)
            _norms[_norms == 0] = 1.0
            _mat = _mat / _norms
            _reps: list[int] = []  # index of each cluster's representative
            _cluster_map: list[int] = [-1] * len(items)
            for i in range(len(items)):
                if _reps:
                    sims = _mat[_reps] @ _mat[i]  # (K,)
                    best = int(_np.argmax(sims))
                    if float(sims[best]) >= threshold:
                        _cluster_map[i] = _reps[best]
                        continue
                _reps.append(i)
                _cluster_map[i] = i
            _cluster_dict: dict[int, list[int]] = {}
            for i, rep in enumerate(_cluster_map):
                _cluster_dict.setdefault(rep, []).append(i)
            clusters = list(_cluster_dict.values())
        except ImportError:
            clusters = []
            for i in range(len(items)):
                joined = False
                for cluster in clusters:
                    rep_item = items[cluster[0]]
                    if sum(x * y for x, y in zip(items[i]["emb"], rep_item["emb"], strict=True)) >= threshold:
                        cluster.append(i)
                        joined = True
                        break
                if not joined:
                    clusters.append([i])

        # 3) Drop singletons; rank by size (then by most-recent updated).
        candidate_clusters = [c for c in clusters if len(c) >= 2]
        candidate_clusters.sort(
            key=lambda c: (-len(c), items[c[0]]["updated"]),
        )
        candidate_clusters = candidate_clusters[:max_clusters]

        if not candidate_clusters:
            return []

        # 4) For each cluster, ask MLXChat to summarise + classify.
        #    Skipped when skip_llm=True (fast lane at very high cosine threshold).
        out: list[dict[str, Any]] = []
        for ci, cluster in enumerate(candidate_clusters):
            members = []
            for idx in cluster:
                it = items[idx]
                body = self._read_body(it["path"])
                members.append({
                    "id": it["id"],
                    "id_short": it["id"][:8],
                    "title": it["title"],
                    "type": it["type"],
                    "tags": it["tags"],
                    "updated": it["updated"],
                    "body_preview": (body[:600] + ("…" if len(body) > 600 else "")),
                })

            if skip_llm:
                out.append({
                    "cluster_id": ci,
                    "size": len(members),
                    "members": members,
                    "summary": "",
                    "relationship": "duplicate",
                    "rationale": f"High-confidence cluster (cosine ≥ {threshold:.2f}); LLM skipped.",
                })
                continue

            # Build LLM prompt with all members, capped to avoid blowing the
            # context window on large clusters (50+ items × 600 chars each).
            _MAX_CONSOLIDATION_PROMPT_CHARS = 24_000
            member_lines = [
                f"[{m['id_short']}] title: {m['title']}  |  updated: {m['updated']}\n"
                f"{m['body_preview']}"
                for m in members
            ]
            _chars = 0
            _included = []
            for _line in member_lines:
                if _chars + len(_line) > _MAX_CONSOLIDATION_PROMPT_CHARS:
                    break
                _included.append(_line)
                _chars += len(_line) + 5  # 5 for "\n---\n" separator
            prompt = "Cluster:\n\n" + "\n---\n".join(_included)
            try:
                chat = self._ensure_chat()
                chat_out = chat_with_timeout(
                    chat, timeout=60,
                    model=self.cfg.helper_model,
                    messages=[
                        {"role": "system", "content": _CONSOLIDATE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.0, "max_tokens": 384, "thinking": False},
                )
                if chat_out is None:
                    _log.warning("consolidate: LLM timeout for cluster %d", ci)
                    continue
                text = ((chat_out.get("message") or {}).get("content") or "").strip()
            except Exception as exc:
                _log.warning("consolidate: LLM call failed for cluster %d: %s", ci, exc)
                text = ""
            text = strip_llm_output(text)
            try:
                data = json.loads(text) if text else {}
            except (ValueError, TypeError):
                data = {}
            out.append({
                "cluster_id": ci,
                "size": len(members),
                "members": members,
                "summary": (data.get("summary") or "").strip(),
                "relationship": data.get("relationship") if data.get("relationship") in
                    ("duplicate", "evolution", "facets", "unrelated") else "unrelated",
                "rationale": (data.get("rationale") or "").strip(),
            })
        return out

    def synthesize_cross_cluster(
        self,
        *,
        threshold: float | None = None,
        min_cluster_size: int | None = None,
        max_clusters: int | None = None,
        min_confidence: str | None = None,
        dry_run: bool = False,
    ) -> builtins.list[dict[str, Any]]:
        """Generate emergent insights from semantically related memory clusters.

        Unlike consolidation (which asks "are these the same thing?"), synthesis
        asks "what do these memories collectively IMPLY that none states alone?"
        Results are saved as ``type=synthesis`` memories with provenance links.

        Algorithm:
        1. Cluster all durable (non-reference, non-synthesis) memories at a
           looser cosine threshold than consolidation (default 0.78 vs 0.85).
        2. Drop clusters smaller than ``min_cluster_size`` (default 3).
        3. For each cluster, check if an up-to-date synthesis already exists
           (provenance hash match) — skip if so.
        4. Call MLXChat with ``_SYNTHESIS_SYSTEM_PROMPT``. If confidence meets
           the floor and title is non-null, save a new ``type=synthesis`` record.
        5. Return the list of results (proposed or saved).

        With ``dry_run=True``, returns proposals without saving anything.
        """
        import hashlib
        import struct

        from memo.flags import flag_float, flag_int, flag_str

        threshold = threshold if threshold is not None else (
            flag_float("MEMO_SYNTHESIS_THRESHOLD") or 0.78
        )
        min_cluster_size = min_cluster_size if min_cluster_size is not None else (
            flag_int("MEMO_SYNTHESIS_MIN_CLUSTER") or 3
        )
        max_clusters = max_clusters if max_clusters is not None else (
            flag_int("MEMO_SYNTHESIS_MAX_CLUSTERS") or 20
        )
        min_confidence = min_confidence or flag_str("MEMO_SYNTHESIS_MIN_CONFIDENCE") or "medium"
        _conf_rank = {"low": 0, "medium": 1, "high": 2}
        min_conf_rank = _conf_rank.get(min_confidence, 1)

        store_conn = self.store._conn

        # 1) Pull all non-synthesis, non-reference embeddings.
        rows = store_conn.execute(
            "SELECT vec.id AS id, vec.embedding AS emb, "
            "       meta.title, meta.type, meta.tags, meta.path, meta.updated "
            "FROM vec JOIN meta ON meta.id = vec.id "
            "WHERE meta.type NOT IN ('reference', 'synthesis') "
            "ORDER BY meta.updated DESC",
        ).fetchall()

        items: list[dict[str, Any]] = []
        for r in rows:
            blob = r["emb"]
            v = list(struct.unpack(f"<{len(blob)//4}f", blob))
            items.append({
                "id": r["id"],
                "title": r["title"],
                "type": r["type"],
                "path": r["path"],
                "updated": r["updated"],
                "emb": v,
            })

        if not items:
            return []

        # 2) Greedy single-link clustering (same algorithm as consolidation).
        def _dot(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b, strict=True))

        clusters: list[list[int]] = []
        for i in range(len(items)):
            joined = False
            for cluster in clusters:
                rep = items[cluster[0]]
                if _dot(items[i]["emb"], rep["emb"]) >= threshold:
                    cluster.append(i)
                    joined = True
                    break
            if not joined:
                clusters.append([i])

        candidate_clusters = [c for c in clusters if len(c) >= min_cluster_size]
        candidate_clusters.sort(key=lambda c: (-len(c), items[c[0]]["updated"]))
        candidate_clusters = candidate_clusters[:max_clusters]

        if not candidate_clusters:
            return []

        # 3) Load existing synthesis provenance hashes to skip duplicates.
        existing_hashes: set[str] = set()
        existing_rows = store_conn.execute(
            "SELECT meta.path FROM meta WHERE meta.type = 'synthesis'",
        ).fetchall()
        for er in existing_rows:
            p = er["path"]
            if p:
                ep = self._resolve_existing(p)
                if ep.is_file():
                    try:
                        ep_post = frontmatter.loads(ep.read_text(encoding="utf-8"))
                        _ep_extra: dict = ep_post.get("extra") or {}  # type: ignore[assignment]
                        h = str(_ep_extra.get("synthesis_sources_hash") or "").strip()
                        if h:
                            existing_hashes.add(h)
                    except (OSError, ValueError) as exc:
                        # A bad existing synthesis file silently missed its hash →
                        # duplicate synthesis on the next run. Log the breadcrumb.
                        _log.debug("synthesis: could not read hash from %s: %s", p, exc)

        # 4) LLM synthesis pass.
        chat = self._ensure_chat()

        out: list[dict[str, Any]] = []
        saved = 0

        for cluster in candidate_clusters:
            source_ids = [items[idx]["id"] for idx in cluster]
            sources_hash = hashlib.sha256(
                ",".join(sorted(source_ids)).encode()
            ).hexdigest()[:16]

            if sources_hash in existing_hashes:
                continue  # up-to-date synthesis already exists

            members = []
            for idx in cluster:
                it = items[idx]
                body = self._read_body(it["path"])
                members.append({
                    "id": it["id"],
                    "id_short": it["id"][:8],
                    "title": it["title"],
                    "type": it["type"],
                    "body_preview": (body[:500] + ("…" if len(body) > 500 else "")),
                })

            _MAX_CHARS = 20_000
            lines = [
                f"[{m['id_short']}] ({m['type']}) {m['title']}\n{m['body_preview']}"
                for m in members
            ]
            _chars = 0
            _included: list[str] = []
            for line in lines:
                if _chars + len(line) > _MAX_CHARS:
                    break
                _included.append(line)
                _chars += len(line) + 5

            prompt = "Cluster:\n\n" + "\n---\n".join(_included)

            try:
                chat_out = chat_with_timeout(
                    chat, timeout=60,
                    model=self.cfg.helper_model,
                    messages=[
                        {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.0, "max_tokens": 512, "thinking": False},
                )
                if chat_out is None:
                    _log.warning("synthesize: LLM timeout for cluster")
                    continue
                text = ((chat_out.get("message") or {}).get("content") or "").strip()
            except Exception as exc:
                _log.warning("synthesize: LLM call failed for cluster: %s", exc)
                continue

            text = strip_llm_output(text)
            try:
                data = json.loads(text) if text else {}
            except (ValueError, TypeError):
                data = {}

            title = (data.get("title") or "").strip()
            body = (data.get("body") or "").strip()
            confidence = data.get("confidence") if data.get("confidence") in _conf_rank else "low"
            rationale = (data.get("rationale") or "").strip()

            result: dict[str, Any] = {
                "sources": source_ids,
                "sources_hash": sources_hash,
                "title": title,
                "body": body,
                "confidence": confidence,
                "rationale": rationale,
                "saved": False,
            }

            if (
                title
                and body
                and _conf_rank.get(str(confidence), 0) >= min_conf_rank
                and not dry_run
            ):
                try:
                    rec = self.save(
                        content=body,
                        title=title,
                        type_="synthesis",
                        tags=["synthesis"],
                        extra={
                            "synthesis_sources": source_ids,
                            "synthesis_sources_hash": sources_hash,
                            "synthesis_rationale": rationale,
                            "synthesis_confidence": confidence,
                        },
                    )
                    result["saved"] = True
                    result["id"] = rec.id
                    existing_hashes.add(sources_hash)
                    saved += 1
                except Exception as exc:
                    _log.warning("synthesize: save failed: %s", exc)

            out.append(result)

        _log.info("synthesize: processed %d clusters → %d saved", len(out), saved)
        return out

    def gc(self, *, fix: bool = False) -> dict[str, builtins.list[str]]:
        """Find orphans between the store and the memory dir.

        - `orphan_store`: store rows whose `.md` is missing on disk.
        - `orphan_disk`: `.md` files with an `id` frontmatter that the
          store doesn't know about. (Untagged `.md` files — no `id` —
          are ignored: they're user-authored content, not memories.)

        With `fix=True`, deletes orphan store rows. `.md` files are
        never deleted automatically — that's destructive and the user
        should review them first. Use `memo reindex` to absorb
        orphan disk files into the store.
        """
        orphan_store: list[str] = []
        orphan_disk: list[str] = []

        # Store-side: walk meta, check file existence (with legacy fallback).
        for r in self.store.list_recent(limit=100_000):
            if not self._resolve_existing(r["path"]).is_file():
                orphan_store.append(r["id"])
                if fix:
                    self.store.delete(r["id"])

        # Disk-side: walk memory dir, check ids in store.
        if self.cfg.memory_dir.is_dir():
            for md_path in self.cfg.memory_dir.rglob("*.md"):
                try:
                    post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
                except Exception as exc:
                    _log.debug("gc: skipping %s (parse error): %s", md_path.name, exc)
                    continue
                md_id = post.get("id")
                if not md_id or not isinstance(md_id, str):
                    continue
                if self.store.get(md_id) is None:
                    orphan_disk.append(str(md_path.relative_to(self.cfg.memory_dir)))

        # Stale synthesis: type=synthesis memories where ≥1 source no longer
        # exists in the store. The synthesis is derived knowledge — if its
        # sources are gone the insight is unverifiable and should be archived.
        stale_synthesis: list[str] = []
        synth_rows = self.store._conn.execute(
            "SELECT meta.id, meta.path FROM meta WHERE meta.type = 'synthesis'",
        ).fetchall()
        for sr in synth_rows:
            p = sr["path"]
            if not p:
                continue
            sp = self._resolve_existing(p)
            if not sp.is_file():
                continue  # already caught above by orphan_store walk
            try:
                post = frontmatter.loads(sp.read_text(encoding="utf-8"))
                _extra: dict = post.get("extra") or {}  # type: ignore[assignment]
                source_ids = _extra.get("synthesis_sources") or []
                if not source_ids:
                    continue
                for sid in source_ids:
                    if self.store.get(sid) is None:
                        stale_synthesis.append(sr["id"])
                        if fix:
                            self.lifecycle.archive_memoria(sr["id"])
                        break
            except Exception as exc:
                _log.debug("gc: stale-synthesis check failed for %s: %s", sr["id"][:8], exc)

        return {
            "orphan_store": orphan_store,
            "orphan_disk": orphan_disk,
            "stale_synthesis": stale_synthesis,
        }

