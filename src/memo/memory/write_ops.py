"""Write-path operations for `Memory` — save pipeline + internal helpers.

`_WriteOpsMixin` holds save() and its helpers. update() lives in update_ops.py
and forget/unforget/delete() in delete_ops.py — all three are joined on the
Memory facade via multiple inheritance. The real attributes/managers live on the
`Memory` facade; this mixin only declares the typed contract via `_MemoryBase`.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import frontmatter

from memo._trace import current_trace
from memo.fact_extraction import upsert_declared_fact_edges
from memo.flags import flag_bool
from memo.memory._base import _MemoryBase
from memo.memory.record import (
    _DERIVE_SYSTEM_PROMPT,
    _VALID_TYPES,
    MemoryRecord,
    _derive_title,
    _extract_provenance,
    _log,
    _normalise_tags,
    _now_iso,
    _slugify,
    bump_support_if_enabled,
    in_derived_save_scope,
    is_reference_noise,
    markdown_body,
)
from memo.prompt_overrides import resolve_prompt
from memo.save_gate import resolve_gate
from memo.tiers import REFERENCE_TYPES
from memo.util import sha256_short as _sha256_short


class _StrictDedupRefused(ValueError):
    """Raised inside the near-dup gate's best-effort try/except to signal an
    intentional refusal (strict preset) rather than a dedup-check failure —
    the surrounding except re-raises this instead of swallowing it."""


_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "decision",
        re.compile(
            r"\b(decided?\s+to|we\s+will\s+use|going\s+with|chosen?\s+to|from\s+now\s+on|"
            r"the\s+decision\s+is|decidimos)\b",
            re.I,
        ),
    ),
    (
        "preference",
        re.compile(
            r"\b(i\s+prefer|prefer\s+to|always\s+use|i\s+like\s+to|"
            r"i\s+don'?t\s+want|prefiero|siempre\s+usar)\b",
            re.I,
        ),
    ),
    (
        "bug",
        re.compile(
            r"\b(bug:|issue:|found\s+that|the\s+problem\s+was|root\s+cause|"
            r"error\s+was|causa\s+ra[ií]z|el\s+problema\s+era)\b",
            re.I,
        ),
    ),
    (
        "fact",
        re.compile(
            r"\b(turns?\s+out|discovered\s+that|learned\s+that|actually\s+the|"
            r"resulta\s+que|descubr[íi]\s+que)\b",
            re.I,
        ),
    ),
]


def _infer_type_from_content(content: str) -> str | None:
    """Zero-cost regex-based type inference. Returns a type string or None."""
    if not flag_bool("MEMO_CAPTURE_PATTERN_TYPES"):
        return None
    snippet = content[:600]
    for type_name, pattern in _TYPE_PATTERNS:
        if pattern.search(snippet):
            return type_name
    return None


def _upsert_declared_fact_edges_best_effort(
    memory: _MemoryBase,
    *,
    record_id: str,
    title: str,
    type_: str,
    created: str,
    updated: str,
    extra: dict[str, Any] | None,
) -> None:
    try:
        upsert_declared_fact_edges(
            memory.fact_edges,
            record_id=record_id,
            title=title,
            type_=type_,
            created=created,
            updated=updated,
            extra=extra,
        )
    except Exception as exc:
        _log.debug("fact edge extraction skipped during save(%s): %s", record_id[:8], exc)


def _graph_entities_from_extra(extra: dict[str, Any]) -> list[dict[str, str]]:
    """Normalize saved ``extra['entities']`` into GraphStore rows."""
    raw = extra.get("entities") or []
    if not isinstance(raw, list):
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        name = ""
        type_ = "concept"
        if isinstance(item, str):
            name = item.strip()
        elif isinstance(item, dict):
            name = str(item.get("name") or "").strip()
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in {"person", "project", "technology", "file", "org", "concept"}:
                type_ = item_type
        if not name:
            continue
        key = f"{name.lower()}:{type_}"
        if key in seen:
            continue
        seen.add(key)
        out.append({"name": name, "type": type_})
    return out


class _WriteOpsMixin(_MemoryBase):
    # -- save ---------------------------------------------------------------

    @contextlib.contextmanager
    def _data_dir_write_lock(self) -> Iterator[None]:
        """Serialize path-sensitive markdown writes across instances/processes."""
        with self._save_path_lock:
            if self._data_lock_depth:
                self._data_lock_depth += 1
                try:
                    yield
                finally:
                    self._data_lock_depth -= 1
                return
            from memo.atomic_io import authority_write_lock

            with authority_write_lock(self.cfg.memory_dir):
                self._data_lock_depth = 1
                try:
                    yield
                finally:
                    self._data_lock_depth = 0

    def _atomic_write_text(self, rel_path: str, text: str) -> None:
        """Publish UTF-8 text atomically without exposing a partial target."""
        path = self._safe_path_under(self.cfg.memory_dir, rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            # Persist the directory entry where the platform supports it.
            with contextlib.suppress(OSError):
                dir_fd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
        finally:
            with contextlib.suppress(FileNotFoundError):
                tmp_path.unlink()

    def _presence_bump_save(self) -> None:
        try:
            from memo import presence

            presence.bump(self.cfg.state_dir, saves=1)
        except Exception:  # noqa: S110  # decoration — never break a save
            pass

    def _record_graph_entities_from_extra(
        self,
        *,
        record_id: str,
        created_iso: str,
        extra: dict[str, Any],
        extractor: str,
    ) -> None:
        entities = _graph_entities_from_extra(extra)
        if not entities:
            return
        try:
            self.graph.record_extraction(
                memory_id=record_id,
                memory_date=created_iso[:10],
                entities=entities,
                extracted_at=_now_iso(),
                extractor=extractor,
                extractor_version="save-v1",
                confidence=0.95 if extractor == "explicit" else 0.45,
            )
        except Exception as exc:
            _log.debug("graph entity write skipped for %s: %s", record_id[:8], exc)

    def _derive_metadata(self, content: str) -> dict[str, Any]:
        """Use the configured helper LLM to derive
        {title, type, tags} from raw content. Returns a dict with
        whatever keys the model produced (any can be None on parse
        failure). Caller decides whether to fill missing fields.

        Failure modes are absorbed: a bad LLM response yields an empty
        dict and the caller falls back to heuristics. We never propagate
        an LLM error up to a save() call — the save must succeed even
        if the helper is broken.
        """
        chat = self._ensure_chat()
        try:
            out = chat.chat(
                model=self.cfg.helper_model,
                messages=[
                    {
                        "role": "system",
                        "content": resolve_prompt(
                            "derive", _DERIVE_SYSTEM_PROMPT, self.cfg.state_dir
                        ),
                    },
                    # Cap input to keep the prompt cheap. The helper only
                    # needs the gist, not the full body.
                    {"role": "user", "content": content[:2000]},
                ],
                options={"temperature": 0.0, "max_tokens": 256},
            )
            text = (out.get("message") or {}).get("content") or ""
        except Exception as exc:
            _log.warning("_derive_metadata LLM call failed: %s", exc)
            return {}
        # Tolerate markdown code fences even though the prompt forbids them.
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        try:
            data = json.loads(text)
        except (ValueError, TypeError) as exc:
            _log.warning("_derive_metadata JSON parse failed (%r…): %s", text[:80], exc)
            return {}
        if not isinstance(data, dict):
            return {}
        derived: dict[str, Any] = {}
        t_title = data.get("title") or ""
        if isinstance(t_title, str) and t_title.strip():
            derived["title"] = t_title.strip()[:80]
        t_type = data.get("type")
        if isinstance(t_type, str) and t_type in _VALID_TYPES:
            derived["type"] = t_type
        t_tags = data.get("tags") or []
        if isinstance(t_tags, list):
            derived["tags"] = _normalise_tags([t for t in t_tags if isinstance(t, str)])
        return derived

    def save(
        self,
        *,
        content: str,
        title: str | None = None,
        type_: str = "note",
        type: str | None = None,
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        auto_derive: bool = False,
        auto_project: bool = True,
        cwd: str | None = None,
        created: str | None = None,
        defer_embed: bool = False,
        respect_synapse_freeze: bool | None = None,
        skip_memflow_receipt: bool = False,
        topic_key: str | None = None,
        normalized_hash: str | None = None,
        valid_at: str | None = None,
        invalid_at: str | None = None,
    ) -> MemoryRecord:
        """Persist a memory to disk + index.

        - `content`: free-form markdown body (no frontmatter; we add it).
        - `title`: optional. If omitted, derived from the first line of
          content (truncated, slug-safe).
        - `type_`: must be in `_VALID_TYPES`. `type` is accepted as a
          compatibility alias. `note` is the default neutral value.
        - `tags`: optional list. Lower-cased + de-duplicated.
        - `extra`: arbitrary JSON-serialisable metadata bag.
        - `created`: optional ISO8601 override for the `created` field
          (frontmatter + index). When None, defaults to NOW. `updated`
          always reflects this write. Use to back-date imported records
          (e.g. historical WhatsApp messages) so temporal queries see the
          original event time, not ingest time.
        - `auto_derive`: when True, calls the configured helper LLM
          to fill any missing field
          (title is None, type_ is "note" with no tags). Adds ~1-2s
          latency on first call (cold model load) plus ~0.5-1s per save.
          Use for callers (eg. another agent) that don't carry context
          to derive metadata themselves.
        - `defer_embed`: when True, write markdown + metadata + BM25
          index only. Semantic search won't see the record until
          `memo reindex` runs with the embedder available.
        - `respect_synapse_freeze`: when True, query synapse's
          `RealityConflict` ledger before commit and raise
          `WriteRefused` if a blocking freeze-write covers this
          memory's topic. Defaults to the env knob
          `MEMO_RESPECT_SYNAPSE_FREEZE=1` (opt-in). Only fires when
          `extra` carries a `synapse_trace_id` — anonymous saves
          bypass the check.
        - `valid_at`/`invalid_at`: optional ISO8601 world-validity interval
          (distinct from `created`/`updated` learned-time). Pure pass-through
          here — persisted to the row when supplied, else left None. Default
          population (`valid_at = created`) and frontmatter mirroring are
          separate paths.
        """
        if not content or not content.strip():
            raise ValueError("`content` must be non-empty")

        # Auto-attach the synapse trace id when the caller did not carry an
        # explicit one in `extra`. Two transports feed it: the warm-daemon MCP
        # path sets the trace contextvar (from the `x-synapse-trace-id` header,
        # via the server middleware); the subprocess path sets `SYNAPSE_TRACE_ID`
        # in the env. Prefer the contextvar (per-request, precise) and fall back
        # to env (covers direct `memo save` CLI invocations). Lets provenance
        # walks link memo writes back to the synapse session that spawned them.
        ambient_trace = current_trace() or os.environ.get("SYNAPSE_TRACE_ID", "").strip()
        if ambient_trace and (extra is None or not extra.get("synapse_trace_id")):
            extra = dict(extra or {})
            extra["synapse_trace_id"] = ambient_trace

        # Freeze-write protocol: opt-in pre-write check against synapse.
        # Only fires when (a) the caller asked (kwarg or env), (b) the
        # save carries provenance (otherwise we have no agent context
        # to reason about), and (c) synapse is on PATH.
        if respect_synapse_freeze is None:
            respect_synapse_freeze = flag_bool("MEMO_RESPECT_SYNAPSE_FREEZE")
        if respect_synapse_freeze and extra and extra.get("synapse_trace_id"):
            self._enforce_synapse_freeze(
                title=title,
                content=content,
                tags=tags,
                trace_id=str(extra.get("synapse_trace_id") or ""),
            )
        if type is not None:
            _log.warning("save: `type=` is deprecated, use `type_=`")
            if type_ != "note" and type_ != type:
                raise ValueError("Pass either `type_` or `type`, not conflicting values")
            type_ = type
        if type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )

        # Reference-tier noise gate: reject near-empty bulk chunks with no
        # heading/link (the class that accreted as dead index rows before the
        # ingest filter existed). Durable tiers are exempt — short facts and
        # preferences are legitimate. Mirrors the repo/vault ingest filter so
        # this junk cannot re-enter through `memo_save`.
        if type_ in REFERENCE_TYPES and is_reference_noise(content):
            raise ValueError(
                "reference content is near-empty noise (no heading/link, "
                f"<{60} chars); refusing to index",
            )

        # Temporal grounding (MEMO_SAVE_NORMALIZE_DATES, default off): annotate
        # relative date expressions with absolute ISO dates so "ayer"/"last
        # week" stay recoverable later. Anchored to `created` when the caller
        # back-dates (imports resolve against event time); durable tiers only —
        # reference chunks mirror vault files and must not be rewritten.
        # _normalize_relative_dates never raises (returns input on error).
        if flag_bool("MEMO_SAVE_NORMALIZE_DATES") and type_ not in REFERENCE_TYPES:
            import datetime as _dt

            from memo.memory.consolidate_ops import _normalize_relative_dates

            try:
                _ref = _dt.date.fromisoformat(created[:10]) if created else _dt.date.today()
            except ValueError:
                _ref = _dt.date.today()
            content = _normalize_relative_dates(content, _ref)

        if auto_derive:
            # Only fire the LLM if at least one field looks "default-y".
            # User-provided values always win.
            wants_title = title is None
            wants_type = type_ == "note"
            wants_tags = not tags
            # Zero-cost regex pre-pass: detect type before calling the LLM.
            # Avoids the helper-model latency (~1-2s) for clearly typed content.
            if wants_type:
                inferred = _infer_type_from_content(content)
                if inferred:
                    type_ = inferred
                    wants_type = False
            if wants_title or wants_type or wants_tags:
                derived = self._derive_metadata(content)
                if wants_title and derived.get("title"):
                    title = derived["title"]
                if wants_type and derived.get("type"):
                    type_ = derived["type"]
                if wants_tags and derived.get("tags"):
                    tags = derived["tags"]

        title = (title or _derive_title(content)).strip()
        if not title:
            title = "untitled"

        norm_tags = _normalise_tags(tags or [])

        # Auto-tag with the caller's project (git toplevel basename or
        # MEMO_PROJECT_TAG) so per-repo recall can boost the right
        # memories. Skipped when the caller already passed any
        # `project:` tag — explicit always wins.
        if auto_project and flag_bool("MEMO_AUTO_PROJECT_TAG"):
            try:
                from memo.project import current_project_tag, has_project_tag

                if not has_project_tag(norm_tags):
                    pt = current_project_tag(cwd)
                    if pt:
                        norm_tags = _normalise_tags([*norm_tags, pt])
            except Exception as exc:
                _log.warning("auto-project tag failed (cwd=%s): %s", cwd, exc)

        now_iso = _now_iso()
        created_iso = created or now_iso
        # Truncate content for embedding (vec store doesn't truncate;
        # disk file keeps full content). 64KB is the default cap.
        if len(content) > self.cfg.max_content_chars:
            _log.warning(
                "save: content truncated from %d to %d chars (title=%r)",
                len(content),
                self.cfg.max_content_chars,
                title,
            )
        content = content[: self.cfg.max_content_chars]

        # Near-duplicate check: quick vec search before committing. Best-effort
        # — never blocks the save. Gated on MEMO_SAVE_DEDUP_CHECK (default on).
        # Skipped on an empty corpus (no embed cost, no duplicates possible).
        from memo.flags import flag_float

        if flag_bool("MEMO_SAVE_DEDUP_CHECK") and not defer_embed:
            try:
                _existing_sample = self.store.list_recent(limit=1)
                if _existing_sample:
                    _dedup_threshold_flag = flag_float("MEMO_SAVE_DEDUP_THRESHOLD")
                    _dedup_threshold = (
                        0.88 if _dedup_threshold_flag is None else _dedup_threshold_flag
                    )
                    _dedup_q = f"{title}\n{content[:300]}"
                    _dedup_emb = self.embedder.embed_query(_dedup_q)
                    _dedup_hits = self.store.search(_dedup_emb, limit=3)
                    for _dh in _dedup_hits:
                        if (_dh.get("score") or 0.0) >= _dedup_threshold:
                            if resolve_gate(type_).dedup_mode == "refuse" and _dh.get("id"):
                                raise _StrictDedupRefused(
                                    f"near-duplicate of {_dh['id'][:8]} "
                                    f"(sim={_dh.get('score', 0.0):.2f}); strict gate for "
                                    f"type '{type_}' — use `memo update {_dh['id'][:8]}` instead"
                                )
                            _dup_title = _dh.get("title") or (_dh.get("id") or "")[:8]
                            _dup_score = _dh.get("score", 0.0)
                            _dup_id = (_dh.get("id") or "")[:8]
                            # Corroboration (C1): the existing record was just
                            # re-asserted by an independent save. Count it —
                            # this signal used to be discarded with the warning.
                            if _dh.get("id"):
                                bump_support_if_enabled(self.store, [_dh["id"]])
                            # C3: absorb-on-recurrence (flag-gated, default off).
                            # Never in derived-save scope — dream/consolidation
                            # batch saves are merged by the consolidate pass.
                            if (
                                flag_bool("MEMO_SAVE_ABSORB")
                                and not in_derived_save_scope()
                                and _dh.get("id")
                            ):
                                _absorbed = self._absorb_into_existing(_dh["id"], title, content)
                                if _absorbed is not None:
                                    return _absorbed
                            # Demote to debug for dream/consolidation batch saves:
                            # the nudge to `memo update` is only actionable for an
                            # interactive human, and the same dream run's consolidate
                            # pass merges these near-dups anyway.
                            _dedup_level = (
                                logging.DEBUG if in_derived_save_scope() else logging.WARNING
                            )
                            _log.log(
                                _dedup_level,
                                "save: near-duplicate detected — '%s' (sim=%.2f, id=%s). "
                                "Consider `memo update %s` instead of creating a new memory.",
                                _dup_title,
                                _dup_score,
                                _dup_id,
                                _dup_id,
                            )
                            break
            except _StrictDedupRefused:
                raise
            except Exception as _exc:
                _log.debug("save: dedup check skipped: %s", _exc)

        # Topic key upsert (session pattern): the lookup runs INSIDE
        # _save_path_lock (below), alongside the path reuse it feeds — see the
        # comment on the lock for why holding it across the SELECT matters.
        use_existing_id: str | None = None
        existing_path: str | None = None
        reservation_error: Exception | None = None

        body_hash = _sha256_short(content)

        # Generate normalized_hash for exact deduplication (session pattern)
        # Use user-provided if given, otherwise auto-generate
        if normalized_hash is None:
            from memo.flags import flag_bool as _flag_bool

            if _flag_bool("MEMO_DEDUP_EXACT"):
                try:
                    from memo.server_session_patterns import _normalize_hash as _pattern_hash

                    normalized_hash = _pattern_hash(title or "", type_, "project")
                except Exception:
                    _log.debug("pattern hash generation failed")

        extra_for_store = dict(extra or {})
        graph_extractor = "explicit" if extra_for_store.get("entities") else "regex"
        if defer_embed:
            extra_for_store["_memo_embed_pending"] = True
        # Entity extraction (regex, dependency-free, no MLX). Written on EVERY
        # save by default (MEMO_ENTITY_EXTRACT_ON_SAVE) — NOT gated on the
        # retrieval flag — so extra['entities'] exists corpus-wide and
        # _apply_entity_boost / graph expansion work the moment they're enabled,
        # with no backfill. Best-effort: a broken extractor never fails the save.
        if not extra_for_store.get("entities") and flag_bool("MEMO_ENTITY_EXTRACT_ON_SAVE"):
            try:
                from memo.entity_extractor import extract_entities

                ents = extract_entities(f"{title} {content}"[:3000])
                if ents:
                    extra_for_store["entities"] = ents[:20]
            except Exception as exc:
                _log.debug("entity extraction failed during save: %s", exc)

        # Allocate a unique path and create the .md atomically under a lock:
        # `meta.path` is UNIQUE, so two concurrent same-title+date saves probing
        # the same free path would have the loser overwrite the winner's file
        # before its INSERT fails — orphaning the winner's row against the
        # loser's content. The slow embed runs AFTER this block.
        # Write `.md` first — if anything fails after this, the user can recover
        # by re-indexing; writing the index first would point it at a missing file.
        # The topic_key lookup runs under the SAME lock: its result (existing
        # id/path) is consumed by the file write below, so holding the lock
        # across the SELECT closes the TOCTOU vs a concurrent delete.
        with self._data_dir_write_lock():
            # If topic_key provided, check for an existing record with the same
            # topic_key and reuse its id/path (update instead of create).
            if topic_key:
                try:
                    existing = self.store.find_by_topic_key(topic_key)
                    if existing is not None and existing["id"]:
                        use_existing_id = existing["id"]
                        existing_path = existing["path"]
                        _, existing_normalized_hash = self.store.get_dedup_keys(use_existing_id)
                        if normalized_hash is None:
                            normalized_hash = existing_normalized_hash
                        # Preserve original creation date when caller didn't supply one
                        if not created and existing["created"]:
                            created_iso = existing["created"]
                        _log.info(
                            "topic_key upsert: updating existing %s (path=%s)",
                            use_existing_id[:8],
                            existing_path,
                        )
                        # Corroboration (C1): a repeated save with the same
                        # topic_key is a re-assertion of the same fact.
                        bump_support_if_enabled(self.store, [use_existing_id])
                except Exception as exc:
                    _log.debug("topic_key lookup failed: %s", exc)

                # A prior save may have reached Markdown while SQLite was
                # unavailable. Recover that canonical reservation from disk so
                # retrying the same topic cannot create a second memory.
                if use_existing_id is None:
                    for candidate in sorted(self.cfg.memory_dir.rglob("*.md")):
                        try:
                            if candidate.is_symlink():
                                continue
                            candidate_post = frontmatter.loads(
                                candidate.read_text(encoding="utf-8")
                            )
                            candidate_id = str(candidate_post.metadata.get("id") or "")
                            if (
                                candidate_post.metadata.get("topic_key") != topic_key
                                or re.fullmatch(r"[0-9a-f]{32}", candidate_id) is None
                            ):
                                continue
                            candidate_rel = str(candidate.relative_to(self.cfg.memory_dir))
                            self._safe_path_under(self.cfg.memory_dir, candidate_rel)
                            use_existing_id = candidate_id
                            existing_path = candidate_rel
                            if not created and candidate_post.metadata.get("created"):
                                created_iso = str(candidate_post.metadata["created"])
                            if normalized_hash is None:
                                disk_hash = candidate_post.metadata.get("normalized_hash")
                                normalized_hash = str(disk_hash) if disk_hash else None
                            break
                        except (OSError, ValueError):
                            continue

            record_id = use_existing_id or uuid.uuid4().hex

            # Default the world-validity start to learned-time (`created`) when
            # the caller passed no explicit `valid_at`; an explicit value always
            # wins. `invalid_at` stays open (None) until a supersede closes the
            # interval. Both mirror to frontmatter below so markdown stays the
            # source of truth and reindex folds them back. `created_iso` is final
            # here (a topic_key upsert reuses the existing record's created).
            if valid_at is None:
                valid_at = created_iso

            post = frontmatter.Post(
                content,
                id=record_id,
                title=title,
                type=type_,
                tags=norm_tags,
                created=created_iso,
                updated=now_iso,
                valid_at=valid_at,
            )
            post["extra"] = extra_for_store or {}
            # Add verification state (always UNVERIFIED for new saves unless overridden)
            post["verification_state"] = "unverified"
            # verified_at is omitted for new saves (None)
            # invalid_at is omitted while the interval is open (None)
            if invalid_at is not None:
                post["invalid_at"] = invalid_at
            if topic_key is not None:
                post["topic_key"] = topic_key
            if normalized_hash is not None:
                post["normalized_hash"] = normalized_hash

            # For topic key upserts, reuse the existing path instead of creating a new one
            rel_path = (
                existing_path if existing_path else self._build_rel_path(title, now_iso, norm_tags)
            )
            abs_path = self._safe_path_under(self.cfg.memory_dir, rel_path)
            written_disk_text = frontmatter.dumps(post)
            self._atomic_write_text(rel_path, written_disk_text)

            # Publish every topic-key write text-only before releasing the
            # shared lock. This both reserves a new canonical id and makes an
            # existing-topic update coherent immediately: Markdown + FTS point
            # to the new body and the old vector is removed while embedding.
            if topic_key:
                try:
                    self.store.upsert_text_only(
                        id_=record_id,
                        path=rel_path,
                        title=title,
                        type_=type_,
                        tags=norm_tags,
                        created=created_iso,
                        updated=now_iso,
                        body_hash=body_hash,
                        extra=extra_for_store,
                        body_text=content,
                        topic_key=topic_key,
                        normalized_hash=normalized_hash,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                    )
                except Exception as exc:
                    reservation_error = exc

        if reservation_error is not None:
            self._presence_bump_save()
            return self._save_index_pending(
                exc=reservation_error,
                record_id=record_id,
                rel_path=rel_path,
                abs_path=abs_path,
                post=post,
                title=title,
                type_=type_,
                norm_tags=norm_tags,
                created_iso=created_iso,
                now_iso=now_iso,
                body_hash=body_hash,
                content=content,
                extra_for_store=extra_for_store,
                skip_memflow_receipt=skip_memflow_receipt,
                topic_key=topic_key,
                normalized_hash=normalized_hash,
                expected_disk_text=written_disk_text,
                graph_extractor=graph_extractor,
                valid_at=valid_at,
                invalid_at=invalid_at,
            )

        if defer_embed:
            if not topic_key:
                try:
                    self.store.upsert_text_only(
                        id_=record_id,
                        path=rel_path,
                        title=title,
                        type_=type_,
                        tags=norm_tags,
                        created=created_iso,
                        updated=now_iso,
                        body_hash=body_hash,
                        extra=extra_for_store,
                        body_text=content,
                        topic_key=topic_key,
                        normalized_hash=normalized_hash,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                    )
                except Exception as exc:
                    # The .md is already on disk (embed-pending). A failed
                    # text-only reservation must route through the SAME
                    # recovery the topic_key path uses — stamp
                    # `_memo_embed_pending`, best-effort text-only index, log
                    # history + receipt — so the save is never silently lost.
                    self._presence_bump_save()
                    return self._save_index_pending(
                        exc=exc,
                        record_id=record_id,
                        rel_path=rel_path,
                        abs_path=abs_path,
                        post=post,
                        title=title,
                        type_=type_,
                        norm_tags=norm_tags,
                        created_iso=created_iso,
                        now_iso=now_iso,
                        body_hash=body_hash,
                        content=content,
                        extra_for_store=extra_for_store,
                        skip_memflow_receipt=skip_memflow_receipt,
                        topic_key=topic_key,
                        normalized_hash=normalized_hash,
                        expected_disk_text=written_disk_text,
                        graph_extractor=graph_extractor,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                    )
            self._record_graph_entities_from_extra(
                record_id=record_id,
                created_iso=created_iso,
                extra=extra_for_store,
                extractor=graph_extractor,
            )
            _upsert_declared_fact_edges_best_effort(
                self,
                record_id=record_id,
                title=title,
                type_=type_,
                created=created_iso,
                updated=now_iso,
                extra=extra_for_store,
            )
            self.history.log_save(
                ts=now_iso,
                record_id=record_id,
                title=title,
                type_=type_,
                provenance=_extract_provenance(extra_for_store),
            )
            deferred_rec = MemoryRecord(
                id=record_id,
                path=rel_path,
                title=title,
                type=type_,
                tags=norm_tags,
                created=created_iso,
                updated=now_iso,
                body=content,
                extra=extra_for_store,
                valid_at=valid_at,
                invalid_at=invalid_at,
            )
            self._emit_save_receipt(
                deferred_rec,
                deferred=True,
                disabled=skip_memflow_receipt,
            )
            self._presence_bump_save()
            return deferred_rec

        # Embed `title + body`: the title carries the highest-density
        # signal for retrieval ("Astor — OT Report" is a much better
        # match for a query like "occupational therapy report astor"
        # than the body's clinical paragraphs alone). Prepending also
        # protects the title from head-truncation when the body is
        # long — see embedder.py for the truncation rationale.
        #
        # Authority contract: the `.md` is already on disk and IS the source
        # of truth. If embed or the vector upsert fails here, we must NOT
        # leave the memory unrecoverable — we mark it embed-pending on disk
        # (so `memo reindex` re-embeds it) and best-effort index it text-only
        # so BM25 still finds it, then return the record. Never raise past a
        # successful disk write.
        topic_write_superseded = False
        try:
            # Content-addressed embed cache (keyed model+dims+sha256(text)):
            # re-saving identical content — or reverting an edit — reuses the
            # stored vector instead of a fresh forward pass. Same cache the
            # reindex path uses; a model swap changes the key, so a stale
            # vector is never served.
            embedding = self._embed_cached(
                self._compose_for_embed(title, content),
                ctx=f"save id={record_id[:8]}",
            )
            upsert_args: dict[str, Any] = dict(
                id_=record_id,
                path=rel_path,
                title=title,
                type_=type_,
                tags=norm_tags,
                created=created_iso,
                updated=now_iso,
                body_hash=body_hash,
                embedding=embedding,
                extra=extra_for_store,
                body_text=content,
                topic_key=topic_key,
                normalized_hash=normalized_hash,
                valid_at=valid_at,
                invalid_at=invalid_at,
            )
            if topic_key:
                # Embedding is intentionally outside the global file lock.
                # Revalidate the exact canonical bytes before publishing its
                # vector: a newer same-topic writer may have won meanwhile.
                with self._data_dir_write_lock():
                    try:
                        topic_write_superseded = (
                            abs_path.is_symlink()
                            or abs_path.read_text(encoding="utf-8") != written_disk_text
                        )
                    except OSError:
                        topic_write_superseded = True
                    if not topic_write_superseded:
                        self.store.upsert(**upsert_args)
            else:
                self.store.upsert(**upsert_args)

            if not topic_write_superseded:
                self.maybe_emit_chunks(
                    parent_id=record_id,
                    parent_rel=rel_path,
                    title=title,
                    body=content,
                    tags=norm_tags,
                    created=created_iso,
                    updated=now_iso,
                )
                self._record_graph_entities_from_extra(
                    record_id=record_id,
                    created_iso=created_iso,
                    extra=extra_for_store,
                    extractor=graph_extractor,
                )
                _upsert_declared_fact_edges_best_effort(
                    self,
                    record_id=record_id,
                    title=title,
                    type_=type_,
                    created=created_iso,
                    updated=now_iso,
                    extra=extra_for_store,
                )
        except ValueError:
            # A dims/norm validation failure signals a misconfigured embedder
            # or model (e.g. wrong MEMO_EMBEDDER_DIMS) — fail loudly so it isn't
            # masked by silently marking every save embed-pending.
            # Stamp pending marker on the already-written .md so reindex picks it up.
            extra_for_store["_memo_embed_pending"] = True
            post["extra"] = extra_for_store
            with self._data_dir_write_lock():
                should_mark_pending = not topic_key
                if topic_key:
                    try:
                        should_mark_pending = (
                            not abs_path.is_symlink()
                            and abs_path.read_text(encoding="utf-8") == written_disk_text
                        )
                    except OSError:
                        should_mark_pending = False
                if should_mark_pending:
                    self._atomic_write_text(rel_path, frontmatter.dumps(post))
            raise
        except Exception as exc:
            self._presence_bump_save()
            return self._save_index_pending(
                exc=exc,
                record_id=record_id,
                rel_path=rel_path,
                abs_path=abs_path,
                post=post,
                title=title,
                type_=type_,
                norm_tags=norm_tags,
                created_iso=created_iso,
                now_iso=now_iso,
                body_hash=body_hash,
                content=content,
                extra_for_store=extra_for_store,
                skip_memflow_receipt=skip_memflow_receipt,
                topic_key=topic_key,
                normalized_hash=normalized_hash,
                expected_disk_text=written_disk_text,
                graph_extractor=graph_extractor,
                valid_at=valid_at,
                invalid_at=invalid_at,
            )

        if not topic_write_superseded:
            self.history.log_save(
                ts=now_iso,
                record_id=record_id,
                title=title,
                type_=type_,
                provenance=_extract_provenance(extra_for_store),
            )

        rec = MemoryRecord(
            id=record_id,
            path=rel_path,
            title=title,
            type=type_,
            tags=norm_tags,
            created=created_iso,
            updated=now_iso,
            body=content,
            extra=extra_for_store,
            valid_at=valid_at,
            invalid_at=invalid_at,
        )
        if topic_write_superseded:
            _log.info(
                "topic_key upsert %s superseded during embedding; discarding stale vector",
                record_id[:8],
            )
            self._write_gen += 1
            return self.get(record_id) or rec
        self._emit_save_receipt(rec, deferred=False, disabled=skip_memflow_receipt)
        # Cache-tier: write policy (push/dirty) then capacity bound. Both
        # no-op unless MEMO_CACHE_MODE is on. Guarded so a backend hiccup
        # never fails the save itself.
        self._apply_write_policy(rec)
        if self.cache.policy.enabled and self.cache.policy.max_entries > 0:
            try:
                self.cache.evict_if_needed()
            except Exception as exc:
                _log.warning("cache eviction skipped after save: %s", exc)
        # Crossref edges (flag-gated, default off): index [[wikilinks]] and
        # typed '- relation [[target]]' lines so reverse traversal can warn
        # before cascade-affecting supersede/delete. Best-effort — a crossref
        # failure must never fail the save. Not on the recall-hook path.
        from memo.flags import flag_bool as _flag_bool

        if _flag_bool("MEMO_CROSSREF_INDEX"):
            try:
                self.crossref.index_source(record_id, content)
            except Exception as exc:
                _log.debug("save(%s): crossref index skipped — %s", record_id[:8], exc)
        self._presence_bump_save()
        self._write_gen += 1
        return rec

    def _save_index_pending(
        self,
        *,
        exc: Exception,
        record_id: str,
        rel_path: str,
        abs_path: Path,
        post: frontmatter.Post,
        title: str,
        type_: str,
        norm_tags: list[str],
        created_iso: str,
        now_iso: str,
        body_hash: str,
        content: str,
        extra_for_store: dict[str, Any],
        skip_memflow_receipt: bool,
        topic_key: str | None = None,
        normalized_hash: str | None = None,
        expected_disk_text: str | None = None,
        graph_extractor: str = "regex",
        valid_at: str | None = None,
        invalid_at: str | None = None,
    ) -> MemoryRecord:
        """Recovery path when indexing fails AFTER the canonical `.md` is on disk.

        markdown-is-truth: a save that reached disk must never be lost just
        because the embedder or vec upsert hiccuped. We (1) stamp
        `_memo_embed_pending` into the on-disk frontmatter so `memo reindex`
        re-embeds it, (2) best-effort index it text-only so BM25 still surfaces
        it in the meantime, and (3) return the record. The exception is logged,
        not raised.
        """
        superseded = False
        with self._data_dir_write_lock():
            if topic_key and expected_disk_text is not None:
                try:
                    superseded = (
                        abs_path.is_symlink()
                        or abs_path.read_text(encoding="utf-8") != expected_disk_text
                    )
                except OSError:
                    superseded = True
            if not superseded:
                _log.warning(
                    "save: indexing failed after .md write (id=%s, path=%s) — marking "
                    "embed-pending for `memo reindex` to replay: %s",
                    record_id[:8],
                    rel_path,
                    exc,
                )
                extra_for_store["_memo_embed_pending"] = True
                post["extra"] = extra_for_store
                # Marker write and text-only index publication are one critical
                # section for topic-key writes. A newer writer cannot land
                # between the compare and either recovery mutation.
                with contextlib.suppress(Exception):
                    self._atomic_write_text(rel_path, frontmatter.dumps(post))
                with contextlib.suppress(Exception):
                    self.store.upsert_text_only(
                        id_=record_id,
                        path=rel_path,
                        title=title,
                        type_=type_,
                        tags=norm_tags,
                        created=created_iso,
                        updated=now_iso,
                        body_hash=body_hash,
                        extra=extra_for_store,
                        body_text=content,
                        topic_key=topic_key,
                        normalized_hash=normalized_hash,
                        valid_at=valid_at,
                        invalid_at=invalid_at,
                    )
        if superseded:
            _log.info(
                "topic_key upsert %s was superseded after an embed failure; "
                "leaving the newer canonical write untouched",
                record_id[:8],
            )
            current = self.get(record_id)
            if current is not None:
                self._write_gen += 1
                return current
            # Defensive fallback: the canonical file won the comparison but
            # the derived index is temporarily unavailable.
            rec = MemoryRecord(
                id=record_id,
                path=rel_path,
                title=title,
                type=type_,
                tags=norm_tags,
                created=created_iso,
                updated=now_iso,
                body=content,
                extra=extra_for_store,
                valid_at=valid_at,
                invalid_at=invalid_at,
            )
            self._write_gen += 1
            return rec

        # Best-effort graph projection after the canonical Markdown + FTS
        # recovery has been atomically published above.
        with contextlib.suppress(Exception):
            self._record_graph_entities_from_extra(
                record_id=record_id,
                created_iso=created_iso,
                extra=extra_for_store,
                extractor=graph_extractor,
            )
            _upsert_declared_fact_edges_best_effort(
                self,
                record_id=record_id,
                title=title,
                type_=type_,
                created=created_iso,
                updated=now_iso,
                extra=extra_for_store,
            )
        with contextlib.suppress(Exception):
            self.history.log_save(
                ts=now_iso,
                record_id=record_id,
                title=title,
                type_=type_,
                provenance=_extract_provenance(extra_for_store),
            )
        rec = MemoryRecord(
            id=record_id,
            path=rel_path,
            title=title,
            type=type_,
            tags=norm_tags,
            created=created_iso,
            updated=now_iso,
            body=content,
            extra=extra_for_store,
            valid_at=valid_at,
            invalid_at=invalid_at,
        )
        self._emit_save_receipt(rec, deferred=True, disabled=skip_memflow_receipt)
        self._write_gen += 1
        return rec

    def _emit_save_receipt(
        self,
        rec: MemoryRecord,
        *,
        deferred: bool,
        disabled: bool,
    ) -> None:
        """Fire-and-forget memflow receipt for a successful save.

        No-op unless `MEMO_EMIT_RECEIPTS=1`; never raises. `disabled`
        is the synapse-originated opt-out (synapse keeps its own ledger).
        """
        from memo.receipts import emit_receipt

        prov = _extract_provenance(rec.extra or {})
        emit_receipt(
            "save",
            text=f"Memo saved memory {rec.id[:8]} ({rec.type}): {rec.title}",
            meta={
                "id": rec.id,
                "type": rec.type,
                "tags": ",".join(rec.tags),
                "path": rec.path,
                "deferred": deferred,
                "synapse_trace_id": prov.get("synapse_trace_id", ""),
                "synapse_route_reason": prov.get("synapse_route_reason", ""),
                "synapse_agent_id": prov.get("synapse_agent_id", ""),
            },
            disabled=disabled,
        )
        # M2b: also emit to the unified trinity ledger (best-effort,
        # independent of the memflow receipt path).
        self._emit_ledger("save", rec, prov, deferred=deferred)

    def _emit_ledger(
        self,
        op: str,
        rec: MemoryRecord,
        prov: dict[str, Any] | None = None,
        *,
        deferred: bool = False,
    ) -> None:
        """Fire-and-forget ConsciousnessEvent for the unified ledger (M2)."""
        from memo.consciousness_ledger import emit_event

        emit_event(
            op,
            subject_uri=f"memo://memoria/{rec.id}",
            trace_id=(prov or {}).get("synapse_trace_id", "") or "",
            actor=(prov or {}).get("synapse_agent_id") or "memo",
            payload={
                "id": rec.id,
                "type": rec.type,
                "title": rec.title or "",
                "tags": list(rec.tags or []),
                "deferred": deferred,
            },
        )

    def _apply_write_policy(self, rec: MemoryRecord) -> None:
        """Honor MEMO_CACHE_MODE on a fresh save:

        - write_through: push to the backing store now; if the push fails,
          mark the entry dirty so a later flush retries (no lost write).
        - write_back: mark dirty; the push happens on flush / before eviction.
        - read_through / off: nothing (the local store is authoritative).
        Guarded — a backend failure never fails the save.
        """
        policy = self.cache.policy
        if not policy.enabled:
            return
        # A read-through fill already mirrors the backing store — pushing it
        # back (write_through) or marking it dirty (write_back) would be a
        # redundant round-trip, so skip the write policy for those saves.
        if (rec.extra or {}).get("source") == "memo-cache-fill":
            return
        try:
            if policy.write_through:
                backend = self._cache_backend()
                if backend is None or not backend.push(rec):
                    self._mark_dirty(rec.id)
            elif policy.write_back:
                self._mark_dirty(rec.id)
        except Exception as exc:
            _log.warning("cache write policy skipped for %s: %s", rec.id[:8], exc)

    def _absorb_into_existing(
        self, existing_id: str, new_title: str, new_content: str
    ) -> MemoryRecord | None:
        """C3 absorb-on-recurrence: fold a near-duplicate save into the
        EXISTING record. One bounded LLM call rewrites the existing body,
        applied via the versioned update() (pre-update snapshot →
        `memo version rollback` works) with proof_count in extra.

        Returns the updated record, or None on ANY failure — the caller
        falls back to today's warn-and-create, so a save is never lost.
        Runs on save paths only (Stop-hook capture / MCP / dream), never
        the 5s recall hook (which performs no saves)."""
        try:
            existing = self.get(existing_id)
            if existing is None:
                return None
            from memo.flags import flag_int
            from memo.memory.prompts import _ABSORB_SYSTEM_PROMPT
            from memo.memory.record import chat_with_timeout

            chat = self._ensure_chat()
            user = (
                f"EXISTING NOTE (title: {existing.title}):\n{existing.body}\n\n"
                f"NEW OBSERVATION (title: {new_title}):\n{new_content}\n\n"
                "Return ONLY the merged note body."
            )
            timeout_flag = flag_int("MEMO_CONSOLIDATE_TIMEOUT")
            out = chat_with_timeout(
                chat,
                timeout=float(180 if timeout_flag is None else timeout_flag),
                model=self.cfg.llm_model,
                messages=[
                    {"role": "system", "content": _ABSORB_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                options={"temperature": 0.0, "max_tokens": 1024, "thinking": False},
            )
            if out is None:
                return None
            merged = ((out.get("message") or {}).get("content") or "").strip()
            if not merged:
                return None
            new_extra = dict(existing.extra or {})
            new_extra["proof_count"] = int(new_extra.get("proof_count") or 1) + 1
            new_extra["last_absorbed_at"] = _now_iso()
            updated = self.update(existing_id, content=merged, extra=new_extra)
            if updated is not None:
                _log.info(
                    "save: absorbed near-duplicate into %s (proof_count=%d)",
                    existing_id[:8],
                    new_extra["proof_count"],
                )
            return updated  # type: ignore[no-any-return]
        except Exception as exc:
            _log.warning("save: absorb failed (%s) — falling back to new record", exc)
            return None

    # -- internals ----------------------------------------------------------

    def _build_rel_path(self, title: str, now_iso: str, tags: list[str] | None = None) -> str:
        date = now_iso.split("T", 1)[0]
        slug = _slugify(title)[:80] or "untitled"
        # Per-project bucket folder, derived from the project: tag. The sqlite
        # index globs recursively, so this is on-disk organization only — search
        # stays global. Gated; flat and foldered layouts coexist.
        prefix = ""
        if flag_bool("MEMO_STORE_BY_PROJECT"):
            from memo.project import project_bucket

            prefix = f"{project_bucket(tags or [])}/"
        # POSIX path joins. Path is relative to `cfg.memory_dir`.
        base = f"{prefix}{date}-{slug}"
        candidate = f"{base}.md"
        # `meta.path` is UNIQUE. Two saves with the same title on the same day
        # would collide. Append a numeric suffix until the path is free —
        # checking both the index and the on-disk file (per-bucket).
        # include_deleted=True: soft-delete tombstones still occupy the
        # UNIQUE(path) index; without this the allocator silently hands out a
        # path that will collide at INSERT time, and the save degrades to a
        # pendng-only write (invisible to search, not recoverable via reindex).
        n = 2
        while (
            self.store.get_by_path(candidate, include_deleted=True) is not None
            or (self.cfg.memory_dir / candidate).exists()
        ):
            candidate = f"{base}-{n}.md"
            n += 1
        return candidate

    def _safe_path_under(self, root: Path, rel_path: str) -> Path:
        """Resolve a store-provided relative path without following symlinks.

        Store rows are rebuildable and may come from an old/corrupt/imported DB,
        so they are not trusted as filesystem capabilities.  Every component
        below ``root`` must be a real path component (not a symlink), and the
        resolved target must remain inside the configured root.
        """
        from memo.errors import StorageError

        relative = Path(rel_path)
        if not rel_path or relative.is_absolute() or ".." in relative.parts:
            raise StorageError(f"refusing store path outside memory_dir: {rel_path!r}")

        # Use CodeQL's recognized normalize-then-prefix validation pattern.
        # ``realpath`` also resolves every existing symlink component before
        # the path reaches a filesystem sink. The separator suffix prevents
        # sibling-prefix confusion (``/safe/root-evil`` vs ``/safe/root``).
        safe_root = os.path.realpath(root)
        candidate = os.path.realpath(os.path.join(safe_root, rel_path))
        root_prefix = safe_root.rstrip(os.sep) + os.sep
        if not candidate.startswith(root_prefix):
            raise StorageError(f"refusing store path outside memory_dir: {rel_path!r}")
        cursor = Path(safe_root)
        for part in relative.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise StorageError(f"refusing symlink in canonical memory path: {rel_path!r}")
        return Path(candidate)

    def _resolve_existing(self, rel_path: str) -> Path:
        """Resolve a DB-stored path to an absolute `Path`.

        Tries `memory_dir / rel_path` first (current layout). Falls back
        to `vault_path / rel_path` if `vault_path` is set AND the file
        actually exists there (legacy layout: paths in older DB rows
        carry a `<memory_subdir>/...` prefix relative to `vault_path`).

        Returns the new-layout path even when the file doesn't exist on
        either branch — callers that need to CREATE a file always write
        to the new layout.
        """
        new_path = self._safe_path_under(self.cfg.memory_dir, rel_path)
        if new_path.is_file():
            return new_path
        if self.cfg.vault_path is not None:
            legacy = self._safe_path_under(self.cfg.vault_path, rel_path)
            if legacy.is_file():
                return legacy
        return new_path

    def _read_body(self, rel_path: str) -> str:
        # Fast path: curated memories live on disk under memory_dir / vault_path.
        abs_path = self._resolve_existing(rel_path)
        if abs_path.is_file():
            try:
                text = abs_path.read_text(encoding="utf-8")
                post = frontmatter.loads(text)
                return markdown_body(post)
            except (OSError, UnicodeDecodeError) as exc:
                _log.debug("_read_body: disk read failed for %s: %s", rel_path, exc)
        # Fallback: vault-ingest rows (e.g. `notes/01-Projects/Foo.md`,
        # `work/.../bar.md#chunk-3`) don't resolve to disk via `memory_dir`
        # because the label-prefixed path lives outside `data_dir`. The
        # body was written into the FTS table at ingest time — read it
        # from there so retrieval surfaces real snippets instead of "".
        try:
            body = str(self.store.get_fts_body_by_path(rel_path) or "")
            if body:
                return body
        except Exception:
            _log.warning("read_body_fallback: DB error for %s", rel_path, exc_info=True)
        return ""
