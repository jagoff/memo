from __future__ import annotations

from typing import Annotated, Any

from fastmcp import Context
from pydantic import Field

from memo.errors import IdentityConflictError
from memo.mcp_budget import bounded_list
from memo.memory import AmbiguousIdError, Memory
from memo.server_annotations import (
    DESTRUCTIVE,
    READ_ONLY,
    WRITE,
    WRITE_IDEMPOTENT,
    annotated_tool,
)

_PROTECTED_MCP_EXTRA = {
    "federation",
    "learning",
    "outcome_stats",
    "owner_principal",
    "principals",
    "priority",
    "trust_tier",
    "visibility",
    "write_policy",
}


def _safe_mcp_extra(extra: dict[str, Any] | None) -> dict[str, Any]:
    """Drop authority-controlled metadata from caller-supplied MCP extras."""
    safe_extra = dict(extra or {})
    for key in _PROTECTED_MCP_EXTRA:
        safe_extra.pop(key, None)
    return safe_extra


def _record_ledger_recovery(memory: Memory, memory_id: str, record: dict[str, Any]) -> None:
    """Count a `memo_get` against the emission ledger's net-saving estimate
    when `memory_id` was actually DIGESTED earlier in this session -- rendered
    as an `already_in_context` pointer some caller had to follow up on, per the
    design doc's definition of `memo_get_after_digest`.

    Deliberately narrower than "has any entry in this session's ledger":
    `emitted_ledger.read()` also holds ids the recall hook merely injected, or
    a prior MCP call sent in FULL (never suppressed to a pointer) -- a
    `memo_get` on either of those is an ordinary first read, not a recovery
    from a digest, and counting it would inflate `memo_get_after_digest` by
    construction (over the numerator of the very ratio that decides whether to
    promote the feature). `emitted_ledger.digested_ids()` answers the narrower
    question this counter actually needs.

    Conservative attribution, not proof of cause: per the spec, ANY memo_get
    on a previously-DIGESTED id counts against the feature, including one the
    model would have issued anyway (e.g. to double-check exact current
    content) rather than specifically because it followed a digest
    `{id, title, ref}` pointer. This module has no record of WHY a memo_get
    happened, only that the id was digested earlier -- see task-8-report.md
    for what this does and does not prove.

    Fail-open, and on its own try/except separate from `memo_get`'s own
    logic: a counter failure must never affect the record this tool actually
    returns.
    """
    try:
        from memo.flags import flag_bool

        if not flag_bool("MEMO_EMITTED_LEDGER"):
            return

        import json

        from memo import emitted_ledger as el
        from memo.mcp_budget import est_tokens
        from memo.server_common import stage_counters
        from memo.server_session_patterns import _effective_session_id

        state_dir = memory.cfg.state_dir
        session_id = _effective_session_id()
        if memory_id not in el.digested_ids(state_dir, session_id):
            return
        # F1 (task-8 review): charge the whole record `memo_get` actually
        # returns (the full `rec.to_dict()` -- id/path/title/type/tags/
        # timestamps/extra/... -- not just its `body` field), matching the
        # same whole-row basis `apply_ledger`'s tokens_suppressed now uses.
        # A recovery costs the caller the entire returned payload, not one
        # field of it.
        stage_counters(
            state_dir,
            session_id,
            get_after_digest=1,
            tokens_recovered=est_tokens(json.dumps(record, separators=(",", ":"), default=str)),
        )
    except Exception:
        return


def _bounded_lint(report: dict[str, list[dict[str, Any]]], *, limit: int) -> dict[str, Any]:
    """Trim each lint category to `limit` findings and keep the true totals.

    The full report grows with the corpus, not with what a caller can read: on
    a 11k-memory index it serialized to 725k chars, `few_tags` alone 3,962
    entries, so an unbounded return is truncated by the client and the backlog
    size is lost with it. `counts` reports the real per-category size.
    """
    cap = max(0, limit)
    bounded: dict[str, Any] = {cat: rows[:cap] for cat, rows in report.items()}
    bounded["counts"] = {cat: len(rows) for cat, rows in report.items()}
    bounded["limit"] = cap
    return bounded


def _bounded_proposal(proposal: Any, *, member_cap: int) -> Any:
    """Bound a merge proposal's two id lists, keeping the real totals.

    `memory_ids` is the FULL membership of the cluster the proposal covers and
    `archived_ids` is that same set minus the surviving memory -- neither is
    capped by `max_clusters`, so both track the corpus directly. Measured on
    the conformance corpus: 100 ids per proposal at N=2000 and 500 at N=10001,
    which is what carried the tool from 14.6k tokens to 74k and would have
    carried it past 200k at 10k memories.

    The `shown`/`total`/`truncated` metadata is name-prefixed rather than
    splatted bare (as `bounded_list`'s callers elsewhere do) because a
    proposal bounds TWO lists: one bare splat would silently overwrite the
    other's totals.
    """
    if not isinstance(proposal, dict):
        return proposal
    bounded = dict(proposal)
    for field in ("memory_ids", "archived_ids"):
        ids = proposal.get(field)
        if not isinstance(ids, list):
            continue
        kept, meta = bounded_list(ids, cap=member_cap)
        bounded[field] = kept
        bounded.update({f"{field}_{k}": v for k, v in meta.items()})
    return bounded


def _bounded_consolidate(
    out: dict[str, Any], *, cluster_limit: int, member_limit: int
) -> dict[str, Any]:
    """Bound every dimension of `consolidate_all`'s output that tracks the corpus.

    1. Each cluster's `members` list -- a near-duplicate cluster's true size
       tracks the corpus, not `max_clusters`: on a 10k-memory conformance
       corpus (same-topic memories cluster by design) a single cluster held
       every memory in its topic, each member carrying a 600-char body
       preview.
    2. The `clusters` list itself -- `consolidate_all` runs a fast lane
       (cosine >= auto_threshold) and a normal pass (cosine >= threshold) and
       concatenates both, so the true count can run to 2x `max_clusters`
       despite the docstring's "maximum clusters to process" promise; this
       makes the return actually honour that cap.
    3. The `proposals` list and, inside each proposal, the `memory_ids` /
       `archived_ids` lists -- the same two defects one level down. The list
       is the same 2x-`max_clusters` concatenation; the id lists are pure
       corpus scale (see `_bounded_proposal`). Bounding (1) and (2) alone left
       this dimension unbounded, which is why the tool still measured 74,155
       tokens at N=2000 after the first pass at this helper.

    `results` is deliberately not bounded: it is populated only when
    `auto_apply=True`, and this MCP tool always calls `consolidate_all` with
    `auto_apply=False`, so it is always empty on this path.
    """
    clusters = out.get("clusters")
    if not isinstance(clusters, list):
        return out
    member_cap = max(0, member_limit)
    cluster_cap = max(0, cluster_limit)
    trimmed = []
    for cluster in clusters:
        members = cluster.get("members") if isinstance(cluster, dict) else None
        if isinstance(members, list) and len(members) > member_cap:
            kept, meta = bounded_list(members, cap=member_cap)
            cluster = {**cluster, "members": kept, **meta}
        trimmed.append(cluster)

    def _by_size_desc(cluster: Any) -> int:
        return -int(cluster.get("size") or 0) if isinstance(cluster, dict) else 0

    kept_clusters, cluster_meta = bounded_list(trimmed, cap=cluster_cap, key=_by_size_desc)
    result = dict(out)
    result["clusters"] = kept_clusters
    result.update(cluster_meta)

    proposals = out.get("proposals")
    if isinstance(proposals, list):
        kept_proposals, proposal_meta = bounded_list(
            [_bounded_proposal(p, member_cap=member_cap) for p in proposals], cap=cluster_cap
        )
        result["proposals"] = kept_proposals
        result.update({f"proposals_{k}": v for k, v in proposal_meta.items()})
    return result


def _mark_partial(counts: dict[str, int]) -> dict[str, int]:
    """Flag a reindex that could not index everything it was handed.

    `errors` is the subset of `skipped` that FAILED (parse error, embed
    failure, refused path) rather than being deliberately passed over.
    """
    if counts.get("errors"):
        counts["partial"] = 1
    return counts


def register(server: Any, memory: Memory) -> None:
    @annotated_tool(server, **WRITE)
    def memo_save(
        content: Annotated[
            str,
            Field(description="Markdown body to persist. Must be non-empty."),
        ],
        title: Annotated[
            str | None,
            Field(
                description="Optional title; when omitted it is derived from the first line of content."
            ),
        ] = None,
        type: Annotated[
            str,
            Field(
                description="Memory type. One of: decision, fact, bug, feedback, preference, "
                "note, manual, synthesis, procedure, failure_pattern, reference, temp."
            ),
        ] = "note",
        tags: Annotated[
            list[str] | None,
            Field(description="Tags to attach; lower-cased and de-duplicated."),
        ] = None,
        auto_derive: Annotated[
            bool,
            Field(
                description="When true, a helper LLM fills missing metadata (title/type/tags); "
                "adds ~0.5-2s per save."
            ),
        ] = False,
        extract: Annotated[
            bool | None,
            Field(
                description="Decompose content into atomic facts saved individually; "
                "None defers to the MEMO_SAVE_EXTRACT flag (off by default)."
            ),
        ] = None,
        extra: Annotated[
            dict[str, Any] | None,
            Field(
                description="Arbitrary JSON metadata bag stored with the record. "
                "Authority-controlled keys (write_policy, visibility, trust_tier, ...) are stripped."
            ),
        ] = None,
        scope: Annotated[
            str | None,
            Field(
                description="'global' skips the auto project:<repo> tag (global recall tier); "
                "'project' or None keep auto-detection. Other values are rejected."
            ),
        ] = None,
        defer_embed: Annotated[
            bool,
            Field(
                description="Persist markdown and the text index immediately, but leave "
                "the semantic vector pending for a later memo_reindex call."
            ),
        ] = False,
    ) -> dict[str, Any]:
        """Persist `content` to memo.

        Use memo_save for a durable curated fact, decision, or preference;
        use memo_offload for bulk dumps of working context.

        When `extract` is true (defaults to the `MEMO_SAVE_EXTRACT` flag, off),
        the helper LLM decomposes `content` into atomic facts and saves each as
        its own memory (mem0 ADD-model) instead of one opaque blob; `tags`
        propagate to every fact. Returns an extraction summary
        (`status`, `saved` ids, `saved_titles`, counts) rather than a single
        record. If nothing extractable is found, the blob is saved verbatim.
        Normal saves add `action` (`created`, `corroborated`, or `revised`) and
        `index_pending` so callers can distinguish evidence from a new record.

        `scope` controls the auto `project:<repo>` tag for THIS call only:
        `"global"` skips it (the memory lands untagged → the global recall
        tier, +0.10 boost everywhere); `"project"` or None keep the default
        auto-detection. An explicit `project:` tag in `tags` always wins
        either way.

        `defer_embed` mirrors CLI `memo save --defer-embed`: it persists
        markdown + BM25 immediately and marks the semantic vector pending for
        `memo_reindex`. Extraction mode ignores it, matching the CLI.
        """
        from memo.flags import flag_bool
        from memo.memory import WriteRefused

        if scope not in (None, "project", "global"):
            return {
                "error": "invalid_scope",
                "message": f"scope must be 'project' or 'global', got {scope!r}",
            }
        auto_project = scope != "global"
        safe_extra = _safe_mcp_extra(extra)

        if extract is None:
            extract = flag_bool("MEMO_SAVE_EXTRACT")
        if extract:
            from memo.capture import extract_and_save_text

            return extract_and_save_text(
                memory,
                memory.cfg,
                content,
                merge_tags=tags,
                title=title,
                type_=type,
                auto_project=auto_project,
            )

        try:
            rec = memory.save(
                content=content,
                title=title,
                type_=type,
                tags=tags,
                auto_derive=auto_derive,
                auto_project=auto_project,
                extra=safe_extra,
                defer_embed=defer_embed,
            )
        except WriteRefused as exc:
            return {
                "status": "refused",
                "conflict": exc.conflict,
                "message": str(exc),
            }
        except IdentityConflictError as exc:
            return {
                "status": "conflict",
                "error": "identity_conflict",
                "kind": exc.kind,
                "conflicting_ids": [
                    str(item.get("id")) for item in exc.conflicts if item.get("id")
                ],
                "message": str(exc),
            }
        return rec.to_dict()

    @annotated_tool(server, **READ_ONLY)
    def memo_list(
        limit: Annotated[
            int,
            Field(description="Maximum records to return, newest by `updated` first."),
        ] = 20,
        type: Annotated[
            str | None,
            Field(
                description="Filter to one memory type (e.g. 'decision', 'fact'); "
                "None lists every type."
            ),
        ] = None,
    ) -> list[dict[str, Any]]:
        """List recent memories, optionally filtered by memory type.

        Read-only. Use this to browse the corpus before choosing an id for
        memo_get, memo_update, memo_rename, memo_delete, or history tools.
        `limit` caps the number of returned records.
        """
        return [r.to_dict() for r in memory.list(limit=limit, type_=type)]

    @annotated_tool(server, **READ_ONLY)
    def memo_get(
        id: Annotated[
            str,
            Field(description="Full 32-char memory id or a unique prefix (git-style short ids)."),
        ],
    ) -> dict[str, Any] | None:
        """Fetch one memory by id or unique id prefix.

        Read-only. Returns the full memory record, `None` when it does not
        exist, or an ambiguity error when the prefix matches multiple records.
        Use memo_search or memo_list first when you do not know the id.

        When MEMO_EMITTED_LEDGER is on and the resolved id already has an
        entry in this session's emission ledger, this call counts as a
        recovery against the feature's net-saving estimate (see
        `memo_cache_stats`'s `emit_ledger.memo_get_after_digest`) -- the
        conservative rule the spec calls for, not proof this call actually
        followed a digest pointer.
        """
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if not rec:
            return None
        out = rec.to_dict()
        _record_ledger_recovery(memory, rec.id, out)
        return out

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_update(
        id: Annotated[
            str,
            Field(description="Id or unique id prefix of the memory to patch."),
        ],
        title: Annotated[
            str | None,
            Field(description="New title; None leaves the title unchanged."),
        ] = None,
        type: Annotated[
            str | None,
            Field(
                description="New memory type (must be a valid type, e.g. 'decision', 'fact'); "
                "None leaves it unchanged."
            ),
        ] = None,
        tags: Annotated[
            list[str] | None,
            Field(
                description="Full replacement tag list (lower-cased, de-duplicated); "
                "None leaves tags unchanged."
            ),
        ] = None,
        content: Annotated[
            str | None,
            Field(
                description="Full replacement body. Mutually exclusive with "
                "replace_old/replace_new and append."
            ),
        ] = None,
        replace_old: Annotated[
            str | None,
            Field(
                description="Exact string to find in the body; must occur exactly once. "
                "Pass together with replace_new."
            ),
        ] = None,
        replace_new: Annotated[
            str | None,
            Field(description="Replacement text for replace_old. Pass together with replace_old."),
        ] = None,
        append: Annotated[
            str | None,
            Field(
                description="Paragraph appended to the end of the body. Mutually exclusive "
                "with content and replace_old/replace_new."
            ),
        ] = None,
    ) -> dict[str, Any] | None:
        """Patch fields on a memory. `content` replaces the whole body;
        `replace_old`+`replace_new` is a surgical exact-string edit (old must
        occur exactly once — unchanged text stays byte-identical); `append`
        adds a paragraph. All three are versioned (memo_version_rollback).
        """
        if (replace_old is None) != (replace_new is None):
            return {
                "error": "replace_incomplete",
                "message": "pass replace_old and replace_new together",
            }
        replace = (
            (replace_old, replace_new)
            if replace_old is not None and replace_new is not None
            else None
        )
        try:
            rec = memory.update(
                id,
                title=title,
                type_=type,
                tags=tags,
                content=content,
                replace=replace,
                append=append,
            )
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        except IdentityConflictError as exc:
            return {
                "error": "identity_conflict",
                "kind": exc.kind,
                "conflicting_ids": [
                    str(item.get("id")) for item in exc.conflicts if item.get("id")
                ],
                "message": str(exc),
            }
        except ValueError as exc:
            return {"error": "edit_failed", "message": str(exc)}
        return rec.to_dict() if rec else None

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_rename(
        title: Annotated[
            str,
            Field(description="New title for the memory."),
        ],
        id: Annotated[
            str | None,
            Field(
                description="Id or unique prefix of the memory to rename; when omitted, "
                "targets the most recent save made on this device."
            ),
        ] = None,
    ) -> dict[str, Any] | None:
        """Rename one memory title without changing its body or tags.

        Destructive metadata edit. Use after memo_save or memo_search when a
        record has the right content but the wrong title. Pass `id` for a
        specific memory; omit it only immediately after a save, when memo can
        target the most recent local save. Use memo_update instead when you
        need to edit content, type, or tags.
        """
        target = id or memory.last_saved_id()
        if target is None:
            return {
                "error": "no_recent_save",
                "message": "no recent save on this machine; pass `id` explicitly",
            }
        try:
            rec = memory.update(target, title=title)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        return rec.to_dict() if rec else None

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_reindex(
        force: Annotated[
            bool,
            Field(
                description="When true, re-embed every indexed entry even when the on-disk "
                "body is unchanged (e.g. after an embedder model swap)."
            ),
        ] = False,
    ) -> dict[str, int]:
        """Rebuild memo's searchable index from the markdown vault.

        Writes only derived index state; markdown remains the source of truth.
        Use after hand-editing vault files or changing indexing behavior.
        `force` reprocesses records even if memo thinks they are current.

        `errors` counts files that FAILED to index (parse error, embed failure,
        refused path) — a subset of `skipped`, which also counts deliberate
        skips (archives, chronicle, secrets). A non-zero `errors` means the
        index is incomplete: fix the cause and re-run.
        """
        return _mark_partial(memory.reindex(force=force))

    @annotated_tool(server, **DESTRUCTIVE)
    async def memo_delete(
        id: Annotated[
            str,
            Field(
                description="Id or unique prefix of the memory to delete permanently "
                "(markdown file + index rows)."
            ),
        ],
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Permanently delete one memory by id or unique prefix.

        Destructive and irreversible (no trash). Elicitation-capable clients
        are asked to confirm before the delete runs; other clients proceed
        unchanged. Resolves ambiguous short ids safely and returns an error
        instead of guessing. When cross-reference indexing is enabled, the
        response warns about memories that linked to the deleted record.
        """
        from memo.flags import flag_bool
        from memo.server_elicit import abort_result, confirm_destructive, sanitize_fragment

        read_failed = False
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        except Exception:
            # "Could not read" is NOT "does not exist" — see the gate below.
            rec = None
            read_failed = True
        referenced_by: list[str] = []
        if rec is not None and flag_bool("MEMO_CROSSREF_INDEX"):
            try:
                referenced_by = [b.source_id for b in memory.crossref.referencing_sources(rec.id)]
            except Exception:
                referenced_by = []
        # Confirm whenever there may be something to delete. A genuinely
        # missing id (`get` returned None) is a no-op and keeps skipping the
        # gate, but a FAILED read (locked DB, transient storage error) used to
        # take the same branch and delete irreversibly with no prompt at all.
        # Titles are untrusted (auto-capture, LLM derivation) — sanitize so a
        # hostile title can't rewrite the confirmation prompt.
        target = (
            f"'{sanitize_fragment(rec.title)}' ({rec.type})"
            if rec is not None
            else f"id={sanitize_fragment(id)} (could not be read before deleting)"
        )
        linked = (
            f" {len(referenced_by)} memories link to it; their typed edges will dangle."
            if referenced_by
            else ""
        )
        if rec is not None or read_failed:
            gate = await confirm_destructive(
                ctx,
                action="delete",
                detail=(
                    f"Permanently delete {target}?{linked} No trash — recovery "
                    "only via backup / git-sync / versions."
                ),
            )
            if not gate.proceed:
                return abort_result(
                    gate,
                    memory,
                    tool="memo_delete",
                    action="delete",
                    target=f"{target} id={rec.id if rec is not None else id}",
                )
        try:
            out: dict[str, Any] = {"deleted": memory.delete(id)}
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if referenced_by:
            out["cascade_warning"] = (
                f"{len(referenced_by)} memories link to the deleted id; "
                "their typed edges now dangle"
            )
            out["referenced_by"] = referenced_by
        return out

    @annotated_tool(server, **DESTRUCTIVE)
    def memo_forget(
        id: Annotated[
            str,
            Field(description="Id or unique prefix of the memory to hide from recall."),
        ],
        reason: Annotated[
            str | None,
            Field(
                description="Optional free-text reason recorded in the memory's metadata "
                "as forget_reason."
            ),
        ] = None,
    ) -> dict[str, Any]:
        """Mark one memory as forgotten without deleting its history.

        Destructive in retrieval behavior: the memory is hidden from normal
        recall/search surfaces until memo_unforget restores it. Pass `reason`
        to record why the memory should no longer be surfaced.
        """
        try:
            rec = memory.forget(id, reason=reason)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"forgotten": False}
        return {"forgotten": True, "id": rec.id}

    @annotated_tool(server, **WRITE_IDEMPOTENT)
    def memo_unforget(
        id: Annotated[
            str,
            Field(description="Id or unique prefix of the forgotten memory to restore."),
        ],
    ) -> dict[str, Any]:
        """Restore a previously forgotten memory to normal retrieval.

        Idempotent write. Accepts a full id or unique prefix and returns
        whether a matching forgotten record was restored. Use after deciding a
        memory should participate in search and recall again.
        """
        try:
            rec = memory.unforget(id)
        except AmbiguousIdError as exc:
            return {"error": "ambiguous", "prefix": exc.prefix, "matches": exc.matches}
        if rec is None:
            return {"unforgotten": False}
        return {"unforgotten": True, "id": rec.id}

    @annotated_tool(server, **READ_ONLY)
    async def memo_consolidate(
        threshold: float = 0.85,
        max_clusters: Annotated[
            int,
            Field(
                description="Clusters to process and return. Each one costs a "
                "member list plus a merge proposal, so this is the dominant term "
                "in the response size."
            ),
        ] = 10,
        type: str | None = None,
        member_limit: Annotated[
            int,
            Field(
                description="Sample members returned per cluster. A cluster's true "
                "size always comes back in `total`; `truncated` says whether any "
                "members were dropped."
            ),
        ] = 2,
        ctx: Context | None = None,
    ) -> dict[str, Any]:
        """Detect near-duplicate clusters and propose merges.

        Read-only — returns proposals without modifying the corpus.
        Uses the AdvancedConsolidator under the hood (same as
        ``memo_consolidate_list_archived``). With client sampling enabled,
        merge synthesis runs on the calling model up to
        MEMO_SAMPLING_MAX_CALLS (see `synthesizer` field). `max_clusters`
        bounds how many clusters come back; a single cluster's member list
        is bounded separately by `member_limit` -- same-topic memories
        cluster by design, so one cluster can hold most of the corpus.

        Both defaults are derived from MEMO_MCP_RESPONSE_BUDGET_TOKENS rather
        than picked for round numbers: a returned member carries a 600-char
        `body_preview`, so the response costs roughly
        `max_clusters * member_limit * 850` chars and the previous 20 x 20
        spent ~47k tokens against a 10k cap -- 20 clusters cannot be returned
        with any useful member detail at all. 10 x 2 measures 3,978 tokens on
        the conformance corpus and ~9,039 projected for a corpus whose bodies
        actually reach the 600-char preview cap. Two sample members are what
        it takes to SEE a duplicate; `total` still reports the cluster's real
        size, and `memo_get` fetches any member by id. Raise either argument
        to see more and accept a larger response (the budget middleware
        refuses one that overruns rather than silently truncating it).

        Args:
            threshold: Cosine similarity threshold (default 0.85).
            max_clusters: Maximum clusters to process (default 10).
            type: Optional filter by memory type.
            member_limit: Sample members returned per cluster (default 2).
        """
        from memo.server_common import run_synth

        out, synthesizer = await run_synth(
            memory,
            ctx,
            lambda: memory.consolidator.consolidate_all(
                threshold=threshold,
                max_clusters=max_clusters,
                type_=type,
                auto_apply=False,
                dry_run=True,
            ),
        )
        out = _bounded_consolidate(out, cluster_limit=max_clusters, member_limit=member_limit)
        out["synthesizer"] = synthesizer
        return out

    @annotated_tool(server, **READ_ONLY)
    def memo_lint(
        limit: Annotated[
            int,
            Field(
                description="Findings to return per category. True per-category "
                "totals come back under `counts`."
            ),
        ] = 20,
    ) -> dict[str, Any]:
        """Inspect the memory corpus for maintenance issues.

        Read-only. Returns grouped lint findings such as malformed metadata or
        other records that may need cleanup, capped at ``limit`` per category —
        the findings scale with the corpus and a whole report runs past any
        client's response budget. Use memo_update, memo_delete, or vault edits
        separately to fix findings.
        """
        return _bounded_lint(memory.lint(), limit=limit)
