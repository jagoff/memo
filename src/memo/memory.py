"""High-level `Memory` API — saves to vault + indexes to sqlite-vec.

This is the layer that callers (CLI, MCP server, library users)
interact with. Wraps `MLXEmbedder` + `VecStore` + frontmatter writer
into a coherent interface mirroring `mem-vault.Memory`:

- `save(content, ...)` → write `.md` file under
  `vault/memory_subdir/<slug>.md`, embed, index. Returns
  `MemoryRecord`.
- `search(query, limit)` → embed the query, top-k vec search, hydrate
  each hit with metadata + on-disk content snippet.
- `list(type_, limit)` → recent entries by `updated` desc.
- `get(id_)` → full record + body.
- `update(id_, ...)` → patch one or more fields, re-embed if content
  changed (body_hash check).
- `delete(id_)` → remove from vec + meta + delete `.md` file.

The `.md` storage of record uses Obsidian-friendly frontmatter so the
user can edit memories from Obsidian and the next index pass picks
them up via `body_hash` mismatch.

## Frontmatter schema

```yaml
---
id: <uuid4 hex>
title: Short descriptive title
type: decision | fact | bug | feedback | preference | note
tags: [tag1, tag2, tag3]
created: 2026-05-06T19:30:00-03:00
updated: 2026-05-06T19:30:00-03:00
---

(body — markdown, free-form)
```

## Slugify

`<YYYY-MM-DD>-<slug-of-title>.md` — e.g.
`2026-05-06-decision-tema.md`. Same convention as obsidian-rag's
conversation writer.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import frontmatter

from memo.config import Config
from memo.embedder import MLXEmbedder
from memo.llm import MLXChat
from memo.store import VecStore


# JSON-schema prompt for the helper LLM. Kept terse to fit in Qwen3-3B's
# attention without hurting accuracy. Empirically the model follows the
# format strictly under temperature=0; the regex fallback in
# `_derive_metadata` handles the occasional markdown fence wrap.
_DERIVE_SYSTEM_PROMPT = """You classify a memory note into a structured JSON object.

Output ONLY a JSON object with these keys:
- "title": short descriptive title, max 80 chars, no date prefix
- "type": one of "decision", "fact", "bug", "feedback", "preference", "note", "manual"
- "tags": array of 3-6 lowercase tags (mix of project, domain, technique)

Type rules:
- "decision": choice with explicit tradeoff or rationale
- "bug": problem + root cause + fix
- "fact": discovery, gotcha, learned constraint
- "preference": user preference or convention to follow
- "feedback": user feedback on an approach
- "note": catch-all, use when no other type fits

Output ONLY the JSON, no markdown fences, no commentary, no preamble."""

_VALID_TYPES = frozenset(
    {"decision", "fact", "bug", "feedback", "preference", "note", "manual"}
)


class AmbiguousIdError(ValueError):
    """Raised when an id prefix matches more than one record. Carries
    the candidate matches so the caller can surface them in an error."""

    def __init__(self, prefix: str, matches: list[str]) -> None:
        super().__init__(
            f"Ambiguous id prefix {prefix!r}: {len(matches)} matches "
            f"({', '.join(m[:8] for m in matches[:5])}...)",
        )
        self.prefix = prefix
        self.matches = matches


@dataclass(frozen=True)
class MemoryRecord:
    """Public, immutable view of one memory entry.

    Internally the store + on-disk file may diverge briefly during a
    `save()` (vec insert happens after the file write). Callers always
    receive a `MemoryRecord` only after both writes have committed.
    """

    id: str
    path: str  # vault-relative
    title: str
    type: str
    tags: list[str]
    created: str  # ISO8601
    updated: str
    body: str
    extra: dict[str, Any] = field(default_factory=dict)
    score: float | None = None  # populated by `search()`; None for direct fetches.

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "path": self.path,
            "title": self.title,
            "type": self.type,
            "tags": list(self.tags),
            "created": self.created,
            "updated": self.updated,
            "body": self.body,
            "extra": dict(self.extra),
            "score": self.score,
        }


class Memory:
    """High-level memory API. Construct once per process; methods are
    thread-safe (delegate to store/embedder which both serialise their
    critical sections).

    Example:

        cfg = Config.from_env()
        cfg.ensure_dirs()
        mem = Memory(cfg)

        rec = mem.save(
            content="**What**: Migré obsidian-rag a MLX. ...",
            title="MLX migration cierre formal",
            type_="decision",
            tags=["mlx", "obsidian-rag", "migration"],
        )

        hits = mem.search("cómo migré a mlx", limit=5)
        for h in hits:
            print(f"{h.score:.3f} · {h.title}")
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.embedder = MLXEmbedder(
            model_path=cfg.embedder_model,
            expected_dims=cfg.embedder_dims,
        )
        self.store = VecStore(cfg.db_path, dims=cfg.embedder_dims)
        # Lazy: opened on first log call. Audit failures must never
        # propagate to the caller, so HistoryStore swallows its own
        # exceptions internally.
        from memo.history import HistoryStore as _HS
        self.history = _HS(cfg.history_db)
        # Helper LLM is lazy — only constructed when `auto_derive=True`
        # is requested. Cold load of Qwen2.5-3B is ~2-3s; users who
        # don't opt in pay nothing.
        self._chat: MLXChat | None = None

    # -- save ---------------------------------------------------------------

    def _derive_metadata(self, content: str) -> dict[str, Any]:
        """Use the helper LLM (Qwen2.5-3B-Instruct-4bit) to derive
        {title, type, tags} from raw content. Returns a dict with
        whatever keys the model produced (any can be None on parse
        failure). Caller decides whether to fill missing fields.

        Failure modes are absorbed: a bad LLM response yields an empty
        dict and the caller falls back to heuristics. We never propagate
        an LLM error up to a save() call — the save must succeed even
        if the helper is broken.
        """
        if self._chat is None:
            self._chat = MLXChat()
        try:
            out = self._chat.chat(
                model=self.cfg.helper_model,
                messages=[
                    {"role": "system", "content": _DERIVE_SYSTEM_PROMPT},
                    # Cap input to keep the prompt cheap. The helper only
                    # needs the gist, not the full body.
                    {"role": "user", "content": content[:2000]},
                ],
                options={"temperature": 0.0, "max_tokens": 256},
            )
            text = (out.get("message") or {}).get("content") or ""
        except Exception:
            return {}
        # Tolerate markdown code fences even though the prompt forbids them.
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE)
        try:
            data = json.loads(text)
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        derived: dict[str, Any] = {}
        t_title = (data.get("title") or "")
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
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
        auto_derive: bool = False,
    ) -> MemoryRecord:
        """Persist a memory to disk + index.

        - `content`: free-form markdown body (no frontmatter; we add it).
        - `title`: optional. If omitted, derived from the first line of
          content (truncated, slug-safe).
        - `type_`: must be in `_VALID_TYPES`. `note` is the default
          neutral value.
        - `tags`: optional list. Lower-cased + de-duplicated.
        - `extra`: arbitrary JSON-serialisable metadata bag.
        - `auto_derive`: when True, calls the helper LLM
          (`Qwen2.5-3B-Instruct-4bit`) to fill any missing field
          (title is None, type_ is "note" with no tags). Adds ~1-2s
          latency on first call (cold model load) plus ~0.5-1s per save.
          Use for callers (eg. another agent) that don't carry context
          to derive metadata themselves.
        """
        if not content or not content.strip():
            raise ValueError("`content` must be non-empty")
        if type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )

        if auto_derive:
            # Only fire the LLM if at least one field looks "default-y".
            # User-provided values always win.
            wants_title = title is None
            wants_type = type_ == "note"
            wants_tags = not tags
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
        now_iso = _now_iso()
        # Truncate content for embedding (vec store doesn't truncate;
        # disk file keeps full content). 64KB is the default cap.
        content = content[: self.cfg.max_content_chars]

        record_id = uuid.uuid4().hex
        rel_path = self._build_rel_path(title, now_iso)
        body_hash = _sha256_short(content)

        # Write `.md` first — if anything fails after this, the user
        # can recover by re-indexing. Conversely if we write the index
        # first and the disk write fails, the index points to a
        # non-existent file.
        abs_path = self.cfg.vault_path / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(
            content,
            id=record_id,
            title=title,
            type=type_,
            tags=norm_tags,
            created=now_iso,
            updated=now_iso,
            **({"extra": extra} if extra else {}),
        )
        abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Embed `title + body`: the title carries the highest-density
        # signal for retrieval ("Astor — Informe TO" is a much better
        # match for a query like "informe terapia ocupacional astor"
        # than the body's clinical paragraphs alone). Prepending also
        # protects the title from head-truncation when the body is
        # long — see embedder.py for the truncation rationale.
        embedding = self.embedder.embed([_compose_for_embed(title, content)])[0]

        self.store.upsert(
            id_=record_id,
            path=rel_path,
            title=title,
            type_=type_,
            tags=norm_tags,
            created=now_iso,
            updated=now_iso,
            body_hash=body_hash,
            embedding=embedding,
            extra=extra,
        )

        self.history.log_save(
            ts=now_iso, record_id=record_id, title=title, type_=type_,
        )

        return MemoryRecord(
            id=record_id, path=rel_path, title=title, type=type_, tags=norm_tags,
            created=now_iso, updated=now_iso, body=content, extra=extra or {},
        )

    # -- search -------------------------------------------------------------

    def search(
        self, query: str, *, limit: int | None = None, type_: str | None = None,
    ) -> list[MemoryRecord]:
        """Top-k semantic search. Returns records sorted by descending
        cosine similarity. Each result has `.score` populated."""
        if not query or not query.strip():
            return []
        limit = limit or self.cfg.search_default_limit
        # Asymmetric retrieval: queries are embedded WITH the instruction
        # prefix; documents are embedded RAW (in `save()` / `update()`).
        # See `_QUERY_INSTRUCTION_PREFIX` in `embedder.py` for the why.
        emb = self.embedder.embed_query(query)
        rows = self.store.search(emb, limit=limit, type_=type_)
        out: list[MemoryRecord] = []
        for r in rows:
            body = self._read_body(r["path"])
            out.append(
                MemoryRecord(
                    id=r["id"], path=r["path"], title=r["title"], type=r["type"],
                    tags=r["tags"], created=r["created"], updated=r["updated"],
                    body=body, extra=r.get("extra") or {}, score=r.get("score"),
                ),
            )
        return out

    # -- list ---------------------------------------------------------------

    def list(
        self, *, limit: int = 20, type_: str | None = None,
    ) -> list[MemoryRecord]:
        """Recent entries by `updated` desc. Body included for each."""
        rows = self.store.list_recent(limit=limit, type_=type_)
        return [
            MemoryRecord(
                id=r["id"], path=r["path"], title=r["title"], type=r["type"],
                tags=r["tags"], created=r["created"], updated=r["updated"],
                body=self._read_body(r["path"]), extra=r.get("extra") or {},
            )
            for r in rows
        ]

    # -- get ----------------------------------------------------------------

    def resolve_id(self, id_or_prefix: str) -> str | None:
        """Resolve a full id or a unique prefix.

        Returns the canonical 32-char id if `id_or_prefix` matches exactly
        one record (full or prefix), or None if nothing matches. Raises
        `AmbiguousIdError` when 2+ records share the prefix — the caller
        is expected to surface the candidates so the user can disambiguate.

        Why prefix lookup: pasting a 32-char UUID4 from chat is friction.
        Git-style 7-char prefixes are unique with overwhelming probability
        for the corpus sizes memo targets (~thousands).
        """
        if not id_or_prefix:
            return None
        # Fast path: full hex hit.
        if len(id_or_prefix) == 32 and self.store.get(id_or_prefix) is not None:
            return id_or_prefix
        matches = self.store.find_by_prefix(id_or_prefix.lower())
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise AmbiguousIdError(id_or_prefix, matches)
        return None

    def get(self, id_: str) -> MemoryRecord | None:
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        r = self.store.get(resolved)
        if not r:
            return None
        return MemoryRecord(
            id=r["id"], path=r["path"], title=r["title"], type=r["type"],
            tags=r["tags"], created=r["created"], updated=r["updated"],
            body=self._read_body(r["path"]), extra=r.get("extra") or {},
        )

    # -- update -------------------------------------------------------------

    def update(
        self,
        id_: str,
        *,
        title: str | None = None,
        type_: str | None = None,
        tags: list[str] | None = None,
        content: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryRecord | None:
        """Patch one or more fields on an existing record.

        Only the kwargs you pass are touched; everything else stays as-is.
        Re-embed only if `content` changed (body_hash check). The file
        path stays stable — renaming the slug after the fact would break
        wikilinks the user may have created in their vault.
        """
        resolved = self.resolve_id(id_)
        if resolved is None:
            return None
        id_ = resolved
        r = self.store.get(id_)
        if r is None:
            return None
        if type_ is not None and type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )

        new_title = (title.strip() if title else r["title"]) or r["title"]
        new_type = type_ or r["type"]
        new_tags = _normalise_tags(tags) if tags is not None else r["tags"]
        new_extra = extra if extra is not None else r.get("extra") or {}
        now_iso = _now_iso()

        # Body resolution: provided > on-disk > empty.
        old_body = self._read_body(r["path"])
        new_body = (content if content is not None else old_body)
        new_body = new_body[: self.cfg.max_content_chars]
        new_body_hash = _sha256_short(new_body)
        body_changed = new_body_hash != r["body_hash"]
        title_changed = new_title != r["title"]

        abs_path = self.cfg.vault_path / r["path"]
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(
            new_body,
            id=id_,
            title=new_title,
            type=new_type,
            tags=new_tags,
            created=r["created"],
            updated=now_iso,
            **({"extra": new_extra} if new_extra else {}),
        )
        abs_path.write_text(frontmatter.dumps(post), encoding="utf-8")

        # Re-embed when the body OR title changed — both are part of the
        # embed input now (see `_compose_for_embed`). Pure retag/type
        # changes still skip the embedder.
        if body_changed or title_changed:
            embedding = self.embedder.embed([_compose_for_embed(new_title, new_body)])[0]
            self.store.upsert(
                id_=id_, path=r["path"], title=new_title, type_=new_type,
                tags=new_tags, created=r["created"], updated=now_iso,
                body_hash=new_body_hash, embedding=embedding, extra=new_extra,
            )
        else:
            self.store.update_meta(
                id_=id_, title=new_title, type_=new_type, tags=new_tags,
                updated=now_iso, extra=new_extra,
            )

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
        if delta:
            self.history.log_update(
                ts=now_iso, record_id=id_, title=new_title, type_=new_type,
                delta=delta,
            )

        return MemoryRecord(
            id=id_, path=r["path"], title=new_title, type=new_type,
            tags=new_tags, created=r["created"], updated=now_iso,
            body=new_body, extra=new_extra,
        )

    # -- delete -------------------------------------------------------------

    def delete(self, id_: str) -> bool:
        """Remove from store + disk. Returns True if anything was deleted."""
        resolved = self.resolve_id(id_)
        if resolved is None:
            return False
        id_ = resolved
        r = self.store.get(id_)
        if not r:
            return False
        existed = self.store.delete(id_)
        try:
            (self.cfg.vault_path / r["path"]).unlink(missing_ok=True)
        except OSError:
            # File deletion is best-effort — store is the authoritative
            # delete signal. Stale `.md` files get cleaned up by a
            # `memo doctor --gc` pass.
            pass
        if existed:
            self.history.log_delete(
                ts=_now_iso(), record_id=id_, title=r["title"], type_=r["type"],
            )
        return existed

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
            except Exception:
                skipped += 1
                continue
            md_id = post.get("id")
            if not md_id or not isinstance(md_id, str):
                skipped += 1
                continue
            body = post.content or ""
            new_hash = _sha256_short(body)
            existing = self.store.get(md_id)
            rel = str(md_path.relative_to(self.cfg.vault_path))

            title = (post.get("title") or _derive_title(body) or "untitled").strip()
            type_ = post.get("type") or "note"
            if type_ not in _VALID_TYPES:
                type_ = "note"
            tags = _normalise_tags(list(post.get("tags") or []))
            created = post.get("created") or _now_iso()
            updated = post.get("updated") or created
            extra = post.get("extra") or {}

            if existing is None:
                emb = self.embedder.embed([_compose_for_embed(title, body)])[0]
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=created, updated=updated, body_hash=new_hash,
                    embedding=emb, extra=extra if extra else None,
                )
                added += 1
                continue
            if force or existing["body_hash"] != new_hash:
                emb = self.embedder.embed([_compose_for_embed(title, body)])[0]
                self.store.upsert(
                    id_=md_id, path=rel, title=title, type_=type_, tags=tags,
                    created=existing["created"], updated=_now_iso(),
                    body_hash=new_hash, embedding=emb,
                    extra=extra if extra else None,
                )
                reindexed += 1
        return {"checked": checked, "reindexed": reindexed, "added": added, "skipped": skipped}

    def lint(self) -> dict[str, list[dict[str, Any]]]:
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

    def gc(self, *, fix: bool = False) -> dict[str, list[str]]:
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

        # Store-side: walk meta, check file existence.
        for r in self.store.list_recent(limit=100_000):
            if not (self.cfg.vault_path / r["path"]).is_file():
                orphan_store.append(r["id"])
                if fix:
                    self.store.delete(r["id"])

        # Disk-side: walk memory dir, check ids in store.
        if self.cfg.memory_dir.is_dir():
            for md_path in self.cfg.memory_dir.rglob("*.md"):
                try:
                    post = frontmatter.loads(md_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                md_id = post.get("id")
                if not md_id or not isinstance(md_id, str):
                    continue
                if self.store.get(md_id) is None:
                    orphan_disk.append(str(md_path.relative_to(self.cfg.vault_path)))

        return {"orphan_store": orphan_store, "orphan_disk": orphan_disk}

    # -- internals ----------------------------------------------------------

    def _build_rel_path(self, title: str, now_iso: str) -> str:
        date = now_iso.split("T", 1)[0]
        slug = _slugify(title)[:80] or "untitled"
        # Use POSIX path joins; vault is always macOS / iCloud.
        return f"{self.cfg.memory_subdir}/{date}-{slug}.md"

    def _read_body(self, rel_path: str) -> str:
        abs_path = self.cfg.vault_path / rel_path
        if not abs_path.is_file():
            return ""
        try:
            text = abs_path.read_text(encoding="utf-8")
            post = frontmatter.loads(text)
            return post.content
        except Exception:
            return ""


# ── Helpers ──────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(tz=UTC).astimezone().isoformat(timespec="seconds")


def _sha256_short(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


_SLUG_NON_WORD = re.compile(r"[^\w\s-]+")
_SLUG_WS = re.compile(r"[\s_-]+")


def _slugify(s: str) -> str:
    s = s.lower().strip()
    s = _SLUG_NON_WORD.sub("", s)
    s = _SLUG_WS.sub("-", s)
    return s.strip("-")


def _derive_title(content: str) -> str:
    # First non-empty line, stripped of leading markdown markers.
    for raw in content.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = re.sub(r"^[#*\-\"`>\s]+", "", line).rstrip(" .:;")
        if line:
            return line[:80]
    return ""


def _compose_for_embed(title: str, body: str) -> str:
    """Combine title + body into the string passed to the embedder.

    Title-first because: (a) titles carry the highest-density retrieval
    signal in this corpus (memos with terse titles + long bodies dominate),
    (b) head-truncation guarantees the title survives even when body is
    long, (c) avoiding double-prefix when title already appears as an H1
    in the body — we do NOT dedup, the redundancy doesn't hurt the
    embedder and the simpler code is worth the few wasted tokens.
    """
    title = (title or "").strip()
    body = (body or "").strip()
    if not title:
        return body
    if not body:
        return title
    return f"{title}\n\n{body}"


def _normalise_tags(tags: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for t in tags:
        t = (t or "").strip().lower()
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


__all__ = ["AmbiguousIdError", "Memory", "MemoryRecord"]
