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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter

from mem_lmx.config import Config
from mem_lmx.embedder import MLXEmbedder
from mem_lmx.store import VecStore


_VALID_TYPES = frozenset(
    {"decision", "fact", "bug", "feedback", "preference", "note", "manual"}
)


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

    # -- save ---------------------------------------------------------------

    def save(
        self,
        *,
        content: str,
        title: str | None = None,
        type_: str = "note",
        tags: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> MemoryRecord:
        """Persist a memory to disk + index.

        - `content`: free-form markdown body (no frontmatter; we add it).
        - `title`: optional. If omitted, derived from the first line of
          content (truncated, slug-safe).
        - `type_`: must be in `_VALID_TYPES`. `note` is the default
          neutral value.
        - `tags`: optional list. Lower-cased + de-duplicated.
        - `extra`: arbitrary JSON-serialisable metadata bag.
        """
        if not content or not content.strip():
            raise ValueError("`content` must be non-empty")
        if type_ not in _VALID_TYPES:
            raise ValueError(
                f"`type_={type_!r}` not in valid set {sorted(_VALID_TYPES)}",
            )
        title = (title or _derive_title(content)).strip()
        if not title:
            title = "untitled"

        norm_tags = _normalise_tags(tags or [])
        now_iso = _now_iso()
        # Truncate content for embedding (vec store doesn't truncate;
        # disk file keeps full content). 64KB is the default cap.
        content = content[: self.cfg.max_content_size]

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

        # Embed the body (not the title; titles are short and bias the
        # vector toward filename-style matches). Caller can pass a
        # `title:`-prefixed content if they want title-included embed.
        embedding = self.embedder.embed([content])[0]

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
        emb = self.embedder.embed([query])[0]
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

    def get(self, id_: str) -> MemoryRecord | None:
        r = self.store.get(id_)
        if not r:
            return None
        return MemoryRecord(
            id=r["id"], path=r["path"], title=r["title"], type=r["type"],
            tags=r["tags"], created=r["created"], updated=r["updated"],
            body=self._read_body(r["path"]), extra=r.get("extra") or {},
        )

    # -- delete -------------------------------------------------------------

    def delete(self, id_: str) -> bool:
        """Remove from store + disk. Returns True if anything was deleted."""
        r = self.store.get(id_)
        if not r:
            return False
        existed = self.store.delete(id_)
        try:
            (self.cfg.vault_path / r["path"]).unlink(missing_ok=True)
        except OSError:
            # File deletion is best-effort — store is the authoritative
            # delete signal. Stale `.md` files get cleaned up by a
            # future `mem-lmx doctor --gc` pass.
            pass
        return existed

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
    return datetime.now(tz=timezone.utc).astimezone().isoformat(timespec="seconds")


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


__all__ = ["Memory", "MemoryRecord"]
