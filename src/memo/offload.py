"""Context offloading — store bulky payloads content-addressed, return a
deterministic typed synopsis + a drill-down id.

`offload()` writes the payload as a `type=reference` memory (the reference
tier is EXCLUDED from auto-recall, so offloaded blobs never pollute the
recall hook) and dedupes by sha256 via `state_dir/offload/index.json`.
`synopsize()` is no-LLM by design (OpenViking-style): JSON key sampling, CSV
headers, code symbols, else the compress-context rules. The drill-down is the
existing `memo_get(id)`. The interception/canvas layer belongs in synapse —
memo keeps cognition off its surface and only ships the handle primitive.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

_log = logging.getLogger(__name__)

_SYNOPSIS_MAX_CHARS = 600
_CODE_LINE = re.compile(
    r"\s*(def |class |import |from |function |const |let |var |#include|package )"
)
_CODE_SYMBOL = re.compile(r"\s*(def |class |function )")


def synopsize(content: str, *, max_chars: int = _SYNOPSIS_MAX_CHARS) -> tuple[str, str]:
    """Deterministic typed synopsis (kind, text). No LLM call, ever."""
    text = content.strip()
    try:
        obj = json.loads(text)
    except (ValueError, RecursionError):
        obj = None
    if isinstance(obj, dict):
        keys = ", ".join(str(k) for k in list(obj)[:12])
        return "json", f"JSON object · {len(obj)} keys: {keys}"[:max_chars]
    if isinstance(obj, list):
        inner = type(obj[0]).__name__ if obj else "empty"
        return "json", f"JSON array · {len(obj)} items ({inner})"[:max_chars]
    lines = text.splitlines()
    if len(lines) >= 2:
        sep = "\t" if "\t" in lines[0] else "," if "," in lines[0] else None
        if sep is not None and all(
            line.count(sep) == lines[0].count(sep)
            for line in lines[1 : min(len(lines), 6)]
        ):
            return "csv", (
                f"Table · {len(lines) - 1} rows · columns: {lines[0][:200]}"[:max_chars]
            )
    if sum(1 for line in lines[:80] if _CODE_LINE.match(line)) >= 3:
        symbols = "; ".join(line.strip()[:80] for line in lines if _CODE_SYMBOL.match(line))
        return "code", f"Code · {len(lines)} lines · {symbols}"[:max_chars]
    from memo.cli_compress_context import compress

    return "text", compress(text)[:max_chars]


def _index_path(memory: Any) -> Any:
    return memory.cfg.state_dir / "offload" / "index.json"


def _read_index(memory: Any) -> dict[str, str]:
    try:
        return dict(json.loads(_index_path(memory).read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return {}


def offload(memory: Any, content: str, *, title: str | None = None) -> dict[str, Any]:
    """Store `content` content-addressed; return id + synopsis (or error dict)."""
    if not content or not content.strip():
        return {"error": "empty", "message": "offload: empty payload"}
    if len(content) > memory.cfg.max_content_chars:
        return {
            "error": "too_large",
            "message": (
                f"offload: payload {len(content)} chars exceeds "
                f"max_content_chars={memory.cfg.max_content_chars} "
                "(raise MEMO_MAX_CONTENT_CHARS if intended)"
            ),
        }
    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    kind, synopsis = synopsize(content)
    index = _read_index(memory)
    existing = index.get(sha)
    if existing is not None and memory.get(existing) is not None:
        return {
            "id": existing,
            "sha256": sha,
            "kind": kind,
            "synopsis": synopsis,
            "deduplicated": True,
            "drill_down": f"memo_get('{existing[:12]}')",
        }
    label = title or f"offload:{kind} {sha[:12]}"
    # Prefix a self-describing markdown heading so the payload clears the
    # reference-tier noise gate in `save()` (rejects <60-char reference bodies
    # with no heading/link); sha/synopsis/dedup stay over the raw payload and
    # the payload remains retrievable verbatim via the drill-down.
    rec = memory.save(
        content=f"# {label}\n\n{content}",
        title=label,
        type_="reference",
        tags=["offload"],
        extra={"offload_sha256": sha, "offload_kind": kind},
    )
    index[sha] = rec.id
    try:
        path = _index_path(memory)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        _log.warning("offload: index write failed (dedup degraded): %s", exc)
    return {
        "id": rec.id,
        "sha256": sha,
        "kind": kind,
        "synopsis": synopsis,
        "deduplicated": False,
        "drill_down": f"memo_get('{rec.id[:12]}')",
    }
