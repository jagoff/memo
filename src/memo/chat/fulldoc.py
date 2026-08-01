"""Fulldoc inline: when one doc dominates the hits, answer with the whole doc."""

from __future__ import annotations

from typing import Any

from memo.chat.dedup import CHUNK_NUM, dedup_key, normalize_title

_MIN_SHARE = 0.6
_MIN_CHUNKS = 2


def dominant_doc_group(
    sources: list[dict[str, Any]],
    *,
    min_share: float = _MIN_SHARE,
    min_chunks: int = _MIN_CHUNKS,
) -> list[dict[str, Any]] | None:
    if not sources:
        return None
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for s in sources:
        groups.setdefault(dedup_key(s), []).append(s)
    members = max(groups.values(), key=len)
    if len(members) < min_chunks:
        return None
    if len(members) / len(sources) < min_share:
        return None
    return members


def _chunk_numbers(members: list[dict[str, Any]]) -> tuple[set[int], set[int]]:
    nums: set[int] = set()
    totals: set[int] = set()
    for m in members:
        match = CHUNK_NUM.search(str(m.get("title") or ""))
        if match:
            nums.add(int(match.group(1)))
            totals.add(int(match.group(2)))
    return nums, totals


def _full_body(memory: Any, member: dict[str, Any]) -> str:
    """Full body of a memory chunk, falling back to its (700-char) snippet.

    resolve_fulldoc's whole point is inlining the WHOLE doc — reassembling
    from pipeline._SNIPPET_CHARS-truncated snippets loses most of the content.
    """
    snippet = str(member.get("snippet") or "")
    member_id = str(member.get("id") or "")
    if not member_id:
        return snippet
    try:
        record = memory.get(member_id)
    except Exception:
        return snippet
    body = str(getattr(record, "body", "") or "") if record is not None else ""
    return body or snippet


def resolve_fulldoc(memory: Any, members: list[dict[str, Any]]) -> dict[str, Any] | None:
    head = members[0]
    # Vault branch: must complete or return None
    if head.get("source") == "vault":
        if not (head.get("repo_name") and head.get("path")):
            return None
        doc = memory.repo_get_file(str(head["repo_name"]), str(head["path"]))
        if isinstance(doc, dict):
            text = str(doc.get("text") or doc.get("content") or "")
            if text:
                return {
                    "title": normalize_title(head.get("title")),
                    "text": text,
                    "fulldoc_source": "repo",
                }
        return None

    # Memory branch: reassemble if all chunks present
    nums, totals = _chunk_numbers(members)
    if len(totals) != 1 or nums != set(range(1, next(iter(totals)) + 1)):
        return None
    ordered = sorted(
        (m for m in members if CHUNK_NUM.search(str(m.get("title") or ""))),
        key=lambda m: int(CHUNK_NUM.search(str(m["title"])).group(1)),  # type: ignore[union-attr]
    )
    text = "\n\n".join(_full_body(memory, m) for m in ordered)
    return {"title": normalize_title(head.get("title")), "text": text, "fulldoc_source": "memory"}
