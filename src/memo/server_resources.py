from __future__ import annotations

from typing import Any

from memo.flags import flag_int
from memo.memory import AmbiguousIdError, Memory


def register(server: Any, memory: Memory) -> None:
    @server.resource("memo://profile/{scope}")
    def _resource_profile(scope: str) -> str:
        """Expose the bounded profile as a poll-only MCP resource."""
        from memo.memory_profile import build_memory_profile

        try:
            payload = build_memory_profile(memory, scope=scope)
        except ValueError as exc:
            return f"# Invalid profile scope\n\n{exc}\n"
        import json

        return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"

    @server.resource("memo://recent")
    def _resource_recent() -> str:
        recs = memory.list(limit=20)
        if not recs:
            return "# memo · recent\n\n_(no memories yet)_\n"
        out = ["# memo · recent", ""]
        for r in recs:
            tags = ", ".join(r.tags) if r.tags else ""
            out.append(
                f"- **[{r.id[:8]}]** [{r.title}](memo://memory/{r.id}) "
                f"_{r.type}_{(' · ' + tags) if tags else ''}"
            )
        return "\n".join(out) + "\n"

    @server.resource("memo://memory/{id}")
    def _resource_memory(id: str) -> str:
        try:
            rec = memory.get(id)
        except AmbiguousIdError as exc:
            return f"# Ambiguous id `{id}`\n\nMatches:\n\n" + "\n".join(
                f"- `{m}`" for m in exc.matches
            )
        if rec is None:
            return f"# Not found\n\nNo memory for id `{id}`.\n"
        tags = ", ".join(rec.tags) if rec.tags else "—"
        body_chars_flag = flag_int("MEMO_RESOURCE_BODY_CHARS")
        body_chars = 1200 if body_chars_flag is None else body_chars_flag
        body = rec.body or ""
        truncated = False
        if body_chars >= 0 and len(body) > body_chars:
            body = body[:body_chars].rstrip() + "…"
            truncated = True
        return (
            f"# {rec.title}\n\n"
            f"- **id:** `{rec.id}`\n"
            f"- **type:** {rec.type}\n"
            f"- **tags:** {tags}\n"
            f"- **created:** {rec.created}\n"
            f"- **updated:** {rec.updated}\n\n"
            f"---\n\n{body}"
            + ("\n\n_(preview only; use `memo get <id>` for the full body.)_" if truncated else "")
        )
