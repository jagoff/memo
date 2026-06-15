from __future__ import annotations

from typing import Any

from memo.memory import AmbiguousIdError, Memory


def register(server: Any, memory: Memory) -> None:
    @server.resource("memo://recent")
    def _resource_recent() -> str:
        recs = memory.list(limit=20)
        if not recs:
            return "# memo · recent\n\n_(no memorias yet)_\n"
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
            return f"# Not found\n\nNo memoria for id `{id}`.\n"
        tags = ", ".join(rec.tags) if rec.tags else "—"
        return (
            f"# {rec.title}\n\n"
            f"- **id:** `{rec.id}`\n"
            f"- **type:** {rec.type}\n"
            f"- **tags:** {tags}\n"
            f"- **created:** {rec.created}\n"
            f"- **updated:** {rec.updated}\n\n"
            f"---\n\n{rec.body or ''}"
        )
