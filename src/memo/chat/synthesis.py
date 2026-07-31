"""Grounded synthesis over curated sources, with the relative relevance floor."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from memo.chat.dedup import score_of

REFUSAL = "No encontré esa información en mis notas."


def filter_by_relevance(sources: list[dict[str, Any]], *, floor: float) -> list[dict[str, Any]]:
    if floor <= 0 or len(sources) < 2:
        return list(sources)
    top = max(score_of(s) for s in sources)
    if top <= 0:
        return list(sources)
    kept = [s for s in sources if s.get("keep") or score_of(s) >= top * floor]
    return kept if kept else [max(sources, key=score_of)]


def build_messages(
    question: str, sources: list[dict[str, Any]], *, today: str
) -> list[dict[str, str]]:
    header = (
        "Sos un asistente RAG de precisión alta. Respondés EXCLUSIVAMENTE con "
        "información que aparece en los SNIPPETS del mensaje del usuario.\n\n"
        f"Fecha actual: {today}. Usá esta fecha para calcular edades y tiempos exactos."
    )
    rules = (
        "Reglas:\n"
        "- Prosa clara; un párrafo por aspecto; sin marcadores [n] ni citas numeradas.\n"
        "- No agregues conocimiento externo a los SNIPPETS.\n"
        f'- Si los SNIPPETS no responden la pregunta, respondé exactamente: "{REFUSAL}"'
    )
    snippets = "\n\n".join(
        f"[{i + 1}] {s.get('title', '')}\n{s.get('snippet', '')}" for i, s in enumerate(sources)
    )
    return [
        {"role": "system", "content": f"{header}\n\n{rules}"},
        {"role": "user", "content": f"PREGUNTA: {question}\n\nSNIPPETS:\n{snippets}"},
    ]


def synthesize_stream(
    chat: Any,
    model: str,
    question: str,
    sources: list[dict[str, Any]],
    *,
    max_tokens: int,
) -> Iterator[str]:
    from datetime import date

    messages = build_messages(question, sources, today=date.today().strftime("%d/%m/%Y"))
    yield from chat.chat_stream(
        model, messages, options={"temperature": 0.1, "max_tokens": max_tokens}
    )
