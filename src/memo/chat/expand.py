"""Query category gate + LLM multi-query expansion (RRF-fused by the pipeline)."""

from __future__ import annotations

import json
import re
from typing import Any

_LEXICAL_IDENTIFIER_RE = re.compile(
    r"(`[^`]+`|\"[^\"]+\"|\b[a-z0-9]+(?:_[a-z0-9]+)+\b|\b[A-Z0-9]{3,}\b)"
)
_MULTI_HOP_RE = re.compile(
    r"\?.*\?|\by\s+(cu[aá]ndo|d[oó]nde|qui[eé]n|por\s+qu[eé])\b|\band\s+(when|where|who|why)\b",
    re.IGNORECASE | re.DOTALL,
)
_EXPAND_PROMPT = (
    "Sos un asistente de búsqueda. Reformulá la PREGUNTA en {n} variantes DIFERENTES "
    "— cambiá vocabulario, sinónimos, orden, nivel de formalidad, pero mantené la "
    "INTENCIÓN exacta. NO respondas. Cada variante en 1 línea, ≤ 200 caracteres. "
    'Devolvé SOLO JSON: {{"variants": ["...", "..."]}}.\nPREGUNTA: {q}'
)


def classify_query(q: str) -> str:
    if _LEXICAL_IDENTIFIER_RE.search(q or ""):
        return "lexical_exact"
    if _MULTI_HOP_RE.search(q or ""):
        return "multi_hop"
    return "semantic_fuzzy"


def allows_multi_query(category: str) -> bool:
    return category in {"semantic_fuzzy", "multi_hop"}


def _content_of(out: Any) -> str:
    if isinstance(out, dict):
        message = out.get("message")
        if isinstance(message, dict):
            return str(message.get("content", ""))
        return str(out.get("content", "") or out.get("response", ""))
    return str(out)


def expand_query(chat: Any, model: str, question: str, *, n: int = 2) -> list[str]:
    try:
        out = chat.chat(
            model,
            [{"role": "user", "content": _EXPAND_PROMPT.format(n=n, q=question)}],
            options={"temperature": 0.0, "max_tokens": 400},
        )
        content = _content_of(out)
        payload = content[content.index("{") : content.rindex("}") + 1]
        variants = json.loads(payload).get("variants", [])
        return [str(v).strip() for v in variants if str(v).strip()][:n]
    except Exception:
        return []
