"""Opt-in contextual retrieval for memory embeddings.

When enabled, memo prepends a short generated context to the text sent to the
embedder. The stored Markdown and displayed snippets stay unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path

PROMPT_VERSION = "memo-contextual-v1-2026-05-26"
MIN_BODY_CHARS = 300
MAX_BODY_CHARS = 4000
MAX_SUMMARY_CHARS = 240
SUMMARY_MARKER = "[contexto:"


def contextual_retrieval_enabled() -> bool:
    raw = os.environ.get("MEMO_CONTEXTUAL_RETRIEVAL", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def context_cache_path(state_dir: Path) -> Path:
    return Path(state_dir) / "contextual_retrieval_cache.json"


def context_hash(title: str, body: str) -> str:
    payload = "\0".join([PROMPT_VERSION, title.strip(), body.strip()])
    return hashlib.sha256(payload.encode("utf-8", errors="ignore")).hexdigest()[:20]


def build_context_prompt(title: str, body: str) -> str:
    body_snippet = str(body or "").strip()[:MAX_BODY_CHARS]
    return (
        f"Nota: {title.strip() or 'untitled'}\n\n"
        f"<MEMORIA>\n{body_snippet}\n</MEMORIA>\n\n"
        "Dame una sola oración corta, máximo 30 palabras, que ubique esta "
        "memoria dentro del archivo personal del usuario para mejorar búsqueda "
        "semántica. Respondé solo la oración."
    )


def sanitize_context(text: str) -> str:
    clean = " ".join(str(text or "").replace("\n", " ").split())
    if clean.startswith("```"):
        clean = clean.strip("`").strip()
    return clean[:MAX_SUMMARY_CHARS].strip()


def _read_cache(path: Path) -> dict[str, str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def _write_cache(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=True, sort_keys=True, indent=2), encoding="utf-8")


def get_or_generate_context(
    *,
    title: str,
    body: str,
    state_dir: Path,
    generate: Callable[[str], str],
) -> str:
    """Return cached generated context, or an empty string when disabled."""
    if not contextual_retrieval_enabled():
        return ""
    body_text = str(body or "").strip()
    if len(body_text) < MIN_BODY_CHARS:
        return ""

    key = context_hash(title, body_text)
    path = context_cache_path(state_dir)
    cache = _read_cache(path)
    cached = cache.get(key)
    if cached:
        return cached

    try:
        generated = sanitize_context(generate(build_context_prompt(title, body_text)))
    except Exception:
        return ""
    if not generated:
        return ""
    cache[key] = generated
    with suppress(OSError):
        _write_cache(path, cache)
    return generated


def prepend_context(embed_text: str, context: str) -> str:
    context = sanitize_context(context)
    if not context:
        return embed_text
    return f"{SUMMARY_MARKER} {context}]\n\n{embed_text}"


__all__ = [
    "PROMPT_VERSION",
    "SUMMARY_MARKER",
    "build_context_prompt",
    "context_hash",
    "contextual_retrieval_enabled",
    "get_or_generate_context",
    "prepend_context",
    "sanitize_context",
]
