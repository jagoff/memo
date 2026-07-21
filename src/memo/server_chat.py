"""Validation helpers for the MCP HTTP chat streaming route."""

from __future__ import annotations

import json
from typing import Any

MAX_CHAT_QUESTION_CHARS = 32_768
MAX_CHAT_HISTORY_ITEMS = 128
MAX_CHAT_HISTORY_BYTES = 524_288
MAX_CHAT_CONTEXT_BYTES = 262_144
MAX_CHAT_JSON_DEPTH = 10


class ChatPayloadError(ValueError):
    """A chat request is malformed or exceeds semantic resource limits."""


def _require_valid_unicode(value: str, *, field: str) -> None:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ChatPayloadError(f"{field} contains invalid Unicode") from exc


def _json_depth(value: object) -> int:
    max_depth = 0
    stack: list[tuple[object, int]] = [(value, 1)]
    while stack:
        current, depth = stack.pop()
        max_depth = max(max_depth, depth)
        if isinstance(current, dict):
            stack.extend((child, depth + 1) for child in current.values())
        elif isinstance(current, list):
            stack.extend((child, depth + 1) for child in current)
    return max_depth


def _json_size(value: object) -> int:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return len(encoded.encode("utf-8"))


def _bounded_json_size(value: object, *, field: str) -> int:
    try:
        return _json_size(value)
    except UnicodeEncodeError as exc:
        raise ChatPayloadError(f"{field} contains invalid Unicode") from exc


def _validate_question(body: dict[str, Any]) -> str:
    question = body.get("question", "")
    if not isinstance(question, str):
        raise ChatPayloadError("question must be a string")
    _require_valid_unicode(question, field="question")
    if not question.strip():
        raise ChatPayloadError("empty question")
    if len(question) > MAX_CHAT_QUESTION_CHARS:
        raise ChatPayloadError("question is too large")
    return question


def _validate_k(body: dict[str, Any]) -> int:
    value = body.get("k", 7)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 100:
        raise ChatPayloadError("k must be an integer from 1 to 100")
    return value


def _validate_type(body: dict[str, Any]) -> str | None:
    value = body.get("type")
    if value is not None and not isinstance(value, str):
        raise ChatPayloadError("type must be a string or null")
    if value is not None:
        _require_valid_unicode(value, field="type")
    return value


def _validate_history(body: dict[str, Any]) -> list[dict[str, Any]] | None:
    value = body.get("history")
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ChatPayloadError("history must be an array of JSON objects or null")
    if len(value) > MAX_CHAT_HISTORY_ITEMS:
        raise ChatPayloadError("history has too many items")
    if _json_depth(value) > MAX_CHAT_JSON_DEPTH:
        raise ChatPayloadError("history is nested too deeply")
    if _bounded_json_size(value, field="history") > MAX_CHAT_HISTORY_BYTES:
        raise ChatPayloadError("history is too large")
    return value


def _validate_context(body: dict[str, Any]) -> dict[str, Any] | None:
    value = body.get("context")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ChatPayloadError("context must be a JSON object or null")
    if _json_depth(value) > MAX_CHAT_JSON_DEPTH:
        raise ChatPayloadError("context is nested too deeply")
    if _bounded_json_size(value, field="context") > MAX_CHAT_CONTEXT_BYTES:
        raise ChatPayloadError("context is too large")
    return value


def validate_chat_payload(
    body: object,
) -> tuple[str, int, str | None, list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Validate and unpack a chat request before invoking ``Memory``."""

    if not isinstance(body, dict):
        raise ChatPayloadError("body must be a JSON object")

    question = _validate_question(body)
    k = _validate_k(body)
    type_ = _validate_type(body)
    history = _validate_history(body)
    context = _validate_context(body)
    return question, k, type_, history, context
