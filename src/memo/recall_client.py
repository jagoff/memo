from __future__ import annotations

import logging
from pathlib import Path

from memo import embed_protocol

_log = logging.getLogger(__name__)


def _send_request(
    state_dir: Path, payload: dict, timeout: float, max_retries: int = 3
) -> str | None:
    from memo.recall_socket import _socket_path

    return embed_protocol.send_request_with_retry(
        _socket_path(state_dir),
        payload,
        timeout=timeout,
        max_retries=max_retries,
        base_delay=0.1,
    )


def connect_and_recall(
    state_dir: Path,
    prompt: str,
    cwd: str | None,
    timeout: float = 1.0,
    *,
    session_id: str | None = None,
    turn: int | None = None,
    client: str | None = None,
) -> str | None:
    req: dict = {"prompt": prompt, "cwd": cwd or ""}
    # Forward the knobs THIS process resolved. The daemon is a separate
    # long-lived process that does not inherit the hook's environment, so a
    # request without them leaves it resolving MEMO_RECALL_TOKEN_BUDGET /
    # _TOP_K from its own chain — which the LaunchAgent does not set. An
    # operator capping injections in settings.json silently got the defaults.
    # Only forward what was actually set, so an unset knob still lets the
    # daemon resolve its own and an older daemon ignores the extra keys.
    from memo.flags import REGISTRY, flag_int

    for key, flag in (("token_budget", "MEMO_RECALL_TOKEN_BUDGET"), ("top_k", "MEMO_RECALL_TOP_K")):
        value = flag_int(flag)
        # Forward only a value moved OFF its registry default: the daemon
        # resolves the same default on its own, so sending it would add wire
        # bytes for nothing and mask a future default change.
        if value is not None and value != REGISTRY[flag].default:
            req[key] = int(value)
    if session_id is not None:
        req["session_id"] = session_id
    if turn is not None:
        req["turn"] = turn
    if client is not None:
        req["client"] = client
    # No retries on the hook path: 4 attempts x 2s timeout + backoff was ~9s
    # worst case against the 5s hook budget. One failed attempt -> subprocess
    # fallback immediately.
    return _send_request(state_dir, req, timeout, max_retries=0)


def connect_and_send(state_dir: Path, payload: dict, timeout: float = 5.0) -> str | None:
    return _send_request(state_dir, payload, timeout)
