from __future__ import annotations

from pathlib import Path

from memo import embed_protocol


def _send_request(state_dir: Path, payload: dict, timeout: float) -> str | None:
    from memo.recall_socket import _socket_path

    return embed_protocol.send_request_line(_socket_path(state_dir), payload, timeout=timeout)


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
    if session_id is not None:
        req["session_id"] = session_id
    if turn is not None:
        req["turn"] = turn
    if client is not None:
        req["client"] = client
    return _send_request(state_dir, req, timeout)


def connect_and_send(state_dir: Path, payload: dict, timeout: float = 5.0) -> str | None:
    return _send_request(state_dir, payload, timeout)
