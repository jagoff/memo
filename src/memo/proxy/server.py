"""The proxy itself. Import-safe without the [http] extra.

Contract with Claude Code, from its gateway documentation:
  * `anthropic-beta` must be forwarded verbatim or subscription auth breaks.
  * The response body must never be buffered: the byte-level watchdog aborts a
    stream after 180s of silence on the direct API.
  * Any failure forwards the original body rather than failing the request.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from memo.proxy.plan import REGISTRY, Context, TransformPlan, apply_all
from memo.proxy.zones import split

_log = logging.getLogger(__name__)

UPSTREAM_DEFAULT = "https://api.anthropic.com"

# Dropped because they describe *this* connection, not the upstream one.
HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
    }
)


def forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Everything the client sent, minus hop-by-hop. Never logged."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def rewrite_body(
    raw: bytes, ctx: Context, transforms: list[Any] | None = None
) -> tuple[bytes, TransformPlan]:
    """Apply transforms to a Messages payload. Returns the original on any doubt."""
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
    except Exception:
        return raw, TransformPlan()

    zones = split(payload)
    plan = apply_all(zones, ctx, transforms if transforms is not None else REGISTRY)
    if not plan.applied:
        return raw, plan
    try:
        return json.dumps(
            zones.to_payload(payload), ensure_ascii=False
        ).encode("utf-8"), plan
    except Exception:
        _log.warning("proxy: could not re-encode rewritten payload; forwarding original")
        return raw, TransformPlan()


async def _relay_chunks(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    """Pass bytes straight through. Deliberately does no accumulation."""
    async for chunk in source:
        yield chunk


def sniff_usage(chunk: bytes, into: dict[str, int]) -> None:
    """Pick provider usage counters out of a passing SSE chunk.

    Streaming responses carry `usage` on `message_start` and again on
    `message_delta`. We must not buffer the body, so each chunk is scanned as it
    goes by and the counters are merged with max() — later events carry the
    final totals. A malformed chunk is ignored.
    """
    if b'"usage"' not in chunk:
        return
    for line in chunk.split(b"\n"):
        if not line.startswith(b"data: "):
            continue
        try:
            event = json.loads(line[6:])
        except (json.JSONDecodeError, ValueError):
            continue
        source = event.get("usage")
        if not isinstance(source, dict):
            message = event.get("message")
            source = message.get("usage") if isinstance(message, dict) else None
        if not isinstance(source, dict):
            continue
        from memo.proxy.meter import usage_from_response

        for key, value in usage_from_response({"usage": source}).items():
            into[key] = max(into.get(key, 0), value)


def build_app(upstream: str = UPSTREAM_DEFAULT) -> Any:
    """Construct the ASGI app. Imports the [http] extra lazily."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    from memo.config import Config
    from memo.flags import flag_bool, flag_float
    from memo.proxy import meter

    app = FastAPI(title="memo proxy", docs_url=None, redoc_url=None)
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(600.0))

    @app.post("/v1/messages")
    async def messages(request: Request) -> StreamingResponse:
        raw = await request.body()
        headers = forward_headers(dict(request.headers))
        cfg = Config.from_env()
        state_dir = cfg.state_dir

        request_key = _request_key(raw)
        holdout = meter.is_holdout(request_key, flag_float("MEMO_PROXY_HOLDOUT_FRAC") or 0.0)

        plan = TransformPlan()
        body = raw
        if not holdout and flag_bool("MEMO_PROXY_ENABLED"):
            ctx = Context(state_dir=state_dir, session_key=request_key, project=None)
            body, plan = rewrite_body(raw, ctx)

        upstream_req = client.build_request(
            "POST", "/v1/messages", content=body, headers=headers
        )
        response = await client.send(upstream_req, stream=True)

        captured: dict[str, int] = {}

        async def _body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _relay_chunks(response.aiter_raw()):
                    sniff_usage(chunk, captured)
                    yield chunk
            finally:
                await response.aclose()
                usage = meter.usage_from_response({})
                usage.update(captured)
                meter.append(
                    state_dir,
                    meter.Record(
                        request_key=request_key,
                        holdout=holdout,
                        transforms=plan.applied,
                        est_saved_tokens=plan.est_saved_tokens,
                        **usage,
                    ),
                )

        return StreamingResponse(
            _body(),
            status_code=response.status_code,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("content-length", "content-encoding")
            },
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def _request_key(raw: bytes) -> str:
    import hashlib

    return hashlib.sha256(raw).hexdigest()[:32]
