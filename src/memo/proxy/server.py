"""The proxy itself. Import-safe without the [http] extra.

Contract with Claude Code, from its gateway documentation:
  * `anthropic-beta` must be forwarded verbatim or subscription auth breaks.
  * The response body must never be buffered: the byte-level watchdog aborts a
    stream after 180s of silence on the direct API.
  * Any failure forwards the original body rather than failing the request.
  * The incoming path AND query string are forwarded (a live probe against
    Claude Code v2.1.226 shows the real request is
    `/v1/messages?beta=true` — dropping the query would silently disable
    provider beta features).

None of the functions below may raise for a caller: `forward_headers`,
`rewrite_body`, `sniff_usage`, and `record_tool_usage` are each self-contained
try/except boundaries, not just "usually fine" — a caller gets a safe default,
never an exception.

Deliberately NOT `from __future__ import annotations`: PEP 563 stringifies
every annotation, and FastAPI resolves a route handler's parameter types via
`typing.get_type_hints()` against the function's *module* globals — not the
enclosing closure. `Request` is imported inside `build_app()` (never at
module level, so this stays import-safe without the [http] extra), so a
stringified `request: Request` can't be resolved and FastAPI silently
degrades it to a required query parameter named `request` — every real call
then 422s with "query request required". Reproduced directly; see
`src/memo/chat/http.py`, which hit and fixed the identical failure.
"""

import fcntl
import hashlib
import json
import logging
import time
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from memo.proxy import meter
from memo.proxy.plan import Context, TransformPlan, apply_all
from memo.proxy.registry import build_registry
from memo.proxy.zones import Zones, split, stable_head_fingerprint

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

TOOL_USAGE_SCHEMA = "memo.proxy.tool_usage.v1"
_SESSION_HEADER = "x-claude-code-session-id"
# Claude Code always sends _SESSION_HEADER (confirmed by the Task 1 probe);
# this is a defensive fallback for a client that doesn't. It must be STABLE
# across requests, never a per-turn value (a hash of the body, a per-request
# id, etc.) — that stability is what session-scoped state (the toolschemas
# keep-set freeze, the prefix-drift tracker below) depends on. It is
# per-*process*, not a single shared literal like the old "unknown": every
# header-less request within one running proxy maps to the same fallback
# session, generated once at import time.
_NO_SESSION_HEADER_FALLBACK = f"no-session-header-{uuid.uuid4().hex}"

# Per-session cached-HEAD fingerprint, for the runtime half of the design's
# cache-stability rule (docs/SPECS/2026-08-18-token-savings-proxy-context-
# compression-design.md Section 2: a mismatch across turns within one session
# is "a test failure and a logged runtime warning" — the test half already
# existed; this is the runtime half). Bounded like the toolschemas keep-set
# cache, for the same reason (a long-lived proxy process must not leak
# memory), keyed the same way (state_dir, session_key) to avoid collisions.
#
# Tracks `stable_head_fingerprint` (system + tools), NOT `prefix_fingerprint`
# (which also hashes `frozen_messages`). Provider prompt caching matches the
# LONGEST CACHED PREFIX, so appending new turns is cache-FRIENDLY — earlier
# cached bytes stay valid. `frozen_messages` growing every turn as the live
# window advances is normal traffic, not instability; comparing the full
# `prefix_fingerprint` across turns would warn on essentially every request
# in a real conversation, training everyone to ignore the warning — which is
# how the round-1 Critical this check exists to catch stayed invisible in
# the first place. What actually breaks a cache hit is REWRITING content
# that was already cached — `system` or `tools` changing mid-session — and
# that is exactly what `stable_head_fingerprint` isolates.
_MAX_TRACKED_HEAD_SESSIONS = 1000
_last_head_fingerprint: OrderedDict[tuple[str, str], str] = OrderedDict()


def _warn_on_prefix_drift(zones: Zones, ctx: Context) -> None:
    """Log (never raise, never block) if this session's cached-HEAD
    (system + tools) fingerprint differs from the one seen on this
    session's last request — see the module-level comment above for why
    this checks `stable_head_fingerprint`, not the stricter
    `prefix_fingerprint`.

    This is a diagnostic side effect, not part of the request's success
    path — any failure here is swallowed, matching every other function in
    this module.
    """
    try:
        key = (str(ctx.state_dir), ctx.session_key)
        fingerprint = stable_head_fingerprint(zones)
        previous = _last_head_fingerprint.get(key)
        if previous is not None and previous != fingerprint:
            _log.warning(
                "proxy: cached head (system/tools) changed within session "
                "(state_dir=%s, session=%s) — a stable-prefix transform "
                "rewrote it and will force a re-cache instead of a "
                "provider cache hit",
                ctx.state_dir,
                ctx.session_key,
            )
        _last_head_fingerprint[key] = fingerprint
        _last_head_fingerprint.move_to_end(key)
        while len(_last_head_fingerprint) > _MAX_TRACKED_HEAD_SESSIONS:
            _last_head_fingerprint.popitem(last=False)
    except Exception:
        _log.warning("proxy: prefix-stability check failed; continuing unaffected")


def forward_headers(headers: dict[str, str]) -> dict[str, str]:
    """Everything the client sent, minus hop-by-hop. Never logged."""
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


def rewrite_body(
    raw: bytes, ctx: Context, transforms: list[Any] | None = None
) -> tuple[bytes, TransformPlan]:
    """Apply transforms to a Messages payload. Returns the original on any doubt.

    `payload` (the parsed dict) is read again after `apply_all` runs, but only
    through `zones.to_payload(payload)` — never directly. Zones alias the
    caller's message dicts, so a transform mutating live-zone content in place
    also mutates `payload`; routing every post-transform read through
    `to_payload` (which overwrites system/tools/messages from the zones) keeps
    that aliasing from leaking into a re-serialized body that wasn't actually
    approved by `apply_all`.
    """
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("payload is not an object")
    except Exception:
        return raw, TransformPlan()

    zones = split(payload)
    plan = apply_all(zones, ctx, transforms if transforms is not None else build_registry())
    _warn_on_prefix_drift(zones, ctx)
    if not plan.applied:
        # No transform succeeded — forward the pristine original bytes, not a
        # re-serialization. A transform that mutated a zone's aliased dict
        # in place before raising must not have that mutation reach the wire.
        return raw, plan
    try:
        return (
            json.dumps(zones.to_payload(payload), ensure_ascii=False).encode("utf-8"),
            plan,
        )
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
    `message_delta`. We must not buffer the body, so each chunk is scanned as
    it goes by and the counters are merged with max() — later events carry the
    final totals. Never raises: a malformed chunk, a non-object JSON event
    (e.g. a bare JSON array happens to contain the substring `"usage"`), or
    any other unexpected shape is ignored rather than propagated.
    """
    try:
        if b'"usage"' not in chunk:
            return
        for line in chunk.split(b"\n"):
            if not line.startswith(b"data: "):
                continue
            try:
                event = json.loads(line[6:])
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            source = event.get("usage")
            if not isinstance(source, dict):
                message = event.get("message")
                source = message.get("usage") if isinstance(message, dict) else None
            if not isinstance(source, dict):
                continue
            for key, value in meter.usage_from_response({"usage": source}).items():
                into[key] = max(into.get(key, 0), value)
    except Exception:
        _log.warning("proxy: usage sniff failed on a chunk; continuing unaffected")


def tool_usage_path(state_dir: Path) -> Path:
    return Path(state_dir) / "proxy" / "tool_usage.json"


def _extract_tool_names(raw: bytes) -> list[str]:
    """Tool names from every `tool_use` content block in the request's messages."""
    try:
        payload = json.loads(raw)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return []
    names: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            name = block.get("name")
            if isinstance(name, str) and name:
                names.append(name)
    return names


def _read_tool_usage(path: Path) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            data = json.loads(text) if text.strip() else {}
            if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
                sessions = data["sessions"]
        except Exception:
            sessions = {}
    return {"schema": TOOL_USAGE_SCHEMA, "sessions": sessions}


def record_tool_usage(state_dir: Path, session_key: str, raw: bytes) -> None:
    """Record which tools appeared in this request's messages.

    Scans every `tool_use` content block and merges the names it finds into
    `<state_dir>/proxy/tool_usage.json`, keyed by `session_key` so a later
    task can ask "which tools appeared in the last N sessions"
    (`MEMO_PROXY_TOOL_WINDOW_SESSIONS`). A request with no tool_use blocks —
    the common case — writes nothing. Never raises: an unwritable state dir,
    a corrupt existing file, or malformed request JSON all fall through to a
    no-op.
    """
    try:
        names = _extract_tool_names(raw)
        if not names:
            return
        path = tool_usage_path(state_dir)
        lock_path = path.with_suffix(".json.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lockfile:
            fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
            try:
                data = _read_tool_usage(path)
                sessions = data["sessions"]
                existing = sessions.get(session_key)
                tools: set[str] = set()
                if isinstance(existing, dict) and isinstance(existing.get("tools"), list):
                    tools.update(t for t in existing["tools"] if isinstance(t, str))
                tools.update(names)
                sessions[session_key] = {"tools": sorted(tools), "ts": time.time()}
                tmp_path = path.with_suffix(".json.tmp")
                tmp_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                tmp_path.replace(path)
            finally:
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)
    except Exception:
        _log.warning("proxy: could not record tool usage")


def _request_key(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:32]


def _target(path: str, query: str) -> str:
    """Rebuild the upstream request target, query string included."""
    return f"{path}?{query}" if query else path


def build_app(upstream: str = UPSTREAM_DEFAULT) -> Any:
    """Construct the ASGI app. Imports the [http] extra lazily."""
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse

    from memo.config import Config
    from memo.flags import flag_bool, flag_float

    app = FastAPI(title="memo proxy", docs_url=None, redoc_url=None)
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(600.0))

    @app.post("/v1/messages")
    async def messages(request: Request) -> StreamingResponse:
        raw = await request.body()
        headers = forward_headers(dict(request.headers))
        cfg = Config.from_env()
        state_dir = cfg.state_dir

        session_key = request.headers.get(_SESSION_HEADER) or _NO_SESSION_HEADER_FALLBACK
        record_tool_usage(state_dir, session_key, raw)

        request_key = _request_key(raw)
        holdout = meter.is_holdout(request_key, flag_float("MEMO_PROXY_HOLDOUT_FRAC") or 0.0)

        plan = TransformPlan()
        body = raw
        # Recorded on the ledger row below so `summarize` can tell a genuinely
        # treated request apart from a passthrough one: with
        # MEMO_PROXY_ENABLED=0 (what `memo proxy off` sets) this is False and
        # `body` never diverges from `raw` — byte-identical to a control
        # request, so it must not be counted as treated either.
        rewrite_ran = not holdout and flag_bool("MEMO_PROXY_ENABLED")
        if rewrite_ran:
            # session_key (the real Claude Code session id, or the stable
            # per-process fallback above) — NOT request_key, which hashes the
            # raw body and is different on every single turn. A per-turn
            # session_key would defeat any per-session freeze a transform
            # does (e.g. toolschemas' keep-set), regardless of how correct
            # that transform's own caching is. request_key stays reserved for
            # holdout assignment and the measurement ledger below, where
            # per-request identity is exactly what's wanted.
            ctx = Context(state_dir=state_dir, session_key=session_key, project=None)
            body, plan = rewrite_body(raw, ctx)

        upstream_req = client.build_request(
            "POST",
            _target(request.url.path, request.url.query),
            content=body,
            headers=headers,
        )
        response = await client.send(upstream_req, stream=True)

        captured: dict[str, int] = {}

        async def _body() -> AsyncIterator[bytes]:
            try:
                async for chunk in _relay_chunks(response.aiter_raw()):
                    sniff_usage(chunk, captured)
                    yield chunk
            finally:
                try:
                    await response.aclose()
                except Exception:
                    _log.warning("proxy: error closing upstream response")
                # Only record a row when a real measurement was actually
                # observed: a successful response (2xx) that `sniff_usage`
                # actually found at least one usage counter in. Anything
                # else -- a 4xx from a payload a transform corrupted, a 5xx
                # overload, a non-streaming body `sniff_usage` can't parse --
                # has no honest usage numbers, and appending an all-zero row
                # would corrupt the very ledger the savings ratio is computed
                # from (worst case: it makes the failure mode that produced
                # it look like the BEST result). `captured` already carries
                # all four counters once any chunk was sniffed successfully
                # (`sniff_usage` always merges the full four-key dict), so
                # it doubles as both the presence check and the payload.
                if captured and 200 <= response.status_code < 300:
                    meter.append(
                        state_dir,
                        meter.Record(
                            request_key=request_key,
                            holdout=holdout,
                            rewritten=rewrite_ran,
                            transforms=plan.applied,
                            est_saved_tokens=plan.est_saved_tokens,
                            saved_by=plan.saved_by,
                            **captured,
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
