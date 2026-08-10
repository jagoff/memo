from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from memo.memory import Memory

if TYPE_CHECKING:
    from pathlib import Path

    from memo import emitted_ledger as el

    _LedgerBatch = tuple[Path, str, list[el.Entry]]

_log = logging.getLogger("memo.server")


async def run_synth[T](memory: Memory, ctx: Any, fn: Callable[[], T]) -> tuple[T, str]:
    """Run a sync synthesis closure with per-request client sampling.

    Returns ``(result, synthesizer_label)``. Inside MCP scope with
    MEMO_SAMPLING_SYNTH_ENABLED on and a sampling-capable client, synthesis
    routes to the client's model; otherwise MLX. Never raises because of
    sampling — ``fn``'s own errors pass through untouched.
    """
    import anyio

    from memo.sampling import sampling_scope, state_from_ctx

    state = state_from_ctx(ctx)

    def _run() -> T:
        if state is None:
            return fn()
        with sampling_scope(state):
            return fn()

    result = await anyio.to_thread.run_sync(_run)
    if state is not None and state.used_client:
        label = f"client:{state.model_hint}"
    else:
        # Attribution is cosmetic — never let a label lookup break the tool.
        try:
            label = f"mlx:{memory.cfg.llm_model}"
        except Exception:
            label = "mlx:unknown"
    return result, label


def now_ms() -> int:
    import time

    return int(time.time() * 1000)


def _mcp_client_name() -> str | None:
    """Best-effort name of the connected MCP client, from the initialize
    handshake's ``clientInfo.name`` (e.g. ``devin`` / ``opencode`` / ``devin-desktop``).
    Lets every MCP consult self-attribute even when the caller passed no
    ``source=`` and set no ``MEMO_SOURCE`` — so agent-class consumers stop
    showing up as the anonymous ``mcp:unknown``. Fully guarded: returns None off
    a request or on any FastMCP/MCP API drift."""
    try:
        from fastmcp.server.dependencies import get_context

        ctx = get_context()
        name = ctx.session.client_params.clientInfo.name  # type: ignore[union-attr]
        return (name or "").strip().lower() or None
    except Exception:
        return None


def log_consult(
    memory: Memory,
    *,
    tool: str,
    query: str,
    hits: list[dict[str, Any]],
    t0_ms: int,
    source: str = "",
) -> None:
    """Record an MCP consult into the shared recall ring buffer.

    ``source`` identifies the calling layer. Attribution precedence:
    explicit ``source=`` → ``MEMO_SOURCE`` env (mirrors the CLI's
    ``log_cli_consult``) → the MCP client's declared ``clientInfo.name``. The
    last tier means agent-class clients (devin / opencode / devin-desktop …) are
    attributed automatically from the handshake instead of showing up as the
    anonymous ``mcp:unknown`` consumer — no per-call args or env needed.
    """
    try:
        from memo.dashboard import append_recall_log
        from memo.flags import flag_str

        src = (source or flag_str("MEMO_SOURCE") or "").strip().lower() or _mcp_client_name()
        append_recall_log(
            memory.cfg.state_dir,
            prompt=query or "",
            hits=hits or [],
            via=f"mcp:{tool}",
            source=src,
            latency_ms=now_ms() - t0_ms,
        )
    except Exception as exc:
        _log.warning("consult recall-log write failed for %s: %s", tool, exc)


def _default_text_of(hit: dict[str, Any]) -> str:
    return str(hit.get("body") or "")


def _default_id_of(hit: dict[str, Any]) -> str:
    return str(hit.get("id") or "")


# Request-scoped staging area for ledger writes `apply_ledger` would
# otherwise commit immediately. FastMCP runs every sync tool's body (the
# default; see FunctionTool.run_in_thread) inside `anyio.to_thread.run_sync`,
# which executes in a COPIED contextvars context -- a `ContextVar.set()`
# performed there is invisible to the caller once the thread returns (this is
# standard `contextvars.Context.run()` semantics, not a fastmcp quirk;
# verified directly against a real FastMCP server + Client dispatch, not just
# reasoned about). So the response-budget middleware (`mcp_budget.py`) opens
# a stage by binding a fresh MUTABLE list here BEFORE dispatching into that
# thread, and `apply_ledger` APPENDS to the existing list rather than
# rebinding the var: mutating a shared object crosses the thread boundary;
# rebinding the var does not, because the thread's copied context is
# discarded when the thread returns.
#
# `None` (the default) means no middleware has opened a stage for this call
# -- true for every direct `apply_ledger(...)` call outside FastMCP's tool
# dispatch (every test in `test_emitted_ledger_apply.py`, and the `call_tool`
# fixture in `tests/conftest.py`, which resolves a tool's `.fn` and invokes
# it directly, bypassing `FunctionTool.run()` and the whole middleware chain
# with it). `apply_ledger` falls back to writing straight through in that
# case, so none of that existing coverage changes behavior.
_LEDGER_STAGE: contextvars.ContextVar[list[_LedgerBatch] | None] = contextvars.ContextVar(
    "_LEDGER_STAGE", default=None
)


def open_ledger_stage() -> contextvars.Token[list[_LedgerBatch] | None]:
    """Start staging ledger writes for the current request.

    Called by the response-budget middleware BEFORE ``call_next`` -- binding
    a fresh list here (rather than ``apply_ledger`` binding one lazily) is
    what makes the cross-thread visibility described above work at all.
    Always pair with exactly one of ``commit_ledger_stage`` /
    ``discard_ledger_stage`` in a ``finally``, so a stage can never survive
    past its request.
    """
    return _LEDGER_STAGE.set([])


def commit_ledger_stage(token: contextvars.Token[list[_LedgerBatch] | None]) -> None:
    """Write every batch staged since the matching ``open_ledger_stage`` to
    disk, then close the stage.

    No extra guarding needed around each write: ``emitted_ledger.append`` is
    already fail-open internally (a write it cannot make costs tokens, never
    correctness), so a bad batch here already can't raise into the
    middleware calling this.
    """
    from memo import emitted_ledger as el

    for state_dir, session_id, entries in _LEDGER_STAGE.get() or []:
        el.append(state_dir, session_id, entries)
    _LEDGER_STAGE.reset(token)


def discard_ledger_stage(token: contextvars.Token[list[_LedgerBatch] | None]) -> None:
    """Close the stage from the matching ``open_ledger_stage`` WITHOUT
    writing anything staged since.

    Called when the payload that would have justified those writes never
    actually reached the caller: the response-budget middleware substituted
    a ``response_budget_exceeded`` error, or the tool body raised. A residual
    gap stays open and is accepted: a failure AFTER the middleware returns
    (a transport-level error) can still leave a phantom entry -- this closes
    the routine, reproducible in-process case, not every path to the wire.
    """
    _LEDGER_STAGE.reset(token)


def apply_ledger(
    memory: Memory,
    tool: str,
    hits: list[dict[str, Any]],
    *,
    text_of: Callable[[dict[str, Any]], str] = _default_text_of,
    id_of: Callable[[dict[str, Any]], str] = _default_id_of,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop bodies this session has already put in the context window.

    Returns ``(hits_to_serialize, extra_payload_keys)``. The extra keys are
    empty whenever nothing was suppressed, so a cold session's payload is
    byte-identical to the pre-feature one.

    ``text_of``/``id_of`` read a hit's emitted text and memory id; override
    them for a tool whose rows don't use ``body``/``id`` (e.g. a ``snippet``
    field) — different MCP tools disagree on the key names. A hit whose
    accessor returns an empty string, default or custom, is always sent in
    full and NEVER recorded: an empty string hashes to a fixed value that
    self-matches regardless of real content, and an empty id would collapse
    distinct memories onto one shared ledger key — either would let this call
    or a later one silently digest content the model never saw.

    Fail-open on everything, including flag resolution itself: flag off, tool
    not allowlisted, no session id, a corrupt on-disk Markdown config (flag
    resolution reads it before falling back to the built-in default), or any
    other exception -> the caller's hits pass through untouched. A ledger
    that misbehaves must cost tokens, never content. (``_effective_session_id``
    never actually returns an empty id today — it mints a process-scoped
    fallback — so "no session id" is a defensive case, not a reachable one.)

    The actual disk write for newly-emitted hits is DEFERRED, not immediate:
    see ``_LEDGER_STAGE`` below. Inside a real MCP call, the write only lands
    once ``mcp_budget``'s response-budget middleware has confirmed the caller
    actually received these bodies -- never for a payload it substituted a
    ``response_budget_exceeded`` error for, and never for a call whose tool
    body went on to raise after this ran.
    """
    try:
        from memo.flags import flag_bool, flag_str

        if not flag_bool("MEMO_EMITTED_LEDGER"):
            return hits, {}
        allow = {
            t.strip() for t in (flag_str("MEMO_EMITTED_LEDGER_TOOLS") or "").split(",") if t.strip()
        }
        if tool not in allow:
            return hits, {}

        import time

        from memo import emitted_ledger as el
        from memo.server_session_patterns import _effective_session_id

        state_dir = memory.cfg.state_dir
        session_id = _effective_session_id()

        # An id-less or bodyless hit stays out of partition()'s view entirely,
        # so it can neither be digested itself nor be recorded under a shared
        # "" key that an unrelated memory could later match against.
        safe_hits = [h for h in hits if id_of(h) and text_of(h)]

        known = el.read(state_dir, session_id)
        part = el.partition(safe_hits, known, text_of=text_of, id_of=id_of)

        ref: str | None = None
        if part.full:
            now = int(time.time())
            ref = el.mint_ref([id_of(h) for h in part.full], now)
            entries = [el.Entry.for_text(id_of(h), text_of(h), ref, now, "mcp") for h in part.full]
            pending = _LEDGER_STAGE.get()
            if pending is None:
                # No middleware opened a stage for this call (unit tests that
                # call apply_ledger directly, or call_tool's .fn()-bypasses-
                # dispatch fixture) -- write straight through, matching this
                # function's pre-staging behavior.
                el.append(state_dir, session_id, entries)
            else:
                pending.append((state_dir, session_id, entries))

        digested_ids = {id(h) for h in part.digest}
        out = [h for h in hits if id(h) not in digested_ids]
        if not part.digest:
            return out, {}

        extra: dict[str, Any] = {
            "already_in_context": [
                {
                    "id": id_of(h),
                    "title": str(h.get("title") or ""),
                    "ref": known[id_of(h)].ref,
                }
                for h in part.digest
            ],
            "hint": (
                "bodies already emitted earlier in this session under the listed "
                "ref; call memo_get(id) for any you cannot see above"
            ),
        }
        if ref is not None:
            extra["cache_ref"] = ref
        return out, extra
    except Exception:
        return hits, {}
