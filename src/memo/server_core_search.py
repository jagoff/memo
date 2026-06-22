from __future__ import annotations

import logging
from typing import Any

from memo.memory import Memory
from memo.server_common import log_consult, now_ms

_log = logging.getLogger(__name__)


def _read_notification(memory: Memory) -> str:
    """Read pending idle-capture notification without deleting it."""
    notif_path = memory.cfg.state_dir / "pending_idle_notification.txt"
    try:
        return notif_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _auto_capture(memory: Memory) -> None:
    """Best-effort auto-capture: runs idle capture on the most recent session.

    Called as a side-effect from memo_unified_briefing, memo_search, memo_ask.
    Never raises — failures are logged at debug level and silently swallowed.
    Writes notification to pending file so the next tool call surfaces it.
    """
    from pathlib import Path

    from memo.capture import run_capture_incremental
    from memo.cli_capture import _write_capture_notification
    from memo.session import list_sessions

    try:
        state_dir = memory.cfg.state_dir
        sessions = list_sessions(state_dir, limit=1)
        if not sessions:
            return
        sid = sessions[0].get("session_id")
        transcript_raw = sessions[0].get("transcript_path")
        if not sid or not transcript_raw:
            return
        transcript = Path(transcript_raw)
        if not transcript.is_file():
            return
        result = run_capture_incremental(transcript, sid, debug=False)
        titles = result.get("saved_titles") or []
        if titles:
            _write_capture_notification(state_dir, titles, idle=True)
            _log.info("auto-capture: saved %d insight(s): %s", len(titles), "; ".join(titles[:3]))
    except Exception:
        _log.debug("auto-capture: skipped", exc_info=True)


def register(server: Any, memory: Memory) -> None:
    @server.tool()
    def memo_get_embedder_profile() -> dict[str, Any]:
        cfg = memory.cfg
        try:
            from consciousness_contracts import EmbedderProfile

            profile = EmbedderProfile(
                model_id=cfg.embedder_model,
                dims=int(cfg.embedder_dims),
                normalization="l2",
                provider="memo",
            )
            return profile.to_dict()
        except ImportError:
            return {
                "schema": "consciousness.embedder_profile.v1",
                "model_id": cfg.embedder_model,
                "dims": int(cfg.embedder_dims),
                "normalization": "l2",
                "max_seq_len": None,
                "quantization": None,
                "provider": "memo",
            }

    @server.tool()
    def memo_unified_briefing(cwd: str | None = None, source: str = "") -> dict[str, Any]:
        from memo.briefing import compact_text, synapse_briefing_lines

        t0 = now_ms()
        raw_lines = synapse_briefing_lines(cwd)
        markdown = compact_text("\n".join(raw_lines), max_chars=480)
        lines = markdown.splitlines() if markdown else []
        log_consult(
            memory,
            tool="unified_briefing",
            query=cwd or "briefing",
            hits=[],
            t0_ms=t0,
            source=source,
        )

        # Auto-capture: best-effort side effect — never blocks briefing.
        _auto_capture(memory)

        return {
            "available": bool(lines),
            "markdown": markdown,
            "lines": lines,
            "notification": _read_notification(memory),
        }

    @server.tool()
    def memo_search(
        query: str,
        limit: int = 10,
        type: str | None = None,
        body_chars: int = 280,
        mode: str = "hybrid",
        source: str = "",
    ) -> dict[str, Any]:
        t0 = now_ms()
        out: list[dict[str, Any]] = []
        for r in memory.search(query, limit=limit, type_=type, mode=mode):
            d = r.to_dict()
            body = d.get("body") or ""
            if body_chars >= 0 and len(body) > body_chars:
                d["body"] = body[:body_chars].rstrip() + "…"
                d["body_truncated"] = True
            out.append(d)
        log_consult(memory, tool="search", query=query, hits=out, t0_ms=t0, source=source)

        # Auto-capture: best-effort side effect on every search turn.
        _auto_capture(memory)

        # Read pending idle notification (best-effort, races with writer)
        notification = _read_notification(memory)

        return {
            "hits": out,
            "notification": notification,
        }

    @server.tool()
    def memo_search_trace(
        query: str,
        limit: int = 10,
        type: str | None = None,
        body_chars: int = 280,
        mode: str = "hybrid",
        source: str = "",
    ) -> dict[str, Any]:
        t0 = now_ms()
        envelope = memory.search_with_trace(query, limit=limit, type_=type, mode=mode)
        hits: list[dict[str, Any]] = []
        for r in envelope["hits"]:
            d = r.to_dict()
            body = d.get("body") or ""
            if body_chars >= 0 and len(body) > body_chars:
                d["body"] = body[:body_chars].rstrip() + "…"
                d["body_truncated"] = True
            hits.append(d)
        log_consult(memory, tool="search_trace", query=query, hits=hits, t0_ms=t0, source=source)
        return {"hits": hits, "trace": envelope["trace"]}

    @server.tool()
    def memo_rerank(
        query: str,
        hits: list[dict[str, Any]],
        top_n: int | None = None,
        body_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        return memory.rerank_hits(query, hits, top_n=top_n, body_chars=body_chars)

    @server.tool()
    def memo_embed_query(text: str) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("memo_embed_query: empty text")
        vec = memory.embedder.embed_query(text)
        return {"vector": vec, "dim": len(vec), "model": memory.cfg.embedder_model}

    @server.tool()
    def memo_embed_batch(texts: list[str]) -> dict[str, Any]:
        if not texts:
            return {"vectors": [], "dim": 0, "model": memory.cfg.embedder_model}
        vecs = memory.embedder.embed(texts)
        dim = len(vecs[0]) if vecs else 0
        return {"vectors": vecs, "dim": dim, "model": memory.cfg.embedder_model}

    @server.tool()
    def memo_ask(
        question: str,
        k: int = 5,
        type: str | None = None,
        snippet_chars: int = 800,
        include_repos: bool = True,
        source: str = "",
    ) -> dict[str, Any]:
        t0 = now_ms()
        res = memory.ask(
            question,
            k=k,
            type_=type,
            snippet_chars=snippet_chars,
            include_repos=include_repos,
        )
        out = res if isinstance(res, dict) else {"answer": str(res)}
        cites = out.get("citations") or out.get("sources") or []
        hit_dicts = [c for c in cites if isinstance(c, dict)]
        log_consult(memory, tool="ask", query=question, hits=hit_dicts, t0_ms=t0, source=source)

        # Auto-capture: best-effort side effect on every ask turn.
        _auto_capture(memory)

        # Read pending idle notification (best-effort, races with writer)
        out["notification"] = _read_notification(memory)

        return out

    @server.tool()
    def memo_chat_ask(
        question: str,
        k: int = 7,
        type: str | None = None,
        history: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        source: str = "",
    ) -> dict[str, Any]:
        t0 = now_ms()
        res = memory.chat_ask(
            question,
            k=k,
            type_=type,
            history=history,
            context=context,
        )
        out = res if isinstance(res, dict) else {"answer": str(res)}
        cites = out.get("citations") or out.get("sources") or []
        hit_dicts = [c for c in cites if isinstance(c, dict)]
        log_consult(memory, tool="chat_ask", query=question, hits=hit_dicts, t0_ms=t0, source=source)
        return out
