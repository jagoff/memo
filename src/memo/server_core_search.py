from __future__ import annotations

from typing import Any

from memo.memory import Memory
from memo.server_common import log_consult, now_ms


def register(server: Any, memory: Memory) -> None:
    @server.tool()
    def memory_get_embedder_profile() -> dict[str, Any]:
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
    def memory_unified_briefing(cwd: str | None = None, source: str = "") -> dict[str, Any]:
        from memo.briefing import synapse_briefing_lines

        t0 = now_ms()
        lines = synapse_briefing_lines(cwd)
        log_consult(
            memory,
            tool="unified_briefing",
            query=cwd or "briefing",
            hits=[],
            t0_ms=t0,
            source=source,
        )
        return {
            "available": bool(lines),
            "markdown": "\n".join(lines),
            "lines": lines,
        }

    @server.tool()
    def memory_search(
        query: str,
        limit: int = 10,
        type: str | None = None,
        body_chars: int = 280,
        mode: str = "hybrid",
        source: str = "",
    ) -> list[dict[str, Any]]:
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
        return out

    @server.tool()
    def memory_search_trace(
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
    def memory_rerank(
        query: str,
        hits: list[dict[str, Any]],
        top_n: int | None = None,
        body_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        return memory.rerank_hits(query, hits, top_n=top_n, body_chars=body_chars)

    @server.tool()
    def memory_embed_query(text: str) -> dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("memory_embed_query: empty text")
        vec = memory.embedder.embed_query(text)
        return {"vector": vec, "dim": len(vec), "model": memory.cfg.embedder_model}

    @server.tool()
    def memory_embed_batch(texts: list[str]) -> dict[str, Any]:
        if not texts:
            return {"vectors": [], "dim": 0, "model": memory.cfg.embedder_model}
        vecs = memory.embedder.embed(texts)
        dim = len(vecs[0]) if vecs else 0
        return {"vectors": vecs, "dim": dim, "model": memory.cfg.embedder_model}

    @server.tool()
    def memory_ask(
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
        return out

    @server.tool()
    def memory_chat_ask(
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
