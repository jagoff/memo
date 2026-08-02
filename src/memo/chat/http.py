"""FastAPI surface for the chat UI. Import-safe without the [http] extra."""

import json
import time
import uuid
from pathlib import Path
from typing import Any


def _normalize_history(history: Any) -> list[dict[str, str]] | None:
    """Map inbound UI turns ({role, text} per types.ts:54-57) onto the
    {role, content} shape rewrite._history_topic reads, tolerating either key."""
    if not history:
        return None
    return [
        {
            "role": str(t.get("role", "")),
            "content": str(t.get("content") or t.get("text") or ""),
        }
        for t in history
        if isinstance(t, dict)
    ]


def _spa_response(
    path: str,
    *,
    dist: Path,
    file_response: Any,
    json_response: Any,
) -> Any:
    """Serve one SPA path while keeping unknown API routes as JSON 404s."""
    if path == "api" or path.startswith("api/"):
        return json_response({"error": "unknown API route"}, status_code=404)
    candidate = (dist / path).resolve()
    if path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
        return file_response(candidate)
    return file_response(dist / "index.html")


def build_app(memory: Any, *, dist: Path | None = None) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from starlette.middleware import Middleware

    from memo.chat.config import ChatConfig
    from memo.chat.feedback import (
        ChatFeedback,
        FeedbackStore,
        SourceVote,
        SourceVoteStore,
        question_key,
    )
    from memo.chat.pipeline import chat_stream
    from memo.chat.sessions import SessionStore, iso_ts
    from memo.http_auth import (
        LocalRequestGuardMiddleware,
        RateLimitMiddleware,
        RequestSizeLimitMiddleware,
    )

    cfg = ChatConfig.load(memory.cfg.state_dir)
    sessions = SessionStore(cfg.sessions_dir)
    # Chat has no bearer-token flow: constrain it to same-origin loopback HTTP,
    # reject DNS-rebinding/cross-site requests, and bound local request pressure.
    # `memo chat serve` independently rejects non-loopback bind addresses so the
    # network policy remains safe even before the first request reaches ASGI.
    app = FastAPI(
        title="memo chat",
        docs_url=None,
        redoc_url=None,
        middleware=[
            Middleware(RateLimitMiddleware),
            Middleware(LocalRequestGuardMiddleware),
            Middleware(RequestSizeLimitMiddleware),
        ],
    )

    async def _json_body(request: Request) -> dict[str, Any] | None:
        try:
            body = await request.json()
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    def sessions_history(session_id: str | None) -> list[dict[str, str]] | None:
        if not session_id:
            return None
        try:
            turns = sessions.get(session_id)
        except ValueError:
            return None
        return [{"role": t.get("role", ""), "content": t.get("text", "")} for t in turns][-12:]

    def requested_session_id(body: dict[str, Any]) -> str | None:
        """Validate an optional caller-owned id before doing retrieval work."""
        raw = body.get("chat_session_id")
        if raw is None or raw == "":
            return None
        if not isinstance(raw, str):
            raise ValueError("invalid chat_session_id")
        # SessionStore owns the path-safe id grammar. Reading here validates it
        # without duplicating that security boundary in the HTTP layer.
        try:
            sessions.get(raw)
        except ValueError as exc:
            raise ValueError("invalid chat_session_id") from exc
        return raw

    def request_inputs(
        body: dict[str, Any],
    ) -> tuple[str, int | None, list[dict[str, str]] | None]:
        """Validate the untyped JSON surface before it reaches retrieval."""
        raw_question = body.get("q")
        if not isinstance(raw_question, str) or not raw_question.strip():
            raise ValueError("q required and must be a string")
        if len(raw_question) > 4096:
            raise ValueError("q must be at most 4096 characters")
        raw_k = body.get("k")
        if raw_k is not None and (type(raw_k) is not int or not 1 <= raw_k <= 100):
            raise ValueError("k must be an integer between 1 and 100")
        raw_history = body.get("history")
        if raw_history is not None and not isinstance(raw_history, list):
            raise ValueError("history must be a list")
        return raw_question.strip(), raw_k, _normalize_history(raw_history)

    def _run(
        question: str,
        *,
        given_session_id: str | None,
        history: list[dict[str, str]] | None,
        k: int | None,
    ) -> list[dict[str, Any]]:
        session_id = given_session_id or uuid.uuid4().hex[:12]
        events = []
        for event in chat_stream(memory, question, history=history, k=k):
            if event.get("type") in ("context", "done"):
                event = {**event, "chat_session_id": session_id}
            events.append(event)
        done = next((e for e in events if e.get("type") == "done"), None)
        try:
            sessions.append_turn(session_id, "user", question)
            if done:
                sessions.append_turn(session_id, "assistant", str(done.get("answer", "")))
        except ValueError:
            pass
        return events

    @app.post("/api/ask/stream")
    async def ask_stream(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            question, k, explicit_history = request_inputs(body)
            given_session_id = requested_session_id(body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        session_id = given_session_id or uuid.uuid4().hex[:12]
        history = explicit_history or sessions_history(given_session_id)

        def _generate() -> Any:
            answer = ""
            for event in chat_stream(memory, question, history=history, k=k):
                if event.get("type") in ("context", "done"):
                    event = {**event, "chat_session_id": session_id}
                if event.get("type") == "done":
                    answer = str(event.get("answer", ""))
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            try:
                sessions.append_turn(session_id, "user", question)
                sessions.append_turn(session_id, "assistant", answer)
            except ValueError:
                pass

        return StreamingResponse(
            _generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/ask")
    async def ask(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            question, k, explicit_history = request_inputs(body)
            given_session_id = requested_session_id(body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        history = explicit_history or sessions_history(given_session_id)
        events = _run(
            question,
            given_session_id=given_session_id,
            history=history,
            k=k,
        )
        done = next((e for e in reversed(events) if e.get("type") in {"done", "error"}), None)
        return JSONResponse(done or {"type": "error", "message": "no events"})

    @app.post("/api/feedback")
    async def feedback(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        sources = body.get("sources", [])
        rating = body.get("rating")
        if not isinstance(sources, list):
            return JSONResponse({"error": "sources must be a list"}, status_code=400)
        if rating not in {"up", "down"}:
            return JSONResponse({"error": "rating must be 'up' or 'down'"}, status_code=400)
        fb = ChatFeedback(
            feedback_id=uuid.uuid4().hex[:12],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            chat_session_id=str(body.get("chat_session_id") or ""),
            turn_id=str(body.get("turn_id") or ""),
            query=str(body.get("query") or ""),
            answer=str(body.get("answer") or ""),
            source_ids=[
                s["id"] for s in sources if isinstance(s, dict) and isinstance(s.get("id"), str)
            ],
            rating=rating,
            correction_text=str(body.get("correction_text") or ""),
        )
        FeedbackStore(cfg.feedback_dir).append(fb)
        return {"ok": True, "feedback_id": fb.feedback_id}

    @app.post("/api/feedback/source")
    async def feedback_source(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        query = body.get("query")
        source_id = body.get("source_id")
        rating = body.get("rating")
        if not isinstance(query, str) or not query.strip() or len(query) > 4096:
            return JSONResponse(
                {"error": "query required as a string of at most 4096 characters"},
                status_code=400,
            )
        if not isinstance(source_id, str) or not source_id:
            return JSONResponse({"error": "source_id required as a string"}, status_code=400)
        if rating not in {"up", "down"}:
            return JSONResponse({"error": "rating must be 'up' or 'down'"}, status_code=400)
        try:
            embedding = memory.embedder.embed_query(query)
        except Exception:
            embedding = []
        vote = SourceVote(
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            question_key=question_key(query),
            query=query,
            source_id=source_id,
            rating=rating,
            query_embedding=list(embedding),
        )
        SourceVoteStore(cfg.feedback_dir).record(vote)
        return {"ok": True}

    @app.get("/api/sessions")
    async def list_sessions(limit: int = 50) -> Any:
        return {"sessions": sessions.list_sessions(limit=limit)}

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> Any:
        try:
            turns = sessions.get(session_id)
        except ValueError:
            return JSONResponse({"error": "invalid session id"}, status_code=400)
        return {
            "session_id": session_id,
            "turns": [
                {
                    "role": t.get("role", ""),
                    "text": t.get("text", ""),
                    "at": iso_ts(t["ts"]) if t.get("ts") is not None else None,
                }
                for t in turns
            ],
        }

    @app.post("/api/sessions/delete")
    async def delete_session(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            return {"ok": sessions.delete(str(body.get("session_id") or ""))}
        except ValueError:
            return JSONResponse({"error": "invalid session id"}, status_code=400)

    @app.post("/api/sessions/delete-all")
    async def delete_all_sessions() -> Any:
        return {"ok": True, "deleted": sessions.delete_all()}

    @app.get("/api/suggestions")
    async def suggestions(limit: int = 8) -> Any:
        chips = [{"label": q, "query": q} for q in sessions.recent_queries(limit=limit)]
        return {"chips": chips}

    @app.post("/api/memory/delete")
    async def memory_delete() -> Any:
        return JSONResponse({"error": "deferred to plan 2"}, status_code=501)

    @app.post("/api/insight/capture")
    async def insight_capture() -> Any:
        return JSONResponse({"error": "deferred to plan 2"}, status_code=501)

    if dist is not None and (dist / "index.html").exists():
        from fastapi.staticfiles import StaticFiles

        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{path:path}")
        async def spa(path: str) -> Any:
            return _spa_response(
                path,
                dist=dist,
                file_response=FileResponse,
                json_response=JSONResponse,
            )

    return app
