"""FastAPI surface for the chat UI. Import-safe without the [http] extra."""

import json
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse

_CHAT_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; object-src 'none'"
    ),
    "Referrer-Policy": "no-referrer",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
}


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
) -> Any:
    """Serve one SPA path while keeping unknown API routes as JSON 404s."""
    if path == "api" or path.startswith("api/"):
        return JSONResponse({"error": "unknown API route"}, status_code=404)
    candidate = (dist / path).resolve()
    if path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
        return FileResponse(candidate)
    return FileResponse(dist / "index.html")


class _ChatApi:
    """Bound chat HTTP handlers, split from app wiring for auditable contracts."""

    def __init__(self, memory: Any, cfg: Any, sessions: Any) -> None:
        self.memory = memory
        self.cfg = cfg
        self.sessions = sessions

    @staticmethod
    async def _json_body(request: Request) -> dict[str, Any] | None:
        try:
            body = await request.json()
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    def _sessions_history(self, session_id: str | None) -> list[dict[str, str]] | None:
        if not session_id:
            return None
        try:
            turns = self.sessions.get(session_id)
        except ValueError:
            return None
        return [{"role": t.get("role", ""), "content": t.get("text", "")} for t in turns][-12:]

    def _requested_session_id(self, body: dict[str, Any]) -> str | None:
        raw = body.get("chat_session_id")
        if raw is None or raw == "":
            return None
        if not isinstance(raw, str):
            raise ValueError("invalid chat_session_id")
        try:
            self.sessions.get(raw)
        except ValueError as exc:
            raise ValueError("invalid chat_session_id") from exc
        return raw

    @staticmethod
    def _request_inputs(
        body: dict[str, Any],
    ) -> tuple[str, int | None, list[dict[str, str]] | None]:
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

    def _prepare_ask(
        self,
        body: dict[str, Any],
    ) -> tuple[str, int | None, list[dict[str, str]] | None, str]:
        question, k, explicit_history = self._request_inputs(body)
        given_session_id = self._requested_session_id(body)
        session_id = given_session_id or uuid.uuid4().hex[:12]
        history = explicit_history or self._sessions_history(given_session_id)
        return question, k, history, session_id

    def _append_session(self, session_id: str, question: str, answer: str | None) -> None:
        try:
            self.sessions.append_turn(session_id, "user", question)
            if answer is not None:
                self.sessions.append_turn(session_id, "assistant", answer)
        except ValueError:
            pass

    def _run(
        self,
        question: str,
        *,
        session_id: str,
        history: list[dict[str, str]] | None,
        k: int | None,
    ) -> list[dict[str, Any]]:
        from memo.chat.pipeline import chat_stream

        events = []
        for event in chat_stream(self.memory, question, history=history, k=k):
            if event.get("type") in ("context", "done"):
                event = {**event, "chat_session_id": session_id}
            events.append(event)
        done = next((event for event in events if event.get("type") == "done"), None)
        answer = str(done.get("answer", "")) if done else None
        self._append_session(session_id, question, answer)
        return events

    def _stream_frames(
        self,
        question: str,
        *,
        session_id: str,
        history: list[dict[str, str]] | None,
        k: int | None,
    ) -> Iterator[str]:
        from memo.chat.pipeline import chat_stream

        answer = ""
        for event in chat_stream(self.memory, question, history=history, k=k):
            if event.get("type") in ("context", "done"):
                event = {**event, "chat_session_id": session_id}
            if event.get("type") == "done":
                answer = str(event.get("answer", ""))
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        self._append_session(session_id, question, answer)

    async def ask_stream(self, request: Request) -> Any:
        body = await self._json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            question, k, history, session_id = self._prepare_ask(body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        return StreamingResponse(
            self._stream_frames(
                question,
                session_id=session_id,
                history=history,
                k=k,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def ask(self, request: Request) -> Any:
        body = await self._json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            question, k, history, session_id = self._prepare_ask(body)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
        events = self._run(
            question,
            session_id=session_id,
            history=history,
            k=k,
        )
        done = next((event for event in reversed(events) if event.get("type") in {"done", "error"}), None)
        return JSONResponse(done or {"type": "error", "message": "no events"})

    async def feedback(self, request: Request) -> Any:
        from memo.chat.feedback import ChatFeedback, FeedbackStore

        body = await self._json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        sources = body.get("sources", [])
        rating = body.get("rating")
        if not isinstance(sources, list):
            return JSONResponse({"error": "sources must be a list"}, status_code=400)
        if rating not in {"up", "down"}:
            return JSONResponse({"error": "rating must be 'up' or 'down'"}, status_code=400)
        feedback = ChatFeedback(
            feedback_id=uuid.uuid4().hex[:12],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            chat_session_id=str(body.get("chat_session_id") or ""),
            turn_id=str(body.get("turn_id") or ""),
            query=str(body.get("query") or ""),
            answer=str(body.get("answer") or ""),
            source_ids=[
                source["id"]
                for source in sources
                if isinstance(source, dict) and isinstance(source.get("id"), str)
            ],
            rating=rating,
            correction_text=str(body.get("correction_text") or ""),
        )
        FeedbackStore(self.cfg.feedback_dir).append(feedback)
        return {"ok": True, "feedback_id": feedback.feedback_id}

    async def feedback_source(self, request: Request) -> Any:
        from memo.chat.feedback import SourceVote, SourceVoteStore, question_key

        body = await self._json_body(request)
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
            embedding = self.memory.embedder.embed_query(query)
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
        SourceVoteStore(self.cfg.feedback_dir).record(vote)
        return {"ok": True}

    async def list_sessions(self, limit: int = 50) -> Any:
        return {"sessions": self.sessions.list_sessions(limit=limit)}

    async def get_session(self, session_id: str) -> Any:
        from memo.chat.sessions import iso_ts

        try:
            turns = self.sessions.get(session_id)
        except ValueError:
            return JSONResponse({"error": "invalid session id"}, status_code=400)
        return {
            "session_id": session_id,
            "turns": [
                {
                    "role": turn.get("role", ""),
                    "text": turn.get("text", ""),
                    "at": iso_ts(turn["ts"]) if turn.get("ts") is not None else None,
                }
                for turn in turns
            ],
        }

    async def delete_session(self, request: Request) -> Any:
        body = await self._json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        try:
            return {"ok": self.sessions.delete(str(body.get("session_id") or ""))}
        except ValueError:
            return JSONResponse({"error": "invalid session id"}, status_code=400)

    async def delete_all_sessions(self) -> Any:
        return {"ok": True, "deleted": self.sessions.delete_all()}

    async def suggestions(self, limit: int = 8) -> Any:
        chips = [{"label": query, "query": query} for query in self.sessions.recent_queries(limit=limit)]
        return {"chips": chips}

    @staticmethod
    async def memory_delete() -> Any:
        return JSONResponse({"error": "deferred to plan 2"}, status_code=501)

    @staticmethod
    async def insight_capture() -> Any:
        return JSONResponse({"error": "deferred to plan 2"}, status_code=501)


def _register_api_routes(app: Any, api: _ChatApi) -> None:
    app.add_api_route("/api/ask/stream", api.ask_stream, methods=["POST"])
    app.add_api_route("/api/ask", api.ask, methods=["POST"])
    app.add_api_route("/api/feedback", api.feedback, methods=["POST"])
    app.add_api_route("/api/feedback/source", api.feedback_source, methods=["POST"])
    app.add_api_route("/api/sessions", api.list_sessions, methods=["GET"])
    app.add_api_route("/api/sessions/{session_id}", api.get_session, methods=["GET"])
    app.add_api_route("/api/sessions/delete", api.delete_session, methods=["POST"])
    app.add_api_route("/api/sessions/delete-all", api.delete_all_sessions, methods=["POST"])
    app.add_api_route("/api/suggestions", api.suggestions, methods=["GET"])
    app.add_api_route("/api/memory/delete", api.memory_delete, methods=["POST"])
    app.add_api_route("/api/insight/capture", api.insight_capture, methods=["POST"])


class _SpaEndpoint:
    def __init__(self, dist: Path) -> None:
        self.dist = dist

    async def __call__(self, path: str) -> Any:
        return _spa_response(path, dist=self.dist)


def _mount_spa(app: Any, dist: Path | None) -> None:
    if dist is None or not (dist / "index.html").exists():
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    app.add_api_route("/{path:path}", _SpaEndpoint(dist), methods=["GET"])


def build_app(memory: Any, *, dist: Path | None = None) -> Any:
    from fastapi import FastAPI
    from starlette.middleware import Middleware

    from memo.chat.config import ChatConfig
    from memo.chat.sessions import SessionStore
    from memo.http_auth import (
        LocalRequestGuardMiddleware,
        RateLimitMiddleware,
        RequestSizeLimitMiddleware,
        SecurityHeadersMiddleware,
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
            Middleware(SecurityHeadersMiddleware, headers=_CHAT_SECURITY_HEADERS),
            Middleware(RateLimitMiddleware),
            Middleware(LocalRequestGuardMiddleware),
            Middleware(RequestSizeLimitMiddleware),
        ],
    )
    _register_api_routes(app, _ChatApi(memory, cfg, sessions))
    _mount_spa(app, dist)
    return app
