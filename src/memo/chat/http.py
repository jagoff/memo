"""FastAPI surface for the chat UI. Import-safe without the [http] extra."""

import asyncio
import json
import time
import uuid
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

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
_MAX_HEAVY_IN_FLIGHT = 4


def _strict_text(
    value: Any,
    field: str,
    *,
    required: bool = False,
    max_chars: int | None = None,
) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str) or (required and not value.strip()):
        raise ValueError(f"{field} required and must be a string")
    if max_chars is not None and len(value) > max_chars:
        raise ValueError(f"{field} must be at most {max_chars} characters")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{field} must contain valid UTF-8 text") from exc
    return value


def _normalize_history(history: Any) -> list[dict[str, str]] | None:
    """Map inbound UI turns ({role, text} per types.ts:54-57) onto the
    {role, content} shape rewrite._history_topic reads, tolerating either key."""
    if history is None:
        return None
    normalized = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role", "")
        content = turn.get("content") if turn.get("content") is not None else turn.get("text")
        if not isinstance(role, str) or not isinstance(content, str):
            continue
        normalized.append(
            {
                "role": _strict_text(role, "history role"),
                "content": _strict_text(content, "history content"),
            }
        )
    return normalized


def _source_ids(sources: list[Any]) -> list[str]:
    ids = []
    for source in sources:
        if not isinstance(source, dict) or "id" not in source:
            continue
        ids.append(_strict_text(source["id"], "sources[].id", required=True, max_chars=512))
    return ids


def _next_item(iterator: Iterator[str]) -> tuple[bool, str]:
    try:
        return True, next(iterator)
    except StopIteration:
        return False, ""


class _HeavyLease:
    """Idempotently return one chat-capacity slot."""

    def __init__(self, semaphore: asyncio.Semaphore) -> None:
        self.semaphore = semaphore
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.semaphore.release()


class _SessionLockState:
    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.users = 0


class _SessionLease:
    """Idempotently release one event-loop-local session lock."""

    def __init__(
        self,
        pool: "_SessionLockPool",
        session_id: str,
        state: _SessionLockState,
    ) -> None:
        self.pool = pool
        self.session_id = session_id
        self.state = state
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.state.lock.release()
        self.pool.drop(self.session_id, self.state)


class _SessionLockPool:
    """Serialize work per session without retaining locks for old sessions."""

    def __init__(self) -> None:
        self.states: dict[str, _SessionLockState] = {}

    async def acquire(self, session_id: str) -> _SessionLease:
        # Dictionary operations are atomic with respect to other tasks because
        # this pool is only touched on the owning asyncio event loop.
        state = self.states.get(session_id)
        if state is None:
            state = _SessionLockState()
            self.states[session_id] = state
        state.users += 1
        try:
            await state.lock.acquire()
        except BaseException:
            self.drop(session_id, state)
            raise
        return _SessionLease(self, session_id, state)

    def drop(self, session_id: str, state: _SessionLockState) -> None:
        state.users -= 1
        if state.users == 0 and self.states.get(session_id) is state:
            del self.states[session_id]


class _OperationLease:
    """Idempotently release a shared or exclusive operation gate lease."""

    def __init__(self, gate: "_OperationGate", *, exclusive: bool) -> None:
        self.gate = gate
        self.exclusive = exclusive
        self.released = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        self.gate.release(exclusive=self.exclusive)


class _OperationGate:
    """Fair event-loop-local reader/writer gate for session mutations."""

    def __init__(self) -> None:
        self.readers = 0
        self.writer = False
        self.waiting_writers = 0
        self.changed = asyncio.Event()

    def _wake(self) -> None:
        previous = self.changed
        self.changed = asyncio.Event()
        previous.set()

    async def acquire_shared(self) -> _OperationLease:
        while self.writer or self.waiting_writers:
            changed = self.changed
            await changed.wait()
        self.readers += 1
        return _OperationLease(self, exclusive=False)

    async def acquire_exclusive(self) -> _OperationLease:
        self.waiting_writers += 1
        acquired = False
        try:
            while self.writer or self.readers:
                changed = self.changed
                await changed.wait()
            self.writer = True
            acquired = True
            return _OperationLease(self, exclusive=True)
        finally:
            self.waiting_writers -= 1
            if not acquired and not self.writer and self.waiting_writers == 0:
                self._wake()

    def release(self, *, exclusive: bool) -> None:
        if exclusive:
            self.writer = False
            self._wake()
            return
        self.readers -= 1
        if self.readers == 0:
            self._wake()


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


class _ChatApi:
    """Bound chat HTTP handlers, split from app wiring for auditable contracts."""

    def __init__(
        self,
        memory: Any,
        cfg: Any,
        sessions: Any,
        *,
        json_response: Any,
        streaming_response: Any,
        run_sync: Any,
    ) -> None:
        self.memory = memory
        self.cfg = cfg
        self.sessions = sessions
        self.json_response = json_response
        self.streaming_response = streaming_response
        self.run_sync = run_sync
        self.heavy_slots = asyncio.Semaphore(_MAX_HEAVY_IN_FLIGHT)
        self.session_locks = _SessionLockPool()
        self.operation_gate = _OperationGate()

    @staticmethod
    async def _json_body(request: Any) -> dict[str, Any] | None:
        try:
            body = await request.json()
        except Exception:
            return None
        return body if isinstance(body, dict) else None

    def _requested_session_id(self, body: dict[str, Any]) -> str | None:
        raw = body.get("chat_session_id")
        if raw is None or raw == "":
            return None
        if not isinstance(raw, str):
            raise ValueError("invalid chat_session_id")
        try:
            self.sessions.validate_id(raw)
        except ValueError as exc:
            raise ValueError("invalid chat_session_id") from exc
        return raw

    @staticmethod
    def _request_inputs(
        body: dict[str, Any],
    ) -> tuple[str, int | None, list[dict[str, str]] | None]:
        question = _strict_text(body.get("q"), "q", required=True, max_chars=4096)
        raw_k = body.get("k")
        if raw_k is not None and (type(raw_k) is not int or not 1 <= raw_k <= 100):
            raise ValueError("k must be an integer between 1 and 100")
        raw_history = body.get("history")
        if raw_history is not None and not isinstance(raw_history, list):
            raise ValueError("history must be a list")
        return question.strip(), raw_k, _normalize_history(raw_history)

    def _prepare_ask(
        self,
        body: dict[str, Any],
    ) -> tuple[str, int | None, list[dict[str, str]] | None, str, str | None]:
        question, k, explicit_history = self._request_inputs(body)
        given_session_id = self._requested_session_id(body)
        session_id = given_session_id or uuid.uuid4().hex[:12]
        return question, k, explicit_history, session_id, given_session_id

    async def _resolve_history(
        self,
        explicit_history: list[dict[str, str]] | None,
        given_session_id: str | None,
    ) -> list[dict[str, str]] | None:
        if explicit_history is not None or given_session_id is None:
            return explicit_history
        turns = await self.run_sync(self.sessions.get_recent, given_session_id, limit=12)
        return _normalize_history(turns)

    def _append_session(self, session_id: str, question: str, answer: str | None) -> None:
        try:
            if answer is not None:
                self.sessions.append_exchange(session_id, question, answer)
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
    ) -> Generator[str]:
        from memo.chat.pipeline import chat_stream

        answer = None
        for event in chat_stream(self.memory, question, history=history, k=k):
            if event.get("type") in ("context", "done"):
                event = {**event, "chat_session_id": session_id}
            if event.get("type") == "done":
                answer = str(event.get("answer", ""))
            yield f"data: {json.dumps(event, ensure_ascii=True)}\n\n"
        self._append_session(session_id, question, answer)

    async def _stream_frames_async(
        self,
        question: str,
        *,
        session_id: str,
        history: list[dict[str, str]] | None,
        k: int | None,
    ) -> Any:
        iterator = self._stream_frames(
            question,
            session_id=session_id,
            history=history,
            k=k,
        )
        try:
            while True:
                available, frame = await self.run_sync(_next_item, iterator)
                if not available:
                    return
                yield frame
        finally:
            iterator.close()

    async def _reserve_heavy_slot(self) -> _HeavyLease | None:
        if self.heavy_slots.locked():
            return None
        await self.heavy_slots.acquire()
        return _HeavyLease(self.heavy_slots)

    def _busy_response(self) -> Any:
        return self.json_response(
            {"error": "chat capacity exhausted; retry shortly"},
            status_code=503,
            headers={"Retry-After": "1"},
        )

    async def ask_stream(self, request: Any) -> Any:
        body = await self._json_body(request)
        if body is None:
            return self.json_response({"error": "invalid JSON body"}, status_code=400)
        try:
            question, k, explicit_history, session_id, given_session_id = self._prepare_ask(body)
        except ValueError:
            return self.json_response({"error": "invalid chat request"}, status_code=400)
        operation_lease = await self.operation_gate.acquire_shared()
        session_lease = None
        heavy_lease = None

        def release_all() -> None:
            if heavy_lease is not None:
                heavy_lease.release()
            if session_lease is not None:
                session_lease.release()
            operation_lease.release()

        try:
            session_lease = await self.session_locks.acquire(session_id)
            heavy_lease = await self._reserve_heavy_slot()
            if heavy_lease is None:
                release_all()
                return self._busy_response()
            history = await self._resolve_history(explicit_history, given_session_id)

            return self.streaming_response(
                self._stream_frames_async(
                    question,
                    session_id=session_id,
                    history=history,
                    k=k,
                ),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
                finalizer=release_all,
            )
        except BaseException:
            release_all()
            raise

    async def ask(self, request: Any) -> Any:
        body = await self._json_body(request)
        if body is None:
            return self.json_response({"error": "invalid JSON body"}, status_code=400)
        try:
            question, k, explicit_history, session_id, given_session_id = self._prepare_ask(body)
        except ValueError:
            return self.json_response({"error": "invalid chat request"}, status_code=400)
        operation_lease = await self.operation_gate.acquire_shared()
        session_lease = None
        heavy_lease = None
        try:
            session_lease = await self.session_locks.acquire(session_id)
            heavy_lease = await self._reserve_heavy_slot()
            if heavy_lease is None:
                return self._busy_response()
            history = await self._resolve_history(explicit_history, given_session_id)
            events = await self.run_sync(
                self._run,
                question,
                session_id=session_id,
                history=history,
                k=k,
            )
        finally:
            if heavy_lease is not None:
                heavy_lease.release()
            if session_lease is not None:
                session_lease.release()
            operation_lease.release()
        done = next(
            (event for event in reversed(events) if event.get("type") in {"done", "error"}), None
        )
        return self.json_response(done or {"type": "error", "message": "no events"})

    async def feedback(self, request: Any) -> Any:
        from memo.chat.feedback import ChatFeedback, FeedbackStore

        body = await self._json_body(request)
        if body is None:
            return self.json_response({"error": "invalid JSON body"}, status_code=400)
        sources = body.get("sources", [])
        rating = body.get("rating")
        if not isinstance(sources, list):
            return self.json_response({"error": "sources must be a list"}, status_code=400)
        if not isinstance(rating, str) or rating not in {"up", "down"}:
            return self.json_response({"error": "rating must be 'up' or 'down'"}, status_code=400)
        try:
            chat_session_id = _strict_text(
                body.get("chat_session_id"), "chat_session_id", max_chars=64
            )
            turn_id = _strict_text(body.get("turn_id"), "turn_id", max_chars=512)
            query = _strict_text(body.get("query"), "query", max_chars=4096)
            answer = _strict_text(body.get("answer"), "answer")
            correction_text = _strict_text(body.get("correction_text"), "correction_text")
            source_ids = _source_ids(sources)
        except ValueError:
            return self.json_response({"error": "invalid feedback request"}, status_code=400)
        feedback = ChatFeedback(
            feedback_id=uuid.uuid4().hex[:12],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            chat_session_id=chat_session_id,
            turn_id=turn_id,
            query=query,
            answer=answer,
            source_ids=source_ids,
            rating=rating,
            correction_text=correction_text,
        )
        await self.run_sync(FeedbackStore(self.cfg.feedback_dir).append, feedback)
        return {"ok": True, "feedback_id": feedback.feedback_id}

    async def feedback_source(self, request: Any) -> Any:
        from memo.chat.feedback import SourceVote, SourceVoteStore, question_key

        body = await self._json_body(request)
        if body is None:
            return self.json_response({"error": "invalid JSON body"}, status_code=400)
        rating = body.get("rating")
        if not isinstance(rating, str) or rating not in {"up", "down"}:
            return self.json_response({"error": "rating must be 'up' or 'down'"}, status_code=400)
        try:
            query = _strict_text(body.get("query"), "query", required=True, max_chars=4096)
            source_id = _strict_text(
                body.get("source_id"), "source_id", required=True, max_chars=512
            )
        except ValueError:
            return self.json_response({"error": "invalid source feedback request"}, status_code=400)

        def record_vote() -> None:
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

        lease = await self._reserve_heavy_slot()
        if lease is None:
            return self._busy_response()
        try:
            await self.run_sync(record_vote)
        finally:
            lease.release()
        return {"ok": True}

    async def list_sessions(self, limit: int = 50) -> Any:
        if not 1 <= limit <= 100:
            return self.json_response({"error": "limit must be between 1 and 100"}, status_code=400)
        sessions = await self.run_sync(self.sessions.list_sessions, limit=limit)
        return {"sessions": sessions}

    async def get_session(self, session_id: str) -> Any:
        from memo.chat.sessions import iso_ts

        try:
            turns = await self.run_sync(self.sessions.get, session_id)
        except ValueError:
            return self.json_response({"error": "invalid session id"}, status_code=400)
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

    async def delete_session(self, request: Any) -> Any:
        body = await self._json_body(request)
        if body is None:
            return self.json_response({"error": "invalid JSON body"}, status_code=400)
        session_id = body.get("session_id")
        if not isinstance(session_id, str):
            return self.json_response({"error": "invalid session id"}, status_code=400)
        try:
            self.sessions.validate_id(session_id)
        except ValueError:
            return self.json_response({"error": "invalid session id"}, status_code=400)
        operation_lease = await self.operation_gate.acquire_shared()
        session_lease = None
        try:
            session_lease = await self.session_locks.acquire(session_id)
            return {"ok": await self.run_sync(self.sessions.delete, session_id)}
        finally:
            if session_lease is not None:
                session_lease.release()
            operation_lease.release()

    async def delete_all_sessions(self) -> Any:
        operation_lease = await self.operation_gate.acquire_exclusive()
        try:
            deleted = await self.run_sync(self.sessions.delete_all)
            return {"ok": True, "deleted": deleted}
        finally:
            operation_lease.release()

    async def suggestions(self, limit: int = 8) -> Any:
        if not 1 <= limit <= 100:
            return self.json_response({"error": "limit must be between 1 and 100"}, status_code=400)
        queries = await self.run_sync(self.sessions.recent_queries, limit=limit)
        chips = [{"label": query, "query": query} for query in queries]
        return {"chips": chips}

    async def memory_delete(self) -> Any:
        return self.json_response({"error": "deferred to plan 2"}, status_code=501)

    async def insight_capture(self) -> Any:
        return self.json_response({"error": "deferred to plan 2"}, status_code=501)


def _request_endpoint(handler: Any, request_type: Any) -> Any:
    """Give FastAPI a concrete Request annotation without importing it eagerly."""

    async def endpoint(request: Any) -> Any:
        return await handler(request)

    endpoint.__name__ = handler.__name__
    endpoint.__annotations__["request"] = request_type
    return endpoint


def _register_api_routes(app: Any, api: _ChatApi, request_type: Any) -> None:
    app.add_api_route(
        "/api/ask/stream", _request_endpoint(api.ask_stream, request_type), methods=["POST"]
    )
    app.add_api_route("/api/ask", _request_endpoint(api.ask, request_type), methods=["POST"])
    app.add_api_route(
        "/api/feedback", _request_endpoint(api.feedback, request_type), methods=["POST"]
    )
    app.add_api_route(
        "/api/feedback/source",
        _request_endpoint(api.feedback_source, request_type),
        methods=["POST"],
    )
    app.add_api_route("/api/sessions", api.list_sessions, methods=["GET"])
    app.add_api_route("/api/sessions/{session_id}", api.get_session, methods=["GET"])
    app.add_api_route(
        "/api/sessions/delete",
        _request_endpoint(api.delete_session, request_type),
        methods=["POST"],
    )
    app.add_api_route("/api/sessions/delete-all", api.delete_all_sessions, methods=["POST"])
    app.add_api_route("/api/suggestions", api.suggestions, methods=["GET"])
    app.add_api_route("/api/memory/delete", api.memory_delete, methods=["POST"])
    app.add_api_route("/api/insight/capture", api.insight_capture, methods=["POST"])


class _SpaEndpoint:
    def __init__(self, dist: Path, *, file_response: Any, json_response: Any) -> None:
        self.dist = dist
        self.file_response = file_response
        self.json_response = json_response

    async def __call__(self, path: str) -> Any:
        return _spa_response(
            path,
            dist=self.dist,
            file_response=self.file_response,
            json_response=self.json_response,
        )


def _mount_spa(
    app: Any,
    dist: Path | None,
    *,
    file_response: Any,
    json_response: Any,
) -> None:
    if dist is None or not (dist / "index.html").exists():
        return
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
    app.add_api_route(
        "/{path:path}",
        _SpaEndpoint(dist, file_response=file_response, json_response=json_response),
        methods=["GET"],
    )


def build_app(memory: Any, *, dist: Path | None = None) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.concurrency import run_in_threadpool
    from fastapi.middleware import Middleware
    from fastapi.responses import FileResponse, Response, StreamingResponse

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

    class FinalizingStreamingResponse(StreamingResponse):
        """Run cleanup for every ASGI exit, including early disconnects."""

        def __init__(self, *args: Any, finalizer: Any, **kwargs: Any) -> None:
            self._memo_finalizer = finalizer
            super().__init__(*args, **kwargs)

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            try:
                await super().__call__(scope, receive, send)
            finally:
                self._memo_finalizer()

    def json_response(
        content: Any,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ) -> Any:
        # Escaping non-ASCII makes even malformed third-party/model surrogates
        # safe to encode as UTF-8 JSON instead of failing response construction.
        payload = json.dumps(
            content,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        return Response(
            content=payload,
            status_code=status_code,
            headers=headers,
            media_type="application/json",
        )

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
            Middleware(LocalRequestGuardMiddleware),
            Middleware(RateLimitMiddleware),
            Middleware(RequestSizeLimitMiddleware),
        ],
    )
    api = _ChatApi(
        memory,
        cfg,
        sessions,
        json_response=json_response,
        streaming_response=FinalizingStreamingResponse,
        run_sync=run_in_threadpool,
    )
    _register_api_routes(app, api, Request)
    _mount_spa(app, dist, file_response=FileResponse, json_response=json_response)
    return app
