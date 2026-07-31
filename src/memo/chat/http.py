"""FastAPI surface for the chat UI. Import-safe without the [http] extra."""

import json
import time
import uuid
from pathlib import Path
from typing import Any


def build_app(memory: Any, *, dist: Path | None = None) -> Any:
    from fastapi import FastAPI, Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

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

    cfg = ChatConfig.load(memory.cfg.state_dir)
    sessions = SessionStore(cfg.sessions_dir)
    app = FastAPI(title="memo chat", docs_url=None, redoc_url=None)

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

    def _run(body: dict[str, Any]) -> list[dict[str, Any]]:
        question = str(body.get("q") or "").strip()
        session_id = body.get("chat_session_id") or None
        history = body.get("history") or sessions_history(session_id)
        events = list(chat_stream(memory, question, history=history, k=body.get("k") or None))
        if session_id:
            done = next((e for e in events if e.get("type") == "done"), None)
            sessions.append_turn(session_id, "user", question)
            if done:
                sessions.append_turn(session_id, "assistant", str(done.get("answer", "")))
        return events

    @app.post("/api/ask/stream")
    async def ask_stream(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        question = str(body.get("q") or "").strip()
        if not question:
            return JSONResponse({"error": "q required"}, status_code=400)
        session_id = body.get("chat_session_id") or None
        history = body.get("history") or sessions_history(session_id)

        def _generate() -> Any:
            answer = ""
            for event in chat_stream(memory, question, history=history, k=body.get("k") or None):
                if event.get("type") == "done":
                    answer = str(event.get("answer", ""))
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if session_id:
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
        if not str(body.get("q") or "").strip():
            return JSONResponse({"error": "q required"}, status_code=400)
        events = _run(body)
        done = next((e for e in reversed(events) if e.get("type") in {"done", "error"}), None)
        return JSONResponse(done or {"type": "error", "message": "no events"})

    @app.post("/api/feedback")
    async def feedback(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        fb = ChatFeedback(
            feedback_id=uuid.uuid4().hex[:12],
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            chat_session_id=str(body.get("chat_session_id") or ""),
            turn_id=str(body.get("turn_id") or ""),
            query=str(body.get("query") or ""),
            answer=str(body.get("answer") or ""),
            source_ids=[str(s.get("id")) for s in body.get("sources") or [] if isinstance(s, dict)],
            rating=str(body.get("rating") or ""),
            correction_text=str(body.get("correction_text") or ""),
        )
        FeedbackStore(cfg.feedback_dir).append(fb)
        return {"ok": True, "feedback_id": fb.feedback_id}

    @app.post("/api/feedback/source")
    async def feedback_source(request: Request) -> Any:
        body = await _json_body(request)
        if body is None:
            return JSONResponse({"error": "invalid JSON body"}, status_code=400)
        query = str(body.get("query") or "")
        try:
            embedding = memory.embedder.embed_query(query) if query else []
        except Exception:
            embedding = []
        vote = SourceVote(
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            question_key=question_key(query),
            query=query,
            source_id=str(body.get("source_id") or ""),
            rating=str(body.get("rating") or ""),
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
            candidate = (dist / path).resolve()
            if path and candidate.is_file() and candidate.is_relative_to(dist.resolve()):
                return FileResponse(candidate)
            return FileResponse(dist / "index.html")

    return app
