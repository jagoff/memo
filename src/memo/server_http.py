"""HTTP REST API — expose memo operations as HTTP endpoints.

Provides a standalone HTTP server (not MCP) for external clients.
Run via: `memo http-api --port 8080`

Endpoints mirror MCP tools but return plain JSON for any HTTP client.

Every API route except ``/health`` requires the shared memo bearer token.
Non-loopback binds require explicit acknowledgement and cannot disable auth.
"""

from __future__ import annotations

import logging
import threading
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import JSONResponse

from memo import __version__
from memo.config import Config
from memo.http_auth import (
    HttpAuthConfig,
    HttpAuthRejected,
    load_http_auth_config,
    validate_http_bind,
    verify_http_auth,
)
from memo.memory import Memory

_log = logging.getLogger(__name__)

# Lazy-initialized memory instance (constructed on first request).
_memory: Memory | None = None
# Endpoints are sync handlers running on FastAPI's threadpool: without the
# lock, concurrent first requests would each build a Memory (duplicate sqlite
# conns + a multi-GB embedder load for the loser instance).
_memory_lock = threading.Lock()
_auth_config: HttpAuthConfig | None = None

MAX_REQUEST_BYTES = 1_048_576
MAX_CONTENT_CHARS = 900_000


class _PayloadTooLarge(Exception):
    pass


class RequestSizeLimitMiddleware:
    """Reject oversized declared and streamed request bodies before parsing."""

    def __init__(self, app: Any, *, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        length_error = self._content_length_error(scope)
        if length_error is not None:
            await length_error(scope, receive, send)
            return

        received = 0
        response_started = False

        async def limited_receive() -> dict[str, Any]:
            nonlocal received
            message = await receive()
            if message.get("type") == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _PayloadTooLarge
            return message

        async def tracked_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, tracked_send)
        except _PayloadTooLarge:
            if not response_started:
                await self._reject(scope, receive, send)

    @staticmethod
    async def _reject(scope: dict[str, Any], receive: Any, send: Any) -> None:
        await JSONResponse({"detail": "Request body too large"}, status_code=413)(
            scope, receive, send
        )

    def _content_length_error(self, scope: dict[str, Any]) -> JSONResponse | None:
        raw_length = dict(scope.get("headers") or []).get(b"content-length")
        if raw_length is None:
            return None
        try:
            content_length = int(raw_length)
        except ValueError:
            return JSONResponse({"detail": "Invalid Content-Length"}, status_code=400)
        if content_length > self.max_bytes:
            return JSONResponse({"detail": "Request body too large"}, status_code=413)
        return None


def configure_auth(*, host: str = "127.0.0.1", allow_no_auth: bool = False) -> None:
    """Configure request auth before starting the REST server."""

    global _auth_config
    _auth_config = load_http_auth_config(host=host, allow_no_auth=allow_no_auth)


def _auth_dependency(authorization: str | None = Header(default=None)) -> None:
    global _auth_config
    if _auth_config is None:
        from memo.flags import flag_bool, flag_str

        allow_no_auth = flag_bool("MEMO_HTTP_ALLOW_NO_AUTH")
        host = flag_str("MEMO_HTTP_HOST") or "127.0.0.1"
        _auth_config = load_http_auth_config(host=host, allow_no_auth=allow_no_auth)
    try:
        verify_http_auth(authorization, _auth_config)
    except HttpAuthRejected as exc:
        raise HTTPException(
            status_code=401,
            detail=exc.detail,
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


def _get_memory() -> Memory:
    global _memory
    if _memory is None:
        with _memory_lock:
            if _memory is None:
                _memory = Memory(Config.from_env())
    return _memory


app = FastAPI(
    title="memo HTTP API",
    description="Local-first semantic memory REST API",
    version=__version__,
)
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=MAX_REQUEST_BYTES)

_AUTH = [Depends(_auth_dependency)]


class SaveInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    content: str = Field(min_length=1, max_length=MAX_CONTENT_CHARS)
    title: str | None = Field(default=None, max_length=500)
    type: str = Field(default="note", min_length=1, max_length=64)
    tags: list[Annotated[str, Field(min_length=1, max_length=128)]] | None = Field(
        default=None,
        max_length=64,
    )


class SearchInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=4096)
    limit: int = Field(default=5, ge=1, le=100)
    mode: Literal["hybrid", "vec", "bm25", "exact"] = "hybrid"


# --- Health ---


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


# --- Core CRUD ---


@app.post("/api/memory", dependencies=_AUTH)
def save_memory(input_: SaveInput) -> dict[str, Any]:
    mem = _get_memory()
    result = mem.save(
        content=input_.content,
        title=input_.title,
        type_=input_.type,
        tags=input_.tags,
    )
    return {"id": result.id, "title": result.title, "status": "saved"}


@app.get("/api/memory/{id_}", dependencies=_AUTH)
def get_memory(id_: str) -> dict[str, Any]:
    mem = _get_memory()
    rec = mem.get(id_)
    if not rec:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {
        "id": rec.id,
        "title": rec.title,
        "body": rec.body,
        "tags": rec.tags,
        "type": rec.type,
    }


@app.get("/api/memory", dependencies=_AUTH)
def list_memory(
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    type_: Annotated[str | None, Query(max_length=64)] = None,
) -> dict[str, Any]:
    mem = _get_memory()
    recs = mem.list(limit=limit, type_=type_)
    return {"memories": [{"id": r.id, "title": r.title, "type": r.type} for r in recs]}


@app.delete("/api/memory/{id_}", dependencies=_AUTH)
def delete_memory(id_: str) -> dict[str, Any]:
    mem = _get_memory()
    mem.delete(id_)
    return {"id": id_, "status": "deleted"}


# --- Search ---


@app.post("/api/search", dependencies=_AUTH)
def search_memory(input_: SearchInput) -> dict[str, Any]:
    mem = _get_memory()
    results = mem.search(
        query=input_.query,
        limit=input_.limit,
        mode=input_.mode,
    )
    return {
        "query": input_.query,
        "results": [
            {
                "id": r.id,
                "title": r.title,
                "body": r.body[:200] if r.body else "",
                "score": getattr(r, "score", None),
            }
            for r in results
        ],
    }


# --- Session ---


@app.get("/api/session", dependencies=_AUTH)
def list_sessions(limit: Annotated[int, Query(ge=1, le=100)] = 10) -> dict[str, Any]:
    from memo.session import list_sessions as _list

    mem = _get_memory()
    sessions = _list(mem.cfg.state_dir, limit=limit)
    return {"sessions": sessions}


# --- Stats ---


@app.get("/api/stats", dependencies=_AUTH)
def get_stats() -> dict[str, Any]:
    mem = _get_memory()
    return {
        "total": mem.store.count(),
        "by_type": mem.store.count_by_type(),
    }


# --- Contradiction detection ---


@app.post("/api/contradict/scan", dependencies=_AUTH)
def scan_contradictions(
    top_k: Annotated[int, Query(ge=1, le=100)] = 5,
    sim_floor: Annotated[float, Query(ge=0.0, le=1.0)] = 0.55,
    confidence_threshold: Annotated[float, Query(ge=0.0, le=1.0)] = 0.7,
    min_days_apart: Annotated[int, Query(ge=0, le=36_500)] = 1,
) -> dict[str, Any]:
    """Scan for contradictions between memories."""
    mem = _get_memory()
    result = mem.contradict_scanner.scan_corpus(
        top_k=top_k,
        sim_floor=sim_floor,
        confidence_threshold=confidence_threshold,
        min_days_apart=min_days_apart,
    )
    return {
        "scanned_memories": result.scanned_memories,
        "pairs_examined": result.pairs_examined,
        "pairs_inserted": result.pairs_inserted,
    }


# --- Backup / Git sync ---


@app.post("/api/backup", dependencies=_AUTH)
def create_backup(
    compress: bool = True,
    name: Annotated[str | None, Query(max_length=128)] = None,
) -> dict[str, Any]:
    """Create a backup of the vault."""
    mem = _get_memory()
    metadata = mem.backup.create_backup(compress=compress, name=name)
    return {
        "name": metadata.name,
        "size": metadata.compressed_size,
        "original_size": metadata.original_size,
        "memory_count": metadata.memory_count,
        "checksum": metadata.checksum,
    }


@app.get("/api/backup", dependencies=_AUTH)
def list_backups() -> dict[str, Any]:
    """List all backups."""
    mem = _get_memory()
    backups = mem.backup.list_backups()
    return {
        "backups": [
            {
                "name": b.name,
                "size": b.compressed_size,
                "original_size": b.original_size,
                "memory_count": b.memory_count,
            }
            for b in backups
        ]
    }


def run_server(
    port: int = 8080,
    host: str = "127.0.0.1",
    *,
    allow_no_auth: bool = False,
    allow_non_loopback: bool = False,
) -> None:
    import uvicorn

    configure_auth(host=host, allow_no_auth=allow_no_auth)
    assert _auth_config is not None
    validate_http_bind(host, _auth_config, allow_non_loopback=allow_non_loopback)
    uvicorn.run(app, host=host, port=port, log_level="info")
