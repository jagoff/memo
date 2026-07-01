"""HTTP REST API — expose memo operations as HTTP endpoints.

Provides a standalone HTTP server (not MCP) for external clients.
Run via: `memo http-api --port 8080`

Endpoints mirror MCP tools but return plain JSON for any HTTP client.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from memo import __version__
from memo.config import Config
from memo.memory import Memory

_log = logging.getLogger(__name__)

# Lazy-initialized memory instance (constructed on first request).
_memory: Memory | None = None
# Endpoints are sync handlers running on FastAPI's threadpool: without the
# lock, concurrent first requests would each build a Memory (duplicate sqlite
# conns + a multi-GB embedder load for the loser instance).
_memory_lock = threading.Lock()


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


class SaveInput(BaseModel):
    content: str
    title: str | None = None
    type: str = "note"
    tags: list[str] | None = None


class SearchInput(BaseModel):
    query: str
    limit: int = 5
    mode: str = "hybrid"


# --- Health ---


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


# --- Core CRUD ---


@app.post("/api/memory")
def save_memory(input_: SaveInput) -> dict[str, Any]:
    mem = _get_memory()
    result = mem.save(
        content=input_.content,
        title=input_.title,
        type_=input_.type,
        tags=input_.tags,
    )
    return {"id": result.id, "title": result.title, "status": "saved"}


@app.get("/api/memory/{id_}")
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


@app.get("/api/memory")
def list_memory(limit: int = 20, type_: str | None = None) -> dict[str, Any]:
    mem = _get_memory()
    recs = mem.list(limit=limit, type_=type_)
    return {
        "memories": [
            {"id": r.id, "title": r.title, "type": r.type}
            for r in recs
        ]
    }


@app.delete("/api/memory/{id_}")
def delete_memory(id_: str) -> dict[str, Any]:
    mem = _get_memory()
    mem.delete(id_)
    return {"id": id_, "status": "deleted"}


# --- Search ---


@app.post("/api/search")
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


@app.get("/api/session")
def list_sessions(limit: int = 10) -> dict[str, Any]:
    from memo.session import list_sessions as _list

    mem = _get_memory()
    sessions = _list(mem.cfg.state_dir, limit=limit)
    return {"sessions": sessions}


# --- Stats ---


@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    mem = _get_memory()
    return {
        "total": mem.store.count(),
        "by_type": {},
    }


# --- Contradiction detection ---


@app.post("/api/contradict/scan")
def scan_contradictions(
    top_k: int = 5,
    sim_floor: float = 0.55,
    confidence_threshold: float = 0.7,
    min_days_apart: int = 1,
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


@app.post("/api/backup")
def create_backup(compress: bool = True, name: str | None = None) -> dict[str, Any]:
    """Create a backup of the vault."""
    mem = _get_memory()
    metadata = mem.backup.create_backup(compress=compress, name=name)
    return {"name": metadata.name, "size": metadata.size, "checksum": metadata.checksum}


@app.get("/api/backup")
def list_backups() -> dict[str, Any]:
    """List all backups."""
    mem = _get_memory()
    backups = mem.backup.list_backups()
    return {"backups": [{"name": b.name, "size": b.size} for b in backups]}


def run_server(port: int = 8080, host: str = "127.0.0.1") -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
