from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from memo.errors import MemoError, QueueFullError, StorageError

_log = logging.getLogger("memo.server")


@dataclass
class _Job:
    run: Callable[[], Awaitable[Any]]
    future: asyncio.Future[Any]
    queued_at: float
    started: bool = False
    canceled: bool = False


class McpWriteCoordinator:
    """Single-worker, bounded FIFO for process-local MCP mutation pressure."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self._queue: asyncio.Queue[_Job] | None = (
            asyncio.Queue(maxsize=self.capacity) if self.capacity else None
        )
        self._worker: asyncio.Task[None] | None = None
        self._active = 0
        self._submitted = 0
        self._started = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._canceled = 0
        self._wait_total_ms = 0.0
        self._wait_max_ms = 0.0

    @property
    def enabled(self) -> bool:
        return self._queue is not None

    async def submit(self, run: Callable[[], Awaitable[Any]]) -> Any:
        if self._queue is None:
            return await run()
        loop = asyncio.get_running_loop()
        self._ensure_worker(loop)
        job = _Job(run=run, future=loop.create_future(), queued_at=time.perf_counter())
        try:
            self._queue.put_nowait(job)
        except asyncio.QueueFull as exc:
            self._rejected += 1
            raise QueueFullError(
                f"MCP write queue is full (capacity={self.capacity}); retry the call"
            ) from exc
        self._submitted += 1
        try:
            return await asyncio.shield(job.future)
        except asyncio.CancelledError:
            if not job.started:
                job.canceled = True
                self._canceled += 1
                raise
            # Once mutation began, preserve its real outcome. The caller's task
            # has consumed its cancellation and can await the shielded result.
            return await asyncio.shield(job.future)

    def _ensure_worker(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._worker is None or self._worker.done():
            self._worker = loop.create_task(self._run(), name="memo-mcp-write-coordinator")

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            job = await self._queue.get()
            try:
                if job.canceled:
                    if not job.future.done():
                        job.future.cancel()
                    continue
                job.started = True
                self._started += 1
                self._active = 1
                wait_ms = (time.perf_counter() - job.queued_at) * 1000.0
                self._wait_total_ms += wait_ms
                self._wait_max_ms = max(self._wait_max_ms, wait_ms)
                try:
                    result = await job.run()
                except MemoError as exc:
                    self._failed += 1
                    if not job.future.done():
                        job.future.set_exception(exc)
                except Exception:
                    self._failed += 1
                    _log.exception("coordinated MCP write failed")
                    if not job.future.done():
                        job.future.set_exception(
                            StorageError("coordinated MCP write failed safely")
                        )
                else:
                    self._completed += 1
                    if not job.future.done():
                        job.future.set_result(result)
            finally:
                self._active = 0
                self._queue.task_done()

    def snapshot(self) -> dict[str, Any]:
        depth = self._queue.qsize() if self._queue is not None else 0
        mean = self._wait_total_ms / self._started if self._started else 0.0
        return {
            "enabled": self.enabled,
            "capacity": self.capacity,
            "queue_depth": depth,
            "active": self._active,
            "submitted": self._submitted,
            "started": self._started,
            "completed": self._completed,
            "failed": self._failed,
            "rejected": self._rejected,
            "canceled": self._canceled,
            "wait_mean_ms": round(mean, 3),
            "wait_max_ms": round(self._wait_max_ms, 3),
        }

    async def close(self) -> None:
        if self._worker is None:
            return
        if self._queue is not None:
            await self._queue.join()
        self._worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._worker
        self._worker = None


def make_write_coordinator_middleware(server: Any, coordinator: McpWriteCoordinator) -> Any:
    try:
        from fastmcp.server.middleware import Middleware
    except ImportError:  # pragma: no cover - supported FastMCP provides it
        return None

    class _WriteCoordinatorMiddleware(Middleware):
        async def on_call_tool(self, context: Any, call_next: Any) -> Any:
            name = str(getattr(context.message, "name", "") or "")
            tool = await server.get_tool(name)
            annotations = getattr(tool, "annotations", None)
            if annotations is not None and bool(getattr(annotations, "readOnlyHint", False)):
                return await call_next(context)
            return await coordinator.submit(lambda: call_next(context))

    return _WriteCoordinatorMiddleware()
