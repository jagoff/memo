from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from memo.errors import QueueFullError, StorageError
from memo.server_write_coordinator import (
    McpWriteCoordinator,
    make_write_coordinator_middleware,
)


@pytest.mark.asyncio
async def test_bounded_fifo_rejects_before_running_and_reports_metrics():
    coordinator = McpWriteCoordinator(1)
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def first():
        calls.append("first")
        started.set()
        await release.wait()
        return 1

    async def second():
        calls.append("second")
        return 2

    one = asyncio.create_task(coordinator.submit(first))
    await started.wait()
    two = asyncio.create_task(coordinator.submit(second))
    await asyncio.sleep(0)
    with pytest.raises(QueueFullError) as exc:
        await coordinator.submit(second)
    assert exc.value.retryable is True
    assert calls == ["first"]

    release.set()
    assert await one == 1
    assert await two == 2
    assert calls == ["first", "second"]
    assert coordinator.snapshot()["rejected"] == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_queued_cancellation_is_skipped():
    coordinator = McpWriteCoordinator(1)
    started = asyncio.Event()
    release = asyncio.Event()
    calls: list[str] = []

    async def first():
        started.set()
        await release.wait()

    async def queued():
        calls.append("queued")

    one = asyncio.create_task(coordinator.submit(first))
    await started.wait()
    two = asyncio.create_task(coordinator.submit(queued))
    await asyncio.sleep(0)
    two.cancel()
    with pytest.raises(asyncio.CancelledError):
        await two
    release.set()
    await one
    await asyncio.sleep(0)
    assert calls == []
    assert coordinator.snapshot()["canceled"] == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_started_job_finishes_with_real_outcome_after_caller_cancel():
    coordinator = McpWriteCoordinator(1)
    started = asyncio.Event()
    release = asyncio.Event()

    async def mutation():
        started.set()
        await release.wait()
        return "committed"

    task = asyncio.create_task(coordinator.submit(mutation))
    await started.wait()
    task.cancel()
    release.set()
    assert await task == "committed"
    assert coordinator.snapshot()["completed"] == 1
    await coordinator.close()


@pytest.mark.asyncio
async def test_unexpected_worker_exception_is_safe_typed_error():
    coordinator = McpWriteCoordinator(1)

    async def boom():
        raise RuntimeError("secret internal detail")

    with pytest.raises(StorageError, match="failed safely") as exc:
        await coordinator.submit(boom)
    assert "secret internal detail" not in str(exc.value)
    await coordinator.close()


@pytest.mark.asyncio
async def test_middleware_bypasses_read_only_and_coordinates_writes():
    coordinator = McpWriteCoordinator(1)

    class Server:
        async def get_tool(self, name):
            return SimpleNamespace(annotations=SimpleNamespace(readOnlyHint=name == "read"))

    middleware = make_write_coordinator_middleware(Server(), coordinator)
    called: list[str] = []

    async def next_call(context):
        called.append(context.message.name)
        return context.message.name

    def context(name):
        return SimpleNamespace(message=SimpleNamespace(name=name))

    assert await middleware.on_call_tool(context("read"), next_call) == "read"
    assert coordinator.snapshot()["submitted"] == 0
    assert await middleware.on_call_tool(context("write"), next_call) == "write"
    assert coordinator.snapshot()["submitted"] == 1
    assert called == ["read", "write"]
    await coordinator.close()


@pytest.mark.asyncio
async def test_load_is_strict_fifo_and_leaves_no_pending_jobs():
    coordinator = McpWriteCoordinator(32)
    order: list[int] = []

    async def mutation(index: int) -> int:
        await asyncio.sleep(0)
        order.append(index)
        return index

    tasks = [
        asyncio.create_task(coordinator.submit(lambda index=index: mutation(index)))
        for index in range(32)
    ]
    assert await asyncio.gather(*tasks) == list(range(32))
    snapshot = coordinator.snapshot()
    assert order == list(range(32))
    assert snapshot["completed"] == 32
    assert snapshot["rejected"] == 0
    assert snapshot["queue_depth"] == 0
    await coordinator.close()
