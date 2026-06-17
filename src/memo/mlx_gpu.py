"""Process-global GPU serialization for MLX forward passes.

MLX runs every array op on a single process-global Metal command stream
(`mlx.core`'s default stream). Two Python threads that force
materialization (`mx.eval`, `.tolist()`, `float(mx_scalar)`)
concurrently race that one command queue. When the race corrupts a
command buffer, Metal reports the error from inside its **async**
completion handler: `mlx::core::gpu::check_error` throws there, the
exception can't unwind back into Python, the C++ runtime calls
`std::terminate` -> `abort()`. The whole interpreter dies with SIGABRT
— not a catchable Python exception. (Observed crash: thread
`com.Metal.CompletionQueueDispatch` aborting while a
`ThreadPoolExecutor` thread was inside `mlx::core::eval`.)

memo hits this because the FastMCP HTTP server runs sync tool handlers
on a worker threadpool, so a `memory_search` (embed) and a
`memory_ask` (chat), or two concurrent searches, can enter MLX at once.
The recall daemon already serializes with its own priority lock, but
the MCP/CLI paths had no equivalent. This guard is that equivalent,
applied at the MLX boundary so **every** caller — MCP, CLI, daemon —
shares one serialization point.

Reentrant (`RLock`): `embed_query`->`embed` and `rerank`->`score`
re-enter on the same thread. The guard is a leaf — never acquire
another lock while holding it — so it cannot deadlock.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager

_GPU_LOCK = threading.RLock()


@contextmanager
def gpu_guard() -> Iterator[None]:
    """Serialize an MLX forward pass against all other GPU work in this process.

    Wrap the smallest region that builds and materializes MLX arrays
    (from `mx.array(...)` through the `.tolist()` / `float(...)` that
    forces evaluation). CPU-only tokenization/pooling setup can stay
    outside to keep the held window short.
    """
    _GPU_LOCK.acquire()
    try:
        yield
    finally:
        _GPU_LOCK.release()
