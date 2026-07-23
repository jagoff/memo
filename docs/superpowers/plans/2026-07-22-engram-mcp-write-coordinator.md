# Engram Learnings — MCP Write Coordinator Plan

**Status:** completed on 2026-07-22; capacity 32 graduated to the default on
2026-07-23 after the measured load gate and explicit approval.

**Goal:** Add bounded, observable process-local backpressure to every mutating
MCP call while preserving the data-directory lock as cross-process authority.

## Contract

- Read-only tools bypass the coordinator.
- Queue-full is a typed retryable error returned before tool execution.
- Canceled queued calls never start; started calls finish with their real
  result; unexpected failures become a safe typed internal error.
- Depth, active jobs, rejection count, completed count, and wait latency are
  observable through a read-only diagnostics tool.
- Capacity comes from the flags registry. Default `0` disables coordination
  until load tests justify a production default.

## Implementation

1. Add `MEMO_MCP_WRITE_QUEUE_SIZE` to `flags.py`'s registry.
2. Implement an async FIFO coordinator with bounded admission and metrics.
3. Add FastMCP middleware that resolves tool annotations and coordinates every
   tool whose `readOnlyHint` is false.
4. Register a read-only coordinator status tool on every MCP profile.

## Gates

- Saturation rejects before mutation, FIFO execution, queued cancellation,
  started completion, read bypass, exception safety, and metrics.
- Existing MCP annotations and server lifecycle tests remain green.
