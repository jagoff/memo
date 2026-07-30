# Task 2 implementation report — Memflow mutation fencing

Memflow BASE: `5426e8e5fce83d8ccf98fa7ecba3bcc634531ae2`

Memflow workspace:
`/Users/fer/repos/memflow/.worktrees/memflow-absorption-fence`

Memflow branch: `feat/memflow-absorption-fence`

Status: implementation in progress.

## Scope and invariants

- Add a signature-verified, fail-closed `FenceGate` and immutable cutover
  marker schemas without importing Memo at runtime.
- Preserve exact unfenced `ACTIVE` behavior and source-test parity.
- Fence every real mutation boundary while leaving read-only status available.
- Make admission validation and in-flight accounting atomic.
- Reject QUIESCING writes retryably and RETIRED startup/writes permanently.
- Do not install a marker, create a production sentinel, mutate live state,
  restart or unload a service, bind a listener, or alter configuration.
- Treat Task 2 as delivered only after focused parity, mypy, Ruff, and full
  Memflow verification pass. Independent review remains a separate acceptance
  gate.
