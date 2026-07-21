# memo Exception And Env Flag Policy

## Broad Exception Policy

Broad `except Exception` handlers are allowed only when the layer explicitly
needs fault isolation. Each allowed site should have one of these intents:

- `hook hot path`: never block user work; log structured debug evidence.
- `daemon or maintenance best effort`: capture warning, receipt, or debug context.
- `optional dependency`: degrade gracefully when an optional package is absent.
- `cleanup path`: preserve the primary exception and avoid raising during cleanup.
- `derived index best effort`: auxiliary graph/fact/cache writes may log and
  skip when the primary markdown/sqlite memory write has already succeeded.

Broad handlers are not acceptable for normal user-visible CLI or domain
failures. Those should raise or wrap `memo.errors.MemoError` subclasses so
CLI/MCP callers get a clear message.

The destructive write paths must not silently swallow failures unless a rollback,
receipt, or explicit recovery path is present.

## Raw `MEMO_*` Env Reads

Normal behavioral flags must use `memo.flags`.

Raw `os.environ.get("MEMO_*")` is allowed only for:

- bootstrap/config discovery before the flags registry can safely be used
- storage/model configuration that belongs in `Config`
- tri-state checks where `flag_bool()` would collapse unset into `False`
- low-level cross-process setup where importing higher layers creates a cycle

Each allowed raw read is classified in `memo.dev_audit.RAW_MEMO_ENV_ALLOWED`.
New raw reads must either use `memo.flags`, move into `Config`, or add a clear
classification with a source comment.

Broad exception sites are inventoried by file, lexical scope, and ordinal
within that scope. This keeps the policy stable across formatting-only line
changes while still rejecting additional broad handlers.
