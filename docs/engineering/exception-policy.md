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
- `bounded evaluation trial`: record one provider failure and continue paired
  strategies so a single broken backend cannot invalidate the comparison.

Broad handlers are not acceptable for normal user-visible CLI or domain
failures. Those should raise or wrap `memo.errors.MemoError` subclasses so
CLI/MCP callers get a clear message.

The destructive write paths must not silently swallow failures unless a rollback,
receipt, or explicit recovery path is present.

### Which gate owns which file

Two gates enforce this policy and they do not overlap:

- `tests/test_dev_audit.py` owns the files in
  `dev_audit.BROAD_EXCEPTION_TARGET_FILES`. Every broad catch there is
  classified individually in `BROAD_EXCEPTION_ALLOWED`, with a comment stating
  its fail-open contract. Adding a justified catch is one edit: the
  classification.
- `scripts/quality_gate.py` owns everything else, as a per-file integer budget
  in `eval/quality_baseline.json`. It skips every site the first gate already
  audits, so a classified site is never billed twice.

Moving a file into `BROAD_EXCEPTION_TARGET_FILES` means classifying all of its
existing sites and regenerating the baseline; the file's integer budget then
drops to zero by construction.

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
