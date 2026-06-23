# Memflow operational receipts

Memo owns the corpus. Memflow owns operational continuity (cross-Mac
state, handoffs, presence). Receipts are how memo tells memflow "I
just did X" without orchestrating anything — memflow then replays the
breadcrumb trail across machines and agents.

## What gets receipted

| Operation | Trigger | Payload meta keys |
|---|---|---|
| `save` | `Memory.save()` (incl. `defer_embed`) | `id`, `type`, `tags`, `path`, `deferred`, `synapse_trace_id`, `synapse_route_reason`, `synapse_agent_id` |
| `update` | `Memory.update()` when any field changed | `id`, `type`, `title`, `delta_keys` |
| `delete` | `Memory.delete()` when the row existed | `id`, `type`, `title`, `path` |
| `reindex` | `Memory.reindex()` when ≥1 row was added or re-embedded | `checked`, `reindexed`, `added`, `skipped`, `force` |
| `repo index` | `memo repo index` (pre-existing, commit `8052e38`) | `repo_id`, `url`, `commit_sha`, file/chunk counts |

`memo repo index` has its own knob (`MEMO_MEMFLOW_RECEIPT`, default ON)
that predates this generalisation. The four ops above are gated by
`MEMO_EMIT_RECEIPTS` (default OFF) so existing single-Mac users see no
behaviour change.

## Wire shape

A receipt is a `memflow say <text> --channel memo-receipts --author memo --no-sync`
subprocess call. It is intentionally channel state, not a durable fact:

```
memflow say \
  "Memo saved memory abcd1234 (note): Astor — Informe TO [client=memo operation=save topic=memo-save ...]" \
  --channel memo-receipts \
  --author memo \
  --no-sync
```

Best-effort: no raise on subprocess errors, 5-second timeout. The
helper returns `{"ok": True, "path": ...}` on success and one of:

```python
{"ok": False, "skipped": True, "reason": "MEMO_EMIT_RECEIPTS not set"}
{"ok": False, "skipped": True, "reason": "memflow binary not found"}
{"ok": False, "skipped": True, "reason": "memflow project root not found"}
{"ok": False, "skipped": True, "reason": "disabled"}            # caller opt-out
{"ok": False, "error": "<truncated>"}                            # memflow exited non-zero
```

on skip / failure. Callers (memo's hot paths) don't inspect the
return today; they only fire-and-forget.

## Why opt-in by default

`save` / `update` / `delete` fire on every user-visible memo
operation. Default-ON would shell out to memflow per call —
~50–200ms latency added under load and a noisy event ledger when memo
runs without memflow nearby. `MEMO_EMIT_RECEIPTS=1` is the explicit
opt-in for users on machines that want the cross-Mac breadcrumb trail.

`memo repo index` is rare and high-signal — its `MEMO_MEMFLOW_RECEIPT`
default-ON behaviour was kept unchanged for backward compatibility.

## Synapse-originated writes skip receipts

`MemoSynapseBackend.remember()` passes `skip_memflow_receipt=True`
because synapse already keeps its own write ledger
(`synapse.memory_write_receipt.v1`). Double-counting in memflow would
duplicate the operational signal.

If you wire memo from a different orchestrator and want the same
opt-out, pass `skip_memflow_receipt=True` when calling
`Memory.save(...)`.

## Environment knobs

| Variable | Default | Purpose |
|---|---|---|
| `MEMO_EMIT_RECEIPTS` | unset | `1` enables receipts on save/update/delete/reindex |
| `MEMO_MEMFLOW_RECEIPT` | unset | `0` disables `memo repo index` receipts (default ON otherwise) |
| `MEMO_MEMFLOW_BIN` | unset | Override path to the `memflow` binary |
| `MEMFLOW_PROJECT_ROOT` | unset | Override the discovered project root (walks up for `.memflow/`) |

## Inspecting receipts

```bash
memflow receipt list --source memo            # all memo receipts
memflow receipt list --topic memo-save         # one operation
memflow receipt show <id>                      # one event
```

(Commands are memflow-side; consult its README for the canonical
surface.)
