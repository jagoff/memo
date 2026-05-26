# Synapse adapter (memo as a backend)

Synapse (the cross-backend orchestrator at
[github.com/jagoff/synapse](https://github.com/jagoff/synapse)) federates
memo + memflow into a single `ConsciousnessPacket.v2`. It talks to memo
in two modes:

1. **Subprocess (default)** — `synapse` shells out to `memo search ... --json`
   / `memo save - --json` / `memo contradict list --json`. Pays subprocess
   fork + JSON parse + (cold) MLX load on every call.
2. **In-process via this adapter** — `MemoSynapseBackend(Memory(Config.from_env()))`
   reuses memo's warm `Memory` instance, the recall daemon, and the cached
   embedder. No subprocess. Same shape on the wire.

This document covers the in-process contract, the provenance fields it
persists, and the freeze-write protocol that ties memo to synapse's
Reality Workbench.

## Contract

`memo.synapse_backend.MemoSynapseBackend` implements the synapse
`SynapseBackend` ABC:

```python
from memo.config import Config
from memo.memory import Memory
from memo.synapse_backend import MemoSynapseBackend

backend = MemoSynapseBackend(Memory(Config.from_env()))

health = backend.health()
# {"name": "memo", "available": True, "status": "ready",
#  "detail": "total=651 model=Qwen/Qwen3-Embedding-0.6B"}

refs = backend.collect("astor terapia", k=5, trace_id="synapse://...")
# [{"source": "memo", "uri": "memo://memoria/<id>", "title": ...,
#   "snippet": ..., "score": 0.81, "updated_at": "...",
#   "metadata": {"type": ..., "tags": [...], "path": ...,
#                "provenance": {...}, "extra": {...},
#                "synapse_trace_id": "..."}}, ...]

receipt = backend.remember({
    "kind": "decision",
    "text": "<markdown body>",
    "target": "memo",
    "evidence_paths": ["/abs/path/to/source.md"],
    "metadata": {
        "synapse_trace_id": "synapse://trace/abc",
        "synapse_route_reason": "deep_semantic",
        "synapse_write_policy_schema": "synapse.write_policy.v1",
        "synapse_agent_id": "claude-4-7",
        "synapse_agent_signature": "<sig>",
        "title": "Optional override",
        "tags": ["optional", "tags"],
    },
})
# {"schema": "synapse.memory_write_receipt.v1", "generated_at": ...,
#  "requested_target": "memo", "backend": "memo", "kind": "decision",
#  "uri": "memo://memoria/<id>", "trace_id": ...,
#  "metadata": {"memo_type": "note", "memoria_id": "...", "path": "...",
#               "tags": [...], "provenance": {...}}}
```

Memo never imports synapse — both projects stay sovereign. The wire
shape mirrors `synapse.models.{MemoryWriteRequest, MemoryWriteReceipt,
EvidenceRef, BackendHealth}` but as plain dicts. Synapse wraps with its
own dataclasses on its side.

## Provenance fields

Six keys are recognised as **provenance** (`memo.memory._PROVENANCE_KEYS`):

| Key | Source | Meaning |
|---|---|---|
| `synapse_trace_id` | Synapse | Single end-to-end trace per packet/write |
| `synapse_route_reason` | Synapse | `deep_semantic` / `quick_answer` / etc. |
| `synapse_write_policy_schema` | Synapse | Which write-policy version applied |
| `synapse_write_target` | Synapse | Resolved backend name (defaults to `memo`) |
| `synapse_agent_id` | Synapse | Signed agent identity |
| `synapse_agent_signature` | Synapse | Agent profile signature |

Provenance lives in `meta.extra_json` (current state) AND
`history.events.delta_json` (per-op snapshot). Replay via:

```bash
memo provenance <id>            # pretty table
memo provenance <id> --json     # raw payload
```

```json
{
  "id": "<full id>",
  "current": {
    "synapse_trace_id": "synapse://trace/abc",
    "synapse_agent_id": "claude-4-7",
    ...
  },
  "events": [
    {"ts": "...", "op": "save",   "title": "...", "provenance": {...}},
    {"ts": "...", "op": "update", "title": "...", "provenance": {...}}
  ]
}
```

CLI saves can attach provenance with `--meta KEY=VALUE` (repeatable):

```bash
memo save - --title "Decision" \
  --meta synapse_trace_id=synapse://trace/abc \
  --meta synapse_agent_id=claude-4-7 \
  <<< "Body here"
```

The MCP `memory_save` tool accepts the same payload as `extra={...}`.

## Freeze-write protocol

`Memory.save(respect_synapse_freeze=True)` (or
`MEMO_RESPECT_SYNAPSE_FREEZE=1`) queries the synapse RealityConflict
ledger before commit. If a conflict with `freeze_write=true` AND
`lifecycle_state ∈ {detected, acknowledged}` overlaps the topic, the
write is refused via `memo.memory.WriteRefused`. The MCP tool surfaces
this as a structured payload, never a raise:

```json
{
  "status": "refused",
  "conflict": {
    "conflict_id": "C-42",
    "freeze_write": true,
    "lifecycle_state": "detected",
    "summary": "memo says X but memflow says ¬X",
    "severity": "high"
  },
  "message": "Write refused: synapse conflict C-42 (detected) is freezing this topic"
}
```

The query is derived from the title + first non-empty tags + first
content line. The check only fires when `extra.synapse_trace_id` is set
— anonymous saves bypass it.

`MemoSynapseBackend.remember()` opts into the freeze check by default
(synapse-originated writes always carry a trace). Callers can disable
per-write with `metadata.respect_synapse_freeze = False`.

## Contradictions: pull, not push

Memo's `contradict.py` does **not** push pairs into synapse's conflict
ledger. The direction is pull — synapse's
`conflict_sync.project_memo_contradictions()` calls
`MemoBackend.contradict_list()`, which shells out to
`memo contradict list --json --status open` and projects every row into
a `RealityConflict` for the Reality Workbench.

This is intentional. Synapse has no `conflicts register` verb today,
and projecting on read avoids race conditions between memo's local
triage state and synapse's lifecycle view. See `docs/contradict-loop.md`.

## Environment knobs

| Variable | Default | Purpose |
|---|---|---|
| `MEMO_RESPECT_SYNAPSE_FREEZE` | unset | `1` enables freeze-check on every save with a `synapse_trace_id` |
| `MEMO_SYNAPSE_EXECUTABLE` | unset | Override path to the `synapse` binary (tests, non-PATH installs) |
| `MEMO_SYNAPSE_CLIENT_TIMEOUT` | `8.0` | Subprocess timeout (seconds) for `synapse conflicts` / `synapse packet` calls |
