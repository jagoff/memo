# Memo 4 independence and migration

Memo 4 is a self-contained memory runtime for agents and LLM applications. Its
stable path does not import or call Synapse, Memflow, a sibling checkout, or a
private contracts package. Markdown files remain the source of truth; sqlite
indexes and operational projections are rebuildable.

## Runtime guarantees

- Memo-native, versioned contracts describe traces, write decisions, evidence,
  operational state, task outcomes, and federation bundles.
- Every operational mutation is appended to a per-device hash chain. Verification
  detects missing, reordered, modified, or forked events.
- Write policy freezes unsafe mutations while an unresolved conflict is open.
- Briefing composes current focus, attention, handoffs, and durable recall without
  consulting another daemon.
- Evidence retrieval produces a bounded `EvidencePack` with memory identifiers,
  evidence URIs, freshness, and coverage. Low coverage returns an explicit
  abstention instead of unsupported confidence.
- Task outcomes feed durable utility statistics. Repeated results can promote a
  memory into a procedure or a failure pattern, with evidence links preserved.

Run the local guarantees and a deterministic journal benchmark:

```bash
memo definitive check
memo definitive benchmark --events 1000
memo operational verify
memo evidence "what supports the deployment decision?"
```

## Outcome-driven learning

Record which memories contributed to a real task result:

```bash
memo operational outcome record deploy-42 success \
  --memory MEMORY_ID --artifact https://ci.example/runs/42 \
  --idempotency-key deploy-42:v1

memo operational procedure candidates

memo operational procedure promote "Safe production deploy" \
  --memory MEMORY_ID --kind procedure \
  --reason "Repeated successful production outcomes"
```

Outcome recording is idempotent when the caller reuses the same
`--idempotency-key`; the MCP tool requires this key on every call. Success and
failure statistics update the source memory's
priority and lifecycle evidence; promotion creates an ordinary Markdown memory,
not an opaque model-side state.

## Signed federation

Federation is deny by default. A record is exportable only when its `visible_to`
metadata names the recipient (or when the recipient is the owner). Shared
bundles omit operational journals; owner backup bundles include complete device
chains so causal integrity can be verified after import.

Create a private HMAC key:

```bash
umask 077
openssl rand -hex 32 > ~/.config/memo/federation.key
chmod 600 ~/.config/memo/federation.key
```

Preview, export, verify, and import:

```bash
memo federation preview --principal agent:research
memo federation export research.memo-bundle \
  --principal agent:research --key-file ~/.config/memo/federation.key
memo federation verify research.memo-bundle \
  --principal agent:research --key-file ~/.config/memo/federation.key
memo federation import research.memo-bundle \
  --principal agent:research --key-file ~/.config/memo/federation.key --dry-run
```

Imports are idempotent. Foreign memories are marked untrusted by default and
retain their provenance. Use `--trust-peer` only after establishing the
signer's identity and key out of band. Memo rejects recipient mismatch,
signature failure, oversized bundles, unsafe device identifiers, divergent
journal histories, secret material, and metadata outside the export allowlist.

HMAC keys are intentionally local and symmetric. Exchange them through an
authenticated secret channel; never place them in a vault, bundle, repository,
or command output.

## Migration

The migration is additive and one way:

1. Back up or commit the Markdown vault.
2. Upgrade Memo, preview with `memo migrate-independence`, then apply with
   `memo migrate-independence --write`.
3. Run `memo reindex --rebuild` when the command requests an index refresh.
4. Verify with `memo definitive check`, `memo operational verify`, and
   `memo doctor --strict-runtime`.

The migrator translates known legacy trace, provenance, cache, and operational
fields into Memo-native schemas. Legacy names remain read-time aliases so old
Markdown is not destroyed, but new writes emit only native fields. Re-running
the migration is safe.

Do not delete `memvec.db` as a migration strategy. `memo reindex --rebuild`
preserves user-signal tables that cannot be reconstructed from Markdown.

## Rollback and recovery

The Markdown corpus is version-independent. To roll back the executable, install
the prior Memo version and keep the vault intact. Older versions ignore unknown
front matter. Operational journal files should be retained even if the older
binary does not consume them; restoring the newer binary resumes verification
from the same chain.

If verification reports a fork or tampering, stop writes, preserve the affected
state directory, compare device-chain prefixes, and restore from the last
verified owner bundle or filesystem backup. Never repair a chain by deleting
individual events.
