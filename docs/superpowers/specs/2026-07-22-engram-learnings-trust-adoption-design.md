# memo — Engram learnings for trust and adoption

**Date:** 2026-07-22
**Status:** approved by Fer; ready for implementation planning
**Decision owner:** Fer
**Priority:** A — trust/correctness, then B — adoption/simplicity
**Engram baseline:** `763a6ba432713725d6ce82a2416eec6cbd9ec94e`

## Executive decision

memo should not become an Engram clone and should not replace Markdown, hybrid
retrieval, temporal history, or MLX with a Go/SQLite/FTS-only architecture.

The selected direction is **hybrid convergence**:

1. Make identity, redaction, deduplication, and write recovery stable-core
   invariants.
2. Complete one end-to-end relation and lifecycle path instead of maintaining
   disconnected experimental systems.
3. Consolidate agent installation behind one declarative registry and one
   `memo setup` entry point.
4. Preserve existing public APIs and commands through facades and aliases.
5. Put heuristic or costly behavior behind measurement and graduation; never
   put corruption-prevention invariants behind default-off flags.

This design deliberately reuses memo's stronger capabilities: Markdown as the
source of truth, hybrid semantic retrieval, history/as-of/diff, passive capture,
graph signals, and the existing evaluation and graduation machinery.

## Research basis

The analysis used a local clone of
[Gentleman-Programming/engram](https://github.com/Gentleman-Programming/engram)
at commit `763a6ba` from 2026-07-20. `go test ./...` passed at that revision.
The following Engram paths were inspected directly:

- [architecture](https://github.com/Gentleman-Programming/engram/blob/763a6ba432713725d6ce82a2416eec6cbd9ec94e/docs/ARCHITECTURE.md);
- [store and observation lifecycle](https://github.com/Gentleman-Programming/engram/blob/763a6ba432713725d6ce82a2416eec6cbd9ec94e/internal/store/store.go);
- [MCP save/judge loop](https://github.com/Gentleman-Programming/engram/blob/763a6ba432713725d6ce82a2416eec6cbd9ec94e/internal/mcp/mcp.go);
- [bounded MCP write queue](https://github.com/Gentleman-Programming/engram/blob/763a6ba432713725d6ce82a2416eec6cbd9ec94e/internal/mcp/write_queue.go);
- [declarative setup registry](https://github.com/Gentleman-Programming/engram/blob/763a6ba432713725d6ce82a2416eec6cbd9ec94e/internal/setup/registry.go).

For memo, the audit covered the save path, `VecStore`, schema/migrations,
session-pattern MCP tools, relations, contradictions, lifecycle/verification,
redaction, installers, agent presets, profiles, and graduation. A focused
baseline of `tests/test_session_patterns.py` and `tests/test_memory_write.py`
passed 41 tests.

Passing baselines show that the inspected revisions are internally consistent;
they do not by themselves prove production quality or the proposed changes.

## What Engram gets right

### Composite identity

Engram resolves a `topic_key` inside project and scope, not globally. A write to
an existing topic updates the canonical observation, increments its revision,
and preserves its identity. Exact duplicates update observation counters rather
than creating repeated rows.

The transferable lesson is not Engram's exact schema. It is the contract:
**identity must include its namespace, and repeated evidence must become a
signal rather than corpus noise.**

### Save → candidate → judgment → recall

Engram's MCP save response can include `judgment_required`, candidate pairs,
and judgment IDs. The agent that already understands the content provides the
semantic verdict through `mem_judge`; judged relations are then visible during
retrieval. This avoids an additional model call in the storage service and
makes the decision auditable.

### Storage-boundary privacy

Engram strips `<private>...</private>` in its store before persisting anything.
That placement matters more than the exact regular expression: every caller is
protected, including callers that bypass higher-level capture helpers.

### Deterministic MCP backpressure

Engram uses a bounded, process-local write queue. A full queue rejects work
before mutation. Once a job starts, the caller waits for its actual outcome
instead of racing cancellation against a SQLite mutation. This is a useful
server-level contract as long as cross-process locking remains separate.

### Declarative setup

Engram describes most clients through an adapter registry that owns MCP config
shape, instruction surfaces, detection, and post-install guidance. Marker-bound
instruction blocks and in-place config updates make repeated setup idempotent
and preserve unrelated user configuration.

### Explicit lifecycle

`review_after` is a review reminder, not automatic expiration. Engram exposes
due-review listing and an explicit mark-reviewed operation. This separation is
cleaner than treating age as proof that content is false.

## What memo should not copy

### Do not replace memo's retrieval architecture

Engram is intentionally SQLite/FTS-centric. memo already has vector, BM25,
hybrid retrieval, reranking, graph signals, project/global tiers, and a recall
regression corpus. Relation candidate generation should reuse that pipeline,
not regress to lexical-only matching.

### Do not replace Markdown as source of truth

Engram's SQLite-first model suits a single binary. memo's hand-editable Markdown
and rebuildable index are a product advantage. New identity metadata must either
live in Markdown or be deterministically derivable from it. User signals that
cannot be derived from Markdown must survive `reindex --rebuild`.

### Do not reproduce Engram's monoliths

At the inspected revision, Engram's `internal/store/store.go` is about 7,061
lines, `internal/mcp/mcp.go` about 3,005, and `internal/setup/setup.go` about
1,381. memo must preserve its facade/wiring boundaries and add cohesive
modules instead of expanding `write_ops.py`, `server.py`, or installer god
files.

### Do not rewrite memo in Go

A single Go binary helps Engram's installation story, but a rewrite would
discard memo's MLX integration, Python retrieval stack, established migrations,
and public API. The adoption problem should be solved at the runtime and setup
boundary.

### Do not add another truth counter

memo already has `support_count`, version history, verification, and temporal
validity. Engram's `duplicate_count` is useful as compatibility metadata, but
`support_count` remains memo's canonical corroboration signal.

## Current memo gap analysis

| Area | Existing capability | Gap to close |
|---|---|---|
| Topic identity | `topic_key` is stored and mirrored to frontmatter | `find_by_topic_key()` searches globally, so equal keys can cross project boundaries |
| Exact dedupe | `normalized_hash`, counters, and `support_count` exist | normal `Memory.save()` does not implement one coherent exact-dedupe contract |
| Privacy | capture and ingest paths use `memo.redact` | direct `Memory.save()` is not protected at the final persistence boundary |
| Relations | `memory_relations`, `mem_judge`, and `mem_compare` exist | normal saves do not generate candidates and normal recall does not close the loop |
| Contradictions | a separate contradiction store and tools exist | it competes with `memory_relations` as a second relation truth |
| Review | `review_after` and `mem_review` exist | dates are not populated consistently and there is no complete mark-reviewed cycle |
| Verification | verified/stale/unverified and validity fields exist | lifecycle concepts overlap and can be mistaken for truth invalidation |
| Installation | MCP presets, `install-mcp`, `install-slash`, and mandates exist | client knowledge and orchestration are spread across multiple modules and commands |
| Concurrency | cross-process data-dir locking exists | MCP bursts have no bounded process-local admission layer |

The program therefore closes existing wiring before introducing broad new
surface area.

## Alternatives considered

### Alternative 1 — Minimal patches

Scope `find_by_topic_key`, add redaction to `save()`, and leave relations and
installation as separate experimental systems.

Advantages: smallest diff and fastest short-term delivery.

Disadvantages: preserves duplicated truth, fragmented setup, and partial
lifecycle behavior. Future work would have to reopen the same boundaries.

### Alternative 2 — Hybrid convergence — selected

Preserve stable APIs and storage philosophy, introduce small internal policies
and stores, converge relation truth, and make setup declarative.

Advantages: fixes correctness first, uses memo's stronger retrieval, and gives
adoption a single UX without a rewrite.

Disadvantages: requires staged migrations and disciplined compatibility
adapters.

### Alternative 3 — Engram-style simplification

Move toward a single binary, SQLite-first truth, FTS-first retrieval, and a
smaller MCP surface.

Advantages: simple deployment and a narrow runtime.

Disadvantages: destroys important memo differentiators and carries unacceptable
migration and regression risk.

## Target architecture

```text
public API / CLI / MCP
          │
          ▼
   Memory facade
          │
          ├── IdentityPolicy ─ namespace, topic identity, exact dedupe
          ├── WriteOps ─────── recoverable Markdown/index/history commit
          ├── RelationOps ──── candidate generation and judgment orchestration
          └── LifecycleOps ─── review, verification, invalidate, supersede
                    │
                    ▼
              RelationStore

memo setup ── AgentRegistry ── MCP config + instructions + verification
```

### Component boundaries

#### `memory/identity.py`

A pure policy module with no file or database I/O. It owns:

- namespace derivation and validation;
- topic-key canonicalization;
- title/content normalization inputs for exact dedupe;
- typed resolution outcomes: create, corroborate, revise, or conflict.

This is an implementation boundary, not a required public module name. The
implementation plan may place the pure policy next to existing record helpers
if that avoids a new one-function file without creating an import cycle.

#### `store` identity queries

`VecStore` remains the storage entry point. Scoped lookups, indexes, signal
bumps, and migrations stay behind store methods and write through `_tx()`.
No operation mixin imports a concrete store mixin.

#### `memory/relation_ops.py` and `RelationStore`

The operation layer finds candidates and applies semantic decisions. The store
owns pending/judged relation rows, provenance, idempotency, and annotations.
The old contradiction store becomes a temporary read/import adapter only.

#### `memory/lifecycle_ops.py`

This module composes existing verification and temporal APIs. It does not
invent a parallel archive/truth state machine.

#### server write coordinator

A bounded coordinator wraps mutating MCP handlers. It does not own domain
logic and does not replace the existing cross-process lock.

#### `runtime/agent_registry.py`

One registry composes the shared `consciousness_contracts` presets with
memo-specific profiles, instruction files, protocol modes, detection, and
post-install verification. Existing installers delegate to it.

## Data ownership

| Data | Source of truth | Rebuild behavior |
|---|---|---|
| content, title, type, tags | Markdown | rebuilt into FTS/vector/meta |
| `topic_key`, validity | Markdown frontmatter | rebuilt into meta |
| namespace | derived from Markdown path/tags | rebuilt into meta |
| normalized title/hash | derived index metadata | recomputed |
| access, corroboration, feedback | signal tables | preserved across rebuild |
| relations and judgments | signal tables | preserved across rebuild |
| review evidence | signal tables/frontmatter fields already used for verification | preserved or deterministically recomputed |
| revisions and audit | history/events | preserved |

Hand-edited Markdown wins on reindex. A rebuild must never discard relations,
feedback, support counts, or review evidence merely because they are not
derivable from Markdown.

## Identity model

### Namespace

The canonical namespaces are:

- `project:<slug>` for exactly one project tag;
- `_global` for global memories;
- `_unscoped` when no project can be derived.

Project slug normalization must reuse memo's current project normalization.
There must not be a second spelling algorithm inside `IdentityPolicy`.

A new write carrying multiple incompatible `project:*` tags fails validation.
Historical ambiguous rows are preserved with a null indexed namespace and are
reported by doctor until repaired; migration must not guess.

### Topic identity

The canonical topic identity is:

```text
(namespace, canonical_topic_key)
```

Topic keys use Unicode normalization, trim, whitespace collapse, and casefold.
A partial unique index enforces one active row per non-null identity once the
corpus has no historical conflicts.

Topic resolution has precedence when a key is supplied:

1. Same namespace/key and same normalized content: corroborate.
2. Same namespace/key and changed content: versioned update of the same ID.
3. No same namespace/key: continue to exact-dedupe resolution.

An exact match that already carries a different explicit topic key is an
`IdentityConflict`; memo must not silently discard or replace either key.
When the exact canonical record is unkeyed and the caller supplies the first
topic key, attaching that key is a versioned metadata update. An unkeyed caller
may corroborate an already-keyed exact record without removing its key.

### Exact duplicate identity

For active rows, exact identity is:

```text
(namespace, type, normalized_title, normalized_content_hash)
```

Normalization happens after privacy redaction. A duplicate returns the
canonical ID, increments `support_count`, records `last_seen_at`, and creates no
new Markdown file. `duplicate_count` may be mirrored temporarily for API
compatibility but is not a second confidence source.

Duplicates are not limited to an arbitrary time window. `last_seen_at` and
history preserve recency, while the identical durable claim remains one record.

### Topic revisions

A topic revision keeps:

- canonical ID and path;
- original `created` time;
- prior version in history;
- corroboration and access signals.

It updates content, title/type/tags where allowed, `updated`, normalized hash,
and revision metadata. It follows the same versioned update API used by normal
edits; save must not implement a private overwrite shortcut.

Restoring a deleted row whose topic identity is now occupied returns an
`IdentityConflict`; restore must not overwrite the active replacement.

## Save protocol

The logical flow is:

```text
validate → redact → derive namespace/identity → acquire lock
        → resolve create/corroborate/revise/conflict
        → prepare atomic Markdown update
        → publish Markdown + index/history/signals
        → commit receipt → release lock
        → detect relation candidates after commit
```

### Privacy boundary

Redaction occurs inside the core save path before hashing, candidate generation,
history, receipts, logs, Markdown, or SQLite. Higher-level capture and ingest
redaction remains defense in depth.

If required redaction fails, the save fails before persistence. Secrets must
not be included in exception text or diagnostic receipts.

### Recoverable cross-store commit

Filesystem and SQLite changes cannot form a literal ACID transaction. memo will
extend its existing atomic-write and `_save_index_pending` behavior so every
intermediate state is detectable and repairable:

1. Prepare the redacted Markdown in a same-filesystem temporary file.
2. Preserve the previous version before a topic revision becomes visible.
3. Apply store/history changes through `_tx()` and atomically rename Markdown at
   the defined commit point.
4. On index/embed failure, keep the canonical Markdown, text-only reservation,
   `_memo_embed_pending` marker, history, and structured receipt coherent.

Startup/doctor reconciles unfinished indexed state from canonical Markdown. If
failure injection finds a crash window that these existing mechanisms cannot
detect, the implementation plan may add a minimal state-dir intent journal with
its own recovery tests; this design does not assume one is necessary. No
failure may be reported as a clean rollback when a visible mutation occurred.

### Save result

The stable response remains backward compatible and gains additive fields:

```json
{
  "id": "...",
  "action": "created|corroborated|revised",
  "index_pending": false,
  "judgment_required": true,
  "candidates": [
    {
      "judgment_id": "...",
      "memory_id": "...",
      "title": "...",
      "excerpt": "...",
      "score": 0.0
    }
  ]
}
```

Old clients may ignore the new fields. Candidate generation failure is
fail-open after a successful save and is disclosed in the receipt.

## Relation convergence

### One relation truth

`memory_relations` becomes the only writable relation truth. It stores:

- stable relation/judgment ID;
- source and target IDs;
- relation verb;
- `pending`, `judged`, or `orphaned` status;
- reason and confidence;
- actor/kind/model provenance where supplied;
- created and updated timestamps;
- migration provenance for legacy contradiction rows.

The initial verbs are:

- `supersedes`;
- `conflicts_with`;
- `compatible`;
- `scoped`;
- `related`;
- `not_conflict`.

`not_conflict` is persisted as a judged negative decision so the same pair does
not repeatedly interrupt agents.

### Candidate generation

After a new or revised save commits, RelationOps searches with memo's existing
hybrid retrieval:

- maximum three candidates;
- exclude the saved record and deleted/invalid records;
- project source: same project plus global;
- global source: global only;
- unscoped source: unscoped plus global;
- never silently compare unrelated projects;
- initially limit automatic candidates to `decision` and `preference`, plus
  durable records carrying `architecture`, `config`, or `policy` tags.

`architecture`, `config`, and `policy` are qualifiers, not new memory types.
The program must reuse the valid types defined by `memo.tiers` and must not
expand `_VALID_TYPES` as a side effect.

The stored document never receives the query instruction prefix. Candidate
generation follows memo's existing asymmetric query-prefix contract.

The service performs no extra LLM call. The consuming agent judges because it
already has the semantic context that produced the save.

### Judgment semantics

The consolidated agent-facing operation is conceptually:

```text
memory_judge(judgment_id, decision, reason?, confidence?)
```

The existing `mem_judge` MCP tool remains the compatibility surface and may be
upgraded to this contract. This design does not require registering a duplicate
MCP tool merely to change its name.

Applying the same judgment twice is idempotent. A conflicting second judgment
returns a typed conflict and leaves the original audit trail intact.

- `supersedes` closes the older memory's `invalid_at` through the versioned
  temporal API.
- `conflicts_with` keeps both valid and visibly disputed.
- `compatible`, `scoped`, and `related` annotate without changing validity.
- `not_conflict` suppresses repeat candidacy without creating a positive edge.

Failure to apply a `supersedes` transition leaves the judgment pending; it must
not mark the relation judged while the temporal update failed.

### Read behavior

Judged relations are rendered compactly in search, ask, and unified briefing.
Pending relations appear only in review/diagnostic surfaces. When both sides of
a judged conflict are relevant, recall presents both rather than silently
hiding one side.

Deleted relation endpoints become `orphaned`. Their audit rows remain, but they
do not affect normal retrieval.

## Lifecycle convergence

Three concepts remain distinct:

| Concept | Meaning |
|---|---|
| `valid_at` / `invalid_at` | whether the claim is currently true |
| verification state | confidence that the claim has been checked |
| `review_after` | when checking it again would be useful |

Review never invalidates content automatically.

### Initial schedule policy

Only durable, change-sensitive types receive automatic review dates:

- `preference` and durable records tagged `config`: 90 days;
- `decision`: 180 days;
- durable records tagged `policy` or `architecture`: 365 days;
- other types: no automatic review date.

These are policy defaults, not hard-coded store behavior. Open conflicts make a
record immediately due. Recent explicit verification may lengthen the next
interval within a capped policy. Corroboration raises confidence but does not
postpone freshness review for mutable configuration. Raw age alone never sets
`invalid_at`.

The current universal verified → stale → unverified age transitions must be
reconciled with this policy. A verified record becomes `stale` when its own
`review_after` passes; it remains stale until reviewed or invalidated rather
than becoming semantically "never verified" merely because more time passed.
Records with no review policy do not decay solely from wall-clock age.

### Operations

- `list_due_reviews(project?, limit)`;
- `mark_reviewed(id, evidence?)`;
- `invalidate(id, reason)`;
- `supersede(old_id, new_id, reason)`;
- `judge_relation(...)`.

`mark_reviewed` updates verification evidence and the next review date without
rewriting content. `memo maintain` may report or enqueue due reviews but must
not silently invalidate high-confidence decisions.

## MCP write coordination

The optional coordinator is process-local and bounded. It wraps all mutating
MCP tools, not CLI or internal API calls.

Required semantics:

- queue-full is retryable and returned before mutation;
- canceled jobs that have not started are skipped;
- started jobs complete and return their real outcome;
- panics become typed internal errors;
- queue depth, rejection count, and wait time are observable;
- the data-directory lock remains authoritative across processes.

Queue size is registered through `flags.py`; app code must not read raw
`MEMO_*` environment variables. Engram's size of 32 is an initial load-test
candidate, not an unmeasured memo default.

## Adoption and setup

### One entry point

```bash
memo setup codex
memo setup claude-code
memo setup all
memo setup --detect
memo setup --dry-run
```

Existing `install-mcp`, `install-slash`, and `mandate` commands remain available
and delegate to the same registry during the compatibility window.

### Agent adapter contract

Each adapter declares:

- slug and detection strategy;
- MCP registration strategy/config format;
- stable isolated `memo-mcp` command;
- MCP profile (`core`, `full`, or explicit custom profile);
- instruction surfaces and managed-block style;
- optional plugin/command assets;
- protocol mode (`compact`, `full`, or `off`);
- verification checks;
- restart/reconnect guidance;
- rollback or manual-remediation behavior.

The registry reuses shared presets rather than forking them. memo-specific
instructions and profiles remain local.

### Mutation safety

Setup first builds a plan. `--dry-run` prints file and command diffs without
mutation. File edits are backed up, written atomically, and preserve unknown
keys and user text. Managed instruction blocks are marker-delimited and
idempotent.

External client commands may not always support transactional rollback. An
adapter must declare a compensating command or return a partial-install receipt
with exact remediation. The UX must not claim full rollback when an external
tool already changed state.

Normal MCP startup never rewrites client configuration; setup is explicit.

### Doctor integration

`memo doctor --agent <slug>` verifies:

- detected client and config syntax;
- MCP command resolves to the intended isolated runtime;
- `memo` and `memo-mcp` versions agree;
- selected tool profile and protocol marker are current;
- data/state locations are writable;
- a smoke save/search succeeds under temporary isolated directories;
- the client has been restarted or needs reconnection after an update.

Tests and doctor smoke operations must never touch the real vault.

## Migration design

The current store uses `PRAGMA user_version = 4`. Migration is additive and
idempotent.

### M0 — preflight inventory

Before enforcement, doctor reports:

- multiple project tags;
- duplicate active `(namespace, topic_key)` groups;
- exact duplicate groups;
- relation endpoints that do not exist;
- legacy contradiction rows not yet imported;
- review metadata inconsistencies;
- historical secret-pattern findings without printing secret values.

M0 is read-only.

### M1 — schema and backfill

Add the rebuildable `namespace` and `normalized_title` identity fields, their
non-unique diagnostic indexes, and the missing relation/review provenance
fields. Backfill namespace deterministically from Markdown path and tags.
Ambiguous rows keep a null namespace and produce diagnostics.

Opening an old store must not delete, merge, or rewrite Markdown.

### M2 — new-write enforcement

All new writes use composite identity and storage-boundary redaction. Runtime
locking prevents new duplicates even when historical duplicates still exist.
Create the partial unique topic index only when preflight finds no conflicting
active group.

The indexed-identity capability is checked explicitly; `user_version` records
the additive column migration but does not falsely imply that a corpus with
historical conflicts already has the unique index.

If enforcement cannot be installed because of historical conflicts, memo stays
readable and preserves data. It rejects only ambiguous writes and reports the
repair path.

### M3 — relation import

Import legacy contradiction records into `memory_relations` with a deterministic
migration key and provenance. Re-running the import is a no-op. For one release,
the legacy store remains a read-only compatibility source. There is no dual
write.

After the compatibility window and successful parity checks, archive the old
database and remove the adapter in a separately reviewed change.

### M4 — setup convergence

Introduce `memo setup`, migrate existing command implementations to registry
delegation, and keep command aliases. Upgrades do not modify agent config until
the user explicitly runs setup.

## Error and recovery contract

| Failure | Result |
|---|---|
| validation, namespace, or redaction failure | no mutation; typed domain error |
| identity conflict | no mutation; return both conflicting identities |
| queue full | no mutation; retryable error |
| Markdown publish failure | no committed save; clean the staged temp file |
| index failure after Markdown commit | successful canonical write with `index_pending`; targeted repair |
| candidate search failure | save succeeds; receipt reports relation detection unavailable |
| judgment write failure | relation remains pending |
| supersede temporal failure | no judged supersede; old memory remains valid |
| setup file failure | restore staged file backups |
| non-reversible external setup failure | partial receipt with exact remediation |

Normal domain failures use `MemoError` subclasses. No new path raises bare
`Exception` as its public contract.

## Compatibility contract

- `Memory` remains the core application facade.
- Existing CLI and MCP names remain available during migration.
- Save response changes are additive.
- Markdown remains hand-editable and authoritative.
- Existing project/global retrieval behavior remains unless a labeled eval
  proves a deliberate change.
- `reindex --rebuild`, not database deletion, is the supported rebuild path.
- Signal tables survive rebuild.
- Query prefixing remains query-only.
- Optional MLX imports remain deferred.

## Test strategy

### Characterization first

Before implementation, lock down current public output, history semantics,
Markdown round trips, install commands, and relation tools. Characterization
tests distinguish intentional behavior changes from accidental API drift.

### Identity and dedupe

Required cases:

- equal topic keys in different projects never overwrite each other;
- project, global, and unscoped namespace matrix;
- multiple project tags fail before persistence;
- simultaneous same-topic saves produce one active canonical record;
- exact duplicate returns the same ID and bumps only corroboration signals;
- topic revision preserves creation time and history;
- delete/restore retains identity metadata;
- restore refuses a topic identity occupied by an active replacement;
- hand-edited Markdown wins after reindex;
- migration over historical duplicates is non-destructive and repeatable.

Use property tests for normalization and adversarial Unicode/case/whitespace.

### Privacy

Exercise `Memory.save`, CLI, MCP, capture, ingest, and imports. Assert secret and
`<private>` payloads are absent from Markdown, SQLite/FTS, history, relations,
receipts, logs, and error strings. Include false-positive fixtures so redaction
does not destroy legitimate code or prose.

### Relations and lifecycle

- labeled conflict/supersede/compatible/scoped pairs;
- candidate cap and namespace guards;
- judgment idempotency and conflicting second judgments;
- complete save → judge → search/ask/briefing path;
- supersede temporal history and as-of queries;
- orphan behavior after soft and hard delete;
- due review does not imply invalidation;
- mark-reviewed rescheduling and verification evidence;
- idempotent contradiction migration.

The hybrid candidate generator must match or beat a lexical baseline on
candidate recall while reducing irrelevant judgment burden. Thresholds and the
fixed corpus are committed before tuning so the gate cannot move after results
are known.

### Setup

Every registry adapter runs under a disposable home with isolated `MEMO_DATA_DIR`
and `MEMO_STATE_DIR`. Verify repeated install, preservation of unrelated config,
valid JSON/TOML/YAML, dry-run purity, backup/rollback, correct runtime path,
profiles, markers, reconnection guidance, and partial external-command errors.

### Concurrency and recovery

Inject failures around every cross-store transition, SQLite transaction, atomic
rename, candidate pass, and setup mutation. Exercise concurrent
readers/writers, queue saturation, canceled queued work, started work, abrupt
process termination, WAL contention, and repeated recovery.

### Required gates

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 --cov=memo --cov-report=term-missing
uv run --no-sync memo eval recall --labels eval/regression_labels.json --k 5 --force
```

Also run focused hook, recall-server, runtime-isolation, migration, installer,
history/as-of, session-pattern, redaction, and relation tests. Real MLX coverage
continues under `@pytest.mark.requires_mlx` and the macOS smoke workflow.

## Performance and quality gates

- Recall-hook end-to-end latency stays inside the existing five-second budget.
- Relation detection adds no service-side LLM call.
- Candidate generation is capped at three and runs after commit.
- Save latency, queue wait, queue rejection, and recovery time receive a
  reproducible baseline before defaults change.
- Retrieval precision@K is flat or better and noise@K is flat or lower.
- Trust invariants have zero allowed failures on the namespace, redaction,
  history, and concurrency matrices.
- Setup is idempotent for every supported declarative adapter.

The existing graduation controller may auto-activate only behavior measured by
a faithful evaluator. Relation judgment UX, setup UX, and other behavior not
faithfully measurable offline remain report-only and require human approval.

## Rollout

### Phase 0 — characterization and scorecards

Add fixed identity, privacy, relation, lifecycle, setup, and failure-injection
fixtures. Capture retrieval and latency baselines.

### Phase 1 — trust invariants

Ship storage-boundary redaction, composite namespace lookup, exact dedupe,
corroboration bumping, typed conflicts, and recoverable write receipts. These
are correctness defaults, not experimental toggles.

### Phase 2 — relation convergence

Expand `memory_relations`, add the legacy import adapter, wire post-save
candidates, judgment, and read annotations. Start candidate/annotation behavior
in shadow or explicit opt-in until its corpus gate passes.

### Phase 3 — lifecycle convergence

Add review scheduling, mark-reviewed, and explicit lifecycle operations on top
of existing verification and temporal history. Do not automatically invalidate.

### Phase 4 — adoption

Introduce AgentRegistry, `memo setup`, dry-run plans, doctor integration, and
compatibility aliases. Dogfood with Codex and Claude Code before broad defaults.

### Phase 5 — graduation and cleanup

Graduate only empirically positive behaviors. Remove the legacy contradiction
adapter and old duplicated installer code in separately reviewable changes
after compatibility telemetry and parity checks.

Each phase is independently releasable. Adoption work does not wait for every
relation enhancement, and no phase requires a repository rewrite.

## Success criteria

- No cross-project overwrite is possible through topic upsert.
- No normal save path persists unredacted secrets or `<private>` content.
- Repeated identical evidence strengthens one canonical memory.
- A save can produce candidates, receive one auditable judgment, and expose the
  judged result in normal recall.
- `review_after` never masquerades as truth invalidation.
- Existing public clients continue working through additive responses and
  command aliases.
- A new user can configure a detected agent through one idempotent command and
  diagnose the result.
- Retrieval quality and hook latency remain within their established gates.
- Reindex and interrupted-write recovery preserve Markdown truth and user
  signals.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| historical topic collisions block unique enforcement | non-destructive preflight, null ambiguous namespace, scoped rejection, explicit repair |
| exact dedupe merges legitimate repeated text | require same namespace/type/title/hash; explicit topic-key conflict protection |
| relation candidates create agent noise | max three, eligible durable types, namespace guard, labeled corpus, negative judgments |
| relation annotations consume recall tokens | compact rendering inside existing token budget; pending rows excluded |
| lifecycle creates a second truth system | keep validity, verification, and review meanings separate; reuse temporal API |
| queue hides cross-process races | retain data-dir lock as authoritative and test both layers |
| setup corrupts user config | plan first, parser-based merge, marker blocks, atomic backup, isolated tests |
| migration loses non-Markdown signals | preserve signal tables and test rebuild/restore explicitly |
| modules become new monoliths | pure identity policy, narrow stores, wiring-only CLI/server, file-size review |

## Non-goals

- Rewriting memo in Go or adopting Engram's database as memo's source of truth.
- Replacing hybrid/vector retrieval with FTS-only retrieval.
- Automatically resolving semantic conflicts without an agent or human verdict.
- Automatically invalidating memories because they are old.
- Adding a large new MCP tool surface; existing profiles remain constrained.
- Reworking cloud/Git sync as part of this program.
- Automatically rewriting agent configuration during normal startup or upgrade.
- Destructively merging historical duplicate Markdown during schema migration.

## Follow-up planning boundary

Implementation planning must split this program into independently verifiable
plans in rollout order. Phase 1 cannot be combined with relation, lifecycle,
and setup changes in one large implementation branch. Each plan must name its
migrations, compatibility behavior, focused tests, eval obligations, and
rollback evidence before code changes begin.
