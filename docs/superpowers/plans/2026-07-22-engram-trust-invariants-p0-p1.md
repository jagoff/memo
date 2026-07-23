# Engram Learnings — Trust Invariants P0 + P1 Implementation Plan

> **For implementation:** Execute this plan task-by-task, keep every commit
> independently reviewable, and stop after Task 8. Relation convergence,
> lifecycle convergence, the MCP write coordinator, and `memo setup` belong to
> later plans and must not be pulled into this change.

**Design:**
`docs/superpowers/specs/2026-07-22-engram-learnings-trust-adoption-design.md`

**Goal:** Make memo's normal persistence path incapable of storing known secret
patterns or `<private>` spans, scope topic identity by project/global namespace,
turn exact repeated evidence into corroboration of one canonical memory, and
preserve a recoverable, backward-compatible save contract.

**Architecture:** Add a pure `memo.identity` policy used by write, update,
reindex, migration, and diagnostics. Add derived identity columns to the
rebuildable SQLite index without changing Markdown ownership. Sanitize complete
records at the `Memory` persistence boundary before identity hashing, embedding,
history, receipts, or logs. Resolve topic and exact identities under the
existing cross-process data-dir lock; use the existing atomic Markdown and
text-only/index-pending recovery machinery rather than inventing a second write
system. Expose save outcomes through additive ephemeral fields on
`MemoryRecord`.

**Tech Stack:** Python 3.11+, pytest, Click, SQLite/FTS5/sqlite-vec, python-
frontmatter. Tooling via `uv run --no-sync`.

## Global constraints

- **Correctness is always on.** Storage-boundary pattern redaction, private-span
  stripping, namespace validation, composite topic lookup, and exact
  corroboration are not experimental flags. Entropy redaction remains opt-in.
- **Markdown remains source of truth.** `namespace`, normalized title, and
  normalized content hash are derived index fields. Hand-edited Markdown wins
  on reindex. Do not rewrite the corpus during schema migration.
- **Keep legacy `normalized_hash`.** It currently means the session-pattern
  hash of title/type/scope and accepts caller-supplied values. Add
  `normalized_content_hash`; do not reinterpret or delete the old column or
  frontmatter key.
- **One namespace spelling.** Reuse `memo.project.slugify_project`; do not add a
  second project slug algorithm.
- **One authoritative lock.** The existing `authority_write_lock` remains the
  cross-process serialization boundary. SQLite constraints are defense in
  depth, not a substitute for resolving the complete write decision under the
  lock.
- **No long model work under the lock.** Use optimistic read, compute a needed
  embedding outside the lock, then re-resolve identity under the lock before
  publishing. Never trust the optimistic result as the commit decision.
- **No secret-bearing diagnostics.** Errors, logs, receipts, tests, and doctor
  output may contain IDs, counts, namespaces, and hashes, but never the matched
  secret value.
- **Stable APIs.** `Memory.save()` still returns `MemoryRecord`. CLI and MCP
  commands remain named and shaped as today; new response fields are additive.
- **No phase creep.** Do not touch relation judgment, `memory_relations`, review
  scheduling, contradiction migration, AgentRegistry, installer commands, or a
  process-local MCP queue in this plan.
- **Test isolation.** Use `tmp_cfg` or an explicit isolated `Config`; never
  touch the real vault/state directory. Keep MLX imports deferred.
- **Shared working tree.** Stage explicit paths only; never use `git add -A`,
  `git add .`, destructive checkout/reset, or whole-tree formatting.

## What already exists — preserve and extend

- `Memory.save()` in `src/memo/memory/write_ops.py` already uses an
  `authority_write_lock`, atomic same-filesystem file replacement, a text-only
  identity reservation, `_memo_embed_pending`, and `_save_index_pending`.
- `Memory.update()` in `src/memo/memory/update_ops.py` already snapshots
  versions, embeds before touching Markdown, rolls the file back if the index
  update fails, and logs history/receipts after success.
- `VecStore.bump_support_batch()` already owns the durable `support_count`
  signal. Exact corroboration must consume that same signal rather than create a
  second confidence counter.
- `memo.redact` already has deterministic pattern masking and private-span
  stripping. The missing piece is a mandatory final persistence sanitizer.
- `topic_key`, legacy `normalized_hash`, `duplicate_count`, `last_seen_at`, and
  `revision_count` already exist in schema v3, but topic lookup is global and
  the legacy hash is not a content identity.
- `replace_memory_index()` already atomically rebuilds Markdown-derived
  meta/FTS/vector rows and preserves non-derivable user-signal tables.
- `memo doctor --db`, migration tests, write failure-injection tests, and
  `eval/regression_labels.json` already provide the correct verification seams.

## Canonical identity contract

Use these exact semantics throughout the implementation:

```text
namespace:
  exactly one project tag -> project:<slug>
  no project + auto_project=False -> _global
  no project + auto_project=True  -> _unscoped
  incompatible project tags      -> IdentityConflictError on new writes

topic identity:
  (namespace, canonical_topic_key)

exact identity:
  (namespace, type, normalized_title, normalized_content_hash)
```

Canonical topic/title text uses Unicode NFKC, trim, Unicode-whitespace
collapse, then `casefold()`. Canonical content uses Unicode NFKC, CRLF/CR to LF,
removes trailing whitespace per line, and trims only outer blank space; it does
**not** collapse internal whitespace. Hash canonical content with full SHA-256.
All normalization happens after mandatory persistence redaction.

Topic resolution precedes exact resolution:

1. Same topic + same exact content: corroborate.
2. Same topic + changed content: versioned revision of the same ID/path.
3. No topic match: resolve exact identity.
4. Exact match with a different explicit topic: typed conflict.
5. Exact unkeyed record + first supplied topic: versioned metadata attachment.
6. Unkeyed caller + exact keyed record: corroborate without clearing its key.

If either lookup returns multiple active rows in a legacy-conflicted corpus,
reject only that ambiguous write with a typed conflict; keep the corpus readable.
For rule 1, preserve the canonical title/type/tags: a repeated save is evidence,
not an implicit metadata edit. Callers that intend to change metadata use
`memo update`; a topic revision may carry new metadata together with new body.

---

## Task 1: P0 — characterize and freeze the current trust baseline

**Files:**
- Create: `tests/test_engram_trust_invariants.py`
- Create: `docs/superpowers/reports/2026-07-22-engram-trust-p0.md`

**Purpose:** Turn the audit into reproducible evidence before production code
changes. Intended P1 behavior is expressed as strict xfails, while existing
recovery guarantees remain passing guards.

- [ ] Add isolated fixtures for two project namespaces, one explicit-global
  namespace, and one auto-project failure (`_unscoped`). Use the stub embedder
  and distinct temp data/state directories.
- [ ] Add `xfail(strict=True)` tests proving the current gaps:
  `test_direct_save_redacts_before_any_persistence`,
  `test_same_topic_key_is_isolated_by_project`,
  `test_exact_duplicate_corroborates_one_record`, and
  `test_multiple_project_tags_fail_without_mutation`.
- [ ] In the privacy xfail, inspect Markdown, FTS, `meta.extra_json`, history,
  and captured receipts/log records. Assert the fixture token is absent from
  all of them; never print it on assertion failure—compare booleans/counts.
- [ ] Add passing characterization guards by reusing or parameterizing the
  existing index-pending and atomic rebuild fixtures. At minimum prove:
  Markdown survives an embedding/index failure, a later reindex repairs it,
  and `reindex --rebuild` preserves `support_count`.
- [ ] Record the baseline matrix in the P0 report with columns `invariant`,
  `probe`, `before`, `target`, and `proof command`. Mark the four xfails as
  known P1 gaps, not skipped work.
- [ ] Capture a latency baseline for 100 stub-embedder saves and record p50/p95
  wall time and host/test configuration. This is a comparison baseline, not a
  CI threshold.
- [ ] Run:

```bash
uv run --no-sync pytest tests/test_engram_trust_invariants.py \
  tests/test_memory_write.py tests/test_write_ops_failure.py \
  tests/test_memory_reindex.py tests/test_support_count.py -v
```

  Expected: all characterization guards pass and exactly four tests xfail.

- [ ] Commit only the two new files:

```bash
git add tests/test_engram_trust_invariants.py \
  docs/superpowers/reports/2026-07-22-engram-trust-p0.md
git commit -m "test: characterize Engram trust invariant gaps"
```

---

## Task 2: Add the pure identity policy and typed conflicts

**Files:**
- Create: `src/memo/identity.py`
- Modify: `src/memo/errors.py`
- Modify: `src/memo/memory/record.py`
- Modify: `src/memo/memory/__init__.py`
- Test: `tests/test_identity.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class IdentityKeys:
    namespace: str
    topic_key: str | None
    normalized_title: str
    normalized_content_hash: str

def canonical_topic_key(value: str | None) -> str | None: ...
def normalized_title(value: str) -> str: ...
def normalized_content_hash(value: str) -> str: ...
def namespace_for_write(tags: Sequence[str], *, auto_project: bool) -> str: ...
def namespace_for_index(tags: Sequence[str], *, path: str) -> str | None: ...
def identity_keys(..., auto_project: bool) -> IdentityKeys: ...

class IdentityConflictError(ValidationError): ...
```

- [ ] Write failing table-driven tests for NFKC, whitespace, casefold, newline
  normalization, non-collapsed internal whitespace, and full 64-character
  SHA-256 output.
- [ ] Write failing namespace tests covering one project tag, duplicate spelling
  of the same project, incompatible project tags, explicit global, unscoped,
  `_global/` and `_unscoped/` reindex paths, and ambiguous historical tags
  returning `None` rather than guessing.
- [ ] Implement `memo.identity` as a leaf module. It may import
  `memo.project.slugify_project` and `memo.errors`, but must not import
  `Memory`, `VecStore`, flags, Config, or model code.
- [ ] Make `IdentityConflictError` carry a stable `kind`, incoming identity,
  and conflicting IDs/identities as structured attributes. Its string form must
  omit content and matched secret values.
- [ ] Add optional ephemeral fields to frozen `MemoryRecord`:
  `action: Literal["created", "corroborated", "revised"] | None = None` and
  `index_pending: bool = False`. `to_dict()` includes these fields only when
  `action` is non-null, so list/get responses do not gain meaningless nulls.
- [ ] Re-export the new error from `memo.memory` for compatibility with the
  existing error import convention.
- [ ] Run:

```bash
uv run --no-sync pytest tests/test_identity.py -v
uv run --no-sync ruff check src/memo/identity.py src/memo/errors.py \
  src/memo/memory/record.py src/memo/memory/__init__.py tests/test_identity.py
uv run --no-sync mypy src/memo/identity.py src/memo/errors.py
```

- [ ] Commit:

```bash
git add src/memo/identity.py src/memo/errors.py src/memo/memory/record.py \
  src/memo/memory/__init__.py tests/test_identity.py
git commit -m "feat: define namespaced memory identity contract"
```

---

## Task 3: Enforce privacy at the final persistence boundary

**Files:**
- Modify: `src/memo/redact.py`
- Modify: `src/memo/memory/write_ops.py`
- Modify: `src/memo/memory/update_ops.py`
- Modify: `src/memo/capture_core.py`
- Modify: `src/memo/cli_ingest.py`
- Modify: `src/memo/flags_behavior.py`
- Modify: `src/memo/tui/config/screens.py`
- Modify: `docs/privacy.md`
- Test: `tests/test_redact.py`
- Test: `tests/test_engram_trust_invariants.py`
- Update affected capture/ingest tests that currently expect an opt-out from
  final persisted redaction.

**Interfaces:**

```python
@dataclass(frozen=True)
class SanitizedMemoryInput:
    content: str
    title: str | None
    tags: list[str]
    topic_key: str | None
    normalized_hash: str | None
    extra: dict[str, Any]
    changed: bool

def sanitize_memory_input(..., entropy: bool = False) -> SanitizedMemoryInput: ...
```

- [ ] Write failing pure tests for recursive sanitization of string values and
  keys in nested dict/list/tuple structures. Cover content, title, tags,
  `topic_key`, legacy `normalized_hash`, and `extra`; non-string scalar values
  must round-trip unchanged.
- [ ] Define deterministic field behavior: strip `<private>` spans first, then
  apply pattern masking; drop empty tags; reject private-only/empty content;
  turn an emptied explicit title into `None` so it is safely re-derived; reject
  an explicitly supplied topic key that becomes empty; and reject sanitized
  mapping keys that are empty or collide rather than silently dropping data.
  Add `_redacted` exactly once when any field changes. Keep entropy scanning
  controlled by `MEMO_REDACT_ENTROPY` through `flag_bool` at the caller.
- [ ] Implement the pure recursive sanitizer in `memo.redact`. It must never log
  the original or matched value and must not read flags itself.
- [ ] Call it at the beginning of `Memory.save()`, before auto-title/derive,
  embedding, identity normalization, dedupe lookup, history, receipts, or logs.
  Sanitize before `respect_synapse_freeze` or any helper that may externalize
  caller text.
- [ ] In `Memory.update()`, build the complete prospective record first and
  sanitize that full record before embedding/snapshot/persistence. Sanitize a
  legacy prior body before placing it in version history so an unrelated edit
  cannot copy an old secret into a new history row.
- [ ] Keep capture/ingest redaction as defense in depth. The legacy
  `MEMO_REDACT_SECRETS=0` and `MEMO_PRIVATE_MARKERS=0` settings may disable an
  earlier preprocessing pass but must no longer disable final persisted pattern
  masking/private stripping. Update descriptions, TUI copy, docs, and tests to
  make that boundary explicit. Do not remove the flags during this release.
- [ ] Ensure reindex never introduces an unredacted body into FTS/vector/meta:
  index a sanitized derived representation while leaving hand-edited Markdown
  untouched. Report the finding for doctor in Task 7.
- [ ] Remove the privacy xfail from Task 1 and extend it across direct save,
  direct update, CLI save, MCP save, capture, ingest, reindex, history, FTS,
  frontmatter, receipt metadata, and logs. Include benign SHA/UUID fixtures to
  prevent a false-positive expansion.
- [ ] Run:

```bash
uv run --no-sync pytest tests/test_redact.py \
  tests/test_engram_trust_invariants.py tests/test_capture.py \
  tests/test_capture_core.py tests/test_ingest_enhanced.py \
  tests/test_save_extract.py tests/test_server.py -v
```

- [ ] Commit explicit files only, including every intentionally updated test:

```bash
git add src/memo/redact.py src/memo/memory/write_ops.py \
  src/memo/memory/update_ops.py src/memo/capture_core.py \
  src/memo/cli_ingest.py src/memo/flags_behavior.py \
  src/memo/tui/config/screens.py docs/privacy.md tests/test_redact.py \
  tests/test_engram_trust_invariants.py tests/test_capture.py \
  tests/test_capture_core.py tests/test_ingest_enhanced.py \
  tests/test_save_extract.py tests/test_server.py
git commit -m "feat: enforce storage-boundary memory redaction"
```

---

## Task 4: Add schema v5 identity fields, backfill, and store APIs

**Files:**
- Modify: `src/memo/store/migrations.py`
- Modify: `src/memo/store/schema.py`
- Modify: `src/memo/store/queries.py`
- Modify: `src/memo/store/vec_base.py`
- Modify if required by row projection: `src/memo/store/bm25_queries.py`
- Create: `tests/test_store_migrations.py`
- Test: `tests/test_store.py`
- Test: `tests/test_memory_reindex.py`

**Schema v5 additions to `meta`:**

```sql
namespace TEXT NULL,
normalized_title TEXT NULL,
normalized_content_hash TEXT NULL
```

Add non-unique diagnostic/lookup indexes for topic and exact identity. The
partial active-topic unique index is installed only after a conflict-free
preflight. Track that independently in `schema_meta` (for example,
`identity_topic_unique=enabled|blocked`); `PRAGMA user_version=5` means only
that the additive columns exist.

**Store interfaces:**

```python
def find_active_by_topic_identity(namespace, topic_key) -> list[dict[str, Any]]: ...
def find_active_by_exact_identity(namespace, type_, normalized_title,
                                  normalized_content_hash) -> list[dict[str, Any]]: ...
def get_identity_keys(id_) -> dict[str, str | None]: ...
def corroborate_identity(id_, *, seen_at) -> dict[str, int | str]: ...
def identity_diagnostics() -> dict[str, Any]: ...
def reconcile_identity_constraint() -> str: ...  # enabled | blocked
```

- [ ] Write failing fresh-schema, v4-to-v5, repeated-open, and interrupted-
  migration tests. Assert old Markdown bytes are unchanged.
- [ ] Add v5 columns to fresh DDL and the ordered migration. Backfill
  `normalized_title`, canonical `topic_key`, and content hash from the indexed
  FTS body. Derive namespace from tags/path. Multiple incompatible project tags
  get `namespace=NULL`; do not guess or merge rows.
- [ ] Keep legacy `normalized_hash` intact in both SQLite and Markdown. Add the
  new content hash through every upsert, text-only upsert, path-owner replace,
  metadata update, select projection, and atomic `replace_memory_index` row.
- [ ] Replace global `find_by_topic_key(topic_key)` call sites with the new
  composite API. Return all matches, not `LIMIT 1`, so legacy ambiguity is
  detectable. Add exact-identity lookup with a covering non-unique index.
- [ ] Implement `corroborate_identity()` as one `_tx()` that increments the
  canonical `memory_health.support_count`, updates `last_seen_at`, and mirrors
  `duplicate_count` only for compatibility. It must not advance the
  cross-machine `memory_health.updated_at` clock unless confidence changes;
  preserve the existing support arbitration tests.
- [ ] Preflight active duplicate `(namespace, topic_key)` groups before creating
  the partial unique index. When conflicts exist, stamp `blocked`, leave rows
  readable, and do not fail store startup. When clean, create the index and
  stamp `enabled` transactionally.
- [ ] Make `replace_memory_index()` preflight its proposed rows. If a hand edit
  introduces a topic collision, drop/disable the unique index and atomically
  install the readable index with capability `blocked`; never partially rebuild
  or erase signal tables. A later clean rebuild re-enables it.
- [ ] Add tests for exact legacy groups remaining separate after migration,
  active/deleted topic behavior, ambiguous topic lists, constraint
  block/re-enable, and support preservation across rebuild.
- [ ] Run:

```bash
uv run --no-sync pytest tests/test_store_migrations.py tests/test_store.py \
  tests/test_memory_reindex.py tests/test_support_count.py -v
uv run --no-sync ruff check src/memo/store/migrations.py \
  src/memo/store/schema.py src/memo/store/queries.py \
  src/memo/store/vec_base.py tests/test_store_migrations.py \
  tests/test_store.py tests/test_memory_reindex.py
uv run --no-sync mypy src/memo/store
```

- [ ] Commit:

```bash
git add src/memo/store/migrations.py src/memo/store/schema.py \
  src/memo/store/queries.py src/memo/store/vec_base.py \
  src/memo/store/bm25_queries.py tests/test_store_migrations.py \
  tests/test_store.py tests/test_memory_reindex.py
git commit -m "feat: add rebuildable namespaced identity index"
```

---

## Task 5: Resolve create/corroborate/revise under the write lock

**Files:**
- Modify: `src/memo/memory/write_ops.py`
- Modify: `src/memo/memory/update_ops.py`
- Modify: `src/memo/project.py`
- Modify: `src/memo/server_core_records.py`
- Modify: `src/memo/cli_memory.py`
- Modify: `src/memo/flags_misc.py`
- Test: `tests/test_engram_trust_invariants.py`
- Test: `tests/test_memory_write.py`
- Test: `tests/test_write_ops_buckets.py`
- Test: `tests/test_server_save_scope.py`
- Test: `tests/test_support_count.py`

- [ ] Write failing tests for the complete namespace matrix: same topic in two
  projects creates two IDs; same topic in project/global/unscoped creates three;
  repeated same-namespace topic corroborates; incompatible project tags mutate
  neither disk nor store.
- [ ] Write failing exact-identity tests: exact duplicate returns the original
  ID with `action=corroborated`, creates no Markdown/history row, increments
  support exactly once, and updates `last_seen_at`. Different namespace, type,
  normalized title, or normalized body creates a new record.
- [ ] Write conflict/revision tests for all six precedence rules in the
  canonical contract. A same-topic changed body must keep ID/path/created and
  produce one version snapshot with `action=revised`.
- [ ] Refactor save into a small decision function returning a typed internal
  outcome (`create`, `corroborate`, `revise`, `conflict`). Do not let
  `write_ops.py` grow another monolithic branch; keep normalization in
  `memo.identity` and storage queries in `VecStore`.
- [ ] Use a two-check protocol: optimistic identity lookup may avoid needless
  embedding, but acquire the data-dir lock and repeat both topic and exact
  lookups before any mutation. If embedding is needed, compute it outside the
  lock and pass it to a private update/create helper; re-resolve after acquiring
  the lock because another process may have won the race.
- [ ] For create, write the sanitized Markdown and text-only identity
  reservation coherently before releasing the lock, then preserve the existing
  vector completion/index-pending behavior. Build physical paths from the
  chosen namespace so `_global` and `_unscoped` are distinguishable on reindex;
  keep legacy path resolution working.
- [ ] For revise, call the same versioned update implementation as
  `Memory.update()`; do not duplicate overwrite/history logic inside `save()`.
  Preserve topic key, signals, created timestamp, and path.
- [ ] For corroborate, call only `corroborate_identity()` and return a
  `MemoryRecord` view of the canonical record. Do not rewrite Markdown, embed,
  create history, or emit a second save receipt.
- [ ] Return ephemeral `action` and `index_pending` fields from save. CLI JSON
  and MCP naturally receive them through `to_dict()`; plain CLI output remains
  readable. Candidate/judgment fields are explicitly deferred to Phase 2.
- [ ] Preserve `MEMO_DEDUP_EXACT` as a registered compatibility setting but
  make its description clear that it no longer disables the correctness
  invariant. Remove behavioral branching on the flag.
- [ ] Remove the remaining identity xfails from Task 1.
- [ ] Add a `ThreadPoolExecutor` and a real two-process regression: N concurrent
  identical saves produce one ID/file and `support_count == N-1`; concurrent
  different-project saves never collide.
- [ ] Run:

```bash
uv run --no-sync pytest tests/test_engram_trust_invariants.py \
  tests/test_memory_write.py tests/test_write_ops_buckets.py \
  tests/test_server_save_scope.py tests/test_support_count.py -v
```

- [ ] Commit:

```bash
git add src/memo/memory/write_ops.py src/memo/memory/update_ops.py \
  src/memo/project.py src/memo/server_core_records.py src/memo/cli_memory.py \
  src/memo/flags_misc.py tests/test_engram_trust_invariants.py \
  tests/test_memory_write.py tests/test_write_ops_buckets.py \
  tests/test_server_save_scope.py tests/test_support_count.py
git commit -m "feat: resolve namespaced memory writes deterministically"
```

---

## Task 6: Make update, reindex, delete rollback, and crash recovery coherent

**Files:**
- Modify: `src/memo/memory/update_ops.py`
- Modify: `src/memo/memory/maintain_ops.py`
- Modify if conflict restoration requires it: `src/memo/memory/delete_ops.py`
- Modify: `src/memo/store/queries.py`
- Test: `tests/test_memory_reindex.py`
- Test: `tests/test_write_ops_failure.py`
- Test: `tests/test_memory_write.py`
- Test: `tests/test_cli_migrate_vault.py`

- [ ] Add failing tests that retagging a record recomputes namespace and rejects
  an occupied topic identity without changing Markdown, SQLite, history, or
  signals. Title/body/type changes recompute normalized identity fields.
- [ ] Add reindex tests for `_global`, `_unscoped`, project, ambiguous tags,
  canonicalized topic keys, hand-edited content hashes, and topic collisions.
  Rebuild must preserve support/access/feedback and must not merge exact rows.
- [ ] Thread identity fields through update's embedding and metadata-only paths.
  Check the prospective identity under the existing lock before atomic file
  replacement. A rejected update raises `IdentityConflictError` with no visible
  mutation.
- [ ] During reindex, sanitize only the derived index representation, derive all
  identity fields from the canonical Markdown record/path, and use the atomic
  rebuild preflight from Task 4. Never rewrite hand-edited Markdown.
- [ ] Add failure injection at: revision snapshot, Markdown atomic replace,
  text-only reservation, vector upsert, corroboration transaction, unique-index
  reconciliation, receipt emission, and reindex replacement. For each point,
  assert one truthful outcome: full no-mutation rollback, or canonical Markdown
  plus detectable `index_pending` that reindex repairs.
- [ ] Verify delete/update rollback cannot reactivate or overwrite a topic now
  owned by another active record. If there is no public restore operation,
  cover the internal rollback/reindex path and document that public restore is
  out of scope rather than adding a new command.
- [ ] Do **not** add an intent journal unless a test demonstrates an
  undetectable state with the existing atomic file + text reservation + pending
  marker mechanisms. If a journal is necessary, keep it minimal in state-dir,
  version its entries, recover idempotently on doctor/reindex, and add crash-
  restart tests before merging.
- [ ] Run:

```bash
uv run --no-sync pytest tests/test_memory_reindex.py \
  tests/test_write_ops_failure.py tests/test_memory_write.py \
  tests/test_cli_migrate_vault.py -v
```

- [ ] Commit:

```bash
git add src/memo/memory/update_ops.py src/memo/memory/maintain_ops.py \
  src/memo/memory/delete_ops.py src/memo/store/queries.py \
  tests/test_memory_reindex.py tests/test_write_ops_failure.py \
  tests/test_memory_write.py tests/test_cli_migrate_vault.py
git commit -m "fix: preserve memory identity across recovery and reindex"
```

---

## Task 7: Add read-only trust preflight to doctor and document compatibility

**Files:**
- Create: `src/memo/trust_preflight.py`
- Modify: `src/memo/cli_diag.py`
- Modify: `src/memo/cli_doctor.py`
- Modify: `docs/privacy.md`
- Modify: `CHANGELOG.md`
- Test: `tests/test_cli_doctor.py`
- Test: `tests/test_engram_trust_invariants.py`

**Doctor result shape:**

```json
{
  "trust": {
    "ok": false,
    "identity_constraint": "enabled|blocked|unavailable",
    "multiple_project_tag_rows": 0,
    "topic_collision_groups": 0,
    "exact_duplicate_groups": 0,
    "legacy_identity_rows": 0,
    "secret_pattern_files": 0,
    "private_marker_files": 0
  }
}
```

- [ ] Implement a read-only, no-model preflight. Reuse `memo.identity` and
  `memo.redact` scanners. Return counts and at most sanitized relative paths;
  never include excerpts, matched values, content hashes derived from a secret,
  or raw frontmatter values in user output.
- [ ] Integrate it into `memo doctor --db` JSON and text output. It must never
  auto-merge, rewrite Markdown, drop an index, or repair identity. Existing
  `--gc --fix` semantics remain unrelated.
- [ ] Distinguish corruption from migration state: historical topic/exact groups
  make trust status actionable but do not prevent normal reads; missing v5
  columns report `unavailable` with `memo reindex --rebuild`/upgrade guidance.
- [ ] Add tests with a canary token asserting the complete serialized JSON and
  captured console output do not contain the token. Test clean, blocked,
  ambiguous-tag, legacy-v4, secret-pattern, and private-marker corpora.
- [ ] Update privacy docs and changelog with: mandatory final boundary,
  entropy opt-in, namespace definitions, new additive save fields, legacy
  `normalized_hash` preservation, migration behavior, and repair command.
- [ ] Explicitly state that Phase 1 doctor does not yet report relation/review/
  installer preflight; those arrive with their owning phases.
- [ ] Run:

```bash
uv run --no-sync pytest tests/test_cli_doctor.py \
  tests/test_engram_trust_invariants.py -v
uv run --no-sync ruff check src/memo/trust_preflight.py \
  src/memo/cli_diag.py src/memo/cli_doctor.py tests/test_cli_doctor.py
uv run --no-sync mypy src/memo/trust_preflight.py src/memo/cli_diag.py
```

- [ ] Commit:

```bash
git add src/memo/trust_preflight.py src/memo/cli_diag.py \
  src/memo/cli_doctor.py docs/privacy.md CHANGELOG.md \
  tests/test_cli_doctor.py tests/test_engram_trust_invariants.py
git commit -m "feat: diagnose memory trust invariants"
```

---

## Task 8: P1 verification, scorecard, and release gate

**Files:**
- Modify: `docs/superpowers/reports/2026-07-22-engram-trust-p0.md`
- Create: `docs/superpowers/reports/2026-07-22-engram-trust-p1.md`
- Modify only if verification reveals defects: files owned by Tasks 2–7

- [ ] Confirm `tests/test_engram_trust_invariants.py` has zero xfails/skips and
  every P0 gap now passes as a normal test.
- [ ] Run the focused trust suite:

```bash
uv run --no-sync pytest tests/test_identity.py tests/test_redact.py \
  tests/test_engram_trust_invariants.py tests/test_memory_write.py \
  tests/test_write_ops_failure.py tests/test_write_ops_buckets.py \
  tests/test_memory_reindex.py tests/test_store.py \
  tests/test_store_migrations.py tests/test_support_count.py \
  tests/test_server_save_scope.py tests/test_cli_doctor.py \
  tests/test_cli_migrate_vault.py -v
```

- [ ] Run CI parity in required order:

```bash
uv run --no-sync ruff check src/ tests/
uv run --no-sync mypy src/memo
uv run --no-sync pytest -m "not slow" -n auto --timeout=120 \
  --cov=memo --cov-report=term-missing
```

- [ ] Run retrieval regression because reindex metadata and indexed text changed:

```bash
uv run --no-sync memo eval recall \
  --labels eval/regression_labels.json --k 5 --force
```

  Gate: precision/recall metrics are flat or better and noise is flat or lower.
  If worse, fix the systemic indexing behavior; do not patch individual labels
  or queries.

- [ ] Run the 100-save latency probe from P0 on the same host/config. Record
  before/after p50/p95, file count, support count, and whether exact duplicates
  avoided embeddings. Investigate a p95 regression greater than 10%; do not
  weaken correctness to hide it.
- [ ] Run the slow suite serially when local resources permit:

```bash
uv run --no-sync pytest -m "slow" --timeout=300 -v
```

- [ ] Fill the P1 report with exact commands, commit SHA, environment, outcomes,
  remaining historical diagnostics, and the decision on whether an intent
  journal was needed. Do not claim a check ran if it did not.
- [ ] Reconcile the original scorecard:

  - storage-boundary privacy: zero canary leaks;
  - namespace matrix: zero cross-namespace collisions;
  - exact corroboration: one canonical ID/file, support `N-1`;
  - topic revision: same ID/path/created plus one prior version;
  - failure injection: every outcome rollback or detectable repair;
  - retrieval: gate met;
  - latency: measured and explained.

- [ ] Review the final diff for accidental Phase 2–4 work and unrelated dirty
  files. Stage only the two reports plus any narrowly justified fixes.
- [ ] Commit the proof:

```bash
git add docs/superpowers/reports/2026-07-22-engram-trust-p0.md \
  docs/superpowers/reports/2026-07-22-engram-trust-p1.md
git commit -m "docs: record Engram trust invariant proof"
```

## Definition of done

- Direct save/update, CLI, MCP, capture, ingest, and reindex cannot persist
  known secret patterns or `<private>` content into Markdown-derived stores,
  history, receipts, or logs.
- Project/global/unscoped topic identities coexist without overwrite.
- Repeated exact evidence returns one canonical ID and strengthens
  `support_count`; changed topic content creates a true versioned revision.
- Legacy corpora open non-destructively, surface ambiguity, and remain readable.
- Save responses add truthful `action` and `index_pending` without breaking
  existing clients.
- Recovery tests prove each cross-store failure is either rolled back or
  detectable and repairable.
- CI order, focused tests, retrieval regression, and measured latency gates are
  recorded in the P1 report.
- No relation, lifecycle, coordinator, or setup implementation is included.
