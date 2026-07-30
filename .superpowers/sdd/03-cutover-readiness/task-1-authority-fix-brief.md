# P03-T01 authority hardening brief

BASE: `19570d540a358e7cbe0ec7df68839459eb7a0a76`

Normative task:
`docs/superpowers/plans/2026-07-29-memflow-absorption/03-cutover-readiness.md`,
Task 1.

Prior audit record: Memo `f5e16aae5a9349eb84c48e04cf12c367`.

Status: fix loop not started. No production evidence, manifest freeze, cutover,
or uninstall is authorized by this brief.

## Findings still open at BASE

### BLOCKER — snapshot inputs are not rebound to the published receipts

`tools/memflow_absorption/manifest.py::_load_json()` follows pathnames with
`Path.read_bytes()`. `_receipt_digest()` trusts only the declared `sha256`
inside `snapshot-receipt.json`. A caller can replace or redirect
`source.json`, `usage.json`, or `mapping-candidates.json` after snapshot
publication and still obtain a signed manifest over unreceipted bytes.

The manifest builder must consume each input through one retained,
descriptor-relative, `O_NOFOLLOW` authority and prove that:

- the input path is the exact target named by its adjacent receipt;
- target and receipt are regular, non-symlink files below the retained root;
- receipt bytes are canonical and match the supported exact schema;
- target bytes, size, mode, mtime, device, and inode match the receipt;
- target identity is unchanged before and after the read;
- the root identity remains the retained authority throughout the build;
- malformed, missing, stale, swapped, extra, or duplicate receipts fail
  closed before signing.

Do not add a compatibility fallback that accepts the old partial receipt.

### HIGH — snapshot publication does not attest the target artifact

`SnapshotReceipt` records source metadata plus a content digest, but not the
published target identity/mode/mtime. `create_readonly_snapshot()` does not
reopen and reverify both target and receipt after publication.

Publish a new exact receipt schema that binds at least:

- absolute source and target paths;
- source `(device, inode, size, mtime_ns, mode)`;
- target `(device, inode, size, mtime_ns, mode)`;
- SHA-256 of the exact target bytes;
- canonical receipt schema/version.

The target must be read-only, the receipt must be read-only, and both must be
reopened descriptor-relative with `O_NOFOLLOW` after publish. Retained parent
and target identity checks must make parent/root displacement fail closed.
Use exclusive create, fsync file data, fsync directory entries, and compensate
only artifacts created by the current attempt.

### HIGH — freshness and 90-day coverage are self-asserted

`usage.json["fresh_source_receipts"][device] is True` and
`usage.json["coverage_gaps"] == []` are declarations, not evidence.

Replace them with signed, structured per-device source receipts whose
canonical payload is covered by the existing per-Mac signature authority and
contains:

- device/key/roster identity;
- exact query and extractor version;
- exact source snapshot commit and raw-event-set digest;
- inclusive window start/end;
- issued-at and collected-at;
- ordered hourly coverage buckets for the complete 90-day interval;
- count/digest for every bucket, including explicit zero-event buckets;
- source cursor/watermark and extraction-complete marker.

The verifier must derive expected buckets from the exact UTC window, reject
missing/duplicate/out-of-order/out-of-window buckets, verify every bucket
digest/count against admitted events, and enforce a bounded freshness policy
relative to `frozen_at`. It must not trust a caller-provided boolean or
caller-provided list of gaps.

### HIGH — transforms and fixture coverage are not executable authority

The closed predicate language and pairwise overlap checks introduced before
this BASE are retained; do not reimplement them.

The remaining gap is that any non-empty `transform_id` is accepted and fixture
digests are bound to files without proving that fixtures exercise the route.
Introduce a frozen transform registry whose entries bind:

- exact `transform_id`;
- implementation/module identifier;
- implementation SHA-256 from immutable bytes;
- accepted input/output schema ids;
- deterministic version.

Reject transforms absent from the registry or whose code/schema digest does
not match. Route fixtures must contain a canonical request and expected
result/error, not result-only data. For every source operation, evaluate the
closed predicates against all admitted request fixtures and prove:

- every fixture matches exactly one route;
- every route has at least one matching fixture;
- no fixture falls through or matches multiple routes;
- the selected registered transform deterministically produces the expected
  result/error mapping.

The signed operation-map digest must include the transform registry digest and
the exact canonical fixture authority.

## Required RED tests

Add focused regressions before production changes:

1. symlink `source.json`, `usage.json`, and `mapping-candidates.json`;
2. swap same-size target bytes after receipt publication;
3. replace target while preserving path, size, mtime, and mode;
4. replace receipt with a valid digest for different target bytes/path;
5. rename/recreate snapshot root or parent during manifest read;
6. mutate target or receipt between initial verification and signing;
7. target mode/mtime/device/inode mismatch;
8. old/partial receipt schema rejection;
9. self-declared `fresh_source_receipts=True` without signed freshness proof;
10. missing, duplicate, reordered, forged, and out-of-window hourly buckets;
11. bucket count/digest inconsistent with admitted events;
12. stale `collected_at`/watermark relative to `frozen_at`;
13. unknown transform id and mismatched implementation digest;
14. fixture matches no route, multiple routes, or leaves one route untested;
15. transform output differs from canonical expected result/error;
16. successful exact receipt replay remains deterministic.

Record the failing command and failure count before implementation.

## Owned implementation surface

- `tools/memflow_absorption/snapshot.py`
- `tools/memflow_absorption/manifest.py`
- `tools/memflow_absorption/schemas.py`
- `tools/memflow_absorption/safety.py` only if the retained-root primitive
  cannot be reused as-is
- `tests/tools/test_absorption_snapshot.py`
- `tests/tools/test_absorption_manifest.py`
- sanitized fixture/transform registry files required by the new exact schema

Do not edit Secure Enclave, live configuration, services, launch agents,
Memflow state, or P03-T02/P03-T03 files.

## Acceptance gates

- All required REDs observed, then GREEN.
- Existing focused P03-T01 suite and all Task 1 authority tests pass.
- Ruff, format, strict mypy, `git diff --check`, and frozen-v1 hashes pass.
- Receipt/schema changes are exact and fail closed; no permissive legacy path.
- Exact review package from this BASE.
- Independent durability/security/specification review returns PASS with no
  BLOCKER/HIGH/MEDIUM.
- No production snapshot or manifest is created.
- Memo durable record and live checklist updated after acceptance.
