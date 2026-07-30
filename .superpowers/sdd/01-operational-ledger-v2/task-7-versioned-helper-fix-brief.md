# P01-T07 versioned Secure Enclave helper fix brief

BASE: `ed306e71`

Technical predecessor under review failure: `23a64ccb`.

Design evidence: Memo `9be8463ae2684ef599339fa47600b35c`.

Status: fix loop not started. Operational v2 remains dormant. This brief does
not authorize a production key, activation stamp, Keychain mutation, helper
cleanup, cutover, or service change.

## Objective

Replace runtime Swift compilation and cross-version Keychain ACL assumptions
with a packaged macOS arm64 helper whose immutable hash is permanently bound
to every key it creates.

Upgrades and downgrades may select a different helper only for new keys.
Reopen/sign/destroy of an existing key must always verify and execute the
exact original helper recorded before `SecItemAdd`.

## Closed security model

- Retain the default Keychain ACL. Do not attempt to make future ad-hoc
  identities trusted by an old item.
- Never probe multiple historical helpers against Keychain.
- Never reconstruct a missing/corrupt binding or fall back to the current
  wheel helper.
- Never persist a caller-controlled helper path.
- Same-UID filesystem compromise is outside the approved threat model. If it
  becomes in-scope, this design is insufficient without Developer ID/access
  group authority.
- Only `darwin/arm64` may use the productive backend. Other platforms and
  architectures fail closed.

## Immutable layout

Under Memo's existing secured native-tools authority:

```text
helpers-v1/<helper_sha256>
bindings-v1/<binding_sha256>.json
```

Derive `binding_sha256` as:

```text
SHA256("memo.secure-enclave-binding.v1\0" || service || "\0" || key_id)
```

The canonical binding has the exact fields:

```json
{
  "helper_sha256": "<64 lowercase hex>",
  "key_id": "<validated key id>",
  "schema": "memo.secure_enclave_helper_binding.v1",
  "service": "<validated service>",
  "state": "generating|active|destroying"
}
```

Validation must prove canonical bytes, exact field set/schema, derived
filename, expected uid, regular file, link count one, mode `0600`, bounded
size, and safe descriptor-relative ancestry. The helper path is derived only
from its digest and must be a regular same-uid one-link file with mode `0500`,
exact SHA-256, expected Mach-O architecture/minOS, and successful
`codesign --verify --strict`.

All binding transitions and helper execution hold the existing
`authority_write_lock(native_tools_root)`.

## Transaction protocol

### Generate

1. Use a versioned productive service namespace
   `com.memo.operational-signing.v2`; never reuse the prototype/default
   namespace implicitly.
2. Install or verify the packaged helper into `helpers-v1/<sha256>` using
   exclusive descriptor-relative publication and file/directory fsync.
3. Create a canonical `generating` binding exclusively and fsync it and its
   parent before any Keychain mutation.
4. Execute `SecItemAdd` with that exact helper.
5. Validate the returned public point/key id.
6. Atomically replace `generating` with `active`, fsync file and directory,
   then return success.

If recovery finds `generating`, it must invoke destroy with the bound helper.
Only confirmed delete or `errSecItemNotFound` permits durable binding removal.
Timeout or ambiguous failure retains `generating`.

### Reopen/sign

Reopen is represented by a new backend instance followed by sign/destroy.
Signing accepts only `active`, resolves and re-verifies the bound helper, and
executes it while holding the authority lock. Missing/tampered helper or
binding fails before subprocess execution.

### Destroy

1. Atomically persist `active -> destroying` and fsync.
2. Execute `SecItemDelete` with the bound original helper.
3. Remove and fsync the binding only after confirmed success or
   `errSecItemNotFound`.

Timeout or failure retains `destroying`; retry uses the same helper. A sign
cannot race across `destroying`.

## Helper retention

Reference count is derived from every valid `generating`, `active`, and
`destroying` binding. Do not persist a separate counter.

Do not implement automatic helper GC in this fix. Historical helpers,
including zero-ref helpers, remain. A corrupt binding cancels any future GC.
Do not inspect, execute, or delete the nine residual local helpers found
during read-only design analysis.

## Packaging

- Add the precompiled ad-hoc-signed Mach-O as a darwin-arm64 package asset.
- Preserve Swift source for source distribution/build provenance, but exclude
  it from wheels.
- The generic `py3-none-any` wheel must contain neither Mach-O nor Swift and
  must fail closed if the productive Secure Enclave backend is requested.
- Publish a correctly tagged macOS arm64 platform wheel from a native macOS
  runner. The tag must match the Mach-O `LC_BUILD_VERSION` minimum OS.
- A clean install from the platform wheel must generate/reopen/sign/destroy
  without invoking or requiring `swiftc`.
- Release workflow must publish sdist, generic wheel, and platform wheel for
  the same release and verify their contents independently.

Likely owned files:

- `src/memo/operational_macos_secure_enclave.py`
- `src/memo/native/memo_secure_enclave_helper.swift`
- `src/memo/native/darwin-arm64/memo-secure-enclave-helper`
- `hatch_build.py`
- `pyproject.toml`
- `.github/workflows/publish.yml`
- `.github/workflows/macos-smoke.yml`
- `tests/test_operational_key_store.py`
- `tests/test_release_workflows.py`

Do not edit activation, ledger, roster, signing contracts, Memflow, Synapse, or
live configuration.

## Required RED tests

Write and observe these failures before production code:

1. generating binding is durable before Keychain add;
2. crash after add recovers with original helper;
3. sign after upgrade uses bound helper, never current helper;
4. destroy after upgrade uses bound helper and unlinks only after absence;
5. downgrade preserves newer-key/newer-helper binding;
6. missing binding never falls back;
7. tampered/noncanonical/wrong-mode/symlink binding fails before exec;
8. missing/tampered/wrong-arch/wrong-mode helper fails before exec;
9. destroy timeout retains `destroying`;
10. delete-success/index-cleanup-failure is retryable;
11. concurrent sign/destroy serializes;
12. destroying one of two keys never removes their shared helper;
13. zero-ref helper remains;
14. generic wheel has no Swift or Mach-O;
15. platform wheel is correctly tagged and contains the precompiled helper;
16. clean platform-wheel install never invokes `swiftc`.

The opt-in productive M3 gate must use two genuinely distinct helper builds:

```text
wheel A generate key A
wheel B reopen/sign/destroy key A
wheel B generate key B
downgrade to wheel A
wheel A reopen/sign/destroy key B
```

Both helper hashes/CDHashes must differ and every operation must remain
non-interactive.

## Existing-authority gate

Historical enrolled keys without bindings cannot be migrated automatically.

Before any productive activation, prove from immutable Memo activation/roster
authority that no production P-256 Secure Enclave authority was enrolled by
`23a64ccb` or a prototype. Record exact read-only evidence for every approved
state root and installed production package:

- no operational-v2 root, activation marker, roster/history, epoch marker, or
  migration stamp;
- no P-256 algorithm or key id in whitelisted operational metadata;
- installed production CLI lacks the prototype backend/helper;
- prototype smoke used an isolated UUID service and completed destroy.

The safe result is named `CLEAN_ENROLLED_STATE_KEYCHAIN_UNOBSERVED`. It proves
there is no enrolled authority to migrate; it deliberately does not claim
that no raw Keychain item exists.

A raw orphan without roster, public record, binding, or activation cannot
authorize an operational signature. Never probe historical helpers or query
Keychain to resolve that irrelevant uncertainty. The versioned v2 service
namespace and random key id prevent a new bound key from adopting such an
orphan.

If any enrolled metadata residue exists, stop with `FAIL_METADATA_RESIDUE`.
Migration then requires a separate explicitly authorized procedure using a
known helper and verification against the expected public key.

## Acceptance gates

- RED evidence recorded, then all focused tests GREEN.
- Productive M3 two-wheel upgrade/downgrade gate GREEN without prompts.
- Platform wheel inspected for tag, architecture, minOS, asset hash and
  absence of Swift/runtime compiler dependency.
- Generic wheel remains portable and contains no native helper.
- Ruff, format, strict mypy, release tests and `git diff --check` pass.
- Existing Operational Ledger and frozen-v1 gates pass.
- Exact review package from this BASE.
- Independent security/durability/packaging review returns PASS with no
  BLOCKER/HIGH/MEDIUM.
- No activation stamp, production key, service change, helper deletion, or
  cutover occurs during implementation/review.
