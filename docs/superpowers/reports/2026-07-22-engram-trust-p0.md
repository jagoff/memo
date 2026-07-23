# Engram trust invariants — P0 baseline

Date: 2026-07-22

Baseline commit: `a6bfb94c` (identity types present; persistence behavior still
pre-P1)

Host: Apple arm64, Darwin 27.0.0, Python 3.13.9

## Baseline matrix

| Invariant | Probe | Before | Target | Proof command |
|---|---|---|---|---|
| Final privacy boundary | Direct save with known-key and `<private>` canaries, then scan Markdown/SQLite/history/receipts/logs | Capture/ingest preprocessing could be disabled and direct persistence was not the authority | No raw canary in any persisted or emitted representation | `pytest tests/test_engram_trust_invariants.py tests/test_redact.py` |
| Namespaced topic identity | Same canonical topic in project A, project B, explicit global, and unscoped | Global topic lookup could overwrite across projects | Four IDs and four readable records | `pytest tests/test_engram_trust_invariants.py -k namespace` |
| Exact corroboration | Save identical normalized title/body/type/namespace 100 times | 100 IDs/files and 100 embeddings | One ID/file, support 99, one embedding | latency probe below |
| Namespace validation | Save with two incompatible `project:` tags | Ambiguous ownership could enter storage | Typed conflict and byte-for-byte no mutation | `pytest tests/test_engram_trust_invariants.py -k multiple_project` |
| Recoverable index failure | Inject embed/vector failures, then reindex | Markdown already survived most failures | Every result is rollback or canonical Markdown plus repairable pending index | `pytest tests/test_write_ops_failure.py tests/test_memory_reindex.py` |
| Signal preservation | Rebuild a corpus with support/access/feedback state | Supported by the existing atomic rebuild | Signals unchanged after schema-v5 rebuild | `pytest tests/test_store_migrations.py tests/test_support_count.py` |

The four initial trust gaps are represented by normal P1 regression tests in
`tests/test_engram_trust_invariants.py`; the final suite contains no xfails or
skips. The baseline behavior was reproduced from the detached baseline commit
rather than retained as permanent expected-failure tests.

## 100-save latency baseline

Configuration: isolated temp data/vault/state directories, 4-dimensional
constant stub embedder, default exact/near-dedup settings, one title/body,
`auto_project=False`, receipts/logging suppressed only for measurement noise.
Wall time used `time.perf_counter()` around each `Memory.save()`.

| Metric | P0 |
|---|---:|
| p50 | 4.670 ms |
| p95 | 5.511 ms |
| distinct IDs | 100 |
| Markdown files | 100 |
| embedding calls | 100 |
| support on first ID | 1 |

The probe was run from a temporary detached worktree at `a6bfb94c` with the
current dependency environment, then the worktree was removed.
