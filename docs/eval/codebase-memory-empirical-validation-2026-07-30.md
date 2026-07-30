# Codebase-memory empirical validation — 2026-07-30

## Decision

Memo's codebase-memory work is empirically useful and safe to activate.
Graph-enriched retrieval materially improves path recall over the symmetric
lexical baseline; the complete graph+semantic configuration recovered every
labelled target at `k=10` in three identical runs. Provider absence and artifact
corruption degraded explicitly instead of producing silent evidence, and a real
repository watcher refreshed after a real commit.

The measurements also found four implementation bugs. All four were corrected
before the final runs:

1. CodeGraph symbol lookup scanned the complete nodes table once per query term.
2. Generic symbols such as `test`, and one-term matches, crowded specific
   multi-term identifiers out of the result window.
3. Multiple chunks from one file could consume most of a unified result window,
   and the candidate pool changed too much with the requested limit.
4. A local-repository watcher observed the managed clone rather than the source
   checkout and ignored the branch-ref event that proves a commit happened.

A fifth operational defect was found while building the semantic index: the
default 64-chunk MLX batch exceeded 19 GiB of process footprint. The default is
now 16, with `MEMO_REPO_EMBED_BATCH` retained as an explicit override. The full
test gate then exposed a sixth defect: Tantivy committed its writer without
joining background merge threads, which could race immediate index-directory
cleanup. Close now waits for those threads and is idempotent.

## Provenance and frozen labels

- Reference implementation originally audited:
  `DeusData/codebase-memory-mcp@f2e97ff7c2944e62f57412a33b3e28d379ba2353`.
- Reference HEAD rechecked for this validation:
  `ee9833df2c72b750378273897ffec9410fc2c4f2`.
- Memo base indexed for the reproducible runs:
  `0c3224776b74e2115a21b41ca09434212dfefb69`.
- Label set: [`eval/repo_search_labels.json`](../../eval/repo_search_labels.json),
  36 hand-reviewed queries (22 production and 14 test queries).
- Frozen label SHA-256:
  `5987e2fac2e44f1327b0945613baa95a749a8f07e6c766c16ddf73d93a8f3288`.

One label was corrected before freezing: the provider-degradation query now
accepts `codegraph_loader.py`, because that module is a canonical implementation
of missing-index discovery in addition to the repo provider modules. No expected
path was derived from ranker scores.

The deterministic judge only compares returned paths against the frozen path
set. It does not inspect scores, snippets, provider metadata, or strategy names.
Every label is run adjacently against both strategies:

- `grep-first`: `mode=lexical`;
- `graph-first`: `mode=unified`.

The CLI gate fails if either strategy raises or graph-first recall falls below
grep-first recall.

## Corpus

The exact/full corpus indexed Memo's complete tracked text surface:

| Measurement | Value |
|---|---:|
| Checked files | 1,285 |
| Indexed files | 1,283 |
| Chunks | 4,529 |
| Indexed lines | 319,074 |
| Indexing errors | 0 |
| Git commits analysed | 298 |
| Git co-change pairs | 33,922 |

The CodeGraph index over that managed checkout contained:

| Measurement | Value |
|---|---:|
| Files | 1,040 |
| Nodes | 20,783 |
| Edges | 55,733 |
| Database size | 60.40 MiB |

The semantic-channel experiment used the same complete CodeGraph but embedded
only the 28 unique files named by the frozen labels: 113 chunks and 9,178
lines, all 113 embedded successfully. This makes the semantic comparison
complete for the adjudicated targets without pretending that a long-running
full-corpus embed completed during the experiment.

## Final A/B results

All values below are from three consecutive warm runs at `k=10`; ranking
metrics were identical across the three runs. Latencies are medians of the
reported cumulative search time for 36 queries.

### Full corpus, graph and Git signals, no semantic vectors

| Strategy | Recall@10 | MRR | Precision@10 | Zero-result queries | Failures | Median search time |
|---|---:|---:|---:|---:|---:|---:|
| grep-first | 0.3611 | 0.3611 | 0.9545 | 23 | 0 | 84.3 ms |
| graph-first | 0.8333 | 0.5722 | 0.1000 | 0 | 0 | 2,828.1 ms |

Graph-first adds **47.22 recall points**, raises MRR by **21.11 points**, and
eliminates all 23 empty queries. Its median cost is about 78.6 ms/query versus
2.3 ms/query for the sparse lexical baseline.

### Label-complete semantic slice plus the full graph

| Strategy | Recall@10 | MRR | Precision@10 | Zero-result queries | Failures | Median search time |
|---|---:|---:|---:|---:|---:|---:|
| grep-first | 0.3611 | 0.3611 | 0.9545 | 23 | 0 | 40.2 ms |
| graph-first | **1.0000** | **0.8236** | 0.1778 | 0 | 0 | 4,838.5 ms |

Graph+semantic adds **63.89 recall points**, raises MRR by **46.25 points**, and
recovers every frozen target in all three runs. The median cost is about
134.4 ms/query versus 1.1 ms/query for lexical-only retrieval.

Precision is not directly comparable here: graph-first deliberately returns ten
results for every query, while lexical returns only 22 results across the whole
36-query set. Most labels name one relevant file, so a perfect ten-result window
usually has a maximum judged precision of 0.10. Recall and MRR are the decision
metrics for this sparse path-label design.

## Before/after evidence for the retrieval fixes

On the same complete graph-only corpus, before the structural lookup fixes:

- graph-first recall@10: 0.5278;
- graph-first MRR: 0.3819;
- cumulative graph-first search time: about 5,720 ms.

After indexed segment lookup, multi-term prioritisation, boilerplate filtering,
path diversity, and a stable candidate floor:

- graph-first recall@10: 0.8333;
- graph-first MRR: 0.5722;
- median cumulative graph-first search time: 2,828 ms.

The final implementation therefore gained 30.55 recall points and cut the
graph-only search time by roughly half relative to the first empirical run.
A direct structural micro-probe fell from 118–227 ms to 5–24 ms for the same
query after replacing full scans with CodeGraph's indexed identifier-segment
vocabulary.

## Fault and runtime probes

### Artifact integrity

The 2,223,143-byte Git change-signal artifact was backed up, truncated to 17
bytes, checked through `memo repo status`, and restored.

- During the fault: `ok=false`,
  `ArtifactIntegrityError: artifact size mismatch: expected 2223143, got 17`.
- After restoration: `ok=true`, with the original digest and size.

No corrupted payload was consumed as valid change evidence.

### Missing structural provider

The managed checkout's `.codegraph` directory was temporarily hidden and a real
unified query was executed.

- The query still returned five lexical/Git-backed results.
- Diagnostics reported
  `codegraph: {status: unavailable, reason: index_missing, results: 0}`.
- After restoration, `codegraph status` again reported 1,040 files, 20,783
  nodes, 55,733 edges, and an up-to-date index.

### Incremental watcher

A fresh one-file Git repository was indexed, then watched by the actual
`memo repo watch` CLI with a 150 ms debounce. The tracked file was edited and
committed.

- A pre-commit file event produced an unchanged refresh.
- Git ref updates were debounced into one post-commit refresh.
- The second refresh completed in 0.30 s with `indexed=1`.
- Repo status advanced from commit `4afdee7` to `56ec4bb`.
- Lexical retrieval returned the new `watcher-after-empirical` content.

### Change impact and architecture

After syncing Memo's own CodeGraph against this worktree:

- bounded depth-1 impact was available with no limitations;
- 19 changed files expanded to 642 impacted symbols and their downstream paths;
- audit-mode architecture focused on `search_codegraph_paths` returned six
  findings;
- the pack carried provider generation
  `codegraph:8:1785379694732396520` and explicitly disabled absence claims
  because source-universe completeness was not proven.

### Startup and release policy

Eighty focused tests passed for:

- offline MCP startup by default;
- explicit update/auto-update opt-in;
- no network call or updater process when the flag is unset;
- mandatory Linux and macOS release smoke jobs;
- no downstream publish job bypass through conditional execution.

The publish workflow was already fail-closed; the new test makes that learned
contract durable.

The Tantivy cleanup regression and its surrounding backend suite then passed
420/420 cases across 20 consecutive repetitions, including the exact MCP
retrieval test that exposed the race.

The complete non-slow gate subsequently passed 6,089 tests with one skip and
78.47% coverage (74% required), under 14 parallel workers, randomized ordering,
and 120-second per-test timeouts. Formatting, lint, typing over 461 source files,
and all 168 complexity plus 170 exception budgets also passed. Pytest reported
seven pre-existing unclosed-SQLite `ResourceWarning` instances in unrelated
tests; they did not fail the suite and are not represented here as fixed.

## Activation defaults

The following code-intelligence features remain enabled by default:

- `MEMO_GRAPH_USE_CODEGRAPH=1`;
- `MEMO_CODEGRAPH_DISCOVERY=1`;
- `MEMO_BRIEFING_GRAPH=1`;
- `MEMO_BRIEFING_CODE_DRIFT=1`;
- `MEMO_GAPS_CODE_HUBS=1`.

Normal MCP startup is now offline by default:

- `MEMO_UPDATE_CHECK_ENABLED=0`;
- `MEMO_AUTO_UPDATE=0`.

This is intentional hardening, not an inactive code-memory feature. Operators
can explicitly opt in to either remote-update behavior.

## Reproduction

With a repo already indexed under the name `memo-self` and a CodeGraph index in
its managed checkout:

```bash
memo eval repo-search \
  --labels eval/repo_search_labels.json \
  --repo memo-self \
  --k 10 \
  --json \
  --gate
```

Run the command three times for the stability check. `memo repo status
memo-self --json` reports generation/artifact integrity, and `codegraph status
<managed-clone>` reports structural coverage.

## Boundaries

- This evaluation proves retrieval of the frozen path-labelled tasks; it does
  not prove arbitrary natural-language correctness.
- It does not claim token savings: no agent token counter participated in the
  symmetric search calls.
- The semantic result is target-complete, not full-corpus-complete. Full-corpus
  exact/graph indexing is proven separately.
- Provider absence is represented as unavailable, never as evidence that a
  symbol or relationship does not exist.
- Watcher activation is useful for local source checkouts and commit events; a
  remote-only repo still needs an external fetch/poll trigger to create local
  filesystem events.
