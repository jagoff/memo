# Graph-Native v2 P0+P1 — Curated Projection and Evidence-Aware Retrieval

Date: 2026-07-22  
Status: approved by user directive; implementation authorized through `origin/master`  
Scope: P0 graph substrate and P1 retrieval/evidence  
Follow-ups: P2 memory-to-code traceability; P3 communities, bridges, and synthesis

## Summary

Memo already contains an entity graph, weighted co-occurrence edges, semantic
memory relations, graph navigation, associative recall, and several independent
graph ranking experiments. The pieces are useful but incoherent: hot paths read
raw graph state directly, code-like names pollute the knowledge graph, graph
signals use different ranking semantics, and most ranking-affecting paths remain
off because historical evals were neutral or harmful.

P0+P1 introduces a curated, versioned read projection over the existing raw
graph and one bounded graph-signal engine. Raw evidence remains intact and
rebuildable. Retrieval reads only the active projection. Graph signal can
reorder candidates that already passed ordinary retrieval eligibility, but it
cannot make an otherwise ineligible candidate eligible. Every score effect has
an evidence trace, and every failure degrades to the original ranking.

The implementation will remain default-conservative in source code. After a
successful A/B on the real regression set, the completed capabilities will be
enabled on this machine through memo's Markdown configuration files using
`memo config set`, then verified with `memo config validate` and
`memo config show --effective`. Persistent activation must not use shell
exports, raw `os.environ` reads, or source-default changes.

## Evidence and Decisions

The 2026-07-22 baseline has 9,506 entities, 23,436 memory/entity links,
50,491 co-occurrence edges, 6,312 semantic relations, and 2,898 aliases. Of the
entities, 7,118 are typed as `concept`. The largest communities are dominated
by test names and helpers rather than durable knowledge.

The current 37-prompt graph A/B changed:

- precision@5: 0.800 to 0.806;
- recall@5: 0.333 to 0.333;
- nDCG@5: 0.272 to 0.333;
- MRR: 0.250 to 0.333;
- noise@5: 0.0 to 0.0;
- p50 latency: 29.1 ms to 37.4 ms.

This supports graph-assisted ordering, not broad graph candidate generation.
The user selected:

1. a curated projection rather than destructive in-place cleanup or a new
   universal graph database;
2. early activation after one satisfactory A/B rather than multi-night
   graduation;
3. persistent activation through Markdown config files;
4. continued delivery through verification, commit, integration, and push to
   `origin/master` without intermediate approval pauses.

## Goals

### P0: trustworthy serving substrate

- Preserve raw extracted entities and relations as evidence.
- Record extractor provenance and confidence for new entity-memory mentions.
- Materialize a deterministic, versioned projection for retrieval.
- Quarantine noisy nodes without deleting raw evidence.
- Produce stable namespaced node references that P2 and P3 can reuse.
- Rebuild and cut over the projection atomically.
- Make projection freshness and quality visible in graph diagnostics.

### P1: bounded, explainable retrieval signal

- Use one graph-signal implementation across search, ask, context, and recall.
- Reorder only candidates that already passed the primary eligibility gate.
- Use scale-independent rank fusion instead of adding arbitrary values to vec,
  RRF, and reranker scores.
- Attach honest graph attribution to every touched result.
- Preserve base order exactly when the graph is unavailable, stale, irrelevant,
  or over budget.
- Enable the completed path through machine-local Markdown configuration after
  one successful A/B.

## Non-Goals

P0+P1 does not:

- replace vector, BM25, exact, or cross-encoder retrieval;
- add graph-only candidates to the primary result set;
- use two-hop or unbounded graph walks in hot paths;
- import the full codegraph into `graph.db`;
- synthesize communities or bridges;
- run LLM extraction in search or recall;
- delete raw entities merely because the serving projection rejects them;
- introduce a graph library or a new graph database;
- activate superseded graph algorithms alongside the unified engine.

## Architecture

The design has three planes with one-way dependencies.

### 1. Raw evidence plane

Existing `GraphStore` tables remain the derived evidence substrate:

- `entities`;
- `entity_memory`;
- `entity_edges`;
- `entity_aliases`;
- `semantic_relations`;
- `co_recall`.

`entity_memory` gains provenance fields for future writes:

- `extractor`: `legacy`, `regex`, `llm`, or `explicit`;
- `extractor_version`;
- `confidence`, clamped to `[0, 1]`;
- `updated_at`.

Existing rows migrate to `extractor='legacy'` with conservative confidence.
An LLM/explicit extraction is allowed to replace a regex-only membership, so
default-on regex extraction no longer prevents typed upgrades during dream
maintenance. Raw graph CRUD and current navigation remain compatible.

### 2. Curated projection plane

The projection lives in `graph.db` because it is derived from graph evidence,
but it uses separate versioned tables:

```text
graph_projection_versions(
  version, status, built_at, source_fingerprint,
  node_count, edge_count, rejected_count
)

graph_projection_nodes(
  version, uri, entity_id, entity_type, canonical_key, label,
  doc_freq, degree, quality, is_hub
)

graph_projection_memberships(
  version, memory_id, uri, confidence, evidence_id
)

graph_projection_edges(
  version, a_uri, b_uri, relation, weight, confidence,
  evidence_count, first_seen, last_seen, evidence_ids_json
)

graph_projection_rejections(
  version, entity_id, candidate_uri, quality, reason
)

graph_projection_state(key, value)
```

The active version is stored as `graph_projection_state.active_version`.
Projection construction writes a new version, validates it, and changes the
active version in one transaction. Readers never observe a partial rebuild.
One previous version is retained for diagnostics and rollback; older inactive
versions are garbage-collected after cutover.

P0 uses stable references:

- `entity://<type>/<url-encoded-fold-key>`;
- `memory://<memory-id>` in evidence lists;
- `fact://<fact-id>` when semantic evidence references facts.

P2 will implement `codegraph://<repo-id>/<stable-symbol-id>` through the same
reference contract without making code names knowledge entities.

### 3. Serving plane

One `GraphSignalEngine` depends on a narrow `GraphReadModel` interface rather
than raw SQLite tables. It returns:

```text
GraphSignalResult(
  signals: {memory_id: normalized_signal},
  traces: {memory_id: GraphEvidenceTrace},
  skipped: optional reason,
  elapsed_ms
)
```

Search, ask, context, and recall consume the same result. Navigation may still
offer raw/debug views explicitly, but ordinary serving never bypasses the
projection.

## Projection Eligibility and Quality

Projection decisions are deterministic and explainable. A raw node is rejected
when any hard rule applies:

- it has no live memory membership;
- its normalized key is empty;
- it is a bare boolean/null token, number, or date;
- its shape is an obvious test/helper symbol (`test_*`, call syntax, assertion
  fragments) and it lacks explicit non-code evidence;
- every supporting memory is gone or soft-forgotten;
- its only provenance is malformed legacy data.

Other nodes receive a quality score from:

- extractor provenance/confidence;
- entity type specificity;
- number of distinct live supporting memories;
- ratio of durable memory types to repo/test/reference-only memories;
- canonical alias agreement;
- generic-token and code-shape penalties.

The eligibility threshold is a registered graph flag. Rejected nodes are
recorded with a reason, never deleted. Hubs are not rejected: they remain
available for exact queries and navigation but cannot provide ranking signal
unless the query explicitly resolves to that hub. Edges are projected only
when both endpoints are eligible. Edge evidence retains a bounded, stable
sample of supporting memory IDs.

## Rebuild and Freshness

`memo graph rebuild` becomes the single complete rebuild operation:

1. prune orphan memory links;
2. canonicalize aliases;
3. rebuild raw weighted edges;
4. materialize and validate a projection version;
5. atomically activate it;
6. report raw and projected health statistics.

Normal saves remain cheap. They update raw entity membership and mark the graph
projection dirty. Dream/nightly maintenance rebuilds when there are dirty
memberships or the projection exceeds its configured maximum age. Fresh
memories remain retrievable through vec/BM25 while the projection catches up.

Serving checks projection age before scoring. A missing or expired projection
returns an identity result and a trace reason; it never blocks retrieval.

## P1 Retrieval Flow

The hot-path order is:

1. run normal candidate generation and materialization;
2. apply ordinary similarity/body/forgotten/type eligibility gates;
3. extract query entities against the projected vocabulary;
4. discard query entities below the IDF threshold or marked as non-explicit
   hubs;
5. inspect one-hop projected neighbors within the graph deadline;
6. score only the already-eligible candidates from their projected entities;
7. fuse the base candidate order with the graph order;
8. attach evidence traces;
9. continue normal health/quality handling and top-K selection.

For a query entity `q` and candidate entity `h`, edge strength is:

```text
log1p(edge_weight) * idf(q) * idf(h)
------------------------------------
       sqrt(degree(q) * degree(h))
```

The candidate signal is the bounded sum of its strongest distinct entity
connections, normalized to `[0, 1]`. Ranking uses weighted reciprocal-rank
fusion over the existing candidate set:

```text
final = 1 / (rrf_k + base_rank) + alpha / (rrf_k + graph_rank)
```

`alpha` is bounded to `[0, 0.5]`; the initial configured value is conservative.
A candidate with no graph signal keeps only its base component. Stable
base-order tie breaking makes the transform an identity when graph signals are
empty.

This rank fusion is deliberately independent of raw vec, RRF, or reranker score
scales. P1 does not run graph candidate injection or post-search graph
expansion. Those older experiments are deprecated and either removed or
redirected to the unified engine.

## Evidence and Explanations

Every graph-touched hit receives a `graph_reason` containing:

- projection version;
- mode (`curated_proximity`);
- resolved query and hit entity URIs;
- contributing edge relation, normalized weight, confidence, and IDF values;
- bounded evidence memory IDs;
- total normalized signal;
- whether hub suppression affected the path;
- `confidence='derived'` unless another verified subsystem supplies stronger
  evidence.

Human output stays compact. JSON/MCP output carries the structured reason.
Missing evidence is reported as unavailable; the reason builder never invents
a path or relation.

## Configuration and Activation

All new behavioral switches are registered through `memo.flags`; application
code uses `flag_bool`, `flag_int`, and `flag_float`. Proposed flags are:

- `MEMO_GRAPH_PROJECTION_ENABLED`;
- `MEMO_GRAPH_PROJECTION_MIN_QUALITY`;
- `MEMO_GRAPH_PROJECTION_MAX_AGE_HOURS`;
- existing `MEMO_GRAPH_SIGNAL_ENABLED`;
- existing `MEMO_GRAPH_REASON_ENABLED`;
- existing `MEMO_GRAPH_SEMANTIC_RELATIONS`;
- existing `MEMO_GRAPH_HUB_SUPPRESSION`;
- existing `MEMO_GRAPH_SIGNAL_BUDGET_MS`;
- a bounded `MEMO_GRAPH_SIGNAL_ALPHA`.

The flag registry maps graph flags to `graph-config.md`. Recall-specific
associative controls remain in `recall-config.md`. No application path reads
these values through raw `os.environ`.

Source defaults remain safe for new installations. After implementation and a
successful real-corpus A/B, this machine is activated with `memo config set`.
The exact registered config keys are used rather than editing Markdown by hand.
The final activation procedure must:

1. enable projection, graph signal, reasons, semantic relations, and hub
   suppression;
2. keep associative recall enabled;
3. set the chosen `alpha`, quality threshold, age, and deadline;
4. validate Markdown config;
5. show effective values and sources;
6. rebuild the graph and confirm an active projection;
7. restart or reload long-lived memo services if their config cache requires
   it;
8. run a live search/recall smoke proving the active path.

Because Markdown config is machine-specific, this activation applies only to
the current machine. Cross-machine replication is outside P0+P1.

## Failure Handling

Graph behavior is fail-open with respect to retrieval:

- unavailable/corrupt `graph.db`: preserve base ranking;
- no active projection: preserve base ranking and mark `projection_missing`;
- stale projection: preserve base ranking and mark `projection_stale`;
- deadline exceeded: discard the entire graph contribution for that request;
- malformed node/edge/evidence row: skip it and count a diagnostic;
- failed rebuild validation: leave the prior active version untouched;
- activation/config validation failure: do not restart services or claim the
  feature is active.

Normal domain failures use `MemoError` subclasses. Defensive graph decoration
may absorb failures only at the boundary where identity behavior is guaranteed.

## Observability and Evaluation

`memo graph stats` adds:

- active projection version and age;
- eligible/rejected node counts and rejection reasons;
- projected edge count and weight distribution;
- raw-to-projected coverage;
- orphan/dangling counts;
- hub count;
- dirty membership count;
- last successful and failed rebuild metadata.

Search traces report projection version, touched candidates, skip reason,
deadline status, and graph elapsed time.

The eval set must add graph-focused labels before activation:

- rare-entity ordering wins;
- hub-only queries that must remain unchanged;
- polluted test/helper entities that must not contribute;
- stale/forgotten evidence that must not contribute;
- graph explanation truth cases;
- identity cases for missing/stale/deadline projection.

The selected early-activation gate is one complete real-corpus A/B. Activation
requires:

- no precision regression;
- no recall regression;
- no noise increase;
- positive nDCG or MRR movement;
- p50 latency overhead no greater than 15 ms;
- complete honest explanation coverage for graph-touched hits;
- graph-focused tests and standard regression tests green.

If the implementation fails this gate, it remains config-off and the task is
not complete; tuning the bounded parameters is in scope, but broad candidate
injection is not.

## Testing Strategy

Unit tests cover:

- stable URI construction and canonical aliases;
- provenance upgrades from regex to typed extraction;
- every hard rejection rule and quality boundary;
- hub handling;
- projection edge construction and bounded evidence;
- reciprocal-rank fusion identity, bounds, and deterministic ties;
- reason non-fabrication;
- deadline and stale/missing identity behavior.

Integration tests cover:

- migration of a legacy graph DB;
- shadow rebuild and atomic cutover;
- failed rebuild preserving the previous active version;
- orphan pruning;
- reindex/rebuild parity from isolated Markdown fixtures;
- search, ask, context, recall, CLI, and MCP consuming the same engine;
- `memo config set` writing graph flags to the correct Markdown domain;
- config validation and effective-source reporting.

Verification follows repository order:

1. focused graph/search/recall/config tests;
2. `ruff check src/ tests/`;
3. `mypy src/memo`;
4. non-slow pytest CI-parity suite;
5. `memo eval recall --labels eval/regression_labels.json --k 5 --force`;
6. graph A/B with graph diagnostics;
7. macOS/runtime smoke because recall and config paths are affected;
8. post-activation live smoke.

## P2 and P3 Contracts

P2 must implement a `CodeGraphProvider` that returns the same `NodeRef`,
`EdgeRef`, and evidence shapes using stable `codegraph://` URIs. Memo stores
durable links to those URIs as evidence; it does not copy codegraph ownership
into the knowledge graph.

P3 consumes the active curated projection, never raw entity edges, for
communities and bridges. Synthesis stores its source projection version,
entity URIs, and evidence memory IDs. Community/bridge activation gets its own
spec and eval, but uses the same configuration and fail-open conventions.

## Completion Criteria

P0+P1 is complete only when:

- raw evidence remains intact and legacy DBs migrate safely;
- a healthy active projection is rebuildable from isolated source fixtures;
- all graph-serving consumers use the unified engine;
- obsolete duplicate ranking paths are removed or redirected;
- the full verification sequence passes;
- the real-corpus A/B passes the early-activation gate;
- supported graph features are active through Markdown config on this machine;
- effective config, active projection, and live graph traces prove activation;
- the implementation and activation documentation are committed and pushed to
  `origin/master` without force-push.
