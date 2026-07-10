# Memo Graph Integration - Full Improvement Design

Date: 2026-07-10
Status: design approved; pending user review of written spec
Scope: graph-assisted recall, explanations, navigation, semantic relations, and eval gates

## Summary

Memo should integrate the graph more deeply, but not by blindly increasing graph
weight in search. Prior work showed that the entity graph is useful for manual
navigation while automatic graph retrieval has not consistently improved eval
metrics. The next graph program should therefore treat graph signal as a
measured, explainable, and mostly additive layer first, with ranking changes
gated by regression tests.

The design covers four user-facing improvements:

- better recall: graph candidates and boosts when the query has specific,
  discriminating entities;
- better explanation: each graph-touched result can explain why it is related;
- better navigation: CLI and MCP graph tools become useful for exploring large
  topic areas;
- better reasoning: typed semantic relations such as `supports`, `contradicts`,
  `supersedes`, `extends`, and `causes` supplement raw co-occurrence.

The implementation should be phased. Each phase must be independently shippable
and reversible, because graph changes can easily add hub noise.

## Current State

Memo already has several graph pieces:

- `GraphStore` stores entities, entity-memory links, co-recall counts,
  materialized `entity_edges`, entity aliases, and basic graph stats.
- `memo graph rebuild` canonicalizes duplicate entities and rebuilds weighted
  edges.
- `GraphNavigator` supports path, neighbors, communities, centrality, and export,
  optionally folding in codegraph.
- `graph_proximity.py` can boost search candidates by 1-hop graph proximity using
  IDF to down-weight ubiquitous entities.
- `Memory._fetch_graph_candidates()` can inject graph candidates into RRF when
  `MEMO_GRAPH_RETRIEVAL_ENABLED` is enabled.
- `Memory._apply_graph_expansion()` can append graph-adjacent records after a
  primary search result set.
- `Memory._apply_co_recall_boost()` can boost memories that frequently surfaced
  together.
- `recall_assoc.build_nudge()` can append bounded, unverified graph nudges to
  recall context.
- Temporal fact edges and contradiction/evolution penalties now exist and can be
  used as a source for typed semantic graph relations.
- The MCP surface already has a consolidated `memo_graph` tool, but the product
  contract should be tightened around explainability, limits, and agent use.

The main gap is coherence. These signals are present, but they are spread across
retrieval, recall, navigation, temporal facts, and MCP without one clear
contract for when graph signal is allowed to affect ranking, when it is only
shown as context, and how success is measured.

## Goals

### G1: Improve Recall Without Hub Noise

Graph retrieval should help when the query contains rare or specific entities,
not when it only touches generic hubs such as `memo`, `synapse`, or `rag`.

Required behavior:

- keep graph ranking features default-conservative;
- gate graph boosts on entity rarity using IDF or a similar document-frequency
  measure;
- cap graph candidate contribution in RRF so graph-only candidates cannot
  dominate strong vector/BM25 candidates;
- keep associative nudges separate from verified search results unless promoted
  by the normal ranking pipeline;
- provide trace fields that show whether a hit came from vec, BM25, graph
  candidate injection, graph proximity, expansion, or co-recall.

Expected gain: better recall for cross-topic and entity-rich questions while
maintaining precision on ordinary lexical/semantic queries.

### G2: Add Graph Explanations

Users and agents should be able to see why a memory is graph-related.

Each graph-touched hit should be able to expose a compact `graph_reason` object:

```json
{
  "mode": "proximity",
  "query_entities": ["mlx", "recall hook"],
  "hit_entities": ["recall hook", "daemon"],
  "shared_entities": [],
  "neighbor_edges": [
    {"from": "mlx", "to": "daemon", "weight": 4.0, "idf": 3.1}
  ],
  "path": ["mlx", "daemon"],
  "relations": [],
  "confidence": "derived"
}
```

The first implementation can return partial reasons. It must not invent missing
attribution. If path details are unavailable, the result should say so rather
than guessing.

Expected gain: higher trust, easier debugging, and better agent behavior because
related context is labeled as evidence rather than silently blended into ranking.

### G3: Improve Navigation And Exploration

The graph should become a practical map of the corpus.

CLI and MCP should support:

- `neighbors`: direct neighbors with weights, document frequencies, and sample
  memories;
- `path`: shortest and weighted paths, with memory evidence for each step;
- `why`: explanation between two entities or two memories;
- `explore`: compact entity overview with aliases, top neighbors, related
  memories, and communities;
- `communities`: bounded community summaries that avoid dumping huge JSON by
  default;
- hub filtering: hide or demote entities whose document frequency is too high
  unless explicitly requested.

Expected gain: users can inspect a topic area, understand where concepts meet,
and debug why retrieval pulled a record.

### G4: Add Typed Semantic Relations

Raw co-occurrence says "these things appeared together"; semantic relations say
"how they relate."

Initial relation types:

- `mentions`: derived from entity-memory links;
- `co_occurs`: derived from weighted entity edges;
- `supports`: one memory/fact supports another;
- `contradicts`: one memory/fact conflicts with another;
- `supersedes`: newer information replaces older information;
- `extends`: one memory builds on another;
- `causes`: one entity/fact is described as causing another.

The first semantic relation source should be deterministic where possible:

- contradiction/evolution pairs from `contradict_store`;
- temporal fact edges from the fact store;
- explicit metadata/frontmatter relationships;
- co-recall as behavioral evidence, labeled separately from factual evidence.

LLM extraction may be added later, but it must be offline, flag-gated, and never
part of the hot recall path.

Expected gain: better reasoning over stale facts, contradictions, follow-up
work, and causal/project narratives.

### G5: Measure Graph Value Explicitly

Graph changes should ship only when they are measurable.

Add graph-specific eval outputs:

- `graph_recall_gain`: expected records recovered only through graph signal;
- `graph_noise_rate`: graph-introduced records that match avoid/noise labels;
- `graph_explanation_coverage`: percentage of graph-touched hits with an honest
  reason object;
- `hub_noise_rate`: graph hits introduced through high-frequency entities;
- `latency_ms_graph`: incremental graph time in search/recall paths.

Recall-affecting defaults should remain off or low until they pass the existing
recall eval gate: precision must not drop and noise must not rise.

Expected gain: graph work becomes tunable instead of subjective.

## Non-Goals

This program does not:

- replace vector or BM25 retrieval;
- make graph signal mandatory for every query;
- add a new graph database or heavy graph dependency;
- run LLM relation extraction in recall hooks;
- expose huge raw graph dumps as the default user experience;
- mark graph-related memories as verified facts unless another subsystem
  provides verification.

## Proposed Architecture

### Component 1: Graph Signal Layer

Create one internal module that normalizes graph-derived signals for retrieval.

Responsibilities:

- extract query entities using existing regex and graph vocabulary matching;
- compute entity rarity and hub suppression;
- collect graph candidates, graph proximity boosts, graph expansion candidates,
  and co-recall boosts through one interface;
- return both score effects and attribution data;
- enforce time and candidate limits.

This module should not own storage. It should call `GraphStore`, `VecStore`, and
existing ranking helpers through narrow methods.

### Component 2: Graph Reason Builder

Add a small reason builder that converts graph signal traces into user-facing
explanations.

Inputs:

- query entities;
- hit memory entities;
- graph signal contributions;
- weighted neighbors or paths;
- semantic relations when present.

Outputs:

- compact JSON for CLI/MCP;
- short text for human output;
- no fabricated reasons.

### Component 3: Semantic Relation Store

Add a relation model over memory IDs, entity IDs, or fact IDs.

Suggested fields:

- `source_kind`: `memory`, `entity`, or `fact`;
- `source_id`;
- `target_kind`;
- `target_id`;
- `relation`;
- `weight`;
- `confidence`;
- `evidence_id`;
- `derived_from`;
- `created_at`;
- `valid_at`;
- `invalid_at`.

Storage will live in `graph.db` as a rebuildable relation table. Markdown
remains source of truth for memories; derived relations must be rebuildable from
metadata, fact edges, contradiction records, and optional extraction receipts.

### Component 4: Navigation Surface

Tighten CLI and MCP around bounded, useful outputs.

Human output should summarize by default. JSON output should remain complete but
bounded by `limit`, `depth`, and `include_hubs` controls.

The MCP `memo_graph` contract should stay consolidated. Agents should not need a
large set of separate graph tools.

### Component 5: Eval And Tuning

Extend existing eval rather than inventing a second harness.

Required changes:

- trace graph contribution in search eval runs;
- report graph-specific metrics alongside precision/noise;
- add or extend regression labels for graph-neighbor expected IDs;
- keep dream tuning constrained by the same gate;
- save baselines for graph configs separately from ordinary retrieval baselines.

## Data Flow

### Search / Ask

1. Query enters search or ask.
2. Normal vec/BM25 candidates are collected.
3. Graph Signal Layer extracts query entities and checks rarity/hub gates.
4. If allowed, graph candidates and graph boosts are computed within limits.
5. Ranking fuses candidates and applies conservative graph score effects.
6. Graph Reason Builder attaches explanations to graph-touched hits.
7. Output includes normal results plus optional graph reasons and trace data.

### Recall Hook

1. Recall retrieves primary relevant memories as today.
2. Associative nudge remains bounded and labeled unverified.
3. Any graph work in the hook obeys a strict deadline.
4. If the deadline is exceeded, graph context is skipped, not partially trusted.

### Navigation

1. User or agent calls `memo graph ...` or `memo_graph`.
2. Navigator builds from materialized weighted edges and optional codegraph.
3. Hub filters and limits are applied.
4. Output includes evidence: memory IDs, edge weights, relation types, and
   aliases where useful.

### Relation Ingestion

1. Deterministic sources emit relation candidates: fact edges, contradiction
   pairs, supersession metadata, co-recall, and entity co-occurrence.
2. Relation Store upserts derived relations with provenance.
3. Reindex/rebuild can reconstruct deterministic relations.
4. Optional LLM relation extraction can add relations later under a flag and
   with receipts.

## Error Handling

Graph features must degrade safely:

- graph DB unavailable: search and ask continue without graph signal;
- stale or empty graph: graph contribution is omitted and trace says unavailable;
- path not found: explanation reports no path found;
- relation store unavailable: typed relations are omitted, co-occurrence still
  works;
- deadline exceeded: recall skips graph additions;
- malformed relation rows: ignore bad rows and log diagnostics.

No graph failure should break CRUD, search, ask, or recall.

## Testing Strategy

Unit tests:

- hub suppression and IDF gating;
- graph candidate score scale remains RRF-compatible;
- graph reason builder emits honest partial reasons;
- semantic relation normalization and upsert idempotency;
- navigation limits and hub filtering;
- deadline behavior in recall graph paths.

Integration tests:

- search with graph enabled recovers graph-neighbor expected IDs;
- search with graph enabled does not promote hub-only noise;
- `memo_graph` returns bounded responses for path, why, explore, and
  communities;
- rebuild reconstructs weighted edges and deterministic semantic relations;
- contradiction/supersession relations affect explanations without making stale
  records look current.

Eval:

- run `memo eval recall --labels eval/regression_labels.json --k 5 --force`
  after ranking-affecting changes;
- report graph-specific metrics in eval output;
- add focused labels for graph-only recall wins and graph-noise cases.

## Phased Plan

### Phase 1: Graph Signal Hygiene

Unify existing graph ranking pieces behind one internal signal layer. Keep
defaults conservative. Add trace fields and graph-specific eval metrics.

Deliverables:

- Graph Signal Layer API;
- unified gates for rarity, hub suppression, candidate caps, and deadlines;
- graph trace output in search/debug paths;
- eval metrics for graph contribution.

### Phase 2: Graph Explanations

Add `graph_reason` to search/context/ask outputs and debug tools.

Deliverables:

- reason builder;
- CLI/MCP explain fields;
- honest unavailable/partial states;
- tests for non-fabrication.

### Phase 3: Navigation UX

Make graph exploration concise by default and rich on demand.

Deliverables:

- improved `why`, `explore`, `neighbors`, and `communities` outputs;
- hub filtering controls;
- bounded MCP responses;
- path evidence with memory IDs and edge weights.

### Phase 4: Semantic Relations

Add typed relation storage and deterministic ingestion from existing systems.

Deliverables:

- relation schema;
- deterministic relation builders for contradiction/evolution/fact metadata;
- relation-aware explanations;
- rebuild path.

### Phase 5: Tuning And Promotion

Only after metrics exist, let dream tuning search graph weights/configs and
promote them if they beat baseline.

Deliverables:

- graph-specific tuning baseline;
- auto-revert on precision/noise regression;
- documentation for flags and safe defaults.

## Success Criteria

The program succeeds when:

- graph-enabled eval shows recall gains without precision loss or noise increase;
- graph-touched results explain their graph connection;
- `memo graph` and `memo_graph` are useful without dumping huge JSON by default;
- semantic relations improve stale/contradictory context handling;
- recall hook latency remains within budget;
- graph failures degrade to ordinary search instead of failing requests.

## Fixed Design Decisions

The implementation plan should use these defaults unless a test or compatibility
issue proves they need to change:

- relation storage belongs in `graph.db`, not a new sidecar;
- new flags use the existing `MEMO_GRAPH_*` namespace;
- proposed flags are `MEMO_GRAPH_SIGNAL_ENABLED`,
  `MEMO_GRAPH_REASON_ENABLED`, `MEMO_GRAPH_SEMANTIC_RELATIONS`,
  `MEMO_GRAPH_HUB_SUPPRESSION`, and `MEMO_GRAPH_SIGNAL_BUDGET_MS`;
- graph-specific metrics are diagnostic first;
- hard promotion gates remain the existing precision/noise recall eval gates;
- default human output gets compact graph explanations;
- JSON output carries full `graph_reason` detail;
- LLM relation extraction stays out of scope for the first implementation plan.
