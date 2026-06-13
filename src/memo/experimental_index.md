# Experimental Modules

These modules still ship in the `memo` package but are **not** part of the
stable core product contract. Some already have CLI or MCP entrypoints; that
does not make them part of the supported core path. Their interfaces and
behavior may still change without notice.

The stable core is:

- durable memory CRUD and retrieval
- ambient recall / briefing / recall-daemon
- reindex / doctor / runtime health
- history, diff, and `as-of` time-machine flows

Use this file as the boundary marker for corpus-level experiments and
faster-moving advanced features.

Memo no longer ships autonomous-agent or cognitive-state modules. Those
experiments crossed the product boundary: Synapse owns orchestration and
front-door policy, while Memo owns local semantic storage, retrieval, replay,
and backend-agnostic receipts.

## multimodal.py

Multi-modal memory with universal embeddings. Captures non-text artefacts
(diagrams, screenshots, audio clips) and enables cross-modal semantic search
— e.g. searching for "architecture diagram" can surface image memories. Relies
on external vision/audio embedding models not bundled with the standard install.

## collaborative.py

Collaborative social memory graph. Shares memorias and derived knowledge
connections across multiple users so that discoveries made by one user can
surface for others. Requires a shared storage backend that is not yet
implemented in the core.

## federation.py

Multi-vault federation. Aggregates search results from several independent
memo vaults (different machines or projects) into a single ranked list with
deduplication. Vault discovery and connection management are placeholder
implementations.

## sharing.py

Memory sharing and permissions. Lets users share individual memorias with
other users via temporary links or explicit permission grants, with optional
expiration and comment threads. Requires a network transport layer that is
not bundled.

## encryption.py

AES-256-GCM per-memoria encryption. Encrypts sensitive memories on disk
using a password-derived key (PBKDF2), with support for tag-based selective
encryption or full-corpus encryption. Key management and rotation are not
yet hardened for production use.

## contradict.py

Contradiction and staleness radar. Scans the corpus for pairs of memories
that contradict each other or have become stale using `TemporalAnalyzer`
classifications, stores verdicts in a sidecar `contradictions.db`, and
exposes a triage workflow for resolving them. Scanning a large corpus is
expensive (O(n²) vec lookups).

## chunker.py

Heading-aware markdown chunker. Splits long memories into sub-document
chunks so that individual sections get their own embedding — useful for
audit reports, long notes, or multi-section documents where a single
1024-dim vector dilutes retrieval signal. Not yet wired into the reindex
pipeline.

## crossref.py

Cross-reference and backlink system. Parses Obsidian-style `[[wikilinks]]`
in memoria content, builds a backlink index, suggests links when saving, and
enables multi-hop `ask()` traversals. The index schema is a draft and has
not been tested against the live store.

## contextual.py

Contextual recall enhancement. Maintains a sliding window of recent prompts
and uses it to boost or penalize candidates during retrieval — e.g. memories
referencing entities mentioned in the last 10 prompts get a score boost.
Preference learning (tracking which hits the user follows up on) is stubbed.

## lifecycle.py

Memory lifecycle management. Archives inactive memories to a subdirectory,
promotes frequently-accessed ones, and expires debug or temporary memories
according to configurable policies. Access tracking hooks are not yet
integrated with the core save/search paths.

## navigation.py

Graph-based memory navigation. Extends `graph.py` with BFS shortest-path
finding between entities, connected-component community detection, and
Graphviz DOT / JSON export for visualisation. Complements `mapa` but
operates on the entity graph rather than the embedding space.

## proactive.py

Proactive memory suggestions. Monitors ongoing conversations for patterns
— repeated themes, decision points, discoveries — and surfaces save
suggestions before the user issues a manual `/memo save`. The suggestion
engine uses the helper LLM; confidence scoring is heuristic.

## sync.py

Multi-device sync and automated backup. Computes a content-addressed diff
of two vaults, pushes/pulls changed memories, resolves conflicts, and
produces compressed backups. No transport or conflict-resolution strategy
is finalised.

## versioning.py

Per-memoria version history and diff UI. Stores a snapshot of each memory
on every update, visualises unified diffs between versions, and supports
rollback to any prior state. The version store is separate from `history.db`
and is not yet garbage-collected.
