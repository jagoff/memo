# Graph-native v2 P2+P3 Design

## Goal

Complete the approved graph-native scope after curated retrieval:

1. make links between durable memories and code inspectable in both directions;
2. make graph discovery and synthesis consume the same curated, versioned,
   evidence-bearing projection used by search.

The raw entity graph and codegraph database remain evidence sources. They are
never served directly by these new paths.

## P2: Memory-to-code traceability

### Identity

Code references use:

```text
codegraph://<repo-id>/<url-encoded-stable-symbol-id>
```

`repo-id` is a stable hash of the normalized Git remote (falling back to the
repository root only when no remote exists). `stable-symbol-id` is codegraph's
node id. Code names never become knowledge entities.

### Evidence

Projection rebuild resolves three durable metadata sources:

- `extra.code_refs`: explicit codegraph URIs or structured code references;
- `extra.files_modified`: `modified` file evidence;
- `extra.files_read`: `read` file evidence.

File metadata resolves to codegraph file nodes. Unresolved paths are reported
as unavailable and are not fabricated into projection nodes.

The versioned projection adds `graph_projection_code_links`, containing the
memory id, code URI, relation, file/line metadata and evidence URI. Code nodes
also participate in projection memberships and evidence-bearing co-occurrence
edges, but graph ranking remains bounded to its eligible candidate set.

### Read surface

`Memory.graph_trace()` is the core API. Exactly one of `memory_id` or `code`
is supplied. It returns the active projection version, stable references and
evidence. The same API powers:

- `memo graph trace --memory <id>`;
- `memo graph trace --code <URI|symbol|path>`;
- `memo_graph_trace` MCP.

Missing/stale projection and unknown references return explicit empty results,
never guessed links.

## P3: Discovery and synthesis

### Discovery packet

`Memory.graph_discover()` operates only on the active curated projection. It
returns bounded, deterministic:

- communities (connected components after hub suppression);
- articulation bridges between bounded regions;
- exact edge and memory evidence for each insight candidate.

`memo graph discover` and `memo_graph_discover` expose the packet without
writing memories.

### Synthesis

Existing dream community and bridge passes keep their dedup/save contracts but
take clusters and bridges from `graph_discover()`. Source memories and edge
evidence are persisted in synthesis metadata. If projection is unavailable,
the passes skip explicitly; they do not fall back to the raw graph.

The relevant switches are graph-domain settings and therefore live in
`graph-config.md`:

- `graph.code_trace_enabled`;
- `graph.discovery_enabled`;
- `graph.dream_communities_enabled`;
- `graph.dream_bridges_enabled`;
- `graph.dream_communities_min_size`.

## Safety and activation

- All flags are registered and accessed through `memo.flags`.
- Projection rebuild/cutover stays atomic.
- Read paths are bounded and fail open/read-only.
- Synthesis writes remain gated, deduplicated and provenance-bearing.
- Persistent activation uses `memo config set`, never source-default changes or
  shell exports.

## Verification

Focused unit/CLI/MCP tests prove URI stability, bidirectional traceability,
unavailable behavior, curated-only discovery, synthesis provenance, and config
routing. Final verification follows project CI order plus the real recall A/B.
