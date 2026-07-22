# Graph-native v2 P2+P3 Implementation Plan

1. Add failing tests for stable code URIs and codegraph metadata resolution.
2. Add versioned code-link rows to the projection and expose them in its read
   model and health output.
3. Add `Memory.graph_trace`, CLI and MCP surfaces with explicit unavailable
   behavior.
4. Add failing tests for curated communities, bridges and their evidence
   packets.
5. Implement `Memory.graph_discover`, CLI and MCP surfaces over the active
   projection only.
6. Route dream community/bridge synthesis through the discovery packet and
   retain exact source-memory/edge provenance.
7. Register graph-domain flags and route every on/off to `graph-config.md`.
8. Update reference docs and the experimental boundary.
9. Run focused tests, ruff, mypy, full non-slow pytest, recall regression A/B,
   Markdown config activation, live smoke, merge and push `master`.
