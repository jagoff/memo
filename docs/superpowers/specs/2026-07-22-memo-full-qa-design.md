# memo Full QA Campaign — Design

**Date:** 2026-07-22
**Status:** Approved design, pre-implementation plan
**Campaign type:** Exhaustive layered QA, live dogfood, repair, and regression hardening
**Authority:** The current local worktree, without integrating origin/master first

## Objective

Perform a full product QA of memo from the perspective of real users and MCP
clients. The campaign covers the stable core, advanced and experimental
surfaces, internal wiring, storage, migrations, native MLX runtime, Linux CPU
runtime, Docker, installers, packaging, daemons, hooks, interactive surfaces,
and distribution channels.

The work does not stop at reporting failures. Every confirmed product defect is
reproduced in isolation, reduced to its cause, protected by a regression test,
fixed, and committed atomically. The campaign finishes only after a clean
end-to-end rerun and a traceable coverage report.

This campaign builds on the 2026-07-21 operational audit only as historical
baseline. The prior result is not accepted as current evidence because the
worktree gained substantial temporal, graph, MCP, presence, telemetry, daemon,
dependency, and wiring changes afterward.

## Locked Decisions

| Decision | Approved choice |
|---|---|
| Code authority | Current local worktree exactly as found |
| Remote divergence | Do not merge, rebase, pull, or cherry-pick the three remote-only commits before QA |
| Existing changes | Preserve all pre-existing staged, modified, and untracked files |
| Execution approach | Layered exhaustive campaign |
| Resource level | Real models, model downloads, Docker builds, daemons, local servers, and isolated installs are authorized |
| Fix strategy | Fix defects while advancing, with one atomic commit per coherent defect/domain |
| Data isolation | Never touch the real vault, default state directory, real client configuration, or production daemon state |
| Stable and experimental scope | Test both; classification changes the expected contract, not whether the surface is exercised |
| Interface depth | Real CLI subprocesses and real MCP protocol, not only in-process calls |
| Completion standard | No unexplained surface, skip, failure, timeout, or confirmed defect |

## Meaning of Exhaustive

Exhaustive means every dynamically discoverable installed surface maps to an
explicit scenario and verdict. It does not mean attempting an infinite
Cartesian product of every flag value and every possible input.

Every CLI leaf command, MCP tool, MCP resource, MCP prompt, registered flag,
daemon lifecycle, hook, migration, installer target, package entry point, and
distribution artifact receives:

1. discovery or schema validation;
2. at least one representative successful scenario when the platform supports
   the feature;
3. invalid-input and empty-state coverage where meaningful;
4. persistence, restart, or cleanup validation when it owns state;
5. a clear PASS, FAIL, EXPECTED-UNAVAILABLE, or ENVIRONMENT-BLOCKED verdict.

EXPECTED-UNAVAILABLE is a passing contract only when the product returns the
documented, actionable unsupported-backend response. A traceback, silent
fallback, partial mutation, or misleading success remains a failure.
ENVIRONMENT-BLOCKED requires concrete evidence and cannot be used as a generic
skip.

## Campaign Architecture

The campaign is a sequence of cumulative gates. A gate does not close until its
confirmed defects are fixed and the affected earlier gates pass again.

### Gate 0: Baseline and chain of custody

Record before any QA mutation:

- HEAD, branch, upstream divergence, tags, and worktree status;
- a patch and file list for all pre-existing changes;
- OS, architecture, Python, uv, Docker, Homebrew, compiler, and SQLite versions;
- effective project version and hashes of lockfiles and release manifests;
- installed memo and memo-mcp paths and versions;
- inherited MEMO, SYNAPSE, HF, XDG, and model-related environment names, with
  secrets redacted;
- available disk, memory, ports, model caches, and Docker state;
- test start timestamp and campaign identifier.

The plan must define a machine-readable baseline manifest. Later reports compare
the final worktree and runtime state to this snapshot.

### Gate 1: Surface inventory and wiring

Enumerate surfaces from live code and installed artifacts, never from a stale
hardcoded count:

- recursively walk the Click command tree under every surface profile;
- initialize memo-mcp and list tools, resources, and prompts for every MCP
  profile;
- inspect all server register functions and verify build_server registration;
- map CLI and MCP operations to Memory facade operations and storage paths;
- enumerate flags from the registry and config catalog;
- enumerate entry points, hooks, plugins, packaged assets, daemons, migrations,
  installers, workflows, Docker files, and release manifests;
- compare docs and help examples with live discovery;
- detect imported-but-unregistered modules, dead registrars, duplicate names,
  stale aliases, missing assets, hidden commands without an explicit policy,
  and exposed experimental modules missing from the boundary index.

The output is the coverage manifest. Adding a surface without a scenario is a
harness failure.

### Gate 2: Structural quality and artifact integrity

Run the repository's quality order and the wider release contracts:

- Ruff formatting and lint;
- mypy;
- progressive quality budget;
- lockfile consistency and frozen dependency resolution;
- dependency and supply-chain security checks;
- version agreement across pyproject, plugin manifests, server metadata,
  installer defaults, Docker metadata, and changelog;
- wheel and sdist builds from the current worktree;
- package-content allowlist and secret/runtime-state exclusion checks;
- import-time and deferred-MLX contracts;
- installation of built artifacts into empty environments;
- entry-point and package metadata validation without relying on checkout
  imports.

### Gate 3: Automated verification matrix

Run all existing deterministic and scheduled lanes that can be reproduced
locally:

- Python 3.13 and 3.14;
- non-slow suite with coverage;
- slow suite serially;
- resource-hygiene lane;
- concurrency and database-contract lanes;
- int8 default lane and float32-specific lane;
- test order randomization with replayable seeds;
- repeated focused stability tests;
- bounded property/state-machine tests;
- contract stub and any locally available private-contract integration;
- real MLX smoke;
- no-model and dependency-missing behavior;
- recall, grounding, and graph evaluation suites;
- mutation or targeted fault-injection checks where the existing workflow
  defines them.

An existing test passing is evidence for its contract, but it does not replace
the live user journeys in later gates.

### Gate 4: Native macOS user journeys

Install the built wheel or sdist in fresh isolated runtimes and exercise the
actual memo and memo-mcp executables. Test Python 3.13 and 3.14 where supported,
real Apple Silicon MLX models, model-profile selection, cold and warm caches,
offline reuse, and graceful no-model mode.

The native plane covers human output, JSON output, stdin, TTY behavior, exit
codes, signals, files, SQLite state, sockets, and restarts.

### Gate 5: MCP and network transports

Drive the real MCP protocol:

- initialize handshake and server metadata;
- tools/list, resources/list, prompts/list, schemas, and pagination if exposed;
- call every tool with a mapped scenario;
- read every resource and render every prompt with representative arguments;
- cross-surface round trips in both directions between CLI and MCP;
- agent, full, and any other configured surface profiles;
- stdio startup, shutdown, malformed frames, client disconnect, and stderr
  hygiene;
- HTTP transport on loopback with generated and explicit bearer tokens;
- rejected unauthenticated requests, invalid tokens, unsafe non-loopback binds,
  and explicit allow-non-loopback behavior;
- notification and sampling paths;
- optional REST API parity where semantics overlap.

### Gate 6: Runtime, hooks, daemons, and interactive operation

Exercise recall, embed, ingest, maintenance, and idle daemons through cold start,
warm use, duplicate start, status, stop, restart, stale socket/PID cleanup,
concurrent clients, interrupted work, and shutdown.

Test hook commands with realistic JSON input and strict timeout budgets. Test
first-run noninteractive behavior, recall fail-open behavior, capture, briefing,
presence notifications, statusline, watcher, shell wrappers, and installer
configuration against fake client homes.

Interactive TUI and configuration surfaces are driven through a pseudo-terminal
with deterministic keystrokes, terminal sizes, resize, cancellation, and clean
exit. They are not silently skipped as merely interactive.

Real user launchd labels and client files must not be modified. Plist generation
and lifecycle are tested with unique disposable labels and paths when the
implementation permits it. If the product hardcodes a production label so that
an exact live test would collide with the user's runtime, that collision risk is
recorded and the closest safe lifecycle test is used.

### Gate 7: Linux CPU and Docker

Build the checked-in Dockerfile from the current worktree, including the pinned
base and exact model revision. Verify:

- image build, version assertion, package provenance, and non-root user;
- CPU sentence-transformers embedder and 1024-dimension contract;
- CLI and MCP surface manifests inside the image;
- save, search, recall, graph, history, and other supported user journeys;
- clear unsupported behavior for MLX-only verbs;
- named-volume persistence across container replacement;
- bind mounts, permissions, read-only inputs, and full filesystems;
- stdio MCP and authenticated HTTP MCP;
- health, signals, graceful shutdown, restart, and log behavior;
- offline runtime after the model cache is populated;
- CPU and memory limits, parallel invocations, and representative latency;
- Dockerfile and Dockerfile.glama parity;
- documentation examples, including Compose semantics, even if the repository
  does not ship a compose file.

The worktree-built image is authoritative for code correctness. Published GHCR
tags are also smoke-tested as distribution-channel evidence, with version
differences classified separately from current-worktree defects.

### Gate 8: Installation and distribution

Test isolated installs from wheel, sdist, local installer specification, and
supported tool managers. Validate upgrade and migration paths without replacing
the developer's real installed runtime.

Homebrew coverage includes formula syntax/audit, dependency resolution, isolated
installation where feasible, entry points, model/runtime expectations, and
uninstall cleanup. Published PyPI, GitHub release, and GHCR artifacts are
channel-smoked and compared with documented versions. A published older release
is not mistaken for the current worktree, but broken channel wiring or false
documentation is a finding.

Check all packaged agent assets and MCP installers against disposable homes for
Claude, Codex, Devin, Continue, Goose, and every client named by the code. Verify
idempotency, preservation of unrelated configuration, valid JSON/YAML/TOML, and
the command actually pointing to the intended isolated runtime.

### Gate 9: Resilience, security, privacy, and performance

Exercise:

- concurrent readers/writers, WAL and lock contention;
- abrupt process termination around writes and migrations;
- stale derived indexes, orphan rows, hand-edited Markdown, reindex, and rebuild;
- migrations from representative older schemas and repeated idempotent runs;
- backup before destructive maintenance and restore afterward;
- malformed frontmatter, oversized records, Unicode, path edge cases, symlinks,
  invalid IDs, malicious link/reference input, and ReDoS regression cases;
- HTTP auth, secret redaction, logs, telemetry opt-in/out, and package exclusions;
- daemon and hook latency, memory growth, file descriptors, and repeated-use
  stability;
- retrieval quality and performance with graph on/off and int8/float32;
- cleanup of processes, sockets, ports, containers, volumes, and temporary
  directories.

Performance uses existing contractual budgets where defined. Elsewhere the
campaign records a repeatable baseline and treats statistically meaningful
regressions or user-visible timeouts as findings rather than inventing an
arbitrary threshold.

### Gate 10: Clean closure

After all fixes:

- rebuild all artifacts from the final worktree;
- create new empty sandboxes;
- repeat structural, automated, native, MCP, Docker, and representative
  resilience gates without relying on prior state;
- regenerate the live surface inventory and prove every item is mapped;
- compare final state with the baseline;
- confirm no QA process or runtime resource remains;
- write and commit the final report.

## Component Coverage Map

### Stable core

- save, get, list, edit/update, rename, retag, delete, restore, undo, fix, and
  provenance;
- Markdown/frontmatter round trips and SQLite rebuildability;
- vector, BM25, hybrid, exact, and optional fuzzy retrieval;
- search filters, project/global tiers, reference handling, deduplication,
  reranking, graph signal, explanations, and empty results;
- ask, context, context-pack, recall, briefing, and recall daemon;
- doctor, health, stats, config, reindex, migrate, runtime isolation, and
  strict-runtime diagnosis;
- record history, corpus history, diff, event-time as-of, valid-time as-of,
  invalidation, superseding, and undo.

### Advanced and experimental

- entities, links, graph projection, relations, trace, discovery, navigation,
  communities, bridges, and repo/code graph;
- chat, synthesize, consolidate, dream passes, hype, contradictions, lifecycle,
  contextual recall, confidence, feedback, outcome, usefulness, and graduation;
- sessions, episodes, continuity, reflection, transcript mining, verbatim index,
  and capture;
- ingest, multimodal, OCR, audio, import, export, passport, backup, restore,
  sync, collaborative, secrets, offload, and receipts;
- analytics, evaluation, token accounting, ROI, guard, release, visualization,
  dashboard, and map.

Experimental status permits interface evolution but never corruption, silent
failure, undeclared dependencies, or documentation that claims unsupported
stability.

### Wiring and operational surfaces

- CLI root, aliases, command groups, human help, JSON, stdin, and profile
  visibility;
- MCP registrars, facade calls, schemas, resources, prompts, profiles, and
  transports;
- flag registry, config catalog, environment translation, model pins, and
  deferred optional imports;
- hooks, launch assets, plugins, commands, skills, statusline, installers,
  wrappers, watchers, and daemons;
- build metadata, release workflows, Docker, manifests, and documentation.

## Isolation Contract

All functional data is disposable:

- create a unique campaign root per environment;
- set HOME, MEMO_DATA_DIR, MEMO_STATE_DIR, MEMO_CONFIG_DIR, XDG directories,
  caches, sockets, ports, and fake client homes below that root;
- scrub inherited MEMO, SYNAPSE, HF, and related variables, then add only the
  scenario's explicit configuration;
- redact tokens and paths before committed reports;
- use disposable local Git repositories/remotes for project tags and sync;
- use unique database, socket, PID, service-label, container, volume, network,
  and image names;
- never use the default vault or state directory;
- never run destructive Git commands;
- never stage pre-existing user changes;
- inventory every created resource and clean it at domain close.

Tests added to the repository use tmp_cfg or an explicitly isolated Config.
CliRunner tests set MEMO_NONINTERACTIVE, MEMO_DATA_DIR, and MEMO_STATE_DIR
explicitly. Real MLX coverage remains marked requires_mlx.

## Stateful User Journeys

### Journey A: New native user

Build artifact, install into a fresh tool directory, run first use, configure a
model profile, validate config, run doctor, create the first memory, retrieve it,
install MCP/hook assets into a fake client home, launch memo-mcp, and receive a
briefing.

### Journey B: Complete memory lifecycle

Save a decision and supporting facts, verify Markdown and indexes, search through
every supported mode, ask a grounded question, inspect and edit the record,
rename and retag it, inspect history and diffs, query past and valid-time views,
invalidate and supersede it, undo invalidation, delete, restore, hand-edit the
Markdown, reindex, rebuild, and prove the hand edit wins.

### Journey C: Ambient continuity

Start a session, submit hook input, compare cold and warm recall, start the
daemon, capture an insight, run idle capture, stop the session, start a new
session, load briefing/continuity, and observe the MCP notification.

### Journey D: Temporal correctness

Create facts using relative dates, absolute dates, bare dates, DST boundaries,
positive and negative offsets, open intervals, closed intervals, expired facts,
and deliberately inverted successor intervals. Compare current, event-time
as-of, and valid-time as-of behavior through API, CLI, MCP, fact retrieval, graph
legs, hype fold, reindex, migration, invalidation, and undo.

### Journey E: Corpus intelligence

Ingest a corpus with linked entities, duplicates, near-duplicates,
contradictions, long sections, reference material, and project overlap. Build
the graph and derived indexes, inspect traces and discovery, run consolidation,
contradiction, dream, and quality passes, and verify that later retrieval changes
only as intended.

### Journey F: Portability and recovery

Export, passport, and back up an initialized vault. Import or restore it into a
new sandbox, compare content and behavioral retrieval, create divergent copies,
sync through a disposable local remote, resolve conflicts, reindex, and prove
history and user-signal preservation rules.

### Journey G: Runtime failure and recovery

Exercise daemon lifecycle, duplicate starts, stale sockets/PIDs, client
disconnects, interrupted writes, locked databases, interrupted migrations,
filesystem permission errors, full disk simulation where safe, restart, doctor,
repair, and subsequent successful use.

### Journey H: Cross-interface MCP

Save by MCP and read by CLI; update by CLI and read by MCP; query the same corpus
through stdio and HTTP; inspect resources and prompts; exercise every profile;
restart the server; verify schema stability and persisted state.

### Journey I: Docker CPU user

Build locally, run as the image's non-root user, initialize a named volume, save
and search, launch stdio and HTTP MCP, replace the container, verify persistence,
run offline, constrain resources, interrupt and restart, inspect logs, and
remove all campaign-owned Docker resources.

## Corpus and Input Design

The campaign corpus includes:

- stable decisions and changing facts;
- explicit contradictions and successors;
- records in multiple projects plus global preferences;
- exact transcript phrases and semantically similar paraphrases;
- duplicate titles, duplicate bodies, near-duplicates, and reference-tier text;
- long Markdown with headings, links, code, lists, and frontmatter;
- empty, minimal, maximum-size, Unicode, emoji, combining-character, and
  right-to-left inputs;
- malformed dates, frontmatter, IDs, paths, JSON, YAML, and protocol frames;
- attachments for OCR, image, and audio paths when dependencies are available;
- enough records to exercise ranking, pagination, graph hubs, and latency.

Every scenario declares its setup, command or protocol call, stdin, expected
exit/status, visible-output assertion, state assertion, timeout, cleanup, and
platform capability.

## Failure Ledger and Repair Loop

Every anomaly receives:

- unique finding ID and severity;
- environment and exact reproduction;
- expected and actual behavior;
- stdout, stderr, structured response, and relevant logs;
- minimal fixture and evidence that it reproduces in a fresh sandbox;
- root cause and affected contracts;
- regression test;
- fix commit;
- focused, domain, and full-gate verification results.

Severity definitions:

- S0: data loss/corruption, secret exposure, unsafe execution, or irreversible
  migration;
- S1: stable core, install, CLI/MCP, or daemon unusable;
- S2: advanced/experimental breakage, incompatibility, or faulty recovery;
- S3: UX, documentation, schema, observability, or performance defect.

The loop is:

1. reproduce in a new sandbox;
2. reduce and exclude environmental contamination;
3. write a failing regression or wiring contract;
4. fix the systemic cause with the smallest coherent change;
5. pass focused, module, domain, and affected earlier gates;
6. inspect the diff and stage explicit owned paths only;
7. commit atomically;
8. repeat the user scenario from an empty sandbox.

Flaky failures are repeated under serialized, parallel, randomized, and
resource-constrained conditions until reproduced or demonstrated to be external.
Retries never convert a red result to green.

Stable documented behavior, public schemas, compatibility, and observable
semantics are the default authority. A proposed breaking behavior change is
returned to the user for a product decision before implementation.

## Evidence and Deliverables

The execution plan must choose exact paths, but the durable deliverables are:

1. approved design and implementation plan;
2. baseline and final machine-readable environment manifests;
3. live surface inventory;
4. matrix mapping every surface to scenarios and verdicts;
5. failure ledger with reproductions and dispositions;
6. atomic fix commits and regression tests;
7. timing, memory, retrieval, and stability measurements;
8. final detailed Markdown QA report;
9. cleanup manifest proving no campaign resource remains.

Large raw logs, model caches, test databases, coverage data, and container layers
remain uncommitted. The final report links findings to concise redacted evidence
and commits.

## Acceptance Criteria

The campaign is complete only when:

- every current live CLI, MCP, config, runtime, installer, and distribution
  surface is present in the manifest;
- every manifest row has a justified verdict and evidence;
- all confirmed S0 through S3 defects are fixed, unless an inaccessible external
  dependency makes correction impossible and the report proves that limitation;
- Ruff, mypy, quality gates, dependency contracts, resource hygiene, non-slow,
  slow, int8, float32-specific, randomized/repeated focused, and applicable
  mutation/property tests pass;
- recall and grounding evaluations do not regress;
- real MLX native journeys pass;
- Linux CPU Docker build and supported journeys pass, while MLX-only paths fail
  gracefully;
- MCP stdio and authenticated HTTP pass for every profile;
- artifacts install and run without importing from the checkout;
- migrations, reindex, backup/restore, and interrupted-runtime recovery preserve
  the documented data truths;
- final clean-room reruns pass without relying on previous campaign state;
- the worktree audit proves pre-existing user changes were preserved and only
  intentional QA commits were added;
- no campaign process, port, socket, container, volume, service, or temporary
  directory remains.

## Explicit Non-Goals

- Publishing a release, pushing commits, or opening a pull request.
- Synchronizing the local branch with origin/master before QA.
- Replacing the user's real memo install or modifying the real vault/client
  configuration.
- Treating an older published artifact as proof for the current worktree.
- Refactoring unrelated code solely to improve style or coverage.
- Promoting experimental APIs to stable contracts as a side effect of testing.

## Transition to Planning

After the user reviews this committed specification, the next and only skill is
superpowers:writing-plans. The implementation plan must decompose the campaign
into bounded workstreams, give exact commands and evidence paths, preserve the
gate order, and include the repair loop and atomic-commit discipline.
